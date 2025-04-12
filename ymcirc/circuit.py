"""
A collection of utilities for building circuits.
"""
from __future__ import annotations
import copy
from ymcirc._abstract import LatticeDef
from ymcirc.conventions import LatticeStateEncoder, IRREP_TRUNCATION_DICT_1_3_3BAR, ONE, THREE, THREE_BAR
from ymcirc.lattice_registers import LatticeRegisters
from ymcirc.givens import (
    givens,
    LPFamily,
    bitstring_value_of_LP_family,
    givens_fused_controls,
    compute_LP_family,
    gray_to_index,
)
from ymcirc._abstract.lattice_data import Plaquette
from ymcirc.utilities import _check_circuits_logically_equivalent, _flatten_circuit
from math import ceil
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import RXGate
from typing import List, Tuple, Set, Union, Dict
import numpy as np

# A list of tuples: (state bitstring1, state bitstring2, matrix element)
HamiltonianData = List[Tuple[str, str, float]]


class LatticeCircuitManager:
    """Class for creating quantum simulation circuits from LatticeRegister instances."""
    def __init__(
        self, lattice_encoder: LatticeStateEncoder, mag_hamiltonian: HamiltonianData
    ):
        """Create via a LatticeStateEncoder instance and magnetic Hamiltonian matrix elements."""
        # Copies to avoid inadvertently changing the behavior of the
        # LatticeCircuitManager instance.
        self._encoder = copy.deepcopy(lattice_encoder)
        self._mag_hamiltonian = copy.deepcopy(mag_hamiltonian)
        self._cached_mag_evol_circuit = None
        self._cached_mag_evol_params = {
            "coupling_g": None,
            "dt": None,
            "physical_states_for_control_pruning": None,
            "optimize_circuits": None,
            "control_fusion": None,
        }

        # Determine if lattice is small and periodic. If yes, filter out inconsistent Hamiltonian terms
        # and drop repeated references to the same physical control link in mag_hamiltonian bit strings.
        lattice_size_threshold_for_smallness = 2
        if not lattice_encoder.lattice_def.all_boundary_conds_periodic:
            raise NotImplementedError("Lattices with nonperiodic or mixed boundary conditions not yet supported.")
        else:
            lattice_is_periodic = True
        match lattice_encoder.lattice_def.dim:
            case 1.5:
                lattice_size = lattice_encoder.lattice_def.shape[1]
            case 2:
                lattice_size = lattice_encoder.lattice_def.shape[1]
                if lattice_size != lattice_encoder.lattice_def.shape[0]:
                    raise NotImplementedError("Non-square dim 2 lattices not yet supported.")
            case _:
                raise NotImplementedError(f"Dim {lattice_encoder.lattice_def.dim} lattice not yet supported.")
        lattice_is_small = True if lattice_size <= lattice_size_threshold_for_smallness else False
        
        if lattice_is_small is True and lattice_is_periodic is True:
            #breakpoint()
            # Filter out magnetic hamiltonian terms which are inconsistent (repeated control links must have the same value)
            filtered_and_trimmed_mag_hamiltonian: HamiltonianData = []
            for matrix_element in self._mag_hamiltonian:
                final_state_vertex_multiplicities, final_state_a_links, final_state_c_links = lattice_encoder.decode_bit_string_to_plaquette_state(matrix_element[0])
                initial_state_vertex_multiplicities, initial_state_a_links, initial_state_c_links = lattice_encoder.decode_bit_string_to_plaquette_state(matrix_element[1])
                match lattice_encoder.lattice_def.dim:
                    case 1.5:
                        # c1 and c2 are the same physical link. c3 and c4 are the same physical link.
                        final_state_has_inconsistent_controls = (
                            (final_state_c_links[0] != final_state_c_links[1]) or
                            (final_state_c_links[2] != final_state_c_links[3])
                        )
                        initial_state_has_inconsistent_controls = (
                            (initial_state_c_links[0] != initial_state_c_links[1]) or
                            (initial_state_c_links[2] != initial_state_c_links[3])
                        )
                    case 2:
                        # physical equivalent controls: c1 == c4, c2 == c7, c3 == c6, c5 == c8
                        final_state_has_inconsistent_controls = (
                            (final_state_c_links[0] != final_state_c_links[3]) or
                            (final_state_c_links[1] != final_state_c_links[6]) or
                            (final_state_c_links[2] != final_state_c_links[5]) or
                            (final_state_c_links[4] != final_state_c_links[7])
                        )
                        initial_state_has_inconsistent_controls = (
                            (initial_state_c_links[0] != initial_state_c_links[3]) or
                            (initial_state_c_links[1] != initial_state_c_links[6]) or
                            (initial_state_c_links[2] != initial_state_c_links[5]) or
                            (initial_state_c_links[4] != initial_state_c_links[7])
                        )
                    case _:
                        raise NotImplementedError(f"Dim {lattice_encoder.lattice_def.dim} lattice not yet supported.")
                if (final_state_has_inconsistent_controls is True) or (initial_state_has_inconsistent_controls is True):
                    continue
                else:
                    # Matrix element is consistent on shared controls. Re-encode as bitstring and keep.
                    # Filter out duplicate c links by keeping only the first instance of each physically distinct c link.
                    match lattice_encoder.lattice_def.dim:
                        case 1.5:
                            # c1 and c3 are kept
                            final_state_physical_c_links = (final_state_c_links[0], final_state_c_links[2])
                            initial_state_physical_c_links = (initial_state_c_links[0], initial_state_c_links[2])
                        case 2:
                            # c1, c2, c3, and c5 are kept
                            final_state_physical_c_links = (final_state_c_links[0], final_state_c_links[1], final_state_c_links[2], final_state_c_links[4])
                            initial_state_physical_c_links = (initial_state_c_links[0], initial_state_c_links[2], initial_state_c_links[2], initial_state_c_links[4])
                        case _:
                            raise NotImplementedError(f"Dim {lattice_encoder.lattice_def.dim} lattice not yet supported.")
                    
                    final_state_plaquette = (
                        final_state_vertex_multiplicities,
                        final_state_a_links,
                        final_state_physical_c_links
                    )
                    initial_state_plaquette = (
                        initial_state_vertex_multiplicities,
                        initial_state_a_links,
                        initial_state_physical_c_links
                    )
                    consistent_and_trimmed_matrix_element = (
                        lattice_encoder.encode_plaquette_state_as_bit_string(final_state_plaquette, override_n_c_links_validation=True),
                        lattice_encoder.encode_plaquette_state_as_bit_string(initial_state_plaquette, override_n_c_links_validation=True),
                        matrix_element[2]
                    )
                    filtered_and_trimmed_mag_hamiltonian.append(consistent_and_trimmed_matrix_element)
            
            self._mag_hamiltonian = filtered_and_trimmed_mag_hamiltonian
            breakpoint()
            

    # TODO modularize the logic for walking through a lattice.
    def create_blank_full_lattice_circuit(
        self, lattice: LatticeRegisters
    ) -> QuantumCircuit:
        """
        Return a blank quantum circuit with all link and vertex registers in lattice.

        Length-zero registers are skipped (relevant for d=3/2, T1 vertex registers.)

        The convention is to construct the circuit by iterating over all vertices,
        then for each vertex, to iterate over all the "positive" links leaving the vertex.
        The iteration over vertices is by ordering on the tuples denoting
        lattice coordinates.

        Example for d = 2 with periodic boundary conditions:

        (pbc)        (pbc)        (pbc)
          |            |            |
          |            |            |
          l16          l17          l18
          |            |            |
          |            |            |
        (0,2)--l13---(1,2)--l14---(2,2)--l15---- (pbc)
          |            |            |
          |            |            |
          l10          l11          l12
          |            |            |
          |            |            |
        (0,1)---l7---(1,1)---l8---(2,1)---l9---- (pbc)
          |            |            |
          |            |            |
          l4           l5           l6
          |            |            |
          |            |            |
        (0,0)---l1---(1,0)---l2---(2,0)---l3---- (pbc)


        Will be mapped to the ket

        |(0,0) l1 l4 (0,1) l7 l10 (0,2) l13 l16 (1,0) l2 l5 ...>

        where the left-most tensor factor is the top line in the circuit.

        In d=3/2, the "top" pbc links in the above diagram are omitted
        because they do not exist on that lattice.
        """
        all_lattice_registers: List[QuantumRegister] = []
        for vertex_address in lattice.vertex_addresses:
            # Add the current vertex, and the positive links connected to it.
            # Skip "top" in d = 3/2.
            current_vertex_reg = lattice.get_vertex(vertex_address)
            all_lattice_registers.append(current_vertex_reg)
            for positive_direction in range(1, ceil(lattice.dim) + 1):
                has_no_vertical_periodic_link_three_halves_case = (
                    lattice.dim == 1.5
                    and positive_direction > 1
                    and vertex_address[1] == 1
                )
                if has_no_vertical_periodic_link_three_halves_case:
                    continue

                current_link_reg = lattice.get_link(
                    (vertex_address, positive_direction)
                )
                all_lattice_registers.append(current_link_reg)

        return QuantumCircuit(*all_lattice_registers)

    def apply_electric_trotter_step(
        self,
        master_circuit: QuantumCircuit,
        lattice: LatticeRegisters,
        hamiltonian: list[float],
        coupling_g: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        """
        Perform an electric Trotter step.

        Implementation uses CX and Zs to implement rotations of Z, I Paulis.
        The single link electric Trotter step is constructed through Z, I
        rotations (e.g. e^(i*coeff*IZZZI))). Such rotatons can be constructed
        through parity circuits (see section 4.2 of arXiv:1001.3855).

        Arguments:
            - master_circuit: A QuantumCircuit instance which is built from all
                              the QuantumRegister instances in lattice.
            - lattice: A LatticeRegisters instance which keeps track of all the
                       QuantumRegisters.
            - hamilonian: Pauli decompositon of the single link electric
                          Hamiltonian. The list hamiltonian is contains
                          coefficients s.t. for hamiltonian[i] = coeff, coeff
                          is coeff of bistring(i) with 'Z'=1 and 'I'=0 in the
                          bitstring.
            - coupling_g: The value of the strong coupling constant.
            - dt: The size of the Trotter time step.

        Returns:
            A new QuantumCircuit instance which is master_circuit with the
            electric Trotter step appended.
        """
        N = int(np.log2(len(hamiltonian)))
        angle_mod = ((coupling_g**2) / 2) * dt
        local_circuit = QuantumCircuit(N)

        # The parity circuit primitive of CXs and Zs.
        for i in range(len(hamiltonian)):
            locs = [
                loc
                for loc, bit in enumerate(str("{0:0" + str(N) + "b}").format(i))
                if bit == "1"
            ]
            for j in locs[:-1]:
                local_circuit.cx(j, locs[-1])
            if len(locs) != 0:
                local_circuit.rz(2 * angle_mod * hamiltonian[i], locs[-1])
            for j in locs[:-1]:
                local_circuit.cx(j, locs[-1])

        # Loop over links for electric Hamiltonian
        for link_address in lattice.link_addresses:
            link_qubits = [
                qubit for qubit in lattice.get_link((link_address[0], link_address[1]))
            ]
            master_circuit.compose(local_circuit, qubits=link_qubits, inplace=True)

    # TODO Can we get the circuits in a parameterized way?
    def apply_magnetic_trotter_step(
        self,
        master_circuit: QuantumCircuit,
        lattice: LatticeRegisters,
        coupling_g: float = 1.0,
        dt: float = 1.0,
        optimize_circuits: bool = True,
        physical_states_for_control_pruning: Union[None | Set[str]] = None,
        control_fusion: bool = False,
        cache_mag_evol_circuit: bool = True,
    ) -> None:
        """
        Add one magnetic Trotter step to the entire lattice circuit.

        This is done by iterating over every lattice vertex.At each vertex,
        there's an additional iteration over every "positive" plaquette.
        For each such plaquette, the plaquette-local magnetic Trotter step
        is appended to the circuit.

        Note that this modifies master_circuit directly rather than returning
        a new circuit!

        Arguments:
          - master_circuit: a QuantumCircuit instance which is built from all
                            the QuantumRegister instances in lattice.
          - lattice: a LatticeRegisters instance which keeps track of all the
                     QuantumRegisters.
          - coupling_g: The value of the strong coupling constant.
          - dt: the size of the Trotter time step.
          - optimize_circuits: if True, run the qiskit transpiler on each
                               internal givens rotation with the maximum
                               optimization level before composing with
                               master_circuit.
          - set_of_physical_states: The set of all physical states encoded as bitstrings.
                                    If provided, control pruning of multi-control rotation
                                    gate inside Givens rotation subcircuits will be attempted.
                                    If None, no control pruning is attempted.
          - control_fusion: Optional boolian argument with the default set to False. If it's set
                            to be True, then LP families of givens rotations are first Gray code ordered,
                            then redundant controls are removed.
          - cache_mag_evol_circuit: Optional boolean argument to cache the magnetic Hamiltonian
                                    evolution circuit once generated, and forevermore use that.
        Returns:
          A new QuantumCircuit instance which is master_circuit with the
          Trotter step appended.
        """
        # Create the magnetic Hamiltonian evolution circuit.
        mag_evol_recomputation_needed = (
            (self._cached_mag_evol_circuit is None)
            or (control_fusion != self._cached_mag_evol_params["control_fusion"])
            or (dt != self._cached_mag_evol_params["dt"])
            or (coupling_g != self._cached_mag_evol_params["coupling_g"])
            or (optimize_circuits != self._cached_mag_evol_params["optimize_circuits"])
            or (
                physical_states_for_control_pruning
                != self._cached_mag_evol_params["physical_states_for_control_pruning"]
            )
        )
        if cache_mag_evol_circuit is True and not mag_evol_recomputation_needed:
            print("Fetching cached magnetic evolution circuit.")
            plaquette_local_rotation_circuit = self._cached_mag_evol_circuit
        else:
            print("Building magnetic evolution circuit.")
            plaquette_local_rotation_circuit = self._build_mag_evol_circuit(
                control_fusion,
                physical_states_for_control_pruning,
                coupling_g,
                dt,
                optimize_circuits,
            )
            if cache_mag_evol_circuit is True:
                print("Storing magnetic evolution circuit in cache.")
                self._cached_mag_evol_circuit = plaquette_local_rotation_circuit
                self._cached_mag_evol_params = {
                    "coupling_g": coupling_g,
                    "dt": dt,
                    "physical_states_for_control_pruning": physical_states_for_control_pruning,
                    "optimize_circuits": optimize_circuits,
                    "control_fusion": control_fusion,
                }

        # Stitch magnetic Hamiltonian evolution circuit onto LatticeRegisters.
        # Vertex iteration loop.
        for vertex_address in lattice.vertex_addresses:
            # Skip creating "top vertex" plaquettes for d=3/2.
            has_no_vertical_periodic_link_three_halves_case = (
                lattice.dim == 1.5 and vertex_address[1] == 1
            )
            if has_no_vertical_periodic_link_three_halves_case:
                continue

            # Get the plaquettes for the current vertex.
            print(f"Fetching all positive plaquettes at vertex {vertex_address}.")
            has_only_one_positive_plaquette = lattice.dim == 1.5 or lattice.dim == 2
            if has_only_one_positive_plaquette:
                plaquettes: List[Plaquette] = [
                    lattice.get_plaquettes(vertex_address, 1, 2)
                ]
            else:
                plaquettes: List[Plaquette] = lattice.get_plaquettes(vertex_address)
            print(f"Found {len(plaquettes)} plaquette(s).")

            # TODO test stitching logic when the same control reg shows up on multiple vertices.
            # For each plaquette, apply the the local Trotter step circuit.
            for plaquette in plaquettes:
                # Get qubits for the current plaquette.
                
                # Deal with possibility of duplicated control registers.
                # Iterate through all the control links, and if any are repeated, only the first
                # will be treated as an actual physical control.
                # TODO finish and clean this up.
                distinct_control_links = set(plaquette.control_links_ordered)
                repeated_c_link_mask: Tuple[bool] = tuple([plaquette.control_links_ordered.count(c_link) > 1 for c_link in plaquette.control_links_ordered])
                physical_c_link_mask = []
                encountered_c_links = []
                for idx, c_link in enumerate(plaquette.control_links_ordered):
                    if (c_link not in encountered_c_links) and (repeated_c_link_mask[idx] is True):
                        physical_c_link_mask.append(True)
                    else:
                        physical_c_link_mask.append(False)
                    encountered_c_links.append(c_link)

                #physical_c_link_mask: Tuple[bool] = tuple([repeated_c_link_mask[idx] is True and ])
                has_same_control_link_on_different_vertices = len(plaquette.control_links_ordered) != len(distinct_control_links)
                #if has_same_control_link_on_different_vertices:
                    # check mat_elem for consistency
                print(plaquette.control_links_ordered)
                print(set(plaquette.control_links_ordered))
                print(repeated_c_link_mask)
                print(physical_c_link_mask)
                breakpoint()
                
                # Collect the local qubits for stitching purposes.
                vertex_multiplicity_qubits = []
                a_link_qubits = []
                c_link_qubits = []
                for register in plaquette.vertices:
                    for qubit in register:
                        vertex_multiplicity_qubits.append(qubit)
                for register in plaquette.active_links:
                    for qubit in register:
                        a_link_qubits.append(qubit)
                for register in plaquette.control_links_ordered:
                    for qubit in register:
                        c_link_qubits.append(qubit)
                        
                # Now that we have the qubits for the current plaquette,
                # Stitch the local magnetic evolution circuit into master circuit.
                master_circuit.compose(
                    plaquette_local_rotation_circuit,
                    qubits=[
                        *vertex_multiplicity_qubits,
                        *a_link_qubits,
                        *c_link_qubits
                    ],
                    inplace=True
                )

    def _build_mag_evol_circuit(
        self,
        control_fusion: bool,
        physical_states_for_control_pruning: Union[None | Set[str]],
        coupling_g: float,
        dt: float,
        optimize_circuits: bool,
    ) -> QuantumCircuit:
        """Build the magnetic time-evolution circuit for a plaquette."""
        # Sort the bitstrings corresponding to transitions in the magnetic hamiltonian
        # into LP bins. This step also computes the angle of Givens rotation for each
        # pair of bitstrings.
        lp_bin = LatticeCircuitManager._sort_matrix_elements_into_lp_bins(
            self._mag_hamiltonian,
            coupling_g,
            dt,
        )
        if control_fusion is True:
            # Sort according to Gray-order.
            lp_bin = {
                k: lp_bin[k]
                for k in sorted(
                    lp_bin.keys(),
                    key=lambda x: gray_to_index(bitstring_value_of_LP_family(x)),
                )
            }
        # iterate over all LP bins and apply givens rotation.
        plaquette_circ_n_qubits = len(
            self._mag_hamiltonian[0][0]
        )  # TODO this is a disgusting way to get the size of the magnetic evol circuit per plaquette.
        plaquette_local_rotation_circuit = QuantumCircuit(plaquette_circ_n_qubits)
        for lp_fam, lp_bin_w_angle in lp_bin.items():
            if control_fusion is True:
                fused_circ_for_lp_fam = givens_fused_controls(
                    lp_bin_w_angle, lp_fam, physical_states_for_control_pruning
                )
                plaquette_local_rotation_circuit.compose(
                    fused_circ_for_lp_fam, inplace=True
                )
            else:
                # If control fusion is turned off, givens rotation is applied individually
                # to all bitstrings.
                for bs1, bs2, angle in lp_bin_w_angle:
                    bs1_bs2_circuit = givens(
                        bs1, bs2, angle, physical_states_for_control_pruning
                    )
                    plaquette_local_rotation_circuit.compose(
                        bs1_bs2_circuit, inplace=True
                    )
            if optimize_circuits is True:
                plaquette_local_rotation_circuit = transpile(
                    plaquette_local_rotation_circuit, optimization_level=3
                )

        return plaquette_local_rotation_circuit

    @staticmethod
    def _sort_matrix_elements_into_lp_bins(
        bitstrings_w_matrix_element: List[(str, str, float)],
        coupling_g: float,
        dt: float,
    ) -> Dict[LPFamily, List[(str, str, float)]]:
        """
        Rearrange magnetic Hamiltonian matrix elements to LP family bins.

        This function does two things:
        1. Sorts tuples of bitstrings into LP bins.
        2. Computes angle of Givens rotation from matrix element

        Input:
            - bitstring_w_matrix_element: this is of the form (bs1, bs2, matrix_element)
                where bs1 is the initial state, bs2 is the final state, and matrix_element
                is the amplitude of transition.
            - coupling_g: value of coupling constant being used.
            - dt: timestep being used.

        Output:
            - dictionary where each key is a LP bin, and the corresponding value is a list of transitions
                that have the same LP value. Each transition is of the form (bitstring1, bitstring2, angle).
        """
        lp_bin = {}
        for (
            bit_string_1,
            bit_string_2,
            matrix_elem,
        ) in bitstrings_w_matrix_element:
            angle = -matrix_elem * (1 / (2 * (coupling_g**2))) * dt
            lp_fam = compute_LP_family(bit_string_1, bit_string_2)
            if lp_fam not in lp_bin.keys():
                lp_bin[lp_fam] = []
            lp_bin[lp_fam].append((bit_string_1, bit_string_2, angle))
        return lp_bin


def _test_create_blank_full_lattice_circuit_has_promised_register_order():
    """Check in some cases that we get the ordering promised in the method docstring."""
    # Creating test data.
    # Not physically meaningful, but has the right format.
    iweight_one = (0, 0, 0)
    iweight_three = (1, 0, 0)
    irrep_bitmap = {
        iweight_one: "0",
        iweight_three: "1"
    }
    physical_plaquette_states_3halves_no_vertices_needed = [
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            (iweight_one, iweight_one, iweight_one, iweight_one)
        ),
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_three, iweight_three, iweight_three),
            (iweight_three, iweight_one, iweight_one, iweight_one)
        )
    ]
    physical_plaquette_states_3halves = [
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            (iweight_one, iweight_one, iweight_one, iweight_one)
        ),
        (
            (1, 1, 1, 1),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            (iweight_one, iweight_one, iweight_one, iweight_one)
        ),
        (
            (2, 2, 2, 2),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            (iweight_one, iweight_one, iweight_one, iweight_one)
        ),
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_three, iweight_three, iweight_three),
            (iweight_three, iweight_one, iweight_one, iweight_one)
        )
    ]
    physical_plaquette_states_2d = [
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            (iweight_one, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one)
        ),
        (
            (1, 1, 1, 1),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            (iweight_one, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one)
        ),
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_three, iweight_three, iweight_three),
            (iweight_three, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one)
        )
    ]
    # Hamiltonian bitstrings take the form vertex_bits + active link bits + c link bits.
    # For the "no_vertices" data, vertex_bits is the empty string. The numbers of
    # Vertex bits and link bits can be inferred from the test data (encode integer in bitstring, use link bitmap).
    mag_hamiltonian_2d = [("1110111100000000", "0001000011111111", -0.33), ("0000111100000000", "1111000011111111", 1.0)]
    mag_hamiltonian_3halves = [("1010010111110000", "0000000011110000", 1.0), ("0000000010100101", "1010101000000001", 1.0)]
    mag_hamiltonian_3halves_no_vertices = [("10101111", "11110010", 1.0), ("10010000", "10000001", 1.0), ("11111101", "00000101", 1.0)]
    # Registers for lattices with size 3
    expected_register_order_2d = [
        'v:(0, 0)', 'l:((0, 0), 1)', 'l:((0, 0), 2)',
        'v:(0, 1)', 'l:((0, 1), 1)', 'l:((0, 1), 2)',
        'v:(0, 2)', 'l:((0, 2), 1)', 'l:((0, 2), 2)',
        'v:(1, 0)', 'l:((1, 0), 1)', 'l:((1, 0), 2)',
        'v:(1, 1)', 'l:((1, 1), 1)', 'l:((1, 1), 2)',
        'v:(1, 2)', 'l:((1, 2), 1)', 'l:((1, 2), 2)',
        'v:(2, 0)', 'l:((2, 0), 1)', 'l:((2, 0), 2)',
        'v:(2, 1)', 'l:((2, 1), 1)', 'l:((2, 1), 2)',
        'v:(2, 2)', 'l:((2, 2), 1)', 'l:((2, 2), 2)',
    ]
    expected_register_order_3halves = [
        'v:(0, 0)', 'l:((0, 0), 1)', 'l:((0, 0), 2)',
        'v:(0, 1)', 'l:((0, 1), 1)',
        'v:(1, 0)', 'l:((1, 0), 1)', 'l:((1, 0), 2)',
        'v:(1, 1)', 'l:((1, 1), 1)',
        'v:(2, 0)', 'l:((2, 0), 1)', 'l:((2, 0), 2)',
        'v:(2, 1)', 'l:((2, 1), 1)'
    ]
    expected_register_order_3halves_no_vertices = [
        'l:((0, 0), 1)', 'l:((0, 0), 2)',
        'l:((0, 1), 1)',
        'l:((1, 0), 1)', 'l:((1, 0), 2)',
        'l:((1, 1), 1)',
        'l:((2, 0), 1)', 'l:((2, 0), 2)',
        'l:((2, 1), 1)',
    ]
    # Registers for a lattice ith size 2 (small enough for the same link to control multiple vertices in a single plaquette).
    expected_register_order_2d_small_lattice = [
        'v:(0, 0)', 'l:((0, 0), 1)', 'l:((0, 0), 2)',
        'v:(0, 1)', 'l:((0, 1), 1)', 'l:((0, 1), 2)',
        'v:(1, 0)', 'l:((1, 0), 1)', 'l:((1, 0), 2)',
        'v:(1, 1)', 'l:((1, 1), 1)', 'l:((1, 1), 2)',
    ]
    test_cases = [
        (
            expected_register_order_2d,
            irrep_bitmap,
            physical_plaquette_states_2d,
            2,
            3,
            mag_hamiltonian_2d
        ),
        (
            expected_register_order_2d_small_lattice,
            irrep_bitmap,
            physical_plaquette_states_2d,
            2,
            2,
            mag_hamiltonian_2d
        ),
        (
            expected_register_order_3halves,
            irrep_bitmap,
            physical_plaquette_states_3halves,
            1.5,
            3,
            mag_hamiltonian_3halves
        ),
        (
            expected_register_order_3halves_no_vertices,
            irrep_bitmap,
            physical_plaquette_states_3halves_no_vertices_needed,
            1.5,
            3,
            mag_hamiltonian_3halves_no_vertices
        )
    ]

    # Iterate over all test cases.
    for expected_register_names_ordered, link_bitmap, physical_plaquette_states, dims, size, hamiltonian in test_cases:
        # Initialize registers and create circuit.
        lattice_encoder = LatticeStateEncoder(
            link_bitmap, physical_plaquette_states, LatticeDef(dims, size))
        lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
        circ_mgr = LatticeCircuitManager(
            lattice_encoder=lattice_encoder,
            mag_hamiltonian=hamiltonian
        )
        print(
            f"Checking register order in a circuit constructed from a {dims}-dimensional lattice "
            f"of linear size {size}."
        )
        print(f"Link bitmap: {link_bitmap}\nVertex bitmap: {lattice_encoder.vertex_bitmap}")
        print(f"Expected register ordering: {expected_register_names_ordered}")

        master_circuit = circ_mgr.create_blank_full_lattice_circuit(lattice_registers)
        nonzero_regs = [reg for reg in master_circuit.qregs if len(reg) > 0]
        n_nonzero_regs = len(nonzero_regs)

        # Check that the circuit makes sense.
        assert n_nonzero_regs == len(
            expected_register_names_ordered
        ), f"Expected {len(expected_register_names_ordered)} registers. Encountered {n_nonzero_regs} registers."
        for expected_name, reg in zip(expected_register_names_ordered, nonzero_regs):
            if len(reg) == 0:
                continue
            assert (
                expected_name == reg.name
            ), f"Expected: {expected_name}, encountered: {reg.name}"
            print(f"Verified location of the register for {expected_name}.")

    print("Register order tests passed.")


def _test_apply_magnetic_trotter_step_d_3_2_large_lattice():
    print(
        "Checking that application of magnetic Trotter step works for d=3/2 "
        "on a large enough lattice that no control links are repeated in any "
        "one plaquette."
    )
    # DO NOT CHANGE THE "DUMMY" DATA UNLESS YOU ARE WILLING TO WORK OUT
    # WHAT THE CORRECT "EXPECTED" CIRCUITS ARE. THERE IS
    # STRONG DEPENDENCE BETWEEN THAT AND THESE DUMMY
    # TEST DATA.
    # If you need to make more test data, follow the following process:
    #    1. Come up with a dummy hamiltonian matrix element of the form (plaquette bitstring, plaquette bitstring, mat elem value).
    #    2. Come up with the corresponding physical states which get encoded to these bit strings.
    #    3. For each matrix element * plaquette in the lattice, there will be ONE givens rotation circuit. For each such givens rotation:
    #      3a. Determine which registers are involved in the circuit following conventional ordering of vertices, then active links, then control links.
    #      3b. Determine what the "X" circuit prefix is by comparing the state bitstrings for the matrix element, determining the LP family, and then
    #          mapping each substring in the plaquette encoding onto actual registers in the lattice.
    #      3c. Repeat this exercise with the multi-control rotation, where the type of ladder or project operator involved determines the control states.
    # Ask yourself if you REALLY feel like doing all that before mucking about with this test data.
    dummy_mag_hamiltonian = [
        ("00100000" + "00000000", "01010100" + "10011010", 0.33)  # One matrix element, plaquette only has a_link and c_link substrings.
    ]
    dummy_phys_states = [
        (  # Matches the first encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (ONE, THREE, ONE, ONE),
            (ONE, ONE, ONE, ONE)
        ),
        (  # Matches the second encoded state in the dummy magnetic haimiltonian.
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, ONE),
            (THREE, THREE_BAR, THREE, THREE)
        )
    ]
    expected_master_circuit = QuantumCircuit(18)
    expected_rotation_gates = {  # Data for constructing the expected circuit.
        "angle": -0.165,
        "MCU ctrl state": "000000000000001",  # Little endian per qiskit convention.
        "givens rotations": [
            {
                "pivot": 1,
                "CX targets": [8, 9, 5, 12, 7, 10, 16],
                "MCU ctrls": [8, 0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17]
            },
            {
                "pivot": 7,
                "CX targets": [14, 15, 11, 0, 13, 16, 4],
                "MCU ctrls": [14, 0, 1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17]
            },
            {
                "pivot": 13,
                "CX targets": [2, 3, 17, 6, 1, 4, 10],
                "MCU ctrls": [2, 0, 1, 3, 4, 5, 6, 7, 10, 11, 12, 14, 15, 16, 17]
            }
        ]
    }
    for rotation_data in expected_rotation_gates["givens rotations"]:
        # Build subcircuits.
        Xcirc = QuantumCircuit(18)
        for target in rotation_data["CX targets"]:
            Xcirc.cx(
                control_qubit=rotation_data["pivot"],
                target_qubit=target
            )
        MCU = RXGate(expected_rotation_gates["angle"]).control(num_ctrl_qubits=len(rotation_data["MCU ctrls"]), ctrl_state=expected_rotation_gates["MCU ctrl state"])

        # Construct current expected givens rotation.
        expected_master_circuit.compose(Xcirc, inplace=True)
        expected_master_circuit.append(MCU, rotation_data["MCU ctrls"] + [rotation_data["pivot"]])
        expected_master_circuit.compose(Xcirc, inplace=True)

    print("Expected circuit:")
    print(expected_master_circuit.draw())

    # Create master circuit via the magnetic trotter step code.
    lattice_def = LatticeDef(1.5, 3)
    lattice_encoder = LatticeStateEncoder(
        IRREP_TRUNCATION_DICT_1_3_3BAR,
        dummy_phys_states,
        lattice=lattice_def)
    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    circ_mgr = LatticeCircuitManager(lattice_encoder,
                                     dummy_mag_hamiltonian)
    master_circuit = circ_mgr.create_blank_full_lattice_circuit(
        lattice_registers)
    circ_mgr.apply_magnetic_trotter_step(
        master_circuit,
        lattice_registers,
        optimize_circuits=False
    )
    print("Obtained circuit:")
    print(master_circuit.draw())

    # Checking equivalence via helper methods for
    # (1) flattening a circuit down to a single register and (2) comparing
    # logical equivalence of two circuits.
    assert _check_circuits_logically_equivalent(_flatten_circuit(master_circuit), expected_master_circuit), "Encountered inequivalent circuits. Expected:\n" \
        f"{expected_master_circuit.draw()}\nObtained:\n" \
        f"{master_circuit.draw()}"
    print("Test passed.")


# TODO finish updating this test. Need to add another matrix element which won't fail the consistency test.
def _test_apply_magnetic_trotter_step_d_3_2_small_lattice():
    print(
        "Checking that application of magnetic Trotter step works for d=3/2 "
        "on a small lattice that where some control links are repeated in each "
        "plaquette."
    )
    # DO NOT CHANGE THE "DUMMY" DATA UNLESS YOU ARE WILLING TO WORK OUT
    # WHAT THE CORRECT "EXPECTED" CIRCUITS ARE. THERE IS
    # STRONG DEPENDENCE BETWEEN THAT AND THESE DUMMY
    # TEST DATA.
    # If you need to make more test data, follow the following process:
    #    1. Come up with a dummy hamiltonian matrix element of the form (plaquette bitstring, plaquette bitstring, mat elem value).
    #    2. Come up with the corresponding physical states which get encoded to these bit strings.
    #    3. For each matrix element * plaquette in the lattice, there will be ONE givens rotation circuit. For each such givens rotation:
    #      3a. Determine which registers are involved in the circuit following conventional ordering of vertices, then active links, then control links.
    #      3b. Determine what the "X" circuit prefix is by comparing the state bitstrings for the matrix element, determining the LP family, and then
    #          mapping each substring in the plaquette encoding onto actual registers in the lattice.
    #      3c. Repeat this exercise with the multi-control rotation, where the type of ladder or project operator involved determines the control states.
    # Ask yourself if you REALLY feel like doing all that before mucking about with this test data.
    dummy_mag_hamiltonian = [
        ("00100001" + "00000000", "01010110" + "10011010", 0.33),  # One matrix element, plaquette only has a_link and c_link substrings. Should get filtered out based on c_link consistency.
        ("00100001" + "00000000", "01010110" + "10100000", 0.33)  # One matrix element, plaquette only has a_link and c_link substrings. Should not get filtered out based on c_link consistency.
    ]
    dummy_phys_states = [
        (  # Matches the first encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (ONE, THREE, ONE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (  # Matches the second encoded state in the dummy magnetic haimiltonian.
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            (THREE, THREE_BAR, THREE, THREE)
        ),
        (  # Matches the third encoded state in the dummy magnetic haimiltonian.
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            (THREE, THREE, ONE, ONE)
        )
    ]
    expected_master_circuit = QuantumCircuit(12)
    expected_rotation_gates = {  # Data for constructing the expected circuit. #TODO update to match the new mag hamiltonian data.
        "angle": -0.165,
        "MCU ctrl state": "00000000011",  # Little endian per qiskit convention.
        "givens rotations": [
            {
                "pivot": 1,
                "CX targets": [8, 9, 5, 6, 7, 10],
                "MCU ctrls": [8, 3, 0, 2, 4, 5, 6, 7, 9, 10, 11]
            },
            { # alinks 10;1, 00;2, 11;1, 10;2 clinks 00;1, 00;1 01;1, 01;1
                "pivot": 7,
                "CX targets": [2, 3, 11, 0, 1, 4],
                "MCU ctrls": [2, 9, 0, 1, 3, 4, 5, 6, 8, 10, 11]
            },
        ]
    }
    for rotation_data in expected_rotation_gates["givens rotations"]:
        # Build subcircuits.
        Xcirc = QuantumCircuit(12)
        for target in rotation_data["CX targets"]:
            Xcirc.cx(
                control_qubit=rotation_data["pivot"],
                target_qubit=target
            )
        MCU = RXGate(expected_rotation_gates["angle"]).control(num_ctrl_qubits=len(rotation_data["MCU ctrls"]), ctrl_state=expected_rotation_gates["MCU ctrl state"])

        # Construct current expected givens rotation.
        expected_master_circuit.compose(Xcirc, inplace=True)
        expected_master_circuit.append(MCU, rotation_data["MCU ctrls"] + [rotation_data["pivot"]])
        expected_master_circuit.compose(Xcirc, inplace=True)

    print("Expected circuit:")
    print(expected_master_circuit.draw())

    # Create master circuit via the magnetic trotter step code.
    lattice_def = LatticeDef(1.5, 2)
    lattice_encoder = LatticeStateEncoder(
        IRREP_TRUNCATION_DICT_1_3_3BAR,
        dummy_phys_states,
        lattice=lattice_def)
    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    circ_mgr = LatticeCircuitManager(lattice_encoder,
                                     dummy_mag_hamiltonian)
    master_circuit = circ_mgr.create_blank_full_lattice_circuit(
        lattice_registers)
    circ_mgr.apply_magnetic_trotter_step(
        master_circuit,
        lattice_registers,
        optimize_circuits=False
    )
    print("Obtained circuit:")
    print(master_circuit.draw())

    # Checking equivalence via helper methods for
    # (1) flattening a circuit down to a single register and (2) comparing
    # logical equivalence of two circuits.
    assert _check_circuits_logically_equivalent(_flatten_circuit(master_circuit), expected_master_circuit), "Encountered inequivalent circuits. Expected:\n" \
        f"{expected_master_circuit.draw()}\nObtained:\n" \
        f"{master_circuit.draw()}"
    print("Test passed.")
    raise NotImplementedError()


def _test_apply_magnetic_trotter_step_d_2_large_lattice():
    print(
        "Checking that application of magnetic Trotter step works for d=2 "
        "on a large enough lattice that no control links are repeated in any "
        "one plaquette."
    )
    # DO NOT CHANGE THE "DUMMY" DATA UNLESS YOU ARE WILLING TO WORK OUT
    # WHAT THE CORRECT "EXPECTED" CIRCUITS ARE. THERE IS
    # STRONG DEPENDENCE BETWEEN THAT AND THESE DUMMY
    # TEST DATA.
    # If you need to make more test data, follow the following process:
    #    1. Come up with a dummy hamiltonian matrix element of the form (plaquette bitstring, plaquette bitstring, mat elem value).
    #    2. Come up with the corresponding physical states which get encoded to these bit strings.
    #    3. For each matrix element * plaquette in the lattice, there will be ONE givens rotation circuit. For each such givens rotation:
    #      3a. Determine which registers are involved in the circuit following conventional ordering of vertices, then active links, then control links.
    #      3b. Determine what the "X" circuit prefix is by comparing the state bitstrings for the matrix element, determining the LP family, and then
    #          mapping each substring in the plaquette encoding onto actual registers in the lattice.
    #      3c. Repeat this exercise with the multi-control rotation, where the type of ladder or project operator involved determines the control states.
    # Ask yourself if you REALLY feel like doing all that before mucking about with this test data.
    dummy_mag_hamiltonian = [
        ("0000" + "00100000" + "0000000000000010", "0010" + "01010100" + "1001101000000010", 0.33)  # One matrix element, plaquette has v, a_link, and c_link substrings.
    ]
    dummy_phys_states = [
        (  # Matches the first encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (ONE, THREE, ONE, ONE),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, THREE)
        ),
        (  # Matches the second encoded state in the dummy magnetic haimiltonian.
            (0, 0, 1, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, ONE),
            (THREE, THREE_BAR, THREE, THREE, ONE, ONE, ONE, THREE)
        )
    ]
    expected_master_circuit = QuantumCircuit(18)
    expected_rotation_gates = {  # Data for constructing the expected circuit.
        "angle": -0.165,
        "MCU ctrl state": "0000000000000000000000000011",  # Little endian per qiskit convention.
        "givens rotations": [ # TODO everything below here needs to be updated.
            {
                "pivot": 1,
                "CX targets": [8, 9, 5, 12, 7, 10, 16],
                "MCU ctrls": [8, 0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17]
            },
            {
                "pivot": 7,
                "CX targets": [14, 15, 11, 0, 13, 16, 4],
                "MCU ctrls": [14, 0, 1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17]
            },
            {
                "pivot": 13,
                "CX targets": [2, 3, 17, 6, 1, 4, 10],
                "MCU ctrls": [2, 0, 1, 3, 4, 5, 6, 7, 10, 11, 12, 14, 15, 16, 17]
            }
        ]
    }
    for rotation_data in expected_rotation_gates["givens rotations"]:
        # Build subcircuits.
        Xcirc = QuantumCircuit(18)
        for target in rotation_data["CX targets"]:
            Xcirc.cx(
                control_qubit=rotation_data["pivot"],
                target_qubit=target
            )
        MCU = RXGate(expected_rotation_gates["angle"]).control(num_ctrl_qubits=len(rotation_data["MCU ctrls"]), ctrl_state=expected_rotation_gates["MCU ctrl state"])

        # Construct current expected givens rotation.
        expected_master_circuit.compose(Xcirc, inplace=True)
        expected_master_circuit.append(MCU, rotation_data["MCU ctrls"] + [rotation_data["pivot"]])
        expected_master_circuit.compose(Xcirc, inplace=True)

    print("Expected circuit:")
    print(expected_master_circuit.draw())

    # Create master circuit via the magnetic trotter step code.
    lattice_def = LatticeDef(2, 3)
    lattice_encoder = LatticeStateEncoder(
        IRREP_TRUNCATION_DICT_1_3_3BAR,
        dummy_phys_states,
        lattice=lattice_def)
    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    circ_mgr = LatticeCircuitManager(lattice_encoder,
                                     dummy_mag_hamiltonian)
    master_circuit = circ_mgr.create_blank_full_lattice_circuit(
        lattice_registers)
    circ_mgr.apply_magnetic_trotter_step(
        master_circuit,
        lattice_registers,
        optimize_circuits=False
    )
    print("Obtained circuit:")
    print(master_circuit.draw())

    # Checking equivalence via helper methods for
    # (1) flattening a circuit down to a single register and (2) comparing
    # logical equivalence of two circuits.
    assert _check_circuits_logically_equivalent(_flatten_circuit(master_circuit), expected_master_circuit), "Encountered inequivalent circuits. Expected:\n" \
        f"{expected_master_circuit.draw()}\nObtained:\n" \
        f"{master_circuit.draw()}"
    print("Test passed.")


def _test_apply_magnetic_trotter_step_d_2_small_lattice():
    raise NotImplementedError()


def _run_tests():
    #_test_create_blank_full_lattice_circuit_has_promised_register_order()
    #_test_apply_magnetic_trotter_step_d_3_2_large_lattice()
    _test_apply_magnetic_trotter_step_d_3_2_small_lattice()
    #_test_apply_magnetic_trotter_step_d_2_large_lattice()
    #_test_apply_magnetic_trotter_step_d_2_small_lattice()


if __name__ == "__main__":
    _run_tests()

    print("All tests passed.")
