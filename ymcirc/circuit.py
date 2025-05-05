"""
A collection of utilities for building circuits.
"""
from __future__ import annotations
import copy
from ymcirc._abstract import LatticeDef
from ymcirc.conventions import PlaquetteState, LatticeStateEncoder, IRREP_TRUNCATION_DICT_1_3_3BAR, ONE, THREE, THREE_BAR
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
from qiskit.circuit.library.standard_gates import RXGate, RYGate, RZGate, MCXGate
from typing import List, Tuple, Set, Union, Dict
import numpy as np

# A list of tuples: (state bitstring1, state bitstring2, matrix element)
HamiltonianData = List[Tuple[str, str, float]]


class LatticeCircuitManager:
    """Class for creating quantum simulation circuits from LatticeRegister instances."""
    def __init__(
        self, lattice_encoder: LatticeStateEncoder, mag_hamiltonian: HamiltonianData
    ):
        """
        Create via a LatticeStateEncoder instance and magnetic Hamiltonian matrix elements.

        If the lattice defined by lattice_encoder is small and periodic, then the data in mag_hamiltonian
        will be filtered for consistency. "Small and periodic" means that the lattice has
        periodic boundary conditions, and is small enough that it is possible for a single
        physical lattice link to appear as control links on two different vertices in a given
        plaquette. In this situation, all matrix elements in the magnetic Hamiltonian
        for which the initial or the final state have distinct link state data on one of
        these "shared" control links are discarded, and "duplicate" control links
        are removed from the binary encoding of the plaquette states. This duplicate
        removal is done by removing all but the first instance of a repeated control
        link from the plaquette state.
        """
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
        self._lattice_is_small = False
        self._lattice_is_periodic = False
        lattice_size_threshold_for_smallness = 2
        if not lattice_encoder.lattice_def.all_boundary_conds_periodic:
            raise NotImplementedError("Lattices with nonperiodic or mixed boundary conditions not yet supported.")
        else:
            self._lattice_is_periodic = True
        match lattice_encoder.lattice_def.dim:
            case 1.5:
                lattice_size = lattice_encoder.lattice_def.shape[0]
            case 2:
                lattice_size = lattice_encoder.lattice_def.shape[0]
                if lattice_size != lattice_encoder.lattice_def.shape[1]:
                    raise NotImplementedError("Non-square dim 2 lattices not yet supported.")
            case _:
                raise NotImplementedError(f"Dim {lattice_encoder.lattice_def.dim} lattice not yet supported.")
        self._lattice_is_small = True if lattice_size <= lattice_size_threshold_for_smallness else False

        
        if self._lattice_is_small is True and self._lattice_is_periodic is True:
            # Filter out magnetic Hamiltonian terms which are inconsistent (repeated control links must have the same value)
            filtered_and_trimmed_mag_hamiltonian: HamiltonianData = []
            for matrix_element in self._mag_hamiltonian:
                final_plaquette_state = lattice_encoder.decode_bit_string_to_plaquette_state(matrix_element[0])
                initial_plaquette_state = lattice_encoder.decode_bit_string_to_plaquette_state(matrix_element[1])
                final_state_has_inconsistent_controls = self._plaquette_state_has_inconsistent_controls(final_plaquette_state)
                initial_state_has_inconsistent_controls = self._plaquette_state_has_inconsistent_controls(initial_plaquette_state)
                
                if (final_state_has_inconsistent_controls is True) or (initial_state_has_inconsistent_controls is True):
                    # Matrix element includes plaquette states that are nonsensical on a small, periodic lattice. Skip it.
                    continue
                else:
                    # Matrix element is consistent on shared controls. Trim out duplicate control links, re-encode plaquettes as bitstring, and keep.
                    final_plaquette_state_trimmed_c_links = self._discard_duplicate_controls_from_plaquette_state(final_plaquette_state)
                    initial_plaquette_state_trimmed_c_links = self._discard_duplicate_controls_from_plaquette_state(initial_plaquette_state)
                    consistent_and_trimmed_matrix_element = (
                        lattice_encoder.encode_plaquette_state_as_bit_string(final_plaquette_state_trimmed_c_links, override_n_c_links_validation=True),
                        lattice_encoder.encode_plaquette_state_as_bit_string(initial_plaquette_state_trimmed_c_links, override_n_c_links_validation=True),
                        matrix_element[2]
                    )
                    filtered_and_trimmed_mag_hamiltonian.append(consistent_and_trimmed_matrix_element)

            # Update the magnetic Hamiltonian data with the trimmed, consistent matrix elements.
            self._mag_hamiltonian = filtered_and_trimmed_mag_hamiltonian

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

        This is done by iterating over every lattice vertex. At each vertex,
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
          - physical_states_for_control_pruning: The set of all physical states encoded as bitstrings.
                                                 If provided, control pruning of multi-control rotation
                                                 gate inside Givens rotation subcircuits will be attempted.
                                                 If the lattice is small and periodic, then duplicate control
                                                 links which are shared between vertices will be stripped
                                                 out first.
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
        # Strip out redundant control links if physical state data provided and the lattice
        # is both small enough and periodic (so that the same physical links can be distinct controls).
        if (physical_states_for_control_pruning is not None) and (self._lattice_is_periodic is True) and (self._lattice_is_small is True):
            stripped_physical_states = []
            for plaquette_string in physical_states_for_control_pruning:
                plaquette_state = self._encoder.decode_bit_string_to_plaquette_state(plaquette_string)
                if self._plaquette_state_has_inconsistent_controls(plaquette_state) is True:
                    continue

                plaquette_state_c_links_stripped = self._discard_duplicate_controls_from_plaquette_state(plaquette_state)
                plaquette_state_c_links_stripped_bit_string = self._encoder.encode_plaquette_state_as_bit_string(
                    plaquette_state_c_links_stripped,
                    override_n_c_links_validation=True
                )
                stripped_physical_states.append(plaquette_state_c_links_stripped_bit_string)
            stripped_physical_states = set(stripped_physical_states)
            if len(stripped_physical_states) == 0:
                physical_states_for_control_pruning = None
            else:
                physical_states_for_control_pruning = stripped_physical_states
        
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

            # For each plaquette, apply the the local Trotter step circuit.
            for plaquette in plaquettes:
                # Get qubits for the current plaquette.
                
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
                for c_link_idx, register in enumerate(plaquette.control_links_ordered):
                    # If lattice is small and has PBCs, skip redundant c_link registers.
                    if (self._lattice_is_small is True) and (self._lattice_is_periodic is True):
                        redundant_c_link_idxes_by_dim_dict = {
                            1.5 : [1, 3],
                            2: [3, 5, 6, 7]
                        }
                        try:
                            current_c_link_is_redundant = c_link_idx in redundant_c_link_idxes_by_dim_dict[self._encoder.lattice_def.dim]
                        except KeyError:
                            raise NotImplementedError(f"Dim {self._encoder.lattice_def.dim} lattice not yet supported.")
                        if current_c_link_is_redundant is True:
                            continue
                    
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
        # Iterate over all LP bins and apply givens rotation.
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

    def _plaquette_state_has_inconsistent_controls(self, plaquette: PlaquetteState) -> bool:
        """
        True if "shared" control links have different states; False otherwise.

        For d=3/2, this corresponds to c1 == c2 and c3 == c4.

        For d=2, this corresponds to c1 == c4, c2 == c7, c3 == c6, and c5 == c8.
        
        Note that this only makes sense on a small, periodic lattice, so a ValueError
        is raised if the lattice fails those checks.
        """
        if (self._lattice_is_periodic is False) or (self._lattice_is_small is False):
            raise ValueError("Plaquette state consistency check only makes sense on a small, periodic lattice.")

        c_links = plaquette[2]
        match self._encoder.lattice_def.dim:
            case 1.5:
                plaquette_state_has_inconsistent_controls = (
                    (c_links[0] != c_links[1]) or
                    (c_links[2] != c_links[3])
                )
            case 2:
                plaquette_state_has_inconsistent_controls = (
                    (c_links[0] != c_links[3]) or
                    (c_links[1] != c_links[6]) or
                    (c_links[2] != c_links[5]) or
                    (c_links[4] != c_links[7])
                )
            case _:
                raise NotImplementedError(f"Dim {self._encoder.lattice_def.dim} lattice not yet supported.")

        return plaquette_state_has_inconsistent_controls

    def _discard_duplicate_controls_from_plaquette_state(self, plaquette: PlaquetteState) -> PlaquetteState:
        """
        Return a new instances of the plaquette where duplicate control link data has been discarded.

        Only the first instance of a duplicate control link is kept. For example, on a 2-plaquette d=3/2
        lattice with PBCs, only c1 and c3 are kept since c1 == c2 and c3 == c4. On a 4-plaquette d=2
        lattice with PBCs, only c1, c2, c3, and c5 are kept since c1 == c4, c2 == c7, c3 == c6,
        and c5 == c8.

        Since this only makes sense on a small, periodic lattice, a ValueError
        is raised if the lattice is not small and periodic.
        """
        if (self._lattice_is_periodic is False) or (self._lattice_is_small is False):
            raise ValueError("Plaquette state consistency check only makes sense on a small, periodic lattice.")

        vertex_multiplicities, a_links, c_links = plaquette
        match self._encoder.lattice_def.dim:
            case 1.5:
                physical_c_links = (c_links[0], c_links[2])
            case 2:
                physical_c_links = (c_links[0], c_links[1], c_links[2], c_links[4])
            case _:
                raise NotImplementedError(f"Dim {self._encoder.lattice_def.dim} lattice not yet supported.")
            
        plaquette_with_filtered_c_links = (vertex_multiplicities, a_links, physical_c_links)
        return plaquette_with_filtered_c_links
        

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
    #      3c. Repeat this exercise with the multi-control rotation, where the type of ladder or projector operator involved determines the control states.
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
        (  # Matches the second encoded state in the dummy magnetic hamiltonian.
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
                "MCU ctrls": [8, 0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17] # on ctrls, followed by off, with pivot and distant link regs skipped.
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

        pivot_qubit = [rotation_data["pivot"]]
        angle = expected_rotation_gates["angle"]
        ctrls = rotation_data["MCU ctrls"]
        num_ctrls = len(rotation_data["MCU ctrls"])
        ctrl_state = expected_rotation_gates["MCU ctrl state"]

        circ_with_mcx = QuantumCircuit(18)
        circ_with_mcx.append(RZGate(-1.0*np.pi/2.0), pivot_qubit)
        circ_with_mcx.append(RYGate(-1.0*angle/2.0), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RYGate(1.0*angle/2.0), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RZGate(1.0*np.pi/2.0), pivot_qubit)

        # Construct current expected givens rotation.
        expected_master_circuit.compose(Xcirc, inplace=True)
        expected_master_circuit.compose(circ_with_mcx, inplace=True)
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


def _test_apply_magnetic_trotter_step_d_3_2_small_lattice():
    print(
        "Checking that application of magnetic Trotter step works for d=3/2 "
        "on a small lattice where some control links are repeated in each "
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
    #      3c. Repeat this exercise with the multi-control rotation, where the type of ladder or projector operator involved determines the control states (raising to get to final state is on, projector onto 1 is on).
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
        (  # Matches the second encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            (THREE, THREE_BAR, THREE, THREE)
        ),
        (  # Matches the third encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            (THREE, THREE, ONE, ONE)
        )
    ]
    expected_master_circuit = QuantumCircuit(12)
    # Only expecting one rotation per plaquette, yielding two total Givens rotations.
    expected_rotation_gates = {  # Data for constructing the expected circuit.
        "angle": -0.165,
        "MCU ctrl state": "00000000011",  # Little endian per qiskit convention.
        "givens rotations": [
            {
                "pivot": 1,
                "CX targets": [8, 9, 5, 2, 3, 6],
                "MCU ctrls": [8, 3, 0, 2, 4, 5, 6, 7, 9, 10, 11] # on ctrls first, followed by off, with pivot skipped.
            },
            {
                "pivot": 7,
                "CX targets": [2, 3, 11, 8, 9, 0],
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

        pivot_qubit = [rotation_data["pivot"]]
        angle = expected_rotation_gates["angle"]
        ctrls = rotation_data["MCU ctrls"]
        num_ctrls = len(rotation_data["MCU ctrls"])
        ctrl_state = expected_rotation_gates["MCU ctrl state"]

        circ_with_mcx = QuantumCircuit(12)
        circ_with_mcx.append(RZGate(-1.0*np.pi/2.0), pivot_qubit)
        circ_with_mcx.append(RYGate(-1.0*angle/2.0), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RYGate(1.0*angle/2.0), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RZGate(1.0*np.pi/2.0), pivot_qubit)

        # Construct current expected givens rotation.
        expected_master_circuit.compose(Xcirc, inplace=True)
        expected_master_circuit.compose(circ_with_mcx, inplace=True)
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
    #      3c. Repeat this exercise with the multi-control rotation, where the type of ladder or projector operator involved determines the control states.
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
        (  # Matches the second encoded state in the dummy magnetic hamiltonian.
            (0, 0, 1, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, ONE),
            (THREE, THREE_BAR, THREE, THREE, ONE, ONE, ONE, THREE)
        )
    ]
    expected_master_circuit = QuantumCircuit(45)
    expected_rotation_gates = {  # Data for constructing the expected circuit.
        "angle": -0.165,
        "MCU ctrl state": "000000000000000000000000011",  # Little endian per qiskit convention.
        "givens rotations": [
            {
                "pivot": 20,
                "CX targets": [2, 18, 19, 7, 31, 14, 28, 16],
                "MCU ctrls": [18, 36] + [0, 15, 5, 1, 2, 19, 6, 7, 3, 4, 31, 32, 13, 14, 28, 29, 16, 17, 21, 22, 23, 24, 8, 9, 37] # on ctrls first, followed by off, with pivot skipped.
            },
            {
                "pivot": 25,
                "CX targets": [7, 23, 24, 12, 36, 4, 18, 21],
                "MCU ctrls": [23, 41] + [5, 20, 10, 6, 7, 24, 11, 12, 8, 9, 36, 37, 3, 4, 18, 19, 21, 22, 26, 27, 28, 29, 13, 14, 42]
            },
            {
                "pivot": 15,
                "CX targets": [12, 28, 29, 2, 41, 9, 23, 26],
                "MCU ctrls": [28, 31] + [10, 25, 0, 11, 12, 29, 1, 2, 13, 14, 41, 42, 8, 9, 23, 24, 26, 27, 16, 17, 18, 19, 3, 4, 32]
            },
            {
                "pivot": 35,
                "CX targets": [17, 33, 34, 22, 1, 29, 43, 31],
                "MCU ctrls": [33, 6] + [15, 30, 20, 16, 17, 34, 21, 22, 18, 19, 1, 2, 28, 29, 43, 44, 31, 32, 36, 37, 38, 39, 23, 24, 7]
            },
            {
                "pivot": 40,
                "CX targets": [22, 38, 39, 27, 6, 19, 33, 36],
                "MCU ctrls": [38, 11] + [20, 35, 25, 21, 22, 39, 26, 27, 23, 24, 6, 7, 18, 19, 33, 34, 36, 37, 41, 42, 43, 44, 28, 29, 12]
            },
            {
                "pivot": 30,
                "CX targets": [27, 43, 44, 17, 11, 24, 38, 41],
                "MCU ctrls": [43, 1] + [25, 40, 15, 26, 27, 44, 16, 17, 28, 29, 11, 12, 23, 24, 38, 39, 41, 42, 31, 32, 33, 34, 18, 19, 2]
            },
            {
                "pivot": 5,
                "CX targets": [32, 3, 4, 37, 16, 44, 13, 1],
                "MCU ctrls": [3, 21] + [30, 0, 35, 31, 32, 4, 36, 37, 33, 34, 16, 17, 43, 44, 13, 14, 1, 2, 6, 7, 8, 9, 38, 39, 22]
            },
            {
                "pivot": 10,
                "CX targets": [37, 8, 9, 42, 21, 34, 3, 6],
                "MCU ctrls": [8, 26] + [35, 5, 40, 36, 37, 9, 41, 42, 38, 39, 21, 22, 33, 34, 3, 4, 6, 7, 11, 12, 13, 14, 43, 44, 27]
            },
            {
                "pivot": 0,
                "CX targets": [42, 13, 14, 32, 26, 39, 8, 11],
                "MCU ctrls": [13, 16] + [40, 10, 30, 41, 42, 14, 31, 32, 43, 44, 26, 27, 38, 39, 8, 9, 11, 12, 1, 2, 3, 4, 33, 34, 17]
            }
        ]
    }
    for rotation_data in expected_rotation_gates["givens rotations"]:
        # Build subcircuits.
        Xcirc = QuantumCircuit(45)
        for target in rotation_data["CX targets"]:
            Xcirc.cx(
                control_qubit=rotation_data["pivot"],
                target_qubit=target
            )

        pivot_qubit = [rotation_data["pivot"]]
        angle = expected_rotation_gates["angle"]
        ctrls = rotation_data["MCU ctrls"]
        num_ctrls = len(rotation_data["MCU ctrls"])
        ctrl_state = expected_rotation_gates["MCU ctrl state"]

        circ_with_mcx = QuantumCircuit(45)
        circ_with_mcx.append(RZGate(-1.0*np.pi/2.0), pivot_qubit)
        circ_with_mcx.append(RYGate(-1.0*angle/2.0), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RYGate(1.0*angle/2.0), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RZGate(1.0*np.pi/2.0), pivot_qubit)

        # Construct current expected givens rotation.
        expected_master_circuit.compose(Xcirc, inplace=True)
        expected_master_circuit.compose(circ_with_mcx, inplace=True)
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
    print(
        "Checking that application of magnetic Trotter step works for d=2 "
        "on a small lattice where some control links are repeated in each "
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
    #      3c. Repeat this exercise with the multi-control rotation, where the type of ladder or projector operator involved determines the control states.
    # Ask yourself if you REALLY feel like doing all that before mucking about with this test data.
    dummy_mag_hamiltonian = [
        ("0000" + "00100000" + "0000000000000010", "0010" + "01010100" + "1001101000000010", 0.33), # One matrix element, plaquette has v, a_link, and c_link substrings. Should get filtered out based on c_link consistency.
        ("0000" + "00100000" + "0000000010000010", "0010" + "01010100" + "1001101000100100", 0.33)  # One matrix element, plaquette has v, a_link, and c_link substrings. Should not get filtered out based on c_link consistency.
    ]
    dummy_phys_states = [
        (  # Matches the first encoded state in the dummy magnetic hamiltonian that isn't discarded.
            (0, 0, 0, 0),
            (ONE, THREE, ONE, ONE),
            (ONE, ONE, ONE, ONE, THREE, ONE, ONE, THREE)
        ),
        (  # Matches the second encoded state in the dummy magnetic hamiltonian that isn't discarded.
            (0, 0, 1, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, ONE),
            (THREE, THREE_BAR, THREE, THREE, ONE, THREE, THREE_BAR, ONE)
        )
    ]
    expected_master_circuit = QuantumCircuit(20)
    expected_rotation_gates = {  # Data for constructing the expected circuit.
        "angle": -0.165,
        "MCU ctrl state": "0000000000000000011",  # Little endian per qiskit convention.
        "givens rotations": [
            {
                "pivot": 15,
                "CX targets": [2, 13, 14, 7, 11, 9, 18, 16],
                "MCU ctrls": [13, 16] + [0, 10, 5, 1, 2, 14, 6, 7, 3, 4, 11, 12, 8, 9, 18, 19, 17] # on ctrls first, followed by off, with pivot skipped.
            },
            {
                "pivot": 10,
                "CX targets": [7, 18, 19, 2, 16, 4, 13, 11],
                "MCU ctrls": [18, 11] + [5, 15, 0, 6, 7, 19, 1, 2, 8, 9, 16, 17, 3, 4, 13, 14, 12] # on ctrls first, followed by off, with pivot skipped.
            },
            {
                "pivot": 5,
                "CX targets": [12, 3, 4, 17, 1, 19, 8, 6],
                "MCU ctrls": [3, 6] + [10, 0, 15, 11, 12, 4, 16, 17, 13, 14, 1, 2, 18, 19, 8, 9, 7] # on ctrls first, followed by off, with pivot skipped.
            },
            {
                "pivot": 0,
                "CX targets": [17, 8, 9, 12, 6, 14, 3, 1],
                "MCU ctrls": [8, 1] + [15, 5, 10, 16, 17, 9, 11, 12, 18, 19, 6, 7, 13, 14, 3, 4, 2] # on ctrls first, followed by off, with pivot skipped.
            },
        ]
    }
    for rotation_data in expected_rotation_gates["givens rotations"]:
        # Build subcircuits.
        Xcirc = QuantumCircuit(20)
        for target in rotation_data["CX targets"]:
            Xcirc.cx(
                control_qubit=rotation_data["pivot"],
                target_qubit=target
            )

        pivot_qubit = [rotation_data["pivot"]]
        angle = expected_rotation_gates["angle"]
        ctrls = rotation_data["MCU ctrls"]
        num_ctrls = len(rotation_data["MCU ctrls"])
        ctrl_state = expected_rotation_gates["MCU ctrl state"]

        circ_with_mcx = QuantumCircuit(20)
        circ_with_mcx.append(RZGate(-1.0*np.pi/2.0), pivot_qubit)
        circ_with_mcx.append(RYGate(-1.0*angle/2.0), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RYGate(1.0*angle/2.0), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RZGate(1.0*np.pi/2.0), pivot_qubit)

        # Construct current expected givens rotation.
        expected_master_circuit.compose(Xcirc, inplace=True)
        expected_master_circuit.compose(circ_with_mcx, inplace=True)
        expected_master_circuit.compose(Xcirc, inplace=True)

    print("Expected circuit:")
    print(expected_master_circuit.draw())

    # Create master circuit via the magnetic trotter step code.
    lattice_def = LatticeDef(2, 2)
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


def _run_tests():
    _test_create_blank_full_lattice_circuit_has_promised_register_order()
    _test_apply_magnetic_trotter_step_d_3_2_large_lattice()
    _test_apply_magnetic_trotter_step_d_3_2_small_lattice()
    _test_apply_magnetic_trotter_step_d_2_large_lattice()
    _test_apply_magnetic_trotter_step_d_2_small_lattice()


if __name__ == "__main__":
    _run_tests()

    print("All tests passed.")
