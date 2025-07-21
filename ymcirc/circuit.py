"""
A collection of utilities for building circuits.
"""
from __future__ import annotations
import copy
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
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library.standard_gates import RXGate, CXGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import InverseCancellation
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
        # Parameter for the number of ancillas used in circuit. Initialized to 0.
        self._num_ancillas = 0

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

    def create_blank_full_lattice_circuit(
        self, lattice: LatticeRegisters
    ) -> QuantumCircuit:
        """
        Return a blank quantum circuit with all link and vertex registers in lattice.

        This uses an ordering which is specified in the LatticeData class which is
        a parent class of LatticeRegisters. See the documentation on LatticeData
        for details
        """
        all_lattice_registers: List[QuantumRegister] = [reg for reg in lattice]

        return QuantumCircuit(*all_lattice_registers)

    def compute_num_ancillas_needed_from_mag_trotter_step(
            self,
            master_circuit: QuantumCircuit,
            lattice: LatticeRegisters,
            control_fusion: bool = False,
            physical_states_for_control_pruning: Union[None | Set[str]] = None,
            optimize_circuits: bool = False) -> int:
        """
        Computes the number of ancilla qubits needed to perform v-chain synthesis in the master_circuit
        for MCU decomposition. Function constructs the circuit for asingle magnetic trotter step to find the minimum
        required ancillas needed.

        For MCX decomposition into v-chain, (maximum number of controls in a trotter step - 2)
        is the result (check internal givens rotation function for reference).

        Arguments:
          - master_circuit: a QuantumCircuit instance which is built from all
                            the QuantumRegister instances in lattice.
          - lattice: a LatticeRegisters instance which keeps track of all the
                     QuantumRegisters.
          - control_fusion: Optional boolian argument with the default set to False. If it's set
                            to be True, then LP families of givens rotations are first Gray code ordered,
                            then redundant controls are removed.
          - physical_states_for_control_pruning: The set of all physical states encoded as bitstrings.
                                                 If provided, control pruning of multi-control rotation
                                                 gate inside Givens rotation subcircuits will be attempted.
                                                 If the lattice is small and periodic, then duplicate control
                                                 links which are shared between vertices will be stripped
                                                 out first.
                                                 If None, no control pruning is attempted.
          - optimize_circuits: if True, run the qiskit transpiler on each
                               internal givens rotation with the maximum
                               optimization level before composing with
                               master_circuit.

        Returns:
          int
        """
        max_controls = 0

        # Copy the input circuit to avoid mutating it.
        circ_for_n_ancillas_check = copy.deepcopy(master_circuit)

        self.apply_magnetic_trotter_step(circ_for_n_ancillas_check, lattice,
                                         coupling_g=1.0, dt=1.0, optimize_circuits=optimize_circuits, 
                                         physical_states_for_control_pruning=physical_states_for_control_pruning, 
                                         control_fusion=control_fusion, cache_mag_evol_circuit = False)

        for circuit_instruction in circ_for_n_ancillas_check.data:
            if len(circuit_instruction.operation.name) >= 3 and circuit_instruction.operation.name[:3] == "mcx":
                max_controls = max(circuit_instruction.operation.num_ctrl_qubits, max_controls)

        max_ancillas_needed = max_controls - 2

        return max_ancillas_needed

    @property
    def num_ancillas(self) -> int:
        """The size of the ancilla register LatticeCircuitManager expects."""
        return self._num_ancillas

    @num_ancillas.setter
    def num_ancillas(self, n) -> None:
        """Set the size of the ancilla register LatticeCircuitManager expects."""
        # TODO test these validation checks.
        if not isinstance(n, int):
            raise TypeError(f"Number of ancillas {n} is not an integer.")
        if n < 0:
            raise ValueError(f"Number of ancillas {n} is not nonnegative.")

        self._num_ancillas = n

    def add_ancilla_register_to_quantum_circuit(self, master_circuit: QuantumCircuit) -> None:
        """
        Adds ancilla qubits to the master_circuit based on the value of self.num_ancillas.

        Note that this mutates the circuit!
        """
        master_circuit.add_register(AncillaRegister(self.num_ancillas, "anc"))

    def apply_electric_trotter_step(
        self,
        master_circuit: QuantumCircuit,
        lattice: LatticeRegisters,
        hamiltonian: list[float],
        coupling_g: float = 1.0,
        dt: float = 1.0,
        electric_gray_order: bool = False
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
            - hamiltonian: Pauli decompositon of the single link electric
                          Hamiltonian. The list hamiltonian is contains
                          coefficients s.t. for hamiltonian[i] = coeff, coeff
                          is coeff of bitstring(i) with 'Z'=1 and 'I'=0 in the
                          bitstring.
            - coupling_g: The value of the strong coupling constant.
            - dt: The size of the Trotter time step.
            - electric_gray_order: The Pauli bitstrings corresponding to the Pauli
                                    decomposition of the electric hamiltonian will 
                                    be gray-code ordered if this option is set to be
                                    True. This option is False by default.


        Returns:
            A new QuantumCircuit instance which is master_circuit with the
            electric Trotter step appended.
        """
        N = int(np.log2(len(hamiltonian)))
        angle_mod = ((coupling_g**2) / 2) * dt
        local_circuit = QuantumCircuit(N)

        # Use the index of the local Pauli-decomposed electric hamiltonian to generate the Pauli bitstrings.
        pauli_bitstring_list = [str("{0:0" + str(N) + "b}").format(i) for i in range(len(hamiltonian))]
        pauli_decomposed_hamiltonian = zip(pauli_bitstring_list,hamiltonian)
        # Gray-Order the Pauli-bitstrings if electric_gray_order == True.
        if electric_gray_order == True:
            pauli_decomposed_hamiltonian = sorted(pauli_decomposed_hamiltonian,key=lambda x: gray_to_index(x[0]))

        # The parity circuit primitive of CXs and Zs.
        for pauli_bitstring, coeff in pauli_decomposed_hamiltonian:
            locs = [
                loc
                for loc, bit in enumerate(pauli_bitstring)
                if bit == "1"
            ]
            for j in locs[1:]:
                local_circuit.cx(j, locs[0])
            if len(locs) != 0:
                local_circuit.rz(2 * angle_mod * coeff, locs[0])
            for j in locs[1:]:
                local_circuit.cx(j, locs[0])

        cancel_cx = PassManager([InverseCancellation([CXGate()])])
        local_circuit = cancel_cx.run(local_circuit)

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
                        *c_link_qubits,
                        *master_circuit.ancillas
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
        if (self.num_ancillas > 0):
            plaquette_local_rotation_circuit.add_register(AncillaRegister(self.num_ancillas))
        for lp_fam, lp_bin_w_angle in lp_bin.items():
            if control_fusion is True:
                fused_circ_for_lp_fam = givens_fused_controls(
                    lp_bin_w_angle, lp_fam, physical_states_for_control_pruning, self.num_ancillas,
                )
                plaquette_local_rotation_circuit.compose(
                    fused_circ_for_lp_fam, inplace=True
                )
            else:
                # If control fusion is turned off, givens rotation is applied individually
                # to all bitstrings.
                for bs1, bs2, angle in lp_bin_w_angle:
                    bs1_bs2_circuit = givens(
                        bs1, bs2, angle, physical_states_for_control_pruning, self.num_ancillas,
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
            angle = -matrix_elem * (1 /  (coupling_g**2)) * dt
            lp_fam = compute_LP_family(bit_string_1, bit_string_2)
            if lp_fam not in lp_bin.keys():
                lp_bin[lp_fam] = []
            lp_bin[lp_fam].append((bit_string_1, bit_string_2, angle))
        return lp_bin
