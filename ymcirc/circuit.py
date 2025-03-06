"""
A collection of utilities for building circuits.
"""
from __future__ import annotations
import copy
from ymcirc._abstract import LatticeDef
from ymcirc.conventions import LatticeStateEncoder, IRREP_TRUNCATION_DICT_1_3_3BAR, ONE, THREE, THREE_BAR
from ymcirc.lattice_registers import LatticeRegisters
from ymcirc.givens import givens
from ymcirc._abstract.lattice_data import Plaquette
from ymcirc.utilities import _check_circuits_logically_equivalent, _flatten_circuit
from math import ceil
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister, ControlledGate
from qiskit.circuit.library.standard_gates import RXGate
from typing import List, Tuple, Set
import numpy as np

# A list of tuples: (state bitstring1, state bitstring2, matrix element)
HamiltonianData = List[Tuple[str, str, float]]


class LatticeCircuitManager:
    """Class for creating quantum simulation circuits from LatticeRegister instances."""
    
    def __init__(self, lattice_encoder: LatticeStateEncoder, mag_hamiltonian: HamiltonianData):
        """Create via a LatticeStateEncoder instance and magnetic Hamiltonian matrix elements."""
        # Copies to avoid inadvertently changing the behavior of the
        # LatticeCircuitManager instance.
        self._encoder = copy.deepcopy(lattice_encoder)
        self._mag_hamiltonian = copy.deepcopy(mag_hamiltonian)

    # TODO modularize the logic for walking through a lattice.
    def create_blank_full_lattice_circuit(self, lattice: LatticeRegisters) -> QuantumCircuit:
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
                has_no_vertical_periodic_link_three_halves_case = \
                    lattice.dim == 1.5 and positive_direction > 1 and vertex_address[1] == 1
                if has_no_vertical_periodic_link_three_halves_case:
                    continue

                current_link_reg = lattice.get_link((vertex_address, positive_direction))
                all_lattice_registers.append(current_link_reg)

        return QuantumCircuit(*all_lattice_registers)

    def apply_electric_trotter_step(
            self,
            master_circuit: QuantumCircuit,
            lattice: LatticeRegisters,
            hamiltonian: list[float],
            coupling_g: float = 1.0,
            dt: float = 1.0) -> None:
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
            locs = [loc for loc, bit in enumerate(str('{0:0' + str(N) + 'b}').format(i)) if bit=='1']
            for j in locs[:-1]:
                local_circuit.cx(j, locs[-1])
            if len(locs) != 0:
                local_circuit.rz(2*angle_mod*hamiltonian[i], locs[-1])
            for j in locs[:-1]:
                local_circuit.cx(j, locs[-1])

        # Loop over links for electric Hamiltonian
        for link_address in lattice.link_addresses:
            link_qubits = [qubit for qubit in lattice.get_link((link_address[0], link_address[1]))]
            master_circuit.compose(
                        local_circuit,
                        qubits=link_qubits,
                        inplace=True
                    )

    # TODO Can we get the circuits in a parameterized way?
    def apply_magnetic_trotter_step(
            self,
            master_circuit: QuantumCircuit,
            lattice: LatticeRegisters,
            coupling_g: float = 1.0,
            dt: float = 1.0,
            optimize_circuits: bool = True,
            physical_states_for_control_pruning: Union[None | Set[str]] = None
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

        Returns:
          A new QuantumCircuit instance which is master_circuit with the
          Trotter step appended.
        """
        # Vertex iteration loop.
        for vertex_address in lattice.vertex_addresses:
            # Skip creating "top vertex" plaquettes for d=3/2.
            has_no_vertical_periodic_link_three_halves_case = \
                lattice.dim == 1.5 and vertex_address[1] == 1
            if has_no_vertical_periodic_link_three_halves_case:
                continue

            # Get the plaquettes for the current vertex.
            print(f"Fetching all positive plaquettes at vertex {vertex_address}.")
            has_only_one_positive_plaquette = lattice.dim == 1.5 or lattice.dim == 2
            if has_only_one_positive_plaquette:
                plaquettes: List[Plaquette] = [lattice.get_plaquettes(vertex_address, 1, 2)]
            else:
                plaquettes: List[Plaquette] = lattice.get_plaquettes(vertex_address)
            print(f"Found {len(plaquettes)} plaquette(s).")

            # TODO test stitching logic when the same control reg shows up on multiple vertices.
            # For each plaquette, apply the the local Trotter step circuit.
            for plaquette in plaquettes:

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

                # Append a Givens rotation circuit for each magnetic Hamiltonian
                # matrix element.
                for bit_string_1, bit_string_2, matrix_elem in self._mag_hamiltonian:
                    if physical_states_for_control_pruning is not None:
                        physical_control_qubits = LatticeCircuitManager.prune_controls(
                            bit_string_1=bit_string_1,
                            bit_string_2=bit_string_2,
                            encoded_physical_states=physical_states_for_control_pruning
                        )
                    else:
                        physical_control_qubits = None
                    angle = -matrix_elem * (1 / (2 * (coupling_g**2))) * dt
                    # We use the reverse argument to account for the little-endianness
                    # of QuantumRegisters implemented by qiskit.
                    plaquette_local_rotation_circuit = givens(
                        bit_string_1,
                        bit_string_2,
                        angle,
                        reverse=False,
                        physical_control_qubits=physical_control_qubits)
                    if optimize_circuits is True:
                        plaquette_local_rotation_circuit = transpile(
                            plaquette_local_rotation_circuit, optimization_level=3)

                    # Stitch the Givens rotation into master circuit.
                    master_circuit.compose(
                        plaquette_local_rotation_circuit,
                        qubits=[
                            *vertex_multiplicity_qubits,
                            *a_link_qubits,
                            *c_link_qubits
                        ],
                        inplace=True
                    )

    @staticmethod
    def prune_controls(
            bit_string_1: str,
            bit_string_2: str,
            encoded_physical_states: Set[str]) -> Set[int]:
        """
        Perform control pruning on circ to reduce multi-control unitaries (MCUs).

        This algorithm decreases the two-qubit gate depth of resulting
        circuits by determining the set of necessary control qubits to
        implement a Givens rotation between bit_string_1 and bit_string_2
        without inducing any rotations between states in encoded_physical_states.

        Input:
            - bit_string_1: a physical state encoded in a bit string.
            - bit_string_2: a physical state encoded in a bit string.
            - encoded_physical_states: a set of physical states encoded in a bit string.
        Output:
            - A set of necessary controls to implement a Givens rotation between
              the two bit strings without inducing rotations among states in
              encoded_physical_states.

        Note that errors are raised if the bit string lengths are unequal, or there are non-bit
        characters in state encodings.
        """
        # Validate input data.
        bit_string_inputs_have_unequal_lengths = (len(bit_string_1) != len(bit_string_2)) or \
            (any([len(bit_string_1) != len(encoded_state) for encoded_state in encoded_physical_states]))
        string_inputs_have_non_bit_chars = any([char not in {"0", "1"} for char in bit_string_1]) or \
            any([char not in {"0", "1"} for char in bit_string_2]) or \
            any([char not in {"0", "1"} for bit_string in encoded_physical_states for char in bit_string])
        if bit_string_inputs_have_unequal_lengths:
            raise IndexError(
                "All input state data must be the same length. Encountered:\n"
                f"bit_string_1 = {bit_string_1}\n"
                f"bit_string_2 = {bit_string_2}\n"
                f"encoded_physical_states = {encoded_physical_states}"
            )
        elif string_inputs_have_non_bit_chars:
            raise ValueError(
                "All input state data must be interpretable as bit strings. Encountered:\n"
                f"bit_string_1 = {bit_string_1}\n"
                f"bit_string_2 = {bit_string_2}\n"
                f"encoded_physical_states = {encoded_physical_states}"
            )

        # Find the LP family of the representative, and identify which qubit
        # "L" first appears on.
        representative_P = bit_string_1
        representative_LP_family = LatticeCircuitManager.compute_LP_family(bit_string_1, bit_string_2)
        q_prime_idx = next((idx for idx, LP_val in enumerate(representative_LP_family) if LP_val[0] == "L"), -1)
        if q_prime_idx == -1:
            raise ValueError("Attempting to prune controls on two identical bit strings.")

        # Use the LP family and q', compute the states tilde_p after the prefix
        # circuit. Do this to the representative too.
        representative_P_tilde = LatticeCircuitManager._apply_LP_family_to_bit_string(representative_LP_family, q_prime_idx, representative_P)
        P_tilde = {
         LatticeCircuitManager._apply_LP_family_to_bit_string(representative_LP_family, q_prime_idx, phys_state) for phys_state in encoded_physical_states
        }

        # Iterate through tilde_p, and identify bitstrings that differ at
        # ONLY one qubit other than q'. Add these to the set Q.
        Q_set = set()
        for phys_tilde_state in P_tilde:
            n_bit_differences = 0
            for idx, (rep_char, phys_char) in enumerate(zip(representative_P_tilde, phys_tilde_state)):
                if idx == q_prime_idx:
                    continue
                elif rep_char != phys_char:
                    n_bit_differences += 1
                    diff_bit_idx = idx
            if n_bit_differences == 1:
                Q_set.add(diff_bit_idx)

        # Use Q to eliminate strings from tilde_p that differ at any qubit
        # in Q.
        P_tilde = LatticeCircuitManager._eliminate_phys_states_that_differ_from_rep_at_Q_idx(
            representative_P_tilde, P_tilde, Q_set)

        # Eliminate states from tilde_p which differ at a qubit in Q.
        bit_string_1_tilde = LatticeCircuitManager._apply_LP_family_to_bit_string(representative_LP_family, q_prime_idx, bit_string_1)
        bit_string_2_tilde = LatticeCircuitManager._apply_LP_family_to_bit_string(representative_LP_family, q_prime_idx, bit_string_2)
        should_continue_loop = True
        while should_continue_loop:
            # Find most frequently differing bit idx
            index_counts = [0,] * len(representative_P_tilde)
            for phys_tilde_state in P_tilde:
                for idx, (rep_char, phys_char) in enumerate(zip(representative_P_tilde, phys_tilde_state)):
                    if idx == q_prime_idx:
                        continue
                    elif rep_char != phys_char:
                        index_counts[idx] += 1

            # If there are differences, add the most frequently differing bit
            # idx to Q, and eliminate P_tilde that differ at an index in Q.
            if not all([count == 0 for count in index_counts]):
                max_counts_idx = index_counts.index(max(index_counts))
                Q_set.add(max_counts_idx)

            # Update loop params.
            P_tilde = LatticeCircuitManager._eliminate_phys_states_that_differ_from_rep_at_Q_idx(
                representative_P_tilde, P_tilde, Q_set)
            should_continue_loop = not all([tilde_state in {bit_string_1_tilde, bit_string_2_tilde} for tilde_state in P_tilde])

        return Q_set

    @staticmethod
    def compute_LP_family(bit_string_1: str, bit_string_2: str) -> List[str]:
        """
        Compare each element of bit_string_1 and bit_string_2 to obtain LP family.

        Input:
            - Two equal-length bit strings strings bit_string_1 and bit_string 2.
        Output:
            - A list of the same length as one of the input bitstrings. Each element of
             the return list consists of the following:
                - "P0" if both inputs have a zero at that index.
                - "P1" if both inputs have a one at that index.
                - "L+" if bit_string_2 has a 1 where bit_string_1 has a 0.
                - "L-" if bit_string_2 has a 0 where bit_string_1 has a 1.
        """
        if len(bit_string_1) != len(bit_string_2):
            raise IndexError("Both input bit strings must have the same length.")

        LP_family = []
        for char_1_2_tuple in zip(bit_string_1, bit_string_2):
            match char_1_2_tuple:
                case ("0", "0"):
                    LP_family.append("P0")
                case ("1", "1"):
                    LP_family.append("P1")
                case ("0", "1"):
                    LP_family.append("L+")
                case ("1", "0"):
                    LP_family.append("L-")
                case _:
                    raise ValueError(f"Encountered non-bit character while comparing chars: {char_1_2_tuple}.")

        return LP_family

    @staticmethod
    def _apply_LP_family_to_bit_string(LP_family: List[str], q_prime_idx: int, bit_string: str) -> str:
        """
        Use LP_family to figure out how bit_string will transform under the CX
        change of basis inside Givens rotation circuits. Bit strings are read left-to-right.

        The q_prime_idx is skipped in applying the LP family. Bits in bit_string which are an L-index only
        flip if the bit at q_prime_idx has value 1.
        """
        if bit_string[q_prime_idx] == '0':
            return bit_string

        result_string_list = list(bit_string)
        for idx, op in enumerate(LP_family):
            if idx == q_prime_idx:
                continue  # Never flip the bit at this index.
            match op[0]:
                case "L":
                    result_string_list[idx] = str((int(result_string_list[idx]) + 1) % 2)  # Flip the bit.
                case "P":
                    pass  # Don't flip the bit.

        return "".join(result_string_list)

    @staticmethod
    def _eliminate_phys_states_that_differ_from_rep_at_Q_idx(
            representative: str,
            phys_states_set: Set[str],
            Q_set: Set[int]) -> Set[str]:
        """
        Return a version of phys_states_set that only contains strings which match representative
        at the indices specified by Q_set.
        """
        return set(filter(lambda phys_states_set: not any([
            rep_char != phys_char and idx in Q_set for idx, (rep_char, phys_char) in enumerate(zip(
                representative, phys_states_set))]),
                         phys_states_set))


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
        assert n_nonzero_regs == len(expected_register_names_ordered), f"Expected {len(expected_register_names_ordered)} registers. Encountered {n_nonzero_regs} registers."
        for expected_name, reg in zip(expected_register_names_ordered, nonzero_regs):
            if len(reg) == 0:
                continue
            assert expected_name == reg.name, f"Expected: {expected_name}, encountered: {reg.name}"
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


# TODO finish updating this test.
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
        ("00100001" + "00000000", "01010110" + "10011010", 0.33)  # One matrix element, plaquette only has a_link and c_link substrings.
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
        )
    ]
    expected_master_circuit = QuantumCircuit(12)
    expected_rotation_gates = {  # Data for constructing the expected circuit.
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


def _test_compute_LP_family():
    """Check computation of LP family works."""
    bs1 = "11000"
    bs2 = "10001"
    expected_LP_family = ["P1", "L-", "P0", "P0", "L+"]
    print(
        f"Checking that the bit strings {bs1} and {bs2} "
        f"yield the LP family {expected_LP_family}."
    )

    assert LatticeCircuitManager.compute_LP_family(bs1, bs2) == expected_LP_family, f"Unexpected LP family result: {LatticeCircuitManager.compute_LP_family(bs1, bs2)}"
    print("Test passed.\n")


def _test_compute_LP_family_fails_on_bad_input():
    """LP family computation should fail for unequal length strings."""
    bs1 = "11110"
    bs2 = "00"
    print(f"Checking that computing LP family of {bs1} and {bs2} raises IndexError.")

    try:
        LatticeCircuitManager.compute_LP_family(bs1, bs2)
    except IndexError as e:
        print(f"Test passed. Raised error: {e}\n")
    else:
        assert False, "IndexError not raised."


def _test_compute_LP_family_fails_for_non_bitstrings():
    """LP family computation fails for input not bit strings."""
    bs1 = "a3bb"
    bs2 = "1011"
    print(f"Checking that expected_LP_family({bs1}, {bs2}) raises a ValueError.")

    try:
        LatticeCircuitManager.compute_LP_family(bs1, bs2)
    except ValueError as e:
        print(f"Test passed. Raised error: {e}\n")
    else:
        assert False, "ValueError not raised."


def _test_prune_controls_fails_for_unequal_length_inputs():
    """IndexError should be raised if any of the bit strings in an input has a different length."""
    bs1 = "11001"
    bs2 = "11000"
    phys_states = {bs1, bs2, "11111", "1100"}
    print("Checking that a wrong-length bit string in the physical "
          "states raises an IndexError:\n"
          f"bs1 = {bs1}\n"
          f"bs2 = {bs2}\n"
          f"phys_states = {phys_states}\n")
    try:
        LatticeCircuitManager.prune_controls(bs1, bs2, phys_states)
    except IndexError as e:
        print(f"Test passed. Raised error: {e}\n")
    else:
        assert False, "IndexError not raised."

    bs1 = "11001"
    bs2 = "1100"
    phys_states = {bs1, bs2, "11111", "11000"}
    print("Checking that a wrong-length bit string in one of the input "
          "states raises an IndexError:\n"
          f"bs1 = {bs1}\n"
          f"bs2 = {bs2}\n"
          f"phys_states = {phys_states}\n")
    try:
        print(
            f"bs1 = {bs1}\n"
            f"bs2 = {bs2}\n"
            f"phys_states = {phys_states}")
        LatticeCircuitManager.prune_controls(bs1, bs2, phys_states)
    except IndexError as e:
        print(f"Test passed. Raised error: {e}\n")
    else:
        assert False, "IndexError not raised."
    try:
        print(
            f"bs1 = {bs2}\n"
            f"bs2 = {bs1}\n"
            f"phys_states = {phys_states}")
        LatticeCircuitManager.prune_controls(bs2, bs1, phys_states)
    except IndexError as e:
        print(f"Test passed. Raised error: {e}\n")
    else:
        assert False, "IndexError not raised."


def _test_prune_controls_fails_for_non_bitstrings():
    """Non bit characters in any bit strings should cause a ValueError to be raised."""
    bs1 = "11001"
    bs2 = "11000"
    phys_states = {bs1, bs2, "11111", "1100a"}
    print("Checking that a non-bit char in the physical "
          "states raises an ValueError:\n"
          f"bs1 = {bs1}\n"
          f"bs2 = {bs2}\n"
          f"phys_states = {phys_states}\n")
    try:
        LatticeCircuitManager.prune_controls(bs1, bs2, phys_states)
    except ValueError as e:
        print(f"Test passed. Raised error: {e}\n")
    else:
        assert False, "ValueError not raised."

    bs1 = "11001"
    bs2 = "110a0"
    phys_states = {bs1, bs2, "11111", "11000"}
    print("Checking that a non-bit char in one of the input "
          "states raises an ValueError:\n"
          f"bs1 = {bs1}\n"
          f"bs2 = {bs2}\n"
          f"phys_states = {phys_states}\n")
    try:
        print(
            f"bs1 = {bs1}\n"
            f"bs2 = {bs2}\n"
            f"phys_states = {phys_states}\n")
        LatticeCircuitManager.prune_controls(bs1, bs2, phys_states)
    except ValueError as e:
        print(f"Test passed. Raised error: {e}\n")
    else:
        assert False, "ValueError not raised."
    try:
        print(
            f"bs1 = {bs2}\n"
            f"bs2 = {bs1}\n"
            f"phys_states = {phys_states}\n")
        LatticeCircuitManager.prune_controls(bs2, bs1, phys_states)
    except ValueError as e:
        print(f"Test passed. Raised error: {e}\n")
    else:
        assert False, "ValueError not raised."

def _test_prune_controls_acts_as_expected():
    """Check the output of prune_controls for good input."""
    # Case 1, one control needed.
    bs1 = "11001001"
    bs2 = "11000110"
    phys_states = {bs1, bs2, "11111111"}
    expected_rep_LP_family = ["P1", "P1", "P0", "P0", "L-", "L+", "L+", "L-"]
    expected_q_prime_idx = 4
    expected_bs1_tilde = "11001110"
    expected_bs2_tilde = bs2  # No change since control is 0-valued.
    expected_P_tilde = {
        bs1: expected_bs1_tilde,
        bs2: expected_bs2_tilde,
        "11111111": "11111000"}
    # Elimination rounds: none, none, 2356 -> add 2 to Q, eliminate all but bs1 tilde and bs2 tilde
    expected_Q_set = {2}

    print(
        "Testing the following prune_controls case:\n"
        f"bs1 = {bs1} (as representative)\n"
        f"bs2 = {bs2}\n"
        f"phys_states = {phys_states}\n"
        f"expected_Q_set = {expected_Q_set}"
    )

    # Check computations at various stages in the algorithm give expected results.
    result_rep_LP_family = LatticeCircuitManager.compute_LP_family(bs1, bs2)
    assert result_rep_LP_family == expected_rep_LP_family, f"LP family = {LatticeCircuitManager.compute_LP_family(bs1, bs2)}"
    for pre_LP_state, expected_post_LP_state in expected_P_tilde.items():
        result_post_LP_state = LatticeCircuitManager._apply_LP_family_to_bit_string(expected_rep_LP_family, expected_q_prime_idx, pre_LP_state)
        assert result_post_LP_state == expected_post_LP_state, f"expected tilde state {expected_post_LP_state} != actual tilde state {result_post_LP_state}"

    # Check the return value from the algorithm.
    result_Q_set = LatticeCircuitManager.prune_controls(bs1, bs2, phys_states)
    assert result_Q_set  == expected_Q_set, f"result Q set = {result_Q_set}"
    print("Test passed.")

    # Case 2, two controls needed.
    bs1 = "11000011"
    bs2 = "11001001"  # Differs at 1 qubit at index 6, will be in Q set.
    phys_states = {bs1, bs2, "11111111", "00000000"}
    expected_rep_LP_family = ["P1", "P1", "P0", "P0", "L+", "P0", "L-", "P1"]
    expected_q_prime_idx = 4
    expected_bs1_tilde = bs1  # No change since control is 0-valude
    expected_bs2_tilde = "11001011"
    expected_P_tilde = {
        bs1: expected_bs1_tilde,
        bs2: expected_bs2_tilde,
        "11111111": "11111101",
        "00000000": "00000000"
    }
    # Elimination round 1: none, 6, 2356, 0167 -> add 6 to Q, eliminate all but bs1 tilde, bs2_tilde
    expected_Q_set = {6}

    print(
        "Testing the following prune_controls case:\n"
        f"bs1 = {bs1} (as representative)\n"
        f"bs2 = {bs2}\n"
        f"phys_states = {phys_states}"
        f"expected_Q_set = {expected_Q_set}"
    )

    # Check computations at various stages in the algorithm give expected results.
    result_rep_LP_family = LatticeCircuitManager.compute_LP_family(bs1, bs2)
    assert result_rep_LP_family == expected_rep_LP_family, f"LP family = {LatticeCircuitManager.compute_LP_family(bs1, bs2)}"
    for pre_LP_state, expected_post_LP_state in expected_P_tilde.items():
        result_post_LP_state = LatticeCircuitManager._apply_LP_family_to_bit_string(expected_rep_LP_family, expected_q_prime_idx, pre_LP_state)
        assert result_post_LP_state == expected_post_LP_state, f"expected tilde state {expected_post_LP_state} != actual tilde state {result_post_LP_state}"

    # Check the return value from the algorithm.
    result_Q_set = LatticeCircuitManager.prune_controls(bs1, bs2, phys_states)
    assert result_Q_set  == expected_Q_set, f"result Q set = {result_Q_set}"
    print("Test passed.")


def _test_apply_LP_family():
    lp_family = ["L-", "L-", "L+", "P0", "L+", "P1"]
    q_prime_idx = 0  # The first 'L' in the LP family list.
    print(f"Checking that application of LP family diagonalizing 'circuit' defined by {lp_family} to bit strings works.")

    # Case 1
    bs_with_active_control = '110001'
    expected_result_bs_with_active_control = "101011"
    print(f"Checking that bit string {bs_with_active_control} becomes {expected_result_bs_with_active_control} when applying LP family.")

    actual_result_active_control = LatticeCircuitManager._apply_LP_family_to_bit_string(lp_family, q_prime_idx, bs_with_active_control)
    assert actual_result_active_control == expected_result_bs_with_active_control, f"Unexpected result: {actual_result_active_control}."
    print("Test passed.")

    # Case 2
    bs_with_inactive_control = '011101'
    expected_result_bs_with_inactive_control = bs_with_inactive_control
    print(f"Checking that bit string {bs_with_inactive_control} becomes {expected_result_bs_with_inactive_control} when applying LP family.")

    actual_result_inactive_control = LatticeCircuitManager._apply_LP_family_to_bit_string(lp_family, q_prime_idx, bs_with_inactive_control)
    assert actual_result_inactive_control == expected_result_bs_with_inactive_control, f"Unexpected result: {actual_result_inactive_control}."
    print("Test passed.")


def _test_eliminate_phys_states_that_differ_from_rep_at_Q_idx():
    rep = "10001"
    phys_states = {rep, "10111", "01001", "00011", "10000", "01011"}
    Q_set = {2, 4}
    expected_phys_states = {rep, "01001", "00011", "01011"}
    print(f"Checking that eliminating bitstrings from a set with qubit index set Q = {Q_set} works.")
    print(f"Representative = {rep}.")
    print(f"Initial phys states = {phys_states}")
    print(f"Expected phys states = {expected_phys_states}")

    actual_phys_states = LatticeCircuitManager._eliminate_phys_states_that_differ_from_rep_at_Q_idx(rep, phys_states, Q_set)
    assert actual_phys_states == expected_phys_states, f"Unexpected result set: {actual_phys_states}."

    print("Test passed.")


def _run_tests():
    #_test_create_blank_full_lattice_circuit_has_promised_register_order()
    #_test_apply_magnetic_trotter_step_d_3_2_large_lattice()
    _test_apply_magnetic_trotter_step_d_3_2_small_lattice()
    _test_apply_magnetic_trotter_step_d_2_large_lattice()
    _test_apply_magnetic_trotter_step_d_2_small_lattice()
    _test_compute_LP_family()
    _test_compute_LP_family_fails_on_bad_input()
    _test_compute_LP_family_fails_for_non_bitstrings()
    _test_prune_controls_fails_for_unequal_length_inputs()
    _test_prune_controls_fails_for_non_bitstrings()
    _test_prune_controls_acts_as_expected()
    _test_apply_LP_family()
    _test_eliminate_phys_states_that_differ_from_rep_at_Q_idx()


if __name__ == "__main__":
    _run_tests()

    print("All tests passed.")
