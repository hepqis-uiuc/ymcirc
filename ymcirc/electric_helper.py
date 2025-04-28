"""
Helper functions to construct electric Hamiltonian through Pauli decomposition.

Electric Hamiltonian is diagonal in computational basis. Helper functions first
construct the Pauli decomposition for matrix with 1 at (i,i) and weights
decomposition by the electric Casimir (according to the link_bitmap).
"""
from __future__ import annotations
from ymcirc.conventions import (IrrepWeight, BitString, IrrepBitmap,
                                LatticeStateEncoder, PlaquetteState,
                                ONE, THREE, THREE_BAR, SIX, SIX_BAR, EIGHT)
from ymcirc._abstract import LatticeDef
import numpy as np
from typing import List
import warnings


def _bitwise_addition(bit_string_1: str | BitString,
                      bit_string_2: str | BitString) -> int:
    """Add two bit bit strings bitwise mod 2."""
    return sum([(int(bit_string_1[i]) * int(bit_string_2[i]))
                for i in range(len(bit_string_1))]) % 2


def _diagonal_pauli_string(bit_string: str, N: int,
                           casimir: float) -> List[int]:
    """
    Generate a Pauli decomposition.

    Given the binary-ordered bit string computational basis of order 2^N,
    takes in bit_string and generates the Pauli decomposition to put the revelant
    Casimir at (int(bit_string), int(bit_string)).

    For a given Pauli string p (w/t I == 0; Z == 1), we have that the coefficient
    for decomposing bit_string is (casimir/2^N)(-1)^(bit_add(p, bit_string)).
    """
    pauliz_list = []
    for i in range(2**N):
        i_string = str('{0:0' + str(N) + 'b}').format(i)
        pauliz_list.append((1 / (2**N)) * casimir *
                           ((-1)**(_bitwise_addition(i_string, bit_string))))
    return pauliz_list


def _gt_pattern_iweight_to_casimir(gt_tuple: IrrepWeight) -> float:
    """
    Generate the Casimir eigenvalue from a GT-pattern i-Weight.

    See arXiv:2101.10227, eq 3.
    """
    p = gt_tuple[0] - gt_tuple[1]
    q = gt_tuple[1]
    return (p**2 + q**2 + p * q + 3 * p + 3 * q) / 3.0


def _handle_electric_energy_unphysical_states(
        global_lattice_bit_string: str,
        link_bit_string: str,
        note_unphysical_states: bool,
        stop_on_unphysical_states: bool) -> float:
    """
    Handle unphysical link states in electric energy calculation.
    """
    if (stop_on_unphysical_states):
        # Terminate simulation
        raise ValueError(
            "Computation of electric energy terminated due to unphysical link state "
            + str(link_bit_string) + " in " + str(global_lattice_bit_string))
    elif (note_unphysical_states):
        # Just print an error, but continue simulation with electric energy for the state 0.0
        warning_msg = f"Unphysical link state {link_bit_string} in {global_lattice_bit_string} found during computation of electric energy."
        warnings.warn(warning_msg)
        return 0.0
    else:
        return 0.0


def casimirs(link_bitmap: IrrepBitmap) -> List[float]:
    """Generate a list of Casimir eigenvalues from link bitmap i-Weights."""
    return [
        _gt_pattern_iweight_to_casimir(irrep)
        for irrep in list(link_bitmap.keys())
    ]


def convert_bitstring_to_evalue(
        global_lattice_bit_string: str,
        lattice_encoder: LatticeStateEncoder,
        warn_on_unphysical_links: bool = True,
        error_on_unphysical_links: bool = False) -> float:
    """
    Convert lattice links to total energy.

    Used for computing the average electric energy. 
    Chunks the bitstring into |v l1 l2 ..> for each vertex when 
    there are multiplicity data on the vertices. Default behavior for unphysical states is to note
    them and assign 0.0 energy. This can by logged with warn_on_unphysical_links,
    or a ValueError can be raised with error_on_unphysical_links.
    """
    link_bitmap = lattice_encoder.link_bitmap
    vertex_bitmap = lattice_encoder.vertex_bitmap

    casimirs = [
        _gt_pattern_iweight_to_casimir(irrep)
        for irrep in list(link_bitmap.keys())
    ]
    casimirs_dict = {}

    # encode using iweights (such as (1, 0 ,0))
    for link_idx, iweight in enumerate(list(link_bitmap.keys())):
        casimirs_dict[iweight] = casimirs[link_idx]

    evalue = 0.0
    length_link_bit_string = lattice_encoder.expected_link_bit_string_length

    has_vertex_bitmap_data = not len(vertex_bitmap) == 0
    if not has_vertex_bitmap_data:
        for link_idx in range(0, len(global_lattice_bit_string), length_link_bit_string):
            current_link_bit_string = global_lattice_bit_string[link_idx:link_idx + length_link_bit_string]
            evalue += compute_link_bitstring_electric_evalue(
                global_lattice_bit_string,
                current_link_bit_string,
                lattice_encoder,
                note_unphysical_states=warn_on_unphysical_links,
                stop_on_unphysical_states=error_on_unphysical_links)
    else:
        encoded_vertex_multiplicity_length = lattice_encoder.expected_vertex_bit_string_length
        spatial_dim = lattice_encoder.lattice_def.dim
        if (spatial_dim == 1.5):
            # For a d = 3/2 lattice, the "top" vertices only have one positive link emanating from them.
            # This inconsistency makes it easier to step through the lattice with a "chunk"
            # consisting of (1) a lower vertex with its two associated positive links and (2) an
            # upper vertex with its one associated positive link.
            vertex_and_two_links_substring_length = encoded_vertex_multiplicity_length + 2 * length_link_bit_string
            vertex_and_one_link_substring_length = encoded_vertex_multiplicity_length + length_link_bit_string
            two_vertical_vertices_with_associated_links_substring_length = (
                vertex_and_one_link_substring_length + vertex_and_two_links_substring_length
            )
            all_two_vertical_vertices_with_associated_links_substrings = [
                global_lattice_bit_string[i:i + two_vertical_vertices_with_associated_links_substring_length]
                for i in range(0, len(global_lattice_bit_string), two_vertical_vertices_with_associated_links_substring_length)
            ]

            # Step over the entire lattice two vertices (+ links) at a time.
            for current_two_vertical_vertices_and_positive_links_substring in all_two_vertical_vertices_with_associated_links_substrings:
                current_lower_vertex_with_two_positive_links_substring = \
                    current_two_vertical_vertices_and_positive_links_substring[:vertex_and_two_links_substring_length]
                current_upper_vertex_with_one_positive_link_substring = \
                    current_two_vertical_vertices_and_positive_links_substring[vertex_and_two_links_substring_length:two_vertical_vertices_with_associated_links_substring_length]

                # Extract the two links substrings connected to the lower vertex, and compute contribution to evalue.
                for link_idx in range(encoded_vertex_multiplicity_length, vertex_and_two_links_substring_length, length_link_bit_string):
                    current_link_bit_string = current_lower_vertex_with_two_positive_links_substring[link_idx:link_idx + length_link_bit_string]
                    evalue += compute_link_bitstring_electric_evalue(
                        global_lattice_bit_string,
                        current_link_bit_string,
                        lattice_encoder,
                        note_unphysical_states=warn_on_unphysical_links,
                        stop_on_unphysical_states=error_on_unphysical_links)

                # Extract the one link substring connected to the upper vertex, and compute the contribution to evalue.
                for link_idx in range(encoded_vertex_multiplicity_length, vertex_and_one_link_substring_length,
                               length_link_bit_string):
                    current_link_bit_string = current_upper_vertex_with_one_positive_link_substring[link_idx:link_idx + length_link_bit_string]
                    evalue += compute_link_bitstring_electric_evalue(
                        global_lattice_bit_string,
                        current_link_bit_string,
                        lattice_encoder,
                        note_unphysical_states=warn_on_unphysical_links,
                        stop_on_unphysical_states=error_on_unphysical_links)
        else:
            vertex_and_positive_links_substring_length = encoded_vertex_multiplicity_length + spatial_dim * length_link_bit_string

            all_vertex_and_positive_link_substrings = [
                global_lattice_bit_string[i:i + vertex_and_positive_links_substring_length]
                for i in range(0, len(global_lattice_bit_string), vertex_and_positive_links_substring_length)
            ]

            for current_two_vertical_vertices_and_positive_links_substring in all_vertex_and_positive_link_substrings:
                for link_idx in range(encoded_vertex_multiplicity_length, len(current_two_vertical_vertices_and_positive_links_substring), length_link_bit_string):
                    current_link_bit_string = current_two_vertical_vertices_and_positive_links_substring[link_idx:link_idx + length_link_bit_string]
                    evalue += compute_link_bitstring_electric_evalue(
                        global_lattice_bit_string,
                        current_link_bit_string,
                        lattice_encoder,
                        note_unphysical_states=warn_on_unphysical_links,
                        stop_on_unphysical_states=error_on_unphysical_links)

    return evalue


def electric_hamiltonian(link_bitmap: IrrepBitmap) -> List[float]:
    """Give the total Pauli decomposition."""
    N = len(list(link_bitmap.values())[0])
    casimir_values = [
        _gt_pattern_iweight_to_casimir(irrep)
        for irrep in list(link_bitmap.keys())
    ]
    pauli_strings = [
        _diagonal_pauli_string(bit_string=list(link_bitmap.values())[i],
                               N=N,
                               casimir=casimir_values[i])
        for i in range(len(list(link_bitmap.values())))
    ]

    return [sum(x) for x in zip(*pauli_strings)]


def compute_link_bitstring_electric_evalue(
        global_lattice_bit_string: str,
        link_bit_string: str,
        lattice_encoder: LatticeStateEncoder,
        note_unphysical_states: bool = True,
        stop_on_unphysical_states: bool = False) -> float:
    """
    Compute the electric energy evalue on a link bitstring.

    Options are provided to control whether to issue a warning if
    the link encodes an unphysical state, or to raise an error.
    """
    link_state = lattice_encoder.decode_bit_string_to_link_state(link_bit_string)
    if link_state is None:
        return _handle_electric_energy_unphysical_states(
            global_lattice_bit_string, link_bit_string,
            note_unphysical_states, stop_on_unphysical_states)

    return _gt_pattern_iweight_to_casimir(link_state)


def _test_decomp_1_3_3bar():
    print(
        "Testing if 1,3,3bar electric Hamiltonian is correctly formed from Pauli decomposition..."
    )
    II = np.kron([[1, 0], [0, 1]], [[1, 0], [0, 1]])
    IZ = np.kron([[1, 0], [0, 1]], [[1, 0], [0, -1]])
    ZI = np.kron([[1, 0], [0, -1]], [[1, 0], [0, 1]])
    ZZ = np.kron([[1, 0], [0, -1]], [[1, 0], [0, -1]])

    bmap = {(0, 0, 0): "00", (1, 0, 0): "10", (1, 1, 0): "01"}

    N = len(list(bmap.values())[0])

    # Generate decomp matrix
    decomp = electric_hamiltonian(bmap)
    decomp_matrx = (decomp[0] * II + decomp[1] * IZ + decomp[2] * ZI +
                    decomp[3] * ZZ)

    # Generate keys for comparison matrix
    binary_keys = list(map(lambda x: int(x, 2), list(bmap.values())))
    cmpr_matrx = np.zeros((2**N, 2**N))
    casmrs = casimirs(bmap)
    for i in range(len(binary_keys)):
        cmpr_matrx[binary_keys[i], binary_keys[i]] = casmrs[i]

    # Compare matrices
    assert (np.isclose(cmpr_matrx, decomp_matrx)).all()
    print("Test passed.")


def _test_decomp_1_3_3bar_6_6bar_8():
    print(
        "Testing if 1, 3, 3bar, 6, 6bar, 8 electric Hamiltonian is correctly formed from Pauli decomposition..."
    )
    III = np.kron(np.kron([[1, 0], [0, 1]], [[1, 0], [0, 1]]),
                  [[1, 0], [0, 1]])
    IIZ = np.kron(np.kron([[1, 0], [0, 1]], [[1, 0], [0, 1]]),
                  [[1, 0], [0, -1]])
    IZI = np.kron(np.kron([[1, 0], [0, 1]], [[1, 0], [0, -1]]),
                  [[1, 0], [0, 1]])
    IZZ = np.kron(np.kron([[1, 0], [0, 1]], [[1, 0], [0, -1]]),
                  [[1, 0], [0, -1]])
    ZII = np.kron(np.kron([[1, 0], [0, -1]], [[1, 0], [0, 1]]),
                  [[1, 0], [0, 1]])
    ZIZ = np.kron(np.kron([[1, 0], [0, -1]], [[1, 0], [0, 1]]),
                  [[1, 0], [0, -1]])
    ZZI = np.kron(np.kron([[1, 0], [0, -1]], [[1, 0], [0, -1]]),
                  [[1, 0], [0, 1]])
    ZZZ = np.kron(np.kron([[1, 0], [0, -1]], [[1, 0], [0, -1]]),
                  [[1, 0], [0, -1]])

    bmap = {
        (0, 0, 0): "000",
        (1, 0, 0): "100",
        (1, 1, 0): "001",
        (2, 0, 0): "110",
        (2, 2, 0): "011",
        (2, 1, 0): "111"
    }

    N = len(list(bmap.values())[0])

    # Generate decomp matrix
    decomp = electric_hamiltonian(bmap)
    decomp_matrx = decomp[0] * III + decomp[1] * IIZ + decomp[
        2] * IZI + decomp[3] * IZZ + decomp[4] * ZII + decomp[
            5] * ZIZ + decomp[6] * ZZI + decomp[7] * ZZZ

    # Generate keys for comparison matrix
    binary_keys = list(map(lambda x: int(x, 2), list(bmap.values())))
    cmpr_matrx = np.zeros((2**N, 2**N))
    casmrs = casimirs(bmap)

    for i in range(len(binary_keys)):
        cmpr_matrx[binary_keys[i], binary_keys[i]] = casmrs[i]

    # Compare matrices
    assert (np.isclose(cmpr_matrx, decomp_matrx)).all()
    print("Test passed.")


def _test_convert_bitstring_to_evalue_d_2_ground_state():
    print(
        "Testing if the electric energy |E|^2 is being calculated correctly for d = 2 with "
        "a single multiplicity qubit in the lattice ground state."
    )
    T1_link_bitmap = {ONE: "00", THREE: "10", THREE_BAR: "01"}
    physical_states: PlaquetteState = [  # The only thing this controls is the number of vertex qubits the LatticeStateEncoder infers. Garbage data otherwise.
        (
            (0, 0, 0, 1),  # Since 1 is the highest multiplicity, one vertex qubit.
            (ONE, ONE, ONE, ONE),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        )
    ]
    lattice = LatticeDef(dimensions=2, size=2)
    lattice_encoder = LatticeStateEncoder(T1_link_bitmap, physical_states, lattice)

    # Ground state of the lattice.
    global_lattice_ground_state_bit_string = "0" * 20
    computed_energy = convert_bitstring_to_evalue(
        global_lattice_ground_state_bit_string, lattice_encoder)
    expected_energy = 0

    assert np.isclose(expected_energy, computed_energy), f"Expected energy: {expected_energy}, computed energy: {computed_energy}"
    print("Test passed.")


def _test_convert_bitstring_to_evalue_d_2_excited_state():
    print(
        "Testing if the electric energy |E|^2 is being calculated correctly for d = 2, "
        "with 2 multiplicity qubits in a lattice state with six excited links."
    )
    T1_link_bitmap = {ONE: "00", THREE: "10", THREE_BAR: "01"}
    physical_states: PlaquetteState = [  # The only thing this controls is the number of vertex qubits the LatticeStateEncoder infers. Garbage data otherwise.
        (
            (0, 0, 0, 3),  # Since 3 is the highest multiplicity, two vertex qubits.
            (ONE, ONE, ONE, ONE),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        )
    ]
    lattice = LatticeDef(dimensions=2, size=3)
    lattice_encoder = LatticeStateEncoder(T1_link_bitmap, physical_states, lattice)

    # 9 * 2 + 18 * 2 = 54 qubits.
    # There are 6 links with electric energy of (4/3)
    # so the total energy should be 6*4/3 = 8.
    ground_state_vertex_bitstring = "00" + "00" + "00"
    vertex_three_three_bar = "10" + "10" + "01"
    vertex_three_one = "00" + "10" + "00"
    vertex_one_three_bar = "01" + "00" + "10"
    global_lattice_state_bit_string = (
        ground_state_vertex_bitstring + ground_state_vertex_bitstring + vertex_three_three_bar +
        vertex_three_one + ground_state_vertex_bitstring + vertex_one_three_bar +
        ground_state_vertex_bitstring + vertex_three_three_bar + ground_state_vertex_bitstring
    )
    computed_energy = convert_bitstring_to_evalue(global_lattice_state_bit_string, lattice_encoder)
    expected_energy = 8

    assert np.isclose(expected_energy, computed_energy), f"Expected energy: {expected_energy}, computed energy: {computed_energy}"
    print("Test passed.")


def _test_convert_bitstring_to_evalue_d_3_2_excited_state():
    print(
        "Testing if the electric energy |E|^2 is being calculated correctly for d = 3/2, "
        "with no multiplicity qubits in a lattice state with one excited link."
    )
    T1_link_bitmap = {(0, 0, 0): "00", (1, 0, 0): "10", (1, 1, 0): "01"}
    physical_states: PlaquetteState = [  # The only thing this controls is the number of vertex qubits the LatticeStateEncoder infers. Garbage data otherwise.
        (
            (0, 0, 0, 0),  # All zeros flag that no vertex qubits needed, only trivial multiplicities.
            (ONE, ONE, ONE, ONE),
            (ONE, ONE, ONE, ONE)
        )
    ]
    lattice = LatticeDef(dimensions=1.5, size=2)
    lattice_encoder = LatticeStateEncoder(T1_link_bitmap, physical_states, lattice)

    # 12 qubits in d=3/2, 2x1, T1p. There is one link with energy (4/3)
    global_lattice_state_bit_string = "001000000000"
    computed_energy = convert_bitstring_to_evalue(global_lattice_state_bit_string, lattice_encoder)
    expected_energy = 4/3.0

    assert np.isclose(expected_energy, computed_energy), f"Expected energy: {expected_energy}, computed energy: {computed_energy}"
    print("Test passed.")


def _test_convert_bitstring_to_evalue_ignore_unphysical_links():
    print(
        "Testing if the electric energy |E|^2 is being calculated correctly for d = 3/2, T1 truncation "
        "when there's a link in an unphysical state that's set to be ignored, one vertex qubit, and two excited links.")
    T1_link_bitmap = {(0, 0, 0): "00", (1, 0, 0): "10", (1, 1, 0): "01"}
    physical_states: PlaquetteState = [  # The only thing this controls is the number of vertex qubits the LatticeStateEncoder infers. Garbage data otherwise.
        (
            (0, 0, 0, 1),  # One vertex qubit needed to track these multiplicities.
            (ONE, ONE, ONE, ONE),
            (ONE, ONE, ONE, ONE)
        )
    ]
    lattice = LatticeDef(dimensions=1.5, size=2)
    lattice_encoder = LatticeStateEncoder(T1_link_bitmap, physical_states, lattice)

    # 4 + 12 = 16 qubits in d=3/2, 2x1. There is an unphysical link at ((0, 0), 2),
    # and two excited links.
    vertex_one_unphysical = "0" + "00" + "11"
    vertex_three = "1" + "10"
    vertex_one_one = "0" + "00" + "00"
    vertex_three_bar = "0" + "01"
    global_lattice_state_bit_string = vertex_one_unphysical + vertex_three + vertex_one_one + vertex_three_bar
    computed_energy = convert_bitstring_to_evalue(global_lattice_state_bit_string, lattice_encoder)
    expected_energy = 8/3.0

    assert np.isclose(expected_energy, computed_energy), f"Expected energy: {expected_energy}, computed energy: {computed_energy}"
    print("Test passed.")


def _test_convert_bitstring_to_evalue_raise_error_on_unphysical_links():
    print(
        "Testing the option to raise an error when encountering unphysical links "
        "for the electric energy |E|^2 calculation.")
    T1_link_bitmap = {(0, 0, 0): "00", (1, 0, 0): "10", (1, 1, 0): "01"}
    physical_states: PlaquetteState = [  # The only thing this controls is the number of vertex qubits the LatticeStateEncoder infers. Garbage data otherwise.
        (
            (0, 0, 0, 1),  # One vertex qubit needed to track these multiplicities.
            (ONE, ONE, ONE, ONE),
            (ONE, ONE, ONE, ONE)
        )
    ]
    lattice = LatticeDef(dimensions=1.5, size=2)
    lattice_encoder = LatticeStateEncoder(T1_link_bitmap, physical_states, lattice)

    # 4 + 12 = 16 qubits in d=3/2, 2x1. There is an unphysical link at ((0, 0), 2),
    # and two excited links.
    vertex_one_unphysical = "0" + "00" + "11"
    vertex_three = "1" + "10"
    vertex_one_one = "0" + "00" + "00"
    vertex_three_bar = "0" + "01"
    global_lattice_state_bit_string = vertex_one_unphysical + vertex_three + vertex_one_one + vertex_three_bar

    try:
        convert_bitstring_to_evalue(global_lattice_state_bit_string, lattice_encoder, error_on_unphysical_links=True)
    except ValueError as e:
        print(f"Test passed. Raised error: {e}")
    else:
        raise AssertionError("Failed to raise ValueError.")


def _test_convert_bitstring_to_evalue_d_2_T2_truncation():
    print(
        "Check the electric energy calculation for the T2 truncation. "
        "Lattice state has exactly one three, three bar, six, six bar, and four eights, "
        "with all other links in the one irrep. Total energy should therefore be "
        "2*(4/3) + 2*(10/3) + 4*(9/3) = 64/3"
    )
    T1_link_bitmap = {ONE: "000", THREE: "100", THREE_BAR: "001", SIX: "110", SIX_BAR: "011", EIGHT: "111"}
    physical_states: PlaquetteState = [  # The only thing this controls is the number of vertex qubits the LatticeStateEncoder infers. Garbage data otherwise.
        (
            (0, 0, 0, 3),  # Since 3 is the highest multiplicity, two vertex qubits.
            (ONE, ONE, ONE, ONE),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        )
    ]
    lattice = LatticeDef(dimensions=2, size=3)
    lattice_encoder = LatticeStateEncoder(T1_link_bitmap, physical_states, lattice)

    ground_state_vertex_bitstring = "00" + "000" + "000"
    vertex_three_three_bar = "00" + "100" + "001"
    vertex_one_six = "00" + "000" + "110"
    vertex_eight_one = "00" + "111" + "000"
    vertex_six_bar_one = "00" + "011" + "000"
    global_lattice_state_bit_string = (
        ground_state_vertex_bitstring + vertex_eight_one + vertex_three_three_bar +
        vertex_one_six + ground_state_vertex_bitstring + vertex_eight_one +
        vertex_eight_one + vertex_six_bar_one + vertex_eight_one
    )
    computed_energy = convert_bitstring_to_evalue(global_lattice_state_bit_string, lattice_encoder)
    expected_energy = 64/3.0

    assert np.isclose(expected_energy, computed_energy), f"Expected energy: {expected_energy}, computed energy: {computed_energy}"
    print("Test passed.")


if __name__ == "__main__":
    print("Running tests...")
    _test_decomp_1_3_3bar()
    _test_decomp_1_3_3bar_6_6bar_8()
    _test_convert_bitstring_to_evalue_d_2_ground_state()
    _test_convert_bitstring_to_evalue_d_2_excited_state()
    _test_convert_bitstring_to_evalue_d_3_2_excited_state()
    _test_convert_bitstring_to_evalue_ignore_unphysical_links()
    _test_convert_bitstring_to_evalue_raise_error_on_unphysical_links()
    _test_convert_bitstring_to_evalue_d_2_T2_truncation()
    print("All tests passed.")
