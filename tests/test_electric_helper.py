import numpy as np
import pytest
from ymcirc._abstract import LatticeDef
from ymcirc.conventions import LatticeStateEncoder, ONE, THREE, THREE_BAR, SIX, SIX_BAR, EIGHT, PlaquetteState
from ymcirc.electric_helper import electric_hamiltonian, casimirs, convert_bitstring_to_evalue


def test_decomp_1_3_3bar():
    """Testing if 1,3,3bar electric Hamiltonian is correctly formed from Pauli decomposition..."""

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
    np.testing.assert_allclose(cmpr_matrx, decomp_matrx)


def test_decomp_1_3_3bar_6_6bar_8():
    """Testing if 1,3,3bar,6,6bar,8 electric Hamiltonian is correctly formed from Pauli decomposition..."""

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
    np.testing.assert_allclose(cmpr_matrx, decomp_matrx, atol=1e-08)  # Setting atol to match defualt np.isclose behavior.


def test_convert_bitstring_to_evalue_d_2_ground_state():
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

    np.testing.assert_allclose(expected_energy, computed_energy)


def test_convert_bitstring_to_evalue_d_2_excited_state():
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

    np.testing.assert_allclose(expected_energy, computed_energy)


def test_convert_bitstring_to_evalue_d_3_2_excited_state():
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

    np.testing.assert_allclose(expected_energy, computed_energy)


def test_convert_bitstring_to_evalue_ignore_unphysical_links():
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

    np.testing.assert_allclose(expected_energy, computed_energy)


def test_convert_bitstring_to_evalue_raise_error_on_unphysical_links():
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

    with pytest.raises(ValueError) as e_info:
        convert_bitstring_to_evalue(global_lattice_state_bit_string, lattice_encoder, error_on_unphysical_links=True)


def test_convert_bitstring_to_evalue_d_2_T2_truncation():
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

    np.testing.assert_allclose(expected_energy, computed_energy)
