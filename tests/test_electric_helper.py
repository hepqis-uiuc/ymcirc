import numpy as np
import pytest
from ymcirc.conventions import LatticeStateEncoder
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
    


def test_convert_bitstring_to_evalue_case1():
    """Testing if the electric energy |E|^2 is being calculated correctly for d2, T1p vacuum"""

    # Use the T1 truncation with T1p vertex maps for ease of creating tests.
    # Tests strings may or not physical
    T1_map = {(0, 0, 0): "00", (1, 0, 0): "10", (1, 1, 0): "01"}

    vertex_d2_T1p_map = {
        (((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), 1): '0',
        (((0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 1, 0)), 1): '1',
    }

    # 20 qubits in d=2, 2x2, T1p. Expect zero energy
    example_d2_T1p_string1 = "0" * 20

    lattice_encoder = LatticeStateEncoder(T1_map, vertex_d2_T1p_map)

    expected_energy = 0
    actual_energy = convert_bitstring_to_evalue(example_d2_T1p_string1,
                                                lattice_encoder)

    np.testing.assert_allclose(expected_energy, actual_energy)


def test_convert_bitstring_to_evalue_case2():
    """Testing if the electric energy |E|^2 is being calculated correctly for d2, T1p excited state"""

    # Use the T1 truncation with T1p vertex maps for ease of creating tests.
    # Tests strings may or not physical
    T1_map = {(0, 0, 0): "00", (1, 0, 0): "10", (1, 1, 0): "01"}

    vertex_d2_T1p_map = {
        (((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), 1): '0',
        (((0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 1, 0)), 1): '1',
    }

    # 20 qubits in d=2, 2x2, T1p. There are 6 links with energy (4/3)
    example_d2_T1p_string2 = "10001110100101001000"

    lattice_encoder = LatticeStateEncoder(T1_map, vertex_d2_T1p_map)

    expected_energy = ((4.0 / 3.0) * 6.0)
    actual_energy = convert_bitstring_to_evalue(example_d2_T1p_string2, lattice_encoder)

    np.testing.assert_allclose(expected_energy, actual_energy)


def test_convert_bitstring_to_evalue_case3():
    """Testing if the electric energy |E|^2 is being calculated correctly for d3/2, T1p excited state"""

    # Use the T1 truncation with T1p vertex maps for ease of creating tests.
    # Tests strings may or not physical
    T1_map = {(0, 0, 0): "00", (1, 0, 0): "10", (1, 1, 0): "01"}

    vertex_d32_T1p_map = {
        (((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1): '0',
        (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1): '1',
    }

    # 16 qubits in d=3/2, 2x1, T1p. There are 4 links with energy (4/3)
    example_d32_T1p_string1 = "1000111011010100"

    lattice_encoder = LatticeStateEncoder(T1_map, vertex_d32_T1p_map)

    expected_energy = ((4.0 / 3.0) * 4.0)
    actual_energy = convert_bitstring_to_evalue(example_d32_T1p_string1, lattice_encoder)

    np.testing.assert_allclose(expected_energy, actual_energy)


def test_convert_bitstring_to_evalue_case4():
    """
    Testing if the electric energy |E|^2 is being calculated correctly for d3/2,
    T1p unphysical state with just noting an error.
    """

    # Use the T1 truncation with T1p vertex maps for ease of creating tests.
    # Tests strings may or not physical
    T1_map = {(0, 0, 0): "00", (1, 0, 0): "10", (1, 1, 0): "01"}

    vertex_d32_T1p_map = {
        (((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1): '0',
        (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1): '1',
    }

    # 16 qubits in d=3/2, 2x1, T1p. There is an unphysical link at ((0,0), e_y). Expecting 0.0
    example_d32_T1p_string2 = "1001111011010100"

    lattice_encoder = LatticeStateEncoder(T1_map, vertex_d32_T1p_map)

    expected_energy = 0
    actual_energy = convert_bitstring_to_evalue(example_d32_T1p_string2, lattice_encoder)

    np.testing.assert_allclose(expected_energy, actual_energy)


def test_convert_bitstring_to_evalue_case5():
    """
    Testing if the electric energy |E|^2 is being calculated correctly for d3/2,
    T1p unphysical state with raising an error.
    """

    # Use the T1 truncation with T1p vertex maps for ease of creating tests.
    # Tests strings may or not physical
    T1_map = {(0, 0, 0): "00", (1, 0, 0): "10", (1, 1, 0): "01"}

    vertex_d32_T1p_map = {
        (((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1): '0',
        (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1): '1',
    }

    # 16 qubits in d=3/2, 2x1, T1p. There is an unphysical link at ((0,0), e_y). Expecting 0.0
    example_d32_T1p_string2 = "1001111011010100"

    lattice_encoder = LatticeStateEncoder(T1_map, vertex_d32_T1p_map)

    with pytest.raises(ValueError) as e_info:
        convert_bitstring_to_evalue(example_d32_T1p_string2,
                                    lattice_encoder,
                                    stop_on_unphysical_states=True)
