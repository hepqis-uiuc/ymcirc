"""
Helper functions to construct electric Hamiltonian through Pauli decomposition.

Electric Hamiltonian is diagonal in computational basis. Helper functions first
construct the Pauli decomposition for matrix with 1 at (i,i) and weights
decomposition by the electric Casimir (according to the link_bitmap).
"""
from __future__ import annotations
from lattice_tools.conventions import (
    IrrepWeight, BitString, IrrepBitmap, VertexMultiplicityBitmap
)
import numpy as np
from typing import List


def _bitwise_addition(
        bit_string_1: str | BitString, bit_string_2: str | BitString) -> int:
    """Add two bit bit strings bitwise mod 2."""
    return sum([(int(bit_string_1[i]) * int(bit_string_2[i])) for i in range(len(bit_string_1))]) % 2


def _diagonal_pauli_string(bit_string: str, N: int, casimir: float) -> List[int]:
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
        pauliz_list.append(
            (1 / (2**N)) * casimir * ((-1)**(_bitwise_addition(i_string, bit_string))))
    return pauliz_list


def _gt_pattern_iweight_to_casimir(gt_tuple: IrrepWeight) -> float:
    """
    Generate the Casimir eigenvalue from a GT-pattern i-Weight.

    See arXiv:2101.10227, eq 3.
    """
    p = gt_tuple[0] - gt_tuple[1]
    q = gt_tuple[1]
    return (p**2 + q**2 + p * q + 3 * p + 3 * q) / 3.0


def casimirs(link_bitmap: IrrepBitmap) -> List[float]:
    """Generate a list of Casimir eigenvalues from link bitmap i-Weights."""
    return [_gt_pattern_iweight_to_casimir(irrep) for irrep in list(link_bitmap.keys())]


# TODO, implement this calculation for vertex_bitmap \neq {}.
def convert_bitstring_to_evalue(
        bitstring: str, link_bitmap: IrrepBitmap,
        vertex_bitmap: VertexMultiplicityBitmap) -> float:
    """
    Convert lattice links to total energy.

    Used for computing the average electric energy. Currently raises a
    NotImplemented error if vertex_bitmap is nonempty.
    """
    casimirs = [_gt_pattern_iweight_to_casimir(irrep) for irrep in list(link_bitmap.keys())]
    casimirs_dict = {}

    for i, enc in enumerate(list(link_bitmap.values())):
        casimirs_dict[enc] = casimirs[i]

    evalue = 0.0
    len_string = len(list(link_bitmap.values())[0])

    has_vertex_bitmap_data = not len(vertex_bitmap) == 0
    if not has_vertex_bitmap_data:
        for i in range(0, len(bitstring), len_string):
            evalue += casimirs_dict[bitstring[i:i + len_string]]
    else:
        raise NotImplementedError("Eigenvalue computation when vertex data present is a WIP.")
        # vertex_singlet_length = len(list(vertex_bitmap.values())[0])
        # dim = int(len(list(vertex_bitmap.keys())[0]) / 2.0)
        # total_length = vertex_singlet_length + dim * len_string

        # chunks = [
        #     bitstring[i:i + total_length]
        #     for i in range(0, len(bitstring, total_length))
        # ]

        # for chunk in chunks:
        #     for i in range(vertex_singlet_length, len(chunk), len_string):
        #         evalue += casimirs_dict[bitstring[i:i + len_string]]

    return evalue


def electric_hamiltonian(link_bitmap: IrrepBitmap) -> List[float]:
    """Give the total Pauli decomposition."""
    N = len(list(link_bitmap.values())[0])
    casimir_values = [
        _gt_pattern_iweight_to_casimir(irrep) for irrep in list(link_bitmap.keys())
    ]
    pauli_strings = [
        _diagonal_pauli_string(
            bit_string=list(link_bitmap.values())[i],
            N=N,
            casimir=casimir_values[i]
        )
        for i in range(len(list(link_bitmap.values())))
    ]

    return [sum(x) for x in zip(*pauli_strings)]


# Tests
def _test_decomp_1_3_3bar():
    print("Testing if 1,3,3bar electric Hamiltonian is correctly formed from Pauli decomposition...")
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
    if (np.isclose(cmpr_matrx, decomp_matrx)).all():
        print("Test passed.")
    else:
        print(
            "Failure. Here are the hard-coded matrix and decomp matrix, respectively:"
        )
        print(cmpr_matrx)
        print(decomp_matrx)


def _test_decomp_1_3_3bar_6_6bar_8():
    print("Testing if 1,3,3bar,6,6bar,8 electric Hamiltonian is correctly formed from Pauli decompisition...")
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
    if (np.isclose(cmpr_matrx, decomp_matrx)).all():
        print("Success for test_decomp_1_3_3bar_6_6bar_8")
    else:
        print(
            "Failure. Here are the hard-coded matrix and decomp matrix, respectively:"
        )
        print(cmpr_matrx)
        print(decomp_matrx)


if __name__ == "__main__":
    print("Running test...")
    _test_decomp_1_3_3bar()
    _test_decomp_1_3_3bar_6_6bar_8()
    print("All tests passed.")
