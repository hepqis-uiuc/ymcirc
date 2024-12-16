"""

Helper functions to construct electric hamiltonian through Pauli decomposition. Electric hamiltonian is diagonal in computational basis. 
Helper functions first construct the Pauli decomposition for matrix with 1 at (i,i) and weights decomposition by the electric casimir (according to the link_bitmap)

"""

import numpy as np
import math
from typing import Tuple, List, Dict


# Useful type aliases.
IrrepWeight = Tuple[int, int, int]
LinkState = IrrepWeight  # A useful semantic alias
BitString = str
MultiplicityIndex = int
IrrepBitmap = Dict[IrrepWeight, BitString]
SingletsDef = Tuple[Tuple[IrrepWeight, ...], Tuple[MultiplicityIndex, ...]]
VertexBag = Tuple[Tuple[IrrepWeight, ...], MultiplicityIndex]
VertexState = VertexBag  # Another useful semantic alias
VertexMultiplicityBitmap = Dict[VertexBag, BitString]


def _bit_add(lst1 : str, lst2 : str) -> int:

    # Adds two bitstrings bitwise mod 2

    return sum([(int(lst1[i])*int(lst2[i])) for i in range(len(lst1))]) % 2
    
def _diagonal_pauli_string(bitstring : str, N : int, casimir : int) -> List[int]:

    # Given the binary-ordered bistring computational basis of order 2^N, function takes in a bistring b and generates the Pauli decomp to put the revelant casimir at (int(b), int(b))
    # For a given Pauli string p (w/t I == 0; Z == 1), we have that the coefficient for decomping b is (casimir/2^N)(-1)^(bit_add(p, b))

    pauliz_list = []
    for i in range(2**N):
        i_string = str('{0:0' + str(N) + 'b}').format(i)
        pauliz_list.append((1/(2**N))*casimir*((-1)**(_bit_add(i_string, bitstring))))
    return pauliz_list

def _gf_to_casimir(gf_tuple : Tuple[int]) -> int:

    # Function to generate casimirs from the GT-pattern (see arXiv:2101.10227, eq 3)

    p = gf_tuple[0] - gf_tuple[1]; q = gf_tuple[1]
    return (p**2 + q**2 + p*q + 3*p + 3*q)/3.0


def casimirs(link_bitmap : IrrepBitmap) -> List[int]:

    """
    Function to generate casimirs from link-bitmap GTs
    """

    return [_gf_to_casimir(irrep) for irrep in list(link_bitmap.keys())]


def _convert_bitstring_to_evalue(bitstring : str, link_bitmap : IrrepBitmap, vertex_bitmap : VertexMultiplicityBitmap) -> int:

    # Function to convert lattice links to total energy. Used for computing average electric energy. WIP for vertex_bitmap \neq {}

    casimirs = [_gf_to_casimir(irrep) for irrep in list(link_bitmap.keys())]

    casimirs_dic = {}

    for i, enc in enumerate(list(link_bitmap.values())):
        casimirs_dic[enc] = casimirs[i]

    value = 0; len_string = len(list(link_bitmap.values())[0])

    if (not vertex_bitmap):
        for i in range(0,len(bitstring),len_string):
            value += casimirs_dic[bitstring[i:i+len_string]]  
    else:
        vertex_singlet_length = len(list(vertex_bitmap.values())[0])
        dim = int(len(list(vertex_bitmap.keys())[0])/2.0)
        total_length = vertex_singlet_length  + dim*len_string

        chunks = [bitstring[i:i+total_length] for i in range(0,len(bitstring,total_length))]

        for chunk in chunks:
            for i in range(vertex_singlet_length,len(chunk),len_string):
                value += casimirs_dic[bitstring[i:i+len_string]] 

    return value


def electric_hamiltonian(link_bitmap : IrrepBitmap) -> List[int]:

    # Gives the total Pauli decompsition

    N = len(list(link_bitmap.values())[0])

    casimir_values = [_gf_to_casimir(irrep) for irrep in list(link_bitmap.keys())]
    
    pauli_strings = [_diagonal_pauli_string(list(link_bitmap.values())[i], N, casimir_values[i]) for i in range(len(list(link_bitmap.values())))]

    return [sum(x) for x in zip(*pauli_strings)] 


# Tests 

def _test_decomp_1_3_3bar():

    # Tests if 1,3,3bar electric hamiltonian is correctly formed from Pauli decompisition

    II = np.kron([[1,0],[0,1]], [[1,0],[0,1]])
    IZ = np.kron([[1,0],[0,1]], [[1,0],[0,-1]])
    ZI = np.kron([[1,0],[0,-1]], [[1,0],[0,1]])
    ZZ = np.kron([[1,0],[0,-1]], [[1,0],[0,-1]])

    bmap = {
    (0, 0, 0): "00",
    (1, 0, 0): "10",
    (1, 1, 0): "01"
    }

    N = len(list(bmap.values())[0])

    # Generate decomp matrix

    decomp = electric_hamiltonian(bmap); decomp_matrx = (decomp[0]*II + decomp[1]*IZ + decomp[2]*ZI + decomp[3]*ZZ)

    # Generate keys for comparison matrix 

    binary_keys = list(map(lambda x: int(x, 2), list(bmap.values())))

    cmpr_matrx = np.zeros((2**N,2**N))

    casmrs = casimirs(bmap)

    for i in range(len(binary_keys)):
        cmpr_matrx[binary_keys[i], binary_keys[i]] = casmrs[i]

    # Compare matrices

    if (np.isclose(cmpr_matrx, decomp_matrx)).all():
        print("Success for test_decomp_1_3_3bar")
    else:
        print("Failure. Here are the hard-coded matrix and decomp matrix, respecively")
        print(cmpr_matrx); print(decomp_matrx)


def _test_decomp_1_3_3bar_6_6bar_8():

    # Tests if 1,3,3bar,6,6bar,8 electric hamiltonian is correctly formed from Pauli decompisition

    III = np.kron(np.kron([[1,0],[0,1]], [[1,0],[0,1]]), [[1,0],[0,1]])
    IIZ = np.kron(np.kron([[1,0],[0,1]], [[1,0],[0,1]]), [[1,0],[0,-1]])
    IZI = np.kron(np.kron([[1,0],[0,1]], [[1,0],[0,-1]]), [[1,0],[0,1]])
    IZZ = np.kron(np.kron([[1,0],[0,1]], [[1,0],[0,-1]]), [[1,0],[0,-1]])
    ZII = np.kron(np.kron([[1,0],[0,-1]], [[1,0],[0,1]]), [[1,0],[0,1]])
    ZIZ = np.kron(np.kron([[1,0],[0,-1]], [[1,0],[0,1]]), [[1,0],[0,-1]])
    ZZI = np.kron(np.kron([[1,0],[0,-1]], [[1,0],[0,-1]]), [[1,0],[0,1]])
    ZZZ = np.kron(np.kron([[1,0],[0,-1]], [[1,0],[0,-1]]), [[1,0],[0,-1]])

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

    decomp_matrx = decomp[0]*III + decomp[1]*IIZ + decomp[2]*IZI + decomp[3]*IZZ + decomp[4]*ZII + decomp[5]*ZIZ + decomp[6]*ZZI + decomp[7]*ZZZ

    # Generate keys for comparison matrix 

    binary_keys = list(map(lambda x: int(x, 2), list(bmap.values())))

    cmpr_matrx = np.zeros((2**N,2**N))

    casmrs = casimirs(bmap)

    for i in range(len(binary_keys)):
        cmpr_matrx[binary_keys[i], binary_keys[i]] = casmrs[i]

    # Compare matrices

    if (np.isclose(cmpr_matrx, decomp_matrx)).all():
        print("Success for test_decomp_1_3_3bar_6_6bar_8")
    else:
        print("Failure. Here are the hard-coded matrix and decomp matrix, respecively")
        print(cmpr_matrx); print(decomp_matrx)


if __name__ == "__main__":
    _test_decomp_1_3_3bar()
    _test_decomp_1_3_3bar_6_6bar_8()


