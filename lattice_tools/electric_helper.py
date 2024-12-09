import numpy as np
import math
from qiskit.circuit import QuantumCircuit
import qiskit.quantum_info as qi
#from qiskit import BasicAer
#backend = BasicAer.get_backend('statevector_simulator')
#from qiskit.execute_function import execute

"""

Helper function to construct electric hamiltonian through Pauli decomposition. Electric hamiltonian is diagonal in computational basis. 
Helper function first constructs Pauli decomposition for matrix with 1 at (i,i) and weights decomposition by the electric casimir (according to the link_bitmap)

"""

def bitadd(lst1, lst2):
    return sum([(int(lst1[i])*int(lst2[i])) for i in range(len(lst1))]) % 2
    
def diagonalpaulistring(bitstring, N, casimir):

    # Given the binary-ordered bistring computational basis of order 2^N, function takes in a bistring b and generates the Pauli decomp to put the revelant casimir at (int(b), int(b))
    # For a given Pauli string p (w/t I == 0; Z == 1), we have that the coefficient for decomping b is (casimir/2^N)(-1)^(bitadd(p, b))

    paulizlist = []
    for i in range(2**N):
        istring = str('{0:0' + str(N) + 'b}').format(i)
        paulizlist.append((1/(2**N))*casimir*((-1)**(bitadd(istring, bitstring))))
    return paulizlist

def gf_to_casimir(gf_tuple):

    # Function to generate casimirs from the GF-pattern (see trailhead)

    p = gf_tuple[0] - gf_tuple[1]; q = gf_tuple[1]
    return (p**2 + q**2 + p*q + 3*p + 3*q)/3.0



def casimirs(link_bitmap):

    # Function to generate casimirs from link-bitmap GFs

    N = len(list(link_bitmap.values())[0])

    return [gf_to_casimir(irrep) for irrep in list(link_bitmap.keys())]


def convert_bitstring_to_evalue(bitstring, link_bitmap):

    # Function to convert lattice links to total energy. Used for computing average electric energy

    N = len(list(link_bitmap.values())[0])

    casimirs = [gf_to_casimir(irrep) for irrep in list(link_bitmap.keys())]

    casimirs_dic = {}

    for i, enc in enumerate(list(link_bitmap.values())):
        casimirs_dic[enc] = casimirs[i]

    value = 0

    for i in range(0,len(bitstring),2):
        value += casimirs_dic[bitstring[i:i+2]] 

    return value


def electric_hamiltonian(link_bitmap):

    # Gives the total Pauli decompsition

    N = len(list(link_bitmap.values())[0])

    casimir_values = [gf_to_casimir(irrep) for irrep in list(link_bitmap.keys())]
    
    pauli_strings = [diagonalpaulistring(list(link_bitmap.values())[i], N, casimir_values[i]) for i in range(len(list(link_bitmap.values())))]

    return [sum(x) for x in zip(*pauli_strings)] 


# Tests 

# Examples of Irrep encoding bitmap for 1,3,3bar and 1,3,3bar,6,6bar,8
IRREP_TRUNCATION_DICT_1_3_3BAR = {
    (0, 0, 0): "00",
    (1, 0, 0): "10",
    (1, 1, 0): "01"
}

IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8 = {
    (0, 0, 0): "000",
    (1, 0, 0): "100",
    (1, 1, 0): "001",
    (2, 0, 0): "110",
    (2, 2, 0): "011",
    (2, 1, 0): "111"
}


# Pauli matrices for testing 
III = np.kron(np.kron([[1,0],[0,1]], [[1,0],[0,1]]), [[1,0],[0,1]])
IIZ = np.kron(np.kron([[1,0],[0,1]], [[1,0],[0,1]]), [[1,0],[0,-1]])
IZI = np.kron(np.kron([[1,0],[0,1]], [[1,0],[0,-1]]), [[1,0],[0,1]])
IZZ = np.kron(np.kron([[1,0],[0,1]], [[1,0],[0,-1]]), [[1,0],[0,-1]])
ZII = np.kron(np.kron([[1,0],[0,-1]], [[1,0],[0,1]]), [[1,0],[0,1]])
ZIZ = np.kron(np.kron([[1,0],[0,-1]], [[1,0],[0,1]]), [[1,0],[0,-1]])
ZZI = np.kron(np.kron([[1,0],[0,-1]], [[1,0],[0,-1]]), [[1,0],[0,1]])
ZZZ = np.kron(np.kron([[1,0],[0,-1]], [[1,0],[0,-1]]), [[1,0],[0,-1]])

II = np.kron([[1,0],[0,1]], [[1,0],[0,1]])
IZ = np.kron([[1,0],[0,1]], [[1,0],[0,-1]])
ZI = np.kron([[1,0],[0,-1]], [[1,0],[0,1]])
ZZ = np.kron([[1,0],[0,-1]], [[1,0],[0,-1]])

def test_decomp_1_3_3bar():
    bmap = IRREP_TRUNCATION_DICT_1_3_3BAR

    N = len(list(bmap.values())[0])

    # Generate decomp matrix

    decomp = electric_hamiltonian(bmap); decomp_matrx = (decomp[0]*II + decomp[1]*IZ + decomp[2]*ZI + decomp[3]*ZZ)

    # Generate keys comparison matrix 

    binary_keys = list(map(lambda x: int(x, 2), list(bmap.values())))

    cmpr_matrx = np.zeros((2**N,2**N))

    casmrs = casimirs(bmap)

    for i in range(len(binary_keys)):
        cmpr_matrx[binary_keys[i], binary_keys[i]] = casmrs[i]

    # Compare matrices

    if (np.isclose(cmpr_matrx, decomp_matrx)).all():
        print("Success for test_decomp_1_3_3bar")
    else:
        print("Failure. Here are the hard-coded and decomp matrix, respecively")
        print(cmpr_matrx); print(decomp_matrx)


def test_decomp_1_3_3bar_6_6bar_8():
    bmap = IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8

    N = len(list(bmap.values())[0])

    # Generate decomp matrix

    decomp = electric_hamiltonian(bmap)

    decomp_matrx = decomp[0]*III + decomp[1]*IIZ + decomp[2]*IZI + decomp[3]*IZZ + decomp[4]*ZII + decomp[5]*ZIZ + decomp[6]*ZZI + decomp[7]*ZZZ

    # Generate keys comparison matrix 

    binary_keys = list(map(lambda x: int(x, 2), list(bmap.values())))

    cmpr_matrx = np.zeros((2**N,2**N))

    casmrs = casimirs(bmap)

    for i in range(len(binary_keys)):
        cmpr_matrx[binary_keys[i], binary_keys[i]] = casmrs[i]

    # Compare matrices

    if (np.isclose(cmpr_matrx, decomp_matrx)).all():
        print("Success for test_decomp_1_3_3bar_6_6bar_8")
    else:
        print("Failure. Here are the hard-coded and decomp matrix, respecively")
        print(cmpr_matrx); print(decomp_matrx)



if __name__ == "__main__":
    test_decomp_1_3_3bar()
    test_decomp_1_3_3bar_6_6bar_8()


