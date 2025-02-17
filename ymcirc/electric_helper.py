"""
Helper functions to construct electric Hamiltonian through Pauli decomposition.

Electric Hamiltonian is diagonal in computational basis. Helper functions first
construct the Pauli decomposition for matrix with 1 at (i,i) and weights
decomposition by the electric Casimir (according to the link_bitmap).
"""
from __future__ import annotations
from ymcirc.conventions import (
    IrrepWeight, BitString, IrrepBitmap, VertexMultiplicityBitmap, LatticeStateEncoder
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



def _handle_electric_energy_unphysical_states(global_state: str, link_state: str, note_unphysical_states : bool, 
    stop_on_unphysical_states : bool):
    """ 
    Handling for unphysical link states in electric energy calculation.
    """
    if (stop_on_unphysical_states):
        # Terminate simulation 
        raise ValueError("Computation of electric energy terminated due to unphysical link state " + str(link_state) + " in " + str(global_state))
    elif (note_unphysical_states):
        # Just print an error, but continue simulation with electric energy for the state 0.0
        print("Unphysical link state " + str(link_state) + " in " + str(global_state) + " found during computation of electric energy. " 
            + "Assigning zero energy")
        return 0.0
    else:
        return 0.0


def casimirs(link_bitmap: IrrepBitmap) -> List[float]:
    """Generate a list of Casimir eigenvalues from link bitmap i-Weights."""
    return [_gt_pattern_iweight_to_casimir(irrep) for irrep in list(link_bitmap.keys())]


def convert_bitstring_to_evalue(
        global_bitstring: str, lattice_encoder: LatticeStateEncoder, note_unphysical_states : bool = True, 
        stop_on_unphysical_states : bool = False) -> float:
    """
    Convert lattice links to total energy.

    Used for computing the average electric energy. 
    Chunks the bitstring into |v l1 l2 ..> for each vertex when 
    there are bag states. Default behavior for unphysical states is to note them and assign 0.0 energy
    """

    link_bitmap = lattice_encoder.link_bitmap
    vertex_bitmap = lattice_encoder.vertex_bitmap

    casimirs = [_gt_pattern_iweight_to_casimir(irrep) for irrep in list(link_bitmap.keys())]
    casimirs_dict = {}

    # encode using iweights (such as (1, 0 ,0))
    for i, iweight in enumerate(list(link_bitmap.keys())):
        casimirs_dict[iweight] = casimirs[i]

    evalue = 0.0
    len_string = lattice_encoder.expected_link_bit_string_length


    has_vertex_bitmap_data = not len(vertex_bitmap) == 0
    if not has_vertex_bitmap_data:
        for i in range(0, len(global_bitstring), len_string):
            link_bitstring_chunk = global_bitstring[i:i + len_string]
            link_state = lattice_encoder.decode_bit_string_to_link_state(link_bitstring_chunk)
            # Handle unphysical link state 
            if (link_state is None):
                return _handle_electric_energy_unphysical_states(global_bitstring, link_bitstring_chunk, 
                    note_unphysical_states, stop_on_unphysical_states)
            evalue += casimirs_dict[link_state]
    else:
        vertex_singlet_length = lattice_encoder.expected_vertex_bit_string_length
        # Find the spatial dimension of the lattice from vertex state 
        spatial_dim = int(len(list(vertex_bitmap.keys())[0][0]) / 2.0)
        if (spatial_dim == 1):
            total_length_x = vertex_singlet_length + (spatial_dim+1) * len_string
            total_length_y = vertex_singlet_length + spatial_dim * len_string

            chunks = [
            global_bitstring[i:i + total_length_x+total_length_y]
            for i in range(0, len(global_bitstring), total_length_x+total_length_y)
            ]

            for chunk in chunks:
                x_chunk = chunk[:total_length_x]; y_chunk = chunk[total_length_x:total_length_x+total_length_y]
                
                for i in range(vertex_singlet_length, total_length_x, len_string):
                    link_bitstring_chunk = x_chunk[i:i + len_string]
                    link_state = lattice_encoder.decode_bit_string_to_link_state(link_bitstring_chunk)
                    # Handle unphysical link state 
                    if (link_state is None):
                        return _handle_electric_energy_unphysical_states(global_bitstring, link_bitstring_chunk, 
                            note_unphysical_states, stop_on_unphysical_states)
                    evalue += casimirs_dict[link_state]
                for i in range(vertex_singlet_length, total_length_y, len_string):
                    link_bitstring_chunk = y_chunk[i:i + len_string]
                    link_state = lattice_encoder.decode_bit_string_to_link_state(link_bitstring_chunk)
                    # Handle unphysical link state 
                    if (link_state is None):
                        return _handle_electric_energy_unphysical_states(global_bitstring, link_bitstring_chunk, 
                            note_unphysical_states, stop_on_unphysical_states)
                    evalue += casimirs_dict[link_state]

        else:
            total_length = vertex_singlet_length + spatial_dim * len_string

            chunks = [
            global_bitstring[i:i + total_length]
            for i in range(0, len(global_bitstring), total_length)
            ]

            for chunk in chunks:
                for i in range(vertex_singlet_length, len(chunk), len_string):
                    link_bitstring_chunk = chunk[i:i + len_string]
                    link_state = lattice_encoder.decode_bit_string_to_link_state(link_bitstring_chunk)
                    if (link_state is None):
                        return _handle_electric_energy_unphysical_states(global_bitstring, link_bitstring_chunk, 
                            note_unphysical_states, stop_on_unphysical_states)
                    evalue += casimirs_dict[link_state]
        
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


def _test_convert_bitstring_to_evalue_case1():
    print("Testing if the electric energy |E|^2 is being calculated correctly for d2, T1p vacuum")
 
    # Use the T1 truncation with T1p vertex maps for ease of creating tests. 
    # Tests strings may or not physical
    T1_map = {(0, 0, 0): "00", (1, 0, 0): "10", (1, 1, 0): "01"}

    vertex_d2_T1p_map = {
    (((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), 1): '0',
    (((0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 1, 0)), 1): '1',
    }

    # 20 qubits in d=2, 2x2, T1p. Expect zero energy 
    example_d2_T1p_string1 = "0"*20

    lattice_encoder = LatticeStateEncoder(T1_map, vertex_d2_T1p_map)

    if (np.isclose(0.0, convert_bitstring_to_evalue(example_d2_T1p_string1, lattice_encoder))):
        print("Test passed")
    else:
        print(
            "Failure for d2, T1p vacuum. The calculated energy was " + 
            str(convert_bitstring_to_evalue(example_d2_T1p_string1, lattice_encoder)) + " instead of " + str(0.0)
        )



def _test_convert_bitstring_to_evalue_case2():
    print("Testing if the electric energy |E|^2 is being calculated correctly for d2, T1p excited state")

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

    if (np.isclose((4.0/3.0)*6.0, convert_bitstring_to_evalue(example_d2_T1p_string2, lattice_encoder))):
        print("Test passed")
    else:
        print(
            "Failure d2, T1p excited state. The calculated energy was " + 
            str(convert_bitstring_to_evalue(example_d2_T1p_string2, lattice_encoder)) + " instead of " + str((4.0/3.0)*6.0)
        )

def _test_convert_bitstring_to_evalue_case3():
    print("Testing if the electric energy |E|^2 is being calculated correctly for d3/2, T1p excited state")

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

    if (np.isclose((4.0/3.0)*4.0, convert_bitstring_to_evalue(example_d32_T1p_string1, lattice_encoder))):
        print("Test passed")
    else:
        print(
            "Failure d3/2, T1p excited state. The calculated energy was " + 
            str(convert_bitstring_to_evalue(example_d32_T1p_string1, lattice_encoder)) + " instead of " + str((4.0/3.0)*4.0)
        )

def _test_convert_bitstring_to_evalue_case4():
    print("Testing if the electric energy |E|^2 is being calculated correctly for d3/2, T1p unphysical state "  
        "with just noting an error")

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

    if (np.isclose(0.0, convert_bitstring_to_evalue(example_d32_T1p_string2, lattice_encoder))):
        print("Test passed")
    else:
        print(
            "Failure d3/2, T1p excited state. The calculated energy was " + 
            str(convert_bitstring_to_evalue(example_d32_T1p_string2, lattice_encoder)) + " instead of " + 0.0)


def _test_convert_bitstring_to_evalue_case5():
    print("Testing if the electric energy |E|^2 is being calculated correctly for d3/2, T1p unphysical state " + 
        "with raising an error")

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

    try:
        convert_bitstring_to_evalue(example_d32_T1p_string2, lattice_encoder, stop_on_unphysical_states=True)
    except Exception as e:
        print("Test passed, caught exception: " + str(e))
    


if __name__ == "__main__":
    print("Running test...")
    _test_decomp_1_3_3bar()
    _test_decomp_1_3_3bar_6_6bar_8()
    _test_convert_bitstring_to_evalue_case1()
    _test_convert_bitstring_to_evalue_case2()
    _test_convert_bitstring_to_evalue_case3()
    _test_convert_bitstring_to_evalue_case4()
    _test_convert_bitstring_to_evalue_case5()
    print("All tests passed.")
    

