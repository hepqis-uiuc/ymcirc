#!/usr/bin/env python

from __future__ import annotations
from collections.abc import Iterable
import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)

import numpy as np

from qiskit import QuantumRegister
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector


def one_locations(binary_string):
    "10011 --> [0, 1, 4]"
    return [i for i, b in enumerate(reversed(binary_string)) if b == '1']

def zero_locations(binary_string):
    "10011 --> [2, 3]"
    return [i for i, b in enumerate(reversed(binary_string)) if b == '0']
        

# TODO infer m and N from specification, or make this an argument to the function?
specification = [(0, 2), (2, 0), (1, 3), (3, 1)]
def oracle_1sparse(specs):
    "|i>|0> --> |i>|j>"
    m = 2
    N = 2**m
    circ = QuantumCircuit(2*m, 2*m)
    for spec in specs:
        i, j = spec
        i = np.binary_repr(i, width=m)  # Need padding here..
        j = np.binary_repr(j*4, width=2*m)
        logger.debug(f"{zero_locations(i) = }")
        logger.debug(f"{one_locations(j) = }")

        for loc in zero_locations(i):
            circ.x(loc)
        logger.debug(f"{i = }")
        logger.debug(f"{j = }")
        for loc in one_locations(j):  # Concatenate these for now..
            circ.mcx([0, 1], loc)
        for loc in zero_locations(i):
            circ.x(loc)
    return circ

def prepare_basis_state(i, m):
    "Circuit representing |i>"
    circ = QuantumCircuit(m, m)
    i = np.binary_repr(i, width=m)
    for loc in one_locations(i):
        circ.x(loc)
    return circ

def get_min_qubit_requirements(number: int | list[int]):
    """
    Return number of qubits required for binary encoding of number.

    If input is a list of ints, find the biggest one first, then
    give the qubit requiremnts for that number.
    """
    if isinstance(number, int):
        return len(np.binary_repr(number))
    elif isinstance(number, list):
        return len(np.binary_repr(max(set(number))))
    else:
        raise ValueError("Expected a number, or a list of numbers.")

def flatten(iterable) -> list:
    "Return iterable as a flattened list."
    # Base case for recursion.
    if not isinstance(iterable, Iterable):
        return iterable

    result = []
    for item in iterable:
        flattened_item = flatten(item)
        result += flattened_item if isinstance(flattened_item, Iterable) else [flattened_item]

    return result


def oracle_integer_comparator(specs: list[tuple[int, int]]):
    """
    |i>|j>|0> --> |i>|j>|i > j>

    Takes i and j to be length k bit strings (if one is smaller, it is padded).
    Initializes a scratch quantum register of k qubits.
    """
    n_qubits_i_or_j = get_min_qubit_requirements(flatten(specs))
    n_total_qubits = 3*n_qubits_i_or_j + 1
    i_reg_zero_idx = 0
    j_reg_zero_idx = n_qubits_i_or_j
    scratch_reg_zero_idx = 2*n_qubits_i_or_j
    output_reg_idx = n_total_qubits - 1
    
    # if n_qubits_i_or_j < 1:
    #     raise ValueError("Must specify a positive integer number of qubits.")

    N = 2**n_qubits_i_or_j  # Dimensionality of the Hilbert space

    circ = QuantumCircuit(n_total_qubits)

    # Bitwise scan through |i> and |j> registers.
    # Set corresponding scratch qubit in intermediate |a>
    # register to 1 if |i> has a 1 where |j> is 0,
    # and NO OTHER more-significant scratch qubits have been
    # flipped.
    circ.x(j_reg_zero_idx)
    circ.mcx([i_reg_zero_idx, j_reg_zero_idx], scratch_reg_zero_idx)
    circ.x(j_reg_zero_idx)
    for k in range(1, n_qubits_i_or_j):
        i_reg_idx = k
        j_reg_idx = j_reg_zero_idx + (k - 1)
        scratch_reg_ctrl_idx = scratch_reg_zero_idx + (k - 1)
        scratch_reg_tgt_idx = scratch_reg_zero_idx + k
        circ.x(j_reg_idx)
        circ.x(scratch_reg_ctrl_idx)
        circ.mcx([i_reg_idx, j_reg_idx, scratch_reg_ctrl_idx], scratch_reg_tgt_idx)
        circ.x(scratch_reg_ctrl_idx)
        circ.x(j_reg_idx)

    # Bitwise scan through scratch register.
    # If there is a |1>, set the output register to |1>.
    # Since there is at most one |1> in the scratch
    # register, this is implemented as a bunch of CX gates.
    for k in range(n_qubits_i_or_j):
        scratch_reg_ctrl_idx = scratch_reg_zero_idx + k
        tgt_idx = output_reg_idx
        circ.cx(scratch_reg_ctrl_idx, tgt_idx)

    return circ

def test_oracle_1sparse(oracle, specification):
    m = 4
    A = int(2**(m/2))
    N = 2**m
    for i, j in specification:
        sv = Statevector.from_int(i, N)
        sv = sv.evolve(Operator.from_circuit(oracle))
        print(sv == Statevector.from_int(i+A*j, N))

def test_flatten():
    print(flatten(2) == 2)
    print(flatten([1, 3, 4]) == [1, 3, 4])
    print(flatten([]) == [])
    print(flatten([(1, 3), (3, 1), (2, 4), (4, 2)]) == [1, 3, 3, 1, 2, 4, 4, 2])
    print(flatten([(1, 3), (3, 1), (2, 4), (4, 2)]))
    print(flatten([(1, 2), 3, [4]]) == [1, 2, 3, 4])

def test_oracle_integer_comparator(oracle, n_qubits):
    raise NotImplementedError()
        
def test():
    #print(one_locations('10011'))
    #print(oracle_1sparse([(2, 1),]))
    #print(oracle_1sparse([(0, 3),]))
    #print(oracle_1sparse([(0, 3), (2, 1)]))

    #circ_11 = QuantumCircuit(4,4)
    #circ_11.mcx([0,1], [2, 3])

    print("Testing oracle_1sparse...")
    
    spec = [(2, 1), (1, 2), (0, 3), (3, 0)]
    oracle = oracle_1sparse(spec)
    print(oracle)
    test_oracle_1sparse(oracle, spec)

    '''
    circ_11 = QuantumCircuit(4,4)
    circ_11.x(0)
    circ_11.x(1)
    result = circ_11.compose(oracle)
    print(result)
    '''
    '''
    reg_a = QuantumRegister(3, 'a')
    number_a = QuantumCircuit(reg_a)
    number_a.initialize(2) # Number 2; |010>
    #print(Operator.from_circuit(number_a))

    sv = Statevector([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    #print(sv.evolve(Operator.from_circuit(oracle)))
    sv = Statevector.from_int(12, 16)
    #print(sv)

    svi = Statevector.from_int(2, 16)
    print(svi)
    svf = svi.evolve(Operator.from_circuit(oracle))
    svf2 = Statevector.from_int(2+1*4, 16)
    print(svf == svf2)
    '''

    print("Checking that flatten method works...")
    test_flatten()
    print("Testing oracle_integer_comparator...")
    

    specifications = [(1, 2), (2, 1), (7, 4), (4, 7)]
    oracle_cmp = oracle_integer_comparator(specs=specifications)
    print(oracle_cmp)
    

if __name__ == "__main__":
    test()
    
