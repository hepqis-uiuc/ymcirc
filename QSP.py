#!/usr/bin/env python

import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator


def one_locations(binary_string):
    "10011 --> [0, 1, 4]"
    return [i for i, b in enumerate(reversed(binary_string)) if b == '1']

def zero_locations(binary_string):
    "10011 --> [2, 3]"
    return [i for i, b in enumerate(reversed(binary_string)) if b == '0']
        

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

def test_oracle(oracle, results):
    m = 4
    #for i in range(2**(m/2)):
    for i in (2, 0):
        circ = prepare_basis_state(i, m)
        circ.compose(oracle)
        print(Operator.from_circuit(circ))

def test():
    print(one_locations('10011'))
    #print(oracle_1sparse([(2, 1),]))
    #print(oracle_1sparse([(0, 3),]))
    #print(oracle_1sparse([(0, 3), (2, 1)]))

    #circ_11 = QuantumCircuit(4,4)
    #circ_11.mcx([0,1], [2, 3])

    oracle = oracle_1sparse([(2, 1), (0, 3)])
    test_oracle(oracle, None)

    '''
    circ_11 = QuantumCircuit(4,4)
    circ_11.x(0)
    circ_11.x(1)
    result = circ_11.compose(oracle)
    print(result)
    '''

if __name__ == "__main__":
    test()
    