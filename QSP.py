#!/usr/bin/env python

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

def test_oracle(oracle, specification):
    m = 4
    A = int(2**(m/2))
    N = 2**m
    for i, j in specification:
        sv = Statevector.from_int(i, N)
        sv = sv.evolve(Operator.from_circuit(oracle))
        print(sv == Statevector.from_int(i+A*j, N))

def test():
    #print(one_locations('10011'))
    #print(oracle_1sparse([(2, 1),]))
    #print(oracle_1sparse([(0, 3),]))
    #print(oracle_1sparse([(0, 3), (2, 1)]))

    #circ_11 = QuantumCircuit(4,4)
    #circ_11.mcx([0,1], [2, 3])

    spec = [(2, 1), (1, 2), (0, 3), (3, 0)]
    oracle = oracle_1sparse(spec)
    print(oracle)
    test_oracle(oracle, spec)

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

if __name__ == "__main__":
    test()
    