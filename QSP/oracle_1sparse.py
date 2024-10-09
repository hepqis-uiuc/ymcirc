#!/usr/bin/env python

import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)

from math import log

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
        

def validate_input(input):
    "Some checks to ensure oracle is specified consistently."
    N = len(input)
    m = log(N, 2)

    # Make sure N=2**m for some integer m.
    if m.is_integer():
        m = int(m)
    else:
        raise ValueError(f"{N = } {m = }. m should be integer with N=2**m.")
    
    # Check all values appear once.
    ivals = list(range(N))
    jvals = list(range(N))
    for (i, j) in input:
        try:
            ivals.remove(i)
            jvals.remove(j)
        except ValueError:
            print(f"{i = } {j = }")
            print(f"{ivals = } {jvals = }. Matrix specification not 1-sparse")
            raise
    
    return m

def oracle_1sparse(specs):
    "|i>|0> --> |i>|j>"
    m = validate_input(specs)
    N = 2**m
    circ = QuantumCircuit(2*m, 2*m)
    for spec in specs:
        i, j = spec
        i = np.binary_repr(i, width=m)  # Need padding here..
        j = np.binary_repr(j*N, width=2*m)
        logger.debug(f"{zero_locations(i) = }")
        logger.debug(f"{one_locations(j) = }")
        
        for loc in zero_locations(i):
            circ.x(loc)
        logger.debug(f"{i = }")
        logger.debug(f"{j = }")
        for loc in one_locations(j):  # Concatenate these for now..
            circ.mcx(list(range(m)), loc)
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
    m = validate_input(specification)
    A = 2**m
    N = 2**(2*m)
    for i, j in specification:
        sv = Statevector.from_int(i, N)
        sv = sv.evolve(Operator.from_circuit(oracle))
        print(sv == Statevector.from_int(i+A*j, N))

def test():
    spec = [(2, 1), (1, 2), (0, 3), (3, 0)]
    validate_input(spec)
    oracle = oracle_1sparse(spec)
    print(oracle)
    test_oracle(oracle, spec)

if __name__ == "__main__":
    test()
    