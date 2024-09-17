#!/usr/bin/env python3
# Andrew Lytle
# Sept 2024

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Operator

from qiskit_aer.primitives import Sampler#, Estimator
from qiskit.primitives import Estimator

from decompose_pauli import to_pauli_vec, from_pauli_vec
from Trotter_evol import Trotter_evol2

# Eqs 39-40.
def Hspec(sector='++'):
    sqrt2 = np.sqrt(2)
    match sector:
        case '++':
            _HE = np.array([[0,0,0,0], 
                            [0,16/3,0,0], 
                            [0,0,16/3,0], 
                            [0,0,0,8]])

            _HB = np.array([[6,-2,0,0], 
                            [-2,5,-sqrt2/9,-sqrt2/3], 
                            [0,-sqrt2/9,6,-2/3], 
                            [0,-sqrt2/3,-2/3,6]])
        case '+-':
            _HE = np.array([[0,0],[0,0]])
            _HB = np.array([[0,0],[0,0]])
        case '-+':
            _HE = np.array([[0,0],[0,0]])
            _HB = np.array([[0,0],[0,0]])
        case '--':
            _HE = np.array([[0,0],[0,0]])
            _HB = np.array([[0,0],[0,0]])
    return _HE, _HB

def H(gsq, sector='++'):
    _HE, HB = Hspec(sector)
    return (gsq/2)*_HE + (1/gsq)*_HB

def remove_zero_entries(_d):
    return {x:_d[x] for x in _d if (_d[x] != 0)}

def check_H_specification():
    _HE, _HB = Hspec(sector='++')
    print(remove_zero_entries(to_pauli_vec(_HE)))  # Eq 41.
    print()
    print(remove_zero_entries(to_pauli_vec(_HB)))  # Eq 41.

if __name__ == "__main__":
    check_H_specification()