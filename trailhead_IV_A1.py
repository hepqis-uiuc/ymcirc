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

from decompose_pauli import to_pauli_vec, from_pauli_vec, remove_zero_entries
from Trotter_evol import Trotter_evol2
from trailhead_III_B1 import Tstep2_circ

# Eqs 39-40.
def Hspec(sector='++'):
    sqrt2 = np.sqrt(2)
    if sector == '++':
        _HE = np.array([[0,0,0,0], 
                        [0,16/3,0,0], 
                        [0,0,16/3,0], 
                        [0,0,0,8]])

        _HB = np.array([[6,-2,0,0], 
                        [-2,5,-sqrt2/9,-sqrt2/3], 
                        [0,-sqrt2/9,6,-2/3], 
                        [0,-sqrt2/3,-2/3,6]])
    elif sector == '+-':
        _HE = np.array([[0,0],[0,0]])
        _HB = np.array([[0,0],[0,0]])
    elif sector == '-+':
        _HE = np.array([[0,0],[0,0]])
        _HB = np.array([[0,0],[0,0]])
    elif sector == '--':
        _HE = np.array([[0,0],[0,0]])
        _HB = np.array([[0,0],[0,0]])
    else:
        raise ValueError(f"{sector = } not recognized.")
    
    return _HE, _HB

def H(gsq, sector='++'):
    _HE, _HB = Hspec(sector)
    return (gsq/2)*_HE + (1/gsq)*_HB

def check_H_specification():
    _HE, _HB = Hspec(sector='++')
    print(remove_zero_entries(to_pauli_vec(_HE)))  # Eq 41.
    print()
    print(remove_zero_entries(to_pauli_vec(_HB)))  # Eq 41.

# For Trotterization, the Pauli decomposition is grouped into commuting sets.
def H1_H2_H3(gsq):
    terms1 = ['II', 'ZI', 'IZ', 'XI', 'IX']
    terms2 = ['ZZ', 'XX', 'YY']
    terms3 = ['XZ', 'ZX']
    H_decomp = to_pauli_vec(H(gsq))
    H1 = {P: H_decomp[P] for P in terms1}
    H2 = {P: H_decomp[P] for P in terms2}
    H3 = {P: H_decomp[P] for P in terms3}
    print(f"{H1 = }")
    print(f"{H2 = }")
    print(f"{H3 = }")
    return from_pauli_vec(H1), from_pauli_vec(H2), from_pauli_vec(H3)

def test_H_decomposition(gsq):
    H1, H2, H3 = H1_H2_H3(gsq)
    #print((H1 + H2 + H3) == H(gsq))
    #print((H1 + H2 + H3) - H(gsq))
    print(np.allclose(H1 + H2 + H3, H(gsq), rtol=1e-16))

def Tstep3_circ(th_ZX, th_XZ, dt):
    pass

def test_Trotter_circuits():
    pass

if __name__ == "__main__":
    check_H_specification()
    test_H_decomposition(1)