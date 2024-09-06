#!/usr/bin/env python3
# Andrew Lytle
# Sept 2024

import numpy as np
from scipy.linalg import expm

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import CXGate, RXGate, XGate
from qiskit.quantum_info import Operator, Pauli, process_fidelity

from decompose_pauli import to_pauli_vec, from_pauli_vec

# Eq 14.
_HE = np.array([[0,0,0,0], 
                [0,16/3,0,0], 
                [0,0,16/3,0], 
                [0,0,0,12]])

_HB = np.array([[3,-1/2,-1/2,0], 
                [-1/2,3,-1/2,-1/2], 
                [-1/2,-1/2,3,-1/2], 
                [0,-1/2,-1/2,3]])

def remove_zero_entries(_d):
    return {x:_d[x] for x in _d if (_d[x] != 0)}

print(remove_zero_entries(to_pauli_vec(3*_HE)))  # Eq 15.
print()
print(remove_zero_entries(to_pauli_vec(_HB)))  # Eq 16.

def H(gsq):
    return (gsq/2)*_HE + (1/gsq)*_HB

# For Trotterization, the Pauli decomposition is grouped into commuting sets.
def H1_H2_H3(gsq):
    terms1 = ['II', 'ZI', 'IZ']
    terms2 = ['ZZ', 'XX', 'YY']
    terms3 = ['XI', 'IX']
    H_decomp = to_pauli_vec(H(gsq))
    H1 = {P: H_decomp[P] for P in terms1}
    H2 = {P: H_decomp[P] for P in terms2}
    H3 = {P: H_decomp[P] for P in terms3}
    print(f"{H1 = }")
    print(f"{H2 = }")
    print(f"{H3 = }")
    return from_pauli_vec(H1), from_pauli_vec(H2), from_pauli_vec(H3)

H1, H2, H3 = H1_H2_H3(1)
print(((H1 + H2 + H3)==H(1)).all())
print()

def Tstep1_circ(th_II, th_IZ, th_ZI, dt):
    circ = QuantumCircuit(2,2, global_phase=-th_II*dt)
    circ.rz(2*th_IZ*dt, 0)
    circ.rz(2*th_ZI*dt, 1)
    return circ

def Tstep2_circ(th_xx, th_yy, th_zz, dt):
    circ = QuantumCircuit(2, 2)
    circ.rxx(2*th_xx*dt, 0, 1)
    circ.ryy(2*th_yy*dt, 0, 1)
    circ.rzz(2*th_zz*dt, 0, 1)
    return circ

def Tstep3_circ(th_IX, th_XI, dt):
    circ = QuantumCircuit(2, 2)
    circ.rx(2*th_IX*dt, 0) 
    circ.rx(2*th_XI*dt, 1)
    return circ

dt = 0.2
circ = Tstep1_circ(3+17/6, -1.5, -1.5, dt)
eHtest = expm(-1j*(dt)*H1)
print(Operator.from_circuit(circ) == Operator(eHtest))

circ = Tstep2_circ(-1/4, -1/4, 1/6, dt)
eHtest = expm(-1j*(dt)*H2)
print(Operator.from_circuit(circ) == Operator(eHtest))

circ = Tstep3_circ(-0.5, -0.5, dt)
eHtest = expm(-1j*(dt)*H3)
print(Operator.from_circuit(circ) == Operator(eHtest))

