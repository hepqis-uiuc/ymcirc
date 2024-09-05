#!/usr/bin/env python3
# Andrew Lytle
# Sept 2024

import numpy as np

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
def H1_H2(gsq):
    terms1 = ['II', 'ZI', 'IZ', 'XI', 'IX']
    terms2 = ['ZZ', 'XX', 'YY']
    H_decomp = to_pauli_vec(H(gsq))
    H1 = {P: H_decomp[P] for P in terms1}
    H2 = {P: H_decomp[P] for P in terms2}
    return from_pauli_vec(H1), from_pauli_vec(H2)

H1, H2 = H1_H2(1)
print(((H1 + H2)==H(1)).all())
