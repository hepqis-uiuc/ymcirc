#!/usr/bin/env python3
# Andrew Lytle
# Sept 2024

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Operator

from qiskit_aer.primitives import Sampler

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

def check_H_specification():
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
    #print(f"{H1 = }")
    #print(f"{H2 = }")
    #print(f"{H3 = }")
    return from_pauli_vec(H1), from_pauli_vec(H2), from_pauli_vec(H3)

def test_H_decomposition():
    H1, H2, H3 = H1_H2_H3(1)
    print(((H1 + H2 + H3)==H(1)).all())

def Tstep1_circ(th_II, th_IZ, th_ZI, dt=None):
    if dt is None:
        dt = Parameter("dt")
    circ = QuantumCircuit(2,2, global_phase=-th_II*dt)
    circ.rz(2*th_IZ*dt, 0)
    circ.rz(2*th_ZI*dt, 1)
    return circ

def Tstep2_circ(th_xx, th_yy, th_zz, dt=None):
    if dt is None:
        dt = Parameter("dt")
    circ = QuantumCircuit(2, 2)
    circ.rxx(2*th_xx*dt, 0, 1)
    circ.ryy(2*th_yy*dt, 0, 1)
    circ.rzz(2*th_zz*dt, 0, 1)
    return circ

def Tstep3_circ(th_IX, th_XI, dt=None):
    if dt is None:
        dt = Parameter("dt")
    circ = QuantumCircuit(2, 2)
    circ.rx(2*th_IX*dt, 0) 
    circ.rx(2*th_XI*dt, 1)
    return circ

def test_Trotter_circuits():
    H1, H2, H3 = H1_H2_H3(1)
    dt = 0.2
    circ1 = Tstep1_circ(3+17/6, -1.5, -1.5, dt)
    eHtest = expm(-1j*(dt)*H1)
    print(Operator.from_circuit(circ1) == Operator(eHtest))

    circ2 = Tstep2_circ(-1/4, -1/4, 1/6, dt)
    eHtest = expm(-1j*(dt)*H2)
    print(Operator.from_circuit(circ2) == Operator(eHtest))

    circ3 = Tstep3_circ(-0.5, -0.5, dt)
    eHtest = expm(-1j*(dt)*H3)
    print(Operator.from_circuit(circ3) == Operator(eHtest))

    #Trotter_evol([circ1, circ2, circ3])

def Trotter_evol(state, circuits, order=1):
    circuit = QuantumCircuit(2, 2)
    for _c in circuits:
        circuit = circuit.compose(_c)
    return circuit

def Trotter_evol2(state, circuits, time, Dt, order=1, steps=1):
    circuit = QuantumCircuit(2, 2)
    if order == 1:
        dt = Dt/steps
        for _ in range(steps):
            for _c in circuits:
                circuit = circuit.compose(_c.assign_parameters({time: dt}))
    
    if order == 2:
        dt = Dt/steps
        for _ in range(steps):
            _circuits, _fcircuit = circuits[:-1], circuits[-1]
            _circuits = [_c.assign_parameters({time: dt/2}) for _c in _circuits]
            _fcircuit = _fcircuit.assign_parameters({time: dt})
            _circuits = _circuits + [_fcircuit,] + list(reversed(_circuits))
            for _c in _circuits:
                circuit = circuit.compose(_c)
    return circuit

sampler = Sampler()
dt = 0.5
Trotter_steps = [Tstep1_circ(3+17/6, -1.5, -1.5, dt), 
                 Tstep3_circ(-0.5, -0.5, dt),
                 Tstep2_circ(-1/4, -1/4, 1/6, dt), 
                 ]
circuit = Trotter_evol([], Trotter_steps)
circuit.measure_all()
print(sampler.run(circuit).result().quasi_dists[0][0])

# dts_trotter = np.linspace(0, 1, 10)
# probs_trotter = []
# for dt in dts_trotter:
#     sampler = Sampler()
#     Trotter_steps = [Tstep1_circ(3+17/6, -1.5, -1.5, dt), 
#                  Tstep2_circ(-1/4, -1/4, 1/6, dt), 
#                  Tstep3_circ(-0.5, -0.5, dt)]
#     circuit = Trotter_evol([], Trotter_steps)
#     circuit.measure_all()
#     try:
#         probs_trotter.append(sampler.run(circuit).result().quasi_dists[0][0])
#     except KeyError:
#         probs_trotter.append(0)

def run_Trotter(tmin, tmax, nsteps, order, steps):
    dts = np.linspace(tmin, tmax, nsteps)
    probs = []
    time = Parameter("time")
    for _dt in dts:
        sampler = Sampler()
        Trotter_steps = [Tstep1_circ(3+17/6, -1.5, -1.5, time), 
                        Tstep2_circ(-1/4, -1/4, 1/6, time), 
                        Tstep3_circ(-0.5, -0.5, time)]
        circuit = Trotter_evol2([], Trotter_steps, time, _dt, order=order, steps=steps)
        circuit.measure_all()
        try:
            probs.append(sampler.run(circuit).result().quasi_dists[0][0])
        except KeyError:
            probs.append(0)
    return dts, probs

dts = np.linspace(0, 5, 100)
probs = [np.abs(expm(-1j*dt*H(1))[0,0])**2 for dt in dts]

dts_trotter11, probs_trotter11 = run_Trotter(0, 2, 40, 1, 1)
dts_trotter21, probs_trotter21 = run_Trotter(0, 2, 40, 2, 1)
dts_trotter14, probs_trotter14 = run_Trotter(0, 5, 100, 1, 4)
dts_trotter24, probs_trotter24 = run_Trotter(0, 5, 100, 2, 4)
dts_trotter12, probs_trotter12 = run_Trotter(0, 5, 100, 1, 2)
dts_trotter13, probs_trotter13 = run_Trotter(0, 5, 100, 1, 3)
dts_trotter16, probs_trotter16 = run_Trotter(0, 5, 100, 1, 6)

fig, ax = plt.subplots()
ax.set_ylim([0.0, 1.2])
ax.set_xlim([0, 5])
ax.set_ylabel("P(0)")
ax.set_xlabel("time")
ax.plot(dts, probs, 'k-', label='exact')
ax.plot(dts_trotter11, probs_trotter11, 'o--', ms=2, label='1 step')
ax.plot(dts_trotter12, probs_trotter12, 'o--', ms=2, label = '2 steps')
#ax.plot(dts_trotter13, probs_trotter13, 'o--', ms=2)
#ax.plot(dts_trotter21, probs_trotter21, 'o--', ms=2)
ax.plot(dts_trotter14, probs_trotter14, 'o--', ms=2, label='4 steps')
#ax.plot(dts_trotter24, probs_trotter24, 'o--', ms=2)
ax.plot(dts_trotter16, probs_trotter16, 'o--', ms=2, label='6 steps')
plt.legend()
#plt.show()
plt.savefig("./trailhead_Fig6.pdf")

#dt = Parameter("dt")
#print(Tstep1_circ(3+17/6, -1.5, -1.5, dt))
#Tstep1_circ(3+17/6, -1.5, -1.5, dt).assign_parameters({dt: 0.1})

if __name__ == "__main__":
    check_H_specification()
    test_H_decomposition()
    test_Trotter_circuits()
