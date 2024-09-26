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
    return (gsq/2)*_HE + 1/(2*gsq)*_HB

def electric_EV(gsq, t):
    psi_i = np.array([1, 0, 0, 0])
    _H = H(gsq)
    _HE, _ = Hspec(sector='++')
    psi_t = np.dot(expm(-1j*t*_H), psi_i)
    E2 = _HE
    psiE2psi = np.dot(np.conjugate(psi_t), np.dot(E2, psi_t))
    
    return np.abs(psiE2psi)

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

def Tstep1_circ(th_II, th_IZ, th_ZI, th_IX, th_XI, dt):
    circ = QuantumCircuit(2,2, global_phase=-th_II*dt)
    circ.rz(2*th_IZ*dt, 0)
    circ.rz(2*th_ZI*dt, 1)
    circ.rx(2*th_IX*dt, 0)
    circ.rx(2*th_XI*dt, 1)
    return circ

def Tstep3_circ(th_ZX, th_XZ, dt):
    "Eq. 43 of trailhead."
    circ = QuantumCircuit(2, 2)
    circ.h(1)
    circ.cx(1, 0)
    circ.h(1)
    circ.rz(2*th_ZX*dt, 1)
    circ.rz(2*th_XZ*dt, 0)
    circ.h(1)
    circ.cx(1, 0)
    circ.h(1)
    return circ

def test_Trotter_circuits():
    H1, H2, H3 = H1_H2_H3(1)
    dt = 0.2
    sqrt2 = np.sqrt(2)
    circ1 = Tstep1_circ(7/3+23/8, 1/8-1, -(1/8+1), -2/3, -1/(6*sqrt2), dt)
    eHtest = expm(-1j*(dt)*H1)
    print(Operator.from_circuit(circ1))
    print(Operator(eHtest))
    print(Operator.from_circuit(circ1) == Operator(eHtest))
    circ2 = Tstep2_circ(-1/(18*sqrt2), -1/(18*sqrt2), 1/8-1/3, dt)
    eHtest = expm(-1j*(dt)*H2)
    print(Operator.from_circuit(circ2) == Operator(eHtest))
    circ3 = Tstep3_circ(-1/3, 1/(6*sqrt2), dt)
    eHtest = expm(-1j*(dt)*H3)
    print(Operator.from_circuit(circ3) == Operator(eHtest))

def Trotter_exact():
    H1, H2, H3 = H1_H2_H3(1)
    dts = np.linspace(0, 5, 100)
    _res = [np.dot(expm(-1j*dt*H3), np.dot(expm(-1j*dt*H2), expm(-1j*dt*H1))) for dt in dts]
    probs = [np.abs(_r[0,0])**2 for _r in _res]
    return probs

def run_Trotter(tmin, tmax, tsteps, order, trotter_steps):
    dts = np.linspace(tmin, tmax, tsteps)
    probs = []
    time = Parameter("time")
    sqrt2 = np.sqrt(2)
    for _dt in dts:
        sampler = Sampler()
        Trotter_steps = [Tstep3_circ(-1/3, 1/(6*sqrt2), time), 
                         Tstep2_circ(-1/(18*sqrt2), -1/(18*sqrt2), 1/8-1/3, time), 
                         Tstep1_circ(7/3+23/8, 1/8-1, -(1/8+1), -2/3, -1/(6*sqrt2), time)]
        circuit = Trotter_evol2([], Trotter_steps, time, _dt, order=order, nsteps=trotter_steps)
        circuit.measure_all()
        try:
            probs.append(sampler.run(circuit).result().quasi_dists[0][0])
        except KeyError:
            probs.append(0)
    return dts, probs

def plot_state_evol():
    dts = np.linspace(0, 5, 100)
    probs = [np.abs(expm(-1j*dt*H(1, sector='++'))[0,0])**2 for dt in dts]
    probs2 = Trotter_exact()

    dts_trotter11, probs_trotter11 = run_Trotter(0, 2, 40, 1, 1)
    dts_trotter14, probs_trotter14 = run_Trotter(0, 3, 40, 1, 4)
    print(probs_trotter11)

    fig, ax = plt.subplots()
    ax.set_ylim([0.0, 1.1])
    ax.set_xlim([0, 3])
    ax.set_ylabel("P(0)")
    ax.set_xlabel("time")
    ax.plot(dts, probs, 'k-', label='exact')
    ax.plot(dts, probs2, 'b-', label='Trotter1')
    ax.plot(dts_trotter11, probs_trotter11, 'o--', ms=2, label='1 step')
    ax.plot(dts_trotter14, probs_trotter14, 'o--', ms=2, label='4 steps')

    plt.legend()
    #plt.show()
    plt.savefig("./trailhead_Fig12a.pdf")

def plot_electric_energy():
    xlim = [0, 30]
    dts = np.linspace(*xlim, xlim[1]*10)
    Es_exact = [electric_EV(1, dt) for dt in dts]

    fig, ax = plt.subplots()
    ax.set_ylim([0.0, 2.5])
    ax.set_xlim(xlim)
    ax.set_ylabel("<E^2>")
    ax.set_xlabel("time")
    ax.plot(dts, Es_exact, 'k-', label='exact')
    #ax.plot(dts, evs4, 'o', ms=2, label='4 steps')
    #ax.plot(dts, evs6, 'o', ms=2, label='6 steps')

    plt.legend()
    #plt.show()
    plt.savefig("./trailhead_Fig11b.pdf")

if __name__ == "__main__":
    test_H_decomposition(1)
    test_Trotter_circuits()
    #plot_state_evol()
    plot_electric_energy()