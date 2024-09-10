#!/usr/bin/env python3
# Andrew Lytle
# Sept 2024

from qiskit.circuit import QuantumCircuit

def Trotter_evol2(state, circuits, time, Dt, order=1, nsteps=1):
    """nsteps of Trotter evolution at specified order.

    Args:
        circuits (list) - Circuit decomposition for time evolution. Function assumes
            these are parameterized circuits w/ Parameter("time").
        time - Parameter("time") assumed, allows name to change if needed.
        Dt (float) - Total time evolution parameter.
        order (int) - Order of approximation. order = 1 and 2 currently implemented.
        nsteps (int) - Execute n steps of duration Dt/n, at specified order.
    Returns: Circuit implementing specified Trotter evolution.
    """
    circuit = QuantumCircuit(2, 2)
    if order == 1:
        dt = Dt/nsteps
        for _ in range(nsteps):
            for _c in circuits:
                circuit = circuit.compose(_c.assign_parameters({time: dt}))
    
    if order == 2:
        dt = Dt/nsteps
        for _ in range(nsteps):
            _circuits, _fcircuit = circuits[:-1], circuits[-1]
            _circuits = [_c.assign_parameters({time: dt/2}) for _c in _circuits]
            _fcircuit = _fcircuit.assign_parameters({time: dt})
            _circuits = _circuits + [_fcircuit,] + list(reversed(_circuits))
            for _c in _circuits:
                circuit = circuit.compose(_c)
    return circuit