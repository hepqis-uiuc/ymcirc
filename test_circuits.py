import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import CXGate, RXGate, XGate
from qiskit.quantum_info import Operator, Pauli, process_fidelity

# Verify Cianan's exp(-i alpha chi_13) circuit.
m = 2
alpha = np.pi*np.random.random()
print(f"{alpha = }")
# Matrix form of operator.
op = np.eye(2**m, dtype=complex)
R = Operator(RXGate(alpha))
op[:2,:2] = R
op = Operator(op)
#print(op)

# Circuit form of operator.
circ = QuantumCircuit(2, 2)
Rhalf = Operator(RXGate(alpha/2))
circ.append(Rhalf, [0,])
circ.cy(1, 0)
circ.append(Rhalf, [0,])
circ.cy(1, 0)

print(Operator(circ) == op)

