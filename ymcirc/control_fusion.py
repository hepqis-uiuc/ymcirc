from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import RXGate
import math
import matplotlib.pyplot as plt

qc = QuantumCircuit(10)
mcrx = RXGate(math.pi/3).control(num_ctrl_qubits=6)
qc.append(mcrx,[2,3,8,5,6,7,4])
qc.cx(0,1)
qc.cx(2,3)
qc.cx(0,1)
optimized_qc = transpile(qc, optimization_level=1) 
gate = optimized_qc.data
print("cx"[-2:])
#optimized_qc.draw(output="mpl")
#plt.show()

