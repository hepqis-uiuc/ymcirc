# Praveen Balaji
# Nov 2024


from givens.py import givens
from lattice_tools.py import Plaquette, LatticeRegisters
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import YGate, XGate
from qiskit.circuit.library.standard_gates import RYGate, RXGate
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator


def trotter_step_magnetic(delta_t, plaquette)

def trotter_step_electric(delta_t, link)


def apply_trotter_step(delta_t, evol_type: str = 'both'):
        """
        Apply a single trotter evolution step to the entire lattice.

        evol_type == 'm' for just Magnetic Hamiltonian evolution.
        evol_type == 'e for just Electric Hamiltonian evolution.
        evol_type == anything else for both.
        """
        # TODO surface a step_size argument?
        if evol_type == "m":
            trotter_step_magnetic()
        elif evol_type == "e":
            trotter_step_electric()
        else:
            trotter_step_magnetic()
            trotter_step_electric()

