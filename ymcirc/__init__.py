"""
Quantum circuits, built with Qiskit, that simulate SU(3) lattice gauge theory.

ymcirc encodes the states of a lattice's links and vertices, at an irrep
truncation, in registers of qubits, and builds circuits on them: Trotter
steps of time evolution (electric and magnetic), and measurements of links,
vertices and plaquettes. Lattices are the plaquette chain (d=3/2), the square
lattice (d=2) and the cubic lattice (d=3). Each axis is periodic or open.
The physical plaquette states and the magnetic Hamiltonian's matrix elements
are computed by pyclebsch (https://github.com/hepqis-uiuc/pyclebsch), a
dependency.

A circuit with one Trotter step for a d=3/2 lattice of size 3 at the T1
truncation is built as follows:

from ymcirc._abstract import LatticeDef
from ymcirc.conventions import LatticeStateEncoder
from ymcirc.lattice_registers import LatticeRegisters
from ymcirc.circuit import LatticeCircuitManager
from ymcirc.electric_helper import electric_hamiltonian

lattice = LatticeDef(1.5, 3)
encoder = LatticeStateEncoder.for_truncation(lattice, "T", 1)
registers = LatticeRegisters.from_lattice_state_encoder(encoder)
circ_mgr = LatticeCircuitManager.for_truncation(encoder)
circuit = circ_mgr.create_blank_full_lattice_circuit(registers)
circ_mgr.apply_magnetic_trotter_step(circuit, registers)
circ_mgr.apply_electric_trotter_step(
    circuit, registers, electric_hamiltonian(encoder.link_bitmap))

The time step and the coupling are Parameters of the circuit. The module
docstring of ymcirc.conventions explains truncations and the encoding of
states as bit strings; ymcirc.parsed_lattice_result and
ymcirc.measurement_results turn measured bit strings back into lattice
states and observables. The scripts in the repository's run directory build
and simulate complete examples. A script that builds circuits keeps its work
under if __name__ == "__main__":, because pyclebsch's worker processes import
the main script.
"""
import logging

# Package-level logger
logger = logging.getLogger("ymcirc")
logger.debug("ymcirc logger initialized.")
logger.propagate = False  # To avoid duplication of log messages in root logger.
