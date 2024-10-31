from __future__ import annotations
from conventions import VERTEX_SINGLET_DICT_D_3HALVES_1_3_3BAR_8
from lattice_tools import LatticeRegisters, Plaquette, LatticeVector, LinkUnitVectorLabel
from math import ceil
from itertools import product
from qiskit.circuit import QuantumCircuit, QuantumRegister
from typing import List, Tuple, Dict
import numpy as np


# Placeholder subcircuit for testing braiding.
def make_plaquette_circuit(
        plaquette: Plaquette,
        primed_qubit_idx: int,
        prefix_target_qubit_idxes: list[int],
        rotation_ctrl_idxes: list[int],
        rotation_ctrl_state: str,
        rotation_angle: float) -> QuantumCircuit:
    if len(rotation_ctrl_idxes) != len(rotation_ctrl_state):
        raise ValueError()
    # TODO replace this dummy circuit with actual simulation circuit
    n_qubits = sum([len(register) for register in plaquette.vertex_registers]) + \
        sum([len(register) for register in plaquette.link_registers])
    qc = QuantumCircuit(n_qubits)
    for target_idx in prefix_target_qubit_idxes:
        qc.cx(primed_qubit_idx, target_idx)

    qc.cx(0, 4*len(vertex_registers[0])+1)

    return qc


if __name__ == "__main__":
    #print(VERTEX_SINGLET_DICT_D_3HALVES_133BAR8)
    for singlet in VERTEX_SINGLET_DICT_D_3HALVES_1_3_3BAR_8:
        print(f"Singlet: {singlet} -> {VERTEX_SINGLET_DICT_D_3HALVES_1_3_3BAR_8[singlet]}")
    lattice = LatticeRegisters(dim=2, size=3, n_qubits_per_vertex=2, n_qubits_per_link=2)
    print(f"Created dim {lattice.dim} lattice with vertices:\n{lattice.vertex_register_keys}.")

    #plaquette_local_circuit = QuantumCircuit()

    # master_circuit = QuantumCircuit(
    #     *[lattice.get_link_register(key[0], key[1]) for key in lattice.link_register_keys],
    #     *[lattice.get_vertex_register(key) for key in lattice.vertex_register_keys]
    # )
    # for vertex_address in lattice.vertex_register_keys:
    #     if lattice.dim == 1.5 and vertex_address[1] == 1:
    #         continue
    #     current_vertex_reg = lattice.get_vertex_register(vertex_address)
    #     print(f"Building plaquette at vertex {vertex_address}:", current_vertex_reg)
    #     plaquettes = [lattice.get_plaquette_registers(vertex_address, 1, 2)]#[get_plaquette(lattice, vertex_address, 1, 2)]
    #     for plaquette in plaquettes:
    #         # TODO: delete, just for debug.
    #         for link in plaquette.link_registers:
    #             print(link)
    #         for vertex in plaquette.vertex_registers:
    #             print(vertex)
    #         #local_hamiltonian_evol = QuantumCircuit()
    #         local_hamiltonian_evol = make_plaquette_circuit(plaquette.vertex_registers, plaquette.link_registers)
    #         print(local_hamiltonian_evol.draw())
    #         vertex_qubits = []
    #         link_qubits = []
    #         for register in plaquette.vertex_registers:
    #             for qubit in register:
    #                 vertex_qubits.append(qubit)

    #         for register in plaquette.link_registers:
    #             for qubit in register:
    #                 link_qubits.append(qubit)
            
    #         master_circuit = master_circuit.compose(local_hamiltonian_evol, qubits=[
    #             *vertex_qubits,
    #             *link_qubits
    #         ])

    # print(master_circuit.draw())


        # for link_dir in range(1, ceil(lattice.dim) + 1):
        #     if lattice.dim == 1.5 and vertex_address[1] == 1 and link_dir == 2:
        #         continue

        #     link_address = (vertex_address, link_dir)
        #     current_link_reg = lattice.get_link_register(vertex_address, link_dir)
        #     print(f"Link {link_address}:", current_link_reg)

