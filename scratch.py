from __future__ import annotations
from conventions import VERTEX_SINGLET_BITMAPS, IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8
from lattice_tools import LatticeRegisters, Plaquette, LatticeVector, LinkUnitVectorLabel
from math import ceil, pi
from givens import givens
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
        rotation_ctrl_bitstring: str,
        rotation_angle: float) -> QuantumCircuit:
    if len(rotation_ctrl_idxes) != len(rotation_ctrl_bitstring):
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
    # Create lattice with encodings.
    lattice = LatticeRegisters(
        dim=1.5,
        size=3,
        link_truncation_dict=IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8,
        vertex_singlet_dict=VERTEX_SINGLET_BITMAPS["d=3/2, T2"]
    )

    # Log resulting lattice data.
    print(f"Created dim {lattice.dim} lattice with vertices:\n{lattice.vertex_register_keys}.")
    print(f"It has {lattice.n_qubits_per_link} qubits per link and {lattice.n_qubits_per_vertex}.")
    print("It knows about the following encodings:")
    for irrep, encoding in lattice.link_truncation_bitmap.items():
        print(f"Link irrep encoding: {irrep} -> {encoding}")
    for vertex_bag, encoding in lattice.vertex_singlet_bitmap.items():
        print(f"Vertex singlet bag encoding: {vertex_bag} -> {encoding}")

    # Loop over plaquettes in the lattice to construct one trotter step
    # TODO clean up code below for this.
    # TODO decide whether some of this functionality should be moved to LatticeRegisters or another class.
    plaquette_local_circuit = QuantumCircuit()

    # Assemble all lattice registers into a blank circuit
    master_circuit = QuantumCircuit(
        *[lattice.get_link_register(link_address[0], link_address[1]) for link_address in lattice.link_register_keys],
        *[lattice.get_vertex_register(vertex_address) for vertex_address in lattice.vertex_register_keys]
    )

    # TODO consider interleaving application of electric and magnetic Hamiltonian evolution?

    # Loop over links for electric Hamiltonain
    for link_key in lattice.link_register_keys:
        # TODO Implement
        pass

    # Loop over plaquettes for magnetic Hamiltonian.
    # TODO swap this out for real matrix elements. Format:
    # (state bitstring1, state bitstring2, rotation angle)
    import random
    random.seed(0)
    n_qubits_per_plaquette = 4 * (lattice.n_qubits_per_link + lattice.n_qubits_per_vertex)
    n_matrix_elements = 2
    TEST_DUMMY_MAG_HAMILTONIAN_DATA = [
        (
            ''.join(random.choice('01') for _ in range(n_qubits_per_plaquette)),
            ''.join(random.choice('01') for _ in range(n_qubits_per_plaquette)),
            random.uniform(0, 2 * pi)
        )
        for _ in range(n_matrix_elements)
    ]
    for vertex_address in lattice.vertex_register_keys:
        has_no_vertical_periodic_link_three_halves_case = lattice.dim == 1.5 and vertex_address[1] == 1
        if has_no_vertical_periodic_link_three_halves_case:
            continue

        current_vertex_reg = lattice.get_vertex_register(vertex_address)
        print(f"Fetching all positive plaquettes at {vertex_address}:", current_vertex_reg)
        has_only_one_positive_plaquette = lattice.dim == 1.5 or lattice.dim == 2
        if has_only_one_positive_plaquette:
            plaquettes: List[Plaquette] = [lattice.get_plaquette_registers(vertex_address, 1, 2)]
        else:
            plaquettes: List[Plaquette] = lattice.get_plaquette_registers(vertex_address)
        print(f"Found {len(plaquettes)} plaquette(s).")
        for plaquette in plaquettes:
            # TODO: delete, just for debug.
            for link in plaquette.link_registers:
                print(link)
            for vertex in plaquette.vertex_registers:
                print(vertex)

            # Collect the local qubits for later stitching.
            vertex_qubits = []
            link_qubits = []
            for register in plaquette.vertex_registers:
                for qubit in register:
                    vertex_qubits.append(qubit)
            for register in plaquette.link_registers:
                for qubit in register:
                    link_qubits.append(qubit)
            #local_hamiltonian_evol = QuantumCircuit()
            # Append a givens rotation circuit for each matrix element
            for bitstring_1, bitstring_2, angle in TEST_DUMMY_MAG_HAMILTONIAN_DATA:
                # Get the current givens rotation circuit
                # TODO make it so these are precomputed one time rather than inside nested for loops.
                # Much more efficient to just do it once for all matrix elements, then create copies as needed.
                plaquette_local_rotation_circuit = givens(bitstring_1, bitstring_2, angle)
                #print(plaquette_local_rotation_circuit)

                # Stitch into master circuit.
                master_circuit = master_circuit.compose(plaquette_local_rotation_circuit, qubits=[
                    *vertex_qubits,
                    *link_qubits
                ])

    master_circuit.draw(output="mpl", filename="quantum_circuit.pdf", fold=False)
