"""
A collection of utilities for building circuits.
"""
from __future__ import annotations
from lattice_tools.lattice_registers import LatticeRegisters, Plaquette
from lattice_tools.givens import givens
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from typing import List, Tuple


# A list of tuples: (state bitstring1, state bitstring2, matrix element)
HamiltonianData = List[Tuple[str, str, float]]


# TODO implement electric trotter step.
def apply_electric_trotter_step(
        master_circuit: QuantumCircuit,
        lattice: LatticeRegisters) -> None:
    """
    Placeholder for electric trotter step implementation.

    Should modify master_circuit in place rather than returning a new circuit because that's more efficient.
    """
    # Loop over links for electric Hamiltonain
    for link_key in lattice.link_register_keys:
        raise NotImplementedError()


# TODO Can we get the circuits in a parameterized way?
def apply_magnetic_trotter_step(
        master_circuit: QuantumCircuit,
        lattice: LatticeRegisters,
        hamiltonian: HamiltonianData,
        coupling_g: float = 1.0,
        dt: float = 1.0,
        optimize_circuits: bool = True) -> None:
    """
    Add one Trotter step.

    Note that this modifies master_circuit directly rather than returning a new circuit!

    Arguments:
      - lattice: a LatticeRegisters instance which keeps track of all the QuantumRegisters.
      - master_circuit: a QuantumCircuit instance which is built from all the
                        QuantumRegister instances in lattice.
      - hamiltonian: a dict whose keys are tuples of bitstrings corrresponding to
                     "plaquette final state" and "plaquette initial state", and whose
                     values correspond to numerical matrix element values.
      - coupling_g: the value of the strong coupling constant.
      - dt: the size of the Trotter time step.
      - optimize_circuits: if true, run the qiskit transpiler on each internal givens rotation
                           with the maximum optimization level before composing with master_circuit.

    Returns:
      A new QuantumCircuit instance which is master_circuit with the Trotter step appended.
    """
    # Big picture: iterate over every lattice vertex, and add the trotter step
    # to every "postive" plaquette at that vertex.
    for vertex_address in lattice.vertex_register_keys:
        # Skip creating "top vertex" plaquettes for d=3/2.
        has_no_vertical_periodic_link_three_halves_case = \
            lattice.dim == 1.5 and vertex_address[1] == 1
        if has_no_vertical_periodic_link_three_halves_case:
            continue

        # Get the plaquettes for the current vertex.
        current_vertex_reg = lattice.get_vertex_register(vertex_address)
        print(f"Fetching all positive plaquettes at {vertex_address}:", current_vertex_reg)
        has_only_one_positive_plaquette = lattice.dim == 1.5 or lattice.dim == 2
        if has_only_one_positive_plaquette:
            plaquettes: List[Plaquette] = [lattice.get_plaquette_registers(vertex_address, 1, 2)]
        else:
            plaquettes: List[Plaquette] = lattice.get_plaquette_registers(vertex_address)
        print(f"Found {len(plaquettes)} plaquette(s).")

        # For each plaquette, apply the the local Trotter step circuit.
        for plaquette in plaquettes:
            # Collect the local qubits for stitching purposes.
            vertex_qubits = []
            link_qubits = []
            for register in plaquette.vertex_registers:
                for qubit in register:
                    vertex_qubits.append(qubit)
            for register in plaquette.link_registers:
                for qubit in register:
                    link_qubits.append(qubit)

            # Append a Givens rotation circuit for each magnetic Hamiltonian
            # matrix element.
            for bitstring_1, bitstring_2, matrix_elem in hamiltonian:
                angle = -matrix_elem * (1 / (2 * (coupling_g**2))) * dt
                plaquette_local_rotation_circuit = givens(bitstring_1, bitstring_2, angle)
                if optimize_circuits is True:
                    plaquette_local_rotation_circuit = transpile(
                        plaquette_local_rotation_circuit, optimization_level=3)

                # Stitch the Givens rotation into master circuit.
                master_circuit.compose(
                    plaquette_local_rotation_circuit,
                    qubits=[
                        *vertex_qubits,
                        *link_qubits
                    ],
                    inplace=True
                )

