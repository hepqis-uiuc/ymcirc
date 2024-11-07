"""
This script creates "diet QSP" simulation circuits.

Various lattice sizes and dimensionalities are supported
by working with the LatticeRegisters class in order to
handle addressing QuantumRegisters for lattice degrees
of freedom.

Currently a work in progress.
"""
from __future__ import annotations
from lattice_tools.conventions import (
    VERTEX_SINGLET_BITMAPS,
    IRREP_TRUNCATION_DICT_1_3_3BAR,
    IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8)
from lattice_tools.lattice_registers import LatticeRegisters, Plaquette
from lattice_tools.conventions import MAGNETIC_HAMILTONIANS, LatticeStateEncoder
from math import pi
from lattice_tools.givens import givens
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from typing import List, Tuple


# Various helpers and conveniences

# A list of tuples: (state bitstring1, state bitstring2, matrix element)
HamiltonianData = List[Tuple[str, str, float]]


def _helper_create_dummy_magnetic_hamiltonian(
        lattice: LatticeRegisters, n_matrix_elements: int) -> HamiltonianData:
    """
    Create test magnetic Hamiltonian data.

    This function should be deleted once actual data are available.

    Elements of the return list have the format:
    - (state bitstring1, state bitstring2, matrix element)
    """
    import random
    random.seed(0)
    n_qubits_per_plaquette = 4 * (lattice.n_qubits_per_link + lattice.n_qubits_per_vertex)
    return [
        (
            ''.join(random.choice('01') for _ in range(n_qubits_per_plaquette)),
            ''.join(random.choice('01') for _ in range(n_qubits_per_plaquette)),
            random.uniform(0, 2 * pi)
        )
        for _ in range(n_matrix_elements)
    ]


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
    # Bit picture: iterate over every lattice vertex, and add the trotter step
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
                angle = matrix_elem * (1 / (2 * (coupling_g**2))) * dt
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


if __name__ == "__main__":
    # Configure simulation parameters and data.
    do_electric_evolution = False
    do_magnetic_evolution = True
    dimensionality_and_truncation_string = "d=3/2, T1"
    dimensions = 1.5
    linear_size = 2
    # Set the right vertex and link bitmaps based on dimensionality_and_truncation_string
    vertex_bitmap = {} if dimensionality_and_truncation_string == "d=3/2, T1" else VERTEX_SINGLET_BITMAPS[dimensionality_and_truncation_string]  # Ok to not use vertex DoFs in this case.
    link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR if dimensionality_and_truncation_string[-2:] == "T1" else IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8
    lattice = LatticeRegisters(
        dim=dimensions,
        size=linear_size,
        link_truncation_dict=link_bitmap,
        vertex_singlet_dict=vertex_bitmap
    )

    # Log information about the lattice we created.
    print(f"Created dim {lattice.dim} lattice with vertices:\n{lattice.vertex_register_keys}.")
    print(f"It has {lattice.n_qubits_per_link} qubits per link and {lattice.n_qubits_per_vertex} per vertex.")
    print("It knows about the following encodings:")
    for irrep, encoding in lattice.link_truncation_bitmap.items():
        print(f"Link irrep encoding: {irrep} -> {encoding}")
    for vertex_bag, encoding in lattice.vertex_singlet_bitmap.items():
        print(f"Vertex singlet bag encoding: {vertex_bag} -> {encoding}")

    # Create an encoder for converting between physical states and bit strings.
    # Then we use the encoder to index Hamiltonian data in terms of bit string encodings of plaquettes.
    # This will be used to determine rotation angles in the simulation circuit.
    lattice_encoder = LatticeStateEncoder(link_bitmap=link_bitmap, vertex_bitmap=vertex_bitmap)
    assert lattice.link_truncation_bitmap == lattice_encoder.link_bitmap
    assert lattice.vertex_singlet_bitmap == lattice_encoder.vertex_bitmap
    mag_hamiltonian: List[Tuple[str, str, float]] = []
    for (final_plaquette_state, initial_plaquette_state), matrix_element_value in MAGNETIC_HAMILTONIANS[dimensionality_and_truncation_string].items():
        final_state_bitstring = lattice_encoder.encode_plaquette_state_as_bit_string(final_plaquette_state)
        initial_state_bitstring = lattice_encoder.encode_plaquette_state_as_bit_string(initial_plaquette_state)
        mag_hamiltonian.append((final_state_bitstring, initial_state_bitstring, matrix_element_value))

    # Assemble all lattice registers into a blank circuit
    master_circuit = QuantumCircuit(
        *[lattice.get_link_register(link_address[0], link_address[1]) for link_address in lattice.link_register_keys],
        *[lattice.get_vertex_register(vertex_address) for vertex_address in lattice.vertex_register_keys]
    )

    # Append a single Trotter step over the lattice.
    # Put this inside a for loop for multiple Trotter steps?
    if do_electric_evolution is True:
        apply_electric_trotter_step(master_circuit, lattice)
    if do_magnetic_evolution is True:
        apply_magnetic_trotter_step(
            master_circuit,
            lattice,
            #hamiltonian=TEST_DUMMY_MAG_HAMILTONIAN_LIST,
            hamiltonian=mag_hamiltonian,
            coupling_g=1.0,
            dt=1.0,
            optimize_circuits=False
        )

    # Uncomment for a final attempt at optimization.
    # master_circuit = transpile(master_circuit, optimization_level=3)

    # Uncomment to save circuit diagram.
    # master_circuit.draw(output="mpl", filename="out.pdf", fold=False)
    print("Gate count on master circuit:\n", dict(master_circuit.count_ops()))
