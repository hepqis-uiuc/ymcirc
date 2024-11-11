"""
This script creates "diet QSP" simulation circuits.

Various lattice sizes and dimensionalities are supported
by working with the LatticeRegisters class in order to
handle addressing QuantumRegisters for lattice degrees
of freedom.

Currently a work in progress.
"""
from __future__ import annotations
from pathlib import Path
from lattice_tools.conventions import (
    VERTEX_SINGLET_BITMAPS,
    IRREP_TRUNCATION_DICT_1_3_3BAR,
    IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8)
from lattice_tools.lattice_registers import LatticeRegisters, Plaquette
from lattice_tools.conventions import MAGNETIC_HAMILTONIANS, LatticeStateEncoder
from lattice_tools.givens import givens
from math import floor
from qiskit import transpile
from qiskit_aer.primitives import SamplerV2
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit
from typing import List, Tuple
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt


from qiskit.qasm2 import dumps
from qiskit import qpy


# Various helpers and conveniences

# A list of tuples: (state bitstring1, state bitstring2, matrix element)
HamiltonianData = List[Tuple[str, str, float]]


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
    #dimensionality_and_truncation_string = "d=2, T1"
    dimensionality_and_truncation_string = "d=3/2, T1"
    trunc_string = dimensionality_and_truncation_string[-2:]
    #dimensions = 2
    dimensions = 1.5
    linear_size = 2
    coupling_g = 1.0
    run_circuit_optimization = False
    n_trotter_steps_cases = [1, 2, 3] # Make this a list that iterates from 1 to 3
    sim_times = np.linspace(0, 3.0, num=4) # set num to 20 for comparison with trailhead

    # TODO replace these with dataframes
    # For collecting results
    sim_index = pd.MultiIndex.from_product([n_trotter_steps_cases, sim_times], names=["num_trotter_steps", "time"])
    df_job_results = pd.DataFrame(columns = ["vacuum_persistence_probability"], index=sim_index)
    #vacuum_persistence_probabilities: Dict[Tuple[float, float], float] = {}
    #data_from_jobs: Dict[Tuple[float, float], float] =  {}

    # Set the right vertex and link bitmaps based on dimensionality_and_truncation_string
    vertex_bitmap = {} if dimensionality_and_truncation_string == "d=3/2, T1" else VERTEX_SINGLET_BITMAPS[dimensionality_and_truncation_string]  # Ok to not use vertex DoFs in this case.
    link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR if dimensionality_and_truncation_string[-2:] == "T1" else IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8

    # Create an encoder for converting between physical states and bit strings.
    lattice_encoder = LatticeStateEncoder(link_bitmap=link_bitmap, vertex_bitmap=vertex_bitmap)

    # Iterate over cases of trotter steps and sim times.
    for sim_time in sim_times:
        for n_trotter_steps in n_trotter_steps_cases:
            dt = sim_time / n_trotter_steps
            print(
                f"Simulating evolution to t = {sim_time} in {n_trotter_steps} Trotter steps "
                f"with dt = {dt}."
            )

            # Create lattice, do sanity checks, and log some info.
            lattice = LatticeRegisters(
                dim=dimensions,
                size=linear_size,
                link_truncation_dict=link_bitmap,
                vertex_singlet_dict=vertex_bitmap
            )
            n_qubits_in_lattice = (lattice.n_qubits_per_vertex * len(lattice.vertex_register_keys)) \
            + (lattice.n_qubits_per_link * len(lattice.link_register_keys))
            current_vacuum_state = "0" * n_qubits_in_lattice
            assert lattice.link_truncation_bitmap == lattice_encoder.link_bitmap
            assert lattice.vertex_singlet_bitmap == lattice_encoder.vertex_bitmap
            print(f"Created dim {lattice.dim} lattice with vertices:\n{lattice.vertex_register_keys}.")
            print(f"It has {lattice.n_qubits_per_link} qubits per link and {lattice.n_qubits_per_vertex} per vertex.")
            print("It knows about the following encodings:")
            for irrep, encoding in lattice.link_truncation_bitmap.items():
                print(f"Link irrep encoding: {irrep} -> {encoding}")
            for vertex_bag, encoding in lattice.vertex_singlet_bitmap.items():
                print(f"Vertex singlet bag encoding: {vertex_bag} -> {encoding}")
            print(f"It has the vacuum state: {current_vacuum_state}")

            # Assemble all lattice registers into a blank circuit
            master_circuit = QuantumCircuit(
                *[lattice.get_link_register(link_address[0], link_address[1]) for link_address in lattice.link_register_keys],
                *[lattice.get_vertex_register(vertex_address) for vertex_address in lattice.vertex_register_keys]
            )

            # Use the encoder to index Hamiltonian data in terms of bit string encodings of plaquettes.
            # This will be used to determine rotation angles in the simulation circuit.
            mag_hamiltonian: List[Tuple[str, str, float]] = []
            for (final_plaquette_state, initial_plaquette_state), matrix_element_value in MAGNETIC_HAMILTONIANS[dimensionality_and_truncation_string].items():
                final_state_bitstring = lattice_encoder.encode_plaquette_state_as_bit_string(final_plaquette_state)
                initial_state_bitstring = lattice_encoder.encode_plaquette_state_as_bit_string(initial_plaquette_state)
                mag_hamiltonian.append((final_state_bitstring, initial_state_bitstring, matrix_element_value))
            # print("Mag Hamiltonian matrix elements:", len(mag_hamiltonian))
            # print(sorted([abs(val) for f, i, val in mag_hamiltonian]))
            #assert False

            # Compute the rotation angle per trotter step
            # Append a single Trotter step over the lattice.
            # Put this inside a for loop for multiple Trotter steps?
            for _ in range(n_trotter_steps):
                if do_electric_evolution is True:
                    apply_electric_trotter_step(master_circuit, lattice)
                if do_magnetic_evolution is True:
                    apply_magnetic_trotter_step(
                        master_circuit,
                        lattice,
                        hamiltonian=mag_hamiltonian,
                        coupling_g=coupling_g,
                        dt=dt,
                        optimize_circuits=run_circuit_optimization
                    )

            # Uncomment for a final attempt at optimization.
            master_circuit.measure_all()
            master_circuit = transpile(master_circuit, optimization_level=3)
            print(f"Finished creating a(n) {n_qubits_in_lattice}-qubit master circuit. Gate count:\n", dict(master_circuit.count_ops()))

            # Dump qasm of circuit to file.
            # qasm_file_path =  Path(__file__).parent / Path(f"qasm-{len(lattice.vertex_register_keys)}-plaquettes-in-d={dimensions}-trunc={trunc_string}/n_trotter={n_trotter_steps}-t={sim_time}.qasm")
            # qasm_file_path.parent.mkdir(parents=True, exist_ok=True)
            # with qasm_file_path.open('w') as qasm_file:
            #     qasm_file.write(dumps(master_circuit))
            # continue
            # Dump QPY serialization of circuit to file.
            qpy_file_path =  Path(__file__).parent / Path(f"qpy-2-plaquettes-in-d={dimensions}-trunc={trunc_string}/n_trotter={n_trotter_steps}-t={sim_time}.qpy")
            qpy_file_path.parent.mkdir(parents=True, exist_ok=True)
            with qpy_file_path.open('wb') as qpy_file:
                qpy.dump(master_circuit, qpy_file)
            continue
            assert False

            # Sim config stuff, maybe change?
            sim = AerSimulator()
            sampler = SamplerV2()
            n_shots = 1024

            # Uncomment to save circuit diagram.
            # master_circuit.draw(output="mpl", filename="out.pdf", fold=False)
            
            print("Running simulation...")
            # Run the circuit.
            job = sampler.run([master_circuit], shots = n_shots)
            job_result = job.result()
            print("Finished.")


            # Aggregate data.
            current_sim_idx = (n_trotter_steps, sim_time)
            print(f"Setting data for {current_sim_idx}.")
            for state, counts in job_result[0].data.meas.get_counts().items():
                df_job_results.loc[current_sim_idx, state] = counts
            # Make sure vacuum state data exists.
            if current_vacuum_state not in job_result[0].data.meas.get_counts().keys():
                df_job_results.loc[current_sim_idx, current_vacuum_state] = 0
                df_job_results.loc[current_sim_idx, "vacuum_persistence_probability"] = 0
            else:
                df_job_results.loc[current_sim_idx, "vacuum_persistence_probability"] = df_job_results.loc[current_sim_idx, current_vacuum_state] / n_shots


            print(df_job_results)


print("All simulations complete. Final results:")
print(df_job_results)
print("Plotting...")
for i in range(1, 4):
    extracted_data = df_job_results.xs(1, level = 'num_trotter_steps')
    extracted_data.plot(y = "vacuum_persistence_probability", label = f"$N_T = {i}$", ax=plt.gca())
plt.title('Vacuum persistence probability')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()
print("Saving data to disk...")
assert False
# Save results to disk
result_file_path = Path(__file__).parent / Path("out.json")
with result_file_path.open('w') as json_file:
    #json.dump(vacuum_persistence_amplitude, json_file)
    json.dump(data_from_jobs, json_file)

print("Done. Goodbye!")
