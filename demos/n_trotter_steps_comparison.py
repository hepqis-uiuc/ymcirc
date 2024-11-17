"""
This script creates "diet QSP" simulation circuits.

Various lattice sizes and dimensionalities are supported
by working with the LatticeRegisters class in order to
handle addressing QuantumRegisters for lattice degrees
of freedom.

Currently a work in progress.
"""
from __future__ import annotations

# Hacky way to make lattice_tools imports work
import sys
import os
# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from pathlib import Path
from lattice_tools.conventions import (
    VERTEX_SINGLET_BITMAPS,
    IRREP_TRUNCATION_DICT_1_3_3BAR,
    IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8)
from lattice_tools.circuit import LatticeCircuitManager
from lattice_tools.lattice_registers import LatticeRegisters
from lattice_tools.conventions import MAGNETIC_HAMILTONIANS, LatticeStateEncoder
from math import comb
from qiskit import transpile
from qiskit_aer.primitives import SamplerV2
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit.qasm2 import dumps
from qiskit import qpy


# Filesystem stuff
PROJECT_ROOT = Path(__file__).parent.parent
SERIALIZED_CIRCUITS_DIR = PROJECT_ROOT / "serialized-circuits"
SERIALIZED_CIRCUITS_DIR.mkdir(exist_ok=True)
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
SIM_RESULTS_DIR = PROJECT_ROOT / "sim-results"
SIM_RESULTS_DIR.mkdir(exist_ok=True)


# Configure simulation parameters and data.
do_electric_evolution = False
do_magnetic_evolution = True
#dimensionality_and_truncation_string = "d=2, T1"
dimensionality_and_truncation_string = "d=3/2, T1"
trunc_string = dimensionality_and_truncation_string[-2:]
dimensions = 1.5
linear_size = 2  # To indirectly control the number of plaquettes
coupling_g = 1.0
mag_hamiltonian_matrix_element_threshold = 0.6 # Drop all matrix elements that have an abs value less than this.
run_circuit_optimization = False
n_trotter_steps_cases = [1, 2, 3] # Make this a list that iterates from 1 to 3
sim_times = np.linspace(0.05, 3.0, num=20) # set num to 20 for comparison with trailhead


if __name__ == "__main__":
    # Configure DataFrame for working with simulation result data.
    sim_index = pd.MultiIndex.from_product([n_trotter_steps_cases, sim_times], names=["num_trotter_steps", "time"])
    df_job_results = pd.DataFrame(columns = ["vacuum_persistence_probability"], index=sim_index)

    # Set the right vertex and link bitmaps based on
    # dimensionality_and_truncation_string.
    # OK to not use vertex DOFs for d=3/2, T1.
    vertex_bitmap = {} if dimensionality_and_truncation_string == "d=3/2, T1" else VERTEX_SINGLET_BITMAPS[dimensionality_and_truncation_string]  # Ok to not use vertex DoFs in this case.
    link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR if dimensionality_and_truncation_string[-2:] == "T1" else IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8

    # Create an encoder for converting between physical states and bit strings.
    lattice_encoder = LatticeStateEncoder(link_bitmap=link_bitmap, vertex_bitmap=vertex_bitmap)

    # Use the encoder to index Hamiltonian data in terms of bit string encodings of plaquettes.
    # This will be used to determine rotation angles in the simulation circuit.
    # Because of the givens rotation sim strategy, we can let
    # H_mag = \box + \box^\dagger = 2\box, which is why we multiply by 2.
    mag_hamiltonian: List[Tuple[str, str, float]] = []
    for (final_plaquette_state, initial_plaquette_state), matrix_element_value in MAGNETIC_HAMILTONIANS[dimensionality_and_truncation_string].items():
        if abs(matrix_element_value) < mag_hamiltonian_matrix_element_threshold:
            continue
        final_state_bitstring = lattice_encoder.encode_plaquette_state_as_bit_string(final_plaquette_state)
        initial_state_bitstring = lattice_encoder.encode_plaquette_state_as_bit_string(initial_plaquette_state)
        if ('1' in final_state_bitstring) and ('1' in initial_state_bitstring):
            continue
        mag_hamiltonian.append((final_state_bitstring, initial_state_bitstring, 2*matrix_element_value))

    print(mag_hamiltonian)
    print("Num matrix elements:", len(mag_hamiltonian))
    # TODO generate all parameterized givens rotation circuits here?

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

            # # Assemble all lattice registers into a blank circuit.
            circ_mgr = LatticeCircuitManager(lattice_encoder, mag_hamiltonian)
            master_circuit = circ_mgr.create_blank_full_lattice_circuit(lattice)

            # Compute the rotation angle per trotter step
            # Append a single Trotter step over the lattice.
            # Put this inside a for loop for multiple Trotter steps?
            for _ in range(n_trotter_steps):
                if do_electric_evolution is True:
                    circ_mgr.apply_electric_trotter_step(master_circuit, lattice)
                if do_magnetic_evolution is True:
                    circ_mgr.apply_magnetic_trotter_step(
                        master_circuit,
                        lattice,
                        coupling_g=coupling_g,
                        dt=dt,
                        optimize_circuits=run_circuit_optimization
                    )

            # Uncomment for a final attempt at optimization.
            master_circuit.measure_all()
            master_circuit = transpile(master_circuit, optimization_level=3)
            print("Gate counts:\n", master_circuit.count_ops())

            # TODO Compute this elsewhere, not efficient.
            # Needed to format file paths.
            if lattice.dim >= 2:
                n_plaquettes = int(len(lattice.vertex_register_keys) * comb(lattice.dim, 2))
            else:
                n_plaquettes = int(len(lattice.vertex_register_keys) / 2)

            #Uncomment to write circuits to disk in either QASM or QPY form.
            # Dump qasm of circuit to file.
            #qasm_file_path =  SERIALIZED_CIRCUITS_DIR / Path(f"qasm-{n_plaquettes}-plaquettes-in-d={dimensions}-irrep_trunc={trunc_string}-mat_elem_cut={mag_hamiltonian_matrix_element_threshold}/n_trotter={n_trotter_steps}-t={sim_time}.qasm")
            # qasm_file_path.parent.mkdir(parents=True, exist_ok=True)
            # with qasm_file_path.open('w') as qasm_file:
            #     qasm_file.write(dumps(master_circuit))
            #continue
            
            #Dump QPY serialization of circuit to file.
            # qpy_file_path =  SERIALIZED_CIRCUITS_DIR / Path(f"qpy-{n_plaquettes}-plaquettes-in-d={dimensions}-irrep_trunc={trunc_string}-mat_elem_cut={mag_hamiltonian_matrix_element_threshold}/n_trotter={n_trotter_steps}-t={sim_time}.qpy")
            # qpy_file_path.parent.mkdir(parents=True, exist_ok=True)
            # with qpy_file_path.open('wb') as qpy_file:
            #     qpy.dump(master_circuit, qpy_file)
            #continue

            # Sim config stuff, maybe change?
            sim = AerSimulator()
            sampler = SamplerV2()
            n_shots = 1024

            # Uncomment to save circuit diagram.
            # master_circuit.draw(output="mpl", filename="out.pdf", fold=False)

            print("Running simulation...")
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

            print("Updated df:\n", df_job_results)

    print("All simulations complete. Final results:")
    print(df_job_results)

    print("Plotting...")
    fig, ax = plt.subplots()
    for n_steps in n_trotter_steps_cases:
        extracted_data = df_job_results.xs(n_steps, level = 'num_trotter_steps')
        extracted_data.plot(y = "vacuum_persistence_probability", label = f"$N_T = {n_steps}$", ax=ax)
    plt.title(f'Vacuum persistence probability ({n_plaquettes} plaquettes, mat. trunc = {mag_hamiltonian_matrix_element_threshold})')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()
    fig.savefig(PLOTS_DIR / Path(f"{n_plaquettes}-plaquettes-in-d={dimensions}-irrep_trunc={trunc_string}-mat_elem_cut={mag_hamiltonian_matrix_element_threshold}.pdf"))

    #Uncomment to save all job counts to disk.
    print("Saving data to disk...")
    df_job_results.to_csv(SIM_RESULTS_DIR / Path(f"{n_plaquettes}-plaquettes-in-d={dimensions}-irrep_trunc={trunc_string}-mat_elem_cut={mag_hamiltonian_matrix_element_threshold}.csv"))
    print("Done. Goodbye!")
