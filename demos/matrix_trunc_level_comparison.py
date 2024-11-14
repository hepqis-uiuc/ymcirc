"""
This demo compares performing different numbers of trotter steps.
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
from lattice_tools.circuit import apply_electric_trotter_step, apply_magnetic_trotter_step
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


# Filesystem stuff
PROJECT_ROOT = Path(__file__).parent.parent
SERIALIZED_CIRCUITS_DIR = PROJECT_ROOT / "serialized-circuits"
PLOTS_DIR = PROJECT_ROOT / "plots"
SIM_RESULTS_DIR = PROJECT_ROOT / "sim-results"


# Configure simulation parameters and data.
do_electric_evolution = False
do_magnetic_evolution = True
dimensionality_and_truncation_string = "d=2, T1"
#dimensionality_and_truncation_string = "d=3/2, T1"
trunc_string = dimensionality_and_truncation_string[-2:]
dimensions = 2
linear_size = 2  # To indirectly control the number of plaquettes
coupling_g = 1.0
run_circuit_optimization = False
n_trotter_steps = 2
matrix_elem_truc_levels = [0.95]#[0.0, 0.2, 0.4, 0.6]
sim_times = np.linspace(0.05, 4.0, num=1)


if __name__ == "__main__":
    # Configure DataFrame for working with simulation result data.
    sim_index = pd.MultiIndex.from_product([matrix_elem_truc_levels, sim_times], names=["matrix_element_trunc", "time"])
    df_job_results = pd.DataFrame(columns = ["vacuum_persistence_probability"], index=sim_index)

    # Set the right vertex and link bitmaps based on
    # dimensionality_and_truncation_string.
    # OK to not use vertex DOFs for d=3/2, T1.
    vertex_bitmap = {} if dimensionality_and_truncation_string == "d=3/2, T1" \
        else VERTEX_SINGLET_BITMAPS[dimensionality_and_truncation_string]
    link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR if dimensionality_and_truncation_string[-2:] == "T1" \
        else IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8

    # Create an encoder for converting between physical states and bit strings.
    lattice_encoder = LatticeStateEncoder(link_bitmap=link_bitmap, vertex_bitmap=vertex_bitmap)

    # Or, Iterate over a fixed trotter step with different matrix trucations.
    # Comment out as desired.
    for trunc_level in matrix_elem_truc_levels:
        print(f"Truncating the Hamiltonian at level {trunc_level}.")
        mag_hamiltonian: List[Tuple[str, str, float]] = []
        for (final_plaquette_state, initial_plaquette_state), matrix_element_value in MAGNETIC_HAMILTONIANS[dimensionality_and_truncation_string].items():
            if abs(matrix_element_value) < trunc_level:
                continue
            final_state_bitstring = lattice_encoder.encode_plaquette_state_as_bit_string(final_plaquette_state)
            initial_state_bitstring = lattice_encoder.encode_plaquette_state_as_bit_string(initial_plaquette_state)
            mag_hamiltonian.append((final_state_bitstring, initial_state_bitstring, 2*matrix_element_value))

        #print("Using the Hamiltonian:", mag_hamiltonian)
        print("Num matrix elements:", len(mag_hamiltonian))
        for sim_time in sim_times:
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
            print("Gate counts:\n", master_circuit.count_ops())
            continue

            #Uncomment to write circuits to disk in either QASM or QPY form.
            if lattice.dim >= 2:
                n_plaquettes = int(len(lattice.vertex_register_keys) * comb(lattice.dim, 2))
            else:
                n_plaquettes = int(len(lattice.vertex_register_keys) / 2)

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
            current_sim_idx = (trunc_level, sim_time)
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

    assert False
    print("All simulations complete. Final results:")
    print(df_job_results)

    print("Plotting...")
    fig, ax = plt.subplots()
    for trunc in matrix_elem_truc_levels:
        extracted_data = df_job_results.xs(trunc, level = 'matrix_element_trunc')
        extracted_data.plot(y = "vacuum_persistence_probability", label = f"{trunc}", ax=ax)
    plt.title(f'Vacuum persistence probability ({n_plaquettes} plaquettes)')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()
    fig.savefig(PLOTS_DIR / Path(f"{n_plaquettes}-plaquettes-in-d={dimensions}-irrep_trunc={trunc_string}-n_trotter={n_trotter_steps}.pdf"))

    # Uncomment to save all job counts to disk.
    print("Saving data to disk...")
    df_job_results.to_csv(SIM_RESULTS_DIR / Path(f"{n_plaquettes}-plaquettes-in-d={dimensions}-irrep_trunc={trunc_string}-n_trotter={n_trotter_steps}.csv"))
    print("Done. Goodbye!")
