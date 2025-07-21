"""
This script creates lattice time-evolution circuits.

Various lattice sizes and dimensionalities are supported
by working with the LatticeRegisters class in order to
handle addressing QuantumRegisters for lattice degrees
of freedom.

Currently a work in progress. Current file simulates vacuum persistence probability 
and electric energy.
"""
from __future__ import annotations

# Hacky way to make ymcirc imports work
import sys
import os
# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from pathlib import Path
from ymcirc._abstract import LatticeDef
from ymcirc.conventions import (
    load_magnetic_hamiltonian,
    PHYSICAL_PLAQUETTE_STATES,
    IRREP_TRUNCATION_DICT_1_3_3BAR,
    IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8)
from ymcirc.circuit import LatticeCircuitManager
from ymcirc.lattice_registers import LatticeRegisters
from ymcirc.conventions import LatticeStateEncoder
from ymcirc.parsed_lattice_result import ParsedLatticeResult
from qiskit import transpile
from qiskit_aer.primitives import SamplerV2
from typing import Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ymcirc.electric_helper import electric_hamiltonian
from ymcirc.electric_helper import convert_bitstring_to_evalue

from qiskit.qasm2 import dumps


# Filesystem stuff
PROJECT_ROOT = Path(__file__).parent.parent
SERIALIZED_CIRCUITS_DIR = PROJECT_ROOT / "serialized-circuits"
SERIALIZED_CIRCUITS_DIR.mkdir(exist_ok=True)
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
SIM_RESULTS_DIR = PROJECT_ROOT / "sim-results"
SIM_RESULTS_DIR.mkdir(exist_ok=True)


# Configure simulation parameters and data.
do_electric_evolution = True
do_magnetic_evolution = True
dimensionality_and_truncation_string = "d=3/2, T1"
dim_string, trunc_string = dimensionality_and_truncation_string.split(",")
dim_string = dim_string.strip()
trunc_string = trunc_string.strip()
dimensions = 1.5 if (dim_string == "d=3/2" or dim_string == "d=1.5") else int(dim_string[2:])
linear_size = 2  # To indirectly control the number of plaquettes
coupling_g = 1.0
mag_hamiltonian_matrix_element_threshold = 0.9 # Drop all matrix elements that have an abs value less than this.
run_circuit_optimization = False
n_trotter_steps_cases = [2] # Make this a list that iterates from 1 to 3
sim_times = np.linspace(0.05, 2.5, num=20) # set num to 20 for comparison with trailhead
#sim_times = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
only_include_elems_connected_to_electric_vacuum = False
use_2box_hack = False  # Halves circuit depth by taking box + box^dagger = 2box. Only true if all nonzero matrix elements have the same magnitude.
warn_on_unphysical_links = True  # Emit warning when decoding unphysical plaquette states. Assign 0.0 electric energy.
error_on_unphysical_links = False  # Raise an error when decoding unphysical unphysical states. Terminate simulation.
prune_controls = True
control_fusion = True
electric_gray_order = True
use_ancillas = True # Toggles whether to use ancillas or not 
n_shots = 10000

# Specify plotting options if desired, and whether to save plots/circuits/data to disk
plot_vacuum_persistence = True
plot_electric_energy = True
save_circuits_qasm = False
save_circuits_diagrams = False
save_plots = False
save_data = False


if __name__ == "__main__":
    # Configure DataFrame for working with simulation result data.
    sim_index = pd.MultiIndex.from_product([n_trotter_steps_cases, sim_times], names=["num_trotter_steps", "time"])
    df_job_results = pd.DataFrame(columns = ["vacuum_persistence_probability", "electric_energy"], index=sim_index)

    # Set the right link bitmap based on
    # dimensionality_and_truncation_string.
    if trunc_string in ["T1", "T1p"]:
        link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR
    elif trunc_string in ["T2"]:
        link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8
    else:
        raise ValueError(f"Unknown irrep truncation: '{trunc_string}'.")

    # Define the lattice geometry, and create an encoder for mapping between physical states and bit strings.
    lattice_def = LatticeDef(dimensions=dimensions, size=linear_size, periodic_boundary_conds=True)
    lattice_encoder = LatticeStateEncoder(
        link_bitmap=link_bitmap,
        physical_plaquette_states=PHYSICAL_PLAQUETTE_STATES[dimensionality_and_truncation_string],
        lattice=lattice_def)

    # Use the encoder to index Hamiltonian data in terms of bit string encodings of plaquettes.
    # This will be used to determine rotation angles in the simulation circuit.
    # Because of the givens rotation sim strategy, we can let
    # H_mag = \box + \box^\dagger = 2\box, which is why we multiply by 2.
    # To see how this works, go to the documentation in ymcirc.conventions.
    mag_hamiltonian = load_magnetic_hamiltonian(
        dimensionality_and_truncation_string,
        lattice_encoder,
        mag_hamiltonian_matrix_element_threshold=mag_hamiltonian_matrix_element_threshold,
        only_include_elems_connected_to_electric_vacuum=only_include_elems_connected_to_electric_vacuum,
        use_2box_hack=use_2box_hack
    )
    print("Num matrix elements:", len(mag_hamiltonian))
    # We need the set of physical plaquette states to do control pruning.
    # If we aren't doing control pruning, we set this variable to a flag value of None.
    if prune_controls is True:
        physical_plaquette_states: Set[str] = set(lattice_encoder.encode_plaquette_state_as_bit_string(plaquette) for plaquette in PHYSICAL_PLAQUETTE_STATES[dimensionality_and_truncation_string])
        print("Performing control pruning. Num phys plaquette states:", len(physical_plaquette_states))
    else:
        physical_plaquette_states = None
        print("Skipping control pruning.")

    # TODO generate all parameterized givens rotation circuits here?

    # Iterate over cases of trotter steps and sim times.
    for sim_time in sim_times:
        for n_trotter_steps in n_trotter_steps_cases:
            dt = sim_time / n_trotter_steps
            print(
                f"Simulating evolution to t = {sim_time} in {n_trotter_steps} Trotter steps "
                f"with dt = {dt}."
            )

            # TODO use from_lattice_encoder method for this.
            # Create lattice, do sanity checks, and log some info.
            lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
            current_vacuum_state = "0" * lattice_registers.n_total_qubits
            print(f"Created dim {lattice_registers.dim} lattice with vertices:\n{lattice_registers.vertex_addresses}.")
            print(f"It has {lattice_registers.n_qubits_per_link} qubits per link and {lattice_registers.n_qubits_per_vertex} per vertex.")
            print("It knows about the following encodings:")
            for irrep, encoding in lattice_registers.link_bitmap.items():
                print(f"Link irrep encoding: {irrep} -> {encoding}")
            for multiplicity_index, encoding in lattice_registers.vertex_bitmap.items():
                print(f"Multiplicity index encoding: {multiplicity_index} -> {encoding}")
            print(f"It has the vacuum state: {current_vacuum_state}")

            # Needed to format file paths.
            n_plaquettes = lattice_registers.n_plaquettes

            # # Assemble all lattice registers into a blank circuit.
            circ_mgr = LatticeCircuitManager(lattice_encoder, mag_hamiltonian)
            master_circuit = circ_mgr.create_blank_full_lattice_circuit(lattice_registers)

            # Adds an ancilla register to use in MCX v-chain decomposition if use_ancillas is True
            if (use_ancillas):
                print("Using ancillas. Running a single trotter step to finded the minimum required number of ancillas")
                circ_mgr.add_ancillas_register_to_lattice_registers(master_circuit, lattice_registers, control_fusion=control_fusion, 
                physical_states_for_control_pruning=physical_plaquette_states,
                optimize_circuits=run_circuit_optimization)
                size_of_ancilla_register = circ_mgr.number_of_ancillas_used_in_circuit()
                print(f"Ancilla register of size {size_of_ancilla_register}.\n")
            else:
                print("Not using ancillas\n.")

            # Compute the rotation angle per trotter step
            # Append a single Trotter step over the lattice.
            # Put this inside a for loop for multiple Trotter steps?
            for _ in range(n_trotter_steps):
                if do_magnetic_evolution is True:
                    circ_mgr.apply_magnetic_trotter_step(
                        master_circuit,
                        lattice_registers,
                        coupling_g=coupling_g,
                        dt=dt,
                        optimize_circuits=run_circuit_optimization,
                        physical_states_for_control_pruning=physical_plaquette_states,
                        control_fusion=control_fusion,
                        cache_mag_evol_circuit=True
                    )

                if do_electric_evolution is True:
                    circ_mgr.apply_electric_trotter_step(master_circuit, lattice_registers, electric_hamiltonian(link_bitmap), coupling_g=coupling_g,
                        dt=dt, electric_gray_order=electric_gray_order)

            # Now optionally save circuit diagram/qasm, and log gate counts.
            if save_circuits_diagrams is True:
                master_circuit.draw(
                    output="mpl",
                    filename=f"{n_plaquettes}-plaquettes-in-d={dimensions}-irrep_trunc={trunc_string}-mat_elem_cut={mag_hamiltonian_matrix_element_threshold}-vac_connected_only={only_include_elems_connected_to_electric_vacuum}-n_trotter={n_trotter_steps}-t={sim_time}.pdf", fold=False)

            master_circuit.measure_all()
            print("Transpiling circuit...")
            master_circuit = transpile(master_circuit, optimization_level=3)
            print("Gate counts:\n", master_circuit.count_ops())

            breakpoint()

            if save_circuits_qasm is True:
                qasm_file_path = SERIALIZED_CIRCUITS_DIR / Path(f"qasm-{n_plaquettes}-plaquettes-in-d={dimensions}-irrep_trunc={trunc_string}-mat_elem_cut={mag_hamiltonian_matrix_element_threshold}-vac_connected_only={only_include_elems_connected_to_electric_vacuum}/n_trotter={n_trotter_steps}-t={sim_time}.qasm")
                qasm_file_path.parent.mkdir(parents=True, exist_ok=True)
                with qasm_file_path.open('w') as qasm_file:
                    qasm_file.write(dumps(master_circuit))

            # Set up simulation for current circuit.
            sampler = SamplerV2()

            print("Running simulation...")
            job = sampler.run([master_circuit], shots = n_shots)
            job_result = job.result()
            # Ancilla register added at the end. This means to strip out those
            # qubits for final state, we discard the part of the measurement
            # string after index lattice_registers.n_total_qubits.
            counts_dict_big_endian = {little_endian_state[::-1][:lattice_registers.n_total_qubits]: count for little_endian_state, count in job_result[0].data.meas.get_counts().items()}
            print("Finished.")

            # Aggregate data.
            current_sim_idx = (n_trotter_steps, sim_time)
            print(f"Setting data for {current_sim_idx}.")
            for big_endian_state, counts in counts_dict_big_endian.items():
                df_job_results.loc[current_sim_idx, big_endian_state] = counts
            # Make sure vacuum state data exists.
            if current_vacuum_state not in counts_dict_big_endian.keys():
                df_job_results.loc[current_sim_idx, current_vacuum_state] = 0
                df_job_results.loc[current_sim_idx, "vacuum_persistence_probability"] = 0
                df_job_results.loc[current_sim_idx, "electric_energy"] = 0
            else:
                df_job_results.loc[current_sim_idx, "vacuum_persistence_probability"] = df_job_results.loc[current_sim_idx, current_vacuum_state] / n_shots
                avg_electric_energy = 0
                for state, counts in counts_dict_big_endian.items():
                    print("Encoded state:", state)
                    print("Decoded state:")
                    parsed_state = ParsedLatticeResult(dimensions, linear_size, state, lattice_encoder)
                    for (vertex_address, attached_link_addresses) in parsed_state.get_traversal_order():
                        print(f"v {vertex_address} == {parsed_state.get_vertex(vertex_address)}")
                        for link_address in attached_link_addresses:
                            print(f"l {link_address} == {parsed_state.get_link(link_address)}")
                    avg_electric_energy += convert_bitstring_to_evalue(state, lattice_encoder, warn_on_unphysical_links, error_on_unphysical_links) * (counts / n_shots) / lattice_registers.n_links
                df_job_results.loc[current_sim_idx, "electric_energy"] = avg_electric_energy

            print("Updated df:\n", df_job_results)

    print("All simulations complete. Final results:")
    print(df_job_results)

    # Plot simulation results.
    if plot_vacuum_persistence is True:
        print("Plotting vacuum persistence amplitude...")
        VPP_PLOT_PATH_TO_SAVE = f"{n_plaquettes}-plaquettes-in-d={dimensions}-irrep_trunc={trunc_string}-mat_elem_cut={mag_hamiltonian_matrix_element_threshold}-vac_connected_only={only_include_elems_connected_to_electric_vacuum}_vpp.pdf"
        fig_vpp, ax_vpp = plt.subplots()
        title = f'$\\left|\\left<vac.|U(t)|vac.\\right>\\right|^2$ ({n_plaquettes} plaquettes, mat. trunc = {mag_hamiltonian_matrix_element_threshold})'
        for n_steps in n_trotter_steps_cases:
            extracted_data = df_job_results.xs(n_steps, level='num_trotter_steps')
            extracted_data.plot(y="vacuum_persistence_probability", label=f"$N_T = {n_steps}$", ax=ax_vpp)
        plt.title(title)
        plt.xlabel('Time')
        plt.xticks(rotation=45)
        plt.grid()
        plt.tight_layout()
        plt.show()
        if save_plots is True:
            fig_vpp.savefig(PLOTS_DIR / Path(VPP_PLOT_PATH_TO_SAVE))
    if plot_electric_energy is True:
        EE_PLOT_PATH_TO_SAVE = f"{n_plaquettes}-plaquettes-in-d={dimensions}-irrep_trunc={trunc_string}-mat_elem_cut={mag_hamiltonian_matrix_element_threshold}-vac_connected_only={only_include_elems_connected_to_electric_vacuum}_ee.pdf"
        fig_ee, ax_ee = plt.subplots()
        title = f'Electric energy $\\left|E\\right|^2$ ({n_plaquettes} plaquettes, mat. trunc = {mag_hamiltonian_matrix_element_threshold})'
        for n_steps in n_trotter_steps_cases:
            extracted_data = df_job_results.xs(n_steps, level='num_trotter_steps')
            extracted_data.plot(y="electric_energy", label=f"$N_T = {n_steps}$", ax=ax_ee)
        plt.title(title)
        plt.xlabel('Time')
        plt.xticks(rotation=45)
        plt.grid()
        plt.tight_layout()
        plt.show()
        if save_plots is True:
            fig_ee.savefig(PLOTS_DIR / Path(EE_PLOT_PATH_TO_SAVE))

    # Save data to disk.
    if save_data is True:
        print("Saving data to disk...")
        df_job_results.to_csv(SIM_RESULTS_DIR / Path(f"{n_plaquettes}-plaquettes-in-d={dimensions}-irrep_trunc={trunc_string}-mat_elem_cut={mag_hamiltonian_matrix_element_threshold}-vac_connected_only={only_include_elems_connected_to_electric_vacuum}.csv"))
        print("Done. Goodbye!")
