"""This script demonstrates one way of building a simulation pipeline with ymcirc."""
from __future__ import annotations

# Hacky way to make ymcirc imports work.
import sys
import os
# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import logging
import matplotlib.pyplot as plt
import numpy as np
from run.functions import (
    config_ymcirc_logger, configure_script_options,
    create_circuit, save_circuit, run_circuit_simulations,
    save_circuit_sim_data, plot_data
)
from pathlib import Path
from qiskit import transpile, qpy
from qiskit.qasm3 import load


if __name__ == "__main__":
    # Set project root directory. Change as appropriate.
    PROJECT_ROOT = Path(__file__).parent.parent

    # Set log level for ymcirc.
    config_ymcirc_logger(logging.INFO)

    # Set simulation parameters here. See the docstring on
    # configure_script_options for an explanation of all
    # available options.
    script_options = configure_script_options(
        dimensionality_string="d=3/2",
        truncation_string="T1",
        lattice_size=2,
        sim_times=np.linspace(0.0, 2.5, num=20),
        n_trotter_steps=2,
        n_shots=10000,
        use_ancillas=True,
        control_fusion=True,
        prune_controls=True,
        cache_mag_evol_circuit=True,
        load_circuit_from_file=None,  # Replace with file path if desired.
        save_circuit_to_qasm=False,
        save_circuit_to_qpy=False,
        save_circuit_diagrams=False,
        save_plots=False,
        save_sim_data=False,
        serialized_circ_dir=PROJECT_ROOT / "serialized-circuits",
        plots_dir=PROJECT_ROOT / "plots",
        sim_results_dir=PROJECT_ROOT / "sim-results",
        mag_hamiltonian_matrix_element_threshold=0.9
    )

    # Generate a descriptive prefix for all filenames based on simulation params.
    # Also specify whether to load a circuit from disk instead of creating it.
    simulation_category_str_prefix = f"{script_options['lattice_def'].n_plaquettes}-plaquettes-in-d={script_options['lattice_def'].dim}-irrep_trunc={script_options['truncation_string']}-n_trotter_steps={script_options['n_trotter_steps']}-mat_elem_cut={script_options['mag_hamiltonian_matrix_element_threshold']}-vac_connected_only={script_options['mag_hamiltonian_use_electric_vacuum_transitions_only']}-vchain={script_options['use_ancillas']}-control_fusion={script_options['control_fusion']}-prune_controls={script_options['prune_controls']}"

    # Create or load circuit to simulate, optionally save to disk.
    # Note that if we don't transpile the circuit,
    # some of the larger mcx gates don't get saved when
    # writing a QPY file.
    if script_options["load_circuit_from_file"] is None:
        simulation_circuit = create_circuit(script_options)
        simulation_circuit = transpile(simulation_circuit, optimization_level=3)
    else:
        print("Skipping circuit creation, loading from disk.")
        circuit_load_path = Path(script_options["load_circuit_from_file"])
        if circuit_load_path.exists() is False:
            raise FileExistsError(f"Tried to load nonexistent file: '{circuit_load_path}'")
        if circuit_load_path.suffix == ".qpy":
            with open(circuit_load_path, "rb") as handle:
                simulation_circuit = qpy.load(handle)[0]
        elif circuit_load_path.suffix == ".qasm":
            simulation_circuit = load(circuit_load_path)
        else:
            raise ValueError(f"Attempted to load circuit file of unknown type '{circuit_load_path.suffix}'.")
    if ((script_options["save_circuit_to_qasm"] is True or
         script_options["save_circuit_to_qpy"] is True or
         script_options["save_circuit_diagrams"] is True)
        and
        (script_options["load_circuit_from_file"] is None)):
        save_circuit(simulation_circuit, simulation_category_str_prefix, script_options)

    print(f"Circuit ops count: {simulation_circuit.count_ops()}.")

    # Either run circuits or skip.
    if script_options["n_shots"] is not None:
        sim_data = run_circuit_simulations(simulation_circuit, script_options)
    else:
        sim_data = None  # Neither simulating circuits nor loading sim data.

    # Save circuit execution data if available.
    if sim_data is not None and script_options["save_sim_data"] is True:
        save_circuit_sim_data(sim_data, simulation_category_str_prefix, script_options)

    # Optionally plot circuit execution data, and save plots to disk if desired.
    if script_options["plot_vacuum_persistence"] is True and sim_data is not None:
        title = f'$\\left|\\left<vac.|U(t)|vac.\\right>\\right|^2$ ({script_options["lattice_def"].n_plaquettes} plaquettes, mat. trunc = {script_options["mag_hamiltonian_matrix_element_threshold"]})'
        fig_vpp = plot_data(sim_data, "vacuum_persistence_probability", title, script_options)
        if script_options["save_plots"] is True:
            Path(script_options['plots_dir']).mkdir(exist_ok=True)
            vpp_plot_filename = simulation_category_str_prefix + "_vpp.pdf"
            fig_vpp.savefig(Path(script_options['plots_dir']) / vpp_plot_filename)
        else:
            plt.show()

    if script_options["plot_electric_energy"] is True and sim_data is not None:
        title = f'Electric energy $\\left|E\\right|^2$ ({script_options["lattice_def"].n_plaquettes} plaquettes, mat. trunc = {script_options["mag_hamiltonian_matrix_element_threshold"]})'
        fig_ee = plot_data(sim_data, "electric_energy", title, script_options)
        if script_options["save_plots"] is True:
            Path(script_options['plots_dir']).mkdir(exist_ok=True)
            ee_plot_filename = simulation_category_str_prefix + "_ee.pdf"
            fig_ee.savefig(Path(script_options['plots_dir']) / ee_plot_filename)
        else:
            plt.show()
