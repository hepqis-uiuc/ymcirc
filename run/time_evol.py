"""This module demonstrates one way of constructing and simulating time-evolution circuits with ymcirc."""
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
    create_time_evol_circuit, save_circuit, load_circuit, run_circuit_simulations,
    save_circuit_sim_data, plot_data
)
from pathlib import Path
from qiskit import transpile


if __name__ == "__main__":
    # Set project root directory. Change as appropriate.
    PROJECT_ROOT = Path(__file__).parent.parent

    # Set log level for ymcirc.
    config_ymcirc_logger(logging.INFO)

    # Set simulation parameters here. See the docstring on
    # configure_script_options for an explanation of all
    # available options.
    # NOTE for QPY files: Read/write for circuits WITH ANCILLAS broken
    # due to parse error in Qiskit's QPY serializer (see https://github.com/Qiskit/qiskit/issues/11619).
    script_options = configure_script_options(
        dimensionality_string="d=3",
        truncation_string="B3",
        lattice_size=2,
        sim_times=np.linspace(0.0, 2.5, num=20),
        n_trotter_steps=2,
        n_shots=10000,  # Set to None to skip circuit execution.
        use_ancillas=True,
        control_fusion=True,
        prune_controls=True,
        warn_unphysical_links=True,
        method='matrix_product_state',  # matrix_product_state, statevector, etc. See Qiskit Aer docs. For Aer version 0.17.2, AerSimulator has a hard cap of 63 qubits for MPS simulations. functions.py includes workaround hack for MPS simulations.
        matrix_product_state_max_bond_dimension=8, # Set to None if no limit desired. Set to a small integer value if MPS simulations take a long time. Ignored for non-MPS methods.
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
    simulation_category_str_prefix = f"{script_options['lattice_def'].n_plaquettes}-plaquettes-in-d={script_options['lattice_def'].dim}-irrep_trunc={script_options['truncation_string']}-n_trotter_steps={script_options['n_trotter_steps']}-mat_elem_cut={script_options['mag_hamiltonian_matrix_element_threshold']}-vac_connected_only={script_options['mag_hamiltonian_use_electric_vacuum_transitions_only']}-vchain={script_options['use_ancillas']}-control_fusion={script_options['control_fusion']}-prune_controls={script_options['prune_controls']}"

    # Create or load circuit to simulate, optionally save to disk.
    # Note that if we don't transpile the circuit,
    # some of the larger mcx gates don't get saved when
    # writing a QPY file.
    if script_options["load_circuit_from_file"] is None:
        simulation_circuit = create_time_evol_circuit(script_options)
        # Save the circuit if desired.
        if ((script_options["save_circuit_to_qasm"] is True or
         script_options["save_circuit_to_qpy"] is True or
         script_options["save_circuit_diagrams"] is True)
        and
        (script_options["load_circuit_from_file"] is None)):
            # We do NO optimization and specify the most generic basis gate set possible.
            # This maximizes the portability of the circuit when writing to disk.
            sim_circ_max_portability = transpile(simulation_circuit, basis_gates=["u","cx"], optimization_level=0)
            save_circuit(simulation_circuit, simulation_category_str_prefix, script_options)
        # Now that any circuit saves requested are done, let's optimize the circuit a bit.
        simulation_circuit = transpile(simulation_circuit, optimization_level=3)
    else:
        print("Skipping circuit creation, loading from disk.")
        circuit_file = script_options['serialized_circ_dir'] / script_options['load_circuit_from_file']
        simulation_circuit = load_circuit(circuit_load_path=circuit_file, script_options=script_options)

    print(f"Circuit ops count: {simulation_circuit.count_ops()}.")
    print(f"Depth: {simulation_circuit.depth()}")

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
