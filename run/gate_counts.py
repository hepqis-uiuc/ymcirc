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
import pandas as pd
import numpy as np
from run.functions import (
    config_ymcirc_logger, configure_script_options,
    create_time_evol_circuit, save_circuit, load_circuit,
    save_circuit_sim_data
)
from pathlib import Path
from qiskit import transpile

if __name__ == "__main__":
    # Set project root directory. Change as appropriate.
    PROJECT_ROOT = Path(__file__).parent.parent

    # Set log level for ymcirc.
    config_ymcirc_logger(logging.INFO)
    ancillas_cases = [False, True, True, True]
    fusion_cases =[False, False, True, True]
    pruning_cases = [False, False, False, True]
    df_gate_cost = pd.DataFrame(
        columns=["CX cost", "T cost", "Data qubits", "Ancilla qubits", "Total qubits"],
        #index=pd.MultiIndex.from_product([["d=3/2", "d=2",], [2, 3]], names=["dimensions", "lattice_size"])
        index=pd.MultiIndex.from_arrays([ancillas_cases, fusion_cases, pruning_cases], names=["use_ancillas", "control_fusion", "prune_controls"]))
    #for idx, (dimensionality_string, lattice_size) in enumerate(df_gate_cost.index):
    for idx, (use_ancillas, control_fusion, prune_controls) in enumerate(df_gate_cost.index):
        #print(f"Case {idx+1}/{len(df_gate_cost.index)}: {dimensionality_string}, {lattice_size}")
        print(f"Case {idx+1}/{len(df_gate_cost.index)}: {use_ancillas}, {control_fusion}, {prune_controls}")
        # Set simulation parameters here. See the docstring on
        # configure_script_options for an explanation of all
        # available options.
        # NOTE for QASM files: Read/write for currently broken due to parse
        # error in Qiskit's QASM serializer.
        # NOTE for QPY files: Read/write for circuits WITH ANCILLAS broken
        # due to parse error in Qiskit's QPY serializer (see https://github.com/Qiskit/qiskit/issues/11619).
        script_options = configure_script_options(
            dimensionality_string='d=2',#dimensionality_string,
            truncation_string="T1",
            lattice_size=2,#lattice_size,
            sim_times=np.linspace(0.0, 2.5, num=20),
            n_trotter_steps=1,
            n_shots=None,#10000,
            use_ancillas=use_ancillas,
            control_fusion=control_fusion,
            prune_controls=prune_controls,
            do_magnetic_evolution=True,
            method='statevector',  # matrix_product_state, statevector, etc. See Qiskit Aer docs.
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
            mag_hamiltonian_matrix_element_threshold=0.0
        )

        # Generate a descriptive prefix for all filenames based on simulation params.
        simulation_category_str_prefix = f"gate_costs-{script_options['lattice_def'].n_plaquettes}-plaquettes-in-d={script_options['lattice_def'].dim}-irrep_trunc={script_options['truncation_string']}-n_trotter_steps={script_options['n_trotter_steps']}-mat_elem_cut={script_options['mag_hamiltonian_matrix_element_threshold']}-vac_connected_only={script_options['mag_hamiltonian_use_electric_vacuum_transitions_only']}-vchain={script_options['use_ancillas']}-control_fusion={script_options['control_fusion']}-prune_controls={script_options['prune_controls']}"

        # Create or load circuit to simulate, optionally save to disk.
        if script_options["load_circuit_from_file"] is None:
            simulation_circuit = create_time_evol_circuit(script_options)
            simulation_circuit = transpile(simulation_circuit, optimization_level=3)
            # Save the circuit if desired.
            if ((script_options["save_circuit_to_qasm"] is True or
             script_options["save_circuit_to_qpy"] is True or
             script_options["save_circuit_diagrams"] is True)
            and
            (script_options["load_circuit_from_file"] is None)):
                save_circuit(simulation_circuit, simulation_category_str_prefix, script_options)
        else:
            print("Skipping circuit creation, loading from disk.")
            circuit_file = script_options['serialized_circ_dir'] / script_options['load_circuit_from_file']
            simulation_circuit = load_circuit(circuit_load_path=circuit_file)


        cx_count = simulation_circuit.count_ops()['cx'] if 'cx' in simulation_circuit.count_ops() else 0
        t_count = simulation_circuit.count_ops()['t'] if 't' in simulation_circuit.count_ops() else 0
        t_count += simulation_circuit.count_ops()['tdg'] if 'tdg' in simulation_circuit.count_ops() else 0
        a_count = len(simulation_circuit.ancillas)
        total_q_count = len(simulation_circuit.qubits)
        data_q_count = total_q_count - a_count

        # df_gate_cost.loc[(dimensionality_string, lattice_size), "CX cost"] = simulation_circuit.count_ops()['cx']
        # df_gate_cost.loc[(dimensionality_string, lattice_size), "T cost"] = simulation_circuit.count_ops()['t'] + simulation_circuit.count_ops()['tdg']
        # df_gate_cost.loc[(dimensionality_string, lattice_size), "Data qubits"] = len(simulation_circuit.qubits) - len(simulation_circuit.ancillas)
        # df_gate_cost.loc[(dimensionality_string, lattice_size), "Ancilla qubits"] = len(simulation_circuit.ancillas)
        # df_gate_cost.loc[(dimensionality_string, lattice_size), "Total qubits"] = len(simulation_circuit.qubits)


        df_gate_cost.loc[(use_ancillas, control_fusion, prune_controls), "CX cost"] = cx_count
        df_gate_cost.loc[(use_ancillas, control_fusion, prune_controls), "T cost"] = t_count
        df_gate_cost.loc[(use_ancillas, control_fusion, prune_controls), "Data qubits"] = data_q_count
        df_gate_cost.loc[(use_ancillas, control_fusion, prune_controls), "Ancilla qubits"] = a_count
        df_gate_cost.loc[(use_ancillas, control_fusion, prune_controls), "Total qubits"] = total_q_count

    print(df_gate_cost)
    df_gate_cost.to_csv(f"gate_count_comparison_different_methods.csv")
