"""This module demonstrates constructing and simulating state-preparation circuits with ymcirc."""

# Hacky way to make ymcirc imports work.
import sys
import os
# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import logging
from run.functions import (
    config_ymcirc_logger, configure_script_options, initialize_lattice_tools, save_circuit,
    load_circuit
)
from pathlib import Path
from qiskit import transpile
from qiskit import qpy

if __name__ == "__main__":
    # Set project root directory. Change as appropriate.
    PROJECT_ROOT = Path(__file__).parent.parent

    # Set log level for ymcirc.
    config_ymcirc_logger(logging.INFO)

    script_options = configure_script_options(
        dimensionality_string="d=3/2",
        truncation_string="T1",
        lattice_size=2,
        givens_have_independent_params=True,
        sim_times=[],
        n_trotter_steps=1,  # TODO does this break if higher than 1?
        n_shots=None,
        use_ancillas=True,
        control_fusion=True,
        prune_controls=True,
        cache_mag_evol_circuit=True,
        mag_hamiltonian_matrix_element_threshold=0.0,
        load_circuit_from_file=None,#"state_prep=vqe-2-plaquettes-in-d=1.5-irrep_trunc=T1-mat_elem_cut=0.0-vac_connected_only=False-vchain=True-control_fusion=True-prune_controls=True.qpy",  # Replace with file path if desired.
        save_circuit_to_qpy=True,
        serialized_circ_dir=PROJECT_ROOT / "serialized-circuits"
    )
    script_options['state_prep_method'] = "vqe"  # When there are more methods available, they will be added to configure_script_options.

    # Generate a descriptive prefix for all filenames based on simulation params.
    filename_str_prefix = f"state_prep={script_options['state_prep_method']}-{script_options['lattice_def'].n_plaquettes}-plaquettes-in-d={script_options['lattice_def'].dim}-irrep_trunc={script_options['truncation_string']}-mat_elem_cut={script_options['mag_hamiltonian_matrix_element_threshold']}-vac_connected_only={script_options['mag_hamiltonian_use_electric_vacuum_transitions_only']}-vchain={script_options['use_ancillas']}-control_fusion={script_options['control_fusion']}-prune_controls={script_options['prune_controls']}"

    # Create or load state prep circuit.
    if script_options['load_circuit_from_file'] is None:
        print(f"Creating ground state preparation ciruit using method: {script_options['state_prep_method']}")
        lattice_encoder, physical_plaquette_states, lattice_registers, circ_mgr, state_prep_circuit = initialize_lattice_tools(script_options)
        if script_options['state_prep_method'] == "vqe":
            circ_mgr.apply_magnetic_trotter_step(
                state_prep_circuit,
                lattice_registers,
                optimize_circuits=script_options['optimize_circuits'],
                physical_states_for_control_pruning=physical_plaquette_states,
                control_fusion=script_options['control_fusion'],
                cache_mag_evol_circuit=script_options['cache_mag_evol_circuit'],
                givens_have_independent_params=script_options['givens_have_independent_params']
            )
            state_prep_circuit = transpile(state_prep_circuit, optimization_level=3)  # Optimize decompose "v-chain" gates into their constituents.
            breakpoint()
        else:
            raise NotImplementedError(f"State prep method {script_options['state_prep_method']} unknown.")

        # Optionally save state prep circuit to disk.
        if script_options['save_circuit_to_qpy'] is True:
            save_circuit(state_prep_circuit, filename_str_prefix, script_options)
    else:
        print(f"Skipping circuit creation, loading from disk.\nfile = {script_options['load_circuit_from_file']}")
        circuit_file = script_options['serialized_circ_dir'] / script_options['load_circuit_from_file']
        state_prep_circuit = load_circuit(circuit_load_path=circuit_file)

    print(f"Circuit ops count: {state_prep_circuit.count_ops()}.")
