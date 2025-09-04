"""This module demonstrates constructing and simulating state-preparation circuits with ymcirc."""

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
    config_ymcirc_logger, configure_script_options, initialize_lattice_tools,
    create_time_evol_circuit, save_circuit, run_circuit_simulations,
    save_circuit_sim_data, plot_data
)
from pathlib import Path
from qiskit import transpile, qpy
from qiskit.qasm3 import load

from ymcirc.circuit import LatticeCircuitManager
from ymcirc.conventions import (
    IRREP_TRUNCATIONS, LatticeStateEncoder, load_magnetic_hamiltonian, PHYSICAL_PLAQUETTE_STATES)

if __name__ == "__main__":
    # Set project root directory. Change as appropriate.
    PROJECT_ROOT = Path(__file__).parent.parent

    # Set log level for ymcirc.
    config_ymcirc_logger(logging.INFO)

    # TODO what script options to configure?
    script_options = configure_script_options(
        dimensionality_string="d=3/2",
        truncation_string="T1",
        lattice_size=2,
        sim_times=[],  # TODO too specific
        n_trotter_steps=1,  # TODO too specific
        n_shots=None,  # TODO too specific?
        use_ancillas=True,
        control_fusion=True,
        prune_controls=True,
        cache_mag_evol_circuit=True,
        mag_hamiltonian_matrix_element_threshold=0.0
    )

    # TODO descriptive prefix for all filenames based on sim params.

    # TODO create circuit where all givens rotations are parameterized. Put this in a function and/or circuit manager?
    lattice_encoder, physical_plaquette_states, lattice_registers, circ_mgr, master_circuit = initialize_lattice_tools(script_options)
    physical_plaquette_states_stripped = circ_mgr._strip_redundant_controls_if_small_and_periodic_lattice(physical_plaquette_states)
    circ_mgr.apply_magnetic_trotter_step(
        master_circuit,
        lattice_registers,
        optimize_circuits=script_options['optimize_circuits'],
        physical_states_for_control_pruning=physical_plaquette_states,
        control_fusion=script_options['control_fusion'],
        cache_mag_evol_circuit=script_options['cache_mag_evol_circuit'],
        givens_have_independent_params=True  # TODO this needs to be in script options.
    )
    master_circuit = transpile(master_circuit, optimization_level=3)  # Optimize decompose "v-chain" gates into their constituents.
