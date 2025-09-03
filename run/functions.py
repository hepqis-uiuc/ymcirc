"""This script demonstrates one way of building a simulation pipeline with ymcirc."""
from __future__ import annotations

# Hacky way to make ymcirc imports work.
import sys
import os
# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import copy
import logging
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from pathlib import Path
from qiskit import transpile, qpy
from qiskit_aer.primitives import SamplerV2
from qiskit.circuit import QuantumCircuit
from qiskit.qasm3 import dumps
from typing import Any, Set
from ymcirc._abstract import LatticeDef
from ymcirc.circuit import LatticeCircuitManager
from ymcirc.conventions import (
    IRREP_TRUNCATIONS, LatticeStateEncoder, load_magnetic_hamiltonian, PHYSICAL_PLAQUETTE_STATES)
from ymcirc.electric_helper import convert_bitstring_to_evalue, electric_hamiltonian
from ymcirc.lattice_registers import LatticeRegisters


def configure_script_options(
        dimensionality_string: str,
        truncation_string: str,
        lattice_size: int,
        sim_times: np.ndarray | list,
        n_trotter_steps: int,
        n_shots: int | None,
        use_periodic_boundary_conds: bool = True,
        serialized_circ_dir: Path | str | None = None,
        plots_dir: Path | str | None = None,
        sim_results_dir: Path | str | None = None,
        do_electric_evolution: bool = True,
        do_magnetic_evolution: bool = True,
        coupling_g: float | int = 1.0,
        mag_hamiltonian_matrix_element_threshold: float | int = 0,
        optimize_circuits: bool = False,
        mag_hamiltonian_use_electric_vacuum_transitions_only: bool = False,
        warn_unphysical_links: bool = True,
        error_unphysical_links: bool = False,
        prune_controls: bool = True,
        control_fusion: bool = True,
        electric_gray_order: bool = True,
        cache_mag_evol_circuit: bool = True,
        use_ancillas: bool = True,
        plot_vacuum_persistence: bool = True,
        plot_electric_energy: bool = True,
        save_circuit_to_qasm: bool = False,
        save_circuit_to_qpy: bool = False,
        save_circuit_diagrams: bool = False,
        save_plots: bool = False,
        save_sim_data: bool = False,
        load_circuit_from_file: str | Path | None = None
) -> dict[str, Any]:
    """
    Configure a dictionary of options
    that control the script behavior.

    Inputs:
        - dimensionality_string:
              A string of the form
              "d=[dimesionality]" where "dimensionality"
              can be 3/2, 2, 3, ...

              Currently supported configurations: d=3/2, T1 or T2; d=2, T1.
              See ymcirc.conventions for more info.
        - truncation_string:
              A string specifying what link irrep truncation to use.

              Current supported configurations: 'T1' and 'T2', when using d=3/2,
              'T1' when using d=2.

              See ymcirc.conventions for more info.
        - lattice_size:
              For 2 or more dimensions, the number of vertices in
              each direction for a (hypercubic) lattice. For d=3/2, how many
              vertices long the chain is.
        - sim_times:
              A list of durations to generate simulation circuits for.
              n_trotter_steps will be used for each one.
        - n_trotter_steps:
              How many Trotter steps to use in simulation
              circuits.
        - n_shots:
              How many shots to use if simulating circuit execution. If None,
              no circuit simulations will be run.
        - use_periodic_boundary_conds:
              Use periodic boundary conditions if True.
        - serialized_circ_dir:
              Directory for read/write of circuit QASM/QPY files.
        - plots_dir:
              Directory for read/write of plots.
        - sim_results_dir:
              Directory for read/write of circuit execution data.
        - do_electric_evolution:
              Whether to include electric Hamiltonian in
              simulation circuits.
        - do_magnetic_evolution:
              Whether to include magnetic Hamiltonian in
              simulation circuits.
        - coupling_g: The value of the coupling constant in the Hamiltonian.
        - mag_hamiltonian_matrix_element_threshold:
              A number between zero and
              one. All magnetic Hamiltonian matrix elements less than the
              threshold will be skipped during circuit construction.
        - optimize_circuits:
              Whether to run qiskit's circuit optimizer.
        - mag_hamiltonian_use_electric_vacuum_transitions_only:
              If True, all
              magnetic Hamiltonian matrix elements which don't induce transitions
              to or from the electric vacuum are skipped during circuit
              construction.
        - warn_unphysical_links:
              Whether to emit a warning on unphysical link
              data when analyzing circuit execution results.
        - error_unphysical_links:
              Whether to raise an error on unphysical link
              data when analyzing circuit execution results.
        - prune_controls:
              Whether to perform the control pruning algorithm to
              reduce circuit depth.
        - control_fusion:
              Whether to perform the control fusion algorithm to
              reduce circuit depth.
        - electric_gray_order:
              Whether to use Gray ordering of gates to reduce
              circuit depth
        - cache_mag_evol_circuit:
              Whether the plaquette-local magnetic evolution
              circuit should be cached. Speeds up application
              across lattices and between Trotter steps.
              Usually a good idea to enable.
        - use_ancillas:
              Whether to introduce ancilla qubits to reduce to gate
              depth of multi-control gates.
        - plot_vacuum_persistence:
              Whether to plot all available vacuum
              persistence amplitude data.
        - plot_electric_energy:
              Whether to plot all available electric energy
              data.
        - save_circuit_to_qasm:
              Whether to save generated parameterized circuit to disk
              as QASM file.
        - save_circuit_to_qpy:
              Whether to save generated parameterized circuit to disk
              as QPY file.
        - save_circuit_diagrams:
              Whether to save diagrams of all generated
              circuits to disk as PDFs.
        - save_plots:
              Whether to save all plots generated to disk as PDFs instead of shown immediately.
        - save_sim_data:
              Whether to save all circuit execution data to disk.
        - load_circuit_from_file:
              If a str or Path instance, a parameterized circuit file (either QASM or QPY).
              Skips creating a circuit if not None.
    """
    options: dict[str, Any] = dict()
    options["serialized_circ_dir"] = serialized_circ_dir
    options["plots_dir"] = plots_dir
    options["sim_results_dir"] = sim_results_dir
    options["do_electric_evolution"] = do_electric_evolution
    options["do_magnetic_evolution"] = do_magnetic_evolution
    options["dimensionality_string"] = dimensionality_string
    options["truncation_string"] = truncation_string
    options["lattice_size"] = lattice_size
    options["coupling_g"] = coupling_g
    options["mag_hamiltonian_matrix_element_threshold"] = mag_hamiltonian_matrix_element_threshold
    options["optimize_circuits"] = optimize_circuits
    options["n_trotter_steps"] = n_trotter_steps
    options["sim_times"] = sim_times
    options["mag_hamiltonian_use_electric_vacuum_transitions_only"] = mag_hamiltonian_use_electric_vacuum_transitions_only
    options["warn_unphysical_links"] = warn_unphysical_links
    options["error_unphysical_links"] = error_unphysical_links
    options["prune_controls"] = prune_controls
    options["control_fusion"] = control_fusion
    options["electric_gray_order"] = electric_gray_order
    options["cache_mag_evol_circuit"] = cache_mag_evol_circuit
    options["use_ancillas"] = use_ancillas
    options["n_shots"] = n_shots
    options["plot_vacuum_persistence"] = plot_vacuum_persistence
    options["plot_electric_energy"] = plot_electric_energy
    options["save_circuit_to_qasm"] = save_circuit_to_qasm
    options["save_circuit_to_qpy"] = save_circuit_to_qpy
    options["save_circuit_diagrams"] = save_circuit_diagrams
    options["save_plots"] = save_plots
    options["save_sim_data"] = save_sim_data
    options["use_periodic_boundary_conds"] = use_periodic_boundary_conds
    options["load_circuit_from_file"] = load_circuit_from_file

    # Automatically set some additional options based on user input above.
    options["dimensions"] = 1.5 if (dimensionality_string == "d=3/2" or dimensionality_string == "d=1.5") else int(dimensionality_string[2:])
    options["link_bitmap"] = IRREP_TRUNCATIONS[options["truncation_string"]]
    options["lattice_def"] = LatticeDef(
        dimensions=options["dimensions"],
        size=options["lattice_size"],
        periodic_boundary_conds=options["use_periodic_boundary_conds"])

    return options


def create_time_evol_circuit(script_options: dict[str, Any]) -> QuantumCircuit:
    """
    Create a parameterized QuantumCircuit instance for simulating the lattice
    described by script_options.
    """
    # Create lattice_encoder instance.
    lattice_encoder = create_lattice_encoder(script_options)

    # Load mag Hamiltonian data.
    print("Loading magnetic Hamiltonian data...")
    mag_hamiltonian = load_magnetic_hamiltonian(
        script_options["dimensionality_string"],
        script_options["truncation_string"],
        lattice_encoder,
        mag_hamiltonian_matrix_element_threshold=script_options["mag_hamiltonian_matrix_element_threshold"],
        only_include_elems_connected_to_electric_vacuum=script_options["mag_hamiltonian_use_electric_vacuum_transitions_only"])
    print("Done.")

    # Figure out physical states needed for control pruning.
    if script_options["prune_controls"] is True:
        physical_plaquette_states: Set[str] = set(
            lattice_encoder.encode_plaquette_state_as_bit_string(plaquette) for plaquette in PHYSICAL_PLAQUETTE_STATES[script_options["dimensionality_string"]][script_options["truncation_string"]])
    else:
        physical_plaquette_states = None

    # Initialize the current circuit's registers.
    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    circ_mgr = LatticeCircuitManager(lattice_encoder, mag_hamiltonian)
    master_circuit = circ_mgr.create_blank_full_lattice_circuit(lattice_registers)

    # Add ancilla register if needed.
    if script_options["use_ancillas"] is True:
        circ_mgr.num_ancillas = circ_mgr.compute_num_ancillas_needed_from_mag_trotter_step(
            master_circuit, lattice_registers, control_fusion=script_options["control_fusion"],
            physical_states_for_control_pruning=physical_plaquette_states,
            optimize_circuits=script_options["optimize_circuits"])
        circ_mgr.add_ancilla_register_to_quantum_circuit(master_circuit)

    # Apply Trotter steps.
    for idx in range(script_options["n_trotter_steps"]):
        if script_options["do_magnetic_evolution"] is True:
            print(f"Applying magnetic Trotter step {idx + 1}/{script_options['n_trotter_steps']} across lattice...")
            circ_mgr.apply_magnetic_trotter_step(
                master_circuit,
                lattice_registers,
                optimize_circuits=script_options["optimize_circuits"],
                physical_states_for_control_pruning=physical_plaquette_states,
                control_fusion=script_options["control_fusion"],
                cache_mag_evol_circuit=script_options["cache_mag_evol_circuit"]
            )

        if script_options["do_electric_evolution"] is True:
            print(f"Applying electric Trotter step {idx + 1}/{script_options['n_trotter_steps']} across lattice...")
            circ_mgr.apply_electric_trotter_step(
                master_circuit,
                lattice_registers,
                electric_hamiltonian(lattice_encoder.link_bitmap),
                electric_gray_order=script_options["electric_gray_order"])

    return master_circuit


def save_circuit(circuit: QuantumCircuit, simulation_identifier: str, script_options: dict[str, Any]) -> None:
    """
    Write a QuantumCircuit instance to
    QASM/QPY files and/or save circuit diagrams to pdfs
    depending on script_options.

    Each QASM/QPY file is saved in the
    directory specified by script_options['serialized_circ_dir'].
    """
    print("Saving circuit to disk...")
    # Prep the circuit write directory.
    circuits_dir = script_options['serialized_circ_dir']
    if not isinstance(circuits_dir, Path):
        circuits_dir = Path(circuits_dir)
    circuits_dir.mkdir(exist_ok=True)

    # Prep tbe circuit diagram write directory.
    circuit_diagram_dir = script_options['plots_dir']
    if not isinstance(circuit_diagram_dir, Path):
        circuit_diagram_dir = Path(circuit_diagram_dir)
    circuit_diagram_dir.mkdir(exist_ok=True)

    # Write QASM/QPY file and/or PDF of circuit diagram.
    if script_options["save_circuit_to_qasm"] is True:
        qasm_circuit_filename = simulation_identifier + ".qasm"
        qasm_file_path = circuits_dir / qasm_circuit_filename
        with qasm_file_path.open('w') as qasm_file:
            qasm_file.write(dumps(circuit))
    if script_options["save_circuit_to_qpy"] is True:
        qpy_circuit_filename = simulation_identifier + ".qpy"
        qpy_file_path = circuits_dir / qpy_circuit_filename
        with open(qpy_file_path, "wb") as qpy_file:
            qpy.dump(circuit, qpy_file)
    if script_options["save_circuit_diagrams"] is True:
        diagram_filename = simulation_identifier + ".pdf"
        diagram_file_path = circuit_diagram_dir / diagram_filename
        circuit.draw(
            output="mpl",
            filename=diagram_file_path,
            fold=False
        )


def run_circuit_simulations(circuit: QuantumCircuit, script_options: dict[str, Any]) -> pd.DataFrame:
    """
    Execute a simulation of circuit according
    to the parameters in script_options. There will be
    as many simulations as there are items in sim_times.

    Returns the results as a DataFrame.
    """
    # Set up objects needed for executing circuits and processing results.
    sampler = SamplerV2()
    n_ancilla_qubits = len(circuit.ancillas)
    n_total_qubits = len(circuit.qubits)
    n_data_qubits = n_total_qubits - n_ancilla_qubits
    vacuum_state = "0" * n_data_qubits
    lattice_encoder = create_lattice_encoder(script_options)
    print(f"# data qubits: {n_data_qubits}")
    print(f"# ancilla qubits: {n_ancilla_qubits}")

    # Prepare a circuit for each sim_time by adding a
    # measurement of all registers, assigning parameter values,
    # and then transpiling.
    # In principle, we could set all the dt and coupling_g parameters
    # for each electric or magnetic Trotter step individually,
    # but in this case, we use the same dt for both at each total sim duration,
    # and use one value of the coupling g for all simulations.
    transpiled_circuits_with_assigned_params = []
    for idx, sim_time in enumerate(script_options["sim_times"]):
        print(f"Setting parameters for circuit {idx+1}/{len(script_options['sim_times'])} (sim_time = {sim_time})")
        dt = sim_time / script_options["n_trotter_steps"]
        transpiled_circuit_with_final_measurement = copy.deepcopy(circuit)
        parameter_values = dict()
        for param in transpiled_circuit_with_final_measurement.parameters:
            if 'dt' in param.name:
                parameter_values[param.name] = dt
            elif 'coupling_g' in param.name:
                parameter_values[param.name] = script_options['coupling_g']
        transpiled_circuit_with_final_measurement.assign_parameters(parameter_values, inplace=True)
        transpiled_circuit_with_final_measurement.measure_all()
        transpiled_circuit_with_final_measurement = transpile(transpiled_circuit_with_final_measurement, optimization_level=3)
        transpiled_circuits_with_assigned_params.append(transpiled_circuit_with_final_measurement)

    # Execute circuits.
    print("Executing circuits...")
    job = sampler.run(transpiled_circuits_with_assigned_params, shots=script_options['n_shots'])
    job_results = job.result()

    # Organize job results into dataframe.
    df_job_results = pd.DataFrame(columns=["vacuum_persistence_probability", "electric_energy"], index=script_options["sim_times"])
    for job_result, sim_time in zip(job_results, script_options["sim_times"]):
        # Strip out ancilla bits, and reverse to big-endian convention.
        counts_dict_big_endian = {little_endian_state[::-1][:n_data_qubits]: count for little_endian_state, count in job_result.data.meas.get_counts().items()}
        for big_endian_state, counts in counts_dict_big_endian.items():
            df_job_results.loc[sim_time, big_endian_state] = counts

        # Ensure existence of vacuum state data.
        if vacuum_state not in counts_dict_big_endian.keys():
            df_job_results.loc[sim_time, vacuum_state] = 0

        # Compute vacuum persistence probability and average electric energy per link.
        df_job_results.loc[sim_time, "vacuum_persistence_probability"] = df_job_results.loc[sim_time, vacuum_state] / script_options["n_shots"]
        avg_electric_energy = 0
        for state, counts in counts_dict_big_endian.items():
            avg_electric_energy += convert_bitstring_to_evalue(state, lattice_encoder, script_options["warn_unphysical_links"], script_options["error_unphysical_links"]) * (counts / script_options["n_shots"]) / script_options["lattice_def"].n_links
        df_job_results.loc[sim_time, "electric_energy"] = avg_electric_energy

    return df_job_results


def save_circuit_sim_data(data: pd.DataFrame, simulation_identifier: str, script_options: dict[str, Any]) -> None:
    """
    Save circuit execution data to disk.

    Data is written as CSV files to the directory spcified by
    script_options['sim_results_dir']. The string simulation_identifier
    is combined with the timesteps in script_options to assign each
    data file a name.
    """
    # Prep the circuit write directory.
    sim_results_dir = script_options['sim_results_dir']
    if not isinstance(sim_results_dir, Path):
        sim_results_dir = Path(sim_results_dir)
    sim_results_dir.mkdir(exist_ok=True)

    # Write data to disk.
    data.to_csv(sim_results_dir / (simulation_identifier + ".csv"))


def plot_data(data: pd.DataFrame, col: str, title: str, script_options: dict[str, Any]) -> Figure:
    """
    Create a matplotlib figure plotting whatever "col" is.
    """
    print(f"Plotting {col}...")
    fig, ax = plt.subplots()
    data.plot(y=col, ax=ax)
    plt.title(title)
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    print("Done.")

    return fig


def create_lattice_encoder(script_options: dict[str, Any]) -> None:
    lattice_encoder = LatticeStateEncoder(
        link_bitmap=script_options["link_bitmap"],
        physical_plaquette_states=PHYSICAL_PLAQUETTE_STATES[script_options["dimensionality_string"]][script_options["truncation_string"]],
        lattice=script_options["lattice_def"])

    return lattice_encoder


def config_ymcirc_logger(level) -> None:
    """
    Set using the logging module.

    Options for level (from most to least 'noisy'):
      - logging.DEBUG
      - logging.INFO
      - logging.WARNING (this is the default level)
      - logging.ERROR
      - logging.CRITICAL
    """
    # Set log level
    logger = logging.getLogger("ymcirc")
    logging.getLogger('ymcirc').setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(name)s:%(levelname)s:%(message)s'))
    logger.addHandler(handler)
