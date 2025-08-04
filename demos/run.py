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
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from pathlib import Path
from qiskit import transpile
from qiskit_aer.primitives import SamplerV2
from qiskit.circuit import QuantumCircuit
from qiskit.qasm2 import dumps
from typing import Any, Set
from ymcirc._abstract import LatticeDef
from ymcirc.circuit import LatticeCircuitManager
from ymcirc.conventions import (
    IRREP_TRUNCATION_DICT_1_3_3BAR, IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8,
    LatticeStateEncoder, load_magnetic_hamiltonian, PHYSICAL_PLAQUETTE_STATES)
from ymcirc.electric_helper import convert_bitstring_to_evalue, electric_hamiltonian
from ymcirc.lattice_registers import LatticeRegisters


def configure_script_options(
        dimensionality_and_truncation_string: str,
        lattice_size: int,
        sim_times: np.ndarray | list,
        n_trotter_steps: int,
        n_shots: int | None,
        use_periodic_boundary_conds: bool = True,
        circ_qasm_dir: Path | str | None = None,
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
        use_ancillas: bool = True,
        plot_vacuum_persistence: bool = True,
        plot_electric_energy: bool = True,
        save_circuits_to_qasm: bool = False,
        save_circuit_diagrams: bool = False,
        save_plots: bool = False,
        save_sim_data: bool = False
) -> dict[str, Any]:
    """
    Configure a dictionary of options
    that control the script behavior.

    Inputs:
        - dimensionality_and_truncation_string:
              A string of the form
              "d=[diensionality], [truncation]" where "dimensionality"
              can be 3/2, 2, 3, ... and "truncation" is a string specifying what
              link irrep truncation to use.

              Currently supported configurations: d=3/2, T1 or T2; d=2, T1.
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
        - circ_qasm_dir:
              Directory for read/write of circuit qasm files.
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
        - use_ancillas:
              Whether to introduce ancilla qubits to reduce to gate
              depth of multi-control gates.
        - plot_vacuum_persistence:
              Whether to plot all available vacuum
              persistence amplitude data.
        - plot_electric_energy:
              Whether to plot all available electric energy
              data.
        - save_circuits_to_qasm:
              Whether to save all generated circuits to disk
              as QASM files.
        - save_circuit_diagrams:
              Whether to save diagrams of all generated
              circuits to disk as PDFs.
        - save_plots:
              Whether to save all plots generated to disk as PDFs instead of shown immediately.
        - save_sim_data:
              Whether to save all circuit execution data to disk.
    """
    options: dict[str, Any] = dict()
    options["circ_qasm_dir"] = circ_qasm_dir
    options["plots_dir"] = plots_dir
    options["sim_results_dir"] = sim_results_dir
    options["do_electric_evolution"] = do_electric_evolution
    options["do_magnetic_evolution"] = do_magnetic_evolution
    options["dimensionality_and_truncation_string"] = dimensionality_and_truncation_string
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
    options["use_ancillas"] = use_ancillas
    options["n_shots"] = n_shots
    options["plot_vacuum_persistence"] = plot_vacuum_persistence
    options["plot_electric_energy"] = plot_electric_energy
    options["save_circuits_to_qasm"] = save_circuits_to_qasm
    options["save_circuit_diagrams"] = save_circuit_diagrams
    options["save_plots"] = save_plots
    options["save_sim_data"] = save_sim_data
    options["use_periodic_boundary_conds"] = use_periodic_boundary_conds

    # Automatically set some additional options based on user input above.
    dim_string, trunc_string = dimensionality_and_truncation_string.split(",")
    options["dim_string"] = dim_string.strip()
    options["trunc_string"] = trunc_string.strip()
    options["dimensions"] = 1.5 if (dim_string == "d=3/2" or dim_string == "d=1.5") else int(dim_string[2:])
    if options["trunc_string"] in ["T1", "T1p"]:
        options["link_bitmap"] = IRREP_TRUNCATION_DICT_1_3_3BAR
    elif options["trunc_string"] in ["T2"]:
        options["link_bitmap"] = IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8
    else:
        raise ValueError(f"Unknown irrep truncation: '{trunc_string}'.")
    options["lattice_def"] = LatticeDef(
        dimensions=options["dimensions"],
        size=options["lattice_size"],
        periodic_boundary_conds=options["use_periodic_boundary_conds"])

    return options


def create_circuits(script_options: dict[str, Any]) -> list[QuantumCircuit]:
    """
    Create a list of QuantumCircuit instances corresponding to a set
    of simulations.
    """
    circuits: list[QuantumCircuit] = []

    # Create lattice_encoder instance.
    lattice_encoder = create_lattice_encoder(script_options)

    # Load mag Hamiltonian data.
    mag_hamiltonian = load_magnetic_hamiltonian(
        script_options["dimensionality_and_truncation_string"],
        lattice_encoder,
        mag_hamiltonian_matrix_element_threshold=script_options["mag_hamiltonian_matrix_element_threshold"],
        only_include_elems_connected_to_electric_vacuum=script_options["mag_hamiltonian_use_electric_vacuum_transitions_only"])

    # Figure out physical states needed for control pruning.
    if script_options["prune_controls"] is True:
        physical_plaquette_states: Set[str] = set(
            lattice_encoder.encode_plaquette_state_as_bit_string(plaquette) for plaquette in PHYSICAL_PLAQUETTE_STATES[script_options["dimensionality_and_truncation_string"]])
    else:
        physical_plaquette_states = None

    # Create a simulation circuit for each sim duration.
    for sim_time in script_options["sim_times"]:
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
        dt = sim_time / script_options["n_trotter_steps"]
        for _ in range(script_options["n_trotter_steps"]):
            if script_options["do_magnetic_evolution"] is True:
                circ_mgr.apply_magnetic_trotter_step(
                    master_circuit,
                    lattice_registers,
                    coupling_g=script_options["coupling_g"],
                    dt=dt,
                    optimize_circuits=script_options["optimize_circuits"],
                    physical_states_for_control_pruning=physical_plaquette_states,
                    control_fusion=script_options["control_fusion"],
                    cache_mag_evol_circuit=True
                )

            if script_options["do_electric_evolution"] is True:
                circ_mgr.apply_electric_trotter_step(
                    master_circuit,
                    lattice_registers,
                    electric_hamiltonian(lattice_encoder.link_bitmap),
                    coupling_g=script_options["coupling_g"],
                    dt=dt,
                    electric_gray_order=script_options["electric_gray_order"])

        # Circuit complete.
        circuits.append(master_circuit)

    return circuits


def save_circuits(circuits: list[QuantumCircuit], simulation_identifier: str, script_options: dict[str, Any]) -> None:
    """
    Write a list of QuantumCircuit instances to
    a set of QASM files and/or save circuit diagrams to pdfs
    depending on script_options.

    The string simulation_identifier is combined with the timesteps in script_options to assign
    each circuit in circuits a meaningful filename. Each QASM file is saved in the
    directory specified by script_options['circ_qasm_dir'].
    """
    # Prep the circuit write directory.
    circuits_dir = script_options['circ_qasm_dir']
    if not isinstance(circuits_dir, Path):
        circuits_dir = Path(circuits_dir)
    circuits_dir.mkdir(exist_ok=True)

    # Prep tbe circuit diagram write directory.
    circuit_diagram_dir = script_options['plots_dir']
    if not isinstance(circuit_diagram_dir, Path):
        circuit_diagram_dir = Path(circuit_diagram_dir)
    circuit_diagram_dir.mkdir(exist_ok=True)

    # Iterate over the circuits for each sim time.
    # Write QASM file and/or PDF of circuit diagram.
    circuits_with_sim_time = zip(circuits, script_options["sim_times"])
    for circuit, sim_time in circuits_with_sim_time:
        if script_options["save_circuits_to_qasm"] is True:
            circuit_filename = simulation_identifier + f"-t={sim_time}.qasm"
            qasm_file_path = circuits_dir / circuit_filename
            with qasm_file_path.open('w') as qasm_file:
                qasm_file.write(dumps(circuit))
        if script_options["save_circuit_diagrams"] is True:
            diagram_filename = simulation_identifier + f"-t={sim_time}.pdf"
            diagram_file_path = circuit_diagram_dir / diagram_filename
            circuit.draw(
                output="mpl",
                filename=diagram_file_path,
                fold=False
            )


def run_circuit_simulations(circuits: list[QuantumCircuit], script_options: dict[str, Any]) -> pd.DataFrame:
    """
    Execute a simulation of each QuantumCircuit in circuits according
    to the parameters in script_options. Returns the results as a DataFrame.
    """
    # Set up objects needed for executing circuits and processing results.
    sampler = SamplerV2()
    n_ancilla_qubits = len(circuits[0].ancillas)
    n_total_qubits = len(circuits[0].qubits)
    n_data_qubits = n_total_qubits - n_ancilla_qubits
    vacuum_state = "0" * n_data_qubits
    lattice_encoder = create_lattice_encoder(script_options)
    print(f"# data qubits: {n_data_qubits}")
    print(f"# ancilla qubits: {n_ancilla_qubits}")

    # Prepare circuits for execution by adding a final
    # measurement of all registers, and then transpile.
    transpiled_circuits_with_final_measurement = []
    for circuit in circuits:
        transpiled_circuit_with_final_measurement = copy.deepcopy(circuit)
        transpiled_circuit_with_final_measurement.measure_all()
        transpiled_circuit_with_final_measurement = transpile(transpiled_circuit_with_final_measurement, optimization_level=3)
        transpiled_circuits_with_final_measurement.append(transpiled_circuit_with_final_measurement)

    # Execute circuits.
    job = sampler.run(transpiled_circuits_with_final_measurement, shots=script_options['n_shots'])
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
        physical_plaquette_states=PHYSICAL_PLAQUETTE_STATES[script_options["dimensionality_and_truncation_string"]],
        lattice=script_options["lattice_def"])

    return lattice_encoder


if __name__ == "__main__":
    # Set project root directory. Change as appropriate.
    PROJECT_ROOT = Path(__file__).parent.parent

    # Set simulation parameters here. See the docstring on
    # configure_script_options for an explanation of all
    # available options.
    script_options = configure_script_options(
        dimensionality_and_truncation_string="d=3/2, T1",
        lattice_size=2,
        sim_times=np.linspace(0.0, 2.5, num=20),
        n_trotter_steps=2,
        n_shots=10000,
        use_ancillas=True,
        save_circuits_to_qasm=True,
        save_circuit_diagrams=False,
        save_plots=True,
        save_sim_data=True,
        circ_qasm_dir=PROJECT_ROOT / "serialized-circuits",
        plots_dir=PROJECT_ROOT / "plots",
        sim_results_dir=PROJECT_ROOT / "sim-results",
        mag_hamiltonian_matrix_element_threshold=0.9
    )

    # Generate a descriptive prefix for all filenames based on simulation params.
    simulation_category_str_prefix = f"{script_options['lattice_def'].n_plaquettes}-plaquettes-in-d={script_options['lattice_def'].dim}-irrep_trunc={script_options['trunc_string']}-mat_elem_cut={script_options['mag_hamiltonian_matrix_element_threshold']}-vac_connected_only={script_options['mag_hamiltonian_use_electric_vacuum_transitions_only']}"

    # Create circuit(s) to simulate, optionally save to disk.
    simulation_circuits = create_circuits(script_options)
    if script_options["save_circuits_to_qasm"] is True or script_options["save_circuit_diagrams"] is True:
        save_circuits(simulation_circuits, simulation_category_str_prefix, script_options)

    # Either run circuits or skip.
    if script_options["n_shots"] is not None:
        sim_data = run_circuit_simulations(simulation_circuits, script_options)
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
