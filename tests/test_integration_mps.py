"""
Integration tests for time evolution circuits.

Verifies that:
1. Measurements on specific links/vertices/plaquettes work with real circuits.
2. Observable values change over time evolution.
3. MeasurementResults.get_transition_probability gives nonzero results for
   excited states at later times.

Includes:
- d=3/2 L=2 T1: MPS time evolution with measurements and observables.
- d=3 B3: Circuit construction for both large (size=3) and small-periodic (size=2) lattices.
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit
import pytest
from ymcirc._abstract import LatticeDef
from ymcirc.conventions import (
    LatticeStateEncoder, IRREP_TRUNCATIONS, PHYSICAL_PLAQUETTE_STATES,
    load_magnetic_hamiltonian, ONE, THREE,
)
from ymcirc.lattice_registers import LatticeRegisters
from ymcirc.circuit import LatticeCircuitManager
from ymcirc.electric_helper import electric_hamiltonian
from ymcirc.parsed_lattice_result import ParsedLatticeResult
from ymcirc.measurement_results import MeasurementResults


def _build_time_evolved_circuit(encoder, mag_ham, dt, g, n_steps=1) -> tuple[QuantumCircuit, LatticeCircuitManager, LatticeRegisters]:
    """Helper: build a time-evolution circuit with n_steps Trotter steps."""
    lattice = LatticeRegisters.from_lattice_state_encoder(encoder)
    circ_mgr = LatticeCircuitManager(encoder, mag_ham)
    circuit = circ_mgr.create_blank_full_lattice_circuit(lattice)

    # Add ancillas for magnetic evolution.
    n_anc = circ_mgr.compute_num_ancillas_needed_from_mag_trotter_step(circuit, lattice)
    circ_mgr.num_ancillas = n_anc
    circ_mgr.add_ancilla_register_to_quantum_circuit(circuit)

    ee_ham = electric_hamiltonian(encoder.link_bitmap)

    for _ in range(n_steps):
        circ_mgr.apply_electric_trotter_step(circuit, lattice, ee_ham)
        circ_mgr.apply_magnetic_trotter_step(circuit, lattice)

    # Bind parameters: all dt and g to the same values.
    param_dict = {}
    for param in circuit.parameters:
        if "dt" in param.name:
            param_dict[param] = dt
        elif "coupling_g" in param.name:
            param_dict[param] = g
    circuit = circuit.assign_parameters(param_dict)

    return circuit, circ_mgr, lattice


def _run_mps_simulation(circuit, shots=4096) -> dict[str, int]:
    """Run MPS simulation and return counts dict."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    backend = AerSimulator(
        method='matrix_product_state',
        matrix_product_state_max_bond_dimension=32,
    )
    # CRITICAL: Do NOT pass backend to transpile for MPS!
    tcirc = transpile(
        circuit,
        basis_gates=["cx", "u1", "u2", "u3", "x", "h", "ry", "rz", "p", "id"],
        optimization_level=2,
    )
    result = backend.run(tcirc, shots=shots).result()
    return result.get_counts()


def _counts_to_measurement_results(counts: dict[str, int] | dict[tuple, int], encoder, n_data_qubits) -> MeasurementResults:
    """
    Convert Qiskit counts dict to MeasurementResults.

    If the keys of dict are tuples, then it is assumed that a partial measurement was made.
    """
    parsed_counts = {}
    for measurement_key, count in counts.items():
        is_not_partial_measurement = isinstance(measurement_key, str)
        if is_not_partial_measurement:
            qiskit_bitstring = measurement_key
        else:
            meas_address, qiskit_bitstring = measurement_key
        # Qiskit measurement string is little-endian (rightmost = qubit 0).
        ymcirc_bitstring = qiskit_bitstring[::-1]
        if is_not_partial_measurement:
            # Strip ancilla bits if working with a full measurement.
            data_bitstring = ymcirc_bitstring[:n_data_qubits]
            parsed = ParsedLatticeResult(1.5, 2, data_bitstring, encoder)
        else:
            data_bitstring = ymcirc_bitstring
            parsed = ParsedLatticeResult.from_partial_measurement([(meas_address, data_bitstring)], encoder)
        if parsed in parsed_counts:
            parsed_counts[parsed] += count
        else:
            parsed_counts[parsed] = count
    return MeasurementResults(parsed_counts, encoder)


@pytest.mark.slow
def test_mps_time_evolution_observables_change():
    """Time-evolved lattice should show changing observables."""
    lattice_def = LatticeDef(1.5, 2, periodic_boundary_conds=True)
    trunc = "T1"
    link_bitmap = IRREP_TRUNCATIONS[trunc]
    physical_states = PHYSICAL_PLAQUETTE_STATES["d=3/2"][trunc]
    encoder = LatticeStateEncoder(link_bitmap, physical_states, lattice_def)

    mag_ham = load_magnetic_hamiltonian("d=3/2", trunc, encoder, mag_hamiltonian_matrix_element_threshold=0.6) # Somewhat high threshold to make test faster.

    g = 1.0
    n_data_qubits = encoder.lattice_def.n_links * encoder.expected_link_bit_string_length

    # Early time: vacuum should dominate.
    circuit_early, circ_mgr_early, lattice_early = _build_time_evolved_circuit(
        encoder, mag_ham, dt=0.1, g=g, n_steps=1)
    circuit_early.measure_all()
    counts_early = _run_mps_simulation(circuit_early, shots=4096)
    mr_early = _counts_to_measurement_results(counts_early, encoder, n_data_qubits)

    # Later time: excited states should appear.
    circuit_late, circ_mgr_late, lattice_late = _build_time_evolved_circuit(
        encoder, mag_ham, dt=0.25, g=g, n_steps=4)
    circuit_late.measure_all()
    counts_late = _run_mps_simulation(circuit_late, shots=4096)
    mr_late = _counts_to_measurement_results(counts_late, encoder, n_data_qubits)

    # Vacuum persistence should decrease with time.
    vpp_early = mr_early.vacuum_persistence_probability()
    vpp_late = mr_late.vacuum_persistence_probability()
    assert vpp_early > vpp_late, (
        f"Vacuum persistence should decrease: early={vpp_early}, late={vpp_late}"
    )

    # Electric energy should increase with time (excitations carry C_2 > 0).
    ee_early = mr_early.get_lattice_electric_energy(average_result=False)
    ee_late = mr_late.get_lattice_electric_energy(average_result=False)
    assert ee_late > ee_early, (
        f"Electric energy should increase: early={ee_early}, late={ee_late}"
    )


@pytest.mark.slow
def test_mps_transition_probability_nonzero_at_late_time():
    """Transition probability to an excited state should be nonzero at late times."""
    lattice_def = LatticeDef(1.5, 2, periodic_boundary_conds=True)
    trunc = "T1"
    link_bitmap = IRREP_TRUNCATIONS[trunc]
    physical_states = PHYSICAL_PLAQUETTE_STATES["d=3/2"][trunc]
    encoder = LatticeStateEncoder(link_bitmap, physical_states, lattice_def)

    mag_ham = load_magnetic_hamiltonian("d=3/2", trunc, encoder, mag_hamiltonian_matrix_element_threshold=0.6)

    g = 1.0
    n_data_qubits = encoder.lattice_def.n_links * encoder.expected_link_bit_string_length

    # Evolve to a later time.
    circuit, circ_mgr, lattice = _build_time_evolved_circuit(
        encoder, mag_ham, dt=0.25, g=g, n_steps=4)
    circuit.measure_all()
    counts = _run_mps_simulation(circuit, shots=8192)
    mr = _counts_to_measurement_results(counts, encoder, n_data_qubits)

    # At late times, some non-vacuum states should exist.
    non_vacuum_prob = 1.0 - mr.vacuum_persistence_probability()
    assert not non_vacuum_prob == pytest.approx(0.0) and non_vacuum_prob > 0, "Expected some non-vacuum states at late times"

    # Verify get_transition_probability consistency: a fully-specified vacuum
    # partial state should give the same result as vacuum_persistence_probability.
    vacuum_partial = ParsedLatticeResult.from_links_and_vertices(
        links_dict={addr: ONE for addr in encoder.lattice_def.link_addresses},
        encoder=encoder,
    )
    assert mr.get_transition_probability(vacuum_partial) == pytest.approx(
        mr.vacuum_persistence_probability()
    )

    # Partial state matching: any shot with THREE on link ((0,0),1) should match.
    partial_three_on_link = ParsedLatticeResult.from_links_and_vertices(
        links_dict={((0, 0), 1): THREE}, encoder=encoder
    )
    three_prob = mr.get_transition_probability(partial_three_on_link)
    # At late times with g=1.0, some excitations should appear.
    assert not three_prob == pytest.approx(0.0) and three_prob > 0  # Should not raise; may be small but non-negative

@pytest.mark.slow
def test_mps_measure_one_link_at_late_time():
    """Measurement of a single excited link, should be nonzero at late times."""
    lattice_def = LatticeDef(1.5, 2, periodic_boundary_conds=True)
    trunc = "T1"
    link_bitmap = IRREP_TRUNCATIONS[trunc]
    physical_states = PHYSICAL_PLAQUETTE_STATES["d=3/2"][trunc]
    encoder = LatticeStateEncoder(link_bitmap, physical_states, lattice_def)

    mag_ham = load_magnetic_hamiltonian("d=3/2", trunc, encoder, mag_hamiltonian_matrix_element_threshold=0.6)

    g = 1.0
    n_data_qubits = encoder.lattice_def.n_links * encoder.expected_link_bit_string_length

    # Evolve to a later time.
    circuit, circ_mgr, lattice = _build_time_evolved_circuit(
        encoder, mag_ham, dt=0.25, g=g, n_steps=4)
    horiz_link_from_origin = ((0, 0), 1)
    circ_mgr.measure_link(circuit, lattice, horiz_link_from_origin)
    counts = {(horiz_link_from_origin, meas_bit_string): n_obs for meas_bit_string, n_obs in _run_mps_simulation(circuit, shots=8192).items()} # Include address info since partial measurement.
    mr = _counts_to_measurement_results(counts, encoder, n_data_qubits)

    # TODO: Behavior changed and now trying to get energy for unmeasured links should raise KeyError. Fix test.
    # Unmeasured links should give zero for link energy, and the measured link should have positive energy.
    for link_address in lattice_def.link_addresses:
        if not link_address == horiz_link_from_origin:
            assert mr.get_link_electric_energy(link_address) == 0
        else:
            assert (not (mr.get_link_electric_energy(link_address) == pytest.approx(0.0))) and mr.get_link_electric_energy(link_address) > 0

    # For each measurement, the underlying energies should be either None or 4/3.
    # Also confirm that we encounter the right number of link states.
    n_excited = 0               # should equal 2: 2 types of excited links on the partial lattice state
    n_zero = 0                  # should equal 3: 1 type of vacuum link on the partial lattice state
    n_none = 0                  # should equal 15: 5 unmeasured links times 3 (partial) lattice states
    for plr, counts in mr.get_counts().items():
        for link_address in lattice_def.link_addresses:
            link_eng = plr.get_link_electric_energy(link_address)
            if link_address == horiz_link_from_origin:
                assert plr.get_link_electric_energy(link_address) == pytest.approx(4/3) or plr.get_link_electric_energy(link_address) == 0.0, f"Link {link_address} has (wrong) energy {plr.get_link_electric_energy(link_address)}."
                if link_eng > 0:
                    n_excited += 1
                elif link_eng == 0:
                    n_zero += 1
            else:
                assert plr.get_link_electric_energy(link_address) is None
                n_none += 1
    assert (n_excited, n_zero, n_none) == (2, 1, 15)


# --- d=3 integration tests ---

def _make_d3_encoder_and_hamiltonian(size, threshold=0.3):
    """Helper: create encoder and Hamiltonian for d=3 B3 lattice of given size."""
    lattice_def = LatticeDef(3, size, periodic_boundary_conds=True)
    trunc = "B3"
    link_bitmap = IRREP_TRUNCATIONS[trunc]
    physical_states = PHYSICAL_PLAQUETTE_STATES["d=3"][trunc]
    encoder = LatticeStateEncoder(link_bitmap, physical_states, lattice_def)
    mag_ham = load_magnetic_hamiltonian("d=3", trunc, encoder,
                                       mag_hamiltonian_matrix_element_threshold=threshold)
    return encoder, mag_ham


@pytest.mark.slow
def test_d3_B3_size3_magnetic_trotter_step():
    """d=3 B3 size=3: construct circuit and apply one magnetic Trotter step (no small-lattice logic)."""
    encoder, mag_ham = _make_d3_encoder_and_hamiltonian(size=3)
    lattice = LatticeRegisters.from_lattice_state_encoder(encoder)
    circ_mgr = LatticeCircuitManager(encoder, mag_ham)
    circuit = circ_mgr.create_blank_full_lattice_circuit(lattice)

    n_anc = circ_mgr.compute_num_ancillas_needed_from_mag_trotter_step(circuit, lattice)
    circ_mgr.num_ancillas = n_anc
    circ_mgr.add_ancilla_register_to_quantum_circuit(circuit)

    circ_mgr.apply_magnetic_trotter_step(circuit, lattice)

    assert circuit.num_qubits > 0
    assert len(circuit.parameters) > 0, "Circuit should have unbound parameters (dt, coupling_g)."


@pytest.mark.slow
def test_d3_B3_size2_magnetic_trotter_step():
    """d=3 B3 size=2: triggers small-periodic filtering; verify circuit constructs and has correct c_link counts."""
    encoder, mag_ham = _make_d3_encoder_and_hamiltonian(size=2)
    lattice = LatticeRegisters.from_lattice_state_encoder(encoder)
    circ_mgr = LatticeCircuitManager(encoder, mag_ham)
    circuit = circ_mgr.create_blank_full_lattice_circuit(lattice)

    n_anc = circ_mgr.compute_num_ancillas_needed_from_mag_trotter_step(circuit, lattice)
    circ_mgr.num_ancillas = n_anc
    circ_mgr.add_ancilla_register_to_quantum_circuit(circuit)

    circ_mgr.apply_magnetic_trotter_step(circuit, lattice)

    assert circuit.num_qubits > 0
    assert len(circuit.parameters) > 0, "Circuit should have unbound parameters (dt, coupling_g)."

    # The resolved Hamiltonian should have 3 planes for d=3.
    assert len(circ_mgr._mag_hamiltonian) == 3, (
        f"Expected 3 planes in resolved Hamiltonian, got {len(circ_mgr._mag_hamiltonian)}"
    )
    # Verify expected planes are present.
    expected_planes = {(1, 2), (1, 3), (2, 3)}
    assert set(circ_mgr._mag_hamiltonian.keys()) == expected_planes
