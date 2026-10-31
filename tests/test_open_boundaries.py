"""
Tests of open and mixed boundary conditions end to end.

These tests simulate magnetic and electric Trotter steps and check Gauss's law;
prepare, measure and parse states; and check the size of circuits on d=3
lattices with open boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveStatevector

from ymcirc._abstract import LatticeDef
from ymcirc._abstract.lattice_data import (
    LatticeVector,
    LinkAddress,
    Plaquette,
    PlaquetteState,
    Signature,
)
from ymcirc._pyclebsch_adapter import box_plus_box_dagger_matrix_elements
from ymcirc.circuit import LatticeCircuitManager
from ymcirc.conventions import (
    EIGHT,
    ONE,
    IrrepWeight,
    LatticeStateEncoder,
    physical_plaquette_states,
)
from ymcirc.electric_helper import electric_hamiltonian
from ymcirc.lattice_registers import LatticeRegisters
from ymcirc.parsed_lattice_result import ParsedLatticeResult

# Circuits of up to this many qubits are simulated as a statevector: 16 bytes
# per amplitude, 270 MB at 24 qubits. Larger ones are simulated with Aer's
# matrix product state method with no cap on the bond dimension, which is
# exact up to rounding.
_MAX_STATEVECTOR_QUBITS = 24

# The size of the Trotter step and the coupling in the simulations of Gauss's
# law. Any values that mix states work: with these, each Givens rotation
# turns by its matrix element, which for B3 is of order 1.
_DT = 1.0
_COUPLING = 1.0
# Enough Trotter steps that a state reached in one step is acted on again.
_N_TROTTER_STEPS = 3


def _simulator(n_qubits: int) -> AerSimulator:
    """Aer's statevector simulator for up to _MAX_STATEVECTOR_QUBITS qubits, else its matrix product state simulator."""
    if n_qubits <= _MAX_STATEVECTOR_QUBITS:
        return AerSimulator(method="statevector")
    return AerSimulator(method="matrix_product_state")


def _probabilities_above_atol(circuit: QuantumCircuit) -> dict[str, float]:
    """
    The probabilities of the circuit's computational basis states, from its statevector in double precision, that are larger than ATOL_DEFAULT.

    ATOL_DEFAULT is Qiskit's default absolute tolerance for comparing
    matrices (qiskit.quantum_info.operators.predicates). A probability that
    should be zero is the square of a rounding error in an amplitude, so it
    falls below that tolerance. Keys are bit strings in ymcirc's order, the
    circuit's qubit 0 first.

    When Aer runs a circuit, the result it returns contains only data that
    instructions in the circuit produce. A measurement produces counts, and
    a SaveStatevector instruction (qiskit_aer.library) produces a copy of
    the statevector the simulation has reached at that point. Simulating
    with method="statevector" decides what Aer tracks during the run, not
    what Aer reports. Without a SaveStatevector instruction, the result
    holds no statevector, and result.get_statevector() raises a
    QiskitError. So SaveStatevector is appended as the circuit's last
    instruction, which puts the statevector at the end of the circuit in
    the result.
    """
    n_qubits = circuit.num_qubits
    assert n_qubits <= _MAX_STATEVECTOR_QUBITS
    backend = _simulator(n_qubits)
    compiled = transpile(circuit, backend, optimization_level=0)
    compiled.append(SaveStatevector(compiled.num_qubits), compiled.qubits)
    statevector = np.asarray(backend.run(compiled).result().get_statevector())
    probabilities = np.abs(statevector) ** 2
    # Qiskit's basis-state bit strings are little-endian (rightmost = qubit 0),
    # so each is reversed to put qubit 0 first, as ymcirc's bit strings do.
    return {
        format(index, f"0{n_qubits}b")[::-1]: float(probabilities[index])
        for index in np.flatnonzero(probabilities > ATOL_DEFAULT)
    }


def _measured_bit_string(circuit: QuantumCircuit) -> str:
    """The bit string of one noiseless measurement of every qubit, in ymcirc's order, for a circuit that prepares a basis state."""
    measured = circuit.copy()
    measured.measure_all()
    backend = _simulator(circuit.num_qubits)
    counts = (
        backend.run(transpile(measured, backend, optimization_level=0), shots=1)
        .result()
        .get_counts()
    )
    (qiskit_bit_string,) = counts
    # Qiskit's measured bit strings are little-endian (rightmost = qubit 0),
    # so the bit string is reversed to put qubit 0 first, as ymcirc's do.
    return qiskit_bit_string[::-1]


@pytest.mark.parametrize(
    ("size", "periodic"),
    [
        (2, False),
        ((3, 2), False),
        # 24 qubits, the largest statevector of these tests.
        pytest.param((3, 2), True, marks=pytest.mark.slow),
    ],
)
def test_gauss_law_after_trotter_steps(size, periodic):
    """
    After Trotter steps from the electric vacuum with control pruning on, every plaquette is in a physical state.

    d=2 at B3. Each Trotter step is a magnetic step, then an electric step.
    Control pruning removes controls that distinguish physical states from
    unphysical ones only, so an error in it puts probability on bit strings
    that decode to an unphysical plaquette state. Every bit string with
    probability above Qiskit's ATOL_DEFAULT must decode, at every plaquette,
    to a physical state of the plaquette's plane and signature.
    """
    lattice_def = LatticeDef(2, size, periodic)
    encoder = LatticeStateEncoder.for_truncation(lattice_def, "B", 3)
    assert encoder.expected_vertex_bit_string_length == 0
    physical_states = physical_plaquette_states(lattice_def, "B", 3)
    registers = LatticeRegisters.from_lattice_state_encoder(encoder)
    circ_mgr = LatticeCircuitManager.for_truncation(encoder)
    circuit = circ_mgr.create_blank_full_lattice_circuit(registers)
    for _ in range(_N_TROTTER_STEPS):
        circ_mgr.apply_magnetic_trotter_step(
            circuit,
            registers,
            optimize_circuits=False,
            physical_states_for_control_pruning=physical_states,
            cache_mag_evol_circuit=True,
        )
        circ_mgr.apply_electric_trotter_step(
            circuit, registers, electric_hamiltonian(encoder.link_bitmap)
        )
    circuit.assign_parameters(
        {
            parameter: _DT if parameter.name.startswith("dt") else _COUPLING
            for parameter in circuit.parameters
        },
        inplace=True,
    )

    # With no vertex bits (asserted above), the multiplicity indices of a
    # decoded plaquette are None, so plaquettes are compared by the irreps
    # on their links alone.
    physical_link_irreps = {
        key: {
            (active_links, control_links) for _, active_links, control_links in states
        }
        for key, states in physical_states.items()
    }
    probabilities = _probabilities_above_atol(circuit)
    assert len(probabilities) > 1, (
        "The Trotter steps left the vacuum unchanged, so nothing is checked."
    )
    for bit_string in probabilities:
        result = ParsedLatticeResult(2, lattice_def.size, bit_string, encoder, periodic)
        for bottom_left_vertex, plane in lattice_def.plaquette_addresses:
            plaquette = Plaquette(result, bottom_left_vertex, plane)
            link_irreps = (
                tuple(plaquette.active_links),
                tuple(tuple(links) for links in plaquette.control_links_per_vertex),
            )
            assert link_irreps in physical_link_irreps[(plane, plaquette.signature)], (
                f"Plaquette at {bottom_left_vertex} in plane {plane}, bit string {bit_string}."
            )


def _vacuum_plaquette_state(signature: Signature) -> PlaquetteState:
    """The electric vacuum of a plaquette with this signature: every link carries 1 and every multiplicity index is 0."""
    # Two of a vertex's half-links are the plaquette's active links; the rest are its controls.
    c1, c2, c3, c4 = ((ONE,) * (len(half_links) - 2) for half_links in signature)
    return ((0, 0, 0, 0), (ONE, ONE, ONE, ONE), (c1, c2, c3, c4))


def _states_one_box_transition_from_vacuum(
    lattice_def: LatticeDef, truncation_mode: str, cutoff: int
) -> list[dict[LinkAddress, IrrepWeight]]:
    """
    The electric vacuum, and each state one box transition from it, as the irrep on every link.

    A transition is a nonzero matrix element of box + box^dagger at the
    truncation whose one plaquette state is the vacuum's; it is applied to
    each plaquette of its plane and signature in turn, changing that
    plaquette's active links. The box transitions change no control link,
    and leave every multiplicity index 0.
    """
    vacuum = {link_address: ONE for link_address in lattice_def.link_addresses}
    states = [vacuum]
    matrix_elements = box_plus_box_dagger_matrix_elements(
        lattice_def, truncation_mode, cutoff
    )
    for bottom_left_vertex, plane in lattice_def.plaquette_addresses:
        e1, e2 = plane
        signature = lattice_def._plaquette_signature(bottom_left_vertex, plane)
        vacuum_plaquette = _vacuum_plaquette_state(signature)
        v2 = lattice_def.add_unit_vector_to_vertex_vector(bottom_left_vertex, e1)
        v4 = lattice_def.add_unit_vector_to_vertex_vector(bottom_left_vertex, e2)
        active_link_addresses = [
            (bottom_left_vertex, e1),
            (v2, e2),
            (v4, e1),
            (bottom_left_vertex, e2),
        ]
        for (state_1, state_2), value in matrix_elements[(plane, signature)].items():
            if value == 0 or vacuum_plaquette not in (state_1, state_2):
                continue
            excited = state_2 if state_1 == vacuum_plaquette else state_1
            assert excited[2] == vacuum_plaquette[2]
            assert excited[0] == vacuum_plaquette[0]
            state = dict(vacuum)
            for link_address, irrep in zip(active_link_addresses, excited[1]):
                state[tuple(link_address)] = irrep
            states.append(state)
    return states


def _prepared_measured_and_parsed(
    lattice_def: LatticeDef,
    encoder: LatticeStateEncoder,
    links: dict[LinkAddress, IrrepWeight],
    vertices: dict[LatticeVector, int],
) -> ParsedLatticeResult:
    """
    A physical state of the lattice prepared with X gates, measured noiselessly, and parsed.

    Preparation finds each link's and vertex's qubits through
    LatticeRegisters.get_link and get_vertex, and parsing reads their bits
    at their places in the traversal order, and then applies the X gates.
    """
    registers = LatticeRegisters.from_lattice_state_encoder(encoder)
    circuit = LatticeCircuitManager(encoder, {}).create_blank_full_lattice_circuit(
        registers
    )
    for link_address, irrep in links.items():
        for qubit, bit in zip(
            registers.get_link(link_address),
            encoder.encode_link_state_as_bit_string(irrep),
        ):
            if bit == "1":
                circuit.x(qubit)
    for vertex, multiplicity_index in vertices.items():
        for qubit, bit in zip(
            registers.get_vertex(vertex),
            encoder.encode_vertex_state_as_bit_string(multiplicity_index),
        ):
            if bit == "1":
                circuit.x(qubit)
    return ParsedLatticeResult(
        lattice_def.dim,
        lattice_def.size,
        _measured_bit_string(circuit),
        encoder,
        lattice_def.periodic_boundary_conds,
    )


@pytest.mark.parametrize(
    ("dimensions", "size", "periodic"),
    [
        (2, 2, False),
        (2, (3, 2), False),
        (2, (3, 2), True),
        (2, (3, 3), (True, False)),
        pytest.param(3, (3, 2, 2), False, marks=pytest.mark.slow),
    ],
)
def test_round_trip_through_preparation_measurement_and_parsing(
    dimensions, size, periodic
):
    """
    A physical state, prepared with X gates on the link registers, measured and parsed, gives the same state back.

    Preparation finds each link's qubits through LatticeRegisters.get_link,
    and parsing reads each link's bits at its place in the traversal order,
    so a disagreement between the two shows. The states are the electric
    vacuum and those one box transition from it, at B3.
    """
    lattice_def = LatticeDef(dimensions, size, periodic)
    encoder = LatticeStateEncoder.for_truncation(lattice_def, "B", 3)
    assert encoder.expected_vertex_bit_string_length == 0
    states = _states_one_box_transition_from_vacuum(lattice_def, "B", 3)
    assert len(states) > 1

    for state in states:
        result = _prepared_measured_and_parsed(lattice_def, encoder, state, {})
        assert {
            link_address: result.get_link(link_address)
            for link_address in lattice_def.link_addresses
        } == state


def test_round_trip_with_vertex_qubits():
    """
    On d=2 3x2 open at T2, states with vertex qubits, prepared, measured and parsed, give the same states back.

    T2 puts 8 on links, and 8 x 8 x 8 has two singlets, so the two middle
    vertices, (1, 0) and (1, 1), which have three links, need a vertex bit
    (index 0 is "0", 1 is "1"). Every vertex gets one, as all share one bit
    length: 27 qubits in all, simulated with MPS. The states are the
    electric vacuum and those one box transition from it, with every
    multiplicity index 0, and one with an 8 on every link and index 1 at
    the two middle vertices. The test checks that each plaquette of that
    last state is a physical plaquette state of T2.
    """
    lattice_def = LatticeDef(2, (3, 2), False)
    encoder = LatticeStateEncoder.for_truncation(lattice_def, "T", 2)
    assert encoder.expected_vertex_bit_string_length == 1
    all_eights = {link_address: EIGHT for link_address in lattice_def.link_addresses}
    second_singlet_in_the_middle = {
        vertex: 1 if vertex in [(1, 0), (1, 1)] else 0
        for vertex in lattice_def.vertex_addresses
    }
    physical_states = physical_plaquette_states(lattice_def, "T", 2)
    all_eights_parsed = ParsedLatticeResult.from_links_and_vertices(
        all_eights, second_singlet_in_the_middle, encoder=encoder
    )
    for bottom_left_vertex, plane in lattice_def.plaquette_addresses:
        plaquette = Plaquette(all_eights_parsed, bottom_left_vertex, plane)
        state = (
            tuple(plaquette.vertices),
            tuple(plaquette.active_links),
            tuple(tuple(links) for links in plaquette.control_links_per_vertex),
        )
        assert state in physical_states[(plane, plaquette.signature)]

    vacuum_indices = {vertex: 0 for vertex in lattice_def.vertex_addresses}
    states = [
        (links, vacuum_indices)
        for links in _states_one_box_transition_from_vacuum(lattice_def, "T", 2)
    ]
    states.append((all_eights, second_singlet_in_the_middle))
    for links, vertices in states:
        result = _prepared_measured_and_parsed(lattice_def, encoder, links, vertices)
        assert {
            link_address: result.get_link(link_address)
            for link_address in lattice_def.link_addresses
        } == links
        assert {
            vertex: result.get_vertex(vertex) for vertex in lattice_def.vertex_addresses
        } == vertices


@pytest.mark.slow
def test_structure_d_3_open():
    """
    On the d=3 lattice of shape (3, 2, 2) with open boundaries, at B3: counts of links, plaquettes, signatures, qubits and plaquette circuits.

    20 links: 8 along axis 1, 6 along axis 2, 6 along axis 3. 11 plaquettes:
    4 in plane (1, 2), 4 in (1, 3), 3 in (2, 3). Every plaquette has a
    (plane, signature) of its own, so the magnetic Trotter step builds and
    caches 11 plaquette circuits. 40 qubits: 2 per link.
    """
    lattice_def = LatticeDef(3, (3, 2, 2), False)
    encoder = LatticeStateEncoder.for_truncation(lattice_def, "B", 3)
    registers = LatticeRegisters.from_lattice_state_encoder(encoder)
    circ_mgr = LatticeCircuitManager.for_truncation(encoder)
    circuit = circ_mgr.create_blank_full_lattice_circuit(registers)
    assert (
        lattice_def.n_links,
        lattice_def.n_plaquettes,
        len(lattice_def._planes_and_signatures()),
    ) == (20, 11, 11)
    assert circuit.num_qubits == 40
    circ_mgr.apply_magnetic_trotter_step(
        circuit, registers, optimize_circuits=False, cache_mag_evol_circuit=True
    )
    assert len(circ_mgr._cached_mag_evol_circuits) == 11


def test_structure_d_3_mixed():
    """
    On the d=3 lattice of shape (3, 3, 2), periodic along axes 1 and 2 and open along axis 3, at B3: counts of links, plaquettes, qubits and plaquette circuits.

    45 links: 18 along each of axes 1 and 2, 9 along axis 3. 36 plaquettes:
    18 in plane (1, 2), 9 in each of (1, 3) and (2, 3). 90 qubits: 2 per
    link. The magnetic Trotter step builds and caches one plaquette circuit
    per (plane, signature), 4 of them. Along axis 3, a vertex with z = 0 has
    only the half-link +3 and one with z = 1 only -3, and every vertex has
    +-1 and +-2. So plane (1, 2) has two signatures, one for the plaquettes
    at z = 0 and one for those at z = 1. Planes (1, 3) and (2, 3) have one
    each, since all their plaquettes run from z = 0 to z = 1.
    """
    lattice_def = LatticeDef(3, (3, 3, 2), (True, True, False))
    encoder = LatticeStateEncoder.for_truncation(lattice_def, "B", 3)
    registers = LatticeRegisters.from_lattice_state_encoder(encoder)
    circ_mgr = LatticeCircuitManager.for_truncation(encoder)
    circuit = circ_mgr.create_blank_full_lattice_circuit(registers)
    assert (lattice_def.n_links, lattice_def.n_plaquettes) == (45, 36)
    assert circuit.num_qubits == 90
    circ_mgr.apply_magnetic_trotter_step(
        circuit, registers, optimize_circuits=False, cache_mag_evol_circuit=True
    )
    assert len(circ_mgr._cached_mag_evol_circuits) == 4
