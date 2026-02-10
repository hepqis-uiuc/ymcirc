import pytest
from ymcirc._abstract import LatticeDef
from ymcirc.conventions import (
    LatticeStateEncoder, ONE, THREE, THREE_BAR
)
from ymcirc.parsed_lattice_result import ParsedLatticeResult
from ymcirc.measurement_results import MeasurementResults


@pytest.fixture
def encoder_d32_L2_T1():
    link_bitmap = {ONE: "00", THREE: "10", THREE_BAR: "01"}
    physical_plaquette_states = [
        ((0, 0, 0, 0), (ONE, ONE, ONE, ONE), (ONE, ONE, ONE, ONE)),
        ((0, 0, 0, 0), (THREE, THREE, THREE_BAR, THREE_BAR), (ONE, ONE, ONE, ONE)),
    ]
    lattice = LatticeDef(1.5, 2)
    return LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)


def test_measurement_results_link_electric_energy(encoder_d32_L2_T1):
    """get_link_electric_energy returns weighted average over all shots."""
    encoder = encoder_d32_L2_T1
    # 70 shots of vacuum (all ONE), 30 shots of THREE on link ((0,0),1)
    plr_vacuum = ParsedLatticeResult(1.5, 2, "000000000000", encoder)
    plr_excited = ParsedLatticeResult(1.5, 2, "100000000000", encoder)

    counts = {plr_vacuum: 70, plr_excited: 30}
    mr = MeasurementResults(counts, encoder)

    # Link ((0,0),1): 70 * 0 + 30 * 4/3 = 40. / 100 = 0.4
    assert mr.get_link_electric_energy(((0, 0), 1)) == pytest.approx(0.4)
    # Link ((0,0),2): all vacuum -> 0
    assert mr.get_link_electric_energy(((0, 0), 2)) == pytest.approx(0.0)


def test_measurement_results_lattice_electric_energy(encoder_d32_L2_T1):
    """get_lattice_electric_energy returns expectation value over all shots."""
    encoder = encoder_d32_L2_T1
    plr_vacuum = ParsedLatticeResult(1.5, 2, "000000000000", encoder)
    plr_excited = ParsedLatticeResult(1.5, 2, "100000000000", encoder)

    counts = {plr_vacuum: 70, plr_excited: 30}
    mr = MeasurementResults(counts, encoder)

    # Total: 70 * 0.0 + 30 * 4/3 = 40/3. Expectation = 40/300 = 2/15
    # Wait -- lattice energy is sum over ALL links. plr_excited has THREE on
    # link ((0,0),1) (C_2=4/3) and ONE on all others. So lattice energy = 4/3.
    # Average over shots: (70 * 0 + 30 * 4/3) / 100 = 40/100 = 0.4
    assert mr.get_lattice_electric_energy(average_result=False) == pytest.approx(0.4)

    # With average_result=True: per-link average is computed per-instance then averaged.
    # plr_vacuum: 0/6 = 0. plr_excited: (4/3)/6 = 2/9.
    # Expectation: (70*0 + 30*2/9) / 100 = 60/900 = 1/15
    assert mr.get_lattice_electric_energy(average_result=True) == pytest.approx(1.0 / 15.0)


def test_vacuum_persistence_probability(encoder_d32_L2_T1):
    """vacuum_persistence_probability should be the fraction in vacuum state."""
    encoder = encoder_d32_L2_T1
    plr_vacuum = ParsedLatticeResult(1.5, 2, "000000000000", encoder)
    plr_excited = ParsedLatticeResult(1.5, 2, "100000000000", encoder)

    counts = {plr_vacuum: 80, plr_excited: 20}
    mr = MeasurementResults(counts, encoder)

    assert mr.vacuum_persistence_probability == pytest.approx(0.8)


def test_get_transition_probability(encoder_d32_L2_T1):
    """get_transition_probability returns the empirical probability of a state."""
    encoder = encoder_d32_L2_T1
    plr_vacuum = ParsedLatticeResult(1.5, 2, "000000000000", encoder)
    plr_excited = ParsedLatticeResult(1.5, 2, "100000000000", encoder)
    plr_other = ParsedLatticeResult(1.5, 2, "010101010101", encoder)

    counts = {plr_vacuum: 60, plr_excited: 40}
    mr = MeasurementResults(counts, encoder)

    assert mr.get_transition_probability(plr_vacuum) == pytest.approx(0.6)
    assert mr.get_transition_probability(plr_excited) == pytest.approx(0.4)
    assert mr.get_transition_probability(plr_other) == pytest.approx(0.0)


def test_get_transition_probability_with_partial_state(encoder_d32_L2_T1):
    """get_transition_probability should match partial states against full states."""
    encoder = encoder_d32_L2_T1
    plr_vacuum = ParsedLatticeResult(1.5, 2, "000000000000", encoder)
    plr_excited = ParsedLatticeResult(1.5, 2, "100000000000", encoder)

    counts = {plr_vacuum: 60, plr_excited: 40}
    mr = MeasurementResults(counts, encoder)

    # Partial state: only specifies link ((0,0),1) = THREE.
    # Should match plr_excited (which has THREE on that link) but not plr_vacuum.
    partial_excited = ParsedLatticeResult.from_links_and_vertices(
        links_dict={((0, 0), 1): THREE}, encoder=encoder
    )
    assert mr.get_transition_probability(partial_excited) == pytest.approx(0.4)

    # Partial state matching vacuum: only specifies link ((0,0),1) = ONE.
    # Should match plr_vacuum only.
    partial_vacuum = ParsedLatticeResult.from_links_and_vertices(
        links_dict={((0, 0), 1): ONE}, encoder=encoder
    )
    assert mr.get_transition_probability(partial_vacuum) == pytest.approx(0.6)

    # Fully-specified partial state matching all 6 links to ONE.
    # Should exactly match vacuum.
    full_vacuum = ParsedLatticeResult.from_links_and_vertices(
        links_dict={addr: ONE for addr in encoder.lattice_def.link_addresses},
        encoder=encoder
    )
    assert mr.get_transition_probability(full_vacuum) == pytest.approx(0.6)


def test_measurement_results_empty_counts_raises(encoder_d32_L2_T1):
    """MeasurementResults should raise ValueError for empty counts."""
    with pytest.raises(ValueError):
        MeasurementResults({}, encoder_d32_L2_T1)
