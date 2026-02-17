import pytest
import warnings
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

@pytest.fixture
def encoder_d2_L2_T1():
    link_bitmap = {ONE: "00", THREE: "10", THREE_BAR: "01"}
    physical_plaquette_states = [
        ((0, 1, 0, 0), (ONE, ONE, ONE, ONE), (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)),
        ((0, 0, 0, 1), (THREE, THREE, THREE_BAR, THREE_BAR), (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)),
    ]
    lattice = LatticeDef(2, 2)
    return LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)


def test_measurement_results_polymorphic_init(encoder_d32_L2_T1):
    """Should be able to initialize with either ParsedLatticeResult or string keys."""
    encoder = encoder_d32_L2_T1
    vac_string = "000000000000"
    excited_string = "000000000001"
    counts_str_keys = {vac_string: 10, excited_string: 90}
    counts_plr_keys = {
        ParsedLatticeResult(1.5, 2, vac_string, encoder): 10,
        ParsedLatticeResult(1.5, 2, excited_string, encoder): 90
    }

    mr_from_str_dict = MeasurementResults(counts_str_keys, encoder)
    mr_from_plr_dict = MeasurementResults(counts_plr_keys, encoder)

    for link_address in encoder.lattice_def.link_addresses:
        assert mr_from_str_dict.get_link_electric_energy(link_address) == mr_from_plr_dict.get_link_electric_energy(link_address)
    assert mr_from_str_dict.get_lattice_electric_energy() == mr_from_plr_dict.get_lattice_electric_energy()
    assert mr_from_str_dict.vacuum_persistence_probability() == pytest.approx(0.1)
    assert mr_from_plr_dict.vacuum_persistence_probability() == pytest.approx(0.1)


def test_get_link_electric_energy(encoder_d32_L2_T1):
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


def test_get_link_electric_energy_unphys_mode(encoder_d32_L2_T1):
    """get_link_electric_energy returns weighted average over all shots."""
    encoder = encoder_d32_L2_T1
    plr_excited = ParsedLatticeResult(1.5, 2, "000000000010", encoder) # Link ((1,1), 1): THREE
    plr_excited_one_unphys = ParsedLatticeResult(1.5, 2, "100000000011", encoder) # Link ((1,1), 1): unphys

    counts = {plr_excited: 70, plr_excited_one_unphys: 30}
    mr = MeasurementResults(counts, encoder)
    l_1_1_d_1_expected_energy = 0.7 * 4/3.0
    # # Link ((0,0),1): 70 * 0 + 30 * 4/3 = 40. / 100 = 0.4
    # assert mr.get_link_electric_energy(((0, 0), 1)) == pytest.approx(0.4)
    # # Link ((0,0),2): all vacuum -> 0
    # assert mr.get_link_electric_energy(((0, 0), 2)) == pytest.approx(0.0)
    
    # No warning when flag set to None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert mr.get_link_electric_energy(((1, 1), 1), unphys_mode=None) == pytest.approx(l_1_1_d_1_expected_energy)
        assert len(w) == 0

    # Warning when flag not set by user (default behavior)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mr.get_link_electric_energy(((1, 1), 1))
        assert len(w) == 1

    # Warning when flag set to 'warn'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mr.get_link_electric_energy(((1, 1), 1), unphys_mode='warn')
        assert len(w) == 1

    # Error when flag set to 'err'
    with pytest.raises(KeyError, match="unphysical"):
         mr.get_link_electric_energy(((1, 1), 1), unphys_mode='err')
    

def test_get_lattice_electric_energy(encoder_d32_L2_T1):
    """get_lattice_electric_energy returns expectation value over all shots."""
    encoder = encoder_d32_L2_T1
    plr_vacuum = ParsedLatticeResult(1.5, 2, "000000000000", encoder)
    plr_excited = ParsedLatticeResult(1.5, 2, "100000000000", encoder)

    counts = {plr_vacuum: 70, plr_excited: 30}
    mr = MeasurementResults(counts, encoder)

    # plr_excited has THREE on link ((0,0),1) (C_2=4/3) and ONE on all others.
    # So lattice energy = 4/3.
    # Average over shots: (70 * 0 + 30 * 4/3) / 100 = 40/100 = 0.4
    assert mr.get_lattice_electric_energy(average_result=False) == pytest.approx(0.4)

    # With average_result=True: per-link average is computed per-instance then averaged.
    # plr_vacuum: 0/6 = 0. plr_excited: (4/3)/6 = 2/9.
    # Expectation: (70*0 + 30*2/9) / 100 = 60/900 = 1/15
    assert mr.get_lattice_electric_energy(average_result=True) == pytest.approx(1.0 / 15.0)


def test_get_lattice_electric_energy_unphysical_link(encoder_d32_L2_T1):
    """get_lattice_electric_energy can optionally return a KeyError when hitting an unphysical link."""
    encoder = encoder_d32_L2_T1
    # 70 shots of one excited link ((0, 0), 1) in the THREE and one unphysical, 30 shots of an unphysical state on link ((0,0),1)
    plr_one_excited = ParsedLatticeResult(1.5, 2, "100000000011", encoder)
    plr_unphysical = ParsedLatticeResult(1.5, 2, "110000000000", encoder)

    counts = {plr_one_excited: 70, plr_unphysical: 30}
    mr = MeasurementResults(counts, encoder)

    expected_total_energy = (4.0/3) * 0.7
    expected_average_energy = expected_total_energy/5.0 # 5 physical links, one unphysical in this lattice state.

    # No warning as default behavior.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert expected_total_energy == pytest.approx(mr.get_lattice_electric_energy(average_result=False, unphys_mode=None))
        assert expected_average_energy == pytest.approx(mr.get_lattice_electric_energy(average_result=True, unphys_mode=None))
        assert len(w) == 0

    # Warning when flag set to warn, or as default behavior.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mr.get_lattice_electric_energy(average_result=False, unphys_mode='warn')
        mr.get_lattice_electric_energy(average_result=False)
        assert len(w) == 8  # 2 strings in counts dict * 2 warnings per string * trying the same function 2 times
        assert "unphysical" in str(w[0].message).lower()
        
    # Error when flag set to err.
    with pytest.raises(KeyError, match='unphysical'):
        mr.get_lattice_electric_energy(unphys_mode='err')

def test_get_lattice_electric_energy_unmeasured_link(encoder_d32_L2_T1):
    """
    get_lattice_electric_energy can optionally return a KeyError when hitting an unmeasured link.
    Check flags for controlling this behavior.
    """
    encoder = encoder_d32_L2_T1
    partial_excited = ParsedLatticeResult.from_links_and_vertices(
        links_dict={((0, 0), 1): THREE}, encoder=encoder
    )
    vacuum_one_unphys_link = "000000000011"
    mr = MeasurementResults({
        partial_excited: 60,
        vacuum_one_unphys_link: 40
    }, encoder=encoder)
    expected_total_energy = 4.0/3 * 0.6                 # Note that this is not physically well-motivated since the two states are averaged over two numbers of total physical links.
    expected_average_energy = ((4.0/3)/1.0) * 0.6 # 60% chance of one excited link out of 1 total measured excited links.

    # Default: no error
    assert mr.get_lattice_electric_energy() == pytest.approx(expected_average_energy)
    assert mr.get_lattice_electric_energy(average_result=True) == pytest.approx(expected_average_energy)
    assert mr.get_lattice_electric_energy(average_result=False) == pytest.approx(expected_total_energy)

    # Warning when flag set appropriately
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mr.get_lattice_electric_energy(unphys_mode='warn')
        assert len(w) == 8      # partial_excited generates 6 warnings (5 unphys links, 1 overall report), vacuum generates 2 warnings
        assert "unmeasured" in str(w[0].message).lower()
    
    # Error when flag set appropriately
    with pytest.raises(KeyError, match="unmeasured"):
        mr.get_lattice_electric_energy(skip_unmeasured=False)
    with pytest.raises(KeyError, match="unmeasured"):
        mr.get_lattice_electric_energy(unphys_mode='err')


def test_vacuum_persistence_probability(encoder_d32_L2_T1):
    """vacuum_persistence_probability should be the fraction in vacuum state."""
    encoder = encoder_d32_L2_T1
    plr_vacuum = ParsedLatticeResult(1.5, 2, "000000000000", encoder)
    plr_excited = ParsedLatticeResult(1.5, 2, "100000000000", encoder)

    counts = {plr_vacuum: 80, plr_excited: 20}
    mr = MeasurementResults(counts, encoder)

    assert mr.vacuum_persistence_probability() == pytest.approx(0.8)

def test_vacuum_persistence_probability_some_nontrivial_multiplicities(encoder_d2_L2_T1):
    """
    vacuum_persistence_probability should be the fraction in vacuum state,
    with the ability to ignore bad multiplicities if desired.
    """
    encoder = encoder_d2_L2_T1
    plr_vacuum = ParsedLatticeResult(2, 2, "0" + "0000" + "0" + "0000" + "0" + "0000" + "0" + "0000", encoder)
    plr_vacuum_nontrivial_multiplicities = ParsedLatticeResult(2, 2, "1" + "0000" + "0" + "0000" + "0" + "0000" + "0" + "0000", encoder)
    plr_excited = ParsedLatticeResult(2, 2, "0" + "1000" + "0" + "0000" + "0" + "0000" + "0" + "0000", encoder)

    counts = {plr_vacuum: 70, plr_vacuum_nontrivial_multiplicities: 20, plr_excited: 10}
    mr = MeasurementResults(counts, encoder)

    assert mr.vacuum_persistence_probability() == pytest.approx(0.9)
    assert mr.vacuum_persistence_probability(strict_equality=True) == pytest.approx(0.7)


def test_get_transition_probability(encoder_d32_L2_T1):
    """get_transition_probability returns the empirical probability of a state."""
    encoder = encoder_d32_L2_T1
    plr_vacuum = ParsedLatticeResult(1.5, 2, "000000000000", encoder)
    plr_excited = ParsedLatticeResult(1.5, 2, "100100000010", encoder)
    plr_other = ParsedLatticeResult(1.5, 2, "011101110101", encoder)

    counts = {plr_vacuum: 60, plr_excited: 40}
    mr = MeasurementResults(counts, encoder)

    assert mr.get_transition_probability(plr_vacuum) == pytest.approx(0.6)
    assert mr.get_transition_probability(plr_excited) == pytest.approx(0.4)
    assert mr.get_transition_probability(plr_other) == pytest.approx(0.0)


def test_get_transition_probability_with_partial_state(encoder_d32_L2_T1):
    """get_transition_probability should match partial states against full states."""
    encoder = encoder_d32_L2_T1
    plr_vacuum = ParsedLatticeResult(1.5, 2, "000000000000", encoder)
    plr_excited = ParsedLatticeResult(1.5, 2, "100000000111", encoder)

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

    # If we request strict equality, then instead there should be no matches.
    assert mr.get_transition_probability(partial_vacuum, strict_equality=True) == pytest.approx(0.0)

    # Fully-specified partial state matching all 6 links to ONE.
    # Should exactly match vacuum.
    full_vacuum = ParsedLatticeResult.from_links_and_vertices(
        links_dict={addr: ONE for addr in encoder.lattice_def.link_addresses},
        encoder=encoder
    )
    assert mr.get_transition_probability(full_vacuum) == pytest.approx(0.6)

    # If we request strict equality, this time the probability should be
    # unaffected.
    assert mr.get_transition_probability(full_vacuum, strict_equality=True) == pytest.approx(0.6)


def test_measurement_results_empty_counts_raises(encoder_d32_L2_T1):
    """MeasurementResults should raise ValueError for empty counts."""
    with pytest.raises(ValueError):
        MeasurementResults({}, encoder_d32_L2_T1)
