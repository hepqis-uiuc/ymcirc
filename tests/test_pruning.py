import pytest
import numpy as np
from ymcirc.pruning import _parse_state, _preprocess_data, _preprocess_sectors, _run_meanfield
from ymcirc.conventions import PHYSICAL_PLAQUETTE_STATES, HAMILTONIAN_BOX_TERMS


def test_parse_state_extracts_components():
    """_parse_state decomposes a PlaquetteState tuple."""
    state = ((0, 0, 0, 0),
             ((0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 0)),
             ((0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 0, 0),
              (0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 0, 0)))
    verts, active, control = _parse_state(state)
    assert verts == (0, 0, 0, 0)
    assert len(active) == 4
    assert len(control) == 8


def test_preprocess_data_b8o3():
    """_preprocess_data converts ymcirc data to arrays for MF solver."""
    states = PHYSICAL_PLAQUETTE_STATES["d=2"]["B8o3"]
    box_terms = HAMILTONIAN_BOX_TERMS["d=2"]["B8o3"]
    data = _preprocess_data(states, box_terms)

    assert data["n_states"] == 627
    assert data["E2_active"].shape == (627,)
    assert data["E2_active"].dtype == np.float64
    # Vacuum state (index 1) has E2=0
    assert data["E2_active"][1] == 0.0
    # hop arrays are consistent
    assert len(data["hop_rows"]) == len(data["hop_cols"]) == len(data["hop_vals"])
    # box + box† = 2 × 83 = 166 entries
    assert len(data["hop_rows"]) == 166
    # unique_irreps includes vacuum, 3, 3bar
    assert len(data["unique_irreps"]) == 3


def test_preprocess_sectors_b8o3():
    """Sectors correctly partition states by control config."""
    states = PHYSICAL_PLAQUETTE_STATES["d=2"]["B8o3"]
    box_terms = HAMILTONIAN_BOX_TERMS["d=2"]["B8o3"]
    data = _preprocess_data(states, box_terms)

    from ymcirc.pruning import _preprocess_sectors
    sectors_s1, sectors_ge2 = _preprocess_sectors(
        data["ctrl_int"], data["E2_active"], data["active_int"],
        data["hop_rows"], data["hop_cols"], data["hop_vals"])

    # B8o3: 627 states, 545 sectors total (464 size-1, 81 size>=2)
    n1 = len(sectors_s1['E2'])
    n_ge2 = len(sectors_ge2)
    assert n1 + n_ge2 == 545
    assert n1 == 464
    assert n_ge2 == 81
    # All size>=2 sectors have max size 3
    for sec in sectors_ge2:
        assert len(sec['E2_local']) <= 3
    # Total states accounted for
    total = n1 + sum(len(s['E2_local']) for s in sectors_ge2)
    assert total == 627


def test_run_meanfield_b8o3_strong_coupling():
    """MF at strong coupling: vacuum dominates, E2 near zero."""
    states = PHYSICAL_PLAQUETTE_STATES["d=2"]["B8o3"]
    box_terms = HAMILTONIAN_BOX_TERMS["d=2"]["B8o3"]
    data = _preprocess_data(states, box_terms)
    sectors = _preprocess_sectors(
        data["ctrl_int"], data["E2_active"], data["active_int"],
        data["hop_rows"], data["hop_cols"], data["hop_vals"])

    from ymcirc.pruning import _run_meanfield
    result = _run_meanfield(
        data["n_states"], data["E2_active"], data["ctrl_int"],
        data["active_int"], data["unique_irreps"], data["irrep_to_idx"],
        data["n_ctrl"], data["n_active_links"],
        data["hop_rows"], data["hop_cols"], data["hop_vals"],
        g=2.0, alpha=2.0, _sectors=sectors)

    assert result["converged"]
    # Strong coupling: E2 (4-link total) ~ 0.006
    assert 0.005 < result["E2"] < 0.007
    # Vacuum weight dominates
    vac = (0, 0, 0)
    assert result["weights"][vac] > 0.99
    # Converges in few iterations
    assert result["n_iter"] < 20


def test_run_meanfield_charge_conjugation():
    """MF preserves C-symmetry: w(3) == w(3bar)."""
    states = PHYSICAL_PLAQUETTE_STATES["d=2"]["B8o3"]
    box_terms = HAMILTONIAN_BOX_TERMS["d=2"]["B8o3"]
    data = _preprocess_data(states, box_terms)
    sectors = _preprocess_sectors(
        data["ctrl_int"], data["E2_active"], data["active_int"],
        data["hop_rows"], data["hop_cols"], data["hop_vals"])

    from ymcirc.pruning import _run_meanfield
    result = _run_meanfield(
        data["n_states"], data["E2_active"], data["ctrl_int"],
        data["active_int"], data["unique_irreps"], data["irrep_to_idx"],
        data["n_ctrl"], data["n_active_links"],
        data["hop_rows"], data["hop_cols"], data["hop_vals"],
        g=1.0, alpha=2.0, _sectors=sectors)

    w = result["weights"]
    # B8o3 has 3 irreps: vacuum (0,0,0), fund (1,0,0), antifund (1,1,0)
    # Charge conjugation: w(fund) == w(antifund)
    fund_weights = sorted([v for k, v in w.items() if k != (0, 0, 0)])
    assert len(fund_weights) == 2
    assert abs(fund_weights[0] - fund_weights[1]) < 1e-12


def test_run_meanfield_weights_normalize():
    """MF weights sum to 1."""
    states = PHYSICAL_PLAQUETTE_STATES["d=2"]["B8o3"]
    box_terms = HAMILTONIAN_BOX_TERMS["d=2"]["B8o3"]
    data = _preprocess_data(states, box_terms)

    from ymcirc.pruning import _run_meanfield
    result = _run_meanfield(
        data["n_states"], data["E2_active"], data["ctrl_int"],
        data["active_int"], data["unique_irreps"], data["irrep_to_idx"],
        data["n_ctrl"], data["n_active_links"],
        data["hop_rows"], data["hop_cols"], data["hop_vals"],
        g=1.0, alpha=2.0)

    assert abs(sum(result["weights"].values()) - 1.0) < 1e-10


def test_meanfield_weights_public_api():
    """meanfield_weights returns MFResult with correct fields."""
    from ymcirc.pruning import meanfield_weights, MFResult
    result = meanfield_weights("d=2", "B8o3", g=2.0)

    assert isinstance(result, MFResult)
    assert isinstance(result.weights, dict)
    assert isinstance(result.E2, float)
    assert isinstance(result.n_iter, int)
    assert result.E2 > 0
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_sector_pruning_strong_coupling():
    """At strong coupling, sector pruning achieves massive compression."""
    from ymcirc.pruning import prune_by_sector_probability, SectorPruningResult
    result = prune_by_sector_probability("d=2", "B8o3", g=2.0, delta=0.01)

    assert isinstance(result, SectorPruningResult)
    assert result.n_retained < result.n_total
    assert result.compression > 10  # expect ~100x at g=2
    assert result.n_total == 627
    # Pruned states are a subset of original
    all_states = PHYSICAL_PLAQUETTE_STATES["d=2"]["B8o3"]
    for s in result.states:
        assert s in all_states
    # box_terms keys only reference retained states
    retained_set = set(result.states)
    for (sf, si) in result.box_terms:
        assert sf in retained_set
        assert si in retained_set


def test_sector_pruning_full_inclusion():
    """delta=1.0 retains all states (no pruning)."""
    from ymcirc.pruning import prune_by_sector_probability
    result = prune_by_sector_probability("d=2", "B8o3", g=1.0, delta=1.0)
    assert result.n_retained == result.n_total
    assert result.compression == pytest.approx(1.0)


def test_sector_pruning_format_compatible():
    """Pruned output has correct PlaquetteState tuple format."""
    from ymcirc.pruning import prune_by_sector_probability
    result = prune_by_sector_probability("d=2", "B8o3", g=2.0, delta=0.01)
    for s in result.states:
        verts, active, ctrl = s[0], s[1], s[2]
        assert len(active) == 4
        assert len(ctrl) == 8


def test_irrep_pruning_strong_coupling():
    """At g=2.0, only 3 irreps needed for 1% accuracy."""
    from ymcirc.pruning import prune_by_irrep_importance, IrrepPruningResult
    result = prune_by_irrep_importance("d=2", g=2.0, delta=0.01)

    assert isinstance(result, IrrepPruningResult)
    assert result.n_kept == 3
    assert result.n_total > 100  # lambda_max=35 has 666 irreps
    # Vacuum should have highest weight
    assert result.ground_state_weights[(0, 0)] > 0.5
    assert result.error < 0.01


def test_irrep_pruning_ordering():
    """Importance ordering keeps highest-weight irreps first."""
    from ymcirc.pruning import prune_by_irrep_importance
    result = prune_by_irrep_importance("d=2", g=1.0, delta=0.01)

    # Weights should be monotonically decreasing
    weights = [result.ground_state_weights[pq] for pq in result.kept_irreps]
    for i in range(len(weights) - 1):
        assert weights[i] >= weights[i + 1]


def test_irrep_pruning_weak_coupling():
    """At weak coupling, more irreps are needed."""
    from ymcirc.pruning import prune_by_irrep_importance
    result_strong = prune_by_irrep_importance("d=2", g=2.0, delta=0.01)
    result_weak = prune_by_irrep_importance("d=2", g=0.6, delta=0.01)

    assert result_weak.n_kept > result_strong.n_kept


def test_irrep_pruning_dim_validation():
    """Only d=2 is currently supported."""
    from ymcirc.pruning import prune_by_irrep_importance
    with pytest.raises(ValueError, match="Only d=2 is currently supported"):
        prune_by_irrep_importance("d=3", g=1.0, delta=0.01)


def test_irrep_pruning_delta_variation():
    """Stricter delta requires more irreps."""
    from ymcirc.pruning import prune_by_irrep_importance
    result_loose = prune_by_irrep_importance("d=2", g=1.0, delta=0.1)
    result_strict = prune_by_irrep_importance("d=2", g=1.0, delta=0.001)

    assert result_strict.n_kept > result_loose.n_kept
