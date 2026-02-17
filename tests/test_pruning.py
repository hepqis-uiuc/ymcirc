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
