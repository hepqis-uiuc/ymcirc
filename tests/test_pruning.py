import pytest
import numpy as np
from ymcirc.pruning import _parse_state, _preprocess_data, _preprocess_sectors
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
