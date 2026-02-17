import pytest
import numpy as np
from ymcirc.pruning import _parse_state, _preprocess_data
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
