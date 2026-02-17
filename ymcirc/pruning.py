"""
Runtime Hilbert-space pruning for SU(3) lattice gauge theory.

Provides irrep-level and state-level (sector) pruning via self-consistent
mean-field analysis. Sector pruning exploits the block-diagonal structure
of the Hamiltonian by control-link configuration, sorting sectors by
probability and retaining only those needed to achieve a target accuracy.

Public API:
    meanfield_weights() - run MF, return converged weights
    prune_by_sector_probability() - state-level sector pruning
    prune_by_irrep_importance() - irrep-level importance ordering
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ymcirc.conventions import (
    HAMILTONIAN_BOX_TERMS,
    PHYSICAL_PLAQUETTE_STATES,
)
from ymcirc.electric_helper import gt_pattern_iweight_to_casimir

IrrepWeight = Tuple[int, int, int]
PlaquetteState = tuple


@dataclass
class MFResult:
    """Result of self-consistent mean-field iteration."""
    weights: Dict[IrrepWeight, float]
    E2: float
    gap: float
    n_iter: int


@dataclass
class SectorPruningResult:
    """Result of state-level sector pruning."""
    states: List[PlaquetteState]
    box_terms: Dict[Tuple[PlaquetteState, PlaquetteState], float]
    n_retained: int
    n_total: int
    compression: float
    mf_weights: Dict[IrrepWeight, float]
    sector_probabilities: np.ndarray
    E2_estimate: float


@dataclass
class IrrepPruningResult:
    """Result of irrep-level importance pruning."""
    kept_irreps: List[Tuple[int, int]]
    n_kept: int
    n_total: int
    error: float
    ground_state_weights: Dict[Tuple[int, int], float]


def _parse_state(state: PlaquetteState):
    """Extract (vertices, active_links, control_links) from a PlaquetteState.

    Single point of coupling to the PlaquetteState tuple format.
    """
    return state[0], state[1], state[2]


def _preprocess_data(states, box_terms):
    """Convert ymcirc PHYSICAL_PLAQUETTE_STATES + HAMILTONIAN_BOX_TERMS to arrays.

    Parameters
    ----------
    states : list of PlaquetteState
        From PHYSICAL_PLAQUETTE_STATES[dim][trunc]
    box_terms : dict
        From HAMILTONIAN_BOX_TERMS[dim][trunc], maps (state_f, state_i) -> float

    Returns
    -------
    dict with keys: n_states, E2_active, ctrl_int, active_int,
        unique_irreps, irrep_to_idx, n_ctrl, n_active_links,
        hop_rows, hop_cols, hop_vals, state_index
    """
    n_states = len(states)
    state_index = {}
    E2_active = np.zeros(n_states)
    ctrl_irreps = []
    active_irreps = []

    for i, state in enumerate(states):
        verts, a_links, c_links = _parse_state(state)
        state_index[state] = i
        E2_active[i] = sum(gt_pattern_iweight_to_casimir(R) for R in a_links)
        ctrl_irreps.append(tuple(c_links))
        active_irreps.append(tuple(a_links))

    # Build unique irrep index
    unique_set = set()
    for cr in ctrl_irreps:
        unique_set.update(cr)
    for ar in active_irreps:
        unique_set.update(ar)
    unique_irreps = sorted(unique_set)
    irrep_to_idx = {R: i for i, R in enumerate(unique_irreps)}

    # Convert to integer arrays
    n_ctrl = len(ctrl_irreps[0])
    n_active_links = len(active_irreps[0])
    ctrl_int = np.array([[irrep_to_idx[R] for R in cr] for cr in ctrl_irreps],
                        dtype=np.int32)
    active_int = np.array([[irrep_to_idx[R] for R in ar] for ar in active_irreps],
                          dtype=np.int32)

    # Parse box terms into sparse hopping arrays (box + box†)
    hop_rows = []
    hop_cols = []
    hop_vals = []
    for (state_f, state_i), val in box_terms.items():
        i_f = state_index.get(state_f)
        i_i = state_index.get(state_i)
        if i_f is None or i_i is None:
            continue
        hop_rows.append(i_f)
        hop_cols.append(i_i)
        hop_vals.append(val)
        hop_rows.append(i_i)
        hop_cols.append(i_f)
        hop_vals.append(val)

    return {
        "n_states": n_states,
        "E2_active": E2_active,
        "ctrl_int": ctrl_int,
        "active_int": active_int,
        "unique_irreps": unique_irreps,
        "irrep_to_idx": irrep_to_idx,
        "n_ctrl": n_ctrl,
        "n_active_links": n_active_links,
        "hop_rows": np.array(hop_rows, dtype=np.int32),
        "hop_cols": np.array(hop_cols, dtype=np.int32),
        "hop_vals": np.array(hop_vals, dtype=np.float64),
        "state_index": state_index,
    }
