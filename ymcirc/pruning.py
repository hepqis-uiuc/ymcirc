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


def _preprocess_sectors(ctrl_int, E2_active, active_int,
                       hop_rows, hop_cols, hop_vals):
    """Group states by control-link configuration into sectors.

    Box preserves control links, making H block-diagonal by control config.
    Called once per B-truncation; reused across all g values and MF iterations.

    Returns
    -------
    sectors_size1 : dict with batched arrays for all size-1 sectors
        ctrl_keys: (n1, n_ctrl) int array
        E2: (n1,) float array
        active_int: (n1, n_active_links) int array
    sectors_ge2 : list of dicts, one per sector with >=2 states
        Each dict has: ctrl_key, E2_local, active_int_local, hop_rows/cols/vals
    """
    n_states = len(E2_active)
    n_ctrl_links = ctrl_int.shape[1] if ctrl_int.ndim == 2 else 0
    n_act_links = active_int.shape[1] if active_int.ndim == 2 else 0

    # Map each state to its sector (by control config)
    sector_map = {}
    state_sector = np.empty(n_states, dtype=np.int32)
    state_local_idx = np.empty(n_states, dtype=np.int32)
    sector_keys = []
    sector_states = []

    for i in range(n_states):
        key = tuple(ctrl_int[i])
        if key not in sector_map:
            s_idx = len(sector_keys)
            sector_map[key] = s_idx
            sector_keys.append(key)
            sector_states.append([])
        else:
            s_idx = sector_map[key]
        state_sector[i] = s_idx
        state_local_idx[i] = len(sector_states[s_idx])
        sector_states[s_idx].append(i)

    # Assign hops to sectors and re-index to local indices
    n_hops = len(hop_rows)
    if n_hops > 0:
        hop_sector = state_sector[hop_rows]
        assert np.all(hop_sector == state_sector[hop_cols]), \
            "Hops cross sector boundaries -- box should preserve control configs"
        hop_local_rows = state_local_idx[hop_rows]
        hop_local_cols = state_local_idx[hop_cols]

        # Sort by sector for efficient slicing
        order = np.argsort(hop_sector, kind='stable')
        hop_sector_s = hop_sector[order]
        hop_lr_s = hop_local_rows[order]
        hop_lc_s = hop_local_cols[order]
        hop_v_s = hop_vals[order]

        unique_sectors, starts = np.unique(hop_sector_s, return_index=True)
        ends = np.append(starts[1:], n_hops)
        sector_hop_range = {int(s): (int(starts[i]), int(ends[i]))
                            for i, s in enumerate(unique_sectors)}
    else:
        hop_lr_s = np.array([], dtype=np.int32)
        hop_lc_s = np.array([], dtype=np.int32)
        hop_v_s = np.array([])
        sector_hop_range = {}

    # Separate into size-1 (batch) and size>=2 (loop)
    s1_keys, s1_E2, s1_active = [], [], []
    sge2 = []

    for s_idx in range(len(sector_keys)):
        indices = np.array(sector_states[s_idx], dtype=np.int32)
        if len(indices) == 1:
            s1_keys.append(sector_keys[s_idx])
            s1_E2.append(E2_active[indices[0]])
            s1_active.append(active_int[indices[0]])
        else:
            if s_idx in sector_hop_range:
                a, b = sector_hop_range[s_idx]
                lr, lc, lv = hop_lr_s[a:b], hop_lc_s[a:b], hop_v_s[a:b]
            else:
                lr = np.array([], dtype=np.int32)
                lc = np.array([], dtype=np.int32)
                lv = np.array([])
            sge2.append({
                'ctrl_key': np.array(sector_keys[s_idx], dtype=np.int32),
                'E2_local': E2_active[indices],
                'active_int_local': active_int[indices],
                'hop_rows': lr, 'hop_cols': lc, 'hop_vals': lv,
            })

    sectors_size1 = {
        'ctrl_keys': (np.array(s1_keys, dtype=np.int32) if s1_keys
                      else np.empty((0, n_ctrl_links), dtype=np.int32)),
        'E2': np.array(s1_E2) if s1_E2 else np.empty(0),
        'active_int': (np.array(s1_active, dtype=np.int32) if s1_active
                       else np.empty((0, n_act_links), dtype=np.int32)),
    }

    return sectors_size1, sge2
