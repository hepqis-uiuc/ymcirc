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


def _run_meanfield(n_states, E2_active, ctrl_int, active_int,
                  unique_irreps, irrep_to_idx, n_ctrl, n_active_links,
                  hop_rows, hop_cols, hop_vals,
                  g, alpha=2.0, max_iter=200, tol=1e-10,
                  ctrl_prob_threshold=1e-20, compute_gap=False,
                  _sectors=None):
    """Run sector-averaged self-consistent mean-field for coupling g.

    For each control-link sector, solves the eigenvalue problem independently,
    then averages E2, weights, and gap over sectors weighted by P(c).

    Parameters
    ----------
    n_states : int
    E2_active : ndarray of shape (n_states,)
    ctrl_int : ndarray of shape (n_states, n_ctrl)
    active_int : ndarray of shape (n_states, n_active_links)
    unique_irreps : list of irrep tuples
    irrep_to_idx : dict mapping irrep tuple to integer index
    n_ctrl : int
    n_active_links : int
    hop_rows, hop_cols, hop_vals : sparse hopping matrix (box+box^dag)
    g : coupling constant
    alpha : electric energy coefficient (2 for isolated plaquette)
    max_iter : maximum iterations
    tol : convergence tolerance on weights
    ctrl_prob_threshold : minimum P(c) to include sector
    compute_gap : if True, compute sector-averaged mass gap
    _sectors : precomputed (sectors_size1, sectors_ge2)
    """
    if n_states == 0:
        return {"E2": 0.0, "weights": {}, "n_iter": 0, "converged": True}

    VACUUM = (0, 0, 0)
    n_irreps = len(unique_irreps)

    if _sectors is None:
        sectors_size1, sectors_ge2 = _preprocess_sectors(
            ctrl_int, E2_active, active_int, hop_rows, hop_cols, hop_vals)
    else:
        sectors_size1, sectors_ge2 = _sectors

    e2_coeff = g**2 / alpha
    diag_const = 6.0 / g**2
    hop_coeff = -1.0 / g**2

    weight_arr = np.zeros(n_irreps)
    vac_idx = irrep_to_idx.get(VACUUM, -1)
    if vac_idx >= 0:
        weight_arr[vac_idx] = 1.0

    E2 = 0.0
    n_active_total = 0
    gap_result = np.nan

    converged = False
    for iteration in range(max_iter):
        E2_accum = 0.0
        new_weight_arr = np.zeros(n_irreps)
        total_P = 0.0
        n_active_total = 0
        gap_numerator = 0.0
        gap_denominator = 0.0

        # Batch: size-1 sectors
        n1 = len(sectors_size1['E2'])
        if n1 > 0:
            P_1 = np.prod(weight_arr[sectors_size1['ctrl_keys']], axis=1)
            mask = P_1 > ctrl_prob_threshold
            n_act_1 = int(np.sum(mask))
            if n_act_1 > 0:
                P_active = P_1[mask]
                E2_accum += np.dot(P_active, sectors_size1['E2'][mask])
                ai = sectors_size1['active_int'][mask]
                for link in range(n_active_links):
                    np.add.at(new_weight_arr, ai[:, link], P_active)
                total_P += np.sum(P_active)
                n_active_total += n_act_1

        # Loop: size >= 2 sectors
        for sec in sectors_ge2:
            P_s = np.prod(weight_arr[sec['ctrl_key']])
            if P_s < ctrl_prob_threshold:
                continue

            n_s = len(sec['E2_local'])
            n_active_total += n_s

            H_s = np.diag(e2_coeff * sec['E2_local'] + diag_const)
            lr, lc, lv = sec['hop_rows'], sec['hop_cols'], sec['hop_vals']
            if len(lr) > 0:
                np.add.at(H_s, (lr, lc), hop_coeff * lv)

            evals, evecs = np.linalg.eigh(H_s)
            psi2 = evecs[:, 0] ** 2

            E2_accum += P_s * np.dot(psi2, sec['E2_local'])
            weighted_psi2 = P_s * psi2
            for link in range(n_active_links):
                np.add.at(new_weight_arr, sec['active_int_local'][:, link],
                          weighted_psi2)
            total_P += P_s

            if compute_gap and n_s >= 2:
                gap_numerator += P_s * (evals[1] - evals[0])
                gap_denominator += P_s

        if total_P > 0:
            E2 = E2_accum / total_P
            new_weight_arr /= (total_P * n_active_links)

        max_diff = np.max(np.abs(weight_arr - new_weight_arr))
        weight_arr = new_weight_arr

        if max_diff < tol:
            converged = True
            break

    weights = {unique_irreps[i]: weight_arr[i] for i in range(n_irreps)
               if weight_arr[i] > 0}
    result = {
        "E2": E2,
        "weights": weights,
        "n_iter": iteration + 1,
        "converged": converged,
        "n_active": n_active_total,
        "n_states": n_states,
    }

    if compute_gap:
        if gap_denominator > 0:
            gap_result = gap_numerator / gap_denominator
            result["gap"] = gap_result
            result["xi"] = 1.0 / gap_result if gap_result > 0 else np.inf
        else:
            result["gap"] = np.nan
            result["xi"] = np.nan

    return result


def meanfield_weights(dim: str, trunc: str, g: float, alpha: float = 2.0) -> MFResult:
    """Run self-consistent mean-field and return converged weights.

    Parameters
    ----------
    dim : str
        Dimension string, e.g. "d=2"
    trunc : str
        Truncation label, e.g. "B8o3", "B20o3"
    g : float
        Coupling constant
    alpha : float
        Electric energy coefficient (default 2.0)

    Returns
    -------
    MFResult
        Contains weights, E2, gap, n_iter
    """
    states = PHYSICAL_PLAQUETTE_STATES[dim][trunc]
    box_terms = HAMILTONIAN_BOX_TERMS[dim][trunc]
    data = _preprocess_data(states, box_terms)
    raw = _run_meanfield(
        data["n_states"], data["E2_active"], data["ctrl_int"],
        data["active_int"], data["unique_irreps"], data["irrep_to_idx"],
        data["n_ctrl"], data["n_active_links"],
        data["hop_rows"], data["hop_cols"], data["hop_vals"],
        g, alpha=alpha, compute_gap=True)
    return MFResult(
        weights=raw["weights"],
        E2=raw["E2"],
        gap=raw.get("gap", float("nan")),
        n_iter=raw["n_iter"],
    )


def prune_by_sector_probability(dim: str, trunc: str, g: float,
                                 delta: float = 0.01,
                                 alpha: float = 2.0) -> SectorPruningResult:
    """Prune Hilbert space by sector probability.

    Runs self-consistent mean-field, then sorts sectors by probability P(c)
    and retains only the most probable sectors until the relative error in
    <E^2> is below delta.

    Parameters
    ----------
    dim : str
        Dimension string, e.g. "d=2"
    trunc : str
        Truncation label, e.g. "B8o3", "B20o3"
    g : float
        Coupling constant
    delta : float
        Maximum relative error in <E^2> (default 0.01 = 1%)
    alpha : float
        Electric energy coefficient (default 2.0)

    Returns
    -------
    SectorPruningResult
        Contains pruned states, box_terms, compression factor, and diagnostics
    """
    states_list = PHYSICAL_PLAQUETTE_STATES[dim][trunc]
    box_terms_dict = HAMILTONIAN_BOX_TERMS[dim][trunc]
    data = _preprocess_data(states_list, box_terms_dict)
    sectors = _preprocess_sectors(
        data["ctrl_int"], data["E2_active"], data["active_int"],
        data["hop_rows"], data["hop_cols"], data["hop_vals"])
    sectors_size1, sectors_ge2 = sectors

    # Run full MF
    raw = _run_meanfield(
        data["n_states"], data["E2_active"], data["ctrl_int"],
        data["active_int"], data["unique_irreps"], data["irrep_to_idx"],
        data["n_ctrl"], data["n_active_links"],
        data["hop_rows"], data["hop_cols"], data["hop_vals"],
        g, alpha=alpha, _sectors=sectors)

    E2_ref = raw["E2"]
    weight_arr = np.zeros(len(data["unique_irreps"]))
    for R, w in raw["weights"].items():
        weight_arr[data["irrep_to_idx"][R]] = w

    # Pre-diag all sectors, compute P(c) and E2_c
    e2_coeff = g**2 / alpha
    diag_const = 6.0 / g**2
    hop_coeff = -1.0 / g**2

    P_list = []
    E2_list = []
    # Track which global state indices belong to each sector
    indices_list = []  # list of lists of global state indices

    # Rebuild sector→state mapping from ctrl_int
    sector_map = {}
    for i in range(data["n_states"]):
        key = tuple(data["ctrl_int"][i])
        if key not in sector_map:
            sector_map[key] = []
        sector_map[key].append(i)

    # Size-1 sectors
    n1 = len(sectors_size1['E2'])
    if n1 > 0:
        for j in range(n1):
            P_c = float(np.prod(weight_arr[sectors_size1['ctrl_keys'][j]]))
            P_list.append(P_c)
            E2_list.append(float(sectors_size1['E2'][j]))
            key = tuple(sectors_size1['ctrl_keys'][j])
            indices_list.append(sector_map[key])

    # Size>=2 sectors
    for sec in sectors_ge2:
        P_c = float(np.prod(weight_arr[sec['ctrl_key']]))
        n_s = len(sec['E2_local'])
        H_s = np.diag(e2_coeff * sec['E2_local'] + diag_const)
        lr, lc, lv = sec['hop_rows'], sec['hop_cols'], sec['hop_vals']
        if len(lr) > 0:
            np.add.at(H_s, (lr, lc), hop_coeff * lv)
        evals, evecs = np.linalg.eigh(H_s)
        psi2 = evecs[:, 0]**2
        E2_c = float(np.dot(psi2, sec['E2_local']))
        P_list.append(P_c)
        E2_list.append(E2_c)
        key = tuple(sec['ctrl_key'])
        indices_list.append(sector_map[key])

    # Sort by P(c) descending, accumulate
    P_arr = np.array(P_list)
    E2_arr = np.array(E2_list)
    order = np.argsort(P_arr)[::-1]
    P_arr = P_arr[order]
    E2_arr = E2_arr[order]
    indices_ordered = [indices_list[o] for o in order]

    cum_PE2 = np.cumsum(P_arr * E2_arr)
    cum_P = np.cumsum(P_arr)
    n_sectors = len(P_arr)

    # Find K where error < delta
    if E2_ref > 0:
        E2_topK = cum_PE2 / cum_P
        errors = np.abs(E2_topK - E2_ref) / E2_ref
        # Find smallest K where error < delta
        passing = np.where(errors <= delta)[0]
        if len(passing) > 0:
            K = passing[0] + 1  # 1-indexed
        else:
            K = n_sectors  # keep all
    else:
        K = n_sectors

    # If delta >= 1.0, keep everything
    if delta >= 1.0:
        K = n_sectors

    # Collect retained state indices
    retained_indices = set()
    for k in range(K):
        retained_indices.update(indices_ordered[k])
    retained_indices = sorted(retained_indices)

    # Build pruned states list and box_terms dict
    retained_set = set(retained_indices)
    pruned_states = [states_list[i] for i in retained_indices]
    pruned_box_terms = {}
    for (sf, si), val in box_terms_dict.items():
        i_f = data["state_index"].get(sf)
        i_i = data["state_index"].get(si)
        if i_f is not None and i_i is not None:
            if i_f in retained_set and i_i in retained_set:
                pruned_box_terms[(sf, si)] = val

    n_retained = len(pruned_states)
    n_total = data["n_states"]

    return SectorPruningResult(
        states=pruned_states,
        box_terms=pruned_box_terms,
        n_retained=n_retained,
        n_total=n_total,
        compression=n_total / n_retained if n_retained > 0 else float("inf"),
        mf_weights=raw["weights"],
        sector_probabilities=P_arr[:K],
        E2_estimate=float(cum_PE2[K-1] / cum_P[K-1]) if K > 0 else 0.0,
    )


def _casimir_pq(p: int, q: int) -> float:
    """Compute quadratic Casimir C₂(p,q) for SU(3) irrep.

    Parameters
    ----------
    p, q : int
        Dynkin labels

    Returns
    -------
    float
        C₂(p,q) = (p² + q² + pq + 3p + 3q) / 3
    """
    return (p**2 + q**2 + p*q + 3*p + 3*q) / 3.0


def _dim_pq(p: int, q: int) -> int:
    """Compute dimension of SU(3) irrep (p,q).

    Parameters
    ----------
    p, q : int
        Dynkin labels

    Returns
    -------
    int
        Dimension d(p,q) = (p+1)(q+1)(p+q+2)/2
    """
    return (p + 1) * (q + 1) * (p + q + 2) // 2


def _build_irrep_hamiltonian(g: float, irrep_list: List[Tuple[int, int]],
                            alpha: float = 2.0) -> np.ndarray:
    """Build single-plaquette Hamiltonian in character basis.

    Constructs H for the isolated single plaquette in the gauge-fixed
    (class function) basis. The Hamiltonian is:

        H = (g²/α) C₂(R) δ_{RR'} - (1/g²) [Box + Box†]_{RR'}

    where Box connects (p,q) via tensor product with (1,0):
        (p,q) ⊗ (1,0) → (p+1,q) + (p-1,q+1) + (p,q-1)

    and Box† uses (0,1):
        (p,q) ⊗ (0,1) → (p,q+1) + (p+1,q-1) + (p-1,q)

    Parameters
    ----------
    g : float
        Coupling constant
    irrep_list : list of (p, q) tuples
        Dynkin labels of irreps to include
    alpha : float
        Electric coefficient (2.0 for isolated plaquette)

    Returns
    -------
    H : ndarray of shape (N, N)
        Dense Hamiltonian matrix
    """
    N = len(irrep_list)
    irrep_idx = {pq: i for i, pq in enumerate(irrep_list)}

    # Diagonal: electric energy + constant shift
    H = np.zeros((N, N))
    for i, (p, q) in enumerate(irrep_list):
        H[i, i] = alpha * g**2 * _casimir_pq(p, q) + 6.0 / g**2

    # Off-diagonal: magnetic (Box + Box†)
    # Box: (p,q) ⊗ (1,0)
    for i, (p, q) in enumerate(irrep_list):
        neighbors = [(p + 1, q), (p, q + 1)]
        if p >= 1:
            neighbors += [(p - 1, q + 1), (p - 1, q)]
        if q >= 1:
            neighbors += [(p, q - 1), (p + 1, q - 1)]

        for nb in neighbors:
            j = irrep_idx.get(nb)
            if j is not None:
                H[i, j] += -1.0 / g**2

    return H


def prune_by_irrep_importance(dim: str, g: float, delta: float = 0.01,
                               alpha: float = 2.0,
                               lambda_max: int = 35) -> IrrepPruningResult:
    """Prune irrep basis by ground-state importance ordering.

    MF-independent: uses exact isolated single-plaquette eigenstates.
    Builds H in the character basis up to Λ_max, diagonalizes, ranks
    irreps by ground-state weight |c_{pq}|², keeps top K until error < delta.

    Algorithm:
    1. Generate all SU(3) irreps (p,q) up to Λ_max, ordered by Casimir
    2. Build full Hamiltonian H in character basis
    3. Diagonalize H, extract ground state ψ₀
    4. Compute ground-state weight w_i = |ψ₀[i]|² for each irrep
    5. Sort irreps by descending weight (importance ordering)
    6. Find smallest K where |E₀(K) - E₀(all)| / |E₀(all)| < delta
    7. Return kept irreps and diagnostics

    Parameters
    ----------
    dim : str
        Dimension string (currently only "d=2" supported)
    g : float
        Coupling constant
    delta : float
        Maximum relative error in ground-state energy
    alpha : float
        Electric energy coefficient (2.0 for isolated plaquette)
    lambda_max : int
        Maximum Λ shell to include (Λ = p + q)

    Returns
    -------
    IrrepPruningResult
        Contains:
        - kept_irreps: list of (p,q) tuples in importance order
        - n_kept: number of kept irreps
        - n_total: total number of irreps at lambda_max
        - error: actual relative error for n_kept irreps
        - ground_state_weights: dict mapping (p,q) to |ψ₀|²
    """
    if dim != "d=2":
        raise ValueError(f"Only d=2 is currently supported, got {dim}")

    # Generate all irreps up to lambda_max, ordered by shell
    all_irreps = []
    for shell in range(lambda_max + 1):
        for p in range(shell + 1):
            q = shell - p
            all_irreps.append((p, q))

    n_total = len(all_irreps)

    # Build full Hamiltonian and diagonalize
    H_full = _build_irrep_hamiltonian(g, all_irreps, alpha)
    evals_full, evecs_full = np.linalg.eigh(H_full)
    psi_ref = evecs_full[:, 0]
    E0_ref = evals_full[0]

    # Compute ground-state weights
    weights_arr = psi_ref**2
    ground_state_weights = {pq: float(weights_arr[i])
                            for i, pq in enumerate(all_irreps)}

    # Compute reference E2 (electric energy)
    casimirs_full = np.array([_casimir_pq(p, q) for p, q in all_irreps])
    E2_ref = float(np.dot(weights_arr, casimirs_full))

    # Sort by importance (descending weight)
    irreps_by_importance = sorted(all_irreps, key=lambda pq: -ground_state_weights[pq])

    # Sweep K and find smallest K where error < delta
    # Error is defined relative to E2 (electric energy), not E0
    n_kept = n_total  # default: keep all
    error = 0.0

    for K in range(1, n_total + 1):
        subset = irreps_by_importance[:K]
        casimirs_K = np.array([_casimir_pq(p, q) for p, q in subset])

        H_K = _build_irrep_hamiltonian(g, subset, alpha)
        evals_K, evecs_K = np.linalg.eigh(H_K)
        psi_K = evecs_K[:, 0]
        E2_K = float(np.dot(psi_K**2, casimirs_K))

        err = abs(E2_K - E2_ref) / E2_ref if E2_ref > 0 else 0.0

        if err < delta:
            n_kept = K
            error = err
            break

    # Prepare result
    kept_irreps = irreps_by_importance[:n_kept]

    return IrrepPruningResult(
        kept_irreps=kept_irreps,
        n_kept=n_kept,
        n_total=n_total,
        error=error,
        ground_state_weights=ground_state_weights,
    )
