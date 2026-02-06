"""
Compact magnetic Hamiltonian via reachable subspace analysis.

Computes the subset of Givens rotations that are reachable from a given
initial state (flux tube + vacuum) under Trotterized evolution, then
optionally filters by coefficient magnitude threshold.

Usage:
    from compact_hamiltonian import build_compact_mag_ham, diagonal_flux_tube_links

    ham = build_compact_mag_ham(size=3, flux_tube_links=diagonal_flux_tube_links(3),
                                encoder=encoder, threshold=0.47)

The returned list is compatible with LatticeCircuitManager.
"""

import json
import hashlib
import os
import warnings
import logging
from collections import defaultdict
from itertools import product
from typing import List, Tuple, Optional, Set

warnings.filterwarnings('ignore')
logging.getLogger('ymcirc').setLevel(logging.WARNING)

from ymcirc.conventions import (
    IRREP_TRUNCATIONS, PHYSICAL_PLAQUETTE_STATES,
    LatticeStateEncoder, load_magnetic_hamiltonian
)
from ymcirc._abstract.lattice_data import LatticeDef

HamiltonianData = List[Tuple[str, str, float]]

ONE = (0, 0, 0)
THREE = (1, 0, 0)
THREE_BAR = (1, 1, 0)

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')


def diagonal_flux_tube_links(size):
    """Return the links forming a diagonal flux tube on a size x size torus.

    The path goes: (0,0) -> (1,0) -> (1,1) -> (2,1) -> ... wrapping around.
    Direction 1 = +x, direction 2 = +y.
    """
    links = []
    x, y = 0, 0
    for _ in range(size):
        links.append(((x, y), 1))
        x = (x + 1) % size
        links.append(((x, y), 2))
        y = (y + 1) % size
    return links


def _normalize_link(v, d, s):
    vx, vy = v
    if d < 0:
        if abs(d) == 1:
            vx = (vx - 1) % s
        else:
            vy = (vy - 1) % s
        d = abs(d)
    return ((vx % s, vy % s), d)


def _get_plaq_links(x, y, s):
    """Get active and control link addresses for plaquette at (x,y)."""
    v1 = (x % s, y % s)
    v2 = ((x + 1) % s, y % s)
    v3 = ((x + 1) % s, (y + 1) % s)
    v4 = (x % s, (y + 1) % s)
    active = [(v1, 1), (v2, 2), (v4, 1), (v1, 2)]
    control_dirs = [
        (v1, [-1, -2]), (v2, [-2, 1]), (v3, [1, 2]), (v4, [2, -1]),
    ]
    control = []
    for v, dirs in control_dirs:
        for d in dirs:
            control.append(_normalize_link(v, d, s))
    return active, control


def _bfs_from(seeds, graph):
    visited = set(seeds)
    frontier = set(seeds)
    while frontier:
        next_frontier = set()
        for s in frontier:
            for t in graph[s]:
                if t not in visited:
                    next_frontier.add(t)
                    visited.add(t)
        frontier = next_frontier
    return visited


def _build_neighbor_groups(size):
    """Build the control-link-to-neighbor-active-position mapping.

    Returns neighbor_groups: dict mapping neighbor plaquette coord to
    list of (ctrl_index, active_position) pairs.
    """
    x, y = 0, 0  # Generic plaquette (translational invariance)
    active_P, control_P = _get_plaq_links(x, y, size)

    ctrl_to_neighbors = {}
    for ci, clink in enumerate(control_P):
        neighbors = []
        for ny in range(size):
            for nx in range(size):
                if (nx, ny) == (x, y):
                    continue
                n_active, _ = _get_plaq_links(nx, ny, size)
                for ai, alink in enumerate(n_active):
                    if alink == clink:
                        neighbors.append(((nx, ny), ai))
        ctrl_to_neighbors[ci] = neighbors

    neighbor_groups = defaultdict(list)
    for ci, neighbors in ctrl_to_neighbors.items():
        for n_plaq, a_pos in neighbors:
            neighbor_groups[n_plaq].append((ci, a_pos))

    return neighbor_groups


def compute_reachable_states(
    size: int,
    flux_tube_links: list,
    encoder: LatticeStateEncoder,
    verbose: bool = True,
) -> Set[str]:
    """Compute the set of plaquette bitstrings reachable from flux tube + vacuum.

    Uses the constrained fixed-point iteration algorithm:
    1. Start with initial states (flux tube views + vacuum), BFS to get neighbors.
    2. Extract active-link patterns, enumerate constrained control patterns.
    3. Cross-reference against Hamiltonian states, BFS from new states.
    4. Repeat until convergence.

    Returns the set of reachable plaquette bitstrings.
    """
    if verbose:
        print("Loading full magnetic Hamiltonian (threshold=0)...")
    mag_ham = load_magnetic_hamiltonian(
        'd=2', 'T1', encoder, mag_hamiltonian_matrix_element_threshold=0
    )
    if verbose:
        print(f"  {len(mag_ham)} Givens rotations total")

    ham_states = set()
    graph = defaultdict(set)
    for bs1, bs2, _ in mag_ham:
        ham_states.add(bs1)
        ham_states.add(bs2)
        graph[bs1].add(bs2)
        graph[bs2].add(bs1)

    # Derive bit-index offsets from encoder properties
    v_bits = encoder._expected_vertex_bit_string_length
    l_bits = encoder._expected_link_bit_string_length
    a_start = 4 * v_bits                  # start of active links within VA
    va_end = 4 * v_bits + 4 * l_bits      # end of vertex+active section
    n_ctrl = (encoder._expected_plaquette_bit_string_length - va_end) // l_bits
    ctrl_placeholder = '?' * l_bits

    # Initial states: flux tube bitstrings at each plaquette + vacuum
    flux_tube_set = set(tuple(l) for l in flux_tube_links)

    vac_bs = encoder.encode_plaquette_state_as_bit_string(
        ((0, 0, 0, 0), (ONE, ONE, ONE, ONE), tuple(ONE for _ in range(n_ctrl)))
    )

    flux_bitstrings = {}
    for fy in range(size):
        for fx in range(size):
            active, control = _get_plaq_links(fx, fy, size)
            a_states = tuple(THREE if tuple(l) in flux_tube_set else ONE for l in active)
            c_states = tuple(THREE if tuple(l) in flux_tube_set else ONE for l in control)
            flux_bitstrings[(fx, fy)] = encoder.encode_plaquette_state_as_bit_string(
                ((0, 0, 0, 0), a_states, c_states))

    initial = set(flux_bitstrings.values()) | {vac_bs}
    current_states = set(initial)

    # BFS one step from initial
    for s in list(initial):
        for t in graph[s]:
            current_states.add(t)

    if verbose:
        print(f"  Initial + 1-step BFS: {len(current_states)} states")

    # Build neighbor groups (translational invariance)
    neighbor_groups = _build_neighbor_groups(size)
    distinct_neighbors = sorted(neighbor_groups.keys())
    n_neighbors = len(distinct_neighbors)

    # Fixed-point iteration
    for iteration in range(1, 20):  # Max 20 iterations (converges in 2)
        va_patterns = set(bs[:va_end] for bs in current_states)
        active_only_patterns = set(va[a_start:] for va in va_patterns)

        if verbose:
            print(f"  Iteration {iteration}: {len(current_states)} states, "
                  f"{len(va_patterns)} VA, {len(active_only_patterns)} active patterns")

        # Enumerate constrained control patterns
        possible_controls = set()
        for combo in product(active_only_patterns, repeat=n_neighbors):
            ctrl_bits = [ctrl_placeholder] * n_ctrl
            for n_idx, n_plaq in enumerate(distinct_neighbors):
                n_pattern = combo[n_idx]
                for ci, ai in neighbor_groups[n_plaq]:
                    ctrl_bits[ci] = n_pattern[l_bits*ai:l_bits*(ai+1)]
            ctrl = ''.join(ctrl_bits)
            if '?' not in ctrl:
                possible_controls.add(ctrl)

        if verbose:
            print(f"    Constrained control patterns: {len(possible_controls)}")

        # Find new states
        new_states = set()
        for va in va_patterns:
            for ctrl in possible_controls:
                candidate = va + ctrl
                if candidate in ham_states and candidate not in current_states:
                    new_states.add(candidate)

        if verbose:
            print(f"    New states in Hamiltonian: {len(new_states)}")

        if not new_states:
            if verbose:
                print(f"  Converged at iteration {iteration}.")
            break

        # BFS from new states
        expanded = current_states | new_states
        frontier = new_states.copy()
        while frontier:
            next_frontier = set()
            for s in frontier:
                for t in graph[s]:
                    if t not in expanded:
                        next_frontier.add(t)
                        expanded.add(t)
            frontier = next_frontier

        if verbose:
            print(f"    After BFS: {len(expanded)} total states")
        current_states = expanded

    return current_states


def _cache_key(size, flux_tube_links, threshold):
    """Generate a cache key for the compact Hamiltonian."""
    data = json.dumps({
        'size': size,
        'flux_tube_links': [list(l) for l in flux_tube_links],
        'threshold': threshold,
    }, sort_keys=True)
    return hashlib.md5(data.encode()).hexdigest()[:12]


def build_compact_mag_ham(
    size: int,
    flux_tube_links: list,
    encoder: LatticeStateEncoder,
    threshold: float = 0.47,
    use_cache: bool = True,
    verbose: bool = True,
) -> HamiltonianData:
    """Build compact magnetic Hamiltonian filtered by reachable subspace + threshold.

    Args:
        size: Lattice size (e.g. 3 for 3x3).
        flux_tube_links: List of link addresses forming the flux tube.
        encoder: LatticeStateEncoder for the lattice.
        threshold: Coefficient magnitude threshold (0 = no threshold).
        use_cache: Whether to use/save JSON cache.
        verbose: Print progress.

    Returns:
        HamiltonianData compatible with LatticeCircuitManager.
    """
    cache_file = None
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        key = _cache_key(size, flux_tube_links, threshold)
        cache_file = os.path.join(CACHE_DIR, f'compact_ham_{size}x{size}_t{threshold}_{key}.json')

        if os.path.exists(cache_file):
            if verbose:
                print(f"Loading cached compact Hamiltonian from {cache_file}")
            with open(cache_file) as f:
                data = json.load(f)
            ham = [(d[0], d[1], d[2]) for d in data['hamiltonian']]
            if verbose:
                print(f"  {len(ham)} Givens rotations, {data['n_reachable_states']} reachable states")
            return ham

    # Compute reachable states
    reachable = compute_reachable_states(size, flux_tube_links, encoder, verbose=verbose)

    # Load full Hamiltonian and filter
    if verbose:
        print(f"Filtering by reachable subspace ({len(reachable)} states) "
              f"and threshold={threshold}...")
    mag_ham = load_magnetic_hamiltonian(
        'd=2', 'T1', encoder, mag_hamiltonian_matrix_element_threshold=threshold
    )
    if verbose:
        print(f"  Full Hamiltonian at threshold={threshold}: {len(mag_ham)} Givens")

    compact_ham = [
        (bs1, bs2, c) for bs1, bs2, c in mag_ham
        if bs1 in reachable and bs2 in reachable
    ]

    if verbose:
        print(f"  Compact Hamiltonian: {len(compact_ham)} Givens "
              f"({100*len(compact_ham)/len(mag_ham):.1f}% of thresholded)")

    # Cache
    if cache_file:
        data = {
            'size': size,
            'threshold': threshold,
            'n_reachable_states': len(reachable),
            'n_givens': len(compact_ham),
            'hamiltonian': [[bs1, bs2, c] for bs1, bs2, c in compact_ham],
        }
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        if verbose:
            print(f"  Cached to {cache_file}")

    return compact_ham


if __name__ == '__main__':
    # Quick test: build compact Hamiltonian for 3x3
    size = 3
    lattice_def = LatticeDef(dimensions=2, size=size, periodic_boundary_conds=True)
    physical_states = PHYSICAL_PLAQUETTE_STATES['d=2']['T1']
    encoder = LatticeStateEncoder(IRREP_TRUNCATIONS['T1'], physical_states, lattice_def)

    flux_links = diagonal_flux_tube_links(size)
    print(f"Flux tube links ({size}x{size}): {flux_links}")

    for thresh in [0.47, 0.33, 0.0]:
        print(f"\n{'='*60}")
        print(f"Threshold = {thresh}")
        print(f"{'='*60}")
        ham = build_compact_mag_ham(size, flux_links, encoder,
                                    threshold=thresh, use_cache=True)
        print(f"Result: {len(ham)} Givens rotations")
