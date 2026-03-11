# d=3, PBC Feature Planning

## Summary

This document describes the changes required to support three-dimensional, periodic, cubic lattices (d=3, PBC) in ymcirc, based on a review of the existing codebase.

**Good news**: The abstract base classes in `lattice_data.py` and the concrete implementations in `lattice_registers.py`, `parsed_lattice_result.py`, `measurement_results.py`, and `electric_helper.py` already handle d=3 generically. The `LatticeDef`, `LatticeData`, and `Plaquette` classes support arbitrary integer dimensions >= 2 for periodic cubic lattices. No changes are needed in those files.

**The work concentrates in two files**: `conventions.py` (data registration, straightforward) and `circuit.py` (LatticeCircuitManager logic, more involved). A minor test update is also needed.

---

## 1. Data File Registration in `conventions.py`

**Difficulty**: Low.

The B3 d=3 data files already exist:
- `ymcirc/_ymcirc_data/plaquette-states/B3_dim(3)_cube_PBC_plaquette_states.json.gz`
- `ymcirc/_ymcirc_data/magnetic-hamiltonian-box-term-matrix-elements/B3_dim(3)_cube_PBC_magnetic_hamiltonian.json.gz`

### Data file characteristics (from inspection)

| Property | Value |
|---|---|
| Unique i-weights | `{(0,0,0), (1,0,0), (1,1,0)}` = ONE, THREE, THREE_BAR (same as T1) |
| Max multiplicity index | 0 (trivial; no vertex qubits needed) |
| Plaquette states | 54,035 |
| Hamiltonian entries | 707 |
| Planes | (1,2), (1,3), (2,3) |
| Unique signatures | 1: all vertices fully connected `((1,2,3,-1,-2,-3), ...)` |
| Controls per vertex | 4 (= 2*(d-1)) → 16 total per plaquette |
| FORDER | [1, 2, 3, -1, -2, -3] (default) |
| dim string | "d=3" |
| trunc string | "B3" |

### Changes needed

Add entries to the file path dicts (`conventions.py:167-183`):

```python
_HAMILTONIAN_DATA_FILE_PATHS = {
    ...,
    "d=3": {
        "B3": _HAMILTONIAN_DATA_DIR / "B3_dim(3)_cube_PBC_magnetic_hamiltonian.json.gz"
    }
}
_PLAQUETTE_STATES_DATA_FILE_PATHS = {
    ...,
    "d=3": {
        "B3": _PLAQUETTE_STATES_DATA_DIR / "B3_dim(3)_cube_PBC_plaquette_states.json.gz"
    }
}
```

The lazy dicts `PHYSICAL_PLAQUETTE_STATES` and `HAMILTONIAN_BOX_TERMS` are constructed via dict comprehension over the file path dicts, so they will automatically pick up the new entries.

Although B3 currently uses the same irreps as T1, a dedicated `"B3"` entry should be added to `IRREP_TRUNCATIONS` with its own link bitmap. Future B truncations will not always have the same irrep content as a T truncation, so establishing the B-series namespace now keeps things forward-compatible. A comment noting that B represents a distinct form of irrep truncation from pyclebsch (see that codebase for details) would be appropriate.

```python
IRREP_TRUNCATIONS = {
    ...,
    "B3": {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "01"
    }
}
```

**No changes needed to `LatticeStateEncoder`** — it accepts arbitrary link bitmaps and lattice definitions, and the d=3 LatticeDef is already fully supported by the base class.

---

## 2. `LatticeCircuitManager` in `circuit.py`

This is the main area of work. There are **six** `NotImplementedError` sites in `circuit.py` that block d=3, plus the small-periodic-lattice logic needs new d=3 handling.

### 2a. `__init__` dimension match (line 77-85)

**Difficulty**: Low.

Add `case 3:` to the dimension match statement:

```python
case 3:
    lattice_size = lattice_encoder.lattice_def.shape[0]
    if lattice_size != lattice_encoder.lattice_def.shape[1] or lattice_size != lattice_encoder.lattice_def.shape[2]:
        raise NotImplementedError("Non-cubic dim 3 lattices not yet supported.")
```

### 2b. `__init__` cache control link dirs (lines 88-93)

**Difficulty**: Medium.

Currently, d=2 caches `_cached_ctrl_dirs_d2_small_and_periodic` for the single plane (1,2). For d=3, control link directions vary by plane (see Section 3 below), so we need per-plane caching.

Proposed: create `_cached_ctrl_dirs_d3_small_and_periodic: Dict[Plane, tuple]` mapping each of the 3 planes to its control link dirs. Each entry is obtained by constructing a temporary plaquette for that plane and reading `control_link_dirs_per_vertex`.

### 2c. `__init__` small-periodic Hamiltonian filtering (lines 95-118) — **MAIN CHALLENGE**

**Difficulty**: High. This is the most architecturally significant change.

#### The problem

On a size-2 periodic lattice, some control links on different plaquette vertices are the same physical link. The current code filters the Hamiltonian by (1) discarding states where shared controls have inconsistent values, and (2) trimming duplicate control links from bitstring encodings. In d=2 (single plane), this is done once per state pair.

In d=3, **the sharing pattern depends on the plaquette plane**. For a plaquette in plane (e1, e2), the shared controls are the in-plane direction pairs:
- v1[dir -e1] ↔ v2[dir +e1]
- v1[dir -e2] ↔ v4[dir +e2]
- v2[dir -e2] ↔ v3[dir +e2]
- v3[dir +e1] ↔ v4[dir -e1]

The out-of-plane controls (±e3, where e3 is the direction NOT in the plane) are never shared.

Since the plaquette state bitstring encoding is plane-agnostic (controls are in FORDER order regardless of which directions are "in-plane"), the FORDER indices of shared controls differ between planes. A state that's consistent for one plane may be inconsistent for another.

Furthermore, after trimming duplicate controls, the resulting bitstring length and content differ by plane. This means a single trimmed bitstring key can't serve all planes.

#### Proposed solution

For d=3 small periodic lattices, restructure the filtering to be **per-plane**:

1. In `__init__`, do **not** perform the filtering for d=3 (leave `self._mag_hamiltonian` unmodified).
2. Add a new private method `_resolve_and_trim_for_small_periodic` that:
   - Takes a plane and signature
   - Resolves the Hamiltonian for that plane/signature (same as `_resolve_hamiltonian_for_plaquette`)
   - Decodes each entry's bitstrings
   - Checks consistency for the given plane
   - Trims duplicate controls for the given plane
   - Re-encodes the trimmed bitstrings
   - Returns a flat `List[Tuple[str, str, float]]`
3. In `apply_magnetic_trotter_step`, for d=3 small periodic lattices, call this new method instead of `_resolve_hamiltonian_for_plaquette`.
4. Cache results per `(plane, signature)` key for efficiency.

**Trade-off**: This adds a new code path for d=3, but avoids restructuring the existing d=2 and d=3/2 logic.

#### Alternative proposed solution: unified per-plane filtering for all dimensions

A cleaner but higher-risk approach would be to refactor the existing `__init__` filtering to be per-plane for *all* dimensions, eliminating the separate d=3 code path entirely. This would involve:

1. **Replace the current plane-agnostic filtering loop** (lines 95-118) with a per-plane loop. For each `(bs1, bs2) -> MatrixElementValue` entry, iterate over each plane present in the MatrixElementValue. For each plane, decode bs1/bs2, check consistency for that plane, and trim for that plane.

2. **Restructure `self._mag_hamiltonian`** from `Dict[Tuple[str, str], MatrixElementValue]` to a per-plane resolved format, e.g. `Dict[Plane, Dict[Tuple[str, str], float]]`, where the bitstring keys are already trimmed per-plane. This is essentially the "pre-resolved" Hamiltonian.

3. **Update `_resolve_hamiltonian_for_plaquette`** (and its callers in `apply_magnetic_trotter_step`) to read from the new per-plane structure instead of doing plane/signature lookups on the original nested dict.

4. **Unify `_plaquette_state_has_inconsistent_controls` and `_discard_duplicate_controls_from_plaquette_state`** by always passing a plane parameter. For d=3/2 and d=2 (which have only one plane), the plane argument is always (1, 2) and can be supplied automatically. The `case 1.5` and `case 2` branches would be rewritten to use the same direction-based lookup logic as `case 3`, just with the fixed plane (1, 2).

5. **Collapse the d=2 `_cached_ctrl_dirs_d2_small_and_periodic`** and the proposed d=3 `_cached_ctrl_dirs_d3_small_and_periodic` into a single `_cached_ctrl_dirs_small_and_periodic: Dict[Plane, tuple]` that works for all dimensions.

6. **Merge the qubit-stitching skip logic** (Section 2d) into a single dimension-agnostic branch that uses the per-plane ctrl dir cache, instead of having separate `case 1.5`, `case 2`, `case 3` blocks.

**Benefits**: Eliminates dimension-specific branching in multiple methods, makes adding future dimensions (d=4, etc.) trivial, and reduces overall code complexity.

**Risks**: Touches every small-periodic code path that currently works for d=3/2 and d=2. Requires thorough regression testing. The per-plane Hamiltonian restructure changes the data contract between `__init__` and `apply_magnetic_trotter_step`, so any downstream code that reads `self._mag_hamiltonian` directly would need updating.

#### Detailed sharing analysis for d=3, size-2 PBC

For plane (e1, e2) with out-of-plane direction e3:

| Vertex | Active dirs | Control dirs (all FORDER-sorted) | Shared? |
|---|---|---|---|
| v1 | {+e1, +e2} | {e3, -e3, -e1, -e2} (FORDER-dependent order) | -e1 shared with v2[+e1], -e2 shared with v4[+e2] |
| v2 | {-e1, +e2} | {+e1, e3, -e3, -e2} | +e1 shared with v1[-e1], -e2 shared with v3[+e2] |
| v3 | {-e1, -e2} | {+e1, +e2, e3, -e3} | +e1 shared with v4[-e1], +e2 shared with v2[-e2] |
| v4 | {+e1, -e2} | {+e2, e3, -e3, -e1} | +e2 shared with v1[-e2], -e1 shared with v3[+e1] |

After trimming (keeping first occurrence of each shared link):
- v1: keep all 4 → 4 controls
- v2: drop +e1 (dups v1[-e1]), keep 3 → {-e2, e3, -e3}
- v3: drop +e2 (dups v2[-e2]), keep 3 → {+e1, e3, -e3}
- v4: drop +e2 (dups v1[-e2]) and -e1 (dups v3[+e1]), keep 2 → {e3, -e3}

**Total after trimming**: 4 + 3 + 3 + 2 = **12** unique physical controls (vs 16 total slots).

This generalizes the d=2 pattern where it's 2 + 1 + 1 + 0 = 4 unique (vs 8 slots).

### 2d. `apply_magnetic_trotter_step` qubit stitching skip logic (lines 614-629)

**Difficulty**: Medium.

Add `case 3:` to the dimension match in the qubit stitching loop. The skip pattern follows the trimming pattern above:

- v1 (idx 0): no skips
- v2 (idx 1): skip the control at FORDER index of +e1
- v3 (idx 2): skip the control at FORDER index of +e2
- v4 (idx 3): skip controls at FORDER indices of +e2 and -e1

The skip indices must be computed from the per-plane cached ctrl dirs (Section 2b) and the current plaquette's plane, which is available in the loop context.

Unlike d=2 (where skip indices are pre-computed once since there's only one plane), for d=3 the skip indices need to be computed per-plaquette or looked up per-plane from the cache. Since there are only 3 planes, a per-plane lookup is efficient.

### 2e. `_plaquette_state_has_inconsistent_controls` (lines 878-922)

**Difficulty**: Medium.

Add `case 3:` that checks the 4 in-plane sharing pairs for a given plane.

**Interface change needed**: This method currently takes only a `PlaquetteState` and uses `self._encoder.lattice_def.dim` to decide what to check. For d=3, it also needs to know the **plane**:

- **Add a `plane` parameter** (default None for backward compat). For d=3, raise if plane is None.

The consistency check uses the per-plane cached ctrl dirs to look up FORDER indices:
```python
case 3:
    ctrl_dirs = self._cached_ctrl_dirs_d3_small_and_periodic[plane]
    e1, e2 = plane
    plaquette_state_has_inconsistent_controls = (
        (c_links[0][ctrl_dirs[0].index(-e1)] != c_links[1][ctrl_dirs[1].index(e1)]) or
        (c_links[0][ctrl_dirs[0].index(-e2)] != c_links[3][ctrl_dirs[3].index(e2)]) or
        (c_links[1][ctrl_dirs[1].index(-e2)] != c_links[2][ctrl_dirs[2].index(e2)]) or
        (c_links[2][ctrl_dirs[2].index(e1)] != c_links[3][ctrl_dirs[3].index(-e1)])
    )
```

Note: this is structurally identical to the d=2 case (same 4 sharing pairs), just with plane-dependent FORDER indices.

### 2f. `_discard_duplicate_controls_from_plaquette_state` (lines 924-965)

**Difficulty**: Medium.

Add `case 3:` that trims in-plane direction duplicates for a given plane. Same interface change as 2e (add optional `plane` parameter).

```python
case 3:
    ctrl_dirs = self._cached_ctrl_dirs_d3_small_and_periodic[plane]
    e1, e2 = plane
    physical_c_links = (
        c_links[0],  # v1: keep all
        # v2: drop +e1 (index in v2's ctrl dirs)
        tuple(c for i, c in enumerate(c_links[1]) if i != ctrl_dirs[1].index(e1)),
        # v3: drop +e2
        tuple(c for i, c in enumerate(c_links[2]) if i != ctrl_dirs[2].index(e2)),
        # v4: drop +e2 and -e1
        tuple(c for i, c in enumerate(c_links[3])
              if i != ctrl_dirs[3].index(e2) and i != ctrl_dirs[3].index(-e1)),
    )
```

---

## 3. Test updates

### `tests/test_conventions.py` (line 50-51)

Add the d=3 case to `test_physical_plaquette_state_data_are_valid`:

```python
case "d=3":
    expected_num_c_links = 16  # 4 controls/vertex * 4 vertices
```

### New test coverage to consider

- Test that d=3, B3 plaquette states load and parse correctly
- Test that d=3, B3 Hamiltonian loads correctly with 3 planes
- Test `LatticeCircuitManager` construction for d=3, size >= 3 (no small-lattice logic)
- Test `LatticeCircuitManager` construction for d=3, size=2 (small periodic lattice logic)
- Test consistency checking and control trimming for each plane independently

---

## 4. Files that do NOT need changes

| File | Reason |
|---|---|
| `_abstract/lattice_data.py` | Generic for d >= 2 periodic cubic. `LatticeDef`, `LatticeData`, `Plaquette`, `get_plaquettes`, `get_traversal_order`, `n_plaquettes`, `n_control_links_per_plaquette`, `add_unit_vector_to_vertex_vector` all work for d=3. |
| `lattice_registers.py` | `get_vertex`, `get_link`, `from_lattice_state_encoder` handle d=3 cubic periodic via generic paths. |
| `parsed_lattice_result.py` | Fully generic; parsing/decoding works for any dim. |
| `measurement_results.py` | `NotImplementedError` at line 40 is for non-hypercubic tuple-valued shapes (not relevant for d=3 cubic). |
| `electric_helper.py` | `NotImplementedError` at line 87 is for non-hypercubic lattices (not relevant). |
| `givens.py` | No dimension-specific logic. |
| `utilities.py` | No changes needed. |

---

## 5. Implementation order

Recommended order of implementation:

1. **`conventions.py` data registration** — unblocks everything else.
2. **`circuit.py` `__init__` dimension match** (2a) — minimal, allows LatticeCircuitManager construction for d=3 size >= 3.
3. **`test_conventions.py`** — validate data loading works.
4. **`circuit.py` per-plane ctrl dirs cache** (2b) — prerequisite for small-lattice logic.
5. **`circuit.py` consistency/discard methods** (2e, 2f) — extend with plane parameter.
6. **`circuit.py` `_strip_redundant_controls_if_small_and_periodic_lattice`** (2g) — extend with plane awareness.
7. **`circuit.py` small-periodic filtering** (2c) — main architectural work.
8. **`circuit.py` qubit stitching skip logic** (2d) — depends on per-plane caching.
9. **Integration tests** — end-to-end circuit construction for d=3 lattices.

---

## 6. Outstanding considerations

### 2g. `_strip_redundant_controls_if_small_and_periodic_lattice` (lines 711-742)

**Difficulty**: Medium.

This helper strips redundant controls from `physical_states_for_control_pruning`. It calls `_plaquette_state_has_inconsistent_controls` and `_discard_duplicate_controls_from_plaquette_state` internally, so it inherits their plane-awareness requirements for d=3.

For d=3 small periodic lattices, this method needs to either:
- Accept a `plane` parameter and pass it through to the consistency/discard methods, or
- Perform the stripping per-plane internally and return a per-plane result (e.g. `Dict[Plane, Set[str]]`).

Since `physical_states_for_control_pruning` is consumed downstream by `_build_mag_evol_circuit` (which operates on already-resolved, plane-specific data), the per-plane approach is more natural: each plane's rotation circuit would use its own stripped physical state set.

### Performance

The d=3 B3 data has 54,035 plaquette states and 707 Hamiltonian entries. A size-2 d=3 periodic lattice has 24 plaquettes (8 vertices × 3 planes). The resolved Hamiltonian is cached per (plane, signature), and since there's only 1 unique signature, there are at most 3 cache entries. Performance should be manageable.

### Data stored as strings

The B3 data files store plaquette states and Hamiltonian keys as strings, consistent with existing data files. The `json_loader` utility already applies `ast.literal_eval` during loading, so no special handling is needed.
