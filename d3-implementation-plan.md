# d=3 PBC Implementation Plan

This plan synthesizes `d3-feature-planning.md` and `d3-feature-plan-audit.md` into a single, concrete implementation specification. For the central architectural question (Section 2c in the original plan), this plan commits to the **unified per-plane approach** — restructuring small-periodic filtering to be per-plane for all dimensions, not just d=3.

---

## Design Decision: Unified Per-Plane Filtering

The original plan presented two options for handling the small-periodic Hamiltonian filtering in d=3:

1. **Proposed solution**: Leave `self._mag_hamiltonian` unmodified for d=3; add a new d=3-specific method called from `apply_magnetic_trotter_step`.
2. **Alternative solution**: Refactor filtering to be per-plane for *all* dimensions, eliminating dimension-specific branching.

**This plan adopts option 2.** The rationale:

- Eliminates dimension-specific `case` branches in `_plaquette_state_has_inconsistent_controls`, `_discard_duplicate_controls_from_plaquette_state`, `_strip_redundant_controls_if_small_and_periodic_lattice`, and the qubit-stitching skip logic.
- Makes future dimensions (d=4+) trivial to add.
- Naturally extends to non-periodic and rectangular lattices (where sharing patterns vary by both plane and signature).
- Avoids the correctness risk identified by audit item 2c.2 (mismatch between untrimmed `physical_states_for_control_pruning` and trimmed rotation bitstrings).

**Risks** (from audit): Touches every small-periodic code path that currently works for d=3/2 and d=2. Requires thorough regression testing. Changes the data contract between `__init__` and `apply_magnetic_trotter_step`.

---

## Phase 1: Data Registration (`conventions.py`)

**Difficulty**: Low. No dependencies.

### 1a. File path dicts (lines 167-183)

Add `"d=3"` entries:

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

The `PHYSICAL_PLAQUETTE_STATES` and `HAMILTONIAN_BOX_TERMS` LazyDicts are built via comprehension over these dicts (lines 304-317), so d=3 entries propagate automatically.

### 1b. Irrep truncation entry (lines 221-235)

Add a `"B3"` entry to `IRREP_TRUNCATIONS`. Although B3 currently contains the same irreps as T1, B represents a distinct truncation scheme from pyclebsch. Establishing the B-series namespace now keeps things forward-compatible.

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

### 1c. No other changes needed in `conventions.py`

`LatticeStateEncoder` accepts arbitrary link bitmaps and `LatticeDef` instances. The d=3 `LatticeDef` is already supported by the base class. Confirmed by audit.

---

## Phase 2: `__init__` Dimension Gate (`circuit.py`, line 84)

**Difficulty**: Low. Depends on Phase 1.

Add `case 3:` to the dimension match statement:

```python
case 3:
    lattice_size = lattice_encoder.lattice_def.shape[0]
    if lattice_size != lattice_encoder.lattice_def.shape[1] \
            or lattice_size != lattice_encoder.lattice_def.shape[2]:
        raise NotImplementedError("Non-cubic dim 3 lattices not yet supported.")
```

This unblocks `LatticeCircuitManager` construction for d=3 with `lattice_size >= 3` (no small-lattice logic triggered).

Ensure that there's still a default `case _` which raises a `NotImplementedError`.

---

## Phase 3: Unified Per-Plane Control Dir Cache (`circuit.py`, lines 88-93)

**Difficulty**: Medium. Depends on Phase 2.

Replace the d=2-specific `_cached_ctrl_dirs_d2_small_and_periodic` with a dimension-agnostic per-plane cache:

```python
self._cached_ctrl_dirs_small_and_periodic: Dict[Plane, tuple] = {}
```

Populate it for all dimensions when the lattice is small and periodic:

```python
if self._lattice_is_small and self._lattice_is_periodic:
    _temp_lattice = LatticeRegisters.from_lattice_state_encoder(self._encoder)
    dim = self._encoder.lattice_def.dim
    if dim == 1.5 or dim == 2:
        planes = [(1, 2)]
    elif dim == 3:
        planes = [(1, 2), (1, 3), (2, 3)]
    else:
        raise NotImplementedError(f"Dim {dim} lattice not yet supported.")
    origin = tuple(0 for _ in range(len(self._encoder.lattice_def.shape)))
    for e1, e2 in planes:
        _temp_plaq = _temp_lattice.get_plaquettes(origin, e1, e2)
        self._cached_ctrl_dirs_small_and_periodic[(e1, e2)] = _temp_plaq.control_link_dirs_per_vertex
```

**Audit note (2b.1)**: Use the origin vertex `(0, 0)` or `(0, 0, 0)` for all planes. On periodic cubic lattices all vertices are equivalent. This is explicit and correct for the current scope.

### Migration

All downstream references to `self._cached_ctrl_dirs_d2_small_and_periodic` must change to `self._cached_ctrl_dirs_small_and_periodic[(1, 2)]` in the d=3/2 and d=2 code paths. This is absorbed by Phase 5 (which eliminates the dimension-specific branching entirely).

---

## Phase 4: Per-Plane Consistency and Discard Methods (`circuit.py`, lines 878-965)

**Difficulty**: Medium. Depends on Phase 3.

### 4a. `_plaquette_state_has_inconsistent_controls` (lines 878-922)

**Interface change**: Add a required `plane: Plane` parameter.

The method body becomes dimension-agnostic. The four sharing pairs on a size-2 periodic lattice are the same structural pattern for any plane `(e1, e2)`:

```python
def _plaquette_state_has_inconsistent_controls(
    self, plaquette: PlaquetteState, plane: Plane
) -> bool:
    if not self._lattice_is_periodic or not self._lattice_is_small:
        raise ValueError(
            "Plaquette state consistency check only makes sense "
            "on a small, periodic lattice."
        )

    c_links = plaquette[2]
    dim = self._encoder.lattice_def.dim

    if dim == 1.5:
        # d=3/2: v1==v2 and v3==v4 (no plane dependence, single plane)
        return c_links[0] != c_links[1] or c_links[2] != c_links[3]

    # d=2 and d=3: direction-based lookup using the per-plane cache.
    ctrl_dirs = self._cached_ctrl_dirs_small_and_periodic[plane]
    e1, e2 = plane
    return (
        (c_links[0][ctrl_dirs[0].index(-e1)] != c_links[1][ctrl_dirs[1].index(e1)])
        or (c_links[0][ctrl_dirs[0].index(-e2)] != c_links[3][ctrl_dirs[3].index(e2)])
        or (c_links[1][ctrl_dirs[1].index(-e2)] != c_links[2][ctrl_dirs[2].index(e2)])
        or (c_links[2][ctrl_dirs[2].index(e1)] != c_links[3][ctrl_dirs[3].index(-e1)])
    )
```

Note: the d=3/2 case is structurally different (v1==v2, v3==v4 wholesale equality) and cannot be expressed as direction-based index lookups in the same way as d=2/d=3. Retaining the `if dim == 1.5` special case is the correct approach.

### 4b. `_discard_duplicate_controls_from_plaquette_state` (lines 924-965)

**Interface change**: Add a required `plane: Plane` parameter.

```python
def _discard_duplicate_controls_from_plaquette_state(
    self, plaquette: PlaquetteState, plane: Plane
) -> PlaquetteState:
    if not self._lattice_is_periodic or not self._lattice_is_small:
        raise ValueError(...)

    vertex_multiplicities, a_links, c_links = plaquette
    dim = self._encoder.lattice_def.dim

    if dim == 1.5:
        physical_c_links = (c_links[0], (), c_links[2], ())
    else:
        # d=2 and d=3: direction-based trimming.
        ctrl_dirs = self._cached_ctrl_dirs_small_and_periodic[plane]
        e1, e2 = plane
        physical_c_links = (
            c_links[0],  # v1: keep all
            # v2: drop +e1 (duplicates v1's -e1)
            tuple(c for i, c in enumerate(c_links[1])
                  if i != ctrl_dirs[1].index(e1)),
            # v3: drop +e2 (duplicates v2's -e2)
            tuple(c for i, c in enumerate(c_links[2])
                  if i != ctrl_dirs[2].index(e2)),
            # v4: drop +e2 (duplicates v1's -e2) and -e1 (duplicates v3's +e1)
            tuple(c for i, c in enumerate(c_links[3])
                  if i != ctrl_dirs[3].index(e2) and i != ctrl_dirs[3].index(-e1)),
        )

    return (vertex_multiplicities, a_links, physical_c_links)
```

**Key insight**: For d=2, v4 drops both controls (result: empty tuple), matching the existing behavior. For d=3, v4 drops 2 of 4 controls (result: 2 controls). The same code handles both because the FORDER-indexed direction lookups resolve differently per dimension.

### Trimming summary by dimension

| Vertex | d=3/2 keeps | d=2 keeps | d=3 keeps |
|---|---|---|---|
| v1 | 1 | 2 | 4 |
| v2 | 0 | 1 | 3 |
| v3 | 1 | 1 | 3 |
| v4 | 0 | 0 | 2 |
| **Total** | **2** | **4** | **12** |

---

## Phase 5: Per-Plane `__init__` Hamiltonian Filtering (`circuit.py`, lines 95-118)

**Difficulty**: High. Depends on Phases 3 and 4. This is the core architectural change.

### 5a. Restructure `self._mag_hamiltonian`

Change the in-memory Hamiltonian from:

```
Dict[Tuple[str, str], MatrixElementValue]
  i.e. Dict[(bs1, bs2), Dict[Plane, Dict[Signature, float]]]
```

to a **per-plane-first** resolved format:

```python
# New type alias (keep local to circuit.py):
ResolvedHamiltonianData = Dict[Plane, Dict[Tuple[str, str], Dict[Signature, float]]]
```

where the bitstring keys are already trimmed for the given plane (on small periodic lattices) or untrimmed (on large/non-periodic lattices where no filtering is needed).

### 5b. Per-plane filtering loop

Replace the current filtering loop (lines 95-118) with a per-plane loop. The logic must handle a critical subtlety identified by audit item 2c.5: a single `(bs1, bs2)` pair might be consistent for one plane but inconsistent for another. The filtering must discard per-plane, not per-entry.

```python
if self._lattice_is_small and self._lattice_is_periodic:
    resolved: ResolvedHamiltonianData = {}

    for (final_bs, initial_bs), matrix_elem_value in self._mag_hamiltonian.items():
        final_ps = lattice_encoder.decode_bit_string_to_plaquette_state(final_bs)
        initial_ps = lattice_encoder.decode_bit_string_to_plaquette_state(initial_bs)

        for plane, sig_dict in matrix_elem_value.items():
            # Check consistency for THIS plane.
            if (self._plaquette_state_has_inconsistent_controls(final_ps, plane)
                    or self._plaquette_state_has_inconsistent_controls(initial_ps, plane)):
                continue

            # Trim duplicate controls for THIS plane.
            final_trimmed = self._discard_duplicate_controls_from_plaquette_state(final_ps, plane)
            initial_trimmed = self._discard_duplicate_controls_from_plaquette_state(initial_ps, plane)
            trimmed_key = (
                lattice_encoder.encode_plaquette_state_as_bit_string(
                    final_trimmed, override_n_c_links_validation=True),
                lattice_encoder.encode_plaquette_state_as_bit_string(
                    initial_trimmed, override_n_c_links_validation=True),
            )

            if plane not in resolved:
                resolved[plane] = {}
            # sig_dict is Dict[Signature, float]; keep it intact.
            if trimmed_key in resolved[plane]:
                # Merge signature dicts (shouldn't collide, but be safe).
                resolved[plane][trimmed_key].update(sig_dict)
            else:
                resolved[plane][trimmed_key] = dict(sig_dict)

    self._mag_hamiltonian = resolved
else:
    # Large or non-periodic: pivot to per-plane-first without filtering/trimming.
    resolved: ResolvedHamiltonianData = {}
    for (bs1, bs2), matrix_elem_value in self._mag_hamiltonian.items():
        for plane, sig_dict in matrix_elem_value.items():
            if plane not in resolved:
                resolved[plane] = {}
            resolved[plane][(bs1, bs2)] = dict(sig_dict)
    self._mag_hamiltonian = resolved
```

### 5c. Update `_resolve_hamiltonian_for_plaquette`

With the new per-plane-first structure, this method simplifies:

```python
@staticmethod
def _resolve_hamiltonian_for_plaquette(
    hamiltonian: ResolvedHamiltonianData,
    plane: Plane,
    signature: Signature,
) -> List[Tuple[str, str, float]]:
    plane_data = hamiltonian.get(plane, {})
    resolved: List[Tuple[str, str, float]] = []
    for (bs1, bs2), sig_dict in plane_data.items():
        sig_val = sig_dict.get(signature)
        if sig_val is not None:
            resolved.append((bs1, bs2, float(sig_val)))
    return resolved
```

### 5d. Update `__repr__` (line 122)

The `__repr__` currently prints `self._mag_hamiltonian` directly. Update it to handle the new structure gracefully (audit item 6.2). The simplest approach is to print the number of planes and total entries rather than dumping the full dict:

```python
def __repr__(self):
    class_name = type(self).__name__
    n_planes = len(self._mag_hamiltonian)
    n_entries = sum(len(v) for v in self._mag_hamiltonian.values())
    return (f"{class_name}({self._encoder.__repr__()}, "
            f"<{n_entries} Hamiltonian entries across {n_planes} plane(s)>)")
```

---

## Phase 6: Per-Plane `_strip_redundant_controls_if_small_and_periodic_lattice` (`circuit.py`, lines 711-742)

**Difficulty**: High. Depends on Phases 4 and 5.

### The problem (audit items 2c.2, 2c.4, 6.1)

Currently `_strip_redundant_controls_if_small_and_periodic_lattice` returns `Set[str] | None` — a single set of trimmed bitstrings. For d=3, different planes produce different trimmed bitstrings (different lengths, different content). A single set cannot serve all planes.

### Interface change

Change the return type to `Dict[Plane, Set[str]] | None` when the lattice is small and periodic. When it is not, return `None` (no pruning).

```python
def _strip_redundant_controls_if_small_and_periodic_lattice(
    self,
    physical_states_for_control_pruning: set[str] | None
) -> Dict[Plane, set[str]] | None:
    if (physical_states_for_control_pruning is None
            or not self._lattice_is_periodic
            or not self._lattice_is_small):
        return None  # Caller should use original set unchanged.

    result: Dict[Plane, set[str]] = {}
    for plane in self._cached_ctrl_dirs_small_and_periodic:
        stripped = []
        for plaquette_string in physical_states_for_control_pruning:
            ps = self._encoder.decode_bit_string_to_plaquette_state(plaquette_string)
            if self._plaquette_state_has_inconsistent_controls(ps, plane):
                continue
            trimmed = self._discard_duplicate_controls_from_plaquette_state(ps, plane)
            trimmed_bs = self._encoder.encode_plaquette_state_as_bit_string(
                trimmed, override_n_c_links_validation=True)
            stripped.append(trimmed_bs)
        result[plane] = set(stripped) if stripped else None  # None = no pruning for this plane

    return result
```

### Downstream ripple effects (audit item 6.1)

The following consumers must be updated:

1. **`apply_magnetic_trotter_step`** (line 486): Currently assigns the return value to `physical_states_for_control_pruning` (a `Set[str] | None`). Must change to store the per-plane dict separately and look up the correct set per plaquette plane before passing to `_build_mag_evol_circuit`.

2. **`_build_mag_evol_circuit`** (line 774): Its `physical_states_for_control_pruning` parameter type is `Union[None | Set[str]]`. No change needed to this method's signature — callers just pass the correct per-plane set.

3. **`givens` and `givens_fused_controls`** (`givens.py`): Accept `Set[str] | None`. No change needed — they receive the already-resolved per-plane set.

4. **Cache invalidation** (lines 502-504): Currently compares `physical_states_for_control_pruning` for equality. Must compare the per-plane dict (or the original input, since the per-plane dict is deterministically derived from it).

### Concrete changes in `apply_magnetic_trotter_step`

```python
# At line 486, replace:
physical_states_for_control_pruning = self._strip_redundant_controls_if_small_and_periodic_lattice(
    physical_states_for_control_pruning)

# With:
per_plane_pruning_states: Dict[Plane, set[str]] | None = None
if self._lattice_is_small and self._lattice_is_periodic:
    per_plane_pruning_states = self._strip_redundant_controls_if_small_and_periodic_lattice(
        physical_states_for_control_pruning)

# Then inside the plaquette loop (around line 578), when calling _build_mag_evol_circuit:
effective_pruning_states = (
    per_plane_pruning_states.get(plaquette_plane)
    if per_plane_pruning_states is not None
    else physical_states_for_control_pruning
)
# Pass effective_pruning_states instead of physical_states_for_control_pruning
```

For cache invalidation (lines 498-506), compare against the original `physical_states_for_control_pruning` input (before per-plane splitting), since two calls with the same input will produce the same per-plane output:

```python
or (
    physical_states_for_control_pruning
    != self._cached_mag_evol_params["physical_states_for_control_pruning"]
)
```

This requires keeping the original `physical_states_for_control_pruning` value for comparison and storing it in the cache params dict as before.

---

## Phase 7: Unified Qubit-Stitching Skip Logic (`circuit.py`, lines 609-629)

**Difficulty**: Medium. Depends on Phases 3 and 6.

Replace the dimension-specific `match` with a single dimension-agnostic branch that uses the per-plane ctrl dir cache and the current plaquette's plane.

### Pre-compute per-plane skip indices

Before the vertex loop (after the existing skip-index computation at lines 516-520), compute skip indices for all cached planes:

```python
# Per-plane skip indices for small periodic lattices.
# Maps (plane, vertex_idx) -> set of ctrl_idx values to skip.
_skip_indices: Dict[Tuple[Plane, int], set[int]] = {}
if self._lattice_is_small and self._lattice_is_periodic:
    dim = self._encoder.lattice_def.dim
    if dim == 1.5:
        # d=3/2: skip v2 and v4 entirely (all ctrl indices).
        for plane in self._cached_ctrl_dirs_small_and_periodic:
            n_ctrls_per_vertex = len(self._cached_ctrl_dirs_small_and_periodic[plane][0])
            _skip_indices[(plane, 1)] = set(range(n_ctrls_per_vertex))  # v2: skip all
            _skip_indices[(plane, 3)] = set(range(n_ctrls_per_vertex))  # v4: skip all
    else:
        # d=2 and d=3: direction-based skipping.
        for plane, ctrl_dirs in self._cached_ctrl_dirs_small_and_periodic.items():
            e1, e2 = plane
            _skip_indices[(plane, 1)] = {ctrl_dirs[1].index(e1)}          # v2: skip +e1
            _skip_indices[(plane, 2)] = {ctrl_dirs[2].index(e2)}          # v3: skip +e2
            _skip_indices[(plane, 3)] = {ctrl_dirs[3].index(e2),          # v4: skip +e2
                                         ctrl_dirs[3].index(-e1)}         #     and -e1
```

### In the stitching loop

Replace the `match self._encoder.lattice_def.dim` block (lines 614-627) with:

```python
if self._lattice_is_small and self._lattice_is_periodic:
    skip_set = _skip_indices.get((plaquette_plane, vertex_idx), set())
    if ctrl_idx in skip_set:
        continue
```

This is cleaner, dimension-agnostic, and O(1) per check.

### Verification of d=2 equivalence

For d=2 with plane (1,2):
- v1 (idx 0): no entry in `_skip_indices` -> no skips. Correct.
- v2 (idx 1): skip `{ctrl_dirs[1].index(1)}` = skip the +e1 control. Matches existing `_v2_skip_ctrl_idx`.
- v3 (idx 2): skip `{ctrl_dirs[2].index(2)}` = skip the +e2 control. Matches existing `_v3_skip_ctrl_idx`.
- v4 (idx 3): skip `{ctrl_dirs[3].index(2), ctrl_dirs[3].index(-1)}` = skip both. For d=2, v4 has only 2 controls, so this skips everything. Matches existing `vertex_idx == 3`.

### Verification of d=3 equivalence with sharing analysis

For d=3 with any plane (e1, e2):
- v1: no skips -> 4 controls kept.
- v2: skip +e1 -> 3 controls kept.
- v3: skip +e2 -> 3 controls kept.
- v4: skip +e2 and -e1 -> 2 controls kept.
- Total: 4 + 3 + 3 + 2 = 12. Matches the sharing analysis in the original plan.

---

## Phase 8: Test Updates

**Difficulty**: Low-Medium. Depends on all prior phases.

### 8a. `test_conventions.py` (line 50-51)

Add `case "d=3":` to `test_physical_plaquette_state_data_are_valid`:

```python
case "d=3":
    expected_num_c_links = 16  # 4 controls/vertex * 4 vertices
```

Other tests that iterate over all dimension/truncation combos (`test_no_duplicate_physical_plaquette_states`, `test_no_duplicate_matrix_elements`, etc.) will automatically pick up B3 data once the file path dicts are updated (audit item 3.1). Verify these pass.

### 8b. Integration test (`test_integration_mps.py`)

Add a concrete end-to-end test (audit item 3.2). Following the pattern of existing tests in that file:

**`test_d3_B3_size3_magnetic_trotter_step`**: Construct a `LatticeCircuitManager` for d=3, B3, size=3 (no small-lattice logic). Call `apply_magnetic_trotter_step`. Verify the circuit has the expected number of qubits, parameters, and is non-empty.

**`test_d3_B3_size2_magnetic_trotter_step`**: Same but with size=2 (triggers small-periodic filtering). Verify the circuit constructs without error. Additionally verify that the number of c_link qubits per plaquette matches the trimmed count (12 per plaquette).

### 8c. Unit tests for per-plane filtering

**`test_per_plane_consistency_check`**: For d=3, size=2, construct plaquette states that are consistent for plane (1,2) but inconsistent for plane (1,3). Verify that `_plaquette_state_has_inconsistent_controls` returns the correct boolean for each plane.

**`test_per_plane_control_trimming`**: For d=3, size=2, verify that `_discard_duplicate_controls_from_plaquette_state` produces different-length results for different planes (always 12 total controls, but different positions trimmed).

### 8d. Regression tests

Run the full existing test suite (`uv run pytest -v`) to verify d=3/2 and d=2 behavior is preserved after the unified refactor.

---

## Implementation Order and Dependencies

```
Phase 1: conventions.py data registration
  |
  v
Phase 2: __init__ dimension gate (case 3)
  |
  v
Phase 3: unified per-plane ctrl dir cache
  |
  +---> Phase 4: per-plane consistency/discard methods
  |       |
  |       v
  +---> Phase 5: per-plane __init__ Hamiltonian filtering  [HARD DEPENDENCY on 4]
  |       |
  |       v
  +---> Phase 6: per-plane _strip_redundant_controls       [HARD DEPENDENCY on 4, 5]
  |       |
  |       v
  +---> Phase 7: unified qubit-stitching skip logic        [HARD DEPENDENCY on 3, 6]
          |
          v
        Phase 8: tests                                     [HARD DEPENDENCY on all]
```

Phases 4 and 5 must be implemented and tested together — they share the same interface change (the `plane` parameter). Phase 6 depends on both. Phase 7 depends on Phase 3's cache and Phase 6's per-plane pruning states.

**Regression checkpoint**: After completing Phases 3-7, run the full test suite to verify d=3/2 and d=2 behavior is preserved *before* adding any d=3-specific tests. The unified refactor should be a no-op for existing dimensions.

---

## Files Changed

| File | Phases | Nature of Change |
|---|---|---|
| `ymcirc/conventions.py` | 1 | Add data paths and B3 irrep truncation entry |
| `ymcirc/circuit.py` | 2-7 | Core refactor: unified per-plane filtering, new cache structure, simplified stitching |
| `tests/test_conventions.py` | 8a | Add d=3 case to existing test |
| `tests/test_integration_mps.py` | 8b | New integration tests for d=3 |
| `tests/test_circuit.py` (or new file) | 8c | New unit tests for per-plane filtering |

## Files NOT Changed

| File | Reason |
|---|---|
| `_abstract/lattice_data.py` | Generic for d >= 2 periodic cubic. Note: `_normalize_link_address` and `add_unit_vector_to_vertex_vector` use `shape[0]` for all components — correct for cubic, tech debt for non-cubic (audit items 4.1, 4.2). |
| `lattice_registers.py` | Fully generic. |
| `parsed_lattice_result.py` | Fully generic. |
| `measurement_results.py` | The `NotImplementedError` at line 40 is for non-hypercubic tuple-valued shapes. A d=3 cubic `LatticeDef(3, 4)` has `shape = (4, 4, 4)` which is tuple-valued but hypercubic. Verify during implementation that this path is not triggered (audit item 4.3). |
| `electric_helper.py` | `NotImplementedError` is for non-hypercubic lattices, not relevant. |
| `givens.py` | No dimension-specific logic. Accepts `Set[str] | None` for physical states — no interface change needed. |
| `utilities.py` | No changes needed. |

---

## Performance Notes

The d=3 B3 data has 54,035 plaquette states and 707 Hamiltonian entries. A size-2 d=3 periodic lattice has 24 plaquettes (8 vertices x 3 planes). There is only 1 unique signature for B3, so the resolved Hamiltonian cache has at most 3 entries (one per plane). The per-plane filtering in `__init__` iterates 707 entries x 3 planes = ~2,100 decode/check/trim operations — negligible.

The on-disk data format is unchanged. The restructuring happens in-memory during `__init__`, keeping compact file storage while matching the consumption pattern.

---

## Status

Phases 1-7 are complete. All dimension-specific `case`/`match` branches for small-periodic logic have been eliminated from `circuit.py`. `_strip_redundant_controls_if_small_and_periodic_lattice` now returns `Dict[Plane, set[str] | None] | None`, producing per-plane stripped state sets. Its caller in `apply_magnetic_trotter_step` resolves the correct per-plane set before passing to `_build_mag_evol_circuit`. Cache invalidation compares against the original `physical_states_for_control_pruning` input. The qubit-stitching skip logic uses a pre-computed `_skip_indices: Dict[(Plane, vertex_idx), set[int]]` dict for O(1) lookups, replacing the old dimension-specific `match` block. Phase 8 (tests) remains.

- [x] Phase 1: Data registration in `conventions.py`
- [x] Phase 2: `__init__` dimension gate (`case 3:`)
- [x] Phase 3: Unified per-plane control dir cache
- [x] Phase 4: Per-plane consistency/discard methods
- [x] Phase 5: Per-plane `__init__` Hamiltonian filtering
- [x] Phase 6: Per-plane `_strip_redundant_controls`
- [x] Phase 7: Unified qubit-stitching skip logic
- [ ] Phase 8: Tests
