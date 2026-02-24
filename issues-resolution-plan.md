# Issues Resolution Plan

This document describes the plan for resolving the outstanding issues from the F-order refactor, as documented in `issues.md`.

---

## Issue 1: `_flatten_hamiltonian_value` should not enforce equal floats

**File:** `ymcirc/conventions.py`
**Location:** `_flatten_hamiltonian_value` (around line 224-241)

**Problem:** The function currently checks `abs(f - floats[0]) < 1e-12` when it encounters a dict value, enforcing that all float values across planes/signatures are equal. This will break for nonperiodic or higher-dimensional lattices where the values legitimately differ.

**Definitions:** A "signature" is a length-4 tuple whose elements are themselves tuples of the F-ordered control links at each vertex in a plaquette. This pattern (per-vertex tuples of F-ordered control links) is already used elsewhere in the codebase (e.g., `Plaquette.control_links_per_vertex`), and downstream consumers should use it consistently.

**Resolution:**
1. Remove the equality enforcement check in the dict-value branch.
2. Stop flattening dict values to a single float. Instead, when the value for a matrix element is a dict (keyed by plane/signature), preserve it as-is. When the value is already a plain float, keep it as a float. Downstream consumers that need to access matrix element values must be updated to handle both cases:
   - If the value is a float, use it directly (no plane/signature information required).
   - If the value is a dict, the consumer must supply plane + signature information to look up the relevant float. The signature should be provided as a length-4 tuple of per-vertex F-ordered control link tuples, consistent with the existing pattern used in `Plaquette.control_links_per_vertex`.
3. Update downstream consumers (`load_magnetic_hamiltonian`, `compute_all_rotations_from_just_box_terms`, and any other code that reads Hamiltonian matrix element values) to accept both float and dict values, requiring plane/signature selection when a dict is present.
4. Update the docstring to reflect the new behavior.

---

## Issue 2: `trunc_string` should come from metadata, not be hard-coded

**File:** `ymcirc/conventions.py`
**Location:** `_load_plaquette_states` (around line 249) and `_load_hamiltonian` (around line 258)

**Problem:** Both functions construct `trunc_string` as `f"T{metadata.get('cutoff', '')}"`, which hard-codes the assumption that the truncation label is always `"T"` followed by the cutoff number. The truncation string should instead be extracted directly from metadata.

**Resolution:**
1. Construct `trunc_string` by concatenating `metadata["truncation_mode"]` with `metadata["cutoff"]` (e.g., if `truncation_mode` is `"T"` and `cutoff` is `1`, the result is `"T1"`).
2. Replace the hard-coded `f"T{metadata.get('cutoff', '')}"` construction with this metadata-derived approach in both `_load_plaquette_states` and `_load_hamiltonian`.
3. Store the extracted `trunc_string` in `_DATA_METADATA` as part of the cached metadata if not already present.

---

## Issue 3: Restore the d=2 plaquette state diagram in the module docstring

**File:** `ymcirc/conventions.py`
**Location:** Module-level docstring (lines 98-105 currently)

**Problem:** Prior to commit `5351001`, the docstring contained an ASCII diagram for the d=2 plaquette state showing control link positions around the plaquette. This diagram was removed during the refactor. The current docstring has the per-vertex tuple representation for d=2 but no accompanying ASCII art (unlike d=3/2, which does have a diagram on lines 90-96).

**Resolution:**
1. Add back an ASCII diagram for d=2, placed after the per-vertex tuple representation (after current line 105).
2. Update the diagram labels to use the F-order convention (i.e., label controls by their direction and vertex, matching the per-vertex nested tuple format) rather than the old flat `c1`-`c8` labels.
3. The diagram should show all 8 control links (2 per vertex) with labels like `c_v1_dir-1`, `c_v1_dir-2`, etc., consistent with the default F-order `[1, 2, 3, -1, -2, -3]`.

**Reference:** The old diagram from commit `5351001^` showed:
```
                 c7           c6
                 |            |
         c8 ---- v4 ----l3--- v3 ---- c5
                 |            |
                 l4           l2
                 |            |
         c1 ---- v1 ----l1--- v2 ---- c4
                 |            |
                 c2           c3
```

The updated version should relabel using the per-vertex, FORDER-based naming.

---

## Issue 4: Remove redundant `forder` argument from `LatticeStateEncoder`

**File:** `ymcirc/conventions.py`
**Location:** `LatticeStateEncoder.__init__` (around line 381-479)

**Problem:** `LatticeStateEncoder` takes an explicit `forder` argument in its constructor, but the `lattice` argument (a `LatticeDef`) already has `forder` as a property. This is redundant and could lead to inconsistency if different values are passed.

**Resolution:**
1. Remove the `forder` parameter from `LatticeStateEncoder.__init__`.
2. Instead, read `forder` from `lattice.forder` when constructing the internal `LatticeDef`.
3. Update the constructor's docstring to remove the `forder` parameter documentation and note that F-order is inherited from the `lattice` argument.
4. Search all call sites of `LatticeStateEncoder(...)` across the codebase (including tests) and remove any `forder=...` arguments being passed.
5. Fix any tests that relied on passing `forder` explicitly to the encoder. These tests should instead set `forder` on the `LatticeDef` / lattice object before passing it to the encoder.
6. The `forder` property on the encoder should continue to work, sourcing from `self._lattice.forder`.

**Files to check for call sites:**
- `ymcirc/conventions.py` (internal usage)
- `run/functions.py` (`create_lattice_encoder`)
- `tests/test_conventions.py`
- `tests/test_circuit.py`
- `tests/test_lattice_registers.py`
- `tests/test_parsed_lattice_result.py`
- `tests/test_integration_mps.py`

---

## Issue 5: `decode_bit_string_to_plaquette_state` should check for periodic boundary conditions

**File:** `ymcirc/conventions.py`
**Location:** `LatticeStateEncoder.decode_bit_string_to_plaquette_state` (around line 643-708)

**Problem:** The method does not verify that the lattice has periodic boundary conditions. Decoding logic for nonperiodic lattices has not been implemented, so attempting to decode on a nonperiodic lattice would silently produce incorrect results.

**Resolution:**
1. At the start of `decode_bit_string_to_plaquette_state`, add a check:
   ```python
   if not self._lattice.periodic_boundary_conds:
       raise NotImplementedError(
           "Decoding plaquette states for nonperiodic lattices is not yet supported."
       )
   ```
2. Place this check early in the method, before any decoding logic runs.
3. Update the method's docstring to document that `NotImplementedError` is raised for nonperiodic lattices.

---

## Issue 6: Add `refresh` flag to `get_data_metadata`

**File:** `ymcirc/conventions.py`
**Location:** `get_data_metadata` (around line 265-274)

**Problem:** `get_data_metadata` caches metadata in the module-level `_DATA_METADATA` dict and provides no way to force a reload.

**Resolution:**
1. Add a `refresh: bool = False` parameter to `get_data_metadata`.
2. When `refresh=True`, bypass the cache check and force re-loading from the underlying data source (triggering a fresh lazy-load of the relevant `PHYSICAL_PLAQUETTE_STATES` or `HAMILTONIAN_BOX_TERMS` entry).
3. Update `_DATA_METADATA` with the fresh result before returning.
4. Update the docstring accordingly.

**Considerations:** If `LazyDict` caches its loaded values internally (which it does — it replaces the loader with the result on first access), a true refresh may also need to reset the `LazyDict` entry. Assess whether `LazyDict` supports this or whether an additional mechanism is needed.

---

## Issue 7: Hard-coded control link indexing in `LatticeCircuitManager`

**File:** `ymcirc/circuit.py`
**Location:** `_plaquette_state_has_inconsistent_controls` (~line 803), `_discard_duplicate_controls_from_plaquette_state` (~line 842), `apply_magnetic_trotter_step` (~line 419)

**Problem:** These three methods all index control links by hard-coded vertex and within-vertex indices (e.g., `c_links[0][0]`, `c_links[1][1]`) that implicitly assume the default F-order `[1, 2, 3, -1, -2, -3]`. With a non-default F-order, the mapping of direction-to-index within each vertex's control tuple changes, and these hard-coded indices would refer to the wrong physical links.

**Resolution:**
1. For each method, determine which physical directions (e.g., `+1`, `-2`) are being compared or pruned. The current hard-coded indices correspond to specific directions under the default F-order.
2. Use `Plaquette.compute_control_link_dirs_per_vertex(dim, plane, forder)` (passing `self._encoder.forder`) to get the actual per-vertex direction ordering.
3. Replace hard-coded index lookups with direction-based lookups. For example, instead of `c_links[0][0]`, find the index of the relevant direction within vertex 0's control direction tuple and use that.
4. Alternatively, build a helper (private method on `LatticeCircuitManager`) that returns a mapping from `(vertex_index, direction)` to `(vertex_index, within_vertex_index)` for the current F-order, and use this mapping in all three methods.
5. Update docstrings to explain that the methods are F-order-aware.

**Detailed breakdown for each method:**

### `_plaquette_state_has_inconsistent_controls`
- **d=3/2:** Currently checks `c_links[0] != c_links[1]` and `c_links[2] != c_links[3]`. These compare full per-vertex tuples, which each have only 1 element in d=3/2. Since there's only 1 control per vertex in d=3/2, the F-order doesn't change anything here — the indices are correct regardless of F-order.
- **d=2:** Currently checks specific `[vertex][control_idx]` pairs. These need to be rewritten to identify the shared physical link by direction rather than by index.

### `_discard_duplicate_controls_from_plaquette_state`
- **d=3/2:** Keeps vertices 0 and 2, discards 1 and 3. This is vertex-based, not direction-based within a vertex, so it may be F-order-independent. Verify this.
- **d=2:** Hard-codes which within-vertex indices to keep/discard. Must be updated to use direction-based identification.

### `apply_magnetic_trotter_step`
- Uses a match statement on dimensionality to skip certain controls. Needs to identify which controls to skip based on direction, not index position.

---

## Issue 8: `ParsedLatticeResult` missing `forder` property

**File:** `ymcirc/parsed_lattice_result.py`
**Location:** `__init__` (around line 43-104)

**Problem:** `ParsedLatticeResult` inherits from `LatticeData` (which inherits from `LatticeDef`), so it has an `forder` property. However, its `__init__` calls `super().__init__(dimensions, size, periodic_boundary_conds)` without passing `forder`, so `forder` defaults to the standard `[1, 2, 3, -1, -2, -3]` regardless of what the encoder's lattice uses.

**Resolution:**
1. In `ParsedLatticeResult.__init__`, extract `forder` from `lattice_encoder.forder` (or `lattice_encoder.lattice_def.forder`).
2. Pass it to the superclass init:
   ```python
   super().__init__(dimensions, size, periodic_boundary_conds, forder=lattice_encoder.forder)
   ```
3. Note that `LatticeRegisters` handles this differently: it requires an explicit `forder` creation argument because it doesn't take a `LatticeStateEncoder` argument. For `ParsedLatticeResult`, since it does receive a `lattice_encoder`, pull `forder` from `lattice_encoder.lattice_def.forder` instead of requiring a separate argument.
4. Update the docstring if needed.

---

## Issue 9: `initialize_lattice_tools` in `run/functions.py` ignores F-order metadata

**File:** `run/functions.py`
**Location:** `initialize_lattice_tools` (~line 228) and `configure_script_options` (~line 220)

**Problem:** When creating the `LatticeDef` and `LatticeStateEncoder`, the run module does not read the `f_order` field from the loaded metadata. This means the pipeline always uses the default F-order even if the data was generated with a non-standard F-order.

**Resolution:**
1. After loading data (plaquette states and/or Hamiltonian terms), call `get_data_metadata(dim_string, trunc_string)` to retrieve the metadata including `f_order`.
2. Extract `f_order` from the metadata dict.
3. Pass `forder=f_order` when constructing the `LatticeDef` in `configure_script_options`.
4. Since Issue 4 removes the `forder` arg from `LatticeStateEncoder`, the F-order will flow through from the `LatticeDef` to the encoder automatically.
5. Verify that `time_evol.py` works correctly with this change (the other run scripts are not important).

---

## Issue 10: Test for bad `forder` creation argument

**File:** New test in `tests/` (likely `tests/test_lattice_data.py` or added to an existing test file)

**Problem:** No test currently verifies that creating a `LatticeDef` (or subclass) with an invalid `forder` raises an appropriate exception.

**Resolution:**
1. Determine the best test file location. If `tests/test_lattice_registers.py` already tests `LatticeDef` creation, add there. Otherwise, consider a new test file or add to `tests/test_conventions.py`.
2. Write test cases:
   - `forder` that is not a list → should raise `ValueError` or `TypeError`
   - `forder` with wrong elements (e.g., `[1, 2, 3, 4, 5, 6]`) → should raise `ValueError`
   - `forder` with duplicate elements (e.g., `[1, 1, 3, -1, -2, -3]`) → should raise `ValueError`
   - `forder` with wrong length → should raise `ValueError`
   - `forder` that is a valid permutation → should succeed (positive test)
3. Use `pytest.raises(ValueError)` for the negative cases.

---

## Issue 11: Tests for F-order affecting control link ordering

**File:** Tests across multiple files

**Problem:** There are no tests verifying that changing F-order actually changes control link ordering as expected.

### Sub-issue 11a: `get_plaquettes` with non-standard F-order

**Resolution:**
1. In `tests/test_lattice_registers.py` and/or `tests/test_parsed_lattice_result.py`, add tests that:
   - Create a `LatticeDef` (or `LatticeRegisters` / `ParsedLatticeResult`) with a non-default F-order.
   - Call `get_plaquettes()` for a d=2 lattice.
   - Verify that the `control_links_per_vertex` ordering differs from the default F-order case in the expected way.
2. The expected ordering can be computed manually by applying `Plaquette.compute_control_link_dirs_per_vertex(dim, plane, forder)` with the non-standard forder.

### Sub-issue 11b: Non-standard F-order doesn't break redundant control removal on small lattices

**Resolution:**
1. Write an integration-level test (can go in `tests/test_circuit.py` or a new `tests/test_forder_integration.py`).
2. For both d=3/2 and d=2, with a small periodic lattice (size=2):
   - Create `LatticeStateEncoder` and `LatticeCircuitManager` with a non-standard F-order.
   - Verify that `_plaquette_state_has_inconsistent_controls` correctly identifies inconsistent states.
   - Verify that `_discard_duplicate_controls_from_plaquette_state` correctly prunes the right controls.
   - Optionally, verify that `apply_magnetic_trotter_step` produces a valid circuit (no errors).
3. This test depends on Issue 7 being resolved first (the hard-coded indexing fix).

### Sub-issue 11c: `LatticeStateEncoder` encode/decode with non-standard F-order

**Resolution:**
1. In `tests/test_conventions.py`, add tests that:
   - Create `LatticeStateEncoder` with a non-standard F-order for both d=3/2 and d=2.
   - Encode a known plaquette state to a bit string.
   - Decode the bit string back.
   - Verify round-trip correctness: `decode(encode(state)) == state`.
   - Also verify that the encoded bit string differs from what the default F-order would produce (i.e., the F-order actually affects the encoding).
2. Test for both standard and non-standard F-orders to confirm both work.

---

## Execution Order and Dependencies

The issues should be addressed in the following order, grouped by dependency:

### Phase 1: Foundation fixes (no cross-dependencies)
These can be done in parallel:

- [x] **Issue 1** — `_flatten_hamiltonian_value` float enforcement
- [x] **Issue 2** — `trunc_string` from metadata
- [x] **Issue 3** — Restore d=2 diagram
- [x] **Issue 5** — Periodic boundary check in `decode_bit_string_to_plaquette_state`
- [x] **Issue 6** — `refresh` flag on `get_data_metadata`

### Phase 2: F-order plumbing
These should be done in sequence:

- [ ] **Issue 4** — Remove redundant `forder` arg from `LatticeStateEncoder` (must come before Issue 9, which depends on F-order flowing through `LatticeDef`)
- [ ] **Issue 8** — `ParsedLatticeResult` pass `forder` to superclass init
- [ ] **Issue 7** — Fix hard-coded control link indexing in `LatticeCircuitManager` (most complex change)
- [ ] **Issue 9** — `initialize_lattice_tools` reads F-order from metadata (depends on Issues 4 and 7)

### Phase 3: Tests
These depend on Phase 2:

- [ ] **Issue 10** — Bad `forder` validation test
- [ ] **Issue 11a** — `get_plaquettes` with non-standard F-order
- [ ] **Issue 11b** — Redundant control removal with non-standard F-order (depends on Issue 7)
- [ ] **Issue 11c** — Encoder round-trip with non-standard F-order

### Final step:
- [ ] Run full test suite (`uv run pytest -v`) to verify no regressions.
