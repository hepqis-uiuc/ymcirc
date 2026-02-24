# Issues Round 2 — Resolution Plan

This document describes the plan for resolving the items listed in `issues-round-2.md`.

---

## Small Item 1: Rename `_flatten_hamiltonian_value`

**File:** `ymcirc/conventions.py`
**Location:** `_flatten_hamiltonian_value` (line ~234)

**Problem:** The name `_flatten_hamiltonian_value` no longer describes what the function does. It used to flatten dict values to a single float, but now it either coerces numeric values to `float` or passes through dicts unchanged. "Flatten" implies reducing structure, which no longer applies to the dict case.

**Resolution:**
1. Rename `_flatten_hamiltonian_value` to `_normalize_hamiltonian_value`. The word "normalize" accurately describes what the function does: it enforces a canonical type for each value (numeric → `float`, dict → preserved as-is), without implying structural reduction.
2. Update the single call site in `_load_hamiltonian` to use the new name.
3. Update the docstring to use "normalize" language.

---

## Small Item 2: Audit `get_data_metadata` refresh logic

**File:** `ymcirc/conventions.py`
**Location:** `get_data_metadata` (line ~276)

**Analysis:** The current implementation is:
```python
def get_data_metadata(dim_string: str, trunc_string: str, refresh: bool = False) -> dict:
    key = (dim_string, trunc_string)
    if key not in _DATA_METADATA or refresh:
        _ = PHYSICAL_PLAQUETTE_STATES[dim_string][trunc_string]
    return _DATA_METADATA[key]
```

The `LazyDict` class (in `utilities.py`) does **not** cache loaded values — every `__getitem__` call invokes the loader function, which re-reads from disk and updates `_DATA_METADATA` as a side effect. This means:

- **Without `refresh=True`:** If metadata was already populated (by any prior access to `PHYSICAL_PLAQUETTE_STATES` or `HAMILTONIAN_BOX_TERMS`), the cached metadata is returned without a disk read. This is correct — it avoids redundant I/O.
- **With `refresh=True`:** The `LazyDict` access is forced, triggering a disk reload via `_load_plaquette_states`, which updates `_DATA_METADATA`. The fresh metadata is then returned. This works correctly.

**Conclusion:** The refresh logic is correct. Update the docstrings on both `get_data_metadata` and `LazyDict` to briefly explain this behavior — specifically that `LazyDict` does not cache and re-reads from disk on every access, and that `get_data_metadata` uses `_DATA_METADATA` as its cache layer to avoid redundant disk reads unless `refresh=True` is passed.

---

## Complex Item 1: `compute_all_rotations_from_just_box_terms` dict handling

**File:** `ymcirc/conventions.py`
**Location:** `compute_all_rotations_from_just_box_terms` (line ~357)

**Problem:** Currently raises `NotImplementedError` when a matrix element value is a dict (keyed by plane/signature). The function needs to handle dict-valued matrix elements by propagating the dict structure through the box + box† sum.

**Background — matrix element value structure:**
A matrix element value is one of:
- `float` — a single amplitude (plane/signature-independent).
- `dict` with **plane tuple** keys. Each value is either:
  - `float` — amplitude for that plane (signature-independent).
  - `dict` with **signature** keys → `float` values — amplitude for that specific (plane, signature) combination.

A "signature" is a length-4 tuple of per-vertex F-ordered control link tuples (matching `Plaquette.control_links_per_vertex`), as defined in the Issue 1 section of `issues-resolution-plan.md`.

**Resolution:**

### Step 1: Define type aliases

Define the `Signature` and `Plane` type aliases in `ymcirc/_abstract/lattice_data.py` (alongside other lattice-related type aliases like `LatticeVector`, `LinkAddress`, etc.), and import them into `conventions.py`. Define the `MatrixElementValue` type alias in `conventions.py` (since it is specific to Hamiltonian data encoding, not lattice geometry). The type aliases are:

```python
# A signature is a tuple of 4 per-vertex control link tuples, each sorted by FORDER.
Signature = Tuple[VertexControlLinks, VertexControlLinks, VertexControlLinks, VertexControlLinks]
Plane = Tuple[int, int]

# Matrix element value: either a plain float, or a dict keyed by plane (whose values
# are either floats or dicts keyed by signature → float).
MatrixElementValue = Union[float, Dict[Plane, Union[float, Dict[Signature, float]]]]
```

### Step 2: Update `compute_all_rotations_from_just_box_terms`

Change the return type from `List[Tuple[PlaquetteState, PlaquetteState, float]]` to `Dict[Tuple[PlaquetteState, PlaquetteState], MatrixElementValue]`.

When both `box_amplitude` and `box_dagger_amplitude` are floats (or 0), sum them as before.

When at least one is a dict, compute the box + box† sum by merging the dict structures:
- Build a helper `_sum_matrix_element_values(a, b)` that handles combining two `MatrixElementValue`s:
  - `float + float` → `float` (existing behavior).
  - `float + dict` or `dict + float` → need to add the float to every leaf value in the dict. However, this scenario represents mixing a plane-independent amplitude with a plane-dependent one, which is unusual. For now, raise `ValueError` since it indicates malformed data. Include a `# TODO` comment in the code stating that this behavior should be reconsidered in the future, outlining the proposed logic: broadcast the float to every leaf of the dict (i.e., add the float to each leaf value).
  - `dict + dict` → merge by key, recursively summing values for matching keys, and preserving keys that appear only in one dict.

Update the docstring to reflect the new return type and dict handling.

### Step 3: Update `load_magnetic_hamiltonian`

Change the return type from `List[Tuple[str, str, float]]` to `Dict[Tuple[str, str], MatrixElementValue]`.

The function currently iterates over the list returned by `compute_all_rotations_from_just_box_terms` and encodes plaquette states to bit strings. Update it to:
1. Call `compute_all_rotations_from_just_box_terms` (which now returns a dict).
2. Build a new dict with the same structure, but with encoded (bit string) keys.
3. For the `mag_hamiltonian_matrix_element_threshold` filter: if the value is a `float`, apply the threshold as before. If the value is a `dict`, recurse into the dict structure and drop leaf `float` values whose absolute value is below the threshold. If all leaves are dropped for a given plane key, drop that plane key entirely. If all plane keys are dropped, drop the entire matrix element entry from the result dict.
4. For the `only_include_elems_connected_to_electric_vacuum` filter: apply as before, based on the encoded bit strings.

Remove the unused `forder` parameter from `load_magnetic_hamiltonian` (it was left over from a previous refactor and is not referenced in the function body).

### Step 4: Update `HamiltonianData` type alias in `circuit.py`

Change `HamiltonianData` from `List[Tuple[str, str, float]]` to `Dict[Tuple[str, str], MatrixElementValue]`. Import `MatrixElementValue` from `conventions.py`. Also define an `EncodedPlane` type alias (e.g., `EncodedPlane = Tuple[str, str]`) in `circuit.py` and use it for the encoded plane key in `HamiltonianData`, improving readability of the type signature.

### Step 5: Update `LatticeCircuitManager`

**`__init__`:** The small-lattice filtering logic (lines ~92-116) currently iterates over `self._mag_hamiltonian` as a list of `(str, str, float)` tuples. Update to iterate over the dict's items instead. The filtering (inconsistent controls / duplicate trimming) operates on decoded plaquette states, so the core logic doesn't change — just the iteration pattern and how the filtered result is built (now a dict instead of a list).

**New helper — `_resolve_hamiltonian_for_plaquette`:** Define a private method (or static method) that, given the full `HamiltonianData` dict plus the current plaquette's `plane` and `signature`, returns a `List[Tuple[str, str, float]]` — the traditional flat list of matrix elements. Resolution rules:
- If the value is a `float`, include it (it matches all planes/signatures).
- If the value is a `dict` keyed by plane:
  - Look up the current plane. If the plane key is not present, skip this matrix element.
  - If the value for that plane key is a `float`, include it (it matches all signatures for this plane).
  - If the value for that plane key is a `dict` keyed by signature, look up the current signature. If the signature key is not present, skip. Otherwise, include the float value.

**`apply_magnetic_trotter_step` and `_build_mag_evol_circuit`:** In `apply_magnetic_trotter_step`, for each plaquette, resolve the hamiltonian to a flat `List[Tuple[str, str, float]]` using `_resolve_hamiltonian_for_plaquette` with the current plaquette's plane and signature. Cache the resolved list and corresponding template circuit per unique `(plane, signature)` combination to avoid redundant circuit builds. Pass the resolved flat list to `_build_mag_evol_circuit`, which continues to work with the traditional flat format unchanged. For the common case where all matrix element values are plain floats, the resolution is a no-op and there is effectively one cache entry, preserving current performance.

### Step 6: Update tests in `test_conventions.py`

- **`test_load_magnetic_hamiltonian_constructs_correct_num_rotations`:** Update to match the new dict return type. The test currently checks against a list of 3-tuples; update to check against a dict with the same key/value structure.
- **`test_matrix_element_data_are_valid_*`:** These test `HAMILTONIAN_BOX_TERMS` directly and check `isinstance(mat_elem_val, (float, int))`. Update to also accept `dict` values (since `_load_hamiltonian` now preserves dicts).
- Add a new test for `compute_all_rotations_from_just_box_terms` with dict-valued matrix elements, verifying correct box + box† merging.

### Step 7: Update other downstream consumers

- **`run/functions.py`:** `initialize_lattice_tools` calls `load_magnetic_hamiltonian` and passes the result to `LatticeCircuitManager`. Since `LatticeCircuitManager` will accept the new dict type, this should work without changes to `functions.py` itself.
- **`tests/test_circuit.py`:** Tests that construct `LatticeCircuitManager` with hand-crafted hamiltonian data (e.g., `test_measure_plaquette_deduplicates_shared_registers`) need to use the new dict format. Tests that call `load_magnetic_hamiltonian` will get the new dict format automatically.
- **`tests/test_integration_mps.py`:** Uses `load_magnetic_hamiltonian` → `LatticeCircuitManager`. Should work after both are updated.
- **`archived-work/`:** Do not modify. This directory exists for historical reasons only.

---

## Execution Order and Dependencies

The items should be addressed in the following order:

### Phase A: Small items (no cross-dependencies)
- [x] **Small Item 1** — Rename `_flatten_hamiltonian_value` → `_normalize_hamiltonian_value`
- [x] **Small Item 2** — Audit refresh logic (correct — update docstrings on `get_data_metadata` and `LazyDict`)

### Phase B: Type aliases and core function changes
- [ ] **Complex Step 1** — Define `Signature` and `Plane` type aliases in `_abstract/lattice_data.py`; define `MatrixElementValue` in `conventions.py`
- [ ] **Complex Step 2** — Update `compute_all_rotations_from_just_box_terms` to return `Dict` and handle dict-valued matrix elements (implement `_sum_matrix_element_values` helper)
- [ ] **Complex Step 3** — Update `load_magnetic_hamiltonian` return type to `Dict[Tuple[str, str], MatrixElementValue]`; remove unused `forder` parameter

### Phase C: Circuit manager updates
- [ ] **Complex Step 4** — Update `HamiltonianData` type alias in `circuit.py`
- [ ] **Complex Step 5** — Update `LatticeCircuitManager`:
  - [ ] Update `__init__` small-lattice filtering to work with dict-typed hamiltonian
  - [ ] Implement `_resolve_hamiltonian_for_plaquette` helper
  - [ ] Update `_build_mag_evol_circuit` / `apply_magnetic_trotter_step` to use resolved hamiltonian with per-(plane, signature) caching

### Phase D: Test updates and verification
- [ ] **Complex Step 6** — Update tests in `test_conventions.py`
- [ ] **Complex Step 7** — Update tests in `test_circuit.py` and other downstream consumers
- [ ] Run full test suite (`uv run pytest -v`) to verify no regressions
