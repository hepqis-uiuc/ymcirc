# Plan: Remove matrix element merging, enforce plane+signature filtering

## Summary

All Hamiltonian matrix element data files have already been updated so that every value is a nested dict: `{Plane: {Signature: float}}`. The codebase still treats `MatrixElementValue` as `Union[float, Dict[...]]`, with many branches handling the bare-float case. This plan removes the float case, simplifies the type, and ensures that `_resolve_hamiltonian_for_plaquette` correctly filters by plane and signature using a geometry-derived signature that matches the data file format.

## Key finding: Signature type mismatch

The data files encode signatures as **all existing half-link directions per vertex** (sorted by forder), e.g.:
- d=3/2: `((1, 2, -1), (1, 2, -1), (1, -1, -2), (1, -1, -2))`
- d=2: `((1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2))`

The current `Signature` type alias (`lattice_data.py:24`) defines inner tuples as `Tuple[int, int, int]` (i-weight shape), and the docstring says "i-weight tuples". This is wrong — signatures in the data are direction-label tuples of variable length.

Additionally, `Plaquette.compute_control_link_dirs_per_vertex` returns only control (non-active) directions, not ALL existing directions per vertex:
- d=3/2: `((-1,), (1,), (1,), (-1,))` — 1 element per vertex
- d=2: `((-1, -2), (1, -2), (1, 2), (2, -1))` — 2 elements per vertex

So neither the type alias nor the existing static method matches the data file format.

In `circuit.py:553`, the code currently does `plaquette_signature: Signature = plaquette.control_links_per_vertex`, which on a `LatticeRegisters` plaquette returns `QuantumRegister` tuples — a third mismatch. This path was previously only reached for dict-valued entries, which didn't exist in the old data files.

---

## Step-by-step plan

### 1. Update `Signature` type alias and add `compute_signature`

**File: `ymcirc/_abstract/lattice_data.py`**

- **Update `Signature`** type alias: change inner tuples from `Tuple[int, int, int]` to `Tuple[int, ...]` and update the docstring to describe direction-label tuples (not i-weight tuples).

- **Add `Plaquette.compute_signature(dim, plane, forder)`** static method: similar to `compute_control_link_dirs_per_vertex`, but returns ALL existing half-link directions at each vertex (including active link directions), sorted by forder. This output will match the signature keys in the data files. Implementation: for each of the 4 vertices, compute the existing directions (applying the d=3/2 bottom/top vertex restriction), sort by forder position, and return as a 4-tuple.

### 2. Simplify `MatrixElementValue` and remove float branches in `conventions.py`

**File: `ymcirc/conventions.py`**

- **Change `MatrixElementValue`** from `Union[float, Dict[Plane, Union[float, Dict[Signature, float]]]]` to `Dict[Plane, Dict[Signature, float]]`. Every matrix element value is now always a two-level nested dict.

- **Remove `_normalize_hamiltonian_value`**: the simplified data type means we no longer need to normalize loaded data. Remove the function and update `_load_hamiltonian` to pass dict values through directly.

- **Simplify `_filter_matrix_element_value`**: remove the float branch at the top (`isinstance(value, (int, float))`). Remove the intermediate float branch for plane values (`isinstance(plane_val, (int, float))`). The function should only handle `Dict[Plane, Dict[Signature, float]]`.

- **Simplify `_sum_matrix_element_values`**: remove the `float + float` case, the `float + dict` error case, and the inner `float + float` case for plane values. The function should only handle `dict + dict` at both the plane level and the signature level. Consider whether a `0` default in `compute_all_rotations_from_just_box_terms` (for missing box/box-dagger amplitudes) should become `{}` instead.

- **Update `compute_all_rotations_from_just_box_terms`**: the `box_terms.get((state_1, state_2), 0)` default of `0` (a float) will no longer be valid since `_sum_matrix_element_values` won't accept floats. Change the default to `{}` (empty dict) and update `_sum_matrix_element_values` to handle the empty-dict identity case.

- **Update `load_magnetic_hamiltonian`**: the threshold filtering call to `_filter_matrix_element_value` stays as-is (its internal simplification is handled by the bullet above). Update the logger message since not all entries produce Givens rotations (some are filtered by plane/signature downstream). Update docstring.

### 3. Update `_resolve_hamiltonian_for_plaquette` and `apply_magnetic_trotter_step` in `circuit.py`

**File: `ymcirc/circuit.py`**

- **Simplify `_resolve_hamiltonian_for_plaquette`**: remove the `isinstance(value, (int, float))` branch. The method should now only handle dict-valued entries: look up the plane, then look up the signature within that plane's sub-dict. Skip entries where either lookup misses.

- **Remove the `universal_resolved` fast path** in `apply_magnetic_trotter_step` (lines 490-498). Since all values are now dicts, `has_dict_values` is always True. Every plaquette must go through `_resolve_hamiltonian_for_plaquette` to get only matching matrix elements.

- **Fix the signature computation** (line 553). Replace:
  ```python
  plaquette_signature: Signature = plaquette.control_links_per_vertex
  ```
  with a call to the new `Plaquette.compute_signature(dim, plane, forder)` to produce a direction-label signature matching the data file format. The `dim`, `plane`, and `forder` are available from `self._encoder.lattice_def.dim`, `plaquette.plane`, and `self._encoder.forder` respectively.

- **Update docstrings** on `apply_magnetic_trotter_step`, `_build_mag_evol_circuit`, and `_resolve_hamiltonian_for_plaquette` to reflect that all matrix element values are now dicts.

- **Update the cache key logic**: currently `cache_key = None` for the universal (all-float) case. Remove that branch. The cache key is always `(plaquette_plane, plaquette_signature)`.

### 4. Update the `Signature` type usage in the `__init__` of `LatticeCircuitManager`

**File: `ymcirc/circuit.py`**

- In `__init__`, the type hint for `self._cached_mag_evol_circuits` is `Dict[Tuple[Plane, Signature] | None, QuantumCircuit]`. Remove the `| None` since there's no longer a universal cache entry.

### 5. Update CLAUDE.md

**File: `ymcirc/CLAUDE.md`**

- Update the `MatrixElementValue` bullet in the "Key Domain Concepts" section to reflect that it is now always `Dict[Plane, Dict[Signature, float]]`.
- Update the `Signatures and planes` bullet to clarify that `Signature` now uses direction-label tuples (not i-weight tuples).

### 6. Update tests

**File: `ymcirc/tests/test_conventions.py`**

- **`test_compute_all_rotations_handles_dict_valued_matrix_elements`** (line 145): this test exercises `_sum_matrix_element_values` with both float-valued and dict-valued entries. Update it to only use dict-valued entries (remove the float test branch). Update assertions.

- **`test_load_magnetic_hamiltonian_constructs_correct_num_rotations`** (line 103): this test uses bare-float box terms. Update to use nested dict box terms instead.

- **`test_matrix_element_data_are_valid_*`** (lines 195, 214, 233): these tests check `isinstance(mat_elem_val, (float, int, dict))`. Tighten to check that every value is a `dict` (never a float/int).

- **Add a unit test for `_filter_matrix_element_value`** with the simplified dict-only structure.

- **Add a unit test for `_sum_matrix_element_values`** with dict-only structure, including the empty-dict identity case.

**File: `ymcirc/tests/test_circuit.py`**

- **Add a unit test for `_resolve_hamiltonian_for_plaquette`** that verifies: (a) entries matching the target plane+signature are included, (b) entries with wrong plane are excluded, (c) entries with right plane but wrong signature are excluded.

- **Add a unit test for `Plaquette.compute_signature`** for d=3/2 and d=2 with the default forder, verifying the output matches the known data file signature format.

- Existing integration tests (`test_apply_magnetic_trotter_step_*`) should continue to pass with no changes (or minimal fixture updates) since the data files have already been updated.

### 7. Verify

- Run `uv run pytest -v` to confirm all tests pass.
- Run `uv run pytest --runslow` to also confirm slow tests pass.

---

## Files changed (summary)

| File | Changes |
|------|---------|
| `ymcirc/_abstract/lattice_data.py` | Update `Signature` type alias; add `compute_signature` |
| `ymcirc/conventions.py` | Simplify `MatrixElementValue`; remove `_normalize_hamiltonian_value`; simplify `_filter_matrix_element_value`, `_sum_matrix_element_values`, `compute_all_rotations_from_just_box_terms`, `load_magnetic_hamiltonian` |
| `ymcirc/circuit.py` | Simplify `_resolve_hamiltonian_for_plaquette`; remove `universal_resolved` fast path; fix signature computation; update cache key types; update docstrings |
| `ymcirc/CLAUDE.md` | Update domain concept descriptions |
| `tests/test_conventions.py` | Update existing tests; add unit tests for `_filter_matrix_element_value`, `_sum_matrix_element_values` |
| `tests/test_circuit.py` | Add unit tests for `_resolve_hamiltonian_for_plaquette`, `compute_signature` |

---

## Todo list

- [x] 1. Update `Signature` type alias in `_abstract/lattice_data.py` (change inner tuples to `Tuple[int, ...]`, update docstring)
- [x] 2. Add `Plaquette.compute_signature` static method in `_abstract/lattice_data.py`
- [x] 3. Simplify `MatrixElementValue` type alias in `conventions.py`
- [x] 4. Remove `_normalize_hamiltonian_value` in `conventions.py`; update `_load_hamiltonian` to pass dicts through directly
- [x] 5. Simplify `_filter_matrix_element_value` in `conventions.py`
- [x] 6. Simplify `_sum_matrix_element_values` in `conventions.py` (handle empty-dict identity)
- [x] 7. Update `compute_all_rotations_from_just_box_terms` in `conventions.py` (change default from `0` to `{}`)
- [x] 8. Update `load_magnetic_hamiltonian` docstring in `conventions.py`
- [x] 9. Simplify `_resolve_hamiltonian_for_plaquette` in `circuit.py`
- [x] 10. Remove `universal_resolved` fast path in `apply_magnetic_trotter_step` in `circuit.py`
- [x] 11. Fix signature computation in `apply_magnetic_trotter_step` (use `compute_signature`)
- [x] 12. Update cache key type hint in `LatticeCircuitManager.__init__`
- [x] 13. Update docstrings in `circuit.py`
- [x] 14. Update CLAUDE.md domain concept descriptions
- [x] 15. Update `test_compute_all_rotations_handles_dict_valued_matrix_elements` in `test_conventions.py`
- [x] 16. Update `test_load_magnetic_hamiltonian_constructs_correct_num_rotations` in `test_conventions.py`
- [x] 17. Update `test_matrix_element_data_are_valid_*` tests in `test_conventions.py`
- [x] 18. Add unit test for `_filter_matrix_element_value` (dict-only)
- [x] 19. Add unit test for `_sum_matrix_element_values` (dict-only, including empty-dict identity)
- [x] 20. Add unit test for `_resolve_hamiltonian_for_plaquette` in `test_circuit.py`
- [x] 21. Add unit test for `compute_signature` in `test_circuit.py`
- [x] 22. Run `uv run pytest -v` and verify all tests pass
- [x] 23. Run `uv run pytest --runslow` and verify slow tests pass
