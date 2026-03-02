# Resolution plan for remove-mat-elem-issues.md

## Issue 1 (Major): Make `compute_signature` boundary-aware

### Problem summary

`Plaquette.compute_signature` is a static method that computes which half-link directions exist at each vertex using only `dim`, `plane`, and `forder`. It currently handles the d=3/2 top/bottom vertex constraint (discarding direction +2 or -2), but has no knowledge of **where on the lattice** a plaquette sits. For periodic lattices this is fine—every vertex has the same set of directions. For non-periodic lattices, vertices on the lattice boundary have fewer half-links (e.g., vertex (0,0) on a non-periodic d=3/2 lattice has no -1 direction because there is no neighbor to the left).

The same problem affects `compute_control_link_dirs_per_vertex`, which shares the same direction-existence logic.

### Approach: convert both static methods to instance properties

The `Plaquette.__init__` already receives a `lattice: LatticeData` (which inherits from `LatticeDef`) and a `bottom_left_vertex`. During `__init__`, the control-link construction loop (lines 206-219) already discovers exactly which links exist via `try/except KeyError`—so the plaquette instance already knows its actual control links. We can leverage this.

**Core idea:** Replace the two static methods with instance properties (`signature` and `control_link_dirs_per_vertex`) that compute the correct answer from the plaquette's actual vertex positions and the lattice geometry. The existing `_control_links` dict already encodes the ground truth for control links. For the signature (which includes active + control directions), we combine active directions with the control directions we actually found.

### Step-by-step plan

#### 1. Add a helper method `_existing_dirs_at_vertex` on `Plaquette`

Add a private method that, for a given vertex index (0-3), returns the set of half-link directions that actually exist at that vertex:

```python
def _existing_dirs_at_vertex(self, vertex_idx: int) -> set[int]:
```

Logic:
- Start with the active link directions for this vertex (known from the plane: `{e1, e2}`, `{-e1, e2}`, `{-e1, -e2}`, `{e1, -e2}` for vertices 0-3 respectively). Active links always exist because `Plaquette.__init__` would have raised an error otherwise (via `lattice.get_link`).
- Add the control link directions that were successfully resolved during `__init__` (i.e., the keys present in `self._control_links[vertex_vector]`).
- Return the union.

This gives us the definitive set of directions at each vertex, derived from the lattice geometry, for both periodic and non-periodic cases.

#### 2. Add `signature` property on `Plaquette`

```python
@property
def signature(self) -> Signature:
```

For each vertex (CCW from bottom-left), get `_existing_dirs_at_vertex(idx)`, sort by forder position, convert to tuple. Return as a 4-tuple.

This replaces the static `compute_signature` method.

#### 3. Add `control_link_dirs_per_vertex` property on `Plaquette`

```python
@property
def control_link_dirs_per_vertex(self) -> Tuple[Tuple[int, ...], ...]:
```

For each vertex, extract the direction labels from `self._control_links[vertex_vector].keys()`, sort by forder position, convert to tuple. Return as a 4-tuple.

This replaces the static `compute_control_link_dirs_per_vertex` method.

Note: this property already effectively exists as `control_links_per_vertex` (which returns the link *data*), but we need a version that returns just the *directions*.

#### 4. Remove the static methods

Delete both `compute_signature` and `compute_control_link_dirs_per_vertex` static methods. There are no external callers beyond this codebase, and this is alpha-stage software, so there is no backward-compatibility concern.

#### 5. Update call site in `circuit.py` (line 543)

Change:
```python
plaquette_signature: Signature = Plaquette.compute_signature(
    self._encoder.lattice_def.dim, plaquette_plane, self._encoder.forder
)
```
To:
```python
plaquette_signature: Signature = plaquette.signature
```

The `plaquette` instance is already available at this call site (it's iterated from `plaquettes` on line 540).

#### 6. Update call sites for `compute_control_link_dirs_per_vertex`

All call sites must be migrated to use the new `control_link_dirs_per_vertex` property on a `Plaquette` instance. The static method is being removed entirely.

1. **`circuit.py` line 512** (small periodic lattice d=2 optimization): This is called before any plaquette iteration and uses hardcoded `dim=2, plane=(1,2)`. Refactor to obtain control link directions from a `Plaquette` instance. Since this code is inside `apply_magnetic_trotter_step` which iterates over plaquettes shortly after, one approach is to construct a representative plaquette for this purpose, or move the computation into the per-plaquette loop and cache it after the first plaquette is processed.

2. **`circuit.py` lines 900, 944** (redundant control detection for small periodic lattices): These are inside methods that already have access to plaquette instances. Refactor to use `plaquette.control_link_dirs_per_vertex` from the plaquette being processed.

3. **`parsed_lattice_result.py` line 427**: This is used during bitstring decoding. Currently there is no `Plaquette` instance available here. Refactor to construct a `Plaquette` instance from the available lattice data and vertex information, then use its `control_link_dirs_per_vertex` property.

#### 7. Update tests

- **Rename `test_compute_signature` to `test_signature`**: Since the static method is being removed, the test should target the new `signature` property. Construct actual `Plaquette` instances (using periodic `LatticeData` objects for d=3/2 and d=2) and verify the property output matches the previously expected values.
- **Add a placeholder test for a non-periodic scenario**: Add a test marked with `@pytest.mark.skip` explaining that it should verify `plaquette.signature` correctly reflects missing directions at boundary vertices on a non-periodic lattice, once non-periodic `LatticeDef` support is available.

#### 8. Update `CLAUDE.md`

Update the `Plaquette` class description to document the new `signature` and `control_link_dirs_per_vertex` properties, and remove references to the deleted static methods.

### Files changed

| File | Changes |
|------|---------|
| `ymcirc/_abstract/lattice_data.py` | Add `_existing_dirs_at_vertex`, `signature` property, `control_link_dirs_per_vertex` property; remove both static methods |
| `ymcirc/circuit.py` | Update all `compute_signature` and `compute_control_link_dirs_per_vertex` call sites to use instance properties |
| `ymcirc/parsed_lattice_result.py` | Construct `Plaquette` instance for control link direction lookup |
| `tests/test_circuit.py` | Rename test; rewrite to test instance properties; add skipped non-periodic placeholder test |
| `ymcirc/CLAUDE.md` | Update `Plaquette` class description |

### Risks and considerations

- **Correctness for periodic lattices**: The new properties must produce identical results to the old static methods for all currently-supported periodic configurations. This is testable.
- **`Plaquette.__init__` control link discovery**: The `try/except KeyError` on line 218 silently skips links that don't exist. This is exactly the behavior we want—it means `_control_links` only contains directions that are actually valid. We rely on this being correct.
- **Active link existence**: We assume all 4 active links exist for any plaquette that was successfully constructed. This is guaranteed because `Plaquette.__init__` calls `lattice.get_link` for each active link without catching exceptions.
- **Performance**: Computing the signature from instance data (iterating over `_control_links` keys) is comparable in cost to the old static method. No performance concern.
- **`parsed_lattice_result.py` refactor**: Constructing a `Plaquette` instance where one didn't exist before adds some overhead and requires that the lattice data context is available. This should be the case since the encoder's lattice is accessible, but the exact construction path needs care.
- **Non-periodic `LatticeDef` support**: `_validate_lattice_params` currently raises `NotImplementedError` for non-bool `periodic_boundary_conds` (line 363). The new properties will be ready for non-periodic lattices once that restriction is lifted—no further changes to the signature/control-link-dirs logic itself should be needed.

---

## Issue 2 (Minor): Add nonstandard F-order test for `test_signature`

### Problem summary

The existing `test_compute_signature` (to be renamed `test_signature`) only tests the default forder `[1, 2, 3, -1, -2, -3]`. A nonstandard forder should also be tested to verify the sorting logic works correctly.

### Step-by-step plan

#### 1. Add test cases with a nonstandard forder

In `test_signature`, add assertions using `Plaquette` instances constructed from lattices with a nonstandard forder such as `[-1, 2, -3, 1, -2, 3]`:

- **d=3/2, plane (1, 2), forder `[-1, 2, -3, 1, -2, 3]`**:
  - v1, v2 (bottom): directions `{1, 2, -1}` sorted by this forder → `(-1, 2, 1)`
  - v3, v4 (top): directions `{1, -1, -2}` sorted by this forder → `(-1, 1, -2)`
  - Expected: `((-1, 2, 1), (-1, 2, 1), (-1, 1, -2), (-1, 1, -2))`

- **d=2, plane (1, 2), forder `[-1, 2, -3, 1, -2, 3]`**:
  - All vertices: directions `{1, 2, -1, -2}` sorted by this forder → `(-1, 2, 1, -2)`
  - Expected: `((-1, 2, 1, -2), (-1, 2, 1, -2), (-1, 2, 1, -2), (-1, 2, 1, -2))`

### Files changed

| File | Changes |
|------|---------|
| `tests/test_circuit.py` | Add nonstandard forder assertions to `test_signature` |

---

## Implementation order

1. Issue 1 steps 1-3 (add helper, `signature` property, `control_link_dirs_per_vertex` property)
2. Issue 1 step 4 (remove static methods)
3. Issue 1 steps 5-6 (update all call sites in `circuit.py` and `parsed_lattice_result.py`)
4. Issue 1 step 7 + Issue 2 (rename and rewrite tests, add nonstandard forder cases, add skipped non-periodic placeholder)
5. Issue 1 step 8 (update `CLAUDE.md`)
6. Run `uv run pytest -v` to verify all tests pass
