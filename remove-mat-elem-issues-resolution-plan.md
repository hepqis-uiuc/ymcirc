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

#### 4. Deprecate but keep the static methods [Feedback: There are no external callers beyond this codebase. Remove the static methods.]

Keep `compute_signature` and `compute_control_link_dirs_per_vertex` as static methods for backward compatibility (they still work correctly for periodic lattices). Add a deprecation note in their docstrings pointing to the new instance properties as the preferred API.

Alternatively, if there are no external callers beyond this codebase, they could be removed outright. The choice depends on whether external code uses them. Since this is alpha-stage, removing them is acceptable.

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

There are three call sites in the production code:

1. **`circuit.py` line 512** (small periodic lattice d=2 optimization): This is called before any plaquette iteration and uses hardcoded `dim=2, plane=(1,2)`. For periodic lattices, the static method is correct. However, to be consistent, we could construct a temporary plaquette or keep using the static method here since this code path is guarded by `self._lattice_is_periodic`. **Decision: leave this call site using the static method**, since it's explicitly for periodic lattices and doesn't need boundary awareness. [Feedback: remove the static method call. It's more work now, but will lead to better code in the future.]

2. **`circuit.py` lines 900, 944** (redundant control detection for small periodic lattices): Same situation—guarded by periodicity checks. **Leave as-is.** [Feedback: remove the static method call. It's more work now, but will lead to better code in the future.]

3. **`parsed_lattice_result.py` line 427**: This is used during bitstring decoding. Here the plaquette is being reconstructed from measurement data, not from a `Plaquette` instance. We don't have a `Plaquette` instance available. **Two options:**
   - (a) Keep using the static method here, which is safe because `ParsedLatticeResult` currently only supports periodic lattices.
   - (b) Refactor to construct a temporary `Plaquette` instance and use the property. [Feedback: use option b.]

   **Decision: use option (a) for now**, with a TODO comment noting that this will need updating when non-periodic lattice support is added to `ParsedLatticeResult`.

#### 7. Update tests

- **Modify `test_compute_signature`**: Update to also test the new `signature` property by constructing actual `Plaquette` instances. Verify that the property produces the same result as the static method for periodic lattices.
- **Add a test for a non-periodic scenario**: Construct a non-periodic d=3/2 lattice (once `LatticeDef` supports non-periodic boundary conditions) and verify that `plaquette.signature` correctly reflects the missing directions at boundary vertices. If non-periodic `LatticeDef` support is not yet available, add a placeholder test marked with `@pytest.mark.skip` with a note explaining what it should test. [Feedback: you will have to do the placeholder test option because non-periodic `LatticeDef` support isn't available yet.]

#### 8. Update `CLAUDE.md`

Update the `Plaquette` class description to document the new `signature` and `control_link_dirs_per_vertex` properties.

### Files changed

| File | Changes |
|------|---------|
| `ymcirc/_abstract/lattice_data.py` | Add `_existing_dirs_at_vertex`, `signature` property, `control_link_dirs_per_vertex` property; deprecate or remove static methods |
| `ymcirc/circuit.py` | Update `compute_signature` call (line 543) to use `plaquette.signature` |
| `tests/test_circuit.py` | Update `test_compute_signature` to also test new property; add non-periodic placeholder test |
| `ymcirc/CLAUDE.md` | Update `Plaquette` class description |

### Risks and considerations

- **Correctness for periodic lattices**: The new property must produce identical results to the static method for all currently-supported periodic configurations. This is testable.
- **`Plaquette.__init__` control link discovery**: The `try/except KeyError` on line 218 silently skips links that don't exist. This is exactly the behavior we want—it means `_control_links` only contains directions that are actually valid. We rely on this being correct.
- **Active link existence**: We assume all 4 active links exist for any plaquette that was successfully constructed. This is guaranteed because `Plaquette.__init__` calls `lattice.get_link` for each active link without catching exceptions.
- **Performance**: Computing the signature from instance data (iterating over `_control_links` keys) is comparable in cost to the static method. No performance concern.
- **Non-periodic `LatticeDef` support**: `_validate_lattice_params` currently raises `NotImplementedError` for non-bool `periodic_boundary_conds` (line 363). The signature property will be ready for non-periodic lattices once that restriction is lifted—no further changes to the signature logic itself should be needed.

---

## Issue 2 (Minor): Add nonstandard F-order test for `test_compute_signature` [Feedback: I suppose this test should be renamed since we are removing the `compute_signature` static method in favor of a `signature` property.]

### Problem summary

`test_compute_signature` only tests the default forder `[1, 2, 3, -1, -2, -3]`. A nonstandard forder should also be tested to verify the sorting logic works correctly.

### Step-by-step plan

#### 1. Add test cases with a nonstandard forder

In `test_compute_signature`, add assertions for a nonstandard forder such as `[-1, 2, -3, 1, -2, 3]`:

- **d=3/2, plane (1, 2), forder `[-1, 2, -3, 1, -2, 3]`**:
  - v1, v2 (bottom): directions `{1, 2, -1}` sorted by this forder → `(-1, 2, 1)`
  - v3, v4 (top): directions `{1, -1, -2}` sorted by this forder → `(-1, 1, -2)`
  - Expected: `((-1, 2, 1), (-1, 2, 1), (-1, 1, -2), (-1, 1, -2))`

- **d=2, plane (1, 2), forder `[-1, 2, -3, 1, -2, 3]`**:
  - All vertices: directions `{1, 2, -1, -2}` sorted by this forder → `(-1, 2, 1, -2)`
  - Expected: `((-1, 2, 1, -2), (-1, 2, 1, -2), (-1, 2, 1, -2), (-1, 2, 1, -2))`

#### 2. Also test the new `signature` property with nonstandard forder

If the new property from Issue 1 has been implemented, construct `Plaquette` instances using a `LatticeDef` with the nonstandard forder and verify the property output matches.

### Files changed

| File | Changes |
|------|---------|
| `tests/test_circuit.py` | Add nonstandard forder assertions to `test_compute_signature` |

---

## Implementation order

1. Issue 1 steps 1-6 (add properties, update call sites)
2. Issue 2 (add nonstandard forder tests)
3. Issue 1 steps 7-8 (update tests and docs)
4. Run `uv run pytest -v` to verify all tests pass
