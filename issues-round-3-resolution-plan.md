# Issues round 3 resolution plan

## Findings

The performance bottleneck in `ParsedLatticeResult` construction is real and stems from multiple compounding factors. The most significant cost is the **two `copy.deepcopy()` calls** on lines 102–103 of `parsed_lattice_result.py`, which deep-copy both the `LatticeDef` and the entire `LatticeStateEncoder` (including its internal bitmaps and plaquette state lists) for *every single* `ParsedLatticeResult` instance. Since these objects are identical across all bit strings in a given simulation run, this is pure waste. The `super().__init__()` call is also non-trivial: it invokes `LatticeDef.__init__()`, which rebuilds the full set of vertex addresses and link addresses (via `_configure_lattice`) using `itertools.product` and set operations for every instance, even though the lattice geometry is the same for all results.

A second, subtler problem exists in the `__hash__` and `__eq__` methods (lines 240–253). The `__hash__` method accesses `self.lattice_def`, which is a *property* that performs yet another `copy.deepcopy` of the internal `LatticeDef` every time it is called. Since `ParsedLatticeResult` instances are used as dictionary keys in `MeasurementResults._counts`, hashing occurs frequently — on every insertion and every lookup — making this an especially costly oversight. Similarly, `__eq__` accesses `self.shape` and `self.periodic_boundary_conds` through properties, though these are less expensive than the deep copy. The `get_traversal_order()` call in the init loop (line 74) also reconstructs the traversal list from scratch on each call rather than caching it, though this is a minor cost compared to the deep copies.

## Proposed solution

The solution has two parts: (1) eliminate redundant object construction and copying within `ParsedLatticeResult`, and (2) introduce caching at the call site in `functions.py` to avoid constructing duplicate `ParsedLatticeResult` instances for the same bit string across different simulation time steps. For part (1), the key changes are: stop deep-copying the `LatticeStateEncoder` and `LatticeDef` into each instance (store shared references instead, or cache at class level), cache the hash value after first computation, and fix `__hash__` to avoid calling the `lattice_def` property (which deep-copies). The `LatticeDef` base class initialization can also be optimized by caching the traversal order and avoiding redundant set/list construction when the lattice geometry hasn't changed. For part (2), a simple dictionary mapping bit strings to already-constructed `ParsedLatticeResult` instances can be maintained across the loop in `run_circuit_simulations`, so that each unique bit string is parsed only once per simulation run.

The user's proposed approach (a) of caching `ParsedLatticeResult` instances at the `functions.py` level is sound and should be adopted. Approach (b) of improving the init method itself is also critical — even with caching at the call site, the first construction of each unique bit string must be fast. With 10,000 shots on a d=2, size-2 lattice, there can be on the order of hundreds to thousands of unique bit strings, so the per-construction cost still matters.

## Detailed implementation plan

### Stage 1: Fix `__hash__` and `__eq__` to avoid deep copies

The `__hash__` method on line 242 accesses `self.lattice_def`, which triggers a `copy.deepcopy`. This should instead access `self._lattice_def` directly (or use cached dimension/shape/boundary values). Additionally, `__hash__` should cache its result in an instance attribute (e.g., `self._hash_cache`) so that repeated hashing (which happens constantly when `ParsedLatticeResult` is used as a dict key) is O(1) after the first call. The `__eq__` method is less problematic but should also use `self._lattice_def` directly rather than going through properties that copy.

**Files affected:** `ymcirc/parsed_lattice_result.py`

### Stage 2: Eliminate redundant `copy.deepcopy` in `__init__`

Lines 102–103 deep-copy both the `LatticeDef` and the `LatticeStateEncoder` into each instance. Since these objects are shared across all `ParsedLatticeResult` instances in a simulation run and are never mutated by `ParsedLatticeResult`, these copies are unnecessary. The `_lattice_def` and `_encoder` attributes should store the original references (or at most a shallow copy). The `lattice_def` property (line 117–119) already returns a deep copy to external callers, so internal immutability is preserved. The same change should be applied to the `_create_partial` factory method (lines 285–286) which also deep-copies.

**Files affected:** `ymcirc/parsed_lattice_result.py`

### Stage 3: Optimize `LatticeDef.__init__` overhead

Each `ParsedLatticeResult` construction calls `super().__init__()` which calls `LatticeDef.__init__()`. This rebuilds the vertex address set via `itertools.product` and the link address set via nested loops every time. Since all `ParsedLatticeResult` instances for a given simulation share the same lattice geometry, this work is redundant. Two sub-options:

- **Option A (preferred):** Bypass `LatticeDef.__init__` for `ParsedLatticeResult` entirely by copying the pre-computed lattice data (vertex addresses, link addresses, shape, etc.) from the encoder's `LatticeDef` directly. This can be done by having `ParsedLatticeResult.__init__` call `LatticeDef.__init__` only if needed, or by introducing a lightweight "from existing lattice def" path.
- **Option B:** Cache `LatticeDef` configurations at the class level keyed by (dim, size, periodic_boundary_conds, forder) so that repeated construction reuses precomputed data.

**Files affected:** `ymcirc/_abstract/lattice_data.py`, `ymcirc/parsed_lattice_result.py`

### Stage 4: Cache `get_traversal_order()` results

`get_traversal_order()` in `LatticeDef` (line 517) recomputes the traversal list from scratch on every call. Since the traversal order depends only on lattice geometry (which is immutable after construction), the result should be cached as an instance attribute on first computation using a simple `if self._traversal_order_cache is None` pattern.

**Files affected:** `ymcirc/_abstract/lattice_data.py`

### Stage 5: Add bit-string-level caching in `functions.py`

In the `run_circuit_simulations` function (lines 432–447), introduce a dictionary `plr_cache: dict[str, ParsedLatticeResult]` that maps bit strings to their parsed results. Before constructing a new `ParsedLatticeResult`, check the cache. This avoids re-parsing the same bit string across different simulation time steps (and within the same time step, since `MeasurementResults.__init__` already deduplicates via dict keys). The cache should be keyed by the big-endian bit string.

Alternatively (or additionally), the caching could be pushed into `MeasurementResults.__init__` itself, so that any caller benefits — not just `run_circuit_simulations`. This would involve `MeasurementResults.__init__` accepting an optional `plr_cache` dict parameter.

**Files affected:** `run/functions.py`, optionally `ymcirc/measurement_results.py`

### Stage 6: Testing and validation

Run the existing test suite to confirm no regressions. Then run the `run/time_evol.py` script with the parameters from the issue description and verify that the hang is resolved. Ideally, add a targeted benchmark or test that constructs many `ParsedLatticeResult` instances to guard against future performance regressions.

**Files affected:** `tests/` (new or modified test files)

## Concrete TODO items

1. **Fix `__hash__` deep copy:** Replace `self.lattice_def` with `self._lattice_def` in `__hash__`, and add a `self._hash_cache` attribute that is computed once and returned on subsequent calls.
2. **Fix `__eq__` property access:** Replace `self.dim`, `self.shape`, `self.periodic_boundary_conds` in `__eq__` with direct attribute access to avoid any unnecessary overhead.
3. **Remove `copy.deepcopy` in `__init__` lines 102–103:** Store `lattice_encoder.lattice_def` and `lattice_encoder` as direct references instead of deep copies.
4. **Remove `copy.deepcopy` in `_create_partial` lines 285–286:** Same change as above for the factory method path.
5. **Add lightweight init path for `ParsedLatticeResult`:** Modify `__init__` to copy precomputed lattice data (vertex set, link set, shape, etc.) from the encoder's `LatticeDef` rather than recomputing via `LatticeDef.__init__`.
6. **Cache `get_traversal_order()` result:** Add a `_traversal_order_cache` attribute to `LatticeDef`, populate it on first call, and return it on subsequent calls.
7. **Add `plr_cache` dict in `run_circuit_simulations`:** Before the loop over `job_results` on line 432, create `plr_cache = {}`. Inside the loop, build `counts_dict_big_endian` using cached `ParsedLatticeResult` instances where available.
8. **Optionally add `plr_cache` parameter to `MeasurementResults.__init__`:** Allow callers to pass in and share a cache of `ParsedLatticeResult` instances to further reduce redundant construction.
9. **Run existing tests:** Execute `uv run pytest -v` to verify no regressions.
10. **Run `time_evol.py` benchmark:** Execute the script with the parameters from the issue and confirm the performance improvement.
11. **Add performance regression test (optional):** Write a test that constructs a large number of `ParsedLatticeResult` instances and asserts it completes within a reasonable time bound.
