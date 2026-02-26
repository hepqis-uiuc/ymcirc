# Issues round 3 resolution plan

## Findings

The performance bottleneck in `ParsedLatticeResult` construction is real and was confirmed by profiling. The **dominant cost** is the `copy.deepcopy(lattice_encoder)` call on line 103 of `parsed_lattice_result.py`, which deep-copies the entire `LatticeStateEncoder` — including its internal copy of the `physical_plaquette_states` list — for every single `ParsedLatticeResult` instance. For the d=2, T1 configuration referenced in the issue, this list contains **15,715 entries** (nested tuples), making the deep copy extremely expensive. The **second largest cost** is the `encoder.__repr__()` call on line 104, which serializes the entire plaquette states list into a string representation, also for every instance. Together, these two operations account for over 99% of the per-instance construction time.

Component-level benchmarks (averaged over 100 runs per configuration) confirm this conclusively:

| Component | d=3/2, T1, size=2 (81 plaq. states) | d=2, T1, size=2 (15,715 plaq. states) | d=2, T1, size=4 (15,715 plaq. states) |
|---|---|---|---|
| `copy.deepcopy(encoder)` | 0.9 ms | **219 ms** | **217 ms** |
| `encoder.__repr__()` | 0.2 ms | **42 ms** | **44 ms** |
| `copy.deepcopy(lattice_def)` | 0.02 ms | 0.02 ms | 0.06 ms |
| `LatticeDef.__init__()` | 0.004 ms | 0.004 ms | 0.007 ms |
| `get_traversal_order()` | 0.002 ms | 0.002 ms | 0.006 ms |
| Decode loop (string parsing) | 0.003 ms | 0.003 ms | 0.013 ms |
| **Full PLR construction** | **1.2 ms** | **260 ms** | **261 ms** |

Several conclusions follow from the data:

1. **The cost is dominated by `copy.deepcopy(encoder)` and `encoder.__repr__()`**, not by the lattice traversal/decode loop or `LatticeDef.__init__()`. The decode loop contributes only ~0.003 ms even for the large configuration — completely negligible.
2. **The cost scales with the number of physical plaquette states, not with lattice size.** Going from size=2 (20 bits, 4 vertices, 8 links) to size=4 (80 bits, 16 vertices, 32 links) in d=2, T1 produces virtually no change in construction time (260 ms vs 261 ms), because both use the same 15,715-entry plaquette states list. Meanwhile, going from d=3/2, T1 (81 plaquette states) to d=2, T1 (15,715 plaquette states) at the same lattice size causes a 220x slowdown.
3. **Other costs are negligible.** `copy.deepcopy(lattice_def)`, `LatticeDef.__init__()`, `get_traversal_order()`, and the decode loop together contribute less than 0.1 ms in all configurations. The `__hash__` deep-copy issue (via the `self.lattice_def` property) is real but minor in comparison, since it copies the `LatticeDef` (~0.02 ms), not the encoder.

For the specific configuration in the issue (d=2, T1, size=2, 10,000 shots), even if only ~1,000 unique bit strings appear, constructing `ParsedLatticeResult` for each one takes ~260 ms, yielding a total of ~260 seconds — consistent with the observed "hang."

## Proposed solution

The primary fix is to **stop deep-copying the `LatticeStateEncoder`** into each `ParsedLatticeResult` instance. The encoder is shared across all instances in a simulation run and is never mutated by `ParsedLatticeResult`, so storing a direct reference is safe. The `__repr__` string (line 104) should be computed once and cached on the encoder itself, rather than regenerated from the full encoder for each instance. Together, these two changes eliminate ~261 ms of the ~261 ms per-instance cost for d=2, T1 — effectively reducing construction time to sub-millisecond.

As a secondary measure, caching `ParsedLatticeResult` instances at the call site in `functions.py` (the user's proposed approach (a)) should also be adopted to avoid redundant construction of the same bit string across different simulation time steps. The user's proposed approach (b) of optimizing the init method is validated by the profiling data, but the specific optimization needed is different from what might be expected: it is the `copy.deepcopy(encoder)` and `__repr__` calls that must be eliminated, not the lattice traversal/decode loop. Other minor improvements (`__hash__` caching) are worth doing for correctness and hygiene but will have negligible impact on the observed bottleneck.

## Detailed implementation plan

### Stage 1: Eliminate `copy.deepcopy(encoder)` and `copy.deepcopy(lattice_def)` in `__init__` (lines 102–103)

This is the highest-impact change. Replace `copy.deepcopy(lattice_encoder)` with a direct reference assignment (`self._encoder = lattice_encoder`), and likewise for `self._lattice_def`. The `lattice_def` property (line 117–119) already returns a deep copy to external callers, preserving the public API contract that external code gets its own copy. Internally, `ParsedLatticeResult` never mutates the encoder or lattice def, so a shared reference is safe.

The same change should be applied to the `_create_partial` factory method (lines 285–286), which also deep-copies both objects.

**Files affected:** `ymcirc/parsed_lattice_result.py`

### Stage 2: Cache `encoder.__repr__()` on the encoder itself

The `__repr__()` call on line 104 serializes the full plaquette states list into a string for every instance, costing ~42 ms for d=2, T1. Since the encoder is shared and immutable, this string is identical across all instances. Cache the repr string on the `LatticeStateEncoder` itself (e.g., as `encoder._repr_cache`), computing it once on first access and reusing it across all `ParsedLatticeResult` instances.

**Files affected:** `ymcirc/conventions.py`, `ymcirc/parsed_lattice_result.py`

### Stage 3: Fix `__hash__` to avoid `lattice_def` property deep copy

The `__hash__` method on line 242 accesses `self.lattice_def`, which is a property that performs a `copy.deepcopy`. While this is a minor cost (~0.02 ms) compared to the encoder deep copy, it is still unnecessary and adds up when `ParsedLatticeResult` is used as a dictionary key. Replace `self.lattice_def` with `self._lattice_def` in `__hash__`. Additionally, cache the hash value in `self._hash_cache` so repeated hashing is O(1).

**Files affected:** `ymcirc/parsed_lattice_result.py`

### Stage 4: Add bit-string-level caching in `functions.py`

In the `run_circuit_simulations` function (lines 432–447), introduce a dictionary `plr_cache: dict[str, ParsedLatticeResult]` that maps bit strings to their parsed results. Before constructing a new `ParsedLatticeResult`, check the cache. This avoids re-parsing the same bit string across different simulation time steps.

**Files affected:** `run/functions.py`

### Stage 5: Minor optimizations (low priority)

Fix `__eq__` to use `self._lattice_def` directly rather than going through properties.

**Files affected:** `ymcirc/parsed_lattice_result.py`

### Stage 6: Testing and validation

Run the existing test suite to confirm no regressions. Then run the benchmark scripts (`benchmark_plr.py` and `benchmark_plr2.py`) to verify the performance improvement.

**Files affected:** `tests/` (existing test files)

## Concrete TODO items

- [ ] **Remove `copy.deepcopy(lattice_encoder)` on line 103 of `parsed_lattice_result.py`:** Replace with `self._encoder = lattice_encoder`.
- [ ] **Remove `copy.deepcopy(lattice_encoder.lattice_def)` on line 102:** Replace with `self._lattice_def = lattice_encoder.lattice_def` (noting this itself returns a deep copy from the property — consider using `lattice_encoder._lattice` directly or storing the reference once).
- [ ] **Cache `encoder.__repr__()` on the encoder:** Add a `_repr_cache` attribute to `LatticeStateEncoder` in `conventions.py`, compute it once on first access, and use it in `ParsedLatticeResult.__init__` (line 104) and `_create_partial`.
- [ ] **Apply the same deep-copy fixes to `_create_partial` (lines 285–286):** Remove deep copies of encoder and lattice_def in the factory method.
- [ ] **Fix `__hash__` (line 242):** Replace `self.lattice_def` (property with deep copy) with `self._lattice_def` (direct reference). Add `self._hash_cache` attribute.
- [ ] **Fix `__eq__` (lines 244–253):** Use `self._lattice_def` directly instead of going through properties.
- [ ] **Add `plr_cache` dict in `run_circuit_simulations`:** Before the loop over `job_results` on line 432, create `plr_cache = {}`. Use cached `ParsedLatticeResult` instances where available when building `counts_dict_big_endian`.
- [ ] **Run existing tests:** Execute `uv run pytest -v` to verify no regressions.
- [ ] **Run benchmark scripts:** Execute `benchmark_plr.py` and `benchmark_plr2.py` to confirm the performance improvement.
