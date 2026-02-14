# Refactor ymcirc for F-ordered pyclebsch JSON Data

## Context

pyclebsch now generates JSON data files (plaquette states and magnetic Hamiltonian matrix elements) using a configurable half-link ordering convention called **FORDER**. The default is `[1, 2, 3, -1, -2, -3]`. ymcirc needs to be updated so that (1) it can load the new `.json.gz` file format with its metadata wrapper and nested key structure, and (2) the control link ordering within plaquettes respects FORDER. The data files in `ymcirc/_ymcirc_data/` have already been updated to the new format.

**Scope**: F-order support for d=3/2 and d=2 with periodic BCs only. No d=3, non-periodic BC, or B-truncation changes.

---

## Category 1: JSON Loading Updates

### 1A. Update `json_loader` in `utilities.py`

**File**: `ymcirc/ymcirc/utilities.py`

The data files are now `.json.gz` (gzip-compressed) with a metadata wrapper:
```json
{"metadata": {..., "f_order": [1, 2, 3, -1, -2, -3]}, "data": <list-or-dict>}
```

Update `json_loader` to:
- Open with `gzip.open` instead of `Path.open`
- Extract the `"metadata"` and `"data"` keys from the top-level dict
- Apply `ast.literal_eval` only to the `"data"` portion (same as current)
- Return a 2-tuple `(data, metadata)` instead of just data

Add `import gzip` to the module.

### 1B. Update file path extensions in `conventions.py`

**File**: `ymcirc/ymcirc/conventions.py`

Update `_HAMILTONIAN_DATA_FILE_PATHS` and `_PLAQUETTE_STATES_DATA_FILE_PATHS`:
- Change all `.json` suffixes to `.json.gz`

### 1C. Handle new Hamiltonian data format

**File**: `ymcirc/ymcirc/conventions.py` (new helper) and `ymcirc/ymcirc/utilities.py`

The new Hamiltonian JSON `"data"` has a nested key structure:
```
{str(Pf, Pi): {str(plane): {str(signature): float | merged_float} | merged_float} | merged_float}
```

After `literal_eval` on the outer keys, each value can be:
- A `float` (fully merged across all planes and signatures)
- A `dict` where values are either `float` (merged across signatures within a plane) or `dict[tuple, float]` (unmerged)

Create a helper function (e.g., `_flatten_hamiltonian_value(value) -> float`) that:
- If `value` is a `float`, return it directly
- If `value` is a `dict`, recursively extract the float value(s)
- If values disagree across planes/signatures, raise a `ValueError` (this shouldn't happen for periodic lattices with a single plane, but is a safety check)

Create specialized loader functions that wrap `json_loader`:
- `_load_plaquette_states(path) -> tuple[list[PlaquetteState], dict]`: Calls `json_loader`, returns (list of PlaquetteStates, metadata)
- `_load_hamiltonian(path) -> tuple[dict[(PlaquetteState, PlaquetteState), float], dict]`: Calls `json_loader`, flattens nested values, returns (flat dict, metadata)

### 1D. Update `LazyDict` usage in `conventions.py`

**File**: `ymcirc/ymcirc/conventions.py`

Current `LazyDict` entries call `json_loader` directly. Since `json_loader` now returns `(data, metadata)`, update the `PHYSICAL_PLAQUETTE_STATES` and `HAMILTONIAN_BOX_TERMS` lazy dicts to use the new specialized loaders instead.

Store metadata (including `f_order`) when data is first loaded. Options:
- Create a module-level `_LOADED_METADATA` dict that gets populated when data is loaded
- Or have the specialized loaders return just the data (for `LazyDict` compatibility) and separately expose metadata via a function

**Chosen approach**: Create a `_load_and_cache_metadata` pattern where the specialized loaders store metadata in a module-level `_DATA_METADATA` dict, keyed by `(dim_string, trunc_string)`. Expose via a public function `get_data_metadata(dim_string, trunc_string) -> dict`.

### 1E. Adapt `PlaquetteState` control links to nested per-vertex format

**File**: `ymcirc/ymcirc/conventions.py`

The new JSON stores control links as per-vertex tuples:
```python
# Old flat format:
c_links = (c1, c2, c3, c4)           # d=3/2: 4 i-weights
c_links = (c1, c2, ..., c8)          # d=2: 8 i-weights

# New per-vertex format:
c_links = ((c_at_v1,), (c_at_v2,), (c_at_v3,), (c_at_v4,))   # d=3/2
c_links = ((c1_v1, c2_v1), (c1_v2, c2_v2), ...)               # d=2
```

Update the `PlaquetteState` type alias:
```python
VertexControlLinks = Tuple[LinkState, ...]  # Variable-length tuple of i-weights at one vertex
PlaquetteState = Union[
    Tuple[
        Tuple[MultiplicityIndex, MultiplicityIndex, MultiplicityIndex, MultiplicityIndex],
        Tuple[LinkState, LinkState, LinkState, LinkState],
        Tuple[VertexControlLinks, VertexControlLinks, VertexControlLinks, VertexControlLinks]
    ],
    Tuple[
        Tuple[None, None, None, None],
        Tuple[LinkState, LinkState, LinkState, LinkState],
        Tuple[VertexControlLinks, VertexControlLinks, VertexControlLinks, VertexControlLinks]
    ]
]
```

Update the module-level docstring in `conventions.py` to document the new per-vertex control link format, including updated ASCII diagrams.

---

## Category 2: FORDER Integration

### 2A. Add `forder` parameter to `LatticeDef`

**File**: `ymcirc/ymcirc/_abstract/lattice_data.py`

Add an optional `forder` parameter to `LatticeDef.__init__()`:
```python
def __init__(self, dimensions, size, periodic_boundary_conds=True,
             forder=None):
```
- Default: `[1, 2, 3, -1, -2, -3]` if `None`
- Store as `self._forder` and expose via `@property forder`
- Validate: must be a permutation of `[1, 2, 3, -1, -2, -3]`

Update `__repr__`, `__eq__`, and `__str__` as appropriate. Thread the `forder` parameter through `LatticeData.__init__()` as well.

### 2B. Add `forder` parameter to `LatticeStateEncoder`

**File**: `ymcirc/ymcirc/conventions.py`

Add `forder` parameter to `LatticeStateEncoder.__init__()`:
```python
def __init__(self, link_bitmap, physical_plaquette_states, lattice, forder=None):
```
- Store internally
- Pass to the internal `LatticeDef` copy (line 417/421)
- Expose via `@property forder`

Update `load_magnetic_hamiltonian` to accept and forward `forder`.

### 2C. Replace static `_CONTROL_LINK_DIRS_PER_VERTEX_MAP` with FORDER-based computation

**File**: `ymcirc/ymcirc/_abstract/lattice_data.py`

Replace the class variable `_CONTROL_LINK_DIRS_PER_VERTEX_MAP` with a static method:

```python
@staticmethod
def compute_control_link_dirs_per_vertex(dim, plane, forder):
    """Compute per-vertex control link directions sorted by FORDER.

    For each of the 4 plaquette vertices, returns a tuple of the
    non-active link directions that exist at that vertex, sorted
    by their position in the forder list.

    Returns:
        Tuple of 4 tuples, one per vertex (CCW from bottom-left).
    """
```

Logic:
1. Determine active link directions per vertex from the plane `(e1, e2)`:
   - v1: `{+e1, +e2}`, v2: `{-e1, +e2}`, v3: `{-e1, -e2}`, v4: `{+e1, -e2}`
2. All possible directions: `{1, ..., ceil(dim), -1, ..., -ceil(dim)}`
3. Filter out non-existent directions for d=3/2 (no `-2` at bottom vertices, no `+2` at top vertices)
4. Control dirs = all_existing - active
5. Sort each vertex's controls by `forder.index(d)`

For the d=3/2 filtering: since we don't have the specific vertex coordinates in the static method, use the known pattern: v1,v2 are "bottom" (no `-2`), v3,v4 are "top" (no `+2`). This is correct because for d=3/2, the only plane is (1,2), and the bottom-left vertex is always at y=0.

### 2D. Update `control_links_ordered` property

**File**: `ymcirc/ymcirc/_abstract/lattice_data.py`

Update `Plaquette.control_links_ordered` to use FORDER-based ordering:

```python
@property
def control_links_ordered(self) -> tuple[T, ...]:
    forder = self._lattice.forder
    control_links_ordered = []
    for vertex_vector in self._ordered_vertex_vectors:
        # Get control link addresses at this vertex
        ctrl_addrs = list(self._control_links[vertex_vector].keys())
        # Sort by FORDER position of the direction component
        ctrl_addrs.sort(key=lambda addr: forder.index(addr[1]))
        # Append data in sorted order
        for addr in ctrl_addrs:
            control_links_ordered.append(self._control_links[vertex_vector][addr])
    return tuple(control_links_ordered)
```

Remove the `_CONTROL_LINK_DIRS_PER_VERTEX_MAP` class variable entirely. Update the docstring to explain FORDER-based ordering instead of the old hardcoded convention.

**Concrete ordering change for d=2** with default FORDER `[1,2,3,-1,-2,-3]`:
- Old `v2` controls: `(-2, +1)` → New: `(+1, -2)` (c3 and c4 swap)
- All other vertices unchanged for d=3/2 and d=2

### 2E. Update `LatticeStateEncoder` encoding/decoding for nested c_links

**File**: `ymcirc/ymcirc/conventions.py`

**`__init__`**: Update validation for control link count:
```python
# Old:
n_total_control_links = len(physical_plaquette_states[0][2])

# New:
n_total_control_links = sum(len(v_ctrls) for v_ctrls in physical_plaquette_states[0][2])
```

**`encode_plaquette_state_as_bit_string`**: Update c_links iteration:
```python
# Old:
for c_link in c_links:
    bit_string_encoding += self._link_bitmap[c_link]

# New:
for vertex_controls in c_links:
    for c_link in vertex_controls:
        bit_string_encoding += self._link_bitmap[c_link]
```

Update the validation `len(c_links) != self._lattice.n_control_links_per_plaquette` to compare `sum(len(vc) for vc in c_links)` instead.

Update the method docstring to reflect the per-vertex format.

**`decode_bit_string_to_plaquette_state`**: Update c_links reconstruction:
```python
# Need to know how many controls per vertex.
# For periodic lattices: n_controls_per_vertex = 2 * (dim - 1), constant across all 4 vertices
# For d=3/2: 1 per vertex. For d=2: 2 per vertex.
n_controls_per_vertex = int(2 * (self._lattice.dim - 1))

decoded_c_links_flat = [
    self.decode_bit_string_to_link_state(encoded_link) for encoded_link in
    LatticeStateEncoder._split_string_evenly(c_links_substring, self._expected_link_bit_string_length)
]
# Group into per-vertex tuples
decoded_c_links = tuple(
    tuple(decoded_c_links_flat[i:i + n_controls_per_vertex])
    for i in range(0, len(decoded_c_links_flat), n_controls_per_vertex)
)
```

### 2F. Update small/periodic lattice control link handling in `circuit.py`

**File**: `ymcirc/ymcirc/circuit.py`

**`_plaquette_state_has_inconsistent_controls`** (lines 794-825):

With nested per-vertex c_links and F-ordered controls, the consistency checks change.

For d=3/2 (each vertex has 1 control, sharing: v1==v2, v3==v4):
```python
case 1.5:
    plaquette_state_has_inconsistent_controls = (
        c_links[0] != c_links[1] or  # v1 controls != v2 controls (as tuples)
        c_links[2] != c_links[3]     # v3 controls != v4 controls
    )
```
This is structurally the same as before but now compares tuples of i-weights rather than individual i-weights.

For d=2 (each vertex has 2 controls). With F-order `[1,2,3,-1,-2,-3]`, the control directions per vertex are:
- v1: `(-1, -2)`, v2: `(+1, -2)`, v3: `(+1, +2)`, v4: `(+2, -1)`

On size-2 periodic lattice, physical link sharing:
- v1[0]=dir(-1) shares with v2[0]=dir(+1) → `c_links[0][0] != c_links[1][0]`
- v1[1]=dir(-2) shares with v4[0]=dir(+2) → `c_links[0][1] != c_links[3][0]`
- v2[1]=dir(-2) shares with v3[1]=dir(+2) → `c_links[1][1] != c_links[2][1]`
- v3[0]=dir(+1) shares with v4[1]=dir(-1) → `c_links[2][0] != c_links[3][1]`

```python
case 2:
    plaquette_state_has_inconsistent_controls = (
        (c_links[0][0] != c_links[1][0]) or
        (c_links[0][1] != c_links[3][0]) or
        (c_links[1][1] != c_links[2][1]) or
        (c_links[2][0] != c_links[3][1])
    )
```

**`_discard_duplicate_controls_from_plaquette_state`** (lines 827-852):

For d=3/2: keep v1 and v3 controls, drop v2 and v4:
```python
case 1.5:
    physical_c_links = (c_links[0], (), c_links[2], ())
```

For d=2: keep first occurrence of each shared physical link:
```python
case 2:
    physical_c_links = (
        c_links[0],                      # v1: both controls are first occurrences
        (c_links[1][1],),                 # v2: only second (dir -2) is unique; first (dir +1) duplicates v1[0]
        (c_links[2][0],),                 # v3: only first (dir +1) is unique; second (dir +2) duplicates v2[1]
        ()                               # v4: both duplicate earlier entries
    )
```

**`apply_magnetic_trotter_step`** redundant c_link indices (lines 571-581):

The redundancy indices must match the pruning logic above. With nested format, the flat-index approach changes. Update to use per-vertex indexing:

```python
# Replace flat index check with per-vertex check
redundant_controls_by_dim = {
    1.5: {1, 3},       # vertex indices to skip entirely
    2: ...             # more complex: partial vertex skips
}
```

Alternative approach: instead of checking flat indices, iterate by vertex and check whether each vertex's controls should be included:

```python
for vertex_idx, vertex_controls in enumerate(plaquette.control_links_per_vertex):
    for ctrl_idx, register in enumerate(vertex_controls):
        if should_skip(vertex_idx, ctrl_idx, dim):
            continue
        for qubit in register:
            c_link_qubits.append(qubit)
```

This requires a `control_links_per_vertex` property on Plaquette (see 2G).

### 2G. Add `control_links_per_vertex` property to `Plaquette`

**File**: `ymcirc/ymcirc/_abstract/lattice_data.py`

Add a new property that returns control link data grouped by vertex (matching the per-vertex structure of `PlaquetteState.c_links`):

```python
@property
def control_links_per_vertex(self) -> tuple[tuple[T, ...], ...]:
    """Retrieve control link data grouped by vertex in FORDER-sorted order."""
    forder = self._lattice.forder
    result = []
    for vertex_vector in self._ordered_vertex_vectors:
        ctrl_addrs = list(self._control_links[vertex_vector].keys())
        ctrl_addrs.sort(key=lambda addr: forder.index(addr[1]))
        vertex_data = tuple(self._control_links[vertex_vector][addr] for addr in ctrl_addrs)
        result.append(vertex_data)
    return tuple(result)
```

Then `control_links_ordered` can delegate:
```python
@property
def control_links_ordered(self) -> tuple[T, ...]:
    """Retrieve control links as a flat tuple in FORDER-sorted, vertex-by-vertex order."""
    return tuple(link for vertex in self.control_links_per_vertex for link in vertex)
```

### 2H. Update `from_partial_measurement` in `parsed_lattice_result.py`

**File**: `ymcirc/ymcirc/parsed_lattice_result.py`

Line 425 currently uses `Plaquette._CONTROL_LINK_DIRS_PER_VERTEX_MAP[dim]`. Replace with:

```python
control_link_dirs = Plaquette.compute_control_link_dirs_per_vertex(
    encoder.lattice_def.dim, (e1, e2), encoder.lattice_def.forder
)
```

The rest of the logic (iterating vertex-by-vertex, direction-by-direction) stays structurally the same.

### 2I. Thread `forder` through `LatticeRegisters`

**File**: `ymcirc/ymcirc/lattice_registers.py`

- Add `forder` parameter to `LatticeRegisters.__init__()`, pass to `super().__init__()`
- Update `from_lattice_state_encoder` factory method to pass `forder` from the encoder's lattice_def:
  ```python
  return LatticeRegisters(
      dimensions=lattice_def.dim,
      size=size,
      periodic_boundary_conds=lattice_def.periodic_boundary_conds,
      link_bitmap=lattice_encoder.link_bitmap,
      vertex_bitmap=lattice_encoder.vertex_bitmap,
      forder=lattice_def.forder
  )
  ```

---

## Category 3: Docstring Updates

Update docstrings for all modified public APIs:
- `Plaquette.control_links_ordered` — explain FORDER-based ordering
- `Plaquette.control_links_per_vertex` — new property, document per-vertex structure
- `Plaquette.compute_control_link_dirs_per_vertex` — new static method
- `LatticeDef.__init__` — document `forder` parameter
- `LatticeStateEncoder.__init__` — document `forder` parameter
- `encode_plaquette_state_as_bit_string` — update for nested c_links format
- `decode_bit_string_to_plaquette_state` — update for nested c_links format
- `conventions.py` module docstring — update control link diagrams and format description
- `json_loader` — document new gzip/metadata support

---

## Category 4: Test Updates

### Tests expected to break and need updating:

**`test_conventions.py`** (~1200 lines):
- `test_physical_plaquette_state_data_are_valid` — validation logic for c_links format changes (nested vs flat)
- `test_lattice_encoder_fails_if_plaquette_states_have_wrong_number_of_controls` — validation logic changed
- `test_lattice_encoder_infers_correct_vertex_bitmaps` — PlaquetteState fixtures use old flat c_links
- `test_lattice_encoder_infers_correct_plaquette_length` — plaquette bit string length calculation
- `test_encoding_malformed_plaquette_fails` — error case fixtures need updating
- `test_encoding_good_plaquette` — PlaquetteState fixtures use old flat c_links
- `test_all_mag_hamiltonian_plaquette_states_have_unique_bit_string_encoding` — loads data from updated JSON
- `test_bit_string_decoding_to_plaquette` — decoded format now nested
- `test_decoding_garbage_bit_strings_result_in_none` — decoded format now nested
- `test_decoding_fails_when_len_bit_string_doesnt_match_bitmaps` — may need minor updates
- `test_load_magnetic_hamiltonian_constructs_correct_num_rotations` — JSON loader return format changed
- `test_matrix_element_data_are_valid_*` — JSON loader return format changed
- `test_no_duplicate_physical_plaquette_states` / `test_no_duplicate_matrix_elements` — data loader changes
- `test_hamiltonian_box_terms_no_unexpected_cases` — data loader changes

**`test_lattice_registers.py`** (~817 lines):
- `test_control_link_registers_have_correct_ordering` — hardcoded expected orderings change for d=2
- `test_get_plaquettes` — control link ordering in assertions
- `test_get_registers_in_local_hamiltonian_order` — depends on control link order

**`test_circuit.py`** (~1546 lines):
- `test_apply_magnetic_trotter_step_d_3_2_*` — circuit construction depends on control link ordering and Hamiltonian data format
- `test_apply_magnetic_trotter_step_d_2_*` — d=2 control ordering changes
- `test_measure_plaquette_*` — plaquette measurement depends on control link order
- `test_creating_correct_ancilla_register_*` — may be affected by Hamiltonian data format
- Any test creating `LatticeStateEncoder` with hardcoded PlaquetteState fixtures

**`test_parsed_lattice_result.py`** (~1029 lines):
- `test_from_partial_measurement_plaquette` — control link ordering in decoded plaquettes
- `test_from_partial_measurement_d2_plaquette_control_link_ordering` — directly tests d=2 control ordering
- Tests using hardcoded PlaquetteState fixtures

**`test_givens.py`** (~846 lines):
- Tests using hardcoded bitstrings that encode plaquette states — if the bitstring encoding changes due to control link reordering, these tests break
- `test_prune_controls_acts_as_expected` — depends on control link bitstring positions

**`test_measurement_results.py`** (~199 lines):
- Tests creating encoders with hardcoded PlaquetteState fixtures

**`test_integration_mps.py`** (~233 lines):
- End-to-end tests that load data and construct circuits — should work once all other changes are correct, but may need fixture updates

### Test update strategy:
1. Update all PlaquetteState fixtures from flat c_links to nested per-vertex format
2. Update expected control link orderings for d=2 (c3/c4 swap)
3. Update expected bitstrings where they depend on control link ordering
4. Run tests incrementally: `test_conventions.py` first, then `test_lattice_registers.py`, then `test_circuit.py`, etc.

---

## Verification

1. **Unit tests**: `uv run pytest -v` — all tests should pass after updates
2. **Slow tests**: `uv run pytest --runslow -v` — run integration tests
3. **Spot-check JSON loading**: Verify that `PHYSICAL_PLAQUETTE_STATES["d=3/2"]["T1"]` loads correctly and states have the expected nested c_links structure
4. **Spot-check Hamiltonian loading**: Verify that `HAMILTONIAN_BOX_TERMS["d=3/2"]["T1"]` loads as a flat `{(Pf, Pi): float}` dict
5. **Metadata access**: Verify `get_data_metadata("d=3/2", "T1")["f_order"]` returns `[1, 2, 3, -1, -2, -3]`
6. **Control link ordering**: Verify `Plaquette.compute_control_link_dirs_per_vertex(2, (1,2), [1,2,3,-1,-2,-3])` returns `((-1, -2), (1, -2), (1, 2), (2, -1))` (note v2 changed from old `(-2, 1)`)

---

## Files Modified (Summary)

| File | Changes |
|------|---------|
| `ymcirc/utilities.py` | `json_loader` gzip + metadata support |
| `ymcirc/conventions.py` | File paths, `PlaquetteState` type, encoder init/encode/decode, loaders, metadata, docstrings |
| `ymcirc/_abstract/lattice_data.py` | `LatticeDef` forder param, `Plaquette` ordering rewrite, new static method + property |
| `ymcirc/lattice_registers.py` | Thread `forder` param through init + factory |
| `ymcirc/circuit.py` | Consistency checks, duplicate pruning, redundant control iteration |
| `ymcirc/parsed_lattice_result.py` | `from_partial_measurement` control link dirs |
| `tests/test_conventions.py` | Fixture + assertion updates for nested c_links and new loader |
| `tests/test_lattice_registers.py` | Control link ordering assertions (especially d=2) |
| `tests/test_circuit.py` | Fixture + assertion updates |
| `tests/test_parsed_lattice_result.py` | Control link ordering + fixture updates |
| `tests/test_givens.py` | Bitstring fixture updates if affected |
| `tests/test_measurement_results.py` | Fixture updates |
| `tests/test_integration_mps.py` | May need fixture updates |

## Execution Order

1. `utilities.py` — JSON loader (foundation for everything else)
2. `_abstract/lattice_data.py` — FORDER on LatticeDef, Plaquette ordering rewrite
3. `conventions.py` — PlaquetteState type, file paths, encoder updates, specialized loaders
4. `lattice_registers.py` — Thread forder
5. `circuit.py` — Consistency/pruning/iteration updates
6. `parsed_lattice_result.py` — from_partial_measurement update
7. Tests — update in dependency order: conventions → lattice_registers → circuit → parsed_lattice_result → givens → measurement_results → integration
