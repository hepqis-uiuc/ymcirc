# ymcirc Boundary Conditions: Comprehensive Summary

This document provides a detailed analysis of how ymcirc handles boundary conditions, with a focus on what would need to change to support open boundary conditions (OBC) and mixed boundary conditions. All file paths are relative to `/Users/jasonelhaderi/Documents/School/UIUC/Projects/claude/ymcirc/`.

---

## 1. Architecture Overview

ymcirc generates quantum circuits (via Qiskit 2.x) to simulate lattice SU(3) gauge theories using Trotterized time evolution. The core module dependency flow is:

```
_abstract/lattice_data.py       LatticeDef, LatticeData[T], Plaquette[T]
        |                              |
lattice_registers.py            conventions.py (LatticeStateEncoder)
        |                              |
        +------ circuit.py ------------+   LatticeCircuitManager
                    |
                givens.py
                electric_helper.py
                parsed_lattice_result.py
                measurement_results.py
```

**Key source files:**

| File | Role |
|------|------|
| `ymcirc/_abstract/lattice_data.py` | Lattice geometry, plaquette abstraction, signatures |
| `ymcirc/conventions.py` | Bit-string encoding/decoding, data loading, Hamiltonian processing |
| `ymcirc/circuit.py` | Circuit construction (magnetic + electric Trotter steps) |
| `ymcirc/lattice_registers.py` | Qiskit QuantumRegister mapping to lattice |
| `ymcirc/parsed_lattice_result.py` | Measurement bitstring parsing |
| `ymcirc/measurement_results.py` | Aggregated shot-count analysis |
| `ymcirc/electric_helper.py` | Electric Hamiltonian Pauli decomposition |
| `ymcirc/utilities.py` | LazyDict, JSON loader, helpers |
| `run/functions.py` | High-level script configuration and execution |

---

## 2. Data Model

### 2.1 Data File Organization

All precomputed data lives in `ymcirc/_ymcirc_data/` in two subdirectories:

- `plaquette-states/` -- gauge-invariant plaquette state configurations
- `magnetic-hamiltonian-box-term-matrix-elements/` -- magnetic Hamiltonian box term matrix elements

### 2.2 File Naming Convention

Files follow the pattern: `{TruncationLabel}_dim({DimSpec})[_PBC]_{DataType}.json.gz`

Examples:
- `T1_dim(3_2)_magnetic_hamiltonian.json.gz` -- T1 truncation, d=3/2 (no explicit BC label)
- `B3_dim(2)_PBC_magnetic_hamiltonian.json.gz` -- B3 truncation, d=2, PBC
- `B7_dim(3_2)_PBC_plaquette_states.json.gz` -- B7 truncation, d=3/2, PBC

**Observation:** The older T-series files (T1, T2) lack an explicit `_PBC` in the filename, while the newer B-series files include `_PBC`. All currently shipped data files are for PBC lattices. There are **no OBC data files**.

### 2.3 Complete Data File Inventory

**Magnetic Hamiltonian files (15 files):**

| File | Truncation | Dim | BC |
|------|-----------|-----|-----|
| `T1_dim(3_2)_magnetic_hamiltonian.json.gz` | T1 | 3/2 | PBC (implicit) |
| `T2_dim(3_2)_magnetic_hamiltonian.json.gz` | T2 | 3/2 | PBC (implicit) |
| `T1_dim(2)_magnetic_hamiltonian.json.gz` | T1 | 2 | PBC (implicit) |
| `B3_dim(3_2)_PBC_magnetic_hamiltonian.json.gz` | B3 | 3/2 | PBC |
| `B3_dim(2)_PBC_magnetic_hamiltonian.json.gz` | B3 | 2 | PBC |
| `B3_dim(3)_PBC_magnetic_hamiltonian.json.gz` | B3 | 3 | PBC |
| `B4_dim(2)_PBC_magnetic_hamiltonian.json.gz` | B4 | 2 | PBC |
| `B4_dim(3)_PBC_magnetic_hamiltonian.json.gz` | B4 | 3 | PBC |
| `B5_dim(3_2)_PBC_magnetic_hamiltonian.json.gz` | B5 | 3/2 | PBC |
| `B6_dim(3_2)_PBC_magnetic_hamiltonian.json.gz` | B6 | 3/2 | PBC |
| `B7_dim(3_2)_PBC_magnetic_hamiltonian.json.gz` | B7 | 3/2 | PBC |
| `B7_dim(2)_PBC_magnetic_hamiltonian.json.gz` | B7 | 2 | PBC |
| `B8_dim(3_2)_PBC_magnetic_hamiltonian.json.gz` | B8 | 3/2 | PBC |
| `B9_dim(3_2)_PBC_magnetic_hamiltonian.json.gz` | B9 | 3/2 | PBC |
| `B10_dim(3_2)_PBC_magnetic_hamiltonian.json.gz` | B10 | 3/2 | PBC |

**Plaquette state files (15 files):** Same truncation/dim/BC combinations as above, with `_plaquette_states.json.gz` suffix.

### 2.4 Data File Internal Structure

Each `.json.gz` file contains a JSON object with two top-level keys:
- `"metadata"` -- contains `dim`, `truncation_mode`, `cutoff`, `f_order`, etc.
- `"data"` -- the actual plaquette states or matrix elements

**Hamiltonian data structure:**
```
{
  (plaq_state_1, plaq_state_2): {
    (plane_tuple): {
      (signature_tuple): float_value,
      ...
    },
    ...
  },
  ...
}
```

JSON serializes all dict keys as strings; `_load_hamiltonian()` (conventions.py, line 361) converts plane and signature keys back to tuples via `ast.literal_eval`.

### 2.5 Data Loading Pipeline

The file path registry is defined in `ymcirc/conventions.py` (lines 184-229) as two dicts:
- `_HAMILTONIAN_DATA_FILE_PATHS` -- maps `(dim_string, trunc_string)` to file paths
- `_PLAQUETTE_STATES_DATA_FILE_PATHS` -- same structure for plaquette states

These feed into two `LazyDict` instances:
- `PHYSICAL_PLAQUETTE_STATES` (line 414) -- lazy-loaded plaquette states
- `HAMILTONIAN_BOX_TERMS` (line 424) -- lazy-loaded Hamiltonian box terms

**To support OBC/mixed BC**, new entries would need to be added to both path registries, and new `.json.gz` data files would need to be generated (by pyclebsch or equivalent) for the relevant boundary condition types.

---

## 3. Plaquette Signatures

### 3.1 What a Signature Represents

A **Signature** (type alias at `lattice_data.py`, line 24) is:
```python
Signature = Tuple[
    Tuple[int, ...],   # vertex 1: all existing half-link directions, FORDER-sorted
    Tuple[int, ...],   # vertex 2
    Tuple[int, ...],   # vertex 3
    Tuple[int, ...],   # vertex 4
]
```

For each of the 4 plaquette vertices, the signature contains a tuple of ALL half-link directions that actually exist at that vertex (both active and control links), sorted by FORDER position.

### 3.2 Physical Meaning

The signature encodes the **local topology** at each vertex of a plaquette. On a periodic lattice, every vertex has the same number of links (2 * ceil(dim)), so all signatures for a given (dim, plane) are identical. On a lattice with boundaries, corner and edge vertices would have fewer links, producing different signatures.

As stated in the docstring at `lattice_data.py`, line 222:
> "Physically, the signature encodes whether a plaquette is in the 'interior' of a lattice, on a 'corner', on an 'edge', etc."

### 3.3 How Signatures Are Computed

The `Plaquette.signature` property (`lattice_data.py`, line 213) works by:

1. For each vertex index (0-3), calling `_existing_dirs_at_vertex()` (line 194)
2. That method unions the **active link directions** (from the plaquette plane) with the **control link directions** that were successfully resolved during `__init__`
3. The result is sorted by FORDER position

The key detail is at `Plaquette.__init__` (line 131-134):
```python
try:
    self._control_links[v][(v, link_dir)] = lattice.get_link((v, link_dir), **kwargs)
except KeyError:  # Depending on boundary conditions, KeyErrors can happen that can be skipped.
    continue
```

This means that on an OBC lattice, links that don't exist at boundary vertices would raise `KeyError` during plaquette construction, and those directions would be silently excluded. The signature would then naturally reflect the reduced connectivity.

### 3.4 How Signatures Vary with Boundary Conditions

- **PBC (current):** All plaquettes in a given (dim, plane) have identical signatures because every vertex has the same number of neighbors. For d=3/2 with default FORDER and plane (1,2): `((1, 2, -1), (1, 2, -1), (1, -1, -2), (1, -1, -2))`.

- **OBC (future):** Plaquettes near lattice boundaries would have vertices with missing directions. A corner vertex on a d=2 lattice might only have 2 directions instead of 4, producing a shorter inner tuple. This means:
  - Multiple distinct signatures would exist per (dim, plane) combination
  - The Hamiltonian data files must contain matrix elements keyed by each relevant signature
  - The circuit builder must handle variable-length control link lists

### 3.5 Signature in Circuit Construction

During `apply_magnetic_trotter_step()` (`circuit.py`, line 600-610):
```python
plaquette_plane: Plane = plaquette.plane
plaquette_signature: Signature = plaquette.signature
cache_key = (plaquette_plane, plaquette_signature)
```

The Hamiltonian is resolved per (plane, signature) via `_resolve_hamiltonian_for_plaquette()` (line 784), which looks up the plane in `PlaneKeyedHamiltonianData` and then filters by signature. On PBC lattices this caching is very effective since all plaquettes share the same key. On OBC lattices, there would be multiple distinct cache keys.

---

## 4. Current Boundary Condition Handling

### 4.1 LatticeDef Configuration

`LatticeDef.__init__()` (`lattice_data.py`, line 269) accepts:
```python
periodic_boundary_conds: bool | tuple[bool, ...] = True
```

The **default is True** (all-periodic). The type hint suggests per-direction control is planned, but validation currently rejects non-bool values:

- Line 340-341: `isinstance(periodic_boundary_conds, Iterable)` raises `NotImplementedError("Tuples for boundary conditions not yet supported.")`
- Line 353-354: `isinstance(periodic_boundary_conds, bool)` check with `TypeError` for non-bools

### 4.2 Properties That Check Boundary Conditions

`LatticeDef` provides two BC-related properties:

- `periodic_boundary_conds` (line 486): Returns the stored value (bool or tuple)
- `all_boundary_conds_periodic` (line 496): Returns `True` only if ALL directions are periodic

### 4.3 Where PBC Is Assumed/Enforced

The PBC assumption pervades the codebase. Every module that does lattice geometry operations checks `all_boundary_conds_periodic` and raises `NotImplementedError` for the non-periodic case. See the complete inventory in Section 5 below.

### 4.4 Periodic Wrapping Logic

Modular arithmetic for PBC appears in several places:

- **`add_unit_vector_to_vertex_vector()`** (`lattice_data.py`, line 507-530): Wraps coordinates via `comp % self.shape[0]`. Special handling for d=3/2 where vertical direction is never periodic.

- **`_normalize_link_address()`** (`lattice_data.py`, line 386-416): Handles negative unit vectors by stepping back one vertex, then wraps via modular arithmetic if periodic.

- **`get_vertex()`** in `LatticeRegisters` (`lattice_registers.py`, line 146-156): Wraps vertex coordinates mod shape.

### 4.5 Small Periodic Lattice Special Handling

On periodic lattices small enough (size <= 2) that the same physical link appears as a control at multiple vertices in one plaquette:

1. `LatticeCircuitManager.__init__()` (`circuit.py`, lines 76-154) detects this case and filters Hamiltonian terms with inconsistent shared controls
2. Duplicate control links are trimmed from bitstring encodings via `_discard_duplicate_controls_from_plaquette_state()` (line 950)
3. During `apply_magnetic_trotter_step()`, skip indices are computed per (plane, vertex_idx) to avoid stitching redundant control qubits (lines 556-570)

This logic is **entirely PBC-specific** and would not apply to OBC lattices (where wrapping doesn't occur).

---

## 5. NotImplementedError Inventory

Every `NotImplementedError` in the codebase, with file, line, function, and message:

### 5.1 `ymcirc/_abstract/lattice_data.py`

| Line | Function/Context | Message | BC-Relevant? |
|------|-----------------|---------|--------------|
| 339 | `_validate_lattice_params()` | `"Tuples for lattice size not yet supported."` | Indirectly -- needed for non-hypercubic OBC lattices |
| 341 | `_validate_lattice_params()` | `"Tuples for boundary conditions not yet supported."` | **YES** -- blocks mixed BC (per-direction PBC/OBC) |
| 413 | `_normalize_link_address()` | `(no message)` | **YES** -- blocks OBC link normalization |
| 432 | `n_plaquettes` (property) | `"Number of plaquettes calculation only implemented for periodic lattices."` | **YES** -- OBC lattices have fewer plaquettes |
| 455 | `n_control_links_per_plaquette` (property) | `"Only periodic boundary conditions have been implemented."` | **YES** -- OBC vertices have variable control counts |
| 530 | `add_unit_vector_to_vertex_vector()` | `"Vector addition not yet implemented on nonperiodic lattices."` | **YES** -- core geometry operation |
| 588 | `get_traversal_order()` | `"Iteration through nonperiodic lattices not yet supported."` | **YES** -- needed for all lattice iteration |

### 5.2 `ymcirc/conventions.py`

| Line | Function/Context | Message | BC-Relevant? |
|------|-----------------|---------|--------------|
| 670 | `LatticeStateEncoder.__init__()` | `"Lattices with different lengths along different dimensions not yet supported."` | Indirectly |
| 878 | `decode_bit_string_to_plaquette_state()` | `"Decoding plaquette states for nonperiodic lattices is not yet supported."` | **YES** -- OBC plaquettes have variable control counts |

### 5.3 `ymcirc/circuit.py`

| Line | Function/Context | Message | BC-Relevant? |
|------|-----------------|---------|--------------|
| 82 | `LatticeCircuitManager.__init__()` | `"Lattices with nonperiodic or mixed boundary conditions not yet supported."` | **YES** -- top-level gate blocking OBC |
| 91 | `LatticeCircuitManager.__init__()` | `"Non-square dim 2 lattices not yet supported."` | Indirectly |
| 96 | `LatticeCircuitManager.__init__()` | `"Non-cubic dim 3 lattices not yet supported."` | Indirectly |
| 98 | `LatticeCircuitManager.__init__()` | `"Dim {dim} lattice not yet supported."` | No |
| 111 | `LatticeCircuitManager.__init__()` | `"Dim {dim} lattice not yet supported."` | No |
| 421 | `apply_electric_trotter_step()` | `"Different number of electric dt and electric coupling_g Parameters encountered."` | No |
| 533 | `apply_magnetic_trotter_step()` | `"Different number of magnetic dt and magnetic coupling_g Parameters encountered."` | No |

### 5.4 `ymcirc/lattice_registers.py`

| Line | Function/Context | Message | BC-Relevant? |
|------|-----------------|---------|--------------|
| 154 | `get_vertex()` | `(no message)` | **YES** -- OBC vertex lookup |
| 263 | `from_lattice_state_encoder()` | `"LatticeRegisters for non-hypercubic lattices not yet implemented."` | Indirectly |

### 5.5 `ymcirc/electric_helper.py`

| Line | Function/Context | Message | BC-Relevant? |
|------|-----------------|---------|--------------|
| 87 | `convert_bitstring_to_evalue()` | `"Electric energy computation for non-hypercubic lattices with dim >= 2 not yet implemented."` | Indirectly |

### 5.6 `ymcirc/measurement_results.py`

| Line | Function/Context | Message | BC-Relevant? |
|------|-----------------|---------|--------------|
| 40 | `MeasurementResults.__init__()` | `"Converting state bit strings for lattices with tuple-valued shape not yet supported."` | Indirectly |

### 5.7 Other Files (not directly BC-relevant)

| File | Line | Message |
|------|------|---------|
| `run/state_prep.py` | 64 | `"State prep method {script_options['state_prep_method']} unknown."` |
| `archived-work/lattice_parser.py` | 41 | `"This class is no longer compatible..."` |
| `tests/test_conventions.py` | 51 | `"Test not implemented for dimension {dim_string}."` |
| `tests/test_lattice_registers.py` | 255 | `"Test not implemented for dims = {dims}."` (Note: this is `assert NotImplementedError(...)` which is a bug -- it should be `raise`) |

### 5.8 Placeholder Test

At `tests/test_circuit.py`, line 1768-1772:
```python
@pytest.mark.skip(reason="Non-periodic LatticeDef support not yet available. "
                         "Should verify plaquette.signature reflects missing directions at boundary vertices.")
def test_signature_nonperiodic():
    """Placeholder: verify signature correctness on a non-periodic lattice."""
    pass
```

---

## 6. Data Loading Pipeline and Required Changes for OBC/Mixed BC

### 6.1 Current Pipeline

```
1. json_loader() (utilities.py)
   -- Reads .json.gz, returns (data, metadata)

2. _load_hamiltonian() / _load_plaquette_states() (conventions.py)
   -- Converts string keys to tuples
   -- Caches metadata in _DATA_METADATA

3. HAMILTONIAN_BOX_TERMS / PHYSICAL_PLAQUETTE_STATES (conventions.py)
   -- LazyDict instances keyed by [dim_string][trunc_string]

4. compute_all_rotations_from_just_box_terms() (conventions.py)
   -- Computes box + box-dagger

5. load_magnetic_hamiltonian() (conventions.py)
   -- Encodes plaquette states as bitstrings
   -- Returns Dict[(bitstring, bitstring), MatrixElementValue]

6. LatticeCircuitManager.__init__() (circuit.py)
   -- Pivots to PlaneKeyedHamiltonianData: plane -> (bs1, bs2) -> signature -> float
   -- On small periodic lattices, filters/trims duplicate controls

7. _resolve_hamiltonian_for_plaquette() (circuit.py)
   -- Filters PlaneKeyedHamiltonianData by plane and signature
   -- Returns List[(bs1, bs2, float)]
```

### 6.2 Required Changes for OBC/Mixed BC

**Data file level:**
- pyclebsch must generate new data files for OBC lattices, with distinct signatures for interior, edge, and corner plaquettes
- File naming convention should be extended (e.g., `B3_dim(2)_OBC_magnetic_hamiltonian.json.gz`)
- Alternatively, a single data file per (truncation, dim) could contain all signature variants needed for any BC type

**Path registry (`conventions.py`):**
- `_HAMILTONIAN_DATA_FILE_PATHS` and `_PLAQUETTE_STATES_DATA_FILE_PATHS` need new entries for OBC/mixed BC data, or the keying scheme needs to be extended from `[dim_string][trunc_string]` to `[dim_string][trunc_string][bc_string]`

**`LazyDict` instances:**
- `HAMILTONIAN_BOX_TERMS` and `PHYSICAL_PLAQUETTE_STATES` need to accommodate the new BC dimension in their key hierarchy

**Plaquette state encoding (`LatticeStateEncoder`):**
- `n_control_links_per_plaquette` is currently a single global number. For OBC, different plaquettes have different numbers of controls, so encoding must become variable-length or padded
- `decode_bit_string_to_plaquette_state()` (line 877) explicitly raises `NotImplementedError` for nonperiodic lattices because the number of controls per vertex is not uniform

**`load_magnetic_hamiltonian()`:**
- Currently assumes a uniform control count. Would need to handle variable-length plaquette bitstrings

---

## 7. Plaquette Iteration and Boundary Condition Effects

### 7.1 Current Iteration Pattern

`get_traversal_order()` (`lattice_data.py`, line 534) defines the canonical iteration order:
- Outer loop: sorted vertices
- Inner loop: positive link directions at each vertex
- d=3/2 special case: skip vertical links at the top row

`apply_magnetic_trotter_step()` (`circuit.py`, line 578) iterates:
```
for vertex_address in lattice.vertex_addresses:
    # Skip top-row vertices for d=3/2
    # Get all "positive" plaquettes at this vertex
    for plaquette in plaquettes:
        # Resolve Hamiltonian by (plane, signature)
        # Build or fetch cached rotation circuit
        # Stitch into master circuit
```

### 7.2 How Boundary Conditions Affect Plaquette Existence

**PBC:** Every vertex has plaquettes in all plane orientations. For d=2, every vertex has exactly one plaquette (plane (1,2)). For d=3, every vertex has three plaquettes.

**OBC (future):** Plaquettes whose four vertices would extend beyond the lattice boundary simply do not exist. On a d=2 lattice of size N with OBC:
- PBC has N^2 plaquettes (N^2 vertices, each with one plaquette)
- OBC has (N-1)^2 plaquettes (only vertices at positions (i,j) with i < N-1 and j < N-1 can serve as bottom-left corners)

### 7.3 Changes Needed for OBC Plaquette Iteration

1. **`get_traversal_order()`** (`lattice_data.py`, line 586-588): Currently raises `NotImplementedError` for non-periodic lattices. Must be extended to:
   - Not wrap vertex addresses modularly
   - Exclude links that don't exist at boundary vertices
   - Handle d=3/2 OBC (horizontal direction is no longer periodic)

2. **`add_unit_vector_to_vertex_vector()`** (`lattice_data.py`, line 507-530): Must handle the non-periodic case -- either return the raw (unwrapped) result or raise an error if stepping off the lattice edge.

3. **`n_plaquettes`** (`lattice_data.py`, line 418-432): Must compute the reduced count for OBC. For a d-dimensional hypercubic lattice of size N with full OBC: `C(d,2) * (N-1)^d` instead of `C(d,2) * N^d`.

4. **`n_control_links_per_plaquette`** (`lattice_data.py`, line 444-458): No longer a single number for OBC -- boundary plaquettes have fewer controls. This property may need to become plaquette-specific or return a maximum.

5. **`_normalize_link_address()`** (`lattice_data.py`, line 386-416): The OBC case at line 412-413 needs to either raise `KeyError` (for links that don't exist) or clamp coordinates.

6. **Plaquette construction** (line 82-141 of `Plaquette.__init__`): Already partially handles missing links via the `KeyError` catch at line 133. On an OBC lattice, `get_link()` would raise `KeyError` for nonexistent boundary links, and those control links would be silently excluded. The signature would then naturally reflect the reduced connectivity.

7. **`LatticeCircuitManager.__init__()`** (`circuit.py`, line 81-82): The top-level guard must be removed or relaxed.

8. **Circuit stitching** (`circuit.py`, lines 654-686): The control qubit collection logic assumes PBC-specific skip indices. For OBC, the number of control qubits varies per plaquette, and the circuit template must match.

### 7.4 Mixed Boundary Conditions

For mixed BC (e.g., periodic in one direction, open in another), the `periodic_boundary_conds` parameter would be a tuple of bools. This requires:

1. Removing the `NotImplementedError` at `lattice_data.py` line 341
2. Updating `add_unit_vector_to_vertex_vector()` to check periodicity per-direction
3. Updating `_normalize_link_address()` similarly
4. Generating appropriate data files where signatures reflect mixed topology
5. The `all_boundary_conds_periodic` property already handles tuples correctly (line 496-500)

---

## 8. Summary of Changes Required for OBC/Mixed BC Support

### 8.1 Data Generation (External -- pyclebsch)

- Generate plaquette state and Hamiltonian matrix element data for OBC and mixed BC lattices
- Data files must contain signature keys corresponding to boundary/corner/edge vertices
- Each distinct vertex topology at a plaquette produces a distinct signature

### 8.2 `ymcirc/_abstract/lattice_data.py` (7 `NotImplementedError` sites)

- **Line 341**: Accept tuple-valued `periodic_boundary_conds`
- **Line 413**: Implement OBC link address normalization (raise `KeyError` for nonexistent links)
- **Line 432**: Compute `n_plaquettes` for OBC/mixed lattices
- **Line 455**: Make `n_control_links_per_plaquette` boundary-aware (or deprecate in favor of per-plaquette counts)
- **Line 530**: Implement `add_unit_vector_to_vertex_vector()` without wrapping (or with per-direction wrapping for mixed BC)
- **Line 588**: Implement `get_traversal_order()` for non-periodic lattices

### 8.3 `ymcirc/conventions.py` (1 `NotImplementedError` site + data registry)

- **Line 878**: Implement `decode_bit_string_to_plaquette_state()` for variable control counts
- **Path registries** (lines 184-229): Add OBC/mixed BC data file entries
- **`LatticeStateEncoder`**: Handle variable-length plaquette bitstrings
- **`n_control_links_per_plaquette` usage** (line 649): Adapt to non-uniform control counts

### 8.4 `ymcirc/circuit.py` (1 critical `NotImplementedError` + logic changes)

- **Line 82**: Remove/relax the PBC-only guard
- **Small periodic lattice logic** (lines 100-154): Keep as-is but ensure it doesn't activate for OBC
- **Circuit caching** (line 617-619): The comment at line 617 already notes that on nonperiodic lattices, different plaquettes may have different matrix elements, so the cache key strategy must change
- **Control qubit stitching** (lines 654-686): Must handle variable control counts per plaquette
- **Skip indices** (lines 556-570): Only apply for small periodic lattices (already gated by `self._lattice_is_small and self._lattice_is_periodic`)

### 8.5 `ymcirc/lattice_registers.py` (1 `NotImplementedError` site)

- **Line 154**: Implement `get_vertex()` without modular wrapping for OBC

### 8.6 `run/functions.py`

- **Line 42/253**: The `use_periodic_boundary_conds` parameter is already plumbed through; extend to accept tuple values for mixed BC

---

## 9. Key Architectural Insight: The Signature Mechanism

The existing signature mechanism is the most important enabler for OBC support. The design already anticipates non-uniform vertex connectivity:

1. `Plaquette.__init__()` gracefully handles missing links via `KeyError` catch (line 133)
2. `_existing_dirs_at_vertex()` (line 194) computes directions from what was actually resolved, not from a static formula
3. `signature` (line 213) produces tuples of variable length reflecting actual connectivity
4. `_resolve_hamiltonian_for_plaquette()` (line 784) filters by exact signature match

This means that **if OBC data files are generated with the correct signature keys**, the Hamiltonian resolution pipeline would work without fundamental changes. The main barriers are:
1. The numerous `NotImplementedError` guards throughout the geometry code
2. The assumption of uniform control link counts in the bitstring encoding/decoding
3. The absence of OBC data files

The uniform control count assumption in `LatticeStateEncoder` is the deepest structural challenge, as it affects bitstring lengths and therefore circuit qubit counts per plaquette.
