# Open and Mixed Boundary Conditions: Implementation Study

This document analyzes implementation options for supporting open boundary conditions (OBC) and mixed boundary conditions in ymcirc, informed by comprehensive analysis of both the ymcirc and pyclebsch codebases. See the companion documents for detailed summaries:

- `ymcirc-boundary-conditions-summary.md` -- ymcirc architecture, data model, and NotImplementedError inventory
- `pyclebsch-boundary-conditions-summary.md` -- pyclebsch lattice geometry, signatures, and data generation

---

## Background: The Core Problem

On a **fully periodic lattice**, every plaquette has the same local topology (same number of links at each vertex, same directions). This means:
- One plaquette signature per plane
- One set of matrix elements per plane
- One set of physical plaquette states per plane
- Data files keyed by `(truncation, dim)` are sufficient

On a **non-periodic lattice** (OBC or mixed BC), translational invariance is broken at the boundaries. Different plaquettes can have different **signatures** -- 4-tuples of per-vertex half-link direction sets that encode the local topology. This means:
- Multiple distinct signatures per plane (up to 9 for 2D OBC, up to 18 per plane for 3D OBC, up to 3 for 2D mixed BC)
- Different matrix elements and physical states for each distinct signature
- The current data model (one file per `(truncation, dim)`) is insufficient

The existing ymcirc architecture already partially anticipates this: `Plaquette.__init__()` handles missing links via `KeyError` catch (`lattice_data.py:133`), signatures are computed from actual resolved links (`lattice_data.py:194`), and `_resolve_hamiltonian_for_plaquette()` (`circuit.py:784`) filters by exact signature match. The data file format already nests matrix elements under signature keys within each plane. **The primary gaps are the ~10 BC-relevant `NotImplementedError` guards, the uniform control link count assumption in `LatticeStateEncoder`, and the absence of non-PBC data files.**

---

## Shared Work (Required Regardless of Option)

All three options below require the same set of ymcirc code changes to remove `NotImplementedError` barriers and support variable-connectivity plaquettes. The options differ only in how data is organized and loaded.

### Geometry Layer (`_abstract/lattice_data.py`)
1. **Accept tuple-valued `periodic_boundary_conds`** (line 341) -- enables per-direction BC specification
2. **Implement `_normalize_link_address()` for non-periodic lattices** (line 413) -- raise `KeyError` for links that don't exist at boundaries
3. **Implement `add_unit_vector_to_vertex_vector()` without wrapping** (line 530) -- per-direction wrapping for mixed BC
4. **Implement `get_traversal_order()` for non-periodic lattices** (line 588) -- exclude boundary plaquettes
5. **Make `n_plaquettes` and `n_control_links_per_plaquette` boundary-aware** (lines 432, 455)

### Encoding Layer (`conventions.py`)
6. **Variable-length plaquette bitstrings in `LatticeStateEncoder`** -- boundary plaquettes have fewer control links, so bitstring lengths vary by signature. This is the deepest structural change.
7. **Implement `decode_bit_string_to_plaquette_state()` for nonperiodic lattices** (line 878)

### Circuit Layer (`circuit.py`)
8. **Remove/relax PBC-only guard** (line 82)
9. **Handle variable control qubit counts** in circuit stitching (lines 654-686)
10. **Ensure small-lattice PBC logic doesn't activate for OBC** (lines 100-154, already gated by `self._lattice_is_periodic`)

### Registers Layer (`lattice_registers.py`)
11. **Implement `get_vertex()` without modular wrapping** (line 154)

---

## Option 1: Universal Data Files (One File per Truncation + Dimension)

### Description

Merge all possible signatures for a given `(truncation_mode, cutoff, dimension)` into a single data file. The same file serves PBC, OBC, and any mixed BC lattice of that dimension.

**Key insight:** A plaquette's matrix elements depend only on its signature and the truncation scheme, not on the global boundary conditions of the lattice it sits in. An "interior" plaquette on an OBC lattice has the same signature (and therefore the same matrix elements) as any plaquette on a PBC lattice of the same dimension. The signature fully determines the physics.

For a given dimension, the set of all possible signatures is small and finite:
- **d=2 (plane (1,2)):** At most 9 distinct signatures (3 positions along each of 2 axes: left-edge, interior, right-edge)
- **d=3/2 (plane (1,2)):** At most 3 distinct signatures (interior, left-edge, right-edge along the non-vertical direction)
- **d=3 (3 planes):** At most 18 distinct signatures per plane, ~54 total across all planes

### Data File Changes

- **File naming:** Keep current names (e.g., `B3_dim(2)_magnetic_hamiltonian.json.gz`), or add a `_universal` suffix. Remove the `_PBC` label since files are no longer BC-specific.
- **File contents:** Each file's data dict already has signature as a key under each plane. Universal files simply have more signature entries per plane.
- **Path registry (`conventions.py`):** Simplify from `[dim_string][trunc_string]` to just `[dim_string][trunc_string]` -- same as today, but now each entry covers all BCs.
- **Backward compatibility:** Existing PBC-only files are strict subsets of the universal files. Could keep old files and add new ones, or regenerate.

### pyclebsch Data Generation

To generate universal files, `gen_ymcirc_data.py` would need a single lattice case per `(truncation, dim)` that is large enough to contain all possible signatures. For d=2, a `[4,4,1]` OBC lattice already contains all 9 signature types. The script already handles this -- the active OBC T-truncation cases in `gen_ymcirc_data.py` iterate over all site coordinates.

The only change needed: instead of generating separate files for PBC and OBC, generate one universal file. For each `(truncation, dim)`, pick a large-enough OBC lattice and iterate over all site coordinates. The result is a superset of what any BC configuration needs.

### ymcirc Loading Changes

- **`_resolve_hamiltonian_for_plaquette()`:** Already filters by `(plane, signature)`. No change needed.
- **Cache behavior:** On a PBC lattice, all plaquettes still share one cache key per plane (one signature). On an OBC lattice, up to 9 cache keys per plane. The extra signatures in the data file are simply never looked up.
- **`load_magnetic_hamiltonian()`:** Must handle variable-length bitstrings since different signatures have different control link counts. The encoder needs to know which signature it's encoding for.

### Pros
- **Minimal number of data files** -- same count as today (one per truncation + dim)
- **No BC-specific file selection** -- ymcirc doesn't need to know which BCs are in use to select the right data file
- **Maximum reusability** -- the same file works for any lattice of that dimension, regardless of BCs
- **Simple path registry** -- no new keying dimension needed
- **Forward-compatible** -- if new BC types are added, the same files still work
- **Existing data format is already structured for this** -- signatures are already keys in the nested dict

### Cons
- **Larger files** -- a universal d=2 file contains data for 9 signatures instead of 1. For d=3, up to ~54 signatures. However, the data per signature is the same size as the current PBC data, so d=2 files grow ~9x and d=3 files grow ~54x at most.
- **Regeneration required** -- existing PBC data files must be regenerated (or augmented) to include non-PBC signatures. This is a one-time cost.
- **Plaquette state files get larger too** -- the set of physical plaquette states is the union across all signatures. Some states only appear at boundaries.
- **Computation cost for generation** -- computing matrix elements for all signatures in a single run is expensive (especially for d=3 with B-truncation at high cutoff). Though the OBC cases in `gen_ymcirc_data.py` already do this.

---

## Option 2: BC-Configuration-Keyed Data Files

### Description

Add a boundary condition dimension to the data file keying. Files are keyed by `(truncation, dim, BC_type)` where `BC_type` captures the boundary condition configuration.

### BC_type Classification

BC configurations can be classified by which directions are periodic:
- **`PBC`**: All directions periodic (existing files)
- **`OBC`**: All directions open
- **`mixed_XY`**: Some directions periodic, others open (e.g., `mixed_TF` for d=2 with x-periodic y-open)

For d=2, the distinct BC types are: `PBC`, `OBC`, `mixed_TF`, `mixed_FT`. For d=3, there are more combinations: `PBC`, `OBC`, plus 6 mixed types.

### Data File Changes

- **File naming:** e.g., `B3_dim(2)_OBC_magnetic_hamiltonian.json.gz`, `B3_dim(2)_mixed_TF_magnetic_hamiltonian.json.gz`
- **Path registry:** Extend from `[dim_string][trunc_string]` to `[dim_string][trunc_string][bc_string]`
- **File contents:** Each file contains only the signatures relevant to its BC type

### pyclebsch Data Generation

One lattice case per `(truncation, dim, BC_type)`. The existing cases in `gen_ymcirc_data.py` already follow this pattern (separate cases for PBC and OBC lattices). Mixed BC cases would be added similarly.

### ymcirc Loading Changes

- **`LatticeDef` or `LatticeCircuitManager`** must determine the `bc_string` from the lattice's boundary conditions
- **Path registry lookup** gains a third key dimension
- **`LazyDict` instances** gain a third nesting level
- **`_resolve_hamiltonian_for_plaquette()`:** Same as Option 1 -- filters by signature within the loaded data

### Pros
- **Smaller files** -- each file only contains signatures for its specific BC type (e.g., 3 signatures for mixed BC instead of 9)
- **Clear semantics** -- file names explicitly state which BC type they serve
- **Incremental adoption** -- can add OBC support without modifying existing PBC files at all
- **Faster loading** -- smaller files mean faster lazy-load times

### Cons
- **File proliferation** -- for each `(truncation, dim)`, need up to ~8 BC-type variants (d=3). With current 15 truncation+dim combos, that's up to ~120 files instead of 15.
- **Redundant data** -- the "interior" signature data is identical across PBC, OBC, and all mixed BC files for the same (truncation, dim). It's duplicated in every file.
- **Complex path registry** -- the registry grows significantly and must map BC configurations to file paths
- **pyclebsch generation overhead** -- must generate data for each BC type separately, even though interior plaquettes produce the same data each time
- **Mixed BC combinatorics** -- for d=3, there are `2^3 - 2 = 6` mixed BC types (excluding all-PBC and all-OBC), each potentially needing its own file
- **Not forward-compatible** -- adding a new BC configuration requires generating and shipping new data files

---

## Option 3: Signature-Indexed Data (Shared Signature Pool)

### Description

Instead of organizing data by lattice BC type, organize it by **signature**. Maintain a pool of signature-indexed data, and have ymcirc select the signatures it needs at runtime based on the plaquettes in the current lattice.

Two sub-variants:

### Option 3a: Signatures as Separate Files

Each file contains data for exactly one signature. ymcirc loads only the signature files it needs.

- **File naming:** e.g., `B3_dim(2)_sig_((1,2,-1,-2),(1,2,-1,-2),(1,-1,-2),(1,-1,-2))_magnetic_hamiltonian.json.gz`
- **At runtime:** ymcirc enumerates the distinct signatures on its lattice, loads the corresponding files

### Option 3b: Signatures in a Single File with Lazy Per-Signature Loading

One file per `(truncation, dim)` (like Option 1), but the `LazyDict` mechanism is extended to load individual signatures lazily rather than the entire file at once.

### Pros
- **Maximum granularity and reusability** -- each signature's data is computed once, stored once, usable by any lattice that has that signature
- **No redundancy** -- the "interior" signature is stored exactly once
- **Demand-driven loading (3a)** -- only load signatures actually needed
- **Naturally scales** -- adding new dimensions or BC types doesn't require regenerating existing data

### Cons
- **Many files (3a)** -- up to ~54 files per (truncation, dim) for d=3. With 15 truncation+dim combos: ~810 files.
- **Complex file naming (3a)** -- signature tuples in filenames are unwieldy
- **Discovery problem (3a)** -- ymcirc needs to know which signature files exist without loading them all. Requires a manifest or naming convention.
- **Implementation complexity** -- significant refactor of the data loading pipeline. The current `LazyDict` keyed by `(dim, trunc)` doesn't map cleanly to per-signature access.
- **Marginal benefit over Option 1** -- Option 1 already has signature-level keying within each file. The only advantage of Option 3 is avoiding loading data for unused signatures, which is a minor memory optimization given file sizes.
- **Complicates pyclebsch generation** -- must either generate one file per signature or restructure the output pipeline

---

## Comparative Analysis

| Criterion | Option 1 (Universal) | Option 2 (Per-BC-Type) | Option 3 (Per-Signature) |
|-----------|----------------------|------------------------|--------------------------|
| **Number of data files** | Same as today (~15 pairs) | ~120 pairs (d=3 worst case) | ~810 files (d=3 worst case) |
| **Data redundancy** | None (each signature stored once per file) | High (interior data duplicated across BC types) | None |
| **File size** | Larger (up to ~54x for d=3) | Smaller per file | Smallest per file |
| **Path registry complexity** | Unchanged | 3-key lookup | Manifest + dynamic loading |
| **ymcirc code changes** | Minimal beyond shared work | Moderate (BC-type mapping) | Significant (loading refactor) |
| **pyclebsch generation** | One run per (trunc, dim) | One run per (trunc, dim, BC) | Complex output restructuring |
| **Forward compatibility** | Excellent | Poor (new BC = new files) | Excellent |
| **Backward compatibility** | Good (old files are subsets) | Good (old PBC files unchanged) | Poor (complete restructure) |
| **Runtime performance** | Slightly more memory | Faster load per lattice | Best memory if lazy-loaded |

---

## Recommendation

**Option 1 (Universal Data Files) is the strongest choice.** Here's why:

1. **The data format already supports it.** Matrix element data is already keyed by signature within each plane. Universal files simply have more entries in an existing dict level. No structural format change is needed.

2. **The ymcirc lookup pipeline already supports it.** `_resolve_hamiltonian_for_plaquette()` already filters by `(plane, signature)`. On a PBC lattice, it will find one matching signature; on an OBC lattice, it will find a different one. The same code path works for both.

3. **Minimal file management.** Same number of files as today. No BC-type classification needed. No manifest or discovery mechanism.

4. **Forward-compatible.** Any future BC configuration (e.g., anti-periodic BCs, if ever needed) works with the same data files as long as the signatures are covered.

5. **File size growth is manageable.** For d=2: 9x growth (9 signatures vs 1). For d=3: up to 54x growth. But the base file sizes are modest (the physics data per signature is small), so even 54x is likely well under 100MB per file for reasonable truncation levels. This should be validated empirically.

6. **One pyclebsch generation run per (truncation, dim).** Use a large OBC lattice (e.g., `[4,4,1]` for d=2, `[4,4,4]` for d=3) to capture all signatures. This is already what the active non-PBC cases in `gen_ymcirc_data.py` do.

The main risk is file size at high B-truncation cutoffs for d=3, where 54 signatures each with many irrep combinations could produce large files. If this becomes a problem, Option 3b (lazy per-signature loading within a single file) could be adopted as an optimization without changing the external file structure.

### Suggested Implementation Order

1. **Generate universal data files** using pyclebsch (modify `gen_ymcirc_data.py` to produce one file per (trunc, dim) covering all signatures)
2. **Update ymcirc path registry** to point to universal files (or keep backward-compatible dual entries)
3. **Implement shared geometry/encoding changes** (the 11 items listed in "Shared Work" above)
4. **Implement variable-length bitstring encoding** in `LatticeStateEncoder` (the hardest part)
5. **Test with the existing `test_signature_nonperiodic` placeholder** and add new tests for OBC/mixed BC circuit construction

### Open Question: Plaquette State Files

The above analysis focuses on matrix element data, but the same question applies to plaquette state files. For Option 1, the universal plaquette state file would contain the union of all physical states across all signatures. The plaquette state list doesn't need per-signature keying because it's used for initialization/validation, not for per-plaquette lookup. The current deduplication in `gen_ymcirc_data.py` (line 395: `list(set(plaq_states))`) already handles this correctly.

---

## Appendix: Signature Counts by Configuration

| Lattice | PBCs | Dim | Planes | Signatures per plane | Total distinct signatures |
|---------|------|-----|--------|---------------------|--------------------------|
| `[N,M,1]` | `[T,T,F]` | 2 | 1 | 1 | 1 |
| `[N,M,1]` | `[T,F,F]` | 3/2 | 1 | 1 | 1 |
| `[N,M,1]` | `[F,F,F]` | 2 | 1 | up to 9 | up to 9 |
| `[N,M,1]` | `[T,F,F]` or `[F,T,F]` | 2 | 1 | up to 3 | up to 3 |
| `[N,M,L]` | `[T,T,T]` | 3 | 3 | 1 | 3 |
| `[N,M,L]` | `[F,F,F]` | 3 | 3 | up to 18 | up to 54 |
| `[2,2,1]` | `[F,F,F]` | 2 | 1 | 1 | 1 |
| `[2,2,2]` | `[F,F,F]` | 3 | 3 | 2 | 6 |
| `[4,4,1]` | `[F,F,F]` | 2 | 1 | 9 | 9 |
| `[4,4,4]` | `[F,F,F]` | 3 | 3 | up to 18 | up to 54 |

"Up to" counts assume the lattice is large enough (>= 4 sites along each non-periodic direction) to contain all possible signature types. Smaller lattices have fewer distinct signatures.
