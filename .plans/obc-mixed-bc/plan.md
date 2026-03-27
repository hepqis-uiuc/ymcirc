# OBC / Mixed BC Implementation Plan (Option 1: Universal Data Files)

Complexity: **high**

This plan implements support for open boundary conditions (OBC) and mixed boundary conditions in ymcirc, using the "Universal Data Files" strategy from `obc-mixed-bc-implementation-study.md`. One data file per `(truncation, dim)` contains matrix elements and plaquette states for ALL possible signatures (interior, edge, corner), serving any BC configuration.

The plan has four phases. Phases 1 and 2 are independent of each other and can be worked in parallel. Phase 3 depends on both. Phase 4 (testing) runs throughout but has a final integration pass at the end.

---

## Phase 1: pyclebsch — Generate Universal Data Files

Modify `gen_ymcirc_data.py` to produce universal data files that contain all possible signatures for each `(truncation_mode, cutoff, dim)` combination, then generate the files.

- [x] **1.1 Design universal lattice cases in `gen_ymcirc_data.py`**
  For each existing PBC lattice case, create a corresponding universal case that uses a large-enough OBC lattice to capture all signatures. For d=2, a `[4,4,1]` `[F,F,F]` lattice has all 9 signature types. For d=3/2, a `[4,2,1]` `[F,F,F]` lattice has all 3. For d=3, a `[4,4,4]` `[F,F,F]` lattice has all signature types. The `site_coords_for_comp` should cover all coordinates. Remove BC-specific labels from output filenames (or use a new naming convention without `_PBC`).
  - Acceptance: `gen_ymcirc_data.py` has one lattice case per `(trunc, dim)` that covers all signatures, with clear comments explaining the strategy.
  - Failure: Cannot determine a single lattice configuration that captures all signatures for a given dimension after three attempts.

- [x] **1.2 Update output file naming**
  Remove `_PBC` from generated filenames. New convention: `{TruncLabel}_dim({DimSpec})_magnetic_hamiltonian.json.gz` and `{TruncLabel}_dim({DimSpec})_plaquette_states.json.gz`. The older T-series files already follow this convention (no `_PBC` suffix).
  - Acceptance: Generated files use the new naming convention. Metadata in each file includes `PBCs` and `site_coords_for_comp` reflecting the universal generation lattice.
  - Failure: Naming conflicts with existing files that cannot be resolved cleanly.

- [x] **1.3 Run data generation for T-truncation universal files**
  Generate universal data files for all T-truncation cases: T1 dim(3/2), T1 dim(2), T2 dim(3/2). Verify each file contains the expected number of signatures per plane (e.g., 9 for d=2, 3 for d=3/2).
  - Acceptance: Files generated, loadable, and contain the expected signature count. All existing PBC signatures are present as a subset.
  - Failure: Generation fails or takes unreasonably long (>1 hour per case). Matrix element values for the PBC signature do not match existing PBC-only files.

- [x] **1.4a Debug pyclebsch KeyError for B6+ d=3/2 OBC boundaries**
  `calc_plaquette_elements()` raises `KeyError: ((2, 1, 0), (2, 2, 0))` in `glue_plaquette_site_factors()` when computing matrix elements at OBC boundary vertices for B6–B10 d=3/2. The site factor lookup table is missing entries for irrep combinations that appear at boundary sites with these higher truncations. Investigate the root cause in pyclebsch and fix.
  - Acceptance: B6 d=3/2 universal file generates successfully. PBC-signature matrix elements match existing B6 PBC data.
  - Failure: The fix requires substantial restructuring of pyclebsch's site factor computation. If so, flag for discussion — may need to defer higher B-truncation OBC support.

- [x] **1.4b Run data generation for remaining B-truncation universal files**
  Generate universal data files for: B4–B10 d=3/2 (after 1.4a fix), B3 d=2, and B3 d=3. B3 and B5 d=3/2 are already done. Validate PBC-signature matrix elements against existing PBC data.
  - Acceptance: Files generated. PBC-signature matrix elements match existing data. File sizes are reasonable (<100MB each).
  - Failure: File sizes exceed 100MB, or generation takes >4 hours per case. If so, flag for discussion before proceeding.

- [x] **1.5 Install universal data files in ymcirc**
  Copy generated files to `ymcirc/_ymcirc_data/` subdirectories. Keep existing PBC-only files temporarily (for regression testing) but mark them as deprecated.
  - Acceptance: Universal files present in `_ymcirc_data/`. Both old and new files coexist.
  - Failure: Files too large for the repository.

---

## Phase 2: ymcirc Geometry Layer — Support Non-Periodic Lattices

Implement the foundational geometry and lattice traversal changes in `_abstract/lattice_data.py` and `lattice_registers.py` that all downstream code depends on. This phase does NOT touch the data loading or circuit construction layers.

- [x] **2.1 Accept tuple-valued `periodic_boundary_conds`**
  In `LatticeDef._validate_lattice_params()` (~line 341), replace the `NotImplementedError` with logic to accept a tuple of bools (one per spatial direction). Normalize `bool` input to a tuple internally. Update `periodic_boundary_conds` property accordingly. The existing `all_boundary_conds_periodic` property (~line 496) already handles tuples correctly.
  - Acceptance: `LatticeDef(dim=2, lattice_size=3, periodic_boundary_conds=(True, False))` constructs without error. `all_boundary_conds_periodic` returns `False`. `LatticeDef(dim=2, lattice_size=3, periodic_boundary_conds=True)` still works as before.
  - Failure: Type errors or regressions in existing PBC tests after three fix attempts.

- [x] **2.2 Implement `add_unit_vector_to_vertex_vector()` for non-periodic lattices**
  At ~line 530, implement the non-periodic branch: add the unit vector without modular wrapping. For mixed BCs, wrap only in periodic directions. Raise `KeyError` (or return `None`) if the result would step off the lattice boundary in a non-periodic direction.
  - Acceptance: Adding `+x` to a vertex at the right boundary of a non-periodic x-direction raises `KeyError`. Adding `+x` to an interior vertex returns the correct unwrapped result. Mixed BCs wrap only in periodic directions. Existing PBC behavior unchanged.
  - Failure: Cannot define clean semantics for "stepping off the edge" after three attempts.

- [x] **2.3 Implement `_normalize_link_address()` for non-periodic lattices**
  At ~line 413, implement the non-periodic branch. When a negative direction steps back to a vertex that doesn't exist (would be at coordinate -1 in a non-periodic direction), raise `KeyError`. For mixed BCs, wrap only in periodic directions.
  - Acceptance: Requesting a link at the boundary that would require wrapping in a non-periodic direction raises `KeyError`. Interior links normalize correctly. PBC behavior unchanged.
  - Failure: Edge cases in mixed BC normalization cannot be resolved after five attempts.

- [x] **2.4 Implement `get_traversal_order()` for non-periodic lattices**
  At ~line 588, implement non-periodic traversal. The traversal must enumerate all valid `(vertex, plane)` pairs where the plaquette exists (i.e., all four vertices are within bounds). For an OBC lattice of size N in d=2, this means vertices `(i,j)` with `i < N-1, j < N-1`.
  - Acceptance: For a 3x3 d=2 OBC lattice, traversal yields 4 plaquettes (2x2 grid). For a 3x3 PBC lattice, traversal still yields 9 plaquettes. Plaquette objects constructed during traversal have correct signatures (boundary vertices have fewer control links).
  - Failure: Plaquette construction fails during traversal due to unhandled `KeyError` cascades.

- [x] **2.5 Make `n_plaquettes` boundary-aware**
  At ~line 432, compute the correct plaquette count for OBC and mixed BC lattices. For a d-dimensional hypercubic lattice of size N with full OBC: `C(d,2) * (N-1)^d`. For mixed BCs, each direction contributes either `N` (periodic) or `N-1` (open) to the product.
  - Acceptance: `n_plaquettes` returns correct values for: d=2 PBC size 3 (9), d=2 OBC size 3 (4), d=2 mixed (T,F) size 3 (6). For d=3 OBC size 3: 3*8=24 (three planes, each 2^3=8... actually C(3,2)*(3-1)^2 per plane... need to verify formula).
  - Failure: Formula doesn't generalize to mixed BCs after three attempts.

- [x] **2.6 Handle `n_control_links_per_plaquette` for non-periodic lattices**
  At ~line 455. For non-periodic lattices, the number of control links varies per plaquette. Options: (a) return the maximum across all plaquettes, (b) return `None` and require callers to use per-plaquette counts, or (c) deprecate this property for non-periodic lattices and raise `ValueError`. Choose the option that minimizes downstream changes.
  - Acceptance: Property returns a meaningful value (or raises a clear error) for OBC lattices. All existing callers of this property are identified and handled.
  - Failure: No single strategy works for all callers after examining all usage sites.

- [x] **2.7 Implement `get_vertex()` without modular wrapping in `lattice_registers.py`**
  At ~line 160, implement the non-periodic branch of `get_vertex()`. For non-periodic directions, the vertex coordinates are used as-is (no mod). For out-of-bounds coordinates, raise `KeyError`.
  - Acceptance: `get_vertex((0,0))` works on both PBC and OBC lattices. `get_vertex((N,0))` raises `KeyError` on an OBC lattice of size N but wraps to `(0,0)` on a PBC lattice.
  - Failure: Cannot determine out-of-bounds without access to lattice shape (should already be available via `LatticeDef`).

---

## Phase 3: ymcirc Encoding + Circuit Layer — Wire Up Non-Periodic Support

Connect the geometry layer (Phase 2) to data loading and circuit construction. This phase depends on both Phase 1 (data files exist) and Phase 2 (geometry works).

- [x] **3.1 Update path registry to point to universal data files**
  In `conventions.py` (~lines 183-229), update `_HAMILTONIAN_DATA_FILE_PATHS` and `_PLAQUETTE_STATES_DATA_FILE_PATHS` to point to the new universal filenames (without `_PBC`). For backward compatibility during transition, support both old and new filenames (check for universal file first, fall back to PBC-only file).
  - Acceptance: `HAMILTONIAN_BOX_TERMS["d=2"]["T1"]` loads the universal file. Existing code that loads PBC data still works.
  - Failure: Import-time errors due to missing files.

- [x] **3.2 Implement variable-length bitstring encoding in `LatticeStateEncoder`**
  This is the hardest single step. Currently, `LatticeStateEncoder` assumes uniform `n_control_links_per_plaquette` for all plaquettes (~line 649). For non-periodic lattices, different plaquettes have different control link counts, so bitstring lengths vary by signature.

  Approach: make the encoding signature-aware. Add a method that encodes/decodes plaquette states given a specific signature (which determines the control link count at each vertex). The existing `encode_plaquette_state()` and `decode_bit_string_to_plaquette_state()` should dispatch based on whether the lattice is periodic (uniform encoding) or non-periodic (signature-dependent encoding).
  - Acceptance: Can encode and decode plaquette states for both PBC and OBC lattices. Round-trip `decode(encode(state)) == state` for all physical states of a given signature. Existing PBC encoding is unchanged.
  - Failure: The variable-length encoding breaks assumptions in circuit stitching that depend on uniform bitstring lengths. If so, flag for replanning.

- [x] **3.3 Update `load_magnetic_hamiltonian()` for variable-length bitstrings**
  `load_magnetic_hamiltonian()` in `conventions.py` encodes plaquette states as bitstrings. For universal data files, the plaquette states in the file come from multiple signatures with different control link counts. The encoding must be signature-aware: when encoding a `(plane, signature)` pair's matrix elements, use the control link count implied by that signature.
  - Acceptance: `load_magnetic_hamiltonian()` returns correctly encoded data for universal files. Bitstring pairs for different signatures have different lengths. Existing PBC data loading produces identical results to before.
  - Failure: Cannot determine signature from the data file's key structure. If so, restructure the loading to pass signature information through.

- [x] **3.4 Remove PBC-only guard in `LatticeCircuitManager.__init__()`**
  At `circuit.py` ~line 82, remove or relax the `NotImplementedError` that blocks non-periodic lattices. Ensure the small-lattice PBC logic (~lines 100-154) is properly gated by `self._lattice_is_periodic` (it already appears to be, but verify).
  - Acceptance: `LatticeCircuitManager` can be instantiated with an OBC `LatticeDef`. Small-lattice PBC logic does not activate for OBC lattices.
  - Failure: Other assumptions in `__init__` break for OBC lattices.

- [x] **3.5 Handle variable control qubit counts in circuit stitching**
  In `circuit.py`, `apply_magnetic_trotter_step()` (~lines 600-686) stitches per-plaquette rotation circuits into the master circuit. The control qubit collection logic assumes a uniform number of controls per plaquette. For OBC, the rotation circuit for a boundary plaquette has fewer control qubits.

  The cache key is already `(plane, signature)` (~line 610), so different-sized rotation circuits are cached separately. The stitching logic must use the actual control link count from the plaquette (via `Plaquette.control_links_ordered` or `control_links_per_vertex`), not a global constant.
  - Acceptance: `apply_magnetic_trotter_step()` produces a valid circuit on an OBC lattice. Interior plaquettes get full-size rotation subcircuits; boundary plaquettes get smaller ones. Circuit qubit counts are correct.
  - Failure: The Givens rotation construction in `givens.py` assumes a fixed qubit count and cannot handle variable sizes. If so, investigate and flag.

- [x] **3.6 Implement `decode_bit_string_to_plaquette_state()` for non-periodic lattices**
  At `conventions.py` ~line 878, implement the non-periodic branch. Use the plaquette's signature to determine the expected bitstring length and control link layout.
  - Acceptance: Can decode measurement bitstrings from OBC lattice circuits back into plaquette states. Round-trip consistency with encoding.
  - Failure: Measurement bitstring layout doesn't cleanly separate plaquettes with different control counts.

---

## Phase 4: Testing and Validation

Testing runs throughout, but this phase covers the final integration tests.

- [x] **4.1 Unit tests for geometry layer (Phase 2)**
  Add tests in `tests/` for each geometry change:
  - Tuple-valued BCs in `LatticeDef`
  - `add_unit_vector_to_vertex_vector()` on OBC and mixed BC lattices
  - `_normalize_link_address()` on OBC and mixed BC lattices
  - `get_traversal_order()` on OBC and mixed BC lattices (correct plaquette count, correct signatures)
  - `n_plaquettes` for various BC configurations
  - `get_vertex()` on OBC lattices
  - Unskip and implement `test_signature_nonperiodic` (`tests/test_circuit.py` ~line 1768)
  - Acceptance: All new tests pass. All existing tests still pass (`uv run pytest -v`).
  - Failure: Existing tests break due to Phase 2 changes.

- [x] **4.2 Unit tests for encoding layer (Phase 3)**
  Test variable-length bitstring encoding/decoding:
  - Encode/decode round-trip for each distinct signature on a d=2 OBC lattice
  - Verify PBC encoding is unchanged
  - Test `load_magnetic_hamiltonian()` with universal data files
  - Acceptance: All round-trip tests pass. PBC regression tests pass.
  - Failure: Encoding inconsistencies between signatures.

- [x] **4.3 Integration test: OBC circuit construction**
  Construct a full Trotter step circuit on a small OBC lattice (e.g., 2x2 d=2 OBC, single plaquette):
  - Verify circuit builds without error
  - Verify circuit has the correct number of qubits (fewer than PBC equivalent)
  - Verify measurement and parsing work end-to-end
  - Acceptance: Full Trotter step circuit (magnetic + electric) builds and runs on simulator for a 2x2 OBC lattice.
  - Failure: Circuit construction fails or produces incorrect qubit counts.

- [x] **4.4 Integration test: Mixed BC circuit construction**
  Construct a Trotter step circuit on a mixed BC lattice (e.g., 3x3 d=2 with x-periodic, y-open):
  - Verify correct plaquette count (fewer than full PBC, more than full OBC)
  - Verify signatures match expectations (3 distinct types)
  - Acceptance: Circuit builds and runs on simulator.
  - Failure: Mixed BC traversal or signature resolution fails.

- [x] **4.5 Regression: all existing PBC tests pass**
  Run the full test suite to confirm no regressions.
  - Acceptance: `uv run pytest -v` passes with no new failures.
  - Failure: Any existing test fails.

- [x] **4.6 Data validation: universal files match existing PBC data**
  For each `(trunc, dim)` with existing PBC-only data files, verify that the universal file's PBC-signature entries produce identical matrix elements (within floating-point tolerance).
  - Acceptance: All PBC-signature matrix elements in universal files match the existing PBC-only files to within 1e-12 relative error.
  - Failure: Discrepancies beyond floating-point noise.

---

## Dependency Graph

```
Phase 1 (data gen)  ──────────┐
                               ├──→ Phase 3 (encoding + circuit) ──→ Phase 4.2-4.6
Phase 2 (geometry) ───────────┘
         │
         └──→ Phase 4.1 (geometry tests)
```

Phases 1 and 2 can proceed in parallel. Phase 3 requires both to be complete. Phase 4.1 can run as soon as Phase 2 is done; the remaining Phase 4 tests require Phase 3.
