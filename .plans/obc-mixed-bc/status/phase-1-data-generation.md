# Phase 1: pyclebsch — Generate Universal Data Files — Status

**Status: Steps 1.1–1.4a COMPLETED. Steps 1.4b and 1.5 remain.**

## What was done

- **1.1**: Redesigned `gen_ymcirc_data.py` with universal lattice cases (`lattice_cases_T_universal` and `lattice_cases_B_universal`). Each uses an OBC lattice large enough to capture all possible signatures. Old PBC-only and specialized cases moved to commented-out sections.

- **1.2**: Universal files use naming without `_PBC` or `_OBC` suffix (e.g., `T1_dim(2)_magnetic_hamiltonian.json.gz`). Fixed a bug in the original script where d=2 `site_coords_for_comp` used 2-tuples instead of the required 3-tuples.

- **1.3**: Generated and validated T-truncation universal files:
  - T1 d=3/2: 3 signatures, 81 interior-signature matrix elements match old PBC file exactly.
  - T2 d=3/2: 3 signatures, generated successfully.
  - T1 d=2: 9 signatures, 19329 interior-signature matrix elements match old PBC file exactly.
  - B3 d=3/2: 3 signatures, generated successfully.
  - B5 d=3/2: 3 signatures, generated successfully.
  - All files in `pyclebsch/out/`.

- **1.4a**: **FIXED.** Root cause: on OBC lattices with B-truncation, boundary vertices (fewer half-links) allow higher-Casimir irreps in their singlets than adjacent interior vertices (more half-links). When `glue_plaquette_site_factors()` seeds from a boundary site 1 with irreps like `(2,1,0)` or `(2,2,0)`, it tries to look up matching site factors at site 2/3, which don't have those irreps in their (smaller) singlet sets. The s4 lookup already had a guard at line 345, but s2 and s3 did not.

  **Fix**: Added two guards in `glue_plaquette_site_factors()` (`plaquette_matrix_elements.py`):
  - Line 323: `if (s1[0],s1[2]) not in info[2]: return matrix_elements` — early return for s2 mismatch
  - Line 337: `if (s2[1],s2[3]) not in info[3]: continue` — skip for s3 mismatch

  These guards correctly produce no matrix elements for incompatible irrep combinations, consistent with the physics (plaquette link irreps must match at shared links).

  **Validation**: B6 d=3/2 all 3 plaquettes generate successfully (34 + 1000 + 34 matrix elements). PBC regression tests pass (B5 d=3/2: 81 me, T1 d=2: 19329 me). All 14 pyclebsch tests pass.

## What failed

- **B6–B10 d=3/2** previously failed with KeyError — now fixed by step 1.4a.

## Deviations from plan

- **Step 1.5 deferred**: Installing universal files into `ymcirc/_ymcirc_data/` now overwrites the existing PBC-only files and breaks 20 tests because `conventions.py` loading code doesn't handle multi-signature data yet. Installation must be done as part of Phase 3 when the encoding layer is updated.

- **Parallelization disabled**: Set `parallelize = False` in gen_ymcirc_data.py to avoid known EOFError in multiprocessing.

## Next steps

- **Step 1.4b**: Generate remaining B-truncation universal files: B4, B6–B10 d=3/2 (now unblocked), B3 d=2, B3 d=3.
- **Step 1.5**: Install universal files into `ymcirc/_ymcirc_data/` (deferred to Phase 3).
