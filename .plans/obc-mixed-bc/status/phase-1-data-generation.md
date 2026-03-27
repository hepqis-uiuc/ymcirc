# Phase 1: pyclebsch — Generate Universal Data Files — Status

**Status: COMPLETED** (2026-03-26)

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

  **Validation**: B6 d=3/2 all 3 plaquettes generate successfully (34 + 1000 + 34 matrix elements). PBC regression tests pass (B5 d=3/2: 81 me, T1 d=2: 19329 me). All 14 pyclebsch tests pass.

- **1.4b**: **COMPLETED.** Generated all remaining B-truncation universal files:
  - B4 d=3/2, B6–B10 d=3/2: all generated successfully (3 signatures each).
  - B3 d=2: 9 signatures, generated successfully (3.5K hamiltonian, 7.8K plaquette states).
  - B3 d=3: 27 signatures, generated successfully (64K hamiltonian, 1.5M plaquette states).
  - All files under 100MB (largest is B3 d=3 plaquette states at 1.5MB).
  - All files passed internal consistency validation (mat elem states ⊆ plaquette states).

- **1.5**: **COMPLETED.** Copied all universal files from `pyclebsch/out/` to `ymcirc/_ymcirc_data/` subdirectories. Old PBC-only files retained alongside universal files.

## What failed

- **B6–B10 d=3/2** previously failed with KeyError — fixed by step 1.4a.

## Deviations from plan

- **Parallelization disabled**: Set `parallelize = False` in gen_ymcirc_data.py to avoid known EOFError in multiprocessing.
- **Step 1.5 was done alongside 3.1** (path registry update) rather than separately.
- Used a temporary `gen_remaining_universal.py` script to run only the new cases without re-generating existing files.

## Resolution

Phase 1 is fully complete. All universal data files generated and installed.
