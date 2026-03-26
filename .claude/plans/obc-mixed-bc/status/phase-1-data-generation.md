# Phase 1: pyclebsch — Generate Universal Data Files — Status

**Status: Steps 1.1–1.3 COMPLETED. Step 1.4 PARTIALLY BLOCKED. Step 1.5 DEFERRED to Phase 3.**

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

## What failed

- **B6–B10 d=3/2**: `calc_plaquette_elements()` raises `KeyError: ((2, 1, 0), (2, 2, 0))` in `glue_plaquette_site_factors()` when computing matrix elements at OBC boundary vertices. The site factor lookup table doesn't contain entries for irrep combinations that appear at boundary sites with these higher truncations. This is a **pyclebsch limitation**, not a gen_ymcirc_data.py issue.

- **B-truncation d=2 and d=3**: Deferred due to expected long generation times (step 1.4 acceptance criteria allows up to 4 hours per case).

## Deviations from plan

- **Step 1.5 deferred**: Installing universal files into `ymcirc/_ymcirc_data/` now overwrites the existing PBC-only files and breaks 20 tests because `conventions.py` loading code doesn't handle multi-signature data yet. Installation must be done as part of Phase 3 when the encoding layer is updated.

- **Parallelization disabled**: Set `parallelize = False` in gen_ymcirc_data.py to avoid known EOFError in multiprocessing.

## Next steps

- Phase 3 step 3.1: Update conventions.py path registry and loading code to handle multi-signature universal files.
- Phase 3 step 3.1: At that point, install universal files from `pyclebsch/out/` and update `_HAMILTONIAN_DATA_FILE_PATHS` / `_PLAQUETTE_STATES_DATA_FILE_PATHS`.
- **Step 1.4a**: Debug pyclebsch `KeyError` in `glue_plaquette_site_factors()` for B6+ d=3/2 OBC boundary sites — missing site factor entries for irrep combo `((2,1,0),(2,2,0))`.
- **Step 1.4b**: After 1.4a fix, generate B4–B10 d=3/2, B3 d=2, and B3 d=3 universal files.
