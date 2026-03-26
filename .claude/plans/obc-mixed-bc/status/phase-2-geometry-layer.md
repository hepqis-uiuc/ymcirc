# Phase 2: ymcirc Geometry Layer — Status

**Status: COMPLETED** (2026-03-26)

## What was done

All 7 steps implemented in `ymcirc/_abstract/lattice_data.py` and `ymcirc/lattice_registers.py`:

- **2.1**: `_validate_lattice_params()` now accepts tuple-valued `periodic_boundary_conds`. Added `_periodic_boundary_conds_per_direction()` helper that normalizes bool/tuple BCs to a per-direction tuple (d=3/2 vertical is always non-periodic).
- **2.2**: `add_unit_vector_to_vertex_vector()` applies per-direction wrapping (periodic) or raises `KeyError` (non-periodic out-of-bounds).
- **2.3**: `_normalize_link_address()` applies per-direction wrapping/boundary checking. Replaces the old d=3/2 special case with the general per-direction approach.
- **2.4**: `get_traversal_order()` skips links that exit non-periodic boundaries. Replaces the old d=3/2-specific vertical link skip with the general check.
- **2.5**: `n_plaquettes` uses a general formula: for each plane (i,j), count = effective_i * effective_j * product(other_dims), where effective_k = shape[k] if periodic else shape[k]-1.
- **2.6**: `n_control_links_per_plaquette` raises `ValueError` for non-periodic lattices (count is not uniform). Directs callers to `Plaquette.control_links_per_vertex`.
- **2.7**: `get_vertex()` in `lattice_registers.py` wraps periodic directions, validates non-periodic directions.

Also updated `_configure_lattice()` to exclude boundary links from `_all_link_addresses` for non-periodic directions. Removed unused `comb` import and unused `VERTICAL_DIR_LABEL`/`VERTICAL_NUM_VERTICES_D_THREE_HALVES` imports from `lattice_registers.py`.

## What failed

Nothing — all changes are clean.

## Deviations from plan

None.

## Verification

All 208 existing tests pass with zero regressions (`uv run pytest -v`).
