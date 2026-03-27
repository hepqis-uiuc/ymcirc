# Phase 3: ymcirc Encoding + Circuit Layer — Status

**Status: COMPLETE — All steps (3.1–3.6) done. All tests passing.**

## What was done

- **3.1**: **COMPLETED.** Updated `_HAMILTONIAN_DATA_FILE_PATHS` and `_PLAQUETTE_STATES_DATA_FILE_PATHS` in `conventions.py` to point to universal filenames (without `_PBC`) for all cases where universal files now exist. Added B4 d=3/2 entry (new truncation). Cases still using PBC-only files: B4 d=2, B7 d=2, B4 d=3 (no universal files generated for these — they weren't in scope for step 1.4b).

- **3.2**: **COMPLETED.** Key changes made to `LatticeStateEncoder`:
  - **Constructor filtering**: For periodic lattices, the constructor now automatically filters plaquette states from universal data files, keeping only those with the interior-signature control count. This is logged. Exposed filtered states via a new `physical_plaquette_states` property.
  - **Variable control count support**: Removed the uniform control count validation for non-periodic lattices. For non-periodic, `_expected_plaquette_bit_string_length` is set to `None`.
  - **Encoding**: `encode_plaquette_state_as_bit_string()` skips control count validation for non-periodic lattices (already handles variable per-vertex controls naturally by iterating `c_links`).
  - **Decoding**: `decode_bit_string_to_plaquette_state()` now accepts an optional `n_controls_per_vertex` tuple for non-periodic lattices (replaces the old `NotImplementedError`).

- **3.3**: **COMPLETED.** Updated `load_magnetic_hamiltonian()` to skip state pairs with incompatible control counts on periodic lattices. For periodic encoders, it determines the expected control count from the lattice geometry and filters out boundary-signature state pairs before encoding. Non-periodic path encodes all state pairs (variable-length bitstrings are fine since resolution always filters by signature).

- **3.4**: **COMPLETED (2026-03-27).** Removed the `NotImplementedError` at `circuit.py` line 82 that blocked non-periodic lattices. Replaced `if not ... : raise` with `if ...: self._lattice_is_periodic = True`. The `self._lattice_is_periodic` flag already defaults to `False` (line 79), so non-periodic lattices just skip the flag set. All small-lattice PBC logic (lines 101-113 and 120-144) is properly gated by `self._lattice_is_small and self._lattice_is_periodic` — verified it does not activate for OBC lattices. The "else" pivot branch (line 145-151) correctly handles non-periodic lattices by pivoting to per-plane-first format without filtering/trimming.

- **3.5**: **COMPLETED (2026-03-27).** Modified `apply_magnetic_trotter_step()` to handle boundary vertices where plaquettes extend beyond the lattice. For d=1.5 and d=2 (single plaquette per vertex), wrapped `get_plaquettes()` in try/except KeyError — if the plaquette would go off-boundary, the vertex is skipped. For d>=3 (multiple planes per vertex), each plane is constructed individually so that planes extending beyond the boundary are skipped while valid planes at the same vertex are kept. The existing stitching logic (Hamiltonian resolution by plane+signature, qubit collection via `control_links_per_vertex`) already handles variable control counts naturally since it iterates over actual plaquette contents.

- **3.6**: **COMPLETED.** The `decode_bit_string_to_plaquette_state()` method now accepts `n_controls_per_vertex` for non-periodic decoding. End-to-end testing deferred to Phase 4.

## Test fixes applied (prior session, 2026-03-26)

Six test failures in `test_conventions.py` were resolved:

1. **`test_hamiltonian_box_terms_no_unexpected_cases`**: Added `"B4"` to the `d=3/2` expected truncation set (new data file added in step 1.4b).
2. **`test_lattice_encoder_fails_on_bad_creation_args`**: The first assertion expected `ValueError` on inconsistent control lengths, but periodic lattices now silently filter by control count. Updated to verify that filtering produces the correct single state instead of raising.
3. **`test_all_mag_hamiltonian_plaquette_states_have_unique_bit_string_encoding`** (4 parametrized cases: d=3/2 T1, d=3/2 T2, d=2 T1, d=3 B3): These iterated over ALL states from `HAMILTONIAN_BOX_TERMS` including boundary-signature states with fewer control links. Added filtering to only encode interior-signature states (matching the periodic lattice's `n_control_links_per_plaquette`).

## Current test status

Full suite passes: **208 passed, 20 skipped, 0 failures** (as of 2026-03-27, after steps 3.4 and 3.5).

## What failed

- Nothing failed during implementation. Steps 3.4 and 3.5 required no test changes — existing tests continue to pass as-is.

## Previous test updates (prior sessions)

- `test_circuit.py`: 3 places updated to use `encoder.physical_plaquette_states` or `lattice_encoder.physical_plaquette_states` instead of raw `PHYSICAL_PLAQUETTE_STATES[...][...]` for encoding/iteration.
- `test_conventions.py`: `test_physical_plaquette_states_data_valid` changed from asserting exact control count equality to asserting `<= max` since universal files contain boundary-signature states with fewer controls.
