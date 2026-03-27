# Phase 4: Testing and Validation — Status

## 4.1 Unit tests for geometry layer — COMPLETE

**What was done:**

1. Created `tests/test_geometry_obc.py` with 57 tests covering all Phase 2 geometry changes:
   - `TestTupleBoundaryConditions` (11 tests): tuple-valued BCs in `LatticeDef`, validation, mixed BCs, d=3/2 vertical-periodic rejection, backwards compatibility with scalar bool
   - `TestAddUnitVectorOBC` (5 tests): interior steps, boundary KeyErrors, mixed BC wrapping, d=3 and d=3/2 OBC
   - `TestNormalizeLinkAddressOBC` (4 tests): positive/negative dir interior, boundary raises, mixed BC wrapping
   - `TestNPlaquettesOBC` (14 parametrized tests): d=2 OBC/PBC/mixed, d=3 OBC, d=3/2 OBC/PBC
   - `TestControlLinksPerPlaquetteOBC` (2 tests): raises ValueError for OBC, works for PBC
   - `TestTraversalOrderOBC` (6 tests): link counts, boundary vertices, mixed BC, plaquette count consistency, d=3 OBC
   - `TestGetVertexOBC` (4 tests): interior, out-of-bounds, PBC vs OBC behavior, mixed BC
   - `TestLinkCountsOBC` (2 tests): link count formulas for OBC and PBC
   - `TestPlaquetteOBC` (7 tests): corner vs interior control counts, boundary construction, no-repeat check, signature variation, interior signature matches PBC
   - `TestSignatureEnumeration` (2 tests): d=2 OBC 4x4 has 9 signatures, d=3/2 OBC has 3 signatures

2. Unskipped and implemented `test_signature_nonperiodic` in `tests/test_circuit.py`:
   - Verifies signatures at corner, edge, and interior positions on d=2 OBC size=4
   - Verifies d=3/2 OBC left-edge plaquette signature
   - All 4 sub-cases pass

**All 57 new geometry tests pass. The unskipped test_signature_nonperiodic also passes.**

**What failed:** Nothing.

**Deviations from plan:** None.

## 4.2 Unit tests for encoding layer — COMPLETE

**What was done:**

Created `tests/test_obc_encoding_and_integration.py` with 42 tests (1 slow-skipped) covering:

- `TestEncoderInitOBC` (6 tests): OBC encoder has `None` plaquette bitstring length, PBC has fixed length, OBC keeps all states, PBC filters boundary states
- `TestEncodeDecodeRoundTripOBC` (6 tests): round-trip encode/decode for d=3/2 interior, left-edge, and right-edge signatures; d=2 corner and interior signatures; different signatures produce different-length bitstrings
- `TestDecodeRequiresControlsForOBC` (1 test): decode raises ValueError without n_controls_per_vertex on OBC
- `TestPBCEncodingUnchanged` (4 tests): PBC T1 d=3/2 known encoding, PBC d=2 T1 round-trip, expected bitstring lengths unchanged
- `TestLoadMagneticHamiltonianOBC` (5 tests): OBC loads successfully, OBC has multiple bitstring lengths, PBC has single length, d=2 OBC loads, OBC entries have consistent control bit counts within state pairs
- `TestEncodeDecodeAllSignaturesD2` (1 test): all 9 distinct d=2 OBC signatures encode/decode correctly

**What failed:** Initially 3 d=3/2 round-trip tests failed because T1 d=3/2 has no vertex bitmap (max multiplicity 0), so vertices decode as `(None, None, None, None)`, not `(0, 0, 0, 0)`. Fixed by adjusting expected values.

**Deviations from plan:** None.

## 4.3 Integration test: OBC circuit construction — COMPLETE

**What was done:**

Added `TestOBCCircuitConstruction` class (6 tests) in the same file:

- d=3/2 OBC circuit builds without error
- d=2 OBC size-3 circuit builds without error
- OBC lattice has fewer qubits than PBC
- d=2 OBC plaquette count correct (4 for size 3)
- d=3/2 OBC plaquette count correct (3 for size 4)
- d=2 OBC size-4 B3 circuit builds (slow, skipped by default)

**What failed:** Nothing.

**Deviations from plan:** Electric Trotter step was not tested separately (only magnetic). The plan mentioned "magnetic + electric" but the electric step doesn't depend on boundary conditions — it operates per-link, not per-plaquette.

## 4.4 Integration test: Mixed BC circuit construction — COMPLETE

**What was done:**

Added `TestMixedBCCircuitConstruction` class (5 tests):

- d=2 mixed BC plaquette count correct (6 for size 3)
- Mixed BC plaquette count between OBC and PBC
- d=2 mixed BC circuit builds without error
- d=2 mixed BC (x-periodic, y-open) has 3 distinct signatures (using size 4)
- Mixed BC has fewer qubits than PBC

**What failed:** Initially the 3-signature test failed for size 3 (only 2 signatures exist because there's no room for a fully interior row). Fixed by using size 4.

**Deviations from plan:** None.

## 4.5 Regression: all existing PBC tests pass — COMPLETE

**What was done:**

Ran `uv run pytest -v` on the full test suite.

**Result:** 308 passed, 20 skipped, 1900 warnings in 471.69s. No failures.

**What failed:** Nothing.

**Deviations from plan:** None.

## 4.6 Data validation: universal files match existing PBC data — COMPLETE

**What was done:**

Added `TestUniversalDataMatchesPBC` class (7 parametrized tests):

- PBC Hamiltonian loads consistently from universal files (d=3/2 T1, T2, B3)
- OBC data contains PBC signature as subset (d=3/2 T1, T2)
- Interior (PBC) signature matrix element values are nonzero floats (d=3/2 T1, T2)
- d=3/2 universal data has all 3 expected signatures
- d=2 universal data has all 9 expected signatures

**What failed:** Nothing.

**Deviations from plan:** The plan called for comparing universal file values against existing PBC-only files to within 1e-12 tolerance. Since the universal files have *replaced* the PBC-only files (Phase 1 already verified consistency), the validation instead confirms structural correctness: PBC-signature entries exist, have the right count, and produce valid non-empty Hamiltonian data when loaded through the PBC encoder.
