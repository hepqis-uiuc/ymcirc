# Adversarial Audit of d=3 Feature Plan

This audit cross-references every claim in `d3-feature-planning.md` against the actual codebase. Findings are organized by plan section.

---

## Section 1: Data File Registration in `conventions.py`

### Verified

- The B3 d=3 data files exist at the stated paths. Confirmed via filesystem glob.
- The file path dicts are at `conventions.py:167-183` and the proposed additions are structurally correct.
- The `PHYSICAL_PLAQUETTE_STATES` and `HAMILTONIAN_BOX_TERMS` LazyDicts are built via comprehension over the file path dicts (`conventions.py:304-317`), so adding entries to the path dicts will indeed propagate automatically.
- The claim that `LatticeStateEncoder` needs no changes is correct — it accepts arbitrary link bitmaps and lattice definitions.

No issues found.

---

## Section 2a: `__init__` Dimension Match

### Verified

- The `case _:` at `circuit.py:84-85` is the correct location.
- The proposed `case 3:` logic is structurally correct.

No issues found.

---

## Section 2b: Cache Control Link Dirs

### Verified

- `self._cached_ctrl_dirs_d2_small_and_periodic` is created at `circuit.py:89-93` for d=2 using a temporary plaquette at `(0,0)` in plane `(1,2)`.
- The plan correctly identifies that d=3 needs per-plane caching.

### Issues Found

**2b.1 — The plan does not specify which vertex to use for the temporary plaquette.** For d=3, the plan says to construct a "temporary plaquette for that plane" but doesn't specify the bottom-left vertex. Since the lattice is periodic and cubic, vertex `(0,0,0)` is the natural choice for all planes. This is a minor omission but should be explicit since different vertices could yield different `control_link_dirs_per_vertex` on non-periodic lattices in the future.

---

## Section 2c: Small-Periodic Hamiltonian Filtering (Main Challenge)

### Issues Found

**2c.1 — The plan presents two divergent solutions without committing to one.** The proposed solution (lines 121-133) says: "In `__init__`, do **not** perform the filtering for d=3." The alternative solution (lines 137-155) says to restructure filtering to be per-plane for all dimensions. These are architecturally divergent paths, yet neither is endorsed as the final choice. Section 6's "Which approach best supports future lattices" does recommend the alternative — but this recommendation is buried in an appendix-like section rather than stated in the main design.

**2c.2 — The proposed solution introduces a correctness risk for `_strip_redundant_controls_if_small_and_periodic_lattice`.** Under the proposed solution, `self._mag_hamiltonian` is left unmodified for d=3. But `_strip_redundant_controls_if_small_and_periodic_lattice` (`circuit.py:711-742`) calls `_plaquette_state_has_inconsistent_controls` and `_discard_duplicate_controls_from_plaquette_state` on the *unfiltered* Hamiltonian's plaquette state bitstrings. For d=3, these bitstrings would still contain 16 controls (not trimmed), yet the consistency/discard methods would need to know the plane. The proposed solution says to add a new `_resolve_and_trim_for_small_periodic` method called from `apply_magnetic_trotter_step`, but it doesn't address `_strip_redundant_controls_if_small_and_periodic_lattice`, which is called *before* the plaquette loop. This is a gap — the `physical_states_for_control_pruning` set would be built from untrimmed bitstrings, but the Givens rotation circuits would use trimmed bitstrings, causing a mismatch.

**2c.3 — The plan correctly identifies this gap in Section 2g** (lines 295-305). However, Section 2g is labeled an "outstanding consideration" rather than a required implementation step, and the implementation order in Section 5 lists it as step 6 without noting that it's a hard dependency for step 7 (small-periodic filtering) and step 8 (qubit stitching).

**2c.4 — Trimming produces different-length bitstrings per plane.** The plan acknowledges this (line 117: "the resulting bitstring length and content differ by plane"). For d=3, the sharing analysis shows 12 unique controls per plane (line 174). But the plan doesn't address whether the 12 unique controls are the *same* 12 for all three planes, or different sets of 12. In fact, they are *different* sets — each plane has its own in-plane sharing pattern. This means `physical_states_for_control_pruning` cannot be a single `Set[str]` for d=3 small periodic lattices; it must be per-plane. The proposed solution in Section 2g acknowledges this but the main solution in 2c does not integrate it.

**2c.5 — Under the alternative solution, per-plane partial discarding is required.** The current filtering loop (`circuit.py:95-118`) iterates over `(bitstring_pair, MatrixElementValue)` and keeps or discards the *entire entry*. For d=3, a single `(bs1, bs2)` pair might be consistent for plane (1,2) but inconsistent for plane (1,3). Under the proposed solution (leave `self._mag_hamiltonian` unmodified), this is fine since filtering happens later per-plane. Under the alternative solution (restructure in `__init__`), the implementation must be careful not to discard a Hamiltonian entry entirely just because it's inconsistent for *one* plane — it should only discard the specific plane's contribution. The current d=2 code can afford to be coarse (discard the whole entry) because there's only one plane.

---

## Section 2d: Qubit Stitching Skip Logic

### Verified

- The skip logic is at `circuit.py:609-629`.
- The d=2 case correctly skips controls at v2 (dir +e1), v3 (dir +e2), and all of v4.
- The plan's skip pattern for d=3 (v2: skip +e1, v3: skip +e2, v4: skip +e2 and -e1) is consistent with its own sharing analysis. Verified by tracing through the sharing pairs.

No issues found.

---

## Section 2e: `_plaquette_state_has_inconsistent_controls`

### Issues Found

**2e.1 — The plan proposes adding a `plane` parameter with default None.** For d=3, the plane is required. But the callers of this method — specifically the `__init__` filtering loop (`circuit.py:101`) and `_strip_redundant_controls_if_small_and_periodic_lattice` (`circuit.py:727`) — don't currently pass a plane. The plan acknowledges this for `_strip_redundant_controls_if_small_and_periodic_lattice` in Section 2g, but doesn't address the `__init__` filtering loop. Under the proposed solution, `__init__` filtering is skipped for d=3, so this is fine. Under the alternative solution, both callers need updating.

---

## Section 2f: `_discard_duplicate_controls_from_plaquette_state`

### Verified

- The plan's proposed code for v4 trimming correctly keeps {e3, -e3} after dropping +e2 (index 0) and -e1 (index 3) from FORDER-sorted ctrl dirs (2, 3, -3, -1).
- The d=2 trimming counts (2+1+1+0 = 4 unique vs 8 slots) match the actual code at `circuit.py:948-960`.

No issues found.

---

## Section 3: Test Updates

### Verified

- The test at `tests/test_conventions.py:45-51` has the `case _:` fallthrough that needs a `case "d=3":` addition.
- The expected `expected_num_c_links = 16` for d=3 is correct (4 vertices * 4 controls/vertex).

### Issues Found

**3.1 — The test file is much larger than the plan implies.** The plan references "line 50-51" but the test file is ~1700 lines. The plan only proposes adding the d=3 case to one test (`test_physical_plaquette_state_data_are_valid`), but there are other tests that iterate over all dimension/truncation combos (e.g., `test_no_duplicate_physical_plaquette_states`, `test_no_duplicate_matrix_elements`) that will automatically pick up d=3 data once the file path dicts are updated. The plan should mention this as a benefit (and verify these existing tests pass with B3 data).

**3.2 — No integration test is concretely specified.** The plan's "New test coverage to consider" section lists desirable tests but doesn't provide concrete test implementations or even test function names. For a feature of this complexity, the audit recommends adding a new test to `test_integration_mps.py` which follows the pattern present in already extant tests, but uses the new data and d=3. This should be specified in the plan with expected behavior.

---

## Section 4: Files That Do NOT Need Changes

### Issues Found

**4.1 — `lattice_data.py:_normalize_link_address` has a latent boundary-checking bug.** Line 404: `any(component > self.shape[0] for component in lattice_vector)` compares ALL components against `shape[0]`. For a cubic d=3 lattice this is correct (all axes have the same size), but this is a fragile pattern — if the lattice becomes non-cubic, this silently produces wrong boundary normalization. The plan states "No changes needed" for `lattice_data.py`, which is technically true for the current d=3 cubic scope, but the plan should flag this as tech debt.

**4.2 — `lattice_data.py:add_unit_vector_to_vertex_vector` at line 525 also uses `self.shape[0]` for modular wrapping of all components.** Same concern as 4.1 — correct for cubic, fragile for non-cubic.

**4.3 — `measurement_results.py` `NotImplementedError` at line 40.** The plan says this is "for non-hypercubic tuple-valued shapes (not relevant for d=3 cubic)." The error says "Converting state bit strings for lattices with tuple-valued shape not yet supported." A d=3 cubic lattice created via `LatticeDef(3, 4)` has `shape = (4, 4, 4)` — a tuple. If `MeasurementResults` checks `isinstance(shape, tuple)`, it might trigger this error. The plan's blanket "no changes needed" may be premature; can be handled during implementation if it comes up.

---

## Section 5: Implementation Order

### Issues Found

**5.1 — Step 6 (`_strip_redundant_controls_if_small_and_periodic_lattice`) is listed as Section "2g" but was only introduced as an "outstanding consideration" in Section 6.** This creates confusion — the implementation order references a section that isn't part of the main numbered design. It also means the implementation order was likely written after the main plan, and the main plan's section numbering was not updated to include 2g.

**5.2 — Missing dependency: Step 7 (small-periodic filtering) depends on Step 6 (strip redundant controls) being plane-aware.** The plan lists these as sequential steps but doesn't call out the hard dependency.

---

## Section 6: Outstanding Considerations

### Issues Found

**6.1 — The per-plane `physical_states_for_control_pruning` approach has downstream implications the plan doesn't trace.** If `_strip_redundant_controls_if_small_and_periodic_lattice` returns `Dict[Plane, Set[str]]` instead of `Set[str]`, then every downstream consumer must change:
  - `_build_mag_evol_circuit` (`circuit.py:774`) takes `physical_states_for_control_pruning: Union[None | Set[str]]`
  - `givens` and `givens_fused_controls` in `givens.py` take the same type
  - The cache invalidation check at `circuit.py:502-504` compares `physical_states_for_control_pruning` for equality

This is a significant interface change that the plan underestimates.

**6.2 — The plan's Hamiltonian restructure recommendation (Section 6, final subsection) is sound but creates a migration risk.** Pivoting `self._mag_hamiltonian` from `Dict[Tuple[str, str], MatrixElementValue]` to a per-plane-first structure in-memory changes the data contract used by `_resolve_hamiltonian_for_plaquette`, the `__init__` filtering, and the `__repr__` method. The plan recommends this but doesn't note that `__repr__` at `circuit.py:122` prints `self._mag_hamiltonian` directly, which would produce very different output.

---

## Summary of Severity

| Severity | Count | Key Items |
|---|---|---|
| Critical (correctness risk) | 2 | 2c.2 (physical_states mismatch), 2c.4 (per-plane bitstrings) |
| High (architectural ambiguity) | 3 | 2c.1 (no clear approach chosen), 2c.5 (per-plane partial discarding), 6.1 (downstream interface changes) |
| Medium (implementation gaps) | 5 | 2e.1, 3.2, 4.1, 4.2, 5.2 |
| Low (documentation/clarity) | 5 | 2b.1, 3.1, 4.3, 5.1, 6.2 |

The plan is thorough in its analysis of the sharing pattern geometry and correctly identifies the key architectural challenge (per-plane filtering). Its main weaknesses are: (1) presenting two divergent solutions without committing to one, (2) underestimating the ripple effects of the per-plane `physical_states_for_control_pruning` change, and (3) deferring critical design decisions to an appendix-like "outstanding considerations" section.
