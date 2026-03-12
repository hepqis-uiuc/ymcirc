# Adversarial Audit of d=3 Feature Plan

This audit cross-references every claim in `d3-feature-planning.md` against the actual codebase. Findings are organized by plan section.

---

## Section 1: Data File Registration in `conventions.py`

### Verified

- The B3 d=3 data files exist at the stated paths. Confirmed via filesystem glob.
- The file path dicts are at `conventions.py:167-183` and the proposed additions are structurally correct.
- The `PHYSICAL_PLAQUETTE_STATES` and `HAMILTONIAN_BOX_TERMS` LazyDicts are built via comprehension over the file path dicts (`conventions.py:304-317`), so adding entries to the path dicts will indeed propagate automatically.
- The claim that `LatticeStateEncoder` needs no changes is correct — it accepts arbitrary link bitmaps and lattice definitions.

### Issues Found

**1.1 — Line number drift.** The plan references "lines 167-183" for the file path dicts. In the current code, `_HAMILTONIAN_DATA_FILE_PATHS` starts at line 167 and `_PLAQUETTE_STATES_DATA_FILE_PATHS` starts at line 175. These are close enough but worth confirming at implementation time; the plan does not account for edits that may have shifted lines since the plan was drafted. [Comment: Line number drift will always happen during implementation. Therefore this is not an issue worth mentioning.]

**1.2 — Missing B3 entry in `IRREP_TRUNCATIONS` bitmap width validation.** The plan proposes adding `"B3"` to `IRREP_TRUNCATIONS` with the same bitmap as T1 (2-bit strings). This is fine, but the plan doesn't mention that `IRREP_TRUNCATIONS` is **not keyed by dimension** — it is a flat dict (`conventions.py:221-235`). If a future B truncation for a *different* dimension (say d=4) also needs a `"B3"` entry with different irreps, the naming convention will collide. The plan should note this namespace design decision explicitly, or propose a naming scheme like `"B3_d3"`. [Comment: This is not an issue. Irrep truncations will never depend on lattice geometry.]

**1.3 — Missing mention of `_load_hamiltonian` key conversion.** The plan's Section 1 says "no special handling is needed" for string-stored data. However, the `_load_hamiltonian` function (`conventions.py:251-274`) already performs `ast.literal_eval` on plane and signature keys during loading. The plan's statement is technically correct, but it would be prudent to note that this conversion logic in `_load_hamiltonian` is what makes it work — if the B3 data files use a different nested-dict format or key encoding, this would silently produce wrong results. The plan should recommend validating the loaded B3 Hamiltonian structure (planes as tuples, signatures as tuples) in a test. [Comment: This is not an issue. All future data files will use the same key encoding and nested-dict structure.]

---

## Section 2a: `__init__` Dimension Match

### Verified

- The `case _:` at `circuit.py:84-85` is the correct location.
- The proposed `case 3:` logic is structurally correct.

### Issues Found

**2a.1 — Hardcoded `shape[0]` assumption is already present and propagates.** The plan proposes `lattice_size = lattice_encoder.lattice_def.shape[0]` and then checking `shape[1]` and `shape[2]`. This is fine for the cubic check, but worth noting that `LatticeDef._configure_lattice` (`lattice_data.py:363-364`) already enforces cubic shape when given an int size: `self._shape = (size,) * int(dim)`. So the non-cubic check in the plan is belt-and-suspenders — it would only fire if someone bypasses `LatticeDef.__init__` validation (which currently raises `NotImplementedError` for tuple sizes at `lattice_data.py:339`). Not a bug, but the plan should note this redundancy. [Comment: This isn't a big deal.]

---

## Section 2b: Cache Control Link Dirs

### Verified

- `self._cached_ctrl_dirs_d2_small_and_periodic` is created at `circuit.py:89-93` for d=2 using a temporary plaquette at `(0,0)` in plane `(1,2)`.
- The plan correctly identifies that d=3 needs per-plane caching.

### Issues Found

**2b.1 — Incorrect control count in plan's data table.** The plan states "Controls per vertex | 4 (= 2*(d-1)) -> 16 total per plaquette" (line 31). Let's verify: for d=3, `n_control_links_per_plaquette = 4 * 2*(3-1) = 4 * 4 = 16`. This is correct. [Comment: This isn't an issue! Remove it from the audit.]

**2b.2 — The plan does not specify which vertex to use for the temporary plaquette.** For d=3, the plan says to construct a "temporary plaquette for that plane" but doesn't specify the bottom-left vertex. Since the lattice is periodic and cubic, vertex `(0,0,0)` is the natural choice for all planes. This is a minor omission but should be explicit since different vertices could yield different `control_link_dirs_per_vertex` on non-periodic lattices in the future.

---

## Section 2c: Small-Periodic Hamiltonian Filtering (Main Challenge)

### Issues Found

**2c.1 — CRITICAL: The plan's "Proposed Solution" conflicts with its own sharing analysis.** The proposed solution (lines 121-133) says: "In `__init__`, do **not** perform the filtering for d=3." But then the alternative solution (lines 137-155) says to restructure filtering to be per-plane for all dimensions. These are presented as alternatives, yet neither is endorsed as the final choice. The plan leaves the implementer to choose between two architecturally divergent paths without a clear recommendation. Section 6's "Which approach best supports future lattices" (`conventions.py:315-319`) does recommend the alternative — but this recommendation is buried in an appendix-like section rather than stated in the main design.

**2c.2 — The proposed solution introduces a correctness risk for `_strip_redundant_controls_if_small_and_periodic_lattice`.** Under the proposed solution, `self._mag_hamiltonian` is left unmodified for d=3. But `_strip_redundant_controls_if_small_and_periodic_lattice` (`circuit.py:711-742`) calls `_plaquette_state_has_inconsistent_controls` and `_discard_duplicate_controls_from_plaquette_state` on the *unfiltered* Hamiltonian's plaquette state bitstrings. For d=3, these bitstrings would still contain 16 controls (not trimmed), yet the consistency/discard methods would need to know the plane. The proposed solution says to add a new `_resolve_and_trim_for_small_periodic` method called from `apply_magnetic_trotter_step`, but it doesn't address `_strip_redundant_controls_if_small_and_periodic_lattice`, which is called *before* the plaquette loop. This is a gap — the `physical_states_for_control_pruning` set would be built from untrimmed bitstrings, but the Givens rotation circuits would use trimmed bitstrings, causing a mismatch.

**2c.3 — The plan correctly identifies this gap in Section 2g** (lines 295-305). However, Section 2g is labeled an "outstanding consideration" rather than a required implementation step, and the implementation order in Section 5 lists it as step 6 without noting that it's a hard dependency for step 7 (small-periodic filtering) and step 8 (qubit stitching).

**2c.4 — Trimming produces different-length bitstrings per plane.** The plan acknowledges this (line 117: "the resulting bitstring length and content differ by plane"). For d=3, the sharing analysis shows 12 unique controls per plane (line 174). But the plan doesn't address whether the 12 unique controls are the *same* 12 for all three planes, or different sets of 12. In fact, they are *different* sets — each plane has its own in-plane sharing pattern. This means `physical_states_for_control_pruning` cannot be a single `Set[str]` for d=3 small periodic lattices; it must be per-plane. The proposed solution in Section 2g acknowledges this but the main solution in 2c does not integrate it.

**2c.5 — The plan does not account for a Hamiltonian entry that has matrix element values across multiple planes simultaneously.** The current filtering loop (`circuit.py:95-118`) iterates over `(bitstring_pair, MatrixElementValue)` and keeps or discards the *entire entry*. For d=3, a single `(bs1, bs2)` pair might be consistent for plane (1,2) but inconsistent for plane (1,3). Under the proposed solution (leave `self._mag_hamiltonian` unmodified), this is fine since filtering happens later per-plane. Under the alternative solution (restructure in `__init__`), the plan correctly says to iterate per-plane. But the plan doesn't note that the alternative solution must be careful not to discard a Hamiltonian entry entirely just because it's inconsistent for *one* plane — it should only discard the specific plane's contribution. The current d=2 code can afford to be coarse (discard the whole entry) because there's only one plane. [Comment: This is a good observation. Make sure this subitem is somewhere in the summary table at the end of the audit doc.]

---

## Section 2d: Qubit Stitching Skip Logic

### Verified

- The skip logic is at `circuit.py:609-629`.
- The d=2 case correctly skips controls at v2 (dir +e1), v3 (dir +e2), and all of v4.

### Issues Found

**2d.1 — The plan's skip pattern for d=3 is inconsistent with the sharing analysis.** The plan's Section 2d (lines 183-187) lists:
- v2: skip +e1
- v3: skip +e2
- v4: skip +e2 and -e1

But the sharing analysis in the plan's own table (lines 168-172) says:
- v2: drop +e1 (dups v1[-e1]), keep 3 -> {-e2, e3, -e3} (FORDER indices of e3, -e3, -e2)
- v3: drop +e2 (dups v2[-e2]), keep 3 -> {+e1, e3, -e3}
- v4: drop +e2 (dups v1[-e2]) and -e1 (dups v3[+e1]), keep 2 -> {e3, -e3}

The plan states v3 drops +e2, but the sharing table says "v3: drop +e2 (dups v2[-e2])". Let's verify: v2 has control dirs {+e1, e3, -e3, -e2} and v3 has control dirs {+e1, +e2, e3, -e3}. The sharing pair is v2[-e2] <-> v3[+e2], meaning v3's +e2 is a duplicate of v2's -e2. But wait — v2's -e2 is itself kept (it's not dropped). So v3's +e2 duplicates a *kept* entry from v2, which is correct. This checks out.

For v4: the sharing pairs are v1[-e2] <-> v4[+e2] and v3[+e1] <-> v4[-e1]. Both v1[-e2] and v3[+e1] are kept, so v4's +e2 and -e1 are correctly identified as duplicates. This also checks out. The skip pattern is consistent with the analysis. My initial concern was unfounded. [Comment: Since this turned out to be a non-issue, remove it from the audit doc.]

**2d.2 — The skip logic must be plane-aware at runtime.** The plan says "the skip indices need to be computed per-plaquette or looked up per-plane from the cache" (line 191). However, the current code structure pre-computes skip indices *before* the plaquette loop (`circuit.py:516-520`). For d=3, these indices would need to be computed inside the plaquette loop (after `plaquette_plane` is known). This means the `match` statement at `circuit.py:614` would need access to the current plaquette's plane and the per-plane cached ctrl dirs. The plan notes this but doesn't provide the implementation detail of how `plane` gets threaded into the skip logic — it's currently inside a `for vertex_idx, vertex_controls in enumerate(plaquette.control_links_per_vertex)` loop that doesn't reference the plane.

Actually, `plaquette_plane` is available at `circuit.py:551` (`plaquette_plane: Plane = plaquette.plane`), so this is accessible. The plan's gap is just that it doesn't explicitly state that the skip indices should be computed from `plaquette_plane` and the per-plane cache inside the plaquette loop body. [Comment: Clarify this item or remove it from the audit doc if it isn't actually a problem.]

---

## Section 2e: `_plaquette_state_has_inconsistent_controls`

### Issues Found

**2e.1 — The proposed code uses `ctrl_dirs[0].index(-e1)` but doesn't verify that -e1 is actually present.** For d=3, `ctrl_dirs[0]` (v1's control dirs) should be `{e3, -e3, -e1, -e2}` in FORDER order. The lookup `ctrl_dirs[0].index(-e1)` will work. But for non-periodic or boundary plaquettes in the future, a control direction might be absent, causing a `ValueError` from `.index()`. The plan should note this assumption (that all expected directions exist). [Comment: This issue is out-of-scope for the current feature.]

**2e.2 — The plan proposes adding a `plane` parameter with default None.** For d=3, the plane is required. But the callers of this method — specifically the `__init__` filtering loop (`circuit.py:101`) and `_strip_redundant_controls_if_small_and_periodic_lattice` (`circuit.py:727`) — don't currently pass a plane. The plan acknowledges this for `_strip_redundant_controls_if_small_and_periodic_lattice` in Section 2g, but doesn't address the `__init__` filtering loop. Under the proposed solution, `__init__` filtering is skipped for d=3, so this is fine. Under the alternative solution, both callers need updating.

---

## Section 2f: `_discard_duplicate_controls_from_plaquette_state`

### Issues Found

**2f.1 — The plan's proposed code for v4 trimming keeps {e3, -e3}.** Let me verify: v4's control dirs are {+e2, e3, -e3, -e1} (in FORDER order). Dropping +e2 and -e1 leaves {e3, -e3}. With default FORDER [1, 2, 3, -1, -2, -3], the indices of e3 and -e3 in v4's ctrl dirs depend on the actual FORDER-sorted order. Since the ctrl dirs are sorted by FORDER position, for default FORDER v4's ctrl dirs would be ordered as (+e2, e3, -e3, -e1) = (2, 3, -3, -1). Dropping indices for +e2 (index 0) and -e1 (index 3) leaves indices 1 and 2, which are e3 and -e3. This is correct. [Comment: If this is correct, then remove the issue from the audit doc.]

**2f.2 — The trimming counts don't match between d=2 and d=3 for the "total after trimming" formula.** The plan states for d=2: "2 + 1 + 1 + 0 = 4 unique (vs 8 slots)." Let me verify against the actual d=2 code: `_discard_duplicate_controls_from_plaquette_state` for d=2 (`circuit.py:948-960`) keeps v1's 2 controls, v2's 1 (dir -e2 only), v3's 1 (dir +e1 only), v4's 0. That's 2+1+1+0 = 4. Correct. [Comment: If this is correct, then remove the issue from the audit doc.]

---

## Section 3: Test Updates

### Verified

- The test at `tests/test_conventions.py:45-51` has the `case _:` fallthrough that needs a `case "d=3":` addition.
- The expected `expected_num_c_links = 16` for d=3 is correct (4 vertices * 4 controls/vertex).

### Issues Found

**3.1 — The test file is much larger than the plan implies.** The plan references "line 50-51" but the test file is ~1700 lines. The plan only proposes adding the d=3 case to one test (`test_physical_plaquette_state_data_are_valid`), but there are other tests that iterate over all dimension/truncation combos (e.g., `test_no_duplicate_physical_plaquette_states`, `test_no_duplicate_matrix_elements`) that will automatically pick up d=3 data once the file path dicts are updated. The plan should mention this as a benefit (and verify these existing tests pass with B3 data).

**3.2 — No integration test is concretely specified.** The plan's "New test coverage to consider" section lists desirable tests but doesn't provide concrete test implementations or even test function names. For a feature of this complexity, the audit recommends that at least one concrete end-to-end test (constructing a `LatticeCircuitManager` for d=3 and calling `apply_magnetic_trotter_step`) be specified in the plan with expected behavior. [Comment: To make this concrete, we can propose adding a new test to `test_integration_mps.py` which follows the pattern present in already extant tests, but uses the new data and d=3.]

---

## Section 4: Files That Do NOT Need Changes

### Issues Found

**4.1 — `lattice_data.py:_normalize_link_address` has a latent boundary-checking bug that affects d=3.** Line 404: `any(component > self.shape[0] for component in lattice_vector)` compares ALL components against `shape[0]`. For a cubic d=3 lattice this is correct (all axes have the same size), but this is a fragile pattern — if the lattice becomes non-cubic, this silently produces wrong boundary normalization. The plan states "No changes needed" for `lattice_data.py`, which is technically true for the current d=3 cubic scope, but the plan should flag this as tech debt. [Comment: This is an important bug you have identified. Update the severity of it in the table at the end of the audit doc.]

**4.2 — `lattice_data.py:add_unit_vector_to_vertex_vector` at line 525 also uses `self.shape[0]` for modular wrapping of all components.** Same concern as 4.1 — correct for cubic, fragile for non-cubic. [Comment: This is an important bug you have identified. Update the severity of it in the table at the end of the audit doc.]

**4.3 — `circuit.py:539` hardcodes the "only one positive plaquette" check.** The line `has_only_one_positive_plaquette = lattice.dim == 1.5 or lattice.dim == 2` excludes d=3 from the single-plaquette path, which means d=3 will take the `else` branch at line 545: `plaquettes = lattice.get_plaquettes(vertex_address)`. This calls into `get_plaquettes` without specifying e1/e2, which for d>2 returns a list of plaquettes. **This is correct and the plan doesn't need to address it, but the plan also doesn't mention it.** Since this is a change in control flow (d=3 returns a list, d=2 returns a single plaquette wrapped in a list at line 541-543), it's worth noting. [Comment: Remove this item from the audit doc. It's not a real issue.]

**4.4 — `measurement_results.py` `NotImplementedError` at line 40.** The plan says this is "for non-hypercubic tuple-valued shapes (not relevant for d=3 cubic)." Let me verify: the error says "Converting state bit strings for lattices with tuple-valued shape not yet supported." A d=3 cubic lattice created via `LatticeDef(3, 4)` has `shape = (4, 4, 4)` — a tuple. The question is whether any code path passes the *tuple* (as opposed to the int size) to `MeasurementResults`. If `MeasurementResults` gets the `LatticeDef` and checks `isinstance(shape, tuple)`, it might trigger this error. **This needs verification.** The plan's blanket "no changes needed" may be premature. [Comment: Downgrade this to low severity. If it comes up as a problem during implementation, it can be handled at that time.]

**4.5 — `electric_helper.py` NotImplementedError at line 87.** The plan says this is "for non-hypercubic lattices." The actual error message says "Electric energy computation for non-hypercubic lattices with dim >= 2 not yet implemented." A d=3 cubic lattice IS hypercubic, so this is fine if the conditional correctly identifies it as hypercubic. This should be verified. [Comment: Not a real issue. Remove from audit doc.]

---

## Section 5: Implementation Order

### Issues Found

**5.1 — Step 6 (`_strip_redundant_controls_if_small_and_periodic_lattice`) is listed as Section "2g" but was only introduced as an "outstanding consideration" in Section 6.** This creates confusion — the implementation order references a section that isn't part of the main numbered design. It also means the implementation order was likely written after the main plan, and the main plan's section numbering was not updated to include 2g.

**5.2 — Missing dependency: Step 7 (small-periodic filtering) depends on Step 6 (strip redundant controls) being plane-aware.** The plan lists these as sequential steps but doesn't call out the hard dependency.

**5.3 — No step for verifying data file integrity.** Before any code changes, the B3 data files should be validated (correct metadata, correct nested dict structure, all plane/signature keys are proper tuples after loading). This should be Step 0. [Comment: Not necessary, I've already verified the data files. Remove this item from the audit doc.]

---

## Section 6: Outstanding Considerations

### Issues Found

**6.1 — The per-plane `physical_states_for_control_pruning` approach has downstream implications the plan doesn't trace.** If `_strip_redundant_controls_if_small_and_periodic_lattice` returns `Dict[Plane, Set[str]]` instead of `Set[str]`, then every downstream consumer must change:
  - `_build_mag_evol_circuit` (`circuit.py:774`) takes `physical_states_for_control_pruning: Union[None | Set[str]]`
  - `givens` and `givens_fused_controls` in `givens.py` take the same type
  - The cache invalidation check at `circuit.py:502-504` compares `physical_states_for_control_pruning` for equality

This is a significant interface change that the plan underestimates.

**6.2 — Performance claim needs qualification.** The plan says "Performance should be manageable" for 54,035 plaquette states and 707 Hamiltonian entries. But the `__init__` filtering loop decodes and re-encodes every Hamiltonian entry. With the alternative solution's per-plane approach, this becomes `707 * 3 = 2,121` decode/encode cycles. The encoding involves string concatenation of ~48+ characters per plaquette state (16 controls * 2 bits + 4 active links * 2 bits + vertex bits). This is still fast but the plan should note it's 3x the work of d=2. [Comment: 3x overhead isn't a big deal. Remove this item from the audit doc.]

**6.3 — The plan's Hamiltonian restructure recommendation (Section 6, final subsection) is sound but creates a migration risk.** Pivoting `self._mag_hamiltonian` from `Dict[Tuple[str, str], MatrixElementValue]` to a per-plane-first structure in-memory changes the data contract used by `_resolve_hamiltonian_for_plaquette`, the `__init__` filtering, and the `__repr__` method. The plan recommends this but doesn't note that `__repr__` at `circuit.py:122` prints `self._mag_hamiltonian` directly, which would produce very different output.

---

## Cross-Cutting Concerns Not Addressed by the Plan

**CC.1 — No mention of `decode_bit_string_to_plaquette_state` for d=3.** This method (`conventions.py:739-812`) uses `n_controls_per_vertex = int(2 * (self._lattice.dim - 1))` at line 800. For d=3, this gives `int(2 * 2) = 4`, which is correct. The method should work for d=3 without changes. Verified. [Comment: Remove from audit doc since not an issue.]

**CC.2 — `get_plaquettes` for d>2 uses `product(range(1, ceil(self.dim + 1)), ...)` which for d=3 gives `range(1, 4) = [1, 2, 3]`, producing planes `(1,2), (1,3), (2,3)`. This is correct.** But the use of `ceil(self.dim + 1)` would break for non-integer dimensions > 2 (hypothetical d=5/2). Not relevant for d=3 but a latent issue. [Comment: Remove from audit doc since not an issue (there will only be integer dimensions other than d-3/2).]

**CC.3 — No backward compatibility or feature-flagging strategy.** The plan doesn't discuss how to land d=3 support incrementally. For example, should there be a way to disable d=3 while it's partially implemented? Given that the changes touch `__init__` filtering (which runs for all dimensions), a bug in the d=3 path could break existing d=2 functionality. The plan should recommend feature-flagging or at minimum a "d=3 not yet supported" guard that's removed last. [Comment: This is alpha-stage research software. We don't need to worry about backward compatibility or feature-flagging. Remove item from audit doc.]

**CC.4 — The plan doesn't address `apply_electric_trotter_step`.** The electric Trotter step (`circuit.py:335-419`) iterates over all link addresses and applies the electric Hamiltonian. This is dimension-agnostic and should work for d=3 without changes. However, the plan doesn't explicitly verify this, and the `electric_helper.py` file may need verification (see 4.5 above). [Comment: Probably not an issue, and if it does come up, can be handled during implementation. Remove item from audit doc.]

**CC.5 — No discussion of circuit width / qubit count scaling.** A d=3, size-2, B3 lattice has 8 vertices * 3 links/vertex = 24 links, each needing 2 qubits (for T1/B3), plus vertex qubits (0 for trivial multiplicity). That's 48 link qubits minimum. Each plaquette also needs controls stitched in. The plan's performance section only discusses Hamiltonian entry counts, not circuit width. For practical usability, the qubit count should be noted. [Comment: Detailed performance analysis not required. Remove item from audit doc.]

---

## Summary of Severity

| Severity | Count | Key Items |
|---|---|---|
| Critical (correctness risk) | 2 | 2c.2 (physical_states mismatch), 2c.4 (per-plane bitstrings) |
| High (architectural ambiguity) | 2 | 2c.1 (no clear approach chosen), 6.1 (downstream interface changes) |
| Medium (implementation gaps) | 5 | 2d.2, 2e.2, 3.2, 4.4, 5.2 |
| Low (documentation/clarity) | 7 | 1.1, 1.2, 1.3, 2a.1, 2b.2, 4.1-4.2, CC.3 |

The plan is thorough in its analysis of the sharing pattern geometry and correctly identifies the key architectural challenge (per-plane filtering). Its main weaknesses are: (1) presenting two divergent solutions without committing to one, (2) underestimating the ripple effects of the per-plane `physical_states_for_control_pruning` change, and (3) deferring critical design decisions to an appendix-like "outstanding considerations" section.
