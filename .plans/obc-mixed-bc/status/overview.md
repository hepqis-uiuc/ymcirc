# OBC/Mixed BC Implementation — Status Overview

**Status: Phases 1 and 2 substantially complete. Ready for Phase 3.**

Phase 2 (geometry layer) is fully implemented — all 7 steps done, all 208 existing tests pass with zero regressions. Phase 1 steps 1.1–1.3 are complete: `gen_ymcirc_data.py` is updated with universal lattice cases, and T-truncation universal files (T1 d=3/2, T2 d=3/2, T1 d=2) plus B3 and B5 d=3/2 have been generated and validated in `pyclebsch/out/`. Step 1.4 has been split: **1.4a** is to debug the pyclebsch `KeyError` in `glue_plaquette_site_factors()` that blocks B6–B10 d=3/2 generation; **1.4b** is to generate the remaining B-truncation files (B4–B10 d=3/2 after the fix, plus B3 d=2 and B3 d=3). Step 1.5 (installation) is deferred to Phase 3 — installing universal files now breaks tests because `conventions.py` doesn't yet handle multi-signature data. The generated files remain in `pyclebsch/out/` ready for Phase 3 to install and wire up.
