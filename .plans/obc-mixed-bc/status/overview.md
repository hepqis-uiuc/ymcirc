# OBC/Mixed BC Implementation — Status Overview

**Status: Phases 1 and 2 substantially complete. Step 1.4a (pyclebsch KeyError) is FIXED. Ready for Phase 3.**

Phase 2 (geometry layer) is fully implemented — all 7 steps done, all 208 existing tests pass with zero regressions. Phase 1 steps 1.1–1.4a are complete: `gen_ymcirc_data.py` is updated with universal lattice cases, T-truncation universal files (T1 d=3/2, T2 d=3/2, T1 d=2) plus B3 and B5 d=3/2 have been generated and validated in `pyclebsch/out/`, and the pyclebsch KeyError blocking B6+ d=3/2 generation has been fixed (two missing guards in `glue_plaquette_site_factors()`). Step **1.4b** (generate remaining B-truncation files: B4, B6–B10 d=3/2, B3 d=2, B3 d=3) is now unblocked. Step 1.5 (installation) is deferred to Phase 3.
