# d=3 PBC post-implementation test issues

There are some minor issues with new tests that were implemented in phase 8 of the d=3 PBC implementation plan:

1. The "d=3 integration tests" which were added to `test_integration_mps.py` actually just exercise `LatticeCircuitManager`. Move them to `test_circuit.py`.
2. We need a bona-fide integration test for d=3. We could do this by creating a new test which follows the logic in `test_mps_time_evolution_observables_change` for the case of a d=3, B3, size 2 lattice. In order to make this new test run more rapidly, DO NOT add an ancilla register to it (this should be the only deviation from the d=3/2 test case).
