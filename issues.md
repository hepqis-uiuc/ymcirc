# Outstanding issues from ymcirc F-order refactor

## Context
This is a review of the post-implementation results of the refactor described in forder-refactor-plan.md. The issues documented here largely pertain to additional requirements which were not specifically addressed, or edge cases which were not handled or tested for.

If additional information on specific feedback is required, first consult forder-refactor-plan.md (keep in mind that line numbers have changed since the refactor was implemented). Thenconsult the ymcirc codebase itself as appropriate.

## ymcirc/conventions.py

1. The `_flatten_hamiltonian_value` method shouldn't enforce that all float values are equal because in the future, there will be nonperiodic/higher-dimensional lattices where there are differences in the float values across planes/signatures.
2. The `trunc_string` constructed in `_load_plaquette_states` and `_load_hamiltonian` inappropriately hard-codes truncation information. `trunc_string` should be extracted from `metadata`.
3. There was a d=2 diagram of a plaquette state in the module-level doc string which was removed a few commits ago (the abbreviated hash for the commit is 5351001). This diagram should be put back, and updated so that the control links are orderd according to the default F-order.
4. It is redundant for `LatticeStateEncoder` to take an explicit `forder` argument because that information is already a property on the `lattice` argument. The `forder` argument to the `LatticeStateEncoder` constructor should therefore be removed. Docstrings should be updated accordingly, and any broken tests fixed.
5. The method `LatticeStateEncoder.decode_bit_string_to_plaquette_state` should check whether `self._lattice` is periodic or not, and raise a `NotImplementedError` if it isn't `True` since we haven't yet implemented decoding logic for nonperiodic lattices.
6. It would be nice to add a boolean `refresh` flag to the `get_data_metadata` method in case the user wants to re-load data for whatever reason.

## ymcirc/circuit.py
1. The `_plaquette_state_has_inconsistent_controls`, `_discard_duplicate_controls_from_plaquette_state`, and `apply_magnetic_trotter_step` methods of `LatticeCircuitManager` have a significant flaw: they all implicitly hard-code dependence on the default F-order in the indexing of control links. These methods should take into account whatever F-order is specified by `self._encoder.forder` when identifying the indices of redundant control links on small and periodic lattices.

## ymcirc/parsed_lattice_result.py
1. `ParsedLatticeResult` doesn't have an `forder` property on it. Since it's a subclass implementing `LatticeDef`, this property should be set in the call to the superclass's init using `lattice_encoder.forder` (kind of like how this is done in `LatticeRegisters`, except now we are using the `LatticeStateEncoder` to get the F-order information).

## run module
1. The `initialize_lattice_tools` method in `functions.py` fails to make use of the `forder` metadata present in the magnetic Hamiltonian matrix element data as well as in the physical plaquette states. This should be corrected so that F-order is correctly read from loaded metadata, and fed to downstream classes through the circuit generation, simulation, and data analysis pipeline of `time_evol.py`.

## tests
1. There should be a test which checks that trying to create an instance of a subclass implementing `LatticeDef` causes an appropriate exception to be raised when the `forder` creaetion argument is bad (i.e. not a list which is a permutation of the default F-order).
2. There should be multiple tests which check that actually changing F-order yields expected changes in the ordering of control links. For example:
   1. The `get_plaquettes` method on subclasses inmplementing `LatticeDef` (`ParsedLatticeResult` and `LatticeRegisters`, for instance) should be tested to ensure that changing F-order in d=2 changes the ordering of control links correctly.
   2. There should be a test (possibly an integration test rather than a unit test) which confirms that setting a non-standard F-order doesn't break the logic for removing redundant controls on small lattices (which is implemented across several places in `LatticeCircuitManager`). This should be checked for both d=3/2 and d=2.
   3. For both d=3/2 and d=2, and for both standard and non-standard choices of F-order, it should be checked that `LatticeStateEncoder` successfully encodes/decodes plaquette states.
