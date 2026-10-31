# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ymcirc generates quantum circuits (via Qiskit 2.x) to simulate lattice SU(3) gauge theories. Alpha stage — breaking changes expected.

## Commands

### Package Management (uv, not pip)
```bash
uv add <package>             # Add dependency (updates pyproject.toml + uv.lock)
uv add --dev <package>       # Add dev dependency
uv add --upgrade <package>   # Update dependency
uv remove <package>          # Remove dependency
uv lock && uv sync           # Manual lock/sync if environment issues arise
```

### Running
```bash
uv run -m run.time_evol                    # Run a module from the run/ directory
uv run some_script.py                      # Run a standalone script
```

### Testing (pytest)
```bash
uv run pytest -v                           # Run all tests (verbose)
uv run pytest tests/test_circuit.py        # Run one test file
uv run pytest tests/test_mod.py::test_func # Run a single test
uv run pytest --runslow                    # Include slow tests (skipped by default)
```
Mark slow tests with `@pytest.mark.slow`. Test files must be named `test_[something].py` and go in `tests/`.

## Architecture

### Key Domain Concepts
- **i-weights**: Gelfand-Tsetlin pattern representation of SU(3) irreducible representations (irreps)
- **Plaquette states**: gauge-invariant configurations on 4 lattice vertices
- **Trotter steps**: time evolution decomposed into magnetic and electric slices
- **Givens rotations**: multi-qubit rotation gates used to build state-preparation circuits
- **Irrep truncations**: given as pyclebsch takes them, a truncation mode and a cutoff, e.g. `("T", 1)` or `("B", 3)`. T1 = {1, 3, 3̄}, T2 = {1, 3, 3̄, 6, 6̄, 8}; a B truncation with cutoff c keeps the singlets at a vertex whose links' quadratic Casimirs sum to at most c, in the normalization where the Casimir of the 3 is 4/3.
- **F-order**: A configurable half-link ordering convention that determines how control links at each plaquette vertex are sorted. Defined as a permutation of `(1, 2, 3, -1, -2, -3)` (the default), where integers represent lattice directions; `LatticeDef` accepts it as any sequence (a list or a tuple, but not a set) and keeps and returns it as a tuple. F-order is set on `LatticeDef`, passed to pyclebsch when it computes plaquette states and matrix elements, and threaded through `LatticeStateEncoder` and all downstream classes. It affects control link ordering within plaquettes, bit-string encoding/decoding of plaquette states, and the indices of duplicate controls (a link that is a control link at two vertices of a plaquette, on a periodic axis of length 2).
- **Signatures and planes**: Type aliases used in Hamiltonian matrix element data. A `Plane` is a `tuple[int, int]` of positive lattice directions, such as `(1, 2)`. A `Signature` is a 4-tuple of per-vertex direction tuples (`tuple[int, ...]`), containing ALL existing half-link directions at each vertex sorted by the lattice's F-order (matching the output of the `Plaquette.signature` property, which pyclebsch computes). Both are defined in `_abstract/lattice_data.py`. ymcirc annotates directions as `int`; pyclebsch's narrower `LinkDirection` (a `Literal` of ±1, ±2, ±3) is used only where a value is handed to pyclebsch, after it has been checked.
- **`PlaquetteState`** and **`PlaquetteMatrixElements`**: Type aliases defined in `_abstract/lattice_data.py` beside `Plane` and `Signature`, and exported by `ymcirc._abstract`. `PlaquetteMatrixElements` is the format for the magnetic Hamiltonian's matrix elements, as `box_plus_box_dagger_matrix_elements` (in `_pyclebsch_adapter.py`) returns them and `LatticeCircuitManager`'s constructor takes them: `dict[tuple[Plane, Signature], dict[tuple[PlaquetteState, PlaquetteState], float]]`, for each plane and signature a dict from pairs of plaquette states to the matrix element. This is how pyclebsch groups them. Callers pass plaquette states, never bit strings; the manager encodes them with its encoder. A third alias there, `DecodedPlaquetteState`, is what `decode_bit_string_to_plaquette_state` returns: the layout of `PlaquetteState`, with `None` for any multiplicity index or irrep whose bits decode to nothing, and for every multiplicity index when the encoder has no vertex bits.
- **Encoding plaquette states**: `LatticeStateEncoder.encode_plaquette_state_as_bit_string(state, plane, signature)` and `decode_bit_string_to_plaquette_state(bit_string, plane, signature)` take the plaquette's plane and signature, which fix the number of control links at each vertex and which controls repeat another (on a periodic axis of length 2, depending on the plane). A state lists every control; its bit string leaves out the duplicates, and decoding puts them back.

### Module Dependency Flow
```
_abstract/lattice_data.py    Base classes: LatticeData[T], Plaquette[T]
        ↓                      ↓
        ↓               _pyclebsch_adapter.py (plaquette states and matrix elements from pyclebsch)
        ↓                      ↓
lattice_registers.py    conventions.py (LatticeStateEncoder, irrep encodings)
        ↓                      ↓
        └──── circuit.py ──────┘   (LatticeCircuitManager — main entry point)
                  ↓
              givens.py            Givens rotation circuit construction
              electric_helper.py   Electric Hamiltonian Pauli decomposition
              parsed_lattice_result.py   Measurement bitstring → physics results
              measurement_results.py     Aggregated measurement analysis
```

### Core Classes

- **`LatticeDef`** (`_abstract/lattice_data.py`): Foundational class defining lattice geometry: dimensions, size, boundary conditions, and F-order. Parent of `LatticeData[T]`. F-order is set here and inherited by all downstream classes. It wraps (holds, rather than inherits from) a pyclebsch `LatticeDef` of the same lattice, from which it takes which sites, links and plaquettes exist, each plaquette's signature, and which control links of a plaquette repeat another (on a periodic axis of length 2). It translates between ymcirc's coordinates (2-tuples in d=3/2 and d=2) and pyclebsch's 3-tuples. `get_traversal_order()` alone orders registers and bit strings. Its `size` property gives the size argument that recreates the lattice.
- **`LatticeData[T]`** (`_abstract/lattice_data.py`): Generic abstract base inheriting from `LatticeDef`. Adds data of type `T` on each vertex and link (`get_vertex`, `get_link`), `get_plaquettes`, and iteration over the data in `LatticeDef`'s traversal order. Two concrete subclasses: `LatticeRegisters` and `ParsedLatticeResult`.
- **`Plaquette[T]`** (`_abstract/lattice_data.py`): Generic container for the 4 vertices, 4 active links, and control links of a hypercubic lattice plaquette. Control links are grouped per-vertex in F-order via `control_links_per_vertex` (returns tuple of tuples) or flattened via `control_links_ordered`. The `signature` property returns a 4-tuple of per-vertex direction-label tuples (all existing half-link directions sorted by the lattice's F-order), as pyclebsch computes it. The `control_link_dirs_per_vertex` property returns per-vertex control link directions sorted by the lattice's F-order. A vertex's control links are its half-links, as pyclebsch lists them, other than those of the plaquette's active links.
- **`LatticeStateEncoder`** (`conventions.py`): Manages bit-string encodings for link states, vertex states, and plaquette states. Handles irrep truncation levels. Inherits F-order from its `lattice` argument (a `LatticeDef`); there is no separate `forder` parameter on the encoder.
- **`LatticeRegisters`** (`lattice_registers.py`): Maps Qiskit `QuantumRegister`s to lattice links/vertices. Factory method: `from_lattice_state_encoder()`.
- **`LatticeCircuitManager`** (`circuit.py`): Builds simulation circuits. Key methods: `create_blank_full_lattice_circuit()`, `apply_magnetic_trotter_step()`, `apply_electric_trotter_step()`, `measure_link()`, `measure_vertex()`, `measure_plaquette()`. Each plaquette's circuit is built from the matrix elements under its plane and signature (the `Plaquette.signature` property); the constructor refuses a (plane, signature) pair that no plaquette of the lattice has, and states its encoder refuses. Control pruning takes the physical plaquette states of each plane and signature, as `physical_plaquette_states` returns them, and the manager encodes them.
- **`ParsedLatticeResult`** (`parsed_lattice_result.py`): Converts measurement bitstrings back into i-weights and multiplicity indices. Hashable (usable as dict key). Factory methods: `from_links_and_vertices()`, `from_partial_measurement()`. Energy computation: `get_link_electric_energy()`, `get_lattice_electric_energy()`.
- **`MeasurementResults`** (`measurement_results.py`): Aggregates shot counts keyed by `ParsedLatticeResult`. Provides expectation values for electric energy, vacuum persistence probability, and transition probabilities.

### Plaquette States and Matrix Elements from pyclebsch
Physical plaquette states and the matrix elements of box + box^dagger (the magnetic Hamiltonian of one plaquette, up to a prefactor) are computed in memory by pyclebsch, a dependency; `_pyclebsch_adapter.py` is the one module that computes them with pyclebsch and converts them to ymcirc's formats. A lattice and truncation are set up in two steps:
1. `LatticeStateEncoder.for_truncation(lattice, truncation_mode, cutoff)` computes the physical plaquette states (`physical_plaquette_states`, importable from `conventions.py`, which groups them by plane and signature), takes the link bitmap from `IRREP_TRUNCATIONS`, keyed by the set of irreps on the states' links (or uses `link_bitmap` if passed), and records the truncation as the encoder's `truncation`.
2. `LatticeCircuitManager.for_truncation(lattice_encoder)` computes the matrix elements of box + box^dagger at the encoder's truncation (`box_plus_box_dagger_matrix_elements`, whose docstring says why pyclebsch's matrix elements of box suffice) and passes them to the constructor, keyed by plane and signature (`PlaquetteMatrixElements`). The constructor also takes data in this format directly, e.g. hand-written ones in tests.

pyclebsch caches the plaquette states and matrix elements in memory for the lifetime of the process (uncapped by default; see the README for caps and for pyclebsch's optional disk cache of Clebsch-Gordan coefficients). Scripts that call pyclebsch keep their work under `if __name__ == "__main__":`, because pyclebsch's worker processes import the main script.

### Supported Lattice Configurations
- d=3/2 (plaquette chain), d=2 (square lattice) and d=3 (cubic lattice); no more than three dimensions, as in pyclebsch
- In d=2 and d=3, the same length along every axis (`size` an int) or lengths per axis (`size` a tuple, e.g. `LatticeDef(2, (4, 3))`)
- On a periodic axis of length 2, a link can be a control link at two vertices of a plaquette: the encoder leaves such duplicate controls out of plaquette bit strings, and `LatticeCircuitManager` out of each plaquette circuit's qubits; the encoder refuses (with a `ValueError`) a state in which the two disagree
- Any truncation pyclebsch supports whose set of link irreps has an entry in `IRREP_TRUNCATIONS` (among them T1, T2 and B3-B10 in d=3/2), or any truncation with a link bitmap passed to `LatticeStateEncoder.for_truncation`
- Periodic or open boundary conditions along each axis: `periodic_boundary_conds` is a bool for every axis, or, in d=2 and d=3, a tuple of bools, one per axis (stored as a bool when its entries are all equal)
- d=3/2 of size L is the d=2 lattice of shape (L, 2), open along axis 2; it takes a bool only, for the axis along the chain

### Python Version
Requires Python 3.15 or later (`requires-python = ">=3.15"` in `pyproject.toml`; `.python-version` is `3.15`), as pyclebsch does.
