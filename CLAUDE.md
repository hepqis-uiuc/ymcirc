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
- **Irrep truncations**: T1 = {1, 3, 3̄}, T2 = {1, 3, 3̄, 6, 6̄, 8}

### Module Dependency Flow
```
_abstract/lattice_data.py    Base classes: LatticeData[T], Plaquette[T]
        ↓                      ↓
lattice_registers.py    conventions.py (LatticeStateEncoder, irrep encodings)
        ↓                      ↓
        └──── circuit.py ──────┘   (LatticeCircuitManager — main entry point)
                  ↓
              givens.py            Givens rotation circuit construction
              electric_helper.py   Electric Hamiltonian Pauli decomposition
              parsed_lattice_result.py   Measurement bitstring → physics results
```

### Core Classes

- **`LatticeData[T]`** (`_abstract/lattice_data.py`): Generic base for lattice data. Defines geometry (dimensions, size, boundary conditions) and traversal order. Two concrete subclasses: `LatticeRegisters` and `ParsedLatticeResult`.
- **`Plaquette[T]`** (`_abstract/lattice_data.py`): Generic container for the 4 vertices, 4 active links, and control links of a hypercubic lattice plaquette.
- **`LatticeStateEncoder`** (`conventions.py`): Manages bit-string encodings for link states, vertex states, and plaquette states. Handles irrep truncation levels.
- **`LatticeRegisters`** (`lattice_registers.py`): Maps Qiskit `QuantumRegister`s to lattice links/vertices. Factory method: `from_lattice_state_encoder()`.
- **`LatticeCircuitManager`** (`circuit.py`): Builds simulation circuits. Key methods: `create_blank_full_lattice_circuit()`, `apply_magnetic_trotter_step()`, `apply_electric_trotter_step()`.
- **`ParsedLatticeResult`** (`parsed_lattice_result.py`): Converts measurement bitstrings back into i-weights and multiplicity indices.

### Data Files
`ymcirc/_ymcirc_data/` contains JSON files for magnetic Hamiltonian matrix elements and physical plaquette states, loaded lazily via `LazyDict` (from `utilities.py`).

### Supported Lattice Configurations
- d=3/2 (two-leg ladder), T1 and T2 truncations
- d=2 (square lattice), T1 truncation
- Periodic boundary conditions only

### Python Version
Requires Python ~3.12.0 (pinned in `.python-version` and `pyproject.toml`).
