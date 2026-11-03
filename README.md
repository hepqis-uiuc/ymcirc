# ymcirc
A Python package for generating quantum circuits to simulate lattice SU(3) gauge theories.

This codebase is currently at an 'alpha' stage of development. Breaking changes should be expected.

## Installation
This project uses [uv](https://docs.astral.sh/uv/getting-started/) for environment (packages, Python version) management. Ensure that you have uv installed (instructions for various operating systems available at the previously linked-to docs).

Once you have uv installed, use it to execute project scripts, and the correct virtual environment will automatically be used. For example:

``` shell
uv run -m run.time_evol
```
The `-m` flag is necessary since we are executing a module file. For standalone scripts, use the syntax `uv run some_script.py`.

There is no need to manually activate or deactivate the virtual environment. Information about the virtual environment is documented in `pyproject.toml`, and can be viewed by running
```shell
uv pip list
```

### Troubleshooting install issues
If you encounter unexpected errors while attempting to install `ymcirc`, a good first troubleshooting step is to ensure there are no stale versions of packages in the virtual environment `uv` creates (this can cause version conflicts). Perform the following steps to ensure that there are no stale dependencies:
1. Delete the `.venv` directory (if present) inside your cloned copy of `ymcirc`.
2. Update `uv`. Note that the specific [update instructions for `uv`](https://docs.astral.sh/uv/getting-started/installation/#upgrading-uv) depend on how you installed `uv`.
3. Run `uv python upgrade 3.15` (note that the version `3.15` may change in the future).
4. Now that `uv` is up to date and has access to the most recent version of Python, run `uv sync` to rebuild the `.venv` directory in `ymcirc`.

### Using ymcirc in another project
To add ymcirc to a uv project, name a released version by its tag, a branch, or a commit:
```shell
uv add git+https://github.com/hepqis-uiuc/ymcirc --tag v0.2.0     # a released version
uv add git+https://github.com/hepqis-uiuc/ymcirc --branch develop # the latest commit on a branch
uv add git+https://github.com/hepqis-uiuc/ymcirc --rev 5db799a    # a commit, by its hash
```
uv records the choice under `[tool.uv.sources]` in the project's `pyproject.toml`, and the exact commit in `uv.lock`. A project that follows a branch stays at the locked commit until `uv lock --upgrade-package ymcirc` moves it to the branch's latest commit. Running `uv add` again with another tag, branch or commit replaces the previous choice. The release v0.1.0 cannot be installed this way, since it has no `pyproject.toml`.

The project must allow only the versions of Python that ymcirc supports (`requires-python` in ymcirc's `pyproject.toml`), or uv cannot add ymcirc.

### Installation for Windows subsystem for Linux (WSL)
After setting up WSL, download / update Git by running `sudo apt-get install git` before proceeding with the regular installation instructions above. Python does not need to be installed separately: uv installs the version of Python that ymcirc requires.

## Adding, updating, and removing dependencies
To add a package to the project (for example, `numpy`):

``` shell
uv add numpy
```
This automatically updates the `pyproject.toml` file as well as the `uv.lock` file. The former is human readable/editable, while the latter is intended for consumption/editing by uv itself only. (Don't touch it!) Don't use `uv pip` to install packages, because this doesn't automatically update the lockfile.

To add a package to the project that's only needed for development (for example, `pytest`):
```shell
uv add --dev pytest
```

If you want to update a package to a newer version:
```shell
uv add --upgrade numpy
```

To remove a package:
```shell
uv remove numpy
```

### Locking and syncing
The uv tool should automatically lock (generate a machine-readable description of the environment) and sync (update your actual virtual environment) as needed. However, if you run in to environment issues, you might try executing these commands manually as debug steps:
```shell
uv lock
uv sync
```

## Tests
The project uses [pytest](https://docs.pytest.org/en/stable/).
There are also numerous useful [how-to](https://docs.pytest.org/en/stable/how-to/index.html#how-to) guides available.

### Writing new tests
When adding new tests, group the functionality being tested by file. For example, if you were writing tests for a class or function in a module named `ymcirc.some_module`, then all tests for this class should go in a file `tests/test_some_module.py`.

Note that all test files must be named `test_[something].py`!

### Running tests
If you want to run all tests in the `tests` directory, type:
```
uv run pytest -v
```
The `-v` flag is optional and simply outputs additional debug info. Another useful flag is `-s`, which enables displaying all print statements generated while tests are running.

If you want to run all tests in a specific file:
```
uv run pytest tests/test_[file].py
```

If you want to run *just one* test:
```
uv run pytest tests/test_mod.py::test_func
```

There's also more complete documentation on [how to invoke pytest](https://docs.pytest.org/en/stable/how-to/usage.html) which presents some additional features.

### Slow tests
Tests which take a long time to run can be skipped by default. To do this, use the following decorator:
```
@pytest.mark.slow
def test_this_is_some_slow_test():
    [test logic here]
```

`conftest.py` is set up so that any test marked this way will be skipped by default. To include slow tests in a test run:
```
uv run pytest --runslow
```

## Usage
See the `run` directory for example usage and logging configuration of ymcirc. To run the modules within `run`, call (for example):
```
uv run -m run.time_evol
```
Replace `run.time_evol` with the particular desired module in the `run` directory.

Building circuits primarily utilizes four classes:
1. `LatticeDef` (in `ymcirc._abstract`), which sets up the lattice geometry and boundary conditions.
2. `LatticeStateEncoder` (in `ymcirc.conventions`), which defines bitmaps (maps from link irreps and vertex multiplicity indices to bit strings) for encoding gauge field configurations in qubits. In practice, instances are created via `LatticeStateEncoder.for_truncation`, which takes an irrep truncation.
3. `LatticeRegisters` (in `ymcirc.lattice_registers`), which holds the qubit registers, names them according to their location in the lattice so this information is not lost when saving circuits to disk, and provides indexing functionality for working with registers at specific lattice locations. (Note that a QASM file holds each name rewritten as a valid identifier, such as `_l___0__0___1_` for the link register `l:((0, 0), 1)`.)
4. `LatticeCircuitManager` (in `ymcirc.circuit`), created with `LatticeCircuitManager.for_truncation`. This class adds Trotter steps and measurements to a circuit.

Working with measurements primarily makes use of two classes:
1. `ParsedLatticeResult` (in `ymcirc.parsed_lattice_result`), which decodes a bit string measured on the lattice's registers into the irrep on each link and the multiplicity index at each vertex, and provides indexing functionality for these at specific lattice locations. It also computes electric energies of measured links. `ParsedLatticeResult.from_partial_measurement` can be used to create an instance from measurements of only some of the registers.
2. `MeasurementResults` (in `ymcirc.measurement_results`) exists to aggregate shot counts. It is keyed by bit strings or by `ParsedLatticeResult` instances, and enables computation of the (empirical) expectation value of the electric energy, the vacuum persistence probability, and transition probabilities.

### Scripts that build circuits
`LatticeCircuitManager.for_truncation` has pyclebsch compute the magnetic Hamiltonian's matrix elements in parallel, in a pool of worker processes from Python's `multiprocessing`. On macOS and Windows, and on Linux from Python 3.14, each worker starts by importing the script that was run, so all code at the top level of that script runs again in every worker. A worker that reaches `for_truncation` tries to start a pool of its own, which Python refuses with a `RuntimeError`, and the script hangs. To avoid this, put a script's work under `if __name__ == "__main__":`, as the scripts in `run` do (see "Safe importing of main module" in Python's `multiprocessing` documentation).

### Caching of magnetic Hamiltonian data
The physical plaquette states and the magnetic Hamiltonian's matrix elements for a lattice and truncation are computed by [pyclebsch](https://github.com/hepqis-uiuc/pyclebsch), a dependency of `ymcirc`. pyclebsch keeps them in in-memory caches. These caches last for the duration of a running process, and are unlimited by default. To set cache limits, use `pyclebsch.set_physical_plaquette_states_cache_limit(n)` and `pyclebsch.set_plaquette_matrix_elements_cache_limit(n)` to cap them at `n` entries each (an entry holds the results for one plane and plaquette signature at one truncation). When limits are in use, the least-recently used item is evicted from the corresponding cache. For repeated runs, Clebsch-Gordan coefficients can be reused by enabling an optional disk cache which is off by default. To turn it on, use `pyclebsch.set_cgc_cache_dir(path)` or the environment variable `PYCLEBSCH_CACHE_DIR`. See pyclebsch's README for more details on caching behavior.

pyclebsch shows progress bars (from `tqdm`) while it computes. Setting the environment variable `TQDM_DISABLE=1` turns them off, along with every other `tqdm` progress bar in the process. For example, to run `run.time_evol` without progress bars:
```shell
TQDM_DISABLE=1 uv run -m run.time_evol
```
Written this way, the variable is set only for that one command. To turn the progress bars off for every command run afterwards in the same terminal session, run this first:
```shell
export TQDM_DISABLE=1
```

## Supported lattices
ymcirc builds circuits on hypercubic lattices, with geometric data stipulated via a `LatticeDef` (in `ymcirc._abstract`). The following lattices are supported:
- d=3/2, a chain of plaquettes. `LatticeDef("3/2", L)` (or `LatticeDef(1.5, L)`) has L vertices along the chain and is two vertices tall. It has the same vertices, links and plaquettes as the d=2 lattice of shape (L, 2) that is open "across" the chain.
- d=2 and d=3, either with the same number of vertices along every axis (`size` an int, as in `LatticeDef(2, 3)`), or rectangular, with its own number of vertices along each axis (`size` a tuple, as in `LatticeDef(2, (4, 3))` or `LatticeDef(3, (3, 3, 2))`).

Each axis is periodic or open, as set by the argument `periodic_boundary_conds`:
- A bool applies to every axis. The default, `True`, makes every axis periodic.
- In d=2 and d=3, a tuple holds one bool per axis, so a lattice can mix periodic and open axes, as in `LatticeDef(2, (4, 3), periodic_boundary_conds=(True, False))`.
- In d=3/2, only a bool is accepted. It applies to the axis along the chain. The axis across the chain is always open.

## Citations
If you use this code in a paper, please cite:
```bibtex
@article{balaji-2025-perturbat-su,
  eprinttype = {arXiv},
  eprintclass = {hep-lat},
  eprint = {2509.25865},
  issn = {2470-0029},
  month = {May},
  number = {9},
  volume = {113},
  journal = {Physical Review D},
  author = {Balaji, Praveen and Conefrey-Shinozaki, Cianan and Draper, Patrick and Elhaderi, Jason K. and Gupta, Drishti and Hidalgo, Luis and Lytle, Andrew},
  title = {Perturbation theory, irrep truncations, and state preparation methods for quantum simulations of SU(3) lattice gauge theory},
  year = {2026},
  doi = {10.1103/m719-7tdf},
  url = {http://dx.doi.org/10.1103/m719-7tdf},
  publisher = {American Physical Society (APS)}
}
```
