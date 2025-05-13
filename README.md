# ymcirc
A Python package for generating quantum circuits to simulate lattice SU(N) gauge theories.

This codebase is currently at an 'alpha' stage of development. Breaking changes should be expected.

## Installation
The Makefile is configured to automatically set up a Python virtual environment with version ~1.2 of qiskit. To use it:

1. Run `make venv` to create the Python virtual environment.
2. Run `source .venv/bin/activate` to activate the virtual environment.
3. If you want to deactivate the virtual environment, run `deactivate`.
4. Run `make clean` to remove the Python virtual environment. Make sure you deactive the virtual environment before doing this!

Alternatively, you can use the `requirements.txt` file to set up a virtual environment with your favored environment management tool.

### Installation for Windows subsystem for Linux (WSL)
After setting up WSL there is a checklist of programs you will need before proceeding with the regular installation instructions above:

1. Download / update Git by running `sudo apt-get install git`.
2. Download / update Python3 by running `sudo apt install python3 python3-pip`
3. Download the venv package by running `sudo apt install python3.10-venv`

## Tests
The project uses [pytest](https://docs.pytest.org/en/stable/).
There are also numerous useful [how-to](https://docs.pytest.org/en/stable/how-to/index.html#how-to) guides available.

### Writing new tests
When adding new tests, group the functionality being tested by file. For example, if you were writing tests for a class or function in a module named `ymcirc.some_module`, then all tests for this class should go in a file `tests/test_some_module.py`.

Note that all test files must be named `test_[something].py`!

### Running tests
If you want to run all tests in the `test` directory, activate the virtual environment and then type:
```
pytest -v
```
The `-v` flag is optional and simply outputs additional debug info. Another useful flag is `-s`, which enables displaying all print statements generated while tests are running.

If you want to run all tests in a specific file:
```
pytest tests/test_[file].py
```

If you want to run *just one* test:
```
pytest tests/test_mod.py::test_func.
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
pytest --runslow
```
