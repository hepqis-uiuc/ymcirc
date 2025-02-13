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
