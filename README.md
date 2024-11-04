# ymcirc
Circuit Hamiltonians for SU(N) gauge theories

## TOC
* decompose_pauli.py - Pauli string decomposition of matrices  
* test_circuits.py - Verifying Trotter evolution circuits  
* trailhead_III_B1.py - Global basis circuits single plaquette  
* trailhead_IV_A1.py - Global basis circuits two-plaquette PBCs  
* Trotter_evol.py - Trotter evolution routines  
* QSP/ - "Quantum signal processing" approach for 1-sparse matrices  
    * oracle_1sparse.py

## Installation
The Makefile is configured to automatically set up a Python virtual environment with version ~1.2 of qiskit. To use it:

1. Run `make venv` to create the Python virtual environment.
2. Run `source .venv/bin/activate` to activate the virtual environment.
3. If you want to deactivate the virtual environment, run `deactivate`.
4. Run `make clean` to remove the Python virtual environment. Make sure you deactive the virtual environment before doing this!

## Installation for Windows subsystem for Linux (WSL)
After setting up WSL there is a checklist of programs you will need before proceeding with the regular installation instructions above:

1. Download / update Git by running `sudo apt-get install git`.
2. Download / update Python3 by running `sudo apt install python3 python3-pip`
3. Download the venv package by running `sudo apt install python3.10-venv`
