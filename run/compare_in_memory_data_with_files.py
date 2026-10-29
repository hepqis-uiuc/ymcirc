"""
Compare the plaquette states and matrix elements pyclebsch computes with ymcirc's data files.

A one-off check, made before the data files are removed: the data files are
the reference that the in-memory results must reproduce. It makes three
comparisons and prints the result of each.

1. States and matrix elements, for every data file, on the lattice recorded in
   the file's metadata: the physical plaquette states are equal as sets; the
   matrix elements have the same keys at every level (pair of plaquette
   states, plane, signature); and every value agrees within VALUE_TOLERANCE.
2. Link bitmaps, for every data file: the encoder that
   LatticeStateEncoder.for_truncation creates uses the bitmap that the file's
   truncation name had in IRREP_TRUNCATIONS.
3. Circuits, for each case of CIRCUIT_CASES: the magnetic Trotter steps built
   from the files and from the in-memory results, set up as run/functions.py
   does, have equal gate lists, with the Givens rotations within each LP bin
   sorted by their pair of bit strings. Each case runs in its own process.

Usage, from the root of a ymcirc clone:
    uv run python -m run.compare_in_memory_data_with_files
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any

import numpy as np
from qiskit.circuit import ParameterExpression, QuantumCircuit

import run.functions as run_functions
from ymcirc import conventions
from ymcirc._abstract import LatticeDef
from ymcirc._pyclebsch_adapter import (
    PRES,
    _pyclebsch_lattice,
    box_terms,
    physical_plaquette_states,
)
from ymcirc.circuit import LatticeCircuitManager

# Both sources round matrix elements to PRES decimals, so two roundings of one
# number can differ by one unit in the last decimal kept; a larger difference
# is a real disagreement.
VALUE_TOLERANCE = 10**-PRES

# The data files, by dimension and truncation name.
DATA_FILES = [
    ("d=3/2", "T1"),
    ("d=3/2", "T2"),
    ("d=3/2", "B3"),
    ("d=3/2", "B5"),
    ("d=3/2", "B6"),
    ("d=3/2", "B7"),
    ("d=3/2", "B8"),
    ("d=3/2", "B9"),
    ("d=3/2", "B10"),
    ("d=2", "T1"),
    ("d=2", "B3"),
    ("d=2", "B4"),
    ("d=2", "B7"),
    ("d=3", "B3"),
    ("d=3", "B4"),
]
_DIMS_BY_DIM_STRING = {"d=3/2": 1.5, "d=2": 2, "d=3": 3}
# This module's name, under which each circuit case reruns it in a process of
# its own.
_MODULE_NAME = "run.compare_in_memory_data_with_files"

# (dimension, truncation name, lattice size, options of
# run/functions.configure_script_options that differ from its defaults). Each
# builds its magnetic Trotter step from the files in under a minute; the
# size-2 cases check the filtering of size-2 data in each dimension.
CIRCUIT_CASES: list[tuple[str, str, int, dict[str, Any]]] = [
    ("d=3/2", "T1", 3, {}),
    ("d=3/2", "B3", 3, {}),
    ("d=3/2", "B5", 3, {}),
    ("d=3/2", "B6", 3, {}),
    ("d=3/2", "B7", 3, {}),
    ("d=2", "B3", 3, {}),
    ("d=3/2", "T1", 2, {}),
    ("d=2", "B3", 2, {}),
    ("d=2", "T1", 3, {"use_ancillas": False, "prune_controls": False}),
    ("d=3", "B3", 3, {"use_ancillas": False, "prune_controls": False}),
    ("d=3", "B3", 2, {"use_ancillas": False, "prune_controls": False}),
    (
        "d=3",
        "B3",
        2,
        {
            "use_ancillas": False,
            "prune_controls": True,
            "mag_hamiltonian_matrix_element_threshold": 0.7,
        },
    ),
]
# Values given to the circuits' time-step and coupling parameters before
# rotation angles are compared. Any values that are not special (such as 0 or
# 1) will do: the angle is proportional to dt / g**2.
_DT_VALUE = 0.3
_COUPLING_G_VALUE = 1.7


def _truncation_mode_and_cutoff(trunc_string: str) -> tuple[str, int]:
    """Split a truncation name such as "B10" into ("B", 10)."""
    return trunc_string[0], int(trunc_string[1:])


def _file_lattice(dim_string: str, trunc_string: str) -> LatticeDef:
    """Return the lattice a data file was computed on, from its metadata."""
    metadata = conventions.get_data_metadata(dim_string, trunc_string)
    lattice = LatticeDef(
        _DIMS_BY_DIM_STRING[dim_string],
        metadata["num_sites"][0],
        forder=metadata["f_order"],
    )
    computed_on = _pyclebsch_lattice(lattice)
    if (
        list(computed_on.num_sites) != metadata["num_sites"]
        or list(computed_on.PBCs) != metadata["PBCs"]
    ):
        raise ValueError(
            f"{dim_string} {trunc_string}: the adapter computes on "
            f"{computed_on.num_sites}, {computed_on.PBCs}; the file's metadata "
            f"says {metadata['num_sites']}, {metadata['PBCs']}."
        )
    return lattice


def compare_states_and_matrix_elements(dim_string: str, trunc_string: str) -> dict:
    """Compare one data file with the in-memory results (comparison 1)."""
    lattice = _file_lattice(dim_string, trunc_string)
    truncation_mode, cutoff = _truncation_mode_and_cutoff(trunc_string)
    file_states = set(conventions.PHYSICAL_PLAQUETTE_STATES[dim_string][trunc_string])
    memory_states = set(physical_plaquette_states(lattice, truncation_mode, cutoff))
    file_box_terms = conventions.HAMILTONIAN_BOX_TERMS[dim_string][trunc_string]
    memory_box_terms = box_terms(lattice, truncation_mode, cutoff)

    result: dict[str, Any] = {
        "file": f"{dim_string} {trunc_string}",
        "states": len(file_states),
        "states_equal": file_states == memory_states,
        "pairs": len(file_box_terms),
        "pairs_equal": file_box_terms.keys() == memory_box_terms.keys(),
        "plane_keys_equal": True,
        "signature_keys_equal": True,
        "values": 0,
        "max_abs_difference": 0.0,
    }
    for pair in file_box_terms.keys() & memory_box_terms.keys():
        file_planes, memory_planes = file_box_terms[pair], memory_box_terms[pair]
        if file_planes.keys() != memory_planes.keys():
            result["plane_keys_equal"] = False
        for plane in file_planes.keys() & memory_planes.keys():
            file_values, memory_values = file_planes[plane], memory_planes[plane]
            if file_values.keys() != memory_values.keys():
                result["signature_keys_equal"] = False
            for signature in file_values.keys() & memory_values.keys():
                # float(), because pyclebsch's values are numpy floats.
                difference = float(
                    abs(file_values[signature] - memory_values[signature])
                )
                result["values"] += 1
                result["max_abs_difference"] = max(
                    result["max_abs_difference"], difference
                )
    result["passed"] = (
        result["states_equal"]
        and result["pairs_equal"]
        and result["plane_keys_equal"]
        and result["signature_keys_equal"]
        and result["max_abs_difference"] <= VALUE_TOLERANCE
    )
    return result


def _bitmap_named(dim_string: str, trunc_string: str) -> tuple[str, dict]:
    """Return the name and bitmap that run/functions.py used for a file's truncation."""
    if trunc_string in conventions.IRREP_TRUNCATIONS:
        return trunc_string, conventions.IRREP_TRUNCATIONS[trunc_string]
    for name, bitmap in conventions.IRREP_TRUNCATIONS.items():
        if not isinstance(name, str) or "_" not in name:
            continue
        prefix, dims = name.split("_")
        if prefix == trunc_string and dim_string in dims.split(","):
            return name, bitmap
    raise KeyError(f"No truncation name for {dim_string} {trunc_string}.")


def compare_link_bitmap(dim_string: str, trunc_string: str) -> dict:
    """Compare the bitmap of LatticeStateEncoder.for_truncation with a file's named bitmap (comparison 2)."""
    lattice = _file_lattice(dim_string, trunc_string)
    truncation_mode, cutoff = _truncation_mode_and_cutoff(trunc_string)
    name, named_bitmap = _bitmap_named(dim_string, trunc_string)
    bitmap = conventions.LatticeStateEncoder.for_truncation(
        lattice, truncation_mode, cutoff
    ).link_bitmap
    return {
        "file": f"{dim_string} {trunc_string}",
        "name": name,
        "passed": bitmap == named_bitmap,
    }


def _sort_rotations_within_lp_bins() -> None:
    """Make LatticeCircuitManager sort the Givens rotations within each LP bin by their pair of bit strings."""
    original = LatticeCircuitManager._sort_matrix_elements_into_lp_bins

    def sorted_within_bins(*args, **kwargs):
        lp_bins = original(*args, **kwargs)
        return {
            lp_family: sorted(rotations, key=lambda rotation: rotation[:2])
            for lp_family, rotations in lp_bins.items()
        }

    # A deliberate replacement, of a different type than ty infers.
    LatticeCircuitManager._sort_matrix_elements_into_lp_bins = staticmethod(  # ty: ignore[invalid-assignment]
        sorted_within_bins
    )


def _parameter_value(parameter: Any) -> Any:
    """Return a gate parameter as a number, with the time step and coupling bound to fixed values."""
    if not isinstance(parameter, ParameterExpression):
        return parameter
    values = {}
    for symbol in parameter.parameters:
        if symbol.name.startswith("dt"):
            values[symbol] = _DT_VALUE
        elif symbol.name.startswith("coupling_g"):
            values[symbol] = _COUPLING_G_VALUE
        else:
            raise ValueError(f"Unexpected circuit parameter {symbol.name}.")
    return float(parameter.bind(values))


def _gate_list(circuit: QuantumCircuit) -> list[tuple]:
    """Return each gate's name, parameter values, control values and qubit indices, in order."""
    qubit_indices = {qubit: index for index, qubit in enumerate(circuit.qubits)}
    return [
        (
            instruction.operation.name,
            tuple(_parameter_value(p) for p in instruction.operation.params),
            getattr(instruction.operation, "ctrl_state", None),
            tuple(qubit_indices[qubit] for qubit in instruction.qubits),
        )
        for instruction in circuit.data
    ]


def _build_magnetic_step(options: dict[str, Any]) -> QuantumCircuit:
    """Set up the lattice as run/functions.py does, and apply one magnetic Trotter step."""
    (
        _,
        physical_states,
        lattice_registers,
        circ_mgr,
        circuit,
    ) = run_functions.initialize_lattice_tools(options)
    circ_mgr.apply_magnetic_trotter_step(
        circuit,
        lattice_registers,
        optimize_circuits=options["optimize_circuits"],
        physical_states_for_control_pruning=physical_states,
        control_fusion=options["control_fusion"],
        cache_mag_evol_circuit=options["cache_mag_evol_circuit"],
    )
    return circuit


def compare_circuits(case_index: int) -> dict:
    """Build one case's magnetic Trotter step from both sources and compare the gate lists (comparison 3)."""
    dim_string, trunc_string, size, overrides = CIRCUIT_CASES[case_index]
    _sort_rotations_within_lp_bins()
    options = run_functions.configure_script_options(
        dimensionality_string=dim_string,
        truncation_string=trunc_string,
        lattice_size=size,
        sim_times=np.linspace(0, 1, 2),
        n_trotter_steps=1,
        n_shots=None,
        **overrides,
    )
    result: dict[str, Any] = {
        "case": f"{dim_string} {trunc_string} size {size} {json.dumps(overrides)}"
    }

    start = time.perf_counter()
    file_gates = _gate_list(_build_magnetic_step(options))
    result["files_s"] = round(time.perf_counter() - start, 1)

    # Route ymcirc's data lookups to the in-memory results for this lattice.
    truncation_mode, cutoff = _truncation_mode_and_cutoff(trunc_string)
    lattice = options["lattice_def"]
    states = {
        dim_string: {
            trunc_string: physical_plaquette_states(lattice, truncation_mode, cutoff)
        }
    }
    # Deliberate replacements of the files' lazily loaded data, of a different
    # type than ty infers.
    conventions.PHYSICAL_PLAQUETTE_STATES = states  # ty: ignore[invalid-assignment]
    run_functions.PHYSICAL_PLAQUETTE_STATES = states  # ty: ignore[invalid-assignment]
    conventions.HAMILTONIAN_BOX_TERMS = {  # ty: ignore[invalid-assignment]
        dim_string: {trunc_string: box_terms(lattice, truncation_mode, cutoff)}
    }
    start = time.perf_counter()
    memory_gates = _gate_list(_build_magnetic_step(options))
    result["memory_s"] = round(time.perf_counter() - start, 1)

    result["gates"] = len(file_gates)
    result["passed"] = file_gates == memory_gates
    if not result["passed"]:
        result["memory_gates"] = len(memory_gates)
        result["first_difference"] = next(
            (
                index
                for index, (a, b) in enumerate(zip(file_gates, memory_gates))
                if a != b
            ),
            min(len(file_gates), len(memory_gates)),
        )
    return result


def main() -> None:
    """Run the three comparisons and print their results."""
    print("1. States and matrix elements")
    for dim_string, trunc_string in DATA_FILES:
        print(json.dumps(compare_states_and_matrix_elements(dim_string, trunc_string)))
    print("2. Link bitmaps")
    for dim_string, trunc_string in DATA_FILES:
        print(json.dumps(compare_link_bitmap(dim_string, trunc_string)))
    print("3. Circuits")
    for case_index in range(len(CIRCUIT_CASES)):
        completed = subprocess.run(
            [sys.executable, "-m", _MODULE_NAME, "circuit", str(case_index)],
            check=False,  # A failed case is reported below, with its errors.
            capture_output=True,
            text=True,
            env=os.environ,
        )
        output = completed.stdout.strip().splitlines()
        print(
            output[-1]
            if output
            else f"case {case_index} failed: {completed.stderr[-2000:]}"
        )


if __name__ == "__main__":
    # pyclebsch computes in a multiprocessing.Pool, whose worker processes
    # import this module afresh; this guard keeps them from rerunning it.
    if len(sys.argv) == 3 and sys.argv[1] == "circuit":
        print(json.dumps(compare_circuits(int(sys.argv[2]))))
    else:
        main()
