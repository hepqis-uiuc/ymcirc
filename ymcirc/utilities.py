"""Utility classes and modules."""
from __future__ import annotations
import ast
from collections.abc import Mapping
import gzip
import json
import logging
import numpy as np
from pathlib import Path
from qiskit.circuit import CircuitInstruction, QuantumCircuit, QuantumRegister
import time

# Set up module-specific logger
logger = logging.getLogger(__name__)


class LazyDict(Mapping):
    """
    This is a lazy-loading dictionary.

    It makes working with dictionaries which have expensive-to-compute
    values less painful. Taken from:
    https://stackoverflow.com/questions/16669367/setup-dictionary-lazily.

    Note: LazyDict does NOT cache loaded values. Every __getitem__ call
    invokes the loader function with its argument, re-executing the load
    from scratch. If caching is desired, it must be handled externally
    (e.g., by the loader function itself storing results in a separate dict).
    """

    def __init__(self, *args, **kwargs):
        """
        Create a LazyDict.

        Behaves like a regular dict once created, but
        creating it is a bit different. Consider a dict:
        {
            "key_1": expensive_fn(input_1),
            "key_2": expensive_fn(input_2),
            ...
        }

        To create an equivalent LazyDict:
        LazyDict({
            "key_1": (expensive_fn, input_1),
            "key_2": (expensive_fn, input_2),
            ...
        })
        """
        self._raw_dict = dict(*args, **kwargs)

    def __getitem__(self, key):
        func, arg = self._raw_dict.__getitem__(key)
        return func(arg)

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)


def json_loader(json_path: Path) -> tuple[dict | list, dict]:
    """
    Load a gzip-compressed JSON file at json_path and return (data, metadata).

    The file is expected to be a .json.gz file with a top-level dict containing
    "metadata" and "data" keys. The "data" portion has ast.literal_eval applied
    to convert string representations of tuples back to Python objects.

    Returns a 2-tuple of (data, metadata) where data is a dict or list and
    metadata is a dict.
    """
    logger.debug(f"Loading json data from disk.")
    with gzip.open(json_path, 'rt') as json_file:
        raw_top_level = json.load(json_file)
        metadata = raw_top_level["metadata"]
        raw_data = raw_top_level["data"]
        # Safer to use ast.literal_eval than eval to convert data keys to tuples.
        # The latter can execute arbitrary potentially malicious code while
        # the worst case attack vector for literal_eval would be to crash
        # the python process.
        # See https://docs.python.org/3/library/ast.html#ast.literal_eval
        # for more information.
        if isinstance(raw_data, dict):
            result = {ast.literal_eval(key): value for key, value in raw_data.items()}
        elif isinstance(raw_data, list):
            result = [ast.literal_eval(item) for item in raw_data]
    return result, metadata


# The following methods are intended for package-internal use.
def _check_circuits_logically_equivalent(circ1: QuantumCircuit, circ2: QuantumCircuit, strict: bool = False) -> bool:
    """
    Check whether two circuits have the same gates, in the same order, on the same qubits.

    Qubits are identified by their index in their circuit. For a controlled
    gate, the controls are compared as a set, each with the value that
    triggers the gate, and the target qubits in order, so the same controls
    listed in another order match. For any other gate, the qubits are
    compared in order.

    Gates are compared by their matrices where Qiskit provides one; Qiskit
    (2.3) provides none for a multi-controlled gate with 3 or more controls.
    Two controlled gates whose matrices differ, or that have none, match if
    they have the same number of controls and control a gate of the same
    name, whatever that gate's parameters. (Their matrices differ, for
    example, when the same controls, not all triggered by the same value,
    are listed in another order.) If strict
    is True, gates match only if their matrices are equal, so gates without
    a matrix never match.

    Note this depends on gate order!
    """
    ops1 = circ1.data
    ops2 = circ2.data

    if len(ops1) != len(ops2):
        return False

    # Scan through ops and check equivalence.
    for op1, op2 in zip(ops1, ops2):
        if _qubits_acted_on(circ1, op1) != _qubits_acted_on(circ2, op2):
            return False
        try:
            op1_and_op2_same_matrix = np.allclose(op1.matrix, op2.matrix, atol=1e-15)
        except TypeError:  # Happens if no matrix data available. Fall back on other checks.
            op1_and_op2_same_matrix = False
        if op1_and_op2_same_matrix is False:
            if strict is True:
                return False
            elif op1.is_controlled_gate() and op2.is_controlled_gate():
                # The qubits and control values agree (checked above), so
                # compare the number of controls and the gate they control.
                has_same_num_ctrls = op1.operation.num_ctrl_qubits == op2.operation.num_ctrl_qubits
                has_same_base_gate = op1.operation.base_gate.name == op2.operation.base_gate.name
                if not has_same_num_ctrls or not has_same_base_gate:
                    return False
            else:
                return False

    return True


def _qubits_acted_on(circuit: QuantumCircuit, instruction: CircuitInstruction) -> tuple:
    """
    Return the qubits an instruction of circuit acts on, by index, in a form two instructions can be compared by.

    For a controlled gate, return the set of (control qubit, value that
    triggers the gate) pairs and the tuple of target qubits; the same
    controls listed in another order give the same result. For any other
    instruction, return the tuple of its qubits, in order.
    """
    qubit_indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
    if instruction.is_controlled_gate():
        n_ctrls = instruction.operation.num_ctrl_qubits
        # Qiskit's ctrl_state integer has the first control as its least significant bit.
        ctrl_values = f"{instruction.operation.ctrl_state:0{n_ctrls}b}"[::-1]
        return (frozenset(zip(qubit_indices[:n_ctrls], ctrl_values)), tuple(qubit_indices[n_ctrls:]))
    return (tuple(qubit_indices),)


def _flatten_circuit(original_circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of a circuit which uses only one QuantumRegister."""
    # Get the total number of qubits from all registers
    total_qubits = sum(qr.size for qr in original_circuit.qregs)

    # Create a new larger quantum register
    flattened_qr = QuantumRegister(total_qubits, 'q')
    flattened_circuit = QuantumCircuit(flattened_qr)

    # Create a mapping from original qubits to the new flattened qubits
    mapping = {}
    current_index = 0
    for qr in original_circuit.qregs:
        for i in range(qr.size):
            mapping[qr[i]] = flattened_qr[current_index]
            current_index += 1

    # Copy operations to the new circuit using the mapping
    for instruction in original_circuit.data:
        gate = instruction[0]
        qubits = instruction[1]
        # Create a new list of qubits based on the mapping
        new_qubits = [mapping[q] for q in qubits]
        # Append the operation to the new circuit
        flattened_circuit.append(gate, new_qubits)

    return flattened_circuit


def eta_update(state: dict | None,
               processed: int,
               total: int,
               *,
               alpha: float = 0.5) -> tuple[dict | None, float | None]:
    """
    Update ETA state and return (new_state, eta_seconds or None).

    Uses an exponential moving average (EMA) with decay parameter alpha.

    - state: previously returned state dict, or None to start.
             Has keys "t0", "last", "avg".
    - processed: total items processed so far (int). Saved into "last" for
             future reference.
    - total: total items (int).
    - alpha: EMA smoothing factor in (0,1].
    Returns:
    - state: updated state dict (store and pass back next call)
    - eta_seconds: estimated seconds remaining, or None if insufficient data
    """
    # Initialize state on first call, do nothing if no updates since last call.
    if state is None:
        state = {"t0": time.monotonic(), "last": 0, "avg": None}
    now = time.monotonic()
    if processed <= state["last"]:
        return state, None

    # Get elapsed time since last call.
    # use to estimate time-per-item.
    elapsed = now - state["t0"]
    sample = elapsed / processed if processed else None

    # Update EMA, or set if not yet computed.
    if state["avg"] is None:
        state["avg"] = sample
    elif sample is not None:
        state["avg"] = alpha * sample + (1 - alpha) * state["avg"]

    # Record the new processed count for future idempotency checks.
    state["last"] = processed

    # If no valid average, can't return an ETA.
    if state["avg"] is None:
        return state, None

    # Return state and ETA based on current EMA.
    remaining = max(0, total - processed)
    return state, remaining * state["avg"]


def fmt_td(seconds: float) -> str:
    """Render seconds in a human-readable format."""
    s = int(round(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h:{m:02d}m:{s:02d}s"
    if m:
        return f"{m}m:{s:02d}s"
    return f"{s}s"
