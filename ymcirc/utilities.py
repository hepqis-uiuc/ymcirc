"""Utility classes and modules."""
from __future__ import annotations
import ast
from collections.abc import Mapping
import json
import logging
import numpy as np
from pathlib import Path
from qiskit.circuit import QuantumCircuit, QuantumRegister
import time
from typing import Dict, List, Optional, Tuple

# Set up module-specific logger
logger = logging.getLogger(__name__)


class LazyDict(Mapping):
    """
    This is a lazy-loading dictionary.

    It makes working with dictionaries which have expensive-to-compute
    values less painful. Taken from:
    https://stackoverflow.com/questions/16669367/setup-dictionary-lazily.
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


def json_loader(json_path: Path) -> Dict | List:
    """
    Load the json file at json_path, and return the data as dict or list.
    """
    logger.debug(f"Loading json data from disk.")
    with json_path.open('r') as json_file:
        raw_data = json.load(json_file)
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
    return result


# The following methods are intended for package-internal use.
def _check_circuits_logically_equivalent(circ1: QuantumCircuit, circ2: QuantumCircuit, strict: bool = False) -> bool:
    """
    Check two circuits for logical equivalence by checking whether they have the same gates.

    If strict is False, then for larger multi control unitaries, just checks for the right ctrl qubits, target qubit, and ctrl state.

    Note this depends on gate order!
    """
    ops1 = circ1.data
    ops2 = circ2.data

    if len(ops1) != len(ops2):
        return False

    # Scan through ops and check equivalence.
    for idx, (op1, op2) in enumerate(zip(ops1, ops2)):
        try:
            op1_and_op2_same_matrix = np.allclose(op1.matrix, op2.matrix, atol=1e-15)
        except TypeError:  # Happens if no matrix data available. Fall back on other checks.
            op1_and_op2_same_matrix = False
        if op1_and_op2_same_matrix is False:
            if strict is True:
                return False
            elif op1.is_controlled_gate() and op2.is_controlled_gate():
                # Initial sanity check.
                has_same_qubits = set(op1.qubits) == set(op2.qubits)
                has_same_num_ctrls = op1.operation.num_ctrl_qubits == op2.operation.num_ctrl_qubits
                if not has_same_qubits or not has_same_num_ctrls:
                    return False

                # Now check for equality of target qubit.
                op1_target = op1.qubits[-1]
                op2_target = op2.qubits[-1]
                has_same_target = op1_target == op2_target
                if not has_same_target:
                    return False

                # Now let verify that the control qubits are the same, and the
                # control state is identical. We need to obey qiskit's
                # little endian convention when constructing ctrl state strings.
                op1_ctrls = op1.qubits[:-1]
                op2_ctrls = op2.qubits[:-1]
                op1_ctrl_state = f'{op1.operation.ctrl_state:0{op1.operation.num_ctrl_qubits}b}'[::-1]
                op2_ctrl_state = f'{op2.operation.ctrl_state:0{op2.operation.num_ctrl_qubits}b}'[::-1]

                # Now let's gather all the qubit indices showing up among the controls
                # of both operations, and set the value of that qubit in the
                # corresponding control string.
                ctrl_val_comp_dict: dict[int, dict[str, str]] = {idx: {} for idx in range(circ1.num_qubits)}
                for qubit_idx in range(op1.operation.num_ctrl_qubits):
                    # NOTE: This is pretty hacky and could break in the future.
                    ctrl_val_comp_dict[op1_ctrls[qubit_idx]._index]["op1_ctrl_val"] = op1_ctrl_state[qubit_idx]
                    ctrl_val_comp_dict[op2_ctrls[qubit_idx]._index]["op2_ctrl_val"] = op2_ctrl_state[qubit_idx]

                # Now the control string data is organized enough for simple equality checks.
                for idx in ctrl_val_comp_dict.keys():
                    has_no_ctrl_at_current_idx = len(ctrl_val_comp_dict[idx]) == 0
                    ctrl_values_match = len(ctrl_val_comp_dict[idx]) == 2 and (ctrl_val_comp_dict[idx]["op1_ctrl_val"] == ctrl_val_comp_dict[idx]["op2_ctrl_val"])
                    if has_no_ctrl_at_current_idx:
                        continue
                    elif ctrl_values_match:
                        continue
                    else:
                        return False
            else:
                return False

    return True


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


def eta_update(state: Optional[Dict] | None,
               processed: int,
               total: int,
               *,
               alpha: float = 0.2) -> Tuple[Optional[Dict], Optional[float]]:
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
