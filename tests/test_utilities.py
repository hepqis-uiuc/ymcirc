"""Tests of the circuit-comparison helper in ymcirc.utilities."""

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RXGate

from ymcirc.utilities import _check_circuits_logically_equivalent

# Qiskit (2.3.0, as locked) provides a matrix for a multi-controlled X with
# at most 2 controls and none from 3 controls up. The helper compares gates
# with a matrix and gates without one differently, so the tests cover both:
# 2 controls, the most that still have a matrix, and 5, comfortably past the
# point where the matrix disappears.
N_CTRLS_WITH_MATRIX = 2
N_CTRLS_WITHOUT_MATRIX = 5
# The N_CTRLS_WITHOUT_MATRIX controls, a target, and one more qubit to serve
# as a different target.
N_QUBITS = N_CTRLS_WITHOUT_MATRIX + 2
# Any angle at which RY and RX are not the identity.
ANGLE = 0.3


def _circuit(*gates) -> QuantumCircuit:
    """Return an N_QUBITS-qubit circuit with the given gates, each a (method name, args, kwargs) triple."""
    circuit = QuantumCircuit(N_QUBITS)
    for method_name, args, kwargs in gates:
        getattr(circuit, method_name)(*args, **kwargs)
    return circuit


def _mcx(controls, target, ctrl_state=None):
    """Return a multi-controlled X as a gate triple for _circuit."""
    return (
        "mcx",
        (list(controls), target),
        {} if ctrl_state is None else {"ctrl_state": ctrl_state},
    )


def _mcrx(controls, target):
    """Return a multi-controlled RX by ANGLE as a gate triple for _circuit."""
    return (
        "append",
        (RXGate(ANGLE).control(len(controls)), list(controls) + [target]),
        {},
    )


CTRLS_OF_MCX_WITH_MATRIX = list(range(N_CTRLS_WITH_MATRIX))
CTRLS_OF_MCX_WITHOUT_MATRIX = list(range(N_CTRLS_WITHOUT_MATRIX))

# Each case: (description, first circuit, second circuit, expected verdict).
CASES = [
    ("identical CX", _circuit(("cx", (0, 1), {})), _circuit(("cx", (0, 1), {})), True),
    (
        "CX on a different target",
        _circuit(("cx", (0, 1), {})),
        _circuit(("cx", (0, 2), {})),
        False,
    ),
    (
        "CX with control and target exchanged",
        _circuit(("cx", (0, 1), {})),
        _circuit(("cx", (1, 0), {})),
        False,
    ),
    (
        "RY on a different qubit",
        _circuit(("ry", (ANGLE, 0), {})),
        _circuit(("ry", (ANGLE, 1), {})),
        False,
    ),
    (
        "RY by a different angle",
        _circuit(("ry", (ANGLE, 0), {})),
        _circuit(("ry", (2 * ANGLE, 0), {})),
        False,
    ),
    (
        "MCX with a matrix, different target",
        _circuit(_mcx(CTRLS_OF_MCX_WITH_MATRIX, N_QUBITS - 1)),
        _circuit(_mcx(CTRLS_OF_MCX_WITH_MATRIX, N_QUBITS - 2)),
        False,
    ),
    (
        "MCX with a matrix, controls listed in another order",
        _circuit(_mcx(CTRLS_OF_MCX_WITH_MATRIX, N_QUBITS - 1, ctrl_state="01")),
        _circuit(_mcx(CTRLS_OF_MCX_WITH_MATRIX[::-1], N_QUBITS - 1, ctrl_state="10")),
        True,
    ),
    (
        "MCX without a matrix, different target",
        _circuit(_mcx(CTRLS_OF_MCX_WITHOUT_MATRIX, N_QUBITS - 1)),
        _circuit(_mcx(CTRLS_OF_MCX_WITHOUT_MATRIX, N_QUBITS - 2)),
        False,
    ),
    (
        "MCX without a matrix, a different control value",
        _circuit(_mcx(CTRLS_OF_MCX_WITHOUT_MATRIX, N_QUBITS - 1, ctrl_state="00001")),
        _circuit(_mcx(CTRLS_OF_MCX_WITHOUT_MATRIX, N_QUBITS - 1, ctrl_state="00011")),
        False,
    ),
    (
        "MCX without a matrix, controls listed in another order",
        _circuit(_mcx(CTRLS_OF_MCX_WITHOUT_MATRIX, N_QUBITS - 1, ctrl_state="00001")),
        _circuit(
            _mcx(CTRLS_OF_MCX_WITHOUT_MATRIX[::-1], N_QUBITS - 1, ctrl_state="10000")
        ),
        True,
    ),
    (
        "multi-controlled X and RX without matrices, same qubits",
        _circuit(_mcx(CTRLS_OF_MCX_WITHOUT_MATRIX, N_QUBITS - 1)),
        _circuit(_mcrx(CTRLS_OF_MCX_WITHOUT_MATRIX, N_QUBITS - 1)),
        False,
    ),
    (
        "different numbers of gates",
        _circuit(("cx", (0, 1), {})),
        _circuit(("cx", (0, 1), {}), ("cx", (0, 1), {})),
        False,
    ),
]


@pytest.mark.parametrize(
    "first_circuit, second_circuit, expected_verdict",
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_check_circuits_logically_equivalent(
    first_circuit: QuantumCircuit,
    second_circuit: QuantumCircuit,
    expected_verdict: bool,
):
    """Check that the helper compares every gate's kind and the qubits it acts on."""
    assert (
        _check_circuits_logically_equivalent(first_circuit, second_circuit)
        is expected_verdict
    )
    assert (
        _check_circuits_logically_equivalent(second_circuit, first_circuit)
        is expected_verdict
    )
