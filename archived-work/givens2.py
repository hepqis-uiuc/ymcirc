"""
Givens-2 implementation.

Author: Cianan Conefrey-Shinozaki, Oct 2024.
"""
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RXGate


def givens2(strings: list, angle: float, reverse: bool = False) -> QuantumCircuit:
    """
    Build QuantumCircuit rotating two bit strings into each other by angle.

    Expects a nested list of pairs of bitstrings in physicist / big-endian notation,
    e.g. strings = [['10','01'],['00','11']], and a float as an angle.
    Optional reverse arg for little-endian.
    """
    if len(strings) != 2:
        raise ValueError("Need two pairs")

    for ii in range(2):
        for jj in range(2):
            if len(strings[ii][jj]) != len(strings[ii - 1][jj]) or len(
                strings[ii][jj]
            ) != len(strings[ii][jj - 1]):
                raise ValueError("All bitstrings must be same length")

    num_qubits = len(strings[0][0])

    strings_reversed = strings
    for ii in range(2):
        for jj in range(2):
            strings_reversed[ii][jj] = strings[ii][jj][::-1]

    pin = None
    for ii in range(num_qubits - 1, -1, -1):  # This finds where to put RY
        if (
            strings_reversed[0][0][ii] != strings_reversed[0][1][ii]
            and strings_reversed[1][0][ii] != strings_reversed[1][1][ii]
        ):
            pin = ii
            break
    if pin is None:
        raise ValueError("Your bitstrings lack the special sauce")

    lock = None
    type_2 = True
    for ii in range(num_qubits - 1, -1, -1):
        property_1 = strings_reversed[0][0][ii] == strings_reversed[0][1][ii]
        property_2 = strings_reversed[1][0][ii] == strings_reversed[1][1][ii]
        property_3 = strings_reversed[0][0][ii] != strings_reversed[1][0][ii]
        if property_1 and property_2 and property_3:
            lock = ii
            type_2 = False
            break
    if type_2 is True:
        candidates = list(range(num_qubits - 1, -1, -1))
        candidates.remove(pin)
        for ii in candidates:
            if (
                strings_reversed[0][0][ii] != strings_reversed[0][1][ii]
                and strings_reversed[1][0][ii] != strings_reversed[1][1][ii]
            ):
                lock = ii
                break

    if lock is None:
        raise ValueError("Your bitstrings lack the special sauce")

    bad1 = (
        strings[0][0][pin] + strings[0][0][lock]
        == strings[1][0][pin] + strings[1][0][lock]
    )
    bad2 = (
        strings[0][0][pin] + strings[0][0][lock]
        == strings[1][1][pin] + strings[1][1][lock]
    )
    if type_2 and (bad1 or bad2):
        raise ValueError("Your bitstrings lack the special sauce")

    R_ctrls = list(range(0, num_qubits))
    R_ctrls.remove(pin)
    R_ctrls.remove(lock)

    R_ctrl_state = strings[0][0]
    if pin < lock:
        R_ctrl_state = (
            R_ctrl_state[:pin]
            + R_ctrl_state[(pin + 1) : lock]
            + R_ctrl_state[(lock + 1) :]
        )
    else:
        R_ctrl_state = (
            R_ctrl_state[:lock]
            + R_ctrl_state[(lock + 1) : pin]
            + R_ctrl_state[(pin + 1) :]
        )

    R_ctrl_state = R_ctrl_state[::-1]

    R_circ = QuantumCircuit(num_qubits)
    if len(R_ctrls) == 0:
        R_circ.rx(angle, pin)
    else:
        R_circ.append(
            RXGate(angle).control(
                num_ctrl_qubits=num_qubits - 2, ctrl_state=R_ctrl_state
            ),
            R_ctrls + [pin],
        )

    X_2_circ = QuantumCircuit(num_qubits)

    for ii in R_ctrls + [lock]:
        if strings[0][0][ii] != strings[0][1][ii]:
            X_2_circ.cx(pin, ii, ctrl_state=strings[0][1][pin])

    if strings[0][0][pin] == strings[1][0][pin]:
        string_3 = strings[1][0]
        string_4 = strings[1][1]
    else:
        string_3 = strings[1][1]
        string_4 = strings[1][0]

    X_3_circ = QuantumCircuit(num_qubits)

    for ii in R_ctrls:
        if strings[0][0][ii] != string_3[ii]:
            X_3_circ.mcx([lock] + [pin], ii, ctrl_state=string_3[pin] + string_3[lock])

    X_4_circ = QuantumCircuit(num_qubits)

    for ii in R_ctrls:
        if strings[0][1][ii] != string_4[ii]:
            X_4_circ.mcx([lock] + [pin], ii, ctrl_state=string_4[pin] + string_4[lock])

    X_3_circ.compose(X_4_circ, inplace=True)
    R_circ.compose(X_2_circ, inplace=True)
    X_2_circ.compose(R_circ, inplace=True)
    X_2_circ.compose(X_3_circ, inplace=True)
    X_3_circ.compose(X_2_circ, inplace=True)

    return X_3_circ
