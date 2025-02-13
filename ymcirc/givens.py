"""
This module contains methods for constructing Givens rotation circuits.

Original author: Cianan Conefrey-Shinozaki, Oct 2024.

A Givens rotation is a generalization of an X-rotation; it rotates
two multi-qubit states into each other.
"""
from __future__ import annotations
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.circuit.library.standard_gates import RXGate
from qiskit.quantum_info import Operator
from random import random
import numpy as np
from scipy.linalg import expm
from typing import Tuple, List, Set


def givens(
        bit_string_1: str,
        bit_string_2: str,
        angle: float,
        reverse: bool = False,
        physical_control_qubits: Set[int | Qubit] | None = None
) -> QuantumCircuit:
    """
    Build QuantumCircuit rotating two bit strings into each other by angle.

    Expects two bitstrings in physicist / big-endian notation,
    e.g. "1001" or "0100", and a float as an angle.
    Optional reverse arg for outputting a circuit in little-endian convention
    (used by qiskit).

    The resulting circuit sandwiches a multi-control X rotation (MCRX) between
    a pre/post computation circuit consisting of CX gates. The pre/post
    computation is essentially just a change of basis for the MCRX.

    If a set physical_control_qubits consisting is provided which
    contains data indexing qubits in a QuantumRegister, only
    those qubits appearing in physical_control_qubits will be
    used when adding controls to construct the givens rotation. This is
    relevant for "control pruning" where we allow the givens rotation
    to rotate pairs of unphysical states into each other in addition
    to the target states bit_string_1 and bit_string_2.
    """
    # Input validation and sanity checks.
    if len(bit_string_1) != len(bit_string_2):
        raise ValueError('Bit strings must be the same length.')
    no_rotation_is_needed = bit_string_1 == bit_string_2
    if no_rotation_is_needed:
        return QuantumCircuit(len(bit_string_1))  # The identity circuit.

    num_qubits = len(bit_string_1)
    circ = QuantumCircuit(num_qubits)

    # Build the circuit.
    if num_qubits == 1:
        # No pre/post computation needed.
        circ.rx(angle, 0)
        return circ
    else:
        for idx in range(num_qubits):
            current_idx_is_target_idx = bit_string_1[idx] != \
                bit_string_2[idx]
            if current_idx_is_target_idx is True:
                target = idx
                break

        # Configure the multi-controlled X rotation (MCRX) gate.
        ctrls, ctrl_state, multiRX = _build_multiRX(
            bit_string_1,
            bit_string_2,
            angle,
            target,
            physical_control_qubits
        )

        # Construct the pre- and post-MCRX circuits.
        Xcirc = _build_Xcirc(
            bit_string_1, bit_string_2, control=target)

        # Add multiRX to the circuit, specifying
        # The proper control locations and target location
        # via a list of the qubit indices.
        circ.append(
            multiRX,
            ctrls + [target]
        )

        # Assemble the final circuit.
        # Using inplace speeds up circuit composition.
        circ.compose(Xcirc, inplace=True)
        Xcirc.compose(circ, inplace=True)
        if reverse is True:
            Xcirc = Xcirc.reverse_bits()
        return Xcirc


def _build_multiRX(
        bs1_little_endian: str,
        bs2_little_endian: str,
        angle: float,
        target: int,
        physical_control_qubits: Set[int | Qubit] | None
) -> Tuple[List[int], str, QuantumCircuit]:
    """
    Build the multi-control RX gate (MCRX) in Givens rotations.

    If a set physical_control_qubits is provided, any controls
    NOT in that set are ignored when constructing the multi-control.
    This is useful if a set of sufficient control qubits has been determined
    which acts as a Givens rotation on the physical state space.

    Return:
      - (ctrls, ctrl_state, multiRX): a tuple consisting of control qubit
        indices, the bit string state of those those controls which triggers
        the rotation gate, and the circuit implementing the MCRX gate.

    Note that the final qubit in multiRX is the target qubit.
    """
    # Determine the set of control qubits.
    # The target (qu)bit isn't a control by definition.
    num_qubits = len(bs1_little_endian)
    ctrls = list(range(0, num_qubits))
    if physical_control_qubits is not None:
        # Use the physical control qubits determined via a control pruning algorithm.
        unphysical_ctrls = []
        physical_ctrls = []
        for ctrl in ctrls:
            if ctrl not in physical_control_qubits or ctrl == target:
                unphysical_ctrls += [ctrl]
            else:
                physical_ctrls += [ctrl]
        ctrl_state_delete_list = unphysical_ctrls
    else:
        # Do not assume a physical set of control qubits from control pruning.
        ctrl_state_delete_list = [target]
        ctrls.remove(target)
        physical_ctrls = ctrls

    # Construct the bit string ctrl_state which triggers the MCRX gate.
    mask_bitstring = bs1_little_endian if bs1_little_endian[target] == '0' else bs2_little_endian
    if mask_bitstring[target] != '0':
        raise ValueError(
            "The two input bitstrings must differ at the target index.\n"
            f"bs1 = {bs1_little_endian}\n"
            f"bs2 = {bs2_little_endian}\n"
            f"target = {target}."
        )

    ctrl_state = ''.join([char for idx, char in enumerate(mask_bitstring) if idx not in ctrl_state_delete_list])
    # TODO: The following reversal cancels some other reversal of unknown
    # origin. This may indicate an endianness issue elsewhere in the codebase
    # to be figured out.
    ctrl_state = ctrl_state[::-1]

    # Assemble MCRX gate.
    # Note that the final qubit is the target.
    return (
        physical_ctrls,
        ctrl_state,
        RXGate(angle).control(
            num_ctrl_qubits=len(physical_ctrls), ctrl_state=ctrl_state)
    )


def _build_Xcirc(
        bs1_little_endian: str,
        bs2_little_endian: str,
        control: int) -> QuantumCircuit:
    """
    Build pre/post computation change-of-basis circuit in the Givens rotation.

    Assumes all bit strings are little-endian (that is, bit strings are read/indexed right-to-left).
    Applies CX gates conditioned on control, and all other bits where there's a bit flip
    between bs1 and bs2 as targets.
    """
    num_qubits = len(bs1_little_endian)
    Xcirc = QuantumCircuit(num_qubits)
    for idx in range(control+1, num_qubits):
        bit_flip_happens_at_idx = bs1_little_endian[idx] != \
            bs2_little_endian[idx]
        if bit_flip_happens_at_idx is True:
            Xcirc.cx(
                control_qubit=control,
                target_qubit=idx,
                ctrl_state="1")

    return Xcirc


# TODO: Clean up the implementation of givens2.
def givens2(
        strings: list, angle: float, reverse: bool = False) -> QuantumCircuit:
    """
    Build QuantumCircuit rotating two bit strings into each other by angle.

    Expects a nested list of pairs of bitstrings in physicist / big-endian notation,
    e.g. strings = [['10','01'],['00','11']], and a float as an angle.
    Optional reverse arg for little-endian.
    """
    if len(strings) != 2:
        raise ValueError('Need two pairs')

    for ii in range(2):
        for jj in range(2):
            if len(strings[ii][jj]) != len(strings[ii - 1][jj]) or len(strings[ii][jj]) != len(strings[ii][jj-1]):
                raise ValueError('All bitstrings must be same length')

    num_qubits = len(strings[0][0])

    strings_reversed = strings
    for ii in range(2):
        for jj in range(2):
            strings_reversed[ii][jj] = strings[ii][jj][::-1]

    pin = None
    for ii in range(num_qubits-1,-1,-1): # This finds where to put RY
        if strings_reversed[0][0][ii] != strings_reversed[0][1][ii] and strings_reversed[1][0][ii] != strings_reversed[1][1][ii]:
            pin = ii
            break
    if pin is None:
        raise ValueError('Your bitstrings lack the special sauce')

    lock = None
    type_2 = True
    for ii in range(num_qubits-1,-1,-1):
        property_1 = strings_reversed[0][0][ii] == strings_reversed[0][1][ii]
        property_2 = strings_reversed[1][0][ii] == strings_reversed[1][1][ii]
        property_3 = strings_reversed[0][0][ii] != strings_reversed[1][0][ii]
        if property_1 and property_2 and property_3:
            lock = ii
            type_2 = False
            break
    if type_2 is True:
        candidates = list(range(num_qubits-1,-1,-1))
        candidates.remove(pin)
        for ii in candidates:
            if strings_reversed[0][0][ii] != strings_reversed[0][1][ii] and strings_reversed[1][0][ii] != strings_reversed[1][1][ii]:
                lock = ii
                break

    if lock is None:
        raise ValueError('Your bitstrings lack the special sauce')

    bad1 = strings[0][0][pin] + strings[0][0][lock] == strings[1][0][pin] + strings[1][0][lock]
    bad2 = strings[0][0][pin] + strings[0][0][lock] == strings[1][1][pin] + strings[1][1][lock]
    if type_2 and (bad1 or bad2):
        raise ValueError('Your bitstrings lack the special sauce')

    R_ctrls = list(range(0, num_qubits))
    R_ctrls.remove(pin)
    R_ctrls.remove(lock)

    R_ctrl_state = strings[0][0]
    if pin < lock:
        R_ctrl_state = R_ctrl_state[:pin] + R_ctrl_state[(pin+1):lock] + R_ctrl_state[(lock+1):]
    else:
        R_ctrl_state = R_ctrl_state[:lock] + R_ctrl_state[(lock+1):pin] + R_ctrl_state[(pin+1):]

    R_ctrl_state = R_ctrl_state[::-1]

    R_circ = QuantumCircuit(num_qubits)
    if len(R_ctrls) == 0:
        R_circ.rx(angle,pin)
    else:
        R_circ.append(
            RXGate(angle).control(num_ctrl_qubits = num_qubits - 2, ctrl_state = R_ctrl_state),
            R_ctrls + [pin]
        )

    X_2_circ = QuantumCircuit(num_qubits)

    for ii in R_ctrls + [lock]:
        if strings[0][0][ii] != strings[0][1][ii]:
            X_2_circ.cx(pin, ii, ctrl_state = strings[0][1][pin])

    if strings[0][0][pin] == strings[1][0][pin]:
        string_3 = strings[1][0]
        string_4 = strings[1][1]
    else:
        string_3 = strings[1][1]
        string_4 = strings[1][0]

    X_3_circ = QuantumCircuit(num_qubits)

    for ii in R_ctrls:
        if strings[0][0][ii] != string_3[ii]:
            X_3_circ.mcx([lock] + [pin], ii, ctrl_state = string_3[pin] + string_3[lock])

    X_4_circ = QuantumCircuit(num_qubits)

    for ii in R_ctrls:
        if strings[0][1][ii] != string_4[ii]:
            X_4_circ.mcx([lock] + [pin], ii, ctrl_state = string_4[pin] + string_4[lock])

    X_3_circ.compose(X_4_circ, inplace=True)
    R_circ.compose(X_2_circ, inplace=True)
    X_2_circ.compose(R_circ, inplace=True)
    X_2_circ.compose(X_3_circ, inplace=True)
    X_3_circ.compose(X_2_circ, inplace=True)

    return X_3_circ


def _make_bitstring(length):
    # Helper for testing purposes.
    return ''.join(f'{int(random()>0.5)}' for _ in range(length))


def _test_givens():
    print("Testing givens.")

    test_circ = QuantumCircuit(2)
    test_circ.cx(control_qubit=1, target_qubit=0, ctrl_state="1")
    test_circ.append(
        RXGate(1).control(ctrl_state="1"),
        [0, 1]
    )
    test_circ.cx(control_qubit=1, target_qubit=0, ctrl_state="1")

    op_test = Operator(test_circ)
    op_given = givens('01', '10', 1)
    assert op_test.equiv(op_given), "Failed non-random test."
    print("Givens non-random test satisfied.")

    N = int(random()*5+2)
    str1 = _make_bitstring(N)
    str2 = _make_bitstring(N)
    print(f"First random string = {str1}")
    print(f"Second random string = {str2}")

    angle = random()
    print(f"Random angle = {angle}")

    actual_givens_circuit = givens(str1, str2, angle, reverse=True)

    actual_givens_circuit_as_operator = np.array(Operator(actual_givens_circuit))
    H = np.zeros((2**N, 2**N))
    H[int(str1, 2), int(str2, 2)] = 1
    H[int(str2, 2), int(str1, 2)] = 1
    expected_givens_operator = expm(-1j/2*angle*H)

    assert np.isclose(
        a=actual_givens_circuit_as_operator,
        b=expected_givens_operator
    ).all(), f"Failed random test. Constructed and expected givens operators not close. Largest difference = {np.max(actual_givens_circuit_as_operator-expected_givens_operator)}"
    print("Givens random test satisfied.")


def _test_givens2():
    print("Testing givens2.")

    N = int(random()*5+2)
    random_strings = [_make_bitstring(N) for _ in range(4)]
    strings = [[random_strings[0], random_strings[1]], [random_strings[2], random_strings[3]]]

    print(f"Strings are {strings}")

    angle = random()
    print(f"Random angle = {angle}")

    H = np.zeros((2**N, 2**N))
    H[int(strings[0][0], 2), int(strings[0][1], 2)] = 1
    H[int(strings[1][0], 2), int(strings[1][1], 2)] = 1

    H = H+H.transpose()

    expected_givens2_operator = expm(-1j/2*angle*H)
    givens2_circuit_as_operator = np.array(Operator(givens2(strings, angle)))

    assert np.isclose(
        a=givens2_circuit_as_operator,
        b=expected_givens2_operator
    ).all(), f"Failed random test. Constructed and expected givens operators not close. Largest difference = {np.max(givens2_circuit_as_operator-expected_givens2_operator)}"
    print("givens2 random test satisfied.")


def _test_Xcirc():
    print("Verifying that the diagonalization subcircuit is correctly constructed.")

    # Case 1
    bs1 = "1000101"
    bs2 = "1001110"
    control_qubit = 3
    print(f"Case 1: bs1 = {bs1}, bs2 = {bs2}, ctrl_qubit = {control_qubit}.")
    Xcirc_expected = QuantumCircuit(7)
    Xcirc_expected.cx(control_qubit=control_qubit, target_qubit=5)
    Xcirc_expected.cx(control_qubit=control_qubit, target_qubit=6)
    Xcirc = _build_Xcirc(bs1, bs2, control_qubit)

    assert Xcirc_expected == Xcirc, "Encountered inequivalent circuits. Expected:\n" \
        f"{Xcirc_expected.draw()}\nObtained:\n" \
        f"{Xcirc.draw()}"

    print(f"Test passed. Obtained circuit:\n{Xcirc.draw()}\n")

    # Case 2
    bs1 = "00000111111"
    bs2 = "10101111101"
    control_qubit = 0
    print(f"Case 2: bs1 = {bs1}, bs2 = {bs2}, ctrl_qubit = {control_qubit}.")
    Xcirc_expected = QuantumCircuit(11)
    Xcirc_expected.cx(control_qubit=control_qubit, target_qubit=2)
    Xcirc_expected.cx(control_qubit=control_qubit, target_qubit=4)
    Xcirc_expected.cx(control_qubit=control_qubit, target_qubit=9)
    Xcirc = _build_Xcirc(bs1, bs2, control_qubit)

    assert Xcirc_expected == Xcirc, "Encountered inequivalent circuits. Expected:\n" \
        f"{Xcirc_expected.draw()}\nObtained:\n" \
        f"{Xcirc.draw()}"

    print(f"Test passed. Obtained circuit:\n{Xcirc.draw()}\n")


def _test_multiRX():
    print("Verifying that the multiRX subcircuit is correctly constructed.")

    # Case 1
    bs1 = "1011"
    bs2 = "0001"
    control_qubit = 2
    angle = 0.5
    expected_ctrls = [0, 1, 3]
    expected_ctrl_state = "100"
    print(f"Case 1: bs1 = {bs1}, bs2 = {bs2}, ctrl_qubit = {control_qubit}, angle = {angle}.")
    print(f"Expected ctrl qubits = {expected_ctrls}, expected control state = {expected_ctrl_state}.")
    expected_circ = QuantumCircuit(4)
    multiRX_expected_gate = RXGate(angle).control(
        num_ctrl_qubits=len(expected_ctrls), ctrl_state=expected_ctrl_state)
    expected_circ.append(multiRX_expected_gate, expected_ctrls + [control_qubit])

    actual_circ = QuantumCircuit(4)
    actual_ctrls, actual_ctrl_state, multiRX_actual = _build_multiRX(bs1, bs2, angle, target=control_qubit, physical_control_qubits=None)
    actual_circ.append(multiRX_actual, actual_ctrls + [control_qubit])

    assert expected_circ == actual_circ, "Encountered inequivalent circuits. Expected:\n" \
        f"{expected_circ.draw()}\nObtained:\n" \
        f"{actual_circ.draw()}"

    print(f"Test passed. Obtained circuit:\n{actual_circ.draw()}\n")

    # Case 2
    bs1 = "001100010011"
    bs2 = "111100011011"
    control_qubit = 8
    angle = 0.5
    expected_ctrls = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]
    expected_ctrl_state = "11010001100"
    print(f"Case 2: bs1 = {bs1}, bs2 = {bs2}, ctrl_qubit = {control_qubit}, angle = {angle}.")
    print(f"Expected ctrl qubits = {expected_ctrls}, expected control state = {expected_ctrl_state}.")
    expected_circ = QuantumCircuit(12)
    multiRX_expected_gate = RXGate(angle).control(
        num_ctrl_qubits=len(expected_ctrls), ctrl_state=expected_ctrl_state)
    expected_circ.append(multiRX_expected_gate, expected_ctrls + [control_qubit])

    actual_circ = QuantumCircuit(12)
    actual_ctrls, actual_ctrl_state, multiRX_actual = _build_multiRX(bs1, bs2, angle, target=control_qubit, physical_control_qubits=None)
    actual_circ.append(multiRX_actual, actual_ctrls + [control_qubit])

    assert expected_circ == actual_circ, "Encountered inequivalent circuits. Expected:\n" \
        f"{expected_circ.draw()}\nObtained:\n" \
        f"{actual_circ.draw()}"

    print(f"Test passed. Obtained circuit:\n{actual_circ.draw()}\n")


if __name__ == "__main__":
    _test_givens()
    #_test_givens2()  # TODO fix nondeterministic behavior of this test.
    _test_Xcirc()
    _test_multiRX()
    print("All tests passed.")
