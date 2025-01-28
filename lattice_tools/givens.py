"""
This module contains methods for constructing Givens rotation circuits.

Original author: Cianan Conefrey-Shinozaki, Oct 2024.

A Givens rotation is a generalization of an X-rotation; it rotates
two multi-qubit states into each other.
"""
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RXGate
from qiskit.quantum_info import Operator
from random import random
import numpy as np
from scipy.linalg import expm
from typing import Tuple, List


def givens(bit_string_1: str, bit_string_2: str, angle: float, reverse: bool = False) -> QuantumCircuit:
    """
    Build QuantumCircuit rotating two bit strings into each other by angle.

    Expects two bitstrings in physicist / big-endian notation,
    e.g. "1001" or "0100", and a float as an angle.
    Optional reverse arg for outputting a circuit in little-endian convention
    (used by qiskit).

    The resulting circuit sandwiches a multi-control X rotation (MCRX) between
    a pre/post computation circuit consisting of CX gates. The pre/post
    computation is essentially just a change of basis for the MCRX.
    """
    # Input validation and sanity checks.
    if len(bit_string_1) != len(bit_string_2):
        raise ValueError('Bit strings must be the same length.')
    no_rotation_is_needed = bit_string_1 == bit_string_2
    if no_rotation_is_needed:
        return QuantumCircuit(len(bit_string_1))  # The identity circuit.

    # Collect data needed to construct rotation circuit, and initialize.
    bit_string_1_little_endian = bit_string_1[::-1]
    bit_string_2_little_endian = bit_string_2[::-1]
    num_qubits = len(bit_string_1)
    circ = QuantumCircuit(num_qubits)

    # Build the circuit.
    if num_qubits == 1:
        # No pre/post computation needed.
        circ.rx(angle, 0)
        return circ
    else:
        # Find target for the rotation gate, which is the
        # highest index at which the little-endian bit strings differ.
        for idx in range(num_qubits-1, -1, -1):
            current_idx_is_target_idx = bit_string_1_little_endian[idx] != \
                bit_string_2_little_endian[idx]
            if current_idx_is_target_idx is True:
                target = idx
                break

        # Configure the multi-controlled X rotation (MCRX) gate.
        ctrls, ctrl_state, multiRX = _build_multiRX(
            bit_string_1_little_endian,
            bit_string_2_little_endian,
            angle,
            target)

        # Construct the pre- and post-MCRX circuits.
        Xcirc = _build_Xcirc(
            bit_string_1_little_endian, bit_string_2_little_endian, target)

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
        target: int) -> Tuple[List[int], str, QuantumCircuit]:
    """
    Build the multi-control RX gate (MCRX) in Givens rotations.

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
    ctrls.remove(target)

    # Construct the bit string ctrl_state which triggers the MCRX gate.
    ctrl_state = bs1_little_endian[:target] + bs1_little_endian[(target + 1):]
    # TODO: The following reversal cancels some other reversal of unknown
    # origin. This may indicate an endianness issue elsewhere in the codebase
    # to be figured out.
    ctrl_state = ctrl_state[::-1]

    # Assemble MCRX gate.
    # Note that the final qubit is the target.
    return (
        ctrls,
        ctrl_state,
        RXGate(angle).control(
            num_ctrl_qubits=len(ctrls), ctrl_state=ctrl_state)
    )


def _build_Xcirc(
        bs1_little_endian: str,
        bs2_little_endian: str,
        target: int) -> QuantumCircuit:
    """
    Build pre/post computation change-of-basis circuit in the Givens rotation.

    Assumes all bit strings are little-endian.
    """
    num_qubits = len(bs1_little_endian)
    ctrl_val = 0 if bs1_little_endian[target] == "1" else 1
    Xcirc = QuantumCircuit(num_qubits)
    for idx in range(target-1, -1, -1):
        X_gate_needed_at_idx = bs1_little_endian[idx] != \
            bs2_little_endian[idx]
        if X_gate_needed_at_idx is True:
            Xcirc.cx(
                control_qubit=target,
                target_qubit=idx,
                ctrl_state=ctrl_val)

    return Xcirc


# TODO: Clean up the implementation of givens2.
def givens2(
        strings: list, angle: float, reverse: bool = False) -> QuantumCircuit:
    """
    Build QuantumCircuit rotating two bit strings into each other by angle.

    Expects a nested list of pairs of bitstrings in physicist / big-endian notation,
    e.g. strings = [['10','01'],['00','11']], and a float as an angle.
    Optional reverse arg for little-endian.

    Note that ValueErrors are raised under the following circumstances:
      - Unequal string lengths are encountered.
      - There is no suitable "lock"/"pin" found during the algorithm.
      - The string pairs fail various "goodness" checks for the validity of the method.
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
        raise ValueError('Unable to find a pin.')

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
        raise ValueError('Unable to find a lock.')

    bad1 = strings[0][0][pin] + strings[0][0][lock] == strings[1][0][pin] + strings[1][0][lock]
    bad2 = strings[0][0][pin] + strings[0][0][lock] == strings[1][1][pin] + strings[1][1][lock]
    if type_2 and (bad1 or bad2):
        raise ValueError('Bit string pairs failed "goodness" check.')

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

    givens_circuit_as_operator = np.array(Operator(givens(str1, str2, angle)))
    H = np.zeros((2**N, 2**N))
    H[int(str1, 2), int(str2, 2)] = 1
    H[int(str2, 2), int(str1, 2)] = 1
    expected_givens_operator = expm(-1j/2*angle*H)

    assert np.isclose(
        a=givens_circuit_as_operator,
        b=expected_givens_operator
    ).all(), f"Failed test. Constructed and expected givens operators not close. Largest difference = {np.max(givens_circuit_as_operator-expected_givens_operator)}"
    print("Givens test passed.")


def _test_givens2():
    print("Testing givens2.")

    N = 6
    valid_givens2_strings = ['101100','001011','111011','110001']
    strings = [[valid_givens2_strings[0],valid_givens2_strings[1]],[valid_givens2_strings[2],valid_givens2_strings[3]]]
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
    ).all(), f"Failed test. Constructed and expected givens operators not close. Largest difference = {np.max(givens2_circuit_as_operator-expected_givens2_operator)}"
    print("givens2 test passed.")

def _test_givens_value_errors():
    print("Testing that givens raises ValueError for bad input.")

    str_1 = "110"
    str_2 = "0011"

    try:
        givens(str_1, str_2, 0.1)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("ValueError not raised.")

# TODO write some givens2 tests for the raising of ValueError.
    
if __name__ == "__main__":
    _test_givens()
    _test_givens2()
    _test_givens_value_errors()
