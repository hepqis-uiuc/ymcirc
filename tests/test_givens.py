from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from random import random
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RXGate
from qiskit.quantum_info import Operator
from scipy.linalg import expm
from ymcirc.givens import (
    givens, givens2, compute_LP_family, LPFamily, _build_Xcirc,
    _compute_ctrls_and_state_for_givens_MCRX, _CRXGate,
    _apply_LP_family_to_bit_string, prune_controls,
    _eliminate_phys_states_that_differ_from_rep_at_Q_idx,
    fuse_controls, gray_to_index, givens_fused_controls, bitstring_value_of_LP_family,
    LPOperator
)


class BasicTests:
    @staticmethod
    def helper_make_random_bitstring(length):
        """Helper for testing purposes."""
        return "".join(f"{int(random()>0.5)}" for _ in range(length))

    def test_givens():
        """Testing the givens method."""

        test_circ = QuantumCircuit(2)
        test_circ.cx(control_qubit=1, target_qubit=0, ctrl_state="1")
        test_circ.append(RXGate(1).control(ctrl_state="1"), [0, 1])
        test_circ.cx(control_qubit=1, target_qubit=0, ctrl_state="1")

        op_test = Operator(test_circ)
        op_given = givens("01", "10", 1)
        assert op_test.equiv(op_given), "Failed non-random test."
        print("Givens non-random test satisfied.")

        N = int(random() * 5 + 2)
        str1 = BasicTests.helper_make_random_bitstring()(N)
        str2 = BasicTests.helper_make_random_bitstring()(N)
        print(f"First random string = {str1}")
        print(f"Second random string = {str2}")

        angle = random()
        print(f"Random angle = {angle}")

        actual_givens_circuit = givens(str1, str2, angle, reverse=True)

        actual_givens_circuit_as_operator = np.array(Operator(actual_givens_circuit))
        H = np.zeros((2**N, 2**N))
        H[int(str1, 2), int(str2, 2)] = 1
        H[int(str2, 2), int(str1, 2)] = 1
        expected_givens_operator = expm(-1j / 2 * angle * H)

        assert np.isclose(
            a=actual_givens_circuit_as_operator, b=expected_givens_operator
        ).all(), f"Failed random test. Constructed and expected givens operators not close. Largest difference = {np.max(actual_givens_circuit_as_operator-expected_givens_operator)}"
        print("Givens random test satisfied.")

    # TODO: Fix nondeterministic behavior in this test
    def test_givens2():
        """Testing the givens2 method."""

        N = int(random() * 5 + 2)
        random_strings = [BasicTests.helper_make_random_bitstring()(N) for _ in range(4)]
        strings = [
            [random_strings[0], random_strings[1]],
            [random_strings[2], random_strings[3]],
        ]

        print(f"Strings are {strings}")

        angle = random()
        print(f"Random angle = {angle}")

        H = np.zeros((2**N, 2**N))
        H[int(strings[0][0], 2), int(strings[0][1], 2)] = 1
        H[int(strings[1][0], 2), int(strings[1][1], 2)] = 1

        H = H + H.transpose()

        expected_givens2_operator = expm(-1j / 2 * angle * H)
        givens2_circuit_as_operator = np.array(Operator(givens2(strings, angle)))

        assert np.isclose(
            a=givens2_circuit_as_operator, b=expected_givens2_operator
        ).all(), f"Failed random test. Constructed and expected givens operators not close. Largest difference = {np.max(givens2_circuit_as_operator-expected_givens2_operator)}"
        print("givens2 random test satisfied.")


def test_Xcirc():
    print("Verifying that the diagonalization subcircuit is correctly constructed.")

    # Case 1
    bs1 = "1000101"
    bs2 = "1001110"
    control_qubit = 3
    print(f"Case 1: bs1 = {bs1}, bs2 = {bs2}, ctrl_qubit = {control_qubit}.")
    Xcirc_expected = QuantumCircuit(7)
    Xcirc_expected.cx(control_qubit=control_qubit, target_qubit=5)
    Xcirc_expected.cx(control_qubit=control_qubit, target_qubit=6)
    lp_fam = compute_LP_family(bs1, bs2)
    Xcirc = _build_Xcirc(lp_fam, control_qubit)

    assert Xcirc_expected == Xcirc, (
        "Encountered inequivalent circuits. Expected:\n"
        f"{Xcirc_expected.draw()}\nObtained:\n"
        f"{Xcirc.draw()}"
    )

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
    lp_fam = compute_LP_family(bs1, bs2)
    Xcirc = _build_Xcirc(lp_fam, control_qubit)

    assert Xcirc_expected == Xcirc, (
        "Encountered inequivalent circuits. Expected:\n"
        f"{Xcirc_expected.draw()}\nObtained:\n"
        f"{Xcirc.draw()}"
    )

    print(f"Test passed. Obtained circuit:\n{Xcirc.draw()}\n")


def test_building_MCRX_gate():
    print("Verifying that the multiRX subcircuit is correctly constructed.")

    # Case 1
    bs1 = "1011"
    bs2 = "0001"
    control_qubit = 2
    angle = 0.5
    expected_ctrls = [0, 1, 3]
    expected_ctrl_state = "100"
    print(
        f"Case 1: bs1 = {bs1}, bs2 = {bs2}, ctrl_qubit = {control_qubit}, angle = {angle}."
    )
    print(
        f"Expected ctrl qubits = {expected_ctrls}, expected control state = {expected_ctrl_state}."
    )
    expected_circ = QuantumCircuit(4)
    multiRX_expected_gate = RXGate(angle).control(
        num_ctrl_qubits=len(expected_ctrls), ctrl_state=expected_ctrl_state
    )
    expected_circ.append(multiRX_expected_gate, expected_ctrls + [control_qubit])

    actual_circ = QuantumCircuit(4)
    actual_ctrls, actual_ctrl_state = _compute_ctrls_and_state_for_givens_MCRX(
        bs1, bs2, target=control_qubit
    )
    multiRX_actual = _CRXGate(len(actual_ctrls), actual_ctrl_state, angle)
    actual_circ.append(multiRX_actual, actual_ctrls + [control_qubit])

    assert expected_circ == actual_circ, (
        "Encountered inequivalent circuits. Expected:\n"
        f"{expected_circ.draw()}\nObtained:\n"
        f"{actual_circ.draw()}"
    )

    print(f"Test passed. Obtained circuit:\n{actual_circ.draw()}\n")

    # Case 2
    bs1 = "001100010011"
    bs2 = "111100011011"
    control_qubit = 8
    angle = 0.5
    expected_ctrls = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]
    expected_ctrl_state = "11010001100"
    print(
        f"Case 2: bs1 = {bs1}, bs2 = {bs2}, ctrl_qubit = {control_qubit}, angle = {angle}."
    )
    print(
        f"Expected ctrl qubits = {expected_ctrls}, expected control state = {expected_ctrl_state}."
    )
    expected_circ = QuantumCircuit(12)
    multiRX_expected_gate = RXGate(angle).control(
        num_ctrl_qubits=len(expected_ctrls), ctrl_state=expected_ctrl_state
    )
    expected_circ.append(multiRX_expected_gate, expected_ctrls + [control_qubit])

    actual_circ = QuantumCircuit(12)
    actual_ctrls, actual_ctrl_state = _compute_ctrls_and_state_for_givens_MCRX(
        bs1, bs2, target=control_qubit
    )
    multiRX_actual = _CRXGate(len(actual_ctrls), actual_ctrl_state, angle)
    actual_circ.append(multiRX_actual, actual_ctrls + [control_qubit])

    assert expected_circ == actual_circ, (
        "Encountered inequivalent circuits. Expected:\n"
        f"{expected_circ.draw()}\nObtained:\n"
        f"{actual_circ.draw()}"
    )

    print(f"Test passed. Obtained circuit:\n{actual_circ.draw()}\n")


def test_compute_LP_family():
    """Check computation of LP family works."""
    bs1 = "11000"
    bs2 = "10001"
    expected_LP_family = LPFamily(("P", "L", "P", "P", "L"))
    print(
        f"Checking that the bit strings {bs1} and {bs2} "
        f"yield the LP family {expected_LP_family}."
    )

    assert (
        compute_LP_family(bs1, bs2) == expected_LP_family
    ), f"Unexpected LP family result: {compute_LP_family(bs1, bs2)}"
    print("Test passed.\n")


def test_compute_LP_family_fails_on_bad_input():
    """LP family computation should fail for unequal length strings."""
    bs1 = "11110"
    bs2 = "00"
    print(f"Checking that computing LP family of {bs1} and {bs2} raises IndexError.")

    with pytest.raises(IndexError) as e_info:
        compute_LP_family(bs1, bs2)


def test_compute_LP_family_fails_for_non_bitstrings():
    """LP family computation fails for input not bit strings."""
    bs1 = "a3bb"
    bs2 = "1011"
    print(f"Checking that expected_LP_family({bs1}, {bs2}) raises a ValueError.")

    with pytest.raises(ValueError) as e_info:
        compute_LP_family(bs1, bs2)


def test_prune_controls_acts_as_expected():
    """Check the output of prune_controls for good input."""
    # Case 1, one control needed.
    bs1 = "11001001"
    bs2 = "11000110"
    ctrls, ctrl_state = _compute_ctrls_and_state_for_givens_MCRX(bs1, bs2, 4)
    phys_states = {bs1, bs2, "11111111"}
    expected_rep_LP_family = LPFamily(("P", "P", "P", "P", "L", "L", "L", "L"))
    expected_q_prime_idx = 4
    expected_bs1_tilde = "11001110"
    expected_bs2_tilde = bs2  # No change since control is 0-valued.
    expected_P_tilde = {
        bs1: expected_bs1_tilde,
        bs2: expected_bs2_tilde,
        "11111111": "11111000",
    }
    # Elimination rounds: none, none, 2356 -> add 2 to Q, eliminate all but bs1 tilde and bs2 tilde
    expected_Q_set = {2}
    pruned_idx = ctrls.index(2)
    expected_pruned_ctrls = [2]
    expected_pruned_ctrl_state = ctrl_state[pruned_idx]

    print(
        "Testing the following prune_controls case:\n"
        f"bs1 = {bs1} (as representative)\n"
        f"bs2 = {bs2}\n"
        f"phys_states = {phys_states}\n"
        f"expected_Q_set = {expected_Q_set}"
    )

    # Check computations at various stages in the algorithm give expected results.
    result_rep_LP_family = compute_LP_family(bs1, bs2)
    assert (
        result_rep_LP_family == expected_rep_LP_family
    ), f"LP family = {compute_LP_family(bs1, bs2)}"
    for pre_LP_state, expected_post_LP_state in expected_P_tilde.items():
        result_post_LP_state = _apply_LP_family_to_bit_string(
            expected_rep_LP_family, expected_q_prime_idx, pre_LP_state
        )
        assert (
            result_post_LP_state == expected_post_LP_state
        ), f"expected tilde state {expected_post_LP_state} != actual tilde state {result_post_LP_state}"

    # Check the return value from the algorithm.
    result_pruned_ctrls, result_pruned_ctrl_state = prune_controls(
        expected_rep_LP_family, ctrls, ctrl_state, phys_states
    )
    assert (
        result_pruned_ctrls == expected_pruned_ctrls
    ), f"result pruned_ctrls = {result_pruned_ctrls}"
    assert (
        result_pruned_ctrl_state == expected_pruned_ctrl_state
    ), f"result pruned_ctrl_state = {result_pruned_ctrl_state}"
    print("Test passed.")

    # Case 2, two controls needed.
    bs1 = "11000011"
    bs2 = "11001001"  # Differs at 1 qubit at index 6, will be in Q set.
    ctrls, ctrl_state = _compute_ctrls_and_state_for_givens_MCRX(bs1, bs2, 4)
    phys_states = {bs1, bs2, "11111111", "00000000"}
    expected_rep_LP_family = LPFamily(("P", "P", "P", "P", "L", "P", "L", "P"))
    expected_q_prime_idx = 4
    expected_bs1_tilde = bs1  # No change since control is 0-valude
    expected_bs2_tilde = "11001011"
    expected_P_tilde = {
        bs1: expected_bs1_tilde,
        bs2: expected_bs2_tilde,
        "11111111": "11111101",
        "00000000": "00000000",
    }
    # Elimination round 1: none, 6, 2356, 0167 -> add 6 to Q, eliminate all but bs1 tilde, bs2_tilde
    expected_Q_set = {6}
    pruned_idx = ctrls.index(6)
    expected_pruned_ctrls = [6]
    expected_pruned_ctrl_state = ctrl_state[pruned_idx]

    print(
        "Testing the following prune_controls case:\n"
        f"bs1 = {bs1} (as representative)\n"
        f"bs2 = {bs2}\n"
        f"phys_states = {phys_states}\n"
        f"expected_Q_set = {expected_Q_set}"
    )

    # Check computations at various stages in the algorithm give expected results.
    result_rep_LP_family = compute_LP_family(bs1, bs2)
    assert (
        result_rep_LP_family == expected_rep_LP_family
    ), f"Expected: {expected_rep_LP_family}, Result: {result_rep_LP_family}"
    for pre_LP_state, expected_post_LP_state in expected_P_tilde.items():
        result_post_LP_state = _apply_LP_family_to_bit_string(
            expected_rep_LP_family, expected_q_prime_idx, pre_LP_state
        )
        assert (
            result_post_LP_state == expected_post_LP_state
        ), f"expected tilde state {expected_post_LP_state} != actual tilde state {result_post_LP_state}"

    # Check the return value from the algorithm.
    result_pruned_ctrls, result_pruned_ctrl_state = prune_controls(
        expected_rep_LP_family, ctrls, ctrl_state, phys_states
    )
    assert (
        result_pruned_ctrls == expected_pruned_ctrls
    ), f"result pruned_ctrls = {result_pruned_ctrls}"
    assert (
        result_pruned_ctrl_state == expected_pruned_ctrl_state
    ), f"result pruned_ctrl_state = {result_pruned_ctrl_state}"
    print("Test passed.")


def test_apply_LP_family():
    lp_family = ("L", "L", "L", "P", "L", "P")
    q_prime_idx = 0  # The first 'L' in the LP family list.
    print(
        f"Checking that application of LP family diagonalizing 'circuit' defined by {lp_family} to bit strings works."
    )

    # Case 1
    bs_with_active_control = "110001"
    expected_result_bs_with_active_control = "101011"
    print(
        f"Checking that bit string {bs_with_active_control} becomes {expected_result_bs_with_active_control} when applying LP family."
    )

    actual_result_active_control = _apply_LP_family_to_bit_string(
        lp_family, q_prime_idx, bs_with_active_control
    )
    assert (
        actual_result_active_control == expected_result_bs_with_active_control
    ), f"Unexpected result: {actual_result_active_control}."
    print("Test passed.")

    # Case 2
    bs_with_inactive_control = "011101"
    expected_result_bs_with_inactive_control = bs_with_inactive_control
    print(
        f"Checking that bit string {bs_with_inactive_control} becomes {expected_result_bs_with_inactive_control} when applying LP family."
    )

    actual_result_inactive_control = _apply_LP_family_to_bit_string(
        lp_family, q_prime_idx, bs_with_inactive_control
    )
    assert (
        actual_result_inactive_control == expected_result_bs_with_inactive_control
    ), f"Unexpected result: {actual_result_inactive_control}."
    print("Test passed.")


def test_eliminate_phys_states_that_differ_from_rep_at_Q_idx():
    rep = "10001"
    phys_states = {rep, "10111", "01001", "00011", "10000", "01011"}
    Q_set = {2, 4}
    expected_phys_states = {rep, "01001", "00011", "01011"}
    print(
        f"Checking that eliminating bitstrings from a set with qubit index set Q = {Q_set} works."
    )
    print(f"Representative = {rep}.")
    print(f"Initial phys states = {phys_states}")
    print(f"Expected phys states = {expected_phys_states}")

    actual_phys_states = _eliminate_phys_states_that_differ_from_rep_at_Q_idx(
        rep, phys_states, Q_set
    )
    assert (
        actual_phys_states == expected_phys_states
    ), f"Unexpected result set: {actual_phys_states}."

    print("Test passed.")


def test_control_fusion_fails_for_inconsistent_lp_bin():
    bin_value = "10110101"
    bitstring_list = [("0000", "1111", 1.0)]
    with pytest.raises(ValueError) as e_info:
        fuse_controls(bin_value, bitstring_list)


def test_control_fusion():
    bitstring_list = [
        ("00000000", "01001010", 1.0),
        ("10001000", "11000010", 1.0),
        ("01100000", "00101010", 1.0),
        ("00001000", "01000010", 1.0),
    ]
    expected_control_fused_gates = [
        ([0, 2, 3, 5, 6, 7], "000000"),
        ([0, 2, 3, 4, 5, 6, 7], "0101010"),
        ([0, 2, 3, 4, 5, 6, 7], "1001000"),
    ]
    bin_value = LPFamily(("P", "L", "P", "P", "L", "P", "L", "P"))
    code_generated_fusion = fuse_controls(bin_value, bitstring_list)
    assert code_generated_fusion[1.0] == expected_control_fused_gates
    print("Control fusion test passed.")


def test_gray_to_index():
    bitstring_to_be_gray_ordered = [
        "00000",
        "00001",
        "00010",
        "00011",
        "00100",
        "00101",
        "00110",
        "00111",
        "01000",
        "01001",
        "01010",
        "01011",
        "01100",
        "01101",
        "01110",
        "01111",
        "10000",
        "10001",
        "10010",
        "10011",
        "10100",
        "10101",
        "10110",
        "10111",
        "11000",
        "11001",
        "11010",
        "11011",
        "11100",
        "11101",
        "11110",
        "11111",
    ]
    expected_gray_order = [
        "00000",
        "00001",
        "00011",
        "00010",
        "00110",
        "00111",
        "00101",
        "00100",
        "01100",
        "01101",
        "01111",
        "01110",
        "01010",
        "01011",
        "01001",
        "01000",
        "11000",
        "11001",
        "11011",
        "11010",
        "11110",
        "11111",
        "11101",
        "11100",
        "10100",
        "10101",
        "10111",
        "10110",
        "10010",
        "10011",
        "10001",
        "10000",
    ]
    # expected_gray_code_order = ["0000", "0001", "0011", "0010","0110", "0101","0100", "1000"]
    code_generated_gray_order = sorted(
        bitstring_to_be_gray_ordered, key=lambda x: gray_to_index(x)
    )
    # print(code_generated_gray_order)
    assert expected_gray_order == code_generated_gray_order
    print("Binary to Gray Order passed.")


def test_gray_to_index_for_larger_bitstrings():
    bitstrings_to_be_gray_ordered = [
        "11000000000000000001",
        "11000000000000000010",
        "11000000000000000111",
        "11000000000000000100",
        "10000000000000000000",
        "11000000000000000000",
        "11000000000000000011",
        "11000000000000000110",
        "11000000000000000101",
        "11000000000000001100",
    ]
    expected_gray_order = [
        "11000000000000000000",
        "11000000000000000001",
        "11000000000000000011",
        "11000000000000000010",
        "11000000000000000110",
        "11000000000000000111",
        "11000000000000000101",
        "11000000000000000100",
        "11000000000000001100",
        "10000000000000000000",
    ]
    code_generated_gray_order = sorted(
        bitstrings_to_be_gray_ordered, key=lambda x: gray_to_index(x)
    )
    assert expected_gray_order == code_generated_gray_order
    print("Gray to index for bitstrings of length 20 passed.")


def test_givens_fused_controls():
    bitstring_list = [
        ("00000000", "01001010", 1.0),
        ("00001000", "01000010", 1.0),
        ("01101010", "00100000", 1.0),
        ("00101000", "01100010", 1.0),
    ]
    bin_value = LPFamily(("P", "L", "P", "P", "L", "P", "L", "P"))
    encoded_physical_states = {
        "00000000",
        "01001010",
        "00001000",
        "01000010",
        "01101010",
        "00100000",
        "00101000",
        "01100010",
        "00110000",
    }
    generated_circuit = givens_fused_controls(
        bitstring_list, bin_value, encoded_physical_states=encoded_physical_states
    )
    expected_circuit = QuantumCircuit(8)
    expected_circuit.cx(1, 4)
    expected_circuit.cx(1, 6)
    expected_circuit.append(
        RXGate(1.0).control(num_ctrl_qubits=1, ctrl_state="0"),
        [3, 1],
    )
    expected_circuit.cx(1, 4)
    expected_circuit.cx(1, 6)
    assert expected_circuit == generated_circuit
    print("givens with fused controls test passed.")


def test_bitstring_value_of_LP_fam():
    lp_fam = ("L", "P", "L", "L", "P", "P")
    expected_bitstring = "010011"
    assert expected_bitstring == bitstring_value_of_LP_family(lp_fam)
    print("bitstring_value_of_LP_fam passed.")


def test_LPOperator_works():
    print(
        f"Checking that {LPOperator.__name__} only takes the correct values, "
        "raises a ValueError with invalid values, and is immutable."
    )
    correct_vals = ("L", "L", "P", "P")
    incorrect_val = "Q"
    for val in correct_vals:
        print("Checking creation with value:", val)
        assert LPOperator(val).value == val
        print("Test passed.")

    print(f"Check that {incorrect_val} causes a ValueError.")
    with pytest.raises(ValueError) as e_info:
        LPOperator(incorrect_val)

    print("Check that LPOperator is immutable.")
    lp_op = LPOperator("L")
    with pytest.raises(FrozenInstanceError) as e_info:
        lp_op.value = "P"


def test_LPFamily_works():
    print(
        f"Checking that {LPFamily.__name__} only takes correct values, raises a ValueError with invalid values, and is immutable."
    )

    # Test creation with list of strings.
    lp_family_str = ("L", "L", "P")
    print(f"Testing creation of LPFamily with list of valid strings: {lp_family_str}.")
    assert (
        LPFamily(lp_family_str).value == lp_family_str
    ), f"Encountered value: {LPFamily(lp_family_str).value}, expected: {lp_family_str}."
    print(f"Test passed, LPFamily.value == {lp_family_str}.")

    # Test that creation with bad strings fails.
    bad_lp_family = ("L", "Q")
    print(f"Checking {bad_lp_family} causes ValueError.")
    with pytest.raises(ValueError) as e_info:
        LPFamily(bad_lp_family)

    # Test immutability.
    print("Checking that LPFamily instances are immutable.")
    lp_fam = LPFamily(lp_family_str)
    with pytest.raises(FrozenInstanceError) as e_info:
        lp_fam.value = LPFamily(("P", "L"))
