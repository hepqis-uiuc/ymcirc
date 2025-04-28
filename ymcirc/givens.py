"""
This module contains methods and classes for constructing Givens rotation circuits.

Original author: Cianan Conefrey-Shinozaki, Oct 2024.

A Givens rotation is a generalization of an X-rotation; it rotates
two multi-qubit states into each other.
"""

from __future__ import annotations
from dataclasses import dataclass, FrozenInstanceError
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.circuit import ControlledGate
from qiskit.circuit.library.standard_gates import RXGate, RZGate, RYGate, MCXGate
from qiskit.quantum_info import Operator, Statevector
from random import random
from math import isclose
import numpy as np
import copy
from scipy.linalg import expm
from typing import Tuple, List, Set, Dict, Union


@dataclass(frozen=True)
class LPOperator:
    """
    Immutable wrapper for string reps of ladder/projector operators.

    The following values are allowed:
      - "P": projector onto "0" or "1" state.
      - "L": "raising" or "lowering" ladder operator.
    """

    value: str

    def __post_init__(self):
        valid_vals = ["P", "L"]
        if self.value not in valid_vals:
            raise ValueError(
                f"{LPOperator.__name__} value must be one of the following: {valid_vals}. Encountered: '{self.value}'."
            )


@dataclass(frozen=True)
class LPFamily:
    """
    Immutable wrapper for working with LP families.

    The LPOperator class is used for validation, but data remains a tuple of strings.
    """

    value: Tuple[str]

    def __post_init__(self):
        # Validate.
        if not isinstance(self.value, tuple):
            raise ValueError(f"Input {self.value} not a tuple.")
        for op_val in self.value:
            try:
                LPOperator(op_val)
            except ValueError:
                raise ValueError(
                    f"Encounterd invalid {LPOperator.__name__} value '{op_val}'."
                )


def givens(
    bit_string_1: str,
    bit_string_2: str,
    angle: float,
    encoded_physical_states: Set[str] | None = None,
    reverse: bool = False,
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
        raise ValueError("Bit strings must be the same length.")
    no_rotation_is_needed = bit_string_1 == bit_string_2
    if no_rotation_is_needed:
        return QuantumCircuit(len(bit_string_1))  # The identity circuit.

    num_qubits = len(bit_string_1)

    # Build the circuit.
    if num_qubits == 1:
        # No pre/post computation needed.
        circ.rx(angle, 0)
        return circ
    else:
        for idx in range(num_qubits):
            current_idx_is_target_idx = bit_string_1[idx] != bit_string_2[idx]
            if current_idx_is_target_idx is True:
                target = idx
                break

        ctrls, ctrl_state = _compute_ctrls_and_state_for_givens_MCRX(
            bit_string_1, bit_string_2, target
        )
        lp_fam = compute_LP_family(bit_string_1, bit_string_2)

        # Note that this step will become redundant when control_pruning is turned off
        # i.e., when encoded_physical_states = None.
        pruned_ctrls, pruned_ctrl_state = prune_controls(
            lp_fam, ctrls, ctrl_state, encoded_physical_states
        )
        Xcirc = _build_Xcirc(lp_fam, control=target)

        circ = _CRXCircuit_with_MCX([pruned_ctrl_state, pruned_ctrls], 
            angle, target, num_qubits)

        # Assemble the final circuit.
        # Using inplace speeds up circuit composition.
        circ.compose(Xcirc, inplace=True)
        Xcirc.compose(circ, inplace=True)
        if reverse is True:
            Xcirc = Xcirc.reverse_bits()
        return Xcirc


def givens_fused_controls(
    lp_bin_w_angle: List[(str, str, float)],
    lp_bin: LPFamily,
    encoded_physical_states: Set[str] | None,
    reverse: bool = False,
) -> QuantumCircuit:
    """
    Implements givens rotation using the same logic as the givens function but for fused multiRX's.

    Inputs:
        - lp_bin_w_angle: list of bitstrings of the same LP family and the angle they have to be rotated by.
        - lp_bin: the LP family the bitstrings in the bin belong to.
        - reverse: optional argument to deal with endianess issues

    Output:
        QuantumCircuit object that has the necessary givens rotation.
    """

    num_qubits = len(lp_bin_w_angle[0][0])

    # Input validation and sanity checks.
    for bit_string_1, bit_string_2, angle in lp_bin_w_angle:
        if len(bit_string_1) != len(bit_string_2):
            raise ValueError("Bit strings must be the same length.")
        bs1_bs2_lp_family = compute_LP_family(bit_string_1, bit_string_2)
        lp_bin_matches = bs1_bs2_lp_family.value == lp_bin.value
        if lp_bin_matches is False:
            return ValueError(
                "The LP value of the bitstrings should match with the LP bin value."
            )
        no_rotation_is_needed = bit_string_1 == bit_string_2
        if no_rotation_is_needed:
            return QuantumCircuit(len(bit_string_1))  # The identity circuit.

    circ = QuantumCircuit(num_qubits)

    # Build the circuit.
    if num_qubits == 1:
        # No pre/post computation needed.
        circ.rx(angle, 0)
        return circ
    else:
        for idx in range(num_qubits):
            current_idx_is_target_idx = lp_bin.value[idx] == "L"
            if current_idx_is_target_idx is True:
                target = idx
                break
        # First, fuse controls.
        angle_dict = fuse_controls(lp_bin, lp_bin_w_angle, round_close_angles=True)
        # Now, prune controls.
        # Note that this step will become redundant when control_pruning is turned off
        # i.e., when encoded_physical_states = None.
        for angle, ctrl_list in angle_dict.items():
            for ctrls, ctrl_state in ctrl_list:
                pruned_ctrls, pruned_ctrl_state = prune_controls(
                    lp_bin, ctrls, ctrl_state, encoded_physical_states
                )
                crxcircuit = _CRXCircuit_with_MCX([pruned_ctrl_state, pruned_ctrls], 
            angle, target, num_qubits)
                circ.compose(crxcircuit, inplace=True)
        Xcirc = _build_Xcirc(lp_bin, control=target)

        # Assemble the final circuit.
        # Using inplace speeds up circuit composition.
        circ.compose(Xcirc, inplace=True)
        Xcirc.compose(circ, inplace=True)
        if reverse is True:
            Xcirc = Xcirc.reverse_bits()

        return Xcirc


def _compute_ctrls_and_state_for_givens_MCRX(
    bs1_little_endian: str, bs2_little_endian: str, target: int
) -> Tuple[List[int], str]:
    """

    Find the controls and control state of the multi-control RX gate involved in
    Givens rotations.

    Return:
      - (ctrls, ctrl_state): a tuple consisting of control qubit
        indices, and the bit string state of those those controls which triggers
        the rotation gate.

    Note that the final qubit in multiRX is the target qubit.
    """
    # Determine the set of control qubits.
    # The target (qu)bit isn't a control by definition.
    num_qubits = len(bs1_little_endian)
    ctrls = list(range(0, num_qubits))
    ctrl_state_delete_list = [target]
    ctrls.remove(target)

    # Construct the bit string ctrl_state which triggers the MCRX gate.
    mask_bitstring = (
        bs1_little_endian if bs1_little_endian[target] == "0" else bs2_little_endian
    )
    if mask_bitstring[target] != "0":
        raise ValueError(
            "The two input bitstrings must differ at the target index.\n"
            f"bs1 = {bs1_little_endian}\n"
            f"bs2 = {bs2_little_endian}\n"
            f"target = {target}."
        )

    ctrl_state = "".join(
        [
            char
            for idx, char in enumerate(mask_bitstring)
            if idx not in ctrl_state_delete_list
        ]
    )
    # TODO: The following reversal cancels some other reversal of unknown
    # origin. This may indicate an endianness issue elsewhere in the codebase
    # to be figured out.
    # HYPOTHESIS: The Endianness issue arises from the RXGate applying ctrl_state
    # in reverse. So, the output ctrl_state shouldn't be reversed because it has the
    # right Endiannesss. This correction has been implemented in the helper function
    # _CRXCircuit.

    # Return the controls and the control state.
    return (ctrls, ctrl_state)


def _build_Xcirc(lp_fam: LPFamily, control: int) -> QuantumCircuit:
    """
    Build pre/post computation change-of-basis circuit in the Givens rotation.

    Assumes all bit strings are little-endian (that is, bit strings are read/indexed right-to-left).
    Applies CX gates conditioned on control, and all other bits where there's a bit flip
    between bs1 and bs2 as targets.
    """
    num_qubits = len(lp_fam.value)
    Xcirc = QuantumCircuit(num_qubits)
    for idx in range(control + 1, num_qubits):
        bit_flip_happens_at_idx = lp_fam.value[idx] == "L"
        if bit_flip_happens_at_idx is True:
            Xcirc.cx(control_qubit=control, target_qubit=idx, ctrl_state="1")

    return Xcirc


# TODO: Clean up the implementation of givens2.
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


def prune_controls(
    lp_fam: LPFamily,
    ctrls: List[int],
    ctrl_state: str,
    encoded_physical_states: Set[str] | None,
) -> Tuple[List[int], str]:
    """
    Perform control pruning on circ to reduce multi-control unitaries (MCUs).

    This algorithm decreases the two-qubit gate depth of resulting
    circuits by determining the set of necessary control qubits to
    implement a Givens rotation without inducing any rotations
    between states in encoded_physical_states. If the Set
    encoded_physical_states is None, then the inputs
    (ctrls, ctrl_state) are simply returned as a tuple with
    no changes.

    Input:
        - lp_fam: the LP family the controls and control state belong to
        - ctrls: A list of controls that are needed for the MCRX gate in the Givens rotation.
                Note that a list of controls post-fusion can also be used here.
        - ctrl_state: The control state corresponding to the list of controls
        - encoded_physical_states: a set of physical states encoded in a bit string.
    Output:
        - A tuple of pruned controls and the corresponding control state.

    """

    if encoded_physical_states == None:
        return (ctrls, ctrl_state)
    else:

        q_prime_idx = _determine_target_of_lp_fam(lp_fam)

        # Use the LP family and q', compute the states tilde_p after the prefix
        # circuit. Do this to the representative too.
        P_tilde = {
            _apply_LP_family_to_bit_string(lp_fam, q_prime_idx, phys_state)
            for phys_state in encoded_physical_states
        }

        # if there is a control missing from the ctrls list, we can remove that qubit from
        # all physical states in P_tilde
        for idx in range(len(lp_fam.value)):
            if idx == q_prime_idx:
                continue
            if idx not in ctrls:
                P_tilde = {
                    phys_state[:idx] + "x" + phys_state[idx + 1 :]
                    for phys_state in P_tilde
                }
        # remove target qubit from P_tilde states
        P_tilde = {
            phys_state[:q_prime_idx] + phys_state[q_prime_idx + 1 :]
            for phys_state in P_tilde
        }
        # remove all the x's
        ctrls_with_missing_idxes_added = [idx for idx in range(len(lp_fam.value))]
        ctrls_with_missing_idxes_added.remove(q_prime_idx)
        P_tilde = {
            _remove_redundant_controls(ctrls_with_missing_idxes_added, phys_state)[1]
            for phys_state in P_tilde
        }
        representative_P_tilde = ctrl_state

        # Iterate through tilde_p, and identify bitstrings that differ at
        # ONLY one qubit other than q'. Add these to the set Q.
        Q_set = set()
        for phys_tilde_state in P_tilde:
            n_bit_differences = 0
            for idx, (rep_char, phys_char) in enumerate(
                zip(representative_P_tilde, phys_tilde_state)
            ):
                if rep_char != phys_char:
                    n_bit_differences += 1
                    diff_bit_idx = idx
            if n_bit_differences == 1:
                Q_set.add(diff_bit_idx)

        # Use Q to eliminate strings from tilde_p that differ at any qubit
        # in Q.
        P_tilde = _eliminate_phys_states_that_differ_from_rep_at_Q_idx(
            representative_P_tilde, P_tilde, Q_set
        )

        # Eliminate states from tilde_p which differ at a qubit in Q.
        should_continue_loop = True
        while should_continue_loop:
            # Find most frequently differing bit idx
            index_counts = [
                0,
            ] * len(representative_P_tilde)
            for phys_tilde_state in P_tilde:
                for idx, (rep_char, phys_char) in enumerate(
                    zip(representative_P_tilde, phys_tilde_state)
                ):
                    if rep_char != phys_char:
                        index_counts[idx] += 1

            # If there are differences, add the most frequently differing bit
            # idx to Q, and eliminate P_tilde that differ at an index in Q.
            if not all([count == 0 for count in index_counts]):
                max_counts_idx = index_counts.index(max(index_counts))
                Q_set.add(max_counts_idx)

            # Update loop params.
            P_tilde = _eliminate_phys_states_that_differ_from_rep_at_Q_idx(
                representative_P_tilde, P_tilde, Q_set
            )
            should_continue_loop = not all(
                [tilde_state == ctrl_state for tilde_state in P_tilde]
            )

        pruned_ctrls, pruned_ctrl_state = _find_pruned_ctrl_list_from_Q_set(
            Q_set, ctrls, ctrl_state
        )

        return (pruned_ctrls, pruned_ctrl_state)


def fuse_controls(
    lp_fam: LPFamily,
    lp_bin_w_angle: List[(str, str, float)],
    round_close_angles: bool = True,
) -> Dict[float, tuple[List[int], str]]:
    """
    This function fuses the controls of multiple multi-RX gates of the same LP family.

    Input:
        - lp_fam: The LP family of the multi-RX gates.
        - lp_bin_w_angle: A list of the bitstrings that are to be givens-rotated.
                          The form of each list entry is (bitstring1, bitstring2, angle).
        - round_close_angles: boolean controling whether to use math.isclose to group angles together.
                              uses rel_tol value of 1e-9.

    Output:
        - A dictionary in which the entries are fused controls binned according to angles. Each is of the form List[(controls,control_state)].
    """

    angle_bin = {}
    target = _determine_target_of_lp_fam(lp_fam)
    # If a target couldn't be found, it means that the givens rotation is the identity rotation.
    if target is None:
        raise ValueError("Invalid LP Family")

    # Step 1: build multiRX gates out of bitstrings, and bin them according to angles.
    for bitstring1, bitstring2, angle in lp_bin_w_angle:
        # find the controls of multiRX gate
        ctrls, ctrl_state = _compute_ctrls_and_state_for_givens_MCRX(
            bitstring1, bitstring2, target
        )

        # Find or create an angle bin.
        if angle in angle_bin:
            angle_bin_key = angle
        elif round_close_angles is True:
            angles_within_tol_of_current_angle = [
                angle_bin_val
                for angle_bin_val in angle_bin.keys()
                if isclose(angle, angle_bin_val, rel_tol=1e-09)
            ]
            if len(angles_within_tol_of_current_angle) > 0:
                angle_bin_key = angles_within_tol_of_current_angle[0]
            else:
                angle_bin_key = angle
                angle_bin[angle_bin_key] = []
        else:
            angle_bin_key = angle
            angle_bin[angle_bin_key] = []
        angle_bin[angle_bin_key].append((ctrls, ctrl_state))

    # Step 2: Control fusion
    for angle, ctrls_list in angle_bin.items():
        # Note that the ctrls_list is a list of tuples of the form (ctrls,ctrl_state) where
        # ctrls refer to the control indices and ctrl_state refers to the state the controls
        # are in.

        # sort the controls according to a gray-order.
        sorted_ctrls_list = sorted(ctrls_list, key=lambda x: gray_to_index(x[1]))
        # control fuse
        sorted_ctrls_list = _fuse_ctrls_of_ctrls_list(sorted_ctrls_list)
        sorted_ctrls_list = [
            _remove_redundant_controls(ctrls, ctrl_state)
            for ctrls, ctrl_state in sorted_ctrls_list
        ]
        angle_bin[angle] = sorted_ctrls_list
    return angle_bin


def compute_LP_family(bit_string_1: str, bit_string_2: str) -> LPFamily:
    """
    Compare each element of bit_string_1 and bit_string_2 to obtain LP family.

    Input:
        - Two equal-length bit strings strings bit_string_1 and bit_string 2.
    Output:
        - A list of the same length as one of the input bitstrings. Each element of
            the return list consists of the following:
            - "P" if both inputs have the same value at that index
            - "L" if the inputs differ at that index
    """
    if len(bit_string_1) != len(bit_string_2):
        raise IndexError("Both input bit strings must have the same length.")

    LP_family = []
    for char_1_2_tuple in zip(bit_string_1, bit_string_2):
        match char_1_2_tuple:
            case ("0", "0"):
                LP_family.append("P")
            case ("1", "1"):
                LP_family.append("P")
            case ("0", "1"):
                LP_family.append("L")
            case ("1", "0"):
                LP_family.append("L")
            case _:
                raise ValueError(
                    f"Encountered non-bit character while comparing chars: {char_1_2_tuple}."
                )

    return LPFamily(tuple(LP_family))


def _apply_LP_family_to_bit_string(
    LP_family: LPFamily | List[str], q_prime_idx: int, bit_string: str
) -> str:
    """
    Use LP_family to figure out how bit_string will transform under the CX
    change of basis inside Givens rotation circuits. Bit strings are read left-to-right.

    The q_prime_idx is skipped in applying the LP family. Bits in bit_string which are an L-index only
    flip if the bit at q_prime_idx has value 1.
    """
    if not isinstance(LP_family, LPFamily):
        LP_family = LPFamily(LP_family)

    if bit_string[q_prime_idx] == "0":
        return bit_string

    result_string_list = list(bit_string)
    for idx, op in enumerate(LP_family.value):
        if idx == q_prime_idx:
            continue  # Never flip the bit at this index.
        match op:
            case "L":
                result_string_list[idx] = str(
                    (int(result_string_list[idx]) + 1) % 2
                )  # Flip the bit.
            case "P":
                pass  # Don't flip the bit.

    return "".join(result_string_list)


def _eliminate_phys_states_that_differ_from_rep_at_Q_idx(
    representative: str, phys_states_set: Set[str], Q_set: Set[int]
) -> Set[str]:
    """
    Return a version of phys_states_set that only contains strings which match representative
    at the indices specified by Q_set.
    """
    return set(
        filter(
            lambda phys_states_set: not any(
                [
                    rep_char != phys_char and idx in Q_set
                    for idx, (rep_char, phys_char) in enumerate(
                        zip(representative, phys_states_set)
                    )
                ]
            ),
            phys_states_set,
        )
    )


def _fuse_ctrls_of_ctrls_list(
    sorted_ctrls_list: List[(List[int], str)],
) -> List[(List[int], str)]:
    """
    Fuses the control states when the control states only differ in one bit.
    After fusion, the position in the control state where the fusion occurred is
    replaced by the placeholder string "x".

    For example, if the control states to be fused were "0000", "0100", "1000", "1100",
    the fusion would happen in three steps:
        1. "0000" and "0100" get fused into "0x00"
        2. "1000" and "1100" get fused into "1x00"
        3. "0x00" and "1x00" get fused into "xx00"

    The redundant "x" controls can be removed by the helper function _remove_redundant_controls.
    """
    ctrl_state_length = len(sorted_ctrls_list[0][1])
    for ctrl_state_idx in range(ctrl_state_length - 1, -1, -1):
        sorted_ctrl_list_element_idx = 0
        while sorted_ctrl_list_element_idx < len(sorted_ctrls_list) - 1:
            if _bitstrings_differ_in_one_bit(
                sorted_ctrls_list[sorted_ctrl_list_element_idx][1],
                sorted_ctrls_list[sorted_ctrl_list_element_idx + 1][1],
            ) == (True, ctrl_state_idx):
                ctrls = sorted_ctrls_list[sorted_ctrl_list_element_idx][0]
                # "x" is added as a place holder in the place where controls got fused
                # to indicate that there's no control state at that position anymore.
                new_ctrl_state = (
                    sorted_ctrls_list[sorted_ctrl_list_element_idx][1][:ctrl_state_idx]
                    + "x"
                    + sorted_ctrls_list[sorted_ctrl_list_element_idx][1][
                        ctrl_state_idx + 1 :
                    ]
                )
                del sorted_ctrls_list[
                    sorted_ctrl_list_element_idx : sorted_ctrl_list_element_idx + 2
                ]
                sorted_ctrls_list.insert(
                    sorted_ctrl_list_element_idx, (ctrls, new_ctrl_state)
                )
            sorted_ctrl_list_element_idx += 1
    return sorted_ctrls_list


def _CRXCircuit_with_MCX(ctr_list: List[Union[str, List[int]]], 
    angle: float, target: int, num_qubits: int) -> QuantumCircuit:
    """Returns the circuit decomposition of the RXGate into MCXs using the ABC decomposition (Corollary 4.2, Nielsen and Chaung)"""
    ctrl_state, ctrls = ctr_list
    num_ctrls = len(ctrl_state)
    circ_with_mcx = QuantumCircuit(num_qubits)
    circ_with_mcx.append(RZGate(-1.0*np.pi/2.0), [target])
    circ_with_mcx.append(RYGate(-1.0*angle/2.0), [target])
    circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state[::-1]), ctrls + [target])
    circ_with_mcx.append(RYGate(1.0*angle/2.0), [target])
    circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state[::-1]), ctrls + [target])
    circ_with_mcx.append(RZGate(1.0*np.pi/2.0), [target])
    return circ_with_mcx


# FIX: type hint of lp_fam
def bitstring_value_of_LP_family(lp_fam: List[str] | LPFamily) -> str:
    """Validates form of lp_fam with LPFamily class if List[str]."""
    if not isinstance(lp_fam, LPFamily):
        lp_fam = LPFamily(lp_fam)
    # Assigns a bitstring value to the LP family. This will help in binning LP families and
    # later in the gray code ordering.
    LP_bitstring = ""
    for operator in lp_fam.value:
        if operator == "L":
            LP_bitstring += "0"
        elif operator == "P":
            LP_bitstring += "1"
    return LP_bitstring


def gray_to_index(gray_str: str) -> int:
    """
    Find the index at which a particular bitstring occurs in the gray order.

    This is implemented by first finding the binary equivalent of the bitstring
    (see https://mathworld.wolfram.com/GrayCode.html for an easy introduction).
    Once the binary representation is found, the binary number is converted to
    an integer index, which is the return value.
    """
    binary = [
        int(gray_str[0])
    ]  # The first binary bit is the same as the first Gray code bit
    for i in range(1, len(gray_str)):
        # XOR the previous binary bit with the current Gray code bit
        binary_bit = binary[i - 1] ^ int(gray_str[i])
        binary.append(binary_bit)
    index_value = 0
    for i in range(len(binary)):
        index_value += (binary[-i - 1]) * (2**i)
    return index_value


def _bitstrings_differ_in_one_bit(
    bitstring1: str, bitstring2: str
) -> tuple[bool, Union[int | None]]:
    """Checks if two bitstrings differ just in one bit."""
    if len(bitstring1) != len(bitstring2):
        raise AttributeError("The length of the two bitstrings must be equal")
    else:
        diff_bit_count = 0
        for i in range(len(bitstring1)):
            if bitstring1[i] != bitstring2[i]:
                diff_bit_count += 1
                first_bit_of_diff = i
        if diff_bit_count == 1:
            return True, first_bit_of_diff
        else:
            return False, None


# FIX: type hint of lp_fam here and in prune_controls
def _determine_target_of_lp_fam(lp_fam: LPFamily) -> int:
    """Given a bitstring corresponding to a LP family, this function returns the target qubit"""
    target = None
    if not isinstance(lp_fam, LPFamily):
        lp_fam = LPFamily(lp_fam)
    for i in range(len(lp_fam.value)):
        if lp_fam.value[i] != "L":
            continue
        elif lp_fam.value[i] == "L":
            target = i
            break
    return target


def _remove_redundant_controls(
    ctrls: List[int], ctrl_state: str
) -> tuple[List[int], str]:
    if len(ctrls) != len(ctrl_state):
        raise ValueError("The length of the controls and control state must be equal")
    new_ctrls = []
    new_ctrl_state = ""
    for i in range(len(ctrls)):
        if ctrl_state[i] != "x":
            new_ctrls.append(ctrls[i])
            new_ctrl_state += ctrl_state[i]
    return new_ctrls, new_ctrl_state


def _find_pruned_ctrl_list_from_Q_set(
    Q_set: Set[int], ctrls: List[int], ctrl_state: str
):
    """Finds the pruned control list by only retaining pruned controls"""
    pruned_ctrls = []
    pruned_ctrl_state = ""
    for idx in range(len(ctrls)):
        if idx in Q_set:
            pruned_ctrls.append(ctrls[idx])
            pruned_ctrl_state += ctrl_state[idx]
    return (pruned_ctrls, pruned_ctrl_state)


def _make_bitstring(length):
    # Helper for testing purposes.
    return "".join(f"{int(random()>0.5)}" for _ in range(length))


def _test_givens():
    print("Testing givens.")

    test_circ = QuantumCircuit(2)
    test_circ.cx(control_qubit=1, target_qubit=0, ctrl_state="1")
    test_circ.append(RXGate(1).control(ctrl_state="1"), [0, 1])
    test_circ.cx(control_qubit=1, target_qubit=0, ctrl_state="1")

    op_test = Operator(test_circ)
    op_given = givens("01", "10", 1)
    assert op_test.equiv(op_given), "Failed non-random test."
    print("Givens non-random test satisfied.")

    N = int(random() * 5 + 2)
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
    expected_givens_operator = expm(-1j / 2 * angle * H)

    assert np.isclose(
        a=actual_givens_circuit_as_operator, b=expected_givens_operator
    ).all(), f"Failed random test. Constructed and expected givens operators not close. Largest difference = {np.max(actual_givens_circuit_as_operator-expected_givens_operator)}"
    print("Givens random test satisfied.")


def _test_givens2():
    print("Testing givens2.")

    N = int(random() * 5 + 2)
    random_strings = [_make_bitstring(N) for _ in range(4)]
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


def _test_building_MCRX_gate():
    print("Verifying that the multiRX with mutliCXs subcircuit is correctly constructed.")

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
    expected_circ.append(RZGate(-1.0*np.pi/2.0), [control_qubit])
    expected_circ.append(RYGate(-1.0*angle/2.0), [control_qubit])
    expected_circ.append(MCXGate(num_ctrl_qubits=len(expected_ctrls), 
        ctrl_state=expected_ctrl_state), expected_ctrls + [control_qubit])
    expected_circ.append(RYGate(1.0*angle/2.0), [control_qubit])
    expected_circ.append(MCXGate(num_ctrl_qubits=len(expected_ctrls), 
        ctrl_state=expected_ctrl_state), expected_ctrls + [control_qubit])
    expected_circ.append(RZGate(1.0*np.pi/2.0), [control_qubit])

    actual_ctrls, actual_ctrl_state = _compute_ctrls_and_state_for_givens_MCRX(
        bs1, bs2, target=control_qubit
    )
    actual_circ = _CRXCircuit_with_MCX([actual_ctrl_state, actual_ctrls], angle, control_qubit, 4)

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
    expected_circ.append(RZGate(-1.0*np.pi/2.0), [control_qubit])
    expected_circ.append(RYGate(-1.0*angle/2.0), [control_qubit])
    expected_circ.append(MCXGate(num_ctrl_qubits=len(expected_ctrls), 
        ctrl_state=expected_ctrl_state), expected_ctrls + [control_qubit])
    expected_circ.append(RYGate(1.0*angle/2.0), [control_qubit])
    expected_circ.append(MCXGate(num_ctrl_qubits=len(expected_ctrls), 
        ctrl_state=expected_ctrl_state), expected_ctrls + [control_qubit])
    expected_circ.append(RZGate(1.0*np.pi/2.0), [control_qubit])

    actual_ctrls, actual_ctrl_state = _compute_ctrls_and_state_for_givens_MCRX(
        bs1, bs2, target=control_qubit
    )
    actual_circ = _CRXCircuit_with_MCX([actual_ctrl_state, actual_ctrls], angle, control_qubit, 12)

    assert expected_circ == actual_circ, (
        "Encountered inequivalent circuits. Expected:\n"
        f"{expected_circ.draw()}\nObtained:\n"
        f"{actual_circ.draw()}"
    )

    print(f"Test passed. Obtained circuit:\n{actual_circ.draw()}\n")


def _test_MCRX_with_MCX_is_same_as_MCU():
    print("Verifying that the decomposition of the multiRX into mutliCXs is correct")

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
    statevector_with_mcu = Statevector.from_instruction(expected_circ)


    actual_ctrls, actual_ctrl_state = _compute_ctrls_and_state_for_givens_MCRX(
        bs1, bs2, target=control_qubit
    )
    acutal_circ = _CRXCircuit_with_MCX([actual_ctrl_state, actual_ctrls], angle, control_qubit, 4)
    statevector_with_mcx = Statevector.from_instruction(acutal_circ)

    assert statevector_with_mcx.equiv(statevector_with_mcu), (
        "Encountered inequivalent evolution circuits. Expected Statevector:\n"
        f"{statevector_with_mcx}\nObtained Statevector:\n"
        f"{statevector_with_mcu}"
    )

    print(f"Test passed. Circuits are evolution equivalent")

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
    statevector_with_mcu = Statevector.from_instruction(expected_circ)


    actual_ctrls, actual_ctrl_state = _compute_ctrls_and_state_for_givens_MCRX(
        bs1, bs2, target=control_qubit
    )
    acutal_circ = _CRXCircuit_with_MCX([actual_ctrl_state, actual_ctrls], angle, control_qubit, 12)
    statevector_with_mcx = Statevector.from_instruction(acutal_circ)
        
    assert statevector_with_mcx.equiv(statevector_with_mcu), (
        "Encountered inequivalent evolution circuits. Expected Statevector:\n"
        f"{statevector_with_mcx}\nObtained Statevector:\n"
        f"{statevector_with_mcu}"
    )

    print(f"Test passed. Circuits are evolution equivalent")


def _test_compute_LP_family():
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


def _test_compute_LP_family_fails_on_bad_input():
    """LP family computation should fail for unequal length strings."""
    bs1 = "11110"
    bs2 = "00"
    print(f"Checking that computing LP family of {bs1} and {bs2} raises IndexError.")

    try:
        compute_LP_family(bs1, bs2)
    except IndexError as e:
        print(f"Test passed. Raised error: {e}\n")
    else:
        assert False, "IndexError not raised."


def _test_compute_LP_family_fails_for_non_bitstrings():
    """LP family computation fails for input not bit strings."""
    bs1 = "a3bb"
    bs2 = "1011"
    print(f"Checking that expected_LP_family({bs1}, {bs2}) raises a ValueError.")

    try:
        compute_LP_family(bs1, bs2)
    except ValueError as e:
        print(f"Test passed. Raised error: {e}\n")
    else:
        assert False, "ValueError not raised."


def _test_prune_controls_acts_as_expected():
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


def _test_apply_LP_family():
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


def _test_eliminate_phys_states_that_differ_from_rep_at_Q_idx():
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


def _test_control_fusion_fails_for_inconsistent_lp_bin():
    bin_value = "10110101"
    bitstring_list = [("0000", "1111", 1.0)]
    try:
        fuse_controls(bin_value, bitstring_list)
    except ValueError as e:
        print(f"Test passed. Raised error: {e}")


def _test_control_fusion():
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


def _test_gray_to_index():
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


def _test_gray_to_index_for_larger_bitstrings():
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


def _test_givens_fused_controls():
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
    circ_with_mcx = QuantumCircuit(8)
    circ_with_mcx.append(RZGate(-1.0*np.pi/2.0), [1])
    circ_with_mcx.append(RYGate(-1.0*1.0/2.0), [1])
    circ_with_mcx.append(MCXGate(num_ctrl_qubits=1, ctrl_state="0"), [3, 1])
    circ_with_mcx.append(RYGate(1.0*1.0/2.0), [1])
    circ_with_mcx.append(MCXGate(num_ctrl_qubits=1, ctrl_state="0"), [3, 1])
    circ_with_mcx.append(RZGate(1.0*np.pi/2.0), [1])
    expected_circuit.compose(circ_with_mcx, inplace=True)
    expected_circuit.cx(1, 4)
    expected_circuit.cx(1, 6)
    assert expected_circuit == generated_circuit
    print("givens with fused controls test passed.")


def _test_bitstring_value_of_LP_fam():
    lp_fam = ("L", "P", "L", "L", "P", "P")
    expected_bitstring = "010011"
    assert expected_bitstring == bitstring_value_of_LP_family(lp_fam)
    print("bitstring_value_of_LP_fam passed.")


def _test_LPOperator_works():
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
    try:
        LPOperator(incorrect_val)
    except ValueError as e:
        print(f"Test passed. Raised error: {e}")
    else:
        raise AssertionError("ValueError not raised.")

    print("Check that LPOperator is immutable.")
    lp_op = LPOperator("L")
    try:
        lp_op.value = "P"
    except FrozenInstanceError as e:
        print(f"Test passed. Raised error: {e}")
    else:
        raise AssertionError("FrozenInstanceError not raised.")


def _test_LPFamily_works():
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
    try:
        LPFamily(bad_lp_family)
    except ValueError as e:
        print(f"Test passed. Raised error: {e}")
    else:
        raise AssertionError("ValueError not raised.")

    # Test immutability.
    print("Checking that LPFamily instances are immutable.")
    lp_fam = LPFamily(lp_family_str)
    try:
        lp_fam.value = LPFamily(("P", "L"))
    except FrozenInstanceError as e:
        print(f"Test passed. Raised error: {e}")
    else:
        raise AssertionError("FrozenInstanceError not raised.")


if __name__ == "__main__":
    _test_givens()
    # _test_givens2()  # TODO fix nondeterministic behavior of this test.
    _test_Xcirc()
    _test_building_MCRX_gate()
    _test_MCRX_with_MCX_is_same_as_MCU()
    _test_compute_LP_family()
    _test_compute_LP_family_fails_on_bad_input()
    _test_compute_LP_family_fails_for_non_bitstrings()
    _test_prune_controls_acts_as_expected()
    _test_apply_LP_family()
    _test_eliminate_phys_states_that_differ_from_rep_at_Q_idx()
    _test_control_fusion()
    _test_gray_to_index()
    _test_bitstring_value_of_LP_fam()
    _test_gray_to_index_for_larger_bitstrings()
    _test_control_fusion_fails_for_inconsistent_lp_bin()
    _test_givens_fused_controls()
    _test_LPOperator_works()
    _test_LPFamily_works()

    print("All tests passed.")
