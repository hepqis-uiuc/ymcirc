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
import copy
from scipy.linalg import expm
from typing import Tuple, List, Set, Dict, Union


def givens(
    bit_string_1: str,
    bit_string_2: str,
    angle: float,
    reverse: bool = False,
    physical_control_qubits: Set[int | Qubit] | None = None,
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
    circ = QuantumCircuit(num_qubits)

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

        # Configure the multi-controlled X rotation (MCRX) gate.
        ctrls, ctrl_state, multiRX = _build_multiRX(
            bit_string_1, bit_string_2, angle, target, physical_control_qubits
        )

        # Construct the pre- and post-MCRX circuits.
        Xcirc = _build_Xcirc(
            bitstring_value_of_LP_family(compute_LP_family(bit_string_1, bit_string_2)), control=target
        )

        # Add multiRX to the circuit, specifying
        # The proper control locations and target location
        # via a list of the qubit indices.
        circ.append(multiRX, ctrls + [target])

        # Assemble the final circuit.
        # Using inplace speeds up circuit composition.
        circ.compose(Xcirc, inplace=True)
        Xcirc.compose(circ, inplace=True)
        if reverse is True:
            Xcirc = Xcirc.reverse_bits()
        return Xcirc


def givens_fused_controls(
    lp_bin_w_angle: List[(str, str, float)],
    lp_bin_value: str,
    reverse: bool = False,
    physical_control_qubits: Dict[(str, str), Set[int | Qubit]] | None = None,
) -> QuantumCircuit:
    """
    Implements givens rotation using the same logic as the givens function but for fused multiRX's.

    Inputs:
        - lp_bin_w_angle: list of bitstrings of the same LP family and the angle they have to be rotated by.
        - lp_bin_value: bitstring value of LP bin
        - reverse: optional argument to deal with endianess issues
        - physical_control_qubits: dictionary which provides pruned controls for bitstrings.

    Output:
        QuantumCircuit object that has the necessary givens rotation.
    """
    # Input validation and sanity checks.

    num_qubits = len(lp_bin_w_angle[0][0])

    for bit_string_1, bit_string_2, angle in lp_bin_w_angle:
        if len(bit_string_1) != len(bit_string_2):
            raise ValueError("Bit strings must be the same length.")
        lp_bin_matches = (
            bitstring_value_of_LP_family(compute_LP_family(bit_string_1, bit_string_2))
            == lp_bin_value
        )
        if lp_bin_matches is False:
            return ValueError(
                "The LP value of the bitstrings should match with the LP bin value"
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
            current_idx_is_target_idx = lp_bin_value[idx] == "0"
            if current_idx_is_target_idx is True:
                target = idx
                break
        # Add multiRX to the circuit, specifying
        # The proper control locations and target location
        # via a list of the qubit indices.
        angle_dict = fuse_controls(
            lp_bin_value, lp_bin_w_angle, physical_control_qubits
        )
        for angle, ctrl_list in angle_dict.items():
            for ctrls, ctrl_state in ctrl_list:
                ctrl_state = ctrl_state[::-1]
                multiRX = RXGate(angle).control(
                    num_ctrl_qubits=len(ctrls), ctrl_state=ctrl_state
                )
                circ.append(multiRX, ctrls + [target])
        Xcirc = _build_Xcirc(lp_bin_value, control=target)

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
    physical_control_qubits: Set[int | Qubit] | None,
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
    ctrl_state = ctrl_state[::-1]

    # Assemble MCRX gate.
    # Note that the final qubit is the target.
    return (
        physical_ctrls,
        ctrl_state,
        RXGate(angle).control(
            num_ctrl_qubits=len(physical_ctrls), ctrl_state=ctrl_state
        ),
    )


def _build_Xcirc(bs_of_lp_fam_little_endian: str, control: int) -> QuantumCircuit:
    """
    Build pre/post computation change-of-basis circuit in the Givens rotation.

    Assumes all bit strings are little-endian (that is, bit strings are read/indexed right-to-left).
    Applies CX gates conditioned on control, and all other bits where there's a bit flip
    between bs1 and bs2 as targets.
    """
    num_qubits = len(bs_of_lp_fam_little_endian)
    Xcirc = QuantumCircuit(num_qubits)
    for idx in range(control + 1, num_qubits):
        bit_flip_happens_at_idx = bs_of_lp_fam_little_endian[idx] == "0"
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
    bit_string_1: str, bit_string_2: str, encoded_physical_states: Set[str]
) -> Set[int]:
    """
    Perform control pruning on circ to reduce multi-control unitaries (MCUs).

    This algorithm decreases the two-qubit gate depth of resulting
    circuits by determining the set of necessary control qubits to
    implement a Givens rotation between bit_string_1 and bit_string_2
    without inducing any rotations between states in encoded_physical_states.

    Input:
        - bit_string_1: a physical state encoded in a bit string.
        - bit_string_2: a physical state encoded in a bit string.
        - encoded_physical_states: a set of physical states encoded in a bit string.
    Output:
        - A set of necessary controls to implement a Givens rotation between
            the two bit strings without inducing rotations among states in
            encoded_physical_states.

    Note that errors are raised if the bit string lengths are unequal, or there are non-bit
    characters in state encodings.
    """
    # Validate input data.
    bit_string_inputs_have_unequal_lengths = (
        len(bit_string_1) != len(bit_string_2)
    ) or (
        any(
            [
                len(bit_string_1) != len(encoded_state)
                for encoded_state in encoded_physical_states
            ]
        )
    )
    string_inputs_have_non_bit_chars = (
        any([char not in {"0", "1"} for char in bit_string_1])
        or any([char not in {"0", "1"} for char in bit_string_2])
        or any(
            [
                char not in {"0", "1"}
                for bit_string in encoded_physical_states
                for char in bit_string
            ]
        )
    )
    if bit_string_inputs_have_unequal_lengths:
        raise IndexError(
            "All input state data must be the same length. Encountered:\n"
            f"bit_string_1 = {bit_string_1}\n"
            f"bit_string_2 = {bit_string_2}\n"
            f"encoded_physical_states = {encoded_physical_states}"
        )
    elif string_inputs_have_non_bit_chars:
        raise ValueError(
            "All input state data must be interpretable as bit strings. Encountered:\n"
            f"bit_string_1 = {bit_string_1}\n"
            f"bit_string_2 = {bit_string_2}\n"
            f"encoded_physical_states = {encoded_physical_states}"
        )

    # Find the LP family of the representative, and identify which qubit
    # "L" first appears on.
    representative_P = bit_string_1
    representative_LP_family = compute_LP_family(bit_string_1, bit_string_2)
    q_prime_idx = next(
        (
            idx
            for idx, LP_val in enumerate(representative_LP_family)
            if LP_val[0] == "L"
        ),
        -1,
    )
    if q_prime_idx == -1:
        raise ValueError("Attempting to prune controls on two identical bit strings.")

    # Use the LP family and q', compute the states tilde_p after the prefix
    # circuit. Do this to the representative too.
    representative_P_tilde = _apply_LP_family_to_bit_string(
        representative_LP_family, q_prime_idx, representative_P
    )
    P_tilde = {
        _apply_LP_family_to_bit_string(
            representative_LP_family, q_prime_idx, phys_state
        )
        for phys_state in encoded_physical_states
    }

    # Iterate through tilde_p, and identify bitstrings that differ at
    # ONLY one qubit other than q'. Add these to the set Q.
    Q_set = set()
    for phys_tilde_state in P_tilde:
        n_bit_differences = 0
        for idx, (rep_char, phys_char) in enumerate(
            zip(representative_P_tilde, phys_tilde_state)
        ):
            if idx == q_prime_idx:
                continue
            elif rep_char != phys_char:
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
    bit_string_1_tilde = _apply_LP_family_to_bit_string(
        representative_LP_family, q_prime_idx, bit_string_1
    )
    bit_string_2_tilde = _apply_LP_family_to_bit_string(
        representative_LP_family, q_prime_idx, bit_string_2
    )
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
                if idx == q_prime_idx:
                    continue
                elif rep_char != phys_char:
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
            [
                tilde_state in {bit_string_1_tilde, bit_string_2_tilde}
                for tilde_state in P_tilde
            ]
        )

    return Q_set


def fuse_controls(
    bin_value: str,
    lp_bin_w_angle: List[(str, str, float)],
    pruned_controls: Union[Dict[List[(str, str)], Set[int]] | None] = None,
) -> Dict[float, tuple[List[int], str]]:
    """
    This function fuses the (pruned) controls of multiple multi-RX gates of the same LP family.

    Input:
        - bin_value: The bitstring value of the LP family of the multi-RX gates.
        - lp_bin_w_angle: A list of the bitstrings that are to be givens-rotated.
            The form of each list entry is (bitstring1, bitstring2, angle).
        - pruned_controls: A dictionary which specifies which controls are to be pruned for a particular givens rotation. The entries in this
            dictionary should be of the form pruned_controls[(bitstring1, bitstring2)] = list[int].

    Output:
        - A dictionary in which the entries are fused controls binned according to angles. Each is of the form List[(controls,control_state)].
    """

    angle_bin = {}
    target = _determine_target_of_lp_bitstring(bin_value)
    # If a target couldn't be found, it means that the givens rotation is the identity rotation.
    if target == None:
        return angle_bin

    # Check if the bitstrings are indeed in the LP family specified by the LP bin value.
    for bitstring1, bitstring2, angle in lp_bin_w_angle:
        lp_fam = compute_LP_family(bitstring1, bitstring2)
        lp_fam_bs_value = bitstring_value_of_LP_family(lp_fam)
        if bin_value != lp_fam_bs_value:
            raise ValueError(
                "The LP family of the bin doesn't match with the LP family of the bitstrings"
            )

    # Step 1: build multiRX gates out of bitstrings, and bin them according to angles.
    for bitstring1, bitstring2, angle in lp_bin_w_angle:
        if (
            pruned_controls != None
            and (bitstring1, bitstring2) in pruned_controls.keys()
        ):
            ctrls, ctrl_state, multiRX = _build_multiRX(
                bitstring1,
                bitstring2,
                angle,
                target,
                pruned_controls[(bitstring1, bitstring2)],
            )
        else:
            ctrls, ctrl_state, multiRX = _build_multiRX(
                bitstring1, bitstring2, angle, target, physical_control_qubits=None
            )
        if angle in angle_bin:
            angle_bin[angle].append((ctrls, ctrl_state))
        else:
            angle_bin[angle] = []
            angle_bin[angle].append((ctrls, ctrl_state))

    # Step 2: compare control qubits between all multiRX's. If there are some control qubits that differ,
    # perform decomposition on the controls that differ.
    for angle, ctrls_list in angle_bin.items():
        max_ctrl_qubits = set()
        min_ctrl_state_length = len(ctrls_list[0][0])
        # Obtain the maximum number of controls
        for ctrls, ctrl_state in ctrls_list:
            for ctrl in ctrls:
                if ctrl not in max_ctrl_qubits:
                    max_ctrl_qubits.add(ctrl)
            if len(ctrls) < min_ctrl_state_length:
                min_ctrl_state_length = len(ctrls)
        if min_ctrl_state_length == len(max_ctrl_qubits):
            new_ctrls_list = ctrls_list
        else:
            new_ctrls_list = copy.deepcopy(ctrls_list)
            for ctrls, ctrl_state in ctrls_list:
                new_ctrls, new_ctrl_states = _decompose_pruned_controls(
                    ctrls, ctrl_state, max_ctrl_qubits
                )
                new_ctrls_list.remove((ctrls, ctrl_state))
                for new_ctrl_state_to_be_added in new_ctrl_states:
                    new_ctrls_list.append((new_ctrls, new_ctrl_state_to_be_added))
        angle_bin[angle] = new_ctrls_list

    # Step 3: Control fusion
    for angle, ctrls_list in angle_bin.items():
        # sort the controls so that they are in the right positions.
        sorted_ctrls_list = [
            _sort_controls_list(ctrls, ctrl_state) for ctrls, ctrl_state in ctrls_list
        ]
        # sort the controls according to a gray-order
        sorted_ctrls_list = sorted(
            sorted_ctrls_list, key=lambda x: binary_to_gray(x[1])
        )
        # control fuse
        ctrl_state_length = len(sorted_ctrls_list[0][1])
        for i in range(ctrl_state_length - 1, -1, -1):
            comparison_idx = 0
            while comparison_idx < len(sorted_ctrls_list) - 1:
                if _bitstrings_differ_in_one_bit(
                    sorted_ctrls_list[comparison_idx][1],
                    sorted_ctrls_list[comparison_idx + 1][1],
                ) == (True, i):
                    ctrls = sorted_ctrls_list[comparison_idx][0]
                    new_ctrl_state = (
                        sorted_ctrls_list[comparison_idx][1][:i]
                        + "x"
                        + sorted_ctrls_list[comparison_idx][1][i + 1 :]
                    )
                    del sorted_ctrls_list[comparison_idx : comparison_idx + 2]
                    sorted_ctrls_list.insert(comparison_idx, (ctrls, new_ctrl_state))
                comparison_idx += 1
        sorted_ctrls_list = [
            _remove_redundant_controls(ctrls, ctrl_state)
            for ctrls, ctrl_state in sorted_ctrls_list
        ]
        angle_bin[angle] = sorted_ctrls_list
    return angle_bin


def compute_LP_family(bit_string_1: str, bit_string_2: str) -> List[str]:
    """
    Compare each element of bit_string_1 and bit_string_2 to obtain LP family.

    Input:
        - Two equal-length bit strings strings bit_string_1 and bit_string 2.
    Output:
        - A list of the same length as one of the input bitstrings. Each element of
            the return list consists of the following:
            - "P0" if both inputs have a zero at that index.
            - "P1" if both inputs have a one at that index.
            - "L+" if bit_string_2 has a 1 where bit_string_1 has a 0.
            - "L-" if bit_string_2 has a 0 where bit_string_1 has a 1.
    """
    if len(bit_string_1) != len(bit_string_2):
        raise IndexError("Both input bit strings must have the same length.")

    LP_family = []
    for char_1_2_tuple in zip(bit_string_1, bit_string_2):
        match char_1_2_tuple:
            case ("0", "0"):
                LP_family.append("P0")
            case ("1", "1"):
                LP_family.append("P1")
            case ("0", "1"):
                LP_family.append("L+")
            case ("1", "0"):
                LP_family.append("L-")
            case _:
                raise ValueError(
                    f"Encountered non-bit character while comparing chars: {char_1_2_tuple}."
                )

    return LP_family


def _apply_LP_family_to_bit_string(
    LP_family: List[str], q_prime_idx: int, bit_string: str
) -> str:
    """
    Use LP_family to figure out how bit_string will transform under the CX
    change of basis inside Givens rotation circuits. Bit strings are read left-to-right.

    The q_prime_idx is skipped in applying the LP family. Bits in bit_string which are an L-index only
    flip if the bit at q_prime_idx has value 1.
    """
    if bit_string[q_prime_idx] == "0":
        return bit_string

    result_string_list = list(bit_string)
    for idx, op in enumerate(LP_family):
        if idx == q_prime_idx:
            continue  # Never flip the bit at this index.
        match op[0]:
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


def bitstring_value_of_LP_family(lp_fam: str) -> str:
    # Assigns a bitstring value to the LP family. This will help in binning LP families and
    # later in the gray code ordering.
    LP_bitstring = ""
    for operator in lp_fam:
        if operator == "L+" or operator == "L-":
            LP_bitstring += "0"
        elif operator == "P0" or operator == "P1":
            LP_bitstring += "1"
    return LP_bitstring


def binary_to_gray(binary_str: str) -> str:
    """Convert a binary string to its Gray code representation."""
    gray = []
    gray.append(binary_str[0])  # The first bit is the same
    for i in range(1, len(binary_str)):
        # XOR the current bit with the previous bit
        gray_bit = str(int(binary_str[i]) ^ int(binary_str[i - 1]))
        gray.append(gray_bit)
    return "".join(gray)


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


def _determine_target_of_lp_bitstring(lp_bitstring: str) -> int:
    """Given a bitstring corresponding to a LP family, this function returns the target qubit"""
    target = None
    for i in range(len(lp_bitstring)):
        if lp_bitstring[i] != "0":
            continue
        elif lp_bitstring[i] == "0":
            target = i
            break
    return target


def _decompose_pruned_controls(
    pruned_ctrls: list[int], ctrl_state: str, max_ctrls_needed: Set[int]
) -> tuple[list[int], list[str]]:
    """Decomposes pruned controls in a way appropriate for control fusion"""
    decomp_register = 0
    new_ctrls = copy.deepcopy(pruned_ctrls)
    new_ctrl_state = ctrl_state
    for max_ctrl_element in max_ctrls_needed:
        if max_ctrl_element not in pruned_ctrls:
            decomp_register += 1
            new_ctrls.append(max_ctrl_element)
            new_ctrl_state += "0"
    if decomp_register == 0:
        new_ctrls_to_be_added = [new_ctrl_state]
    if decomp_register != 0:
        new_ctrls_to_be_added = [new_ctrl_state]
        for i in range(-decomp_register, 0):
            old_ctrls_to_be_added = copy.deepcopy(new_ctrls_to_be_added)
            for already_decomposed_ctrl_states in old_ctrls_to_be_added:
                if i != -1:
                    new_ctrl_decomposed1 = (
                        already_decomposed_ctrl_states[:i]
                        + "1"
                        + already_decomposed_ctrl_states[i + 1 :]
                    )
                elif i == -1:
                    new_ctrl_decomposed1 = already_decomposed_ctrl_states[:i] + "1"
                new_ctrls_to_be_added.append(new_ctrl_decomposed1)
    return new_ctrls, new_ctrls_to_be_added


def _sort_controls_list(ctrls: List[int], ctrl_state: str) -> tuple[List[int], str]:
    """Sorts the controls in the order they appear in the quantum circuit."""
    ctrl_state_list = list(ctrl_state)
    ctrls_and_state_zipped = zip(ctrls, ctrl_state_list)
    sorted_ctrls_zipped = sorted(ctrls_and_state_zipped, key=lambda x: x[0])
    ctrls, ctrl_state_list = zip(*sorted_ctrls_zipped)
    ctrls = list(ctrls)
    ctrl_state = "".join(ctrl_state_list)
    return ctrls, ctrl_state


def _remove_redundant_controls(
    ctrls: List[int], ctrl_state: str
) -> tuple[List[int], str]:
    assert len(ctrls) == len(
        ctrl_state
    ), "The length of the controls and control state must be equal"
    new_ctrls = []
    new_ctrl_state = ""
    for i in range(len(ctrls)):
        if ctrl_state[i] != "x":
            new_ctrls.append(ctrls[i])
            new_ctrl_state += ctrl_state[i]
    return new_ctrls, new_ctrl_state


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
    Xcirc = _build_Xcirc(bitstring_value_of_LP_family(compute_LP_family(bs1, bs2)), control_qubit)

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
    Xcirc = _build_Xcirc(bitstring_value_of_LP_family(compute_LP_family(bs1, bs2)), control_qubit)

    assert Xcirc_expected == Xcirc, (
        "Encountered inequivalent circuits. Expected:\n"
        f"{Xcirc_expected.draw()}\nObtained:\n"
        f"{Xcirc.draw()}"
    )

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
    actual_ctrls, actual_ctrl_state, multiRX_actual = _build_multiRX(
        bs1, bs2, angle, target=control_qubit, physical_control_qubits=None
    )
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
    actual_ctrls, actual_ctrl_state, multiRX_actual = _build_multiRX(
        bs1, bs2, angle, target=control_qubit, physical_control_qubits=None
    )
    actual_circ.append(multiRX_actual, actual_ctrls + [control_qubit])

    assert expected_circ == actual_circ, (
        "Encountered inequivalent circuits. Expected:\n"
        f"{expected_circ.draw()}\nObtained:\n"
        f"{actual_circ.draw()}"
    )

    print(f"Test passed. Obtained circuit:\n{actual_circ.draw()}\n")


def _test_compute_LP_family():
    """Check computation of LP family works."""
    bs1 = "11000"
    bs2 = "10001"
    expected_LP_family = ["P1", "L-", "P0", "P0", "L+"]
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


def _test_prune_controls_fails_for_unequal_length_inputs():
    """IndexError should be raised if any of the bit strings in an input has a different length."""
    bs1 = "11001"
    bs2 = "11000"
    phys_states = {bs1, bs2, "11111", "1100"}
    print(
        "Checking that a wrong-length bit string in the physical "
        "states raises an IndexError:\n"
        f"bs1 = {bs1}\n"
        f"bs2 = {bs2}\n"
        f"phys_states = {phys_states}\n"
    )
    try:
        prune_controls(bs1, bs2, phys_states)
    except IndexError as e:
        print(f"Test passed. Raised error: {e}\n")
    else:
        assert False, "IndexError not raised."

    bs1 = "11001"
    bs2 = "1100"
    phys_states = {bs1, bs2, "11111", "11000"}
    print(
        "Checking that a wrong-length bit string in one of the input "
        "states raises an IndexError:\n"
        f"bs1 = {bs1}\n"
        f"bs2 = {bs2}\n"
        f"phys_states = {phys_states}\n"
    )
    try:
        print(f"bs1 = {bs1}\n" f"bs2 = {bs2}\n" f"phys_states = {phys_states}")
        prune_controls(bs1, bs2, phys_states)
    except IndexError as e:
        print(f"Test passed. Raised error: {e}\n")
    else:
        assert False, "IndexError not raised."
    try:
        print(f"bs1 = {bs2}\n" f"bs2 = {bs1}\n" f"phys_states = {phys_states}")
        prune_controls(bs2, bs1, phys_states)
    except IndexError as e:
        print(f"Test passed. Raised error: {e}\n")
    else:
        assert False, "IndexError not raised."


def _test_prune_controls_fails_for_non_bitstrings():
    """Non bit characters in any bit strings should cause a ValueError to be raised."""
    bs1 = "11001"
    bs2 = "11000"
    phys_states = {bs1, bs2, "11111", "1100a"}
    print(
        "Checking that a non-bit char in the physical "
        "states raises an ValueError:\n"
        f"bs1 = {bs1}\n"
        f"bs2 = {bs2}\n"
        f"phys_states = {phys_states}\n"
    )
    try:
        prune_controls(bs1, bs2, phys_states)
    except ValueError as e:
        print(f"Test passed. Raised error: {e}\n")
    else:
        assert False, "ValueError not raised."

    bs1 = "11001"
    bs2 = "110a0"
    phys_states = {bs1, bs2, "11111", "11000"}
    print(
        "Checking that a non-bit char in one of the input "
        "states raises an ValueError:\n"
        f"bs1 = {bs1}\n"
        f"bs2 = {bs2}\n"
        f"phys_states = {phys_states}\n"
    )
    try:
        print(f"bs1 = {bs1}\n" f"bs2 = {bs2}\n" f"phys_states = {phys_states}\n")
        prune_controls(bs1, bs2, phys_states)
    except ValueError as e:
        print(f"Test passed. Raised error: {e}\n")
    else:
        assert False, "ValueError not raised."
    try:
        print(f"bs1 = {bs2}\n" f"bs2 = {bs1}\n" f"phys_states = {phys_states}\n")
        prune_controls(bs2, bs1, phys_states)
    except ValueError as e:
        print(f"Test passed. Raised error: {e}\n")
    else:
        assert False, "ValueError not raised."


def _test_prune_controls_acts_as_expected():
    """Check the output of prune_controls for good input."""
    # Case 1, one control needed.
    bs1 = "11001001"
    bs2 = "11000110"
    phys_states = {bs1, bs2, "11111111"}
    expected_rep_LP_family = ["P1", "P1", "P0", "P0", "L-", "L+", "L+", "L-"]
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
    result_Q_set = prune_controls(bs1, bs2, phys_states)
    assert result_Q_set == expected_Q_set, f"result Q set = {result_Q_set}"
    print("Test passed.")

    # Case 2, two controls needed.
    bs1 = "11000011"
    bs2 = "11001001"  # Differs at 1 qubit at index 6, will be in Q set.
    phys_states = {bs1, bs2, "11111111", "00000000"}
    expected_rep_LP_family = ["P1", "P1", "P0", "P0", "L+", "P0", "L-", "P1"]
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

    print(
        "Testing the following prune_controls case:\n"
        f"bs1 = {bs1} (as representative)\n"
        f"bs2 = {bs2}\n"
        f"phys_states = {phys_states}"
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
    result_Q_set = prune_controls(bs1, bs2, phys_states)
    assert result_Q_set == expected_Q_set, f"result Q set = {result_Q_set}"
    print("Test passed.")


def _test_apply_LP_family():
    lp_family = ["L-", "L-", "L+", "P0", "L+", "P1"]
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
    pruned_controls = {}
    pruned_controls[("00000000", "01001010")] = [0, 2, 5, 6, 7]
    pruned_controls[("10001000", "11000010")] = [0, 2, 3, 4, 5]
    pruned_controls[("01100000", "00101010")] = [0, 2, 3, 4, 5, 6, 7]
    pruned_controls[("00001000", "01000010")] = [0, 2, 3, 4, 5, 6, 7]
    bin_value = "10110101"
    expected_control_fused_gates = [
        ([0, 2, 3, 5, 6, 7], "000000"),
        ([0, 2, 3, 4, 5, 6, 7], "0001000"),
        ([0, 2, 3, 5, 6, 7], "001000"),
        ([0, 2, 3, 4, 5], "01001"),
        ([0, 2, 3, 4, 5, 6, 7], "0101010"),
    ]
    code_generated_fusion = fuse_controls(bin_value, bitstring_list, pruned_controls)
    assert code_generated_fusion[1.0] == expected_control_fused_gates
    print("Control fusion test passed.")


def _test_binary_to_gray():
    bitstring_to_be_gray_ordered = [
        "1000",
        "0010",
        "0100",
        "0011",
        "0000",
        "0001",
        "0101",
    ]
    expected_gray_code_order = ["0000", "0001", "0011", "0010", "0100", "0101", "1000"]
    code_generated_gray_order = sorted(
        bitstring_to_be_gray_ordered, key=lambda x: binary_to_gray(x)
    )
    assert expected_gray_code_order == code_generated_gray_order
    print("Binary to Gray Order passed.")


def _test_givens_fused_controls():
    bitstring_list = [
        ("00000000", "01001010", 1.0),
        ("10001000", "11000010", 1.0),
        ("01100000", "00101010", 1.0),
        ("00001000", "01000010", 1.0),
    ]
    pruned_controls = {}
    pruned_controls[("00000000", "01001010")] = [0, 2, 5, 6, 7]
    pruned_controls[("10001000", "11000010")] = [0, 2, 3, 4, 5]
    pruned_controls[("01100000", "00101010")] = [0, 2, 3, 4, 5, 6, 7]
    pruned_controls[("00001000", "01000010")] = [0, 2, 3, 4, 5, 6, 7]
    bin_value = "10110101"
    generated_circuit = givens_fused_controls(
        bitstring_list, bin_value, physical_control_qubits=pruned_controls
    )
    expected_circuit = QuantumCircuit(8)
    expected_circuit.cx(1, 4)
    expected_circuit.cx(1, 6)
    expected_circuit.append(
        RXGate(1.0).control(num_ctrl_qubits=6, ctrl_state="000000"),
        [0, 2, 3, 5, 6, 7, 1],
    )
    expected_circuit.append(
        RXGate(1.0).control(num_ctrl_qubits=7, ctrl_state="0001000"),
        [0, 2, 3, 4, 5, 6, 7, 1],
    )
    expected_circuit.append(
        RXGate(1.0).control(num_ctrl_qubits=6, ctrl_state="000100"),
        [0, 2, 3, 5, 6, 7, 1],
    )
    expected_circuit.append(
        RXGate(1.0).control(num_ctrl_qubits=5, ctrl_state="10010"), [0, 2, 3, 4, 5, 1]
    )
    expected_circuit.append(
        RXGate(1.0).control(num_ctrl_qubits=7, ctrl_state="0101010"),
        [0, 2, 3, 4, 5, 6, 7, 1],
    )
    expected_circuit.cx(1, 4)
    expected_circuit.cx(1, 6)
    assert expected_circuit == generated_circuit
    print("givens with fused controls test passed.")


if __name__ == "__main__":
    _test_givens()
    # _test_givens2()  # TODO fix nondeterministic behavior of this test.
    _test_Xcirc()
    _test_multiRX()
    _test_compute_LP_family()
    _test_compute_LP_family_fails_on_bad_input()
    _test_compute_LP_family_fails_for_non_bitstrings()
    _test_prune_controls_fails_for_unequal_length_inputs()
    _test_prune_controls_fails_for_non_bitstrings()
    _test_prune_controls_acts_as_expected()
    _test_apply_LP_family()
    _test_eliminate_phys_states_that_differ_from_rep_at_Q_idx()
    _test_control_fusion()
    _test_binary_to_gray()
    _test_control_fusion_fails_for_inconsistent_lp_bin
    _test_givens_fused_controls
    print("All tests passed.")
