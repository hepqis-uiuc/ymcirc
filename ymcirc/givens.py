"""
This module contains methods and classes for constructing Givens rotation circuits.

Original author: Cianan Conefrey-Shinozaki, Oct 2024.

A Givens rotation is a generalization of an X-rotation; it rotates
two multi-qubit states into each other.
"""

from __future__ import annotations
from dataclasses import dataclass
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit import ControlledGate
from qiskit.circuit.library.standard_gates import RXGate, RZGate, RYGate
from qiskit.quantum_info import Operator, Statevector
from random import random
from math import isclose
import numpy as np
import copy
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
    num_ancillas: int = 0,
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

    If num_ancillas > 0, circuit construction uses v-chain
    gate synthesis using the provided number of ancillas. This
    increases qubit cost in order to reduce CX gate depth.
    """
    # Input validation and sanity checks.
    if len(bit_string_1) != len(bit_string_2):
        raise ValueError("Bit strings must be the same length.")

    num_qubits = len(bit_string_1)
    givens_circuit = QuantumCircuit(num_qubits)

    no_rotation_is_needed = (bit_string_1 == bit_string_2) or angle == 0
    if no_rotation_is_needed:
        return givens_circuit

    # Build the circuit.
    if num_qubits == 1:
        # No pre/post computation needed.
        givens_circuit.rx(angle, 0)
        return givens_circuit
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

        # adds an ancilla register is num_ancillas > 0
        if (num_ancillas > 0):
            givens_circuit.add_register(AncillaRegister(num_ancillas))

        Xcirc = _build_Xcirc(lp_fam, control=target)

        CRXcirc = _CRXCircuit_with_MCX([pruned_ctrl_state, pruned_ctrls], 
            angle, target, num_qubits, num_ancillas)

        # Assemble the final circuit.
        # Using inplace speeds up circuit composition.
        givens_circuit.compose(Xcirc, inplace=True)
        givens_circuit.compose(CRXcirc, inplace=True)
        givens_circuit.compose(Xcirc, inplace=True)

        if reverse is True:
            givens_circuit = givens_circuit.reverse_bits()
        return givens_circuit


def givens_fused_controls(
    lp_bin_w_angle: List[(str, str, float)],
    lp_bin: LPFamily,
    encoded_physical_states: Set[str] | None,
    num_ancillas: int = 0,
    reverse: bool = False,
) -> QuantumCircuit:
    """
    Implements givens rotation using the same logic as the givens function but for fused multiRX's.

    Inputs:
        - lp_bin_w_angle: list of bitstrings of the same LP family and the angle they have to be rotated by.
        - lp_bin: the LP family the bitstrings in the bin belong to.
        - encoded_physical_states: the set of physical states for control pruning 
        - num_ancillas: the number of ancillas to use in the local circuit. If num_ancillas > 0, circuit construction uses v-chain
                        gate synthesis using the provided number of ancillas. This
                        increases qubit cost in order to reduce CX gate depth.
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

    givens_circuit = QuantumCircuit(num_qubits)
    if (num_ancillas > 0):
        givens_circuit.add_register(AncillaRegister(num_ancillas))

    # Build the circuit.
    for idx in range(num_qubits):
        current_idx_is_target_idx = lp_bin.value[idx] == "L"
        if current_idx_is_target_idx is True:
            target = idx
            break
    Xcirc = _build_Xcirc(lp_bin, control=target)
    givens_circuit.compose(Xcirc, inplace=True)
    # First, fuse controls.
    angle_dict = fuse_controls(lp_bin, lp_bin_w_angle, round_close_angles=True)
    # Now, prune controls.
    # Note that this step will become redundant when control_pruning is turned off
    # i.e., when encoded_physical_states = None.
    for angle, ctrl_list in angle_dict.items():
        if angle == 0:  # Skip rotations that don't do anything.
            continue
        for ctrls, ctrl_state in ctrl_list:
            pruned_ctrls, pruned_ctrl_state = prune_controls(
                lp_bin, ctrls, ctrl_state, encoded_physical_states
            )
            crxcircuit = _CRXCircuit_with_MCX([pruned_ctrl_state, pruned_ctrls], 
        angle, target, num_qubits, num_ancillas)
            givens_circuit.compose(crxcircuit, inplace=True)

    # Assemble the final circuit.
    # Using inplace speeds up circuit composition.
    givens_circuit.compose(Xcirc, inplace=True)
    if reverse is True:
        givens_circuit = givens_circuit.reverse_bits()

    return givens_circuit


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
    # _CRXCircuit_with_MCX.
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


def _CRXGate(num_ctrls: int, ctrl_state: str, angle: float) -> ControlledGate:
    """Returns a RXGate given num_ctrls and ctrl_state"""
    return RXGate(angle).control(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state[::-1])


def _CRXCircuit_with_MCX(ctrl_list: List[Union[str, List[int]]], 
    angle: float, target: int, num_qubits: int, num_ancillas: int = 0) -> QuantumCircuit:
    """
    Input:
        - ctrl_List: [ctrls_state, ctrls], where ctrl_list corresponds to the control qubit indices and 
        ctrl_state corresponds to the state the controls are in.
        - angle: The rotation angle for the MCU
        - ctrl_state: The target qubit index for the rotation
        - num_qubits: The total qubits in the local rotation
        - num_ancillas: The number of ancillas used for the local rotation. If
                        num_ancillas > 0, circuit construction uses v-chain
                        gate synthesis using the provided number of ancillas.
                        This increases qubit cost in order to reduce CX gate depth.
    Output:
        - a QuantumCircuit with the circuit decomposition of the RXGate into MCXs 
        using the ABC decomposition (Corollary 4.2, Nielsen and Chuang). If ancillas are used,
        then the MCXs are decomposed into CXs with the v-chain method (arXiv:2408.01304, Algorithm 2)

    Note: Qubit indices are in little-endian notation
    """
    ctrl_state, ctrls = ctrl_list
    num_ctrls = len(ctrl_state)
    circ_with_mcx = QuantumCircuit(num_qubits)
    # Add an ancilla register for num_ancillas > 0 and uses v-chain decomposition for MCX 
    if (num_ancillas > 0):
        if (num_ancillas  < num_ctrls - 2):
            ancillas_needed = num_ctrls - 2
            raise ValueError(f"Too few ancillas for givens rotation. Require at least {ancillas_needed}")
        plaquette_ancilla_qubits = AncillaRegister(num_ancillas)
        circ_with_mcx.add_register(plaquette_ancilla_qubits)
        mode = 'v-chain'
    else:
        plaquette_ancilla_qubits = None
        mode = 'noancilla'
    circ_with_mcx.append(RZGate(-1.0*np.pi/2.0), [target])
    circ_with_mcx.append(RYGate(-1.0*angle/2.0), [target])
    circ_with_mcx.mcx(ctrls, target, ancilla_qubits=plaquette_ancilla_qubits, ctrl_state = ctrl_state[::-1], mode=mode)
    circ_with_mcx.append(RYGate(1.0*angle/2.0), [target])
    circ_with_mcx.mcx(ctrls, target, ancilla_qubits=plaquette_ancilla_qubits, ctrl_state = ctrl_state[::-1], mode=mode)
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
