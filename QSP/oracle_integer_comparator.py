#!/usr/bin/env python

from __future__ import annotations
from collections.abc import Iterable
from itertools import product
import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)

import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import SwapGate
from qiskit.quantum_info import Operator, Statevector


def get_min_qubit_requirements(number: int | list[int]):
    """
    Return number of qubits required for binary encoding of number.

    If input is a list of ints, find the biggest one first, then
    give the qubit requiremnts for that number.
    """
    if isinstance(number, int):
        return len(np.binary_repr(number))
    elif isinstance(number, list):
        return len(np.binary_repr(max(set(number))))
    else:
        raise ValueError("Expected a number, or a list of numbers.")


def flatten(iterable) -> list:
    """Return iterable as a flattened list."""
    # Base case for recursion.
    if not isinstance(iterable, Iterable):
        return iterable

    result = []
    for item in iterable:
        flattened_item = flatten(item)
        result += flattened_item if isinstance(flattened_item, Iterable) \
            else [flattened_item]

    return result


def oracle_QBSC(specs: list[tuple[int, int]]):
    """
    |0>|i>|j> --> | i>j >|i>|j>

    Takes i and j to be length k bit strings (if one is smaller, it is padded).
    Initializes a scratch quantum register of k qubits.

    Assumes little-endian indexing.

    Implementation of:
    @article{oliveira2007quantum,
        title={Quantum bit string comparator: circuits and applications},
        author={Oliveira, David Sena and Ramos, Rubens Viana},
        journal={Quantum Comput. Comput},
        volume={7},
        number={1},
        pages={17--26},
        year={2007}
    }
    """
    n_qubits_per_reg = get_min_qubit_requirements(flatten(specs))
    n_total_qubits = 5*n_qubits_per_reg - 1
    j_reg_zero_idx = 0
    j_reg_most_sig_idx = j_reg_zero_idx + (n_qubits_per_reg - 1)
    i_reg_zero_idx = j_reg_most_sig_idx + 1
    i_reg_most_sig_idx = i_reg_zero_idx + (n_qubits_per_reg - 1)
    active_domain_reg_zero_idx = i_reg_most_sig_idx + 1
    active_domain_reg_most_sig_idx = active_domain_reg_zero_idx + (n_qubits_per_reg - 1) - 1
    bit_compare_gr_reg_zero_idx = active_domain_reg_most_sig_idx + 1
    bit_compare_gr_reg_most_sig_idx = bit_compare_gr_reg_zero_idx + (n_qubits_per_reg - 1)
    bit_compare_le_reg_zero_idx = bit_compare_gr_reg_most_sig_idx + 1
    bit_compare_le_reg_most_sig_idx = bit_compare_le_reg_zero_idx + (n_qubits_per_reg - 1)

    # Build the bitwise comparator circuit.
    circ_Uc = QuantumCircuit(4)
    circ_Uc.mcx(control_qubits=[0, 1], target_qubit=2, ctrl_state="10")
    circ_Uc.mcx(control_qubits=[0, 1], target_qubit=3, ctrl_state="01")

    # Build the main circuit.
    circ = QuantumCircuit(n_total_qubits)
    n_qubits_per_reg = get_min_qubit_requirements(flatten(specs))
    for idx in range(n_qubits_per_reg):
        # Apply the Uc subcircuit.
        target_Uc_i = bit_compare_gr_reg_most_sig_idx - idx
        target_Uc_j = bit_compare_le_reg_most_sig_idx - idx
        circ.compose(circ_Uc, qubits=[
            i_reg_most_sig_idx - idx,
            j_reg_most_sig_idx - idx,
            target_Uc_i,
            target_Uc_j
        ], inplace=True)

        # Apply the transfer of domain circuit (if applicable).
        if idx < n_qubits_per_reg - 1:
            active_domain_current_idx = active_domain_reg_most_sig_idx - idx
            circ.mcx(
                control_qubits=[target_Uc_i, target_Uc_j],
                target_qubit=active_domain_current_idx,
                ctrl_state="00")

    # Aggregate results
    for idx in range(n_qubits_per_reg - 1):
        target_Uc_i_current = bit_compare_gr_reg_zero_idx + idx
        target_Uc_j_current = bit_compare_le_reg_zero_idx + idx
        target_Uc_i_next = bit_compare_gr_reg_zero_idx + idx + 1
        target_Uc_j_next = bit_compare_le_reg_zero_idx + idx + 1
        active_domain_current_idx = active_domain_reg_zero_idx + idx
        circ.mcx(
            control_qubits=[target_Uc_i_current, active_domain_current_idx],
            target_qubit=target_Uc_i_next
        )
        circ.mcx(
            control_qubits=[target_Uc_j_current, active_domain_current_idx],
            target_qubit=target_Uc_j_next
        )

    return circ


def oracle_integer_comparison_no_swaps(specs: list[tuple[int, int]]):
    """
    |0>|i>|j> --> | i>j >|i>|j>

    Takes i and j to be length k bit strings (if one is smaller, it is padded).
    Initializes a scratch quantum register of k qubits.

    Assumes little-endian indexing.
    """
    # For later convenience.
    n_qubits_per_reg = get_min_qubit_requirements(flatten(specs))
    n_total_qubits = 4*n_qubits_per_reg + 1
    j_reg_zero_idx = 0
    j_reg_most_sig_idx = j_reg_zero_idx + (n_qubits_per_reg - 1)
    i_reg_zero_idx = j_reg_most_sig_idx + 1
    i_reg_most_sig_idx = i_reg_zero_idx + (n_qubits_per_reg - 1)
    active_domain_reg_zero_idx = i_reg_most_sig_idx + 1
    active_domain_reg_most_sig_idx = active_domain_reg_zero_idx + (n_qubits_per_reg - 1)
    bit_compare_reg_zero_idx = active_domain_reg_most_sig_idx + 1
    bit_compare_reg_most_sig_idx = bit_compare_reg_zero_idx + (n_qubits_per_reg - 1)
    output_reg_idx = n_total_qubits - 1

    circ = QuantumCircuit(n_total_qubits)


    # Flip most significant qubit in domain register as initialization step.
    circ.x(active_domain_reg_most_sig_idx)
    for k in range(n_qubits_per_reg):
        # Set bit compare reg to current qubit.
        i_reg_current_idx = i_reg_most_sig_idx - k
        j_reg_current_idx = j_reg_most_sig_idx - k
        bit_compare_reg_current_idx = bit_compare_reg_most_sig_idx - k
        bit_compare_ctrl_string = "01"
        bit_compare_ctrl_idxes = [i_reg_current_idx, j_reg_current_idx]
        circ.mcx(control_qubits=bit_compare_ctrl_idxes, target_qubit=bit_compare_reg_current_idx, ctrl_state=bit_compare_ctrl_string)

        # Check whether to transfer active domain.
        ad_current_idx = active_domain_reg_most_sig_idx - k
        ad_next_idx = ad_current_idx - 1
        bit_compare_already_tested_idxes = [bit_compare_reg_most_sig_idx - l for l in range(k + 1)]
        domain_transfer_ctrl_idxes = bit_compare_ctrl_idxes + bit_compare_already_tested_idxes
        bc_off_ij_on_ctrl_string = ("0" * len(bit_compare_already_tested_idxes) )+ "11"
        bc_off_ij_off_ctrl_string = ("0" * len(bit_compare_already_tested_idxes)) + "00"
        for idx, ctrl_str in product(
                [ad_current_idx, ad_next_idx], [bc_off_ij_on_ctrl_string, bc_off_ij_off_ctrl_string]):
            circ.mcx(control_qubits=domain_transfer_ctrl_idxes, target_qubit=idx, ctrl_state=ctrl_str)

    # Save the reversed circuit for working on the scratch qubits
    # for a final cleanup step.
    cleanup_circuit = circ.reverse_ops()

    # Final accumulation sweep.
    for k in range(n_qubits_per_reg):
        ad_current_idx = active_domain_reg_zero_idx + k
        bc_current_idx = bit_compare_reg_zero_idx + k
        circ.mcx(
            control_qubits=[ad_current_idx, bc_current_idx],
            target_qubit=output_reg_idx
        )

    return circ.compose(cleanup_circuit)


def oracle_integer_comparison_via_swaps(specs: list[tuple[int, int]]):
    """
    |0>|i>|j> --> | i>j >|i>|j>

    Takes i and j to be length k bit strings (if one is smaller, it is padded).
    Initializes a scratch quantum register of k qubits.

    Assumes little-endian indexing.
    """
    # For later convenience.
    n_qubits_per_reg = get_min_qubit_requirements(flatten(specs))
    most_sig_qubit_idx = n_qubits_per_reg - 1
    j_reg = QuantumRegister(name="j", size=n_qubits_per_reg)
    i_reg = QuantumRegister(name="i", size=n_qubits_per_reg)
    significance_reg = AncillaRegister(name="s", size=n_qubits_per_reg)
    result_reg = QuantumRegister(name="r", size=1)
    registers = [j_reg, i_reg, significance_reg, result_reg]

    # Initialize the circuit.
    circ = QuantumCircuit(*registers)

    # Sweep to determine which qubit in the significance
    # register should determine the output of the
    # algorithm. At most one qubit in the significance
    # register will be flipped.
    circ.x(significance_reg[most_sig_qubit_idx])
    for k in range(n_qubits_per_reg - 1):
        current_ctrl_idx = most_sig_qubit_idx - k
        sig_idx1 = most_sig_qubit_idx - k
        sig_idx2 = most_sig_qubit_idx - (k + 1)
        sig_bit_cswap_if_both_off = SwapGate().control(
            num_ctrl_qubits=2, ctrl_state="00")
        sig_bit_cswap_if_both_on = SwapGate().control(
            num_ctrl_qubits=2, ctrl_state="11")
        circ.append(sig_bit_cswap_if_both_off, [
            i_reg[current_ctrl_idx],
            j_reg[current_ctrl_idx],
            significance_reg[sig_idx1],
            significance_reg[sig_idx2]
        ])
        circ.append(sig_bit_cswap_if_both_on, [
            i_reg[current_ctrl_idx],
            j_reg[current_ctrl_idx],
            significance_reg[sig_idx1],
            significance_reg[sig_idx2]
        ])

    # Save circuit so far for uncomputing ancillae.
    cleanup_circ = circ.reverse_ops()

    # Sweep through significance register to check
    # which qubit comparison between the i and j registers
    # should set the final result.
    for k in range(n_qubits_per_reg):
        current_ctrl_idx = most_sig_qubit_idx - k
        circ.mcx(
            control_qubits=[
                i_reg[current_ctrl_idx],
                j_reg[current_ctrl_idx],
                significance_reg[current_ctrl_idx]
            ],
            target_qubit=result_reg[0],
            ctrl_state="101"
        )

    return circ.compose(cleanup_circ)


def test_flatten():
    """Check that the list flatten method works."""
    print(flatten(2) == 2)
    print(flatten([1, 3, 4]) == [1, 3, 4])
    print(flatten([]) == [])
    print(flatten([(1, 3), (3, 1), (2, 4), (4, 2)]) == [1, 3, 3, 1, 2, 4, 4, 2])
    print(flatten([(1, 3), (3, 1), (2, 4), (4, 2)]))
    print(flatten([(1, 2), 3, [4]]) == [1, 2, 3, 4])


def test_oracle_integer_comparison_no_swaps(oracle, specification):
    """Test the no-swaps comparison circuit."""
    # Qubits per i, j, and scratch registers
    m = get_min_qubit_requirements(flatten(specification))

    test_results = {
        "passed_cases": [],
        "failed_cases": []
    }

    for i, j in specification:
        output_reg_value = 1 if i > j else 0
        print(
            f"Testing |0>|i>|j> = |0>|{i}>|{j}> --> "
            f"|{output_reg_value}>|{i}>|{j}> = | i>j >|i>|j>"
        )
        print(
            "For binary lovers:  "
            f"|0>|{np.binary_repr(i, m)}>|{np.binary_repr(j, m)}> --> "
            f"|{output_reg_value}>|{np.binary_repr(i, m)}>"
            f"|{np.binary_repr(j, m)}>"
        )
        # initial output reg val + m initial ancilla reg val + m initial ancilla reg val + i reg + j reg
        sv = Statevector.from_label(
            "0" + ("0" * m) + ("0" * m) + np.binary_repr(i, m) + np.binary_repr(j, m))
        # final output reg val + 2m uncomputed (i.e. unchanged) ancilla reg val + i reg + j reg
        sv_expected = Statevector.from_label(
            str(output_reg_value) + ("0" * m) + ("0" * m) + np.binary_repr(i, m) + np.binary_repr(j, m))
        sv_evolved = sv.evolve(oracle)
        print("Test passed:", sv_expected == sv_evolved)
        if not sv_expected == sv_evolved:
            test_results["failed_cases"].append((i, j))
            print("Debug info:")
            print("Equiv up to global phase:", sv_evolved.equiv(sv_expected))
            print("Initialized state vector:\n", sv.to_dict())
            print("Expected state vector:\n", sv_expected.to_dict())
            print("Evolved state vector:\n", sv_evolved.to_dict())
        else:
            test_results["passed_cases"].append((i, j))

    print(
        f"Checked all {m} combinations of {m}-bit integer comparisons. "
        f"There were {len(test_results['passed_cases'])} passes "
        f"and {len(test_results['failed_cases'])} failures.")


def test_oracle_QBSC(oracle, specification):
    """
    Not sure what the full output state should look like because there's no uncomputation.
    This just logs the output for manual inspection.
    """
    # Qubits per i, j, and scratch registers
    m = get_min_qubit_requirements(flatten(specification))

    for i, j in specification:
        output_reg_value = 1 if i > j else 0
        print(
            f"Testing |0>|i>|j> = |0>|{i}>|{j}> --> "
            f"|{output_reg_value}>|{i}>|{j}> = | i>j >|i>|j>"
        )
        print(
            "For binary lovers:  "
            f"|0>|{np.binary_repr(i, m)}>|{np.binary_repr(j, m)}> --> "
            f"|{output_reg_value}>|{np.binary_repr(i, m)}>"
            f"|{np.binary_repr(j, m)}>"
        )
        # m intial ancilla reg val +  m initial ancilla reg val + m - 1 initial ancilla reg val + i reg + j reg
        sv = Statevector.from_label(
            ("0" * m) + ("0" * m) + ("0" * (m - 1)) + np.binary_repr(i, m) + np.binary_repr(j, m))
        sv_evolved = sv.evolve(oracle)
        print("Debug info:")
        print("Initialized state vector:\n", sv.to_dict())
        print("Evolved state vector:\n", sv_evolved.to_dict())


def test_oracle_integer_comparison_via_swaps(oracle, specification):
    # Qubits per i, j, and scratch registers
    m = get_min_qubit_requirements(flatten(specification))
    test_results = {
        "passed_cases": [],
        "failed_cases": []
    }

    for i, j in specification:
        output_reg_value = 1 if i > j else 0
        print(
            f"Testing |0>|i>|j> = |0>|{i}>|{j}> --> "
            f"|{output_reg_value}>|{i}>|{j}> = | i>j >|i>|j>"
        )
        print(
            "For binary lovers:  "
            f"|0>|{np.binary_repr(i, m)}>|{np.binary_repr(j, m)}> --> "
            f"|{output_reg_value}>|{np.binary_repr(i, m)}>"
            f"|{np.binary_repr(j, m)}>"
        )
        logging.info("Creating input Statevector...")
        # initial output reg val + initial ancilla reg val + i reg + j reg
        sv = Statevector.from_label(
            "0" + ("0" * m) + np.binary_repr(i, m) + np.binary_repr(j, m))
        logging.info("Done.")
        logging.info("Creating expected output Statevector...")
        # final output reg val + uncomputed (i.e. unchanged) ancilla reg val + i reg + j reg
        sv_expected = Statevector.from_label(
            str(output_reg_value) + ("0" * m) + np.binary_repr(i, m) + np.binary_repr(j, m))
        logging.info("Done.")
        logging.info("Evolving input Statevector with circuit...")
        sv_evolved = sv.evolve(oracle)
        logging.info("Done.")
        print("Test passed:", sv_expected == sv_evolved)
        if not sv_expected == sv_evolved:
            test_results["failed_cases"].append((i, j))
            print("Equiv up to global phase:", sv_evolved.equiv(sv_expected))
            logging.info("Debug info for failed test:")
            logging.info("Initialized state vector:\n", sv.to_dict())
            logging.info("Expected state vector:\n", sv_expected.to_dict())
            logging.info("Evolved state vector:\n", sv_evolved.to_dict())
        else:
            test_results["passed_cases"].append((i, j))

    print(
        f"Checked all {m} combinations of {m}-bit integer comparisons. "
        f"There were {len(test_results['passed_cases'])} passes "
        f"and {len(test_results['failed_cases'])} failures.")


def test():
    print("Checking that flatten method works...")
    test_flatten()

    print("Checking that integer comparison oracle(s) work(s).")
    #spec = [(1, 2), (2, 1), (7, 4), (4, 7)]
    n_qubits_for_test = 4
    spec = [(i, j) for i in range(2**n_qubits_for_test) for j in range(2**n_qubits_for_test)]

    oracle = oracle_integer_comparison_via_swaps(spec)
    print(oracle)
    test_oracle_integer_comparison_via_swaps(oracle, spec)

    # oracle = oracle_QBSC(spec)
    # print(oracle)
    # test_oracle_QBSC(oracle, spec)

    # oracle = oracle_integer_comparison_no_swaps(spec)
    # print(oracle)
    # test_oracle_integer_comparison_no_swaps(oracle, spec)


if __name__ == "__main__":
    test()
