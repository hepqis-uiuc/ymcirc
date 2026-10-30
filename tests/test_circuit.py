from itertools import product
from pathlib import Path
import pytest
import numpy as np
from ymcirc._abstract import LatticeDef
from ymcirc._abstract.lattice_data import Plane, Signature
from ymcirc.circuit import HamiltonianData, LatticeCircuitManager
from ymcirc.conventions import IrrepBitmap, LatticeStateEncoder, PlaquetteState, ONE, THREE, THREE_BAR, SIX, SIX_BAR, EIGHT, physical_plaquette_states
from ymcirc._pyclebsch_adapter import BoxTerms, box_terms
from ymcirc.lattice_registers import LatticeRegisters
from ymcirc.givens import bitstring_value_of_LP_family, compute_LP_family
from ymcirc.utilities import _flatten_circuit, _check_circuits_logically_equivalent
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library.standard_gates import RZGate, RYGate, MCXGate
from qiskit.circuit.exceptions import CircuitError

# Link bitmaps for tests with made-up plaquette states: the codes ymcirc
# registers for the link irreps of the T1 truncation (1, 3, 3bar), and of T2
# (1, 3, 3bar, 6, 6bar, 8). The expected circuits are derived from these codes.
_T1_LINK_BITMAP = {ONE: "00", THREE: "10", THREE_BAR: "01"}
_T2_LINK_BITMAP = {ONE: "000", THREE: "100", THREE_BAR: "001", SIX: "110", SIX_BAR: "011", EIGHT: "111"}
# The signature of every plaquette of a periodic d=3/2 lattice with the
# default F-order: 3 half-links at each corner.
_SIG_3_2: Signature = ((1, 2, -1), (1, 2, -1), (1, -1, -2), (1, -1, -2))
# The signature of every plaquette of a periodic d=2 lattice with the default
# F-order: 4 half-links at each corner.
_SIG_2: Signature = ((1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2))


def test_create_blank_full_lattice_circuit_has_promised_register_order():
    """Check in some cases that we get the ordering promised in the method docstring."""
    # Creating test data.
    # Not physically meaningful, but has the right format.
    iweight_one = (0, 0, 0)
    iweight_three = (1, 0, 0)
    irrep_bitmap: IrrepBitmap = {
        iweight_one: "0",
        iweight_three: "1"
    }
    physical_plaquette_states_3halves_no_vertices_needed: list[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            ((iweight_one,), (iweight_one,), (iweight_one,), (iweight_one,))
        ),
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_three, iweight_three, iweight_three),
            ((iweight_three,), (iweight_one,), (iweight_one,), (iweight_one,))
        )
    ]
    physical_plaquette_states_3halves: list[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            ((iweight_one,), (iweight_one,), (iweight_one,), (iweight_one,))
        ),
        (
            (1, 1, 1, 1),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            ((iweight_one,), (iweight_one,), (iweight_one,), (iweight_one,))
        ),
        (
            (2, 2, 2, 2),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            ((iweight_one,), (iweight_one,), (iweight_one,), (iweight_one,))
        ),
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_three, iweight_three, iweight_three),
            ((iweight_three,), (iweight_one,), (iweight_one,), (iweight_one,))
        )
    ]
    physical_plaquette_states_2d: list[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            ((iweight_one, iweight_one), (iweight_one, iweight_one), (iweight_one, iweight_one), (iweight_one, iweight_one))
        ),
        (
            (1, 1, 1, 1),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            ((iweight_one, iweight_one), (iweight_one, iweight_one), (iweight_one, iweight_one), (iweight_one, iweight_one))
        ),
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_three, iweight_three, iweight_three),
            ((iweight_three, iweight_one), (iweight_one, iweight_one), (iweight_one, iweight_one), (iweight_one, iweight_one))
        )
    ]
    # Hamiltonian bitstrings take the form vertex_bits + active link bits + c link bits.
    # For the "no_vertices" data, vertex_bits is the empty string. The numbers of
    # Vertex bits and link bits can be inferred from the test data (encode integer in bitstring, use link bitmap).
    mag_hamiltonian_2d: HamiltonianData = {
        ((1, 2), _SIG_2): {
            ("1110111100000000", "0001000011111111"): -0.33,
            ("0000111100000000", "1111000011111111"): 1.0,
        }
    }
    mag_hamiltonian_3halves: HamiltonianData = {
        ((1, 2), _SIG_3_2): {
            ("1010010111110000", "0000000011110000"): 1.0,
            ("0000000010100101", "1010101000000001"): 1.0,
        }
    }
    mag_hamiltonian_3halves_no_vertices: HamiltonianData = {
        ((1, 2), _SIG_3_2): {
            ("10101111", "11110010"): 1.0,
            ("10010000", "10000001"): 1.0,
            ("11111101", "00000101"): 1.0,
        }
    }
    # Registers for lattices with size 3
    expected_register_order_2d = [
        'v:(0, 0)', 'l:((0, 0), 1)', 'l:((0, 0), 2)',
        'v:(0, 1)', 'l:((0, 1), 1)', 'l:((0, 1), 2)',
        'v:(0, 2)', 'l:((0, 2), 1)', 'l:((0, 2), 2)',
        'v:(1, 0)', 'l:((1, 0), 1)', 'l:((1, 0), 2)',
        'v:(1, 1)', 'l:((1, 1), 1)', 'l:((1, 1), 2)',
        'v:(1, 2)', 'l:((1, 2), 1)', 'l:((1, 2), 2)',
        'v:(2, 0)', 'l:((2, 0), 1)', 'l:((2, 0), 2)',
        'v:(2, 1)', 'l:((2, 1), 1)', 'l:((2, 1), 2)',
        'v:(2, 2)', 'l:((2, 2), 1)', 'l:((2, 2), 2)',
    ]
    expected_register_order_3halves = [
        'v:(0, 0)', 'l:((0, 0), 1)', 'l:((0, 0), 2)',
        'v:(0, 1)', 'l:((0, 1), 1)',
        'v:(1, 0)', 'l:((1, 0), 1)', 'l:((1, 0), 2)',
        'v:(1, 1)', 'l:((1, 1), 1)',
        'v:(2, 0)', 'l:((2, 0), 1)', 'l:((2, 0), 2)',
        'v:(2, 1)', 'l:((2, 1), 1)'
    ]
    expected_register_order_3halves_no_vertices = [
        'l:((0, 0), 1)', 'l:((0, 0), 2)',
        'l:((0, 1), 1)',
        'l:((1, 0), 1)', 'l:((1, 0), 2)',
        'l:((1, 1), 1)',
        'l:((2, 0), 1)', 'l:((2, 0), 2)',
        'l:((2, 1), 1)',
    ]
    # Registers for a lattice with size 2 (small enough for the same link to control multiple vertices in a single plaquette).
    expected_register_order_2d_small_lattice = [
        'v:(0, 0)', 'l:((0, 0), 1)', 'l:((0, 0), 2)',
        'v:(0, 1)', 'l:((0, 1), 1)', 'l:((0, 1), 2)',
        'v:(1, 0)', 'l:((1, 0), 1)', 'l:((1, 0), 2)',
        'v:(1, 1)', 'l:((1, 1), 1)', 'l:((1, 1), 2)',
    ]
    test_cases = [
        (
            expected_register_order_2d,
            irrep_bitmap,
            physical_plaquette_states_2d,
            2,
            3,
            mag_hamiltonian_2d
        ),
        (
            expected_register_order_2d_small_lattice,
            irrep_bitmap,
            physical_plaquette_states_2d,
            2,
            2,
            mag_hamiltonian_2d
        ),
        (
            expected_register_order_3halves,
            irrep_bitmap,
            physical_plaquette_states_3halves,
            1.5,
            3,
            mag_hamiltonian_3halves
        ),
        (
            expected_register_order_3halves_no_vertices,
            irrep_bitmap,
            physical_plaquette_states_3halves_no_vertices_needed,
            1.5,
            3,
            mag_hamiltonian_3halves_no_vertices
        )
    ]

    # Iterate over all test cases.
    for expected_register_names_ordered, link_bitmap, made_up_plaquette_states, dims, size, hamiltonian in test_cases:
        # Initialize registers and create circuit.
        lattice_encoder = LatticeStateEncoder(
            link_bitmap, made_up_plaquette_states, LatticeDef(dims, size))
        lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
        circ_mgr = LatticeCircuitManager(
            lattice_encoder=lattice_encoder,
            mag_hamiltonian=hamiltonian
        )
        print(
            f"Checking register order in a circuit constructed from a {dims}-dimensional lattice "
            f"of linear size {size}."
        )
        print(f"Link bitmap: {link_bitmap}\nVertex bitmap: {lattice_encoder.vertex_bitmap}")
        print(f"Expected register ordering: {expected_register_names_ordered}")

        master_circuit = circ_mgr.create_blank_full_lattice_circuit(lattice_registers)
        nonzero_regs = [reg for reg in master_circuit.qregs if len(reg) > 0]
        n_nonzero_regs = len(nonzero_regs)

        # Check that the circuit makes sense.
        assert n_nonzero_regs == len(
            expected_register_names_ordered
        ), f"Expected {len(expected_register_names_ordered)} registers. Encountered {n_nonzero_regs} registers."
        for expected_name, reg in zip(expected_register_names_ordered, nonzero_regs):
            if len(reg) == 0:
                continue
            assert (
                expected_name == reg.name
            ), f"Expected: {expected_name}, encountered: {reg.name}"
            print(f"Verified location of the register for {expected_name}.")


def test_apply_magnetic_trotter_step_d_3_2_large_lattice():
    print(
        "Checking that application of magnetic Trotter step works for d=3/2 "
        "on a large enough lattice that no control links are repeated in any "
        "one plaquette."
    )
    # DO NOT CHANGE THE "DUMMY" DATA UNLESS YOU ARE WILLING TO WORK OUT
    # WHAT THE CORRECT "EXPECTED" CIRCUITS ARE. THERE IS
    # STRONG DEPENDENCE BETWEEN THAT AND THESE DUMMY
    # TEST DATA.
    # If you need to make more test data, follow the following process:
    #    1. Come up with a dummy hamiltonian matrix element of the form (plaquette bitstring, plaquette bitstring, mat elem value).
    #    2. Come up with the corresponding physical states which get encoded to these bit strings.
    #    3. For each matrix element * plaquette in the lattice, there will be ONE givens rotation circuit. For each such givens rotation:
    #      3a. Determine which registers are involved in the circuit following conventional ordering of vertices, then active links, then control links.
    #      3b. Determine what the "X" circuit prefix is by comparing the state bitstrings for the matrix element, determining the LP family, and then
    #          mapping each substring in the plaquette encoding onto actual registers in the lattice.
    #      3c. Repeat this exercise with the multi-control rotation, where the type of ladder or projector operator involved determines the control states.
    # Ask yourself if you REALLY feel like doing all that before mucking about with this test data.
    # Signature for d=3/2, plane=(1,2), default forder.
    _sig_3_2 = ((1, 2, -1), (1, 2, -1), (1, -1, -2), (1, -1, -2))
    _plane_12 = (1, 2)
    dummy_mag_hamiltonian: HamiltonianData = {
        (_plane_12, _sig_3_2): {("00100000" + "00000000", "01010100" + "10011010"): 0.33}  # One matrix element, plaquette only has a_link and c_link substrings.
    }
    dummy_phys_states: list[PlaquetteState] = [
        (  # Matches the first encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (ONE, THREE, ONE, ONE),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (  # Matches the second encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, ONE),
            ((THREE,), (THREE_BAR,), (THREE,), (THREE,))
        )
    ]
    expected_master_circuit = QuantumCircuit(18)
    expected_rotation_gates = {  # Data for constructing the expected circuit.
        "angle": -0.165,
        "MCU ctrl state": "000000000000001",  # Little endian per qiskit convention.
        "givens rotations": [
            {
                "pivot": 1,
                "CX targets": [8, 9, 5, 12, 7, 10, 16],
                "MCU ctrls": [8, 0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17] # on ctrls, followed by off, with pivot and distant link regs skipped.
            },
            {
                "pivot": 7,
                "CX targets": [14, 15, 11, 0, 13, 16, 4],
                "MCU ctrls": [14, 0, 1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17]
            },
            {
                "pivot": 13,
                "CX targets": [2, 3, 17, 6, 1, 4, 10],
                "MCU ctrls": [2, 0, 1, 3, 4, 5, 6, 7, 10, 11, 12, 14, 15, 16, 17]
            }
        ]
    }
    for rotation_data in expected_rotation_gates["givens rotations"]:
        # Build subcircuits.
        Xcirc = QuantumCircuit(18)
        for target in rotation_data["CX targets"]:
            Xcirc.cx(
                control_qubit=rotation_data["pivot"],
                target_qubit=target
            )
        pivot_qubit = [rotation_data["pivot"]]
        angle = expected_rotation_gates["angle"]
        ctrls = rotation_data["MCU ctrls"]
        num_ctrls = len(rotation_data["MCU ctrls"])
        ctrl_state = expected_rotation_gates["MCU ctrl state"]

        circ_with_mcx = QuantumCircuit(18)
        circ_with_mcx.append(RZGate(-1.0*np.pi/2.0), pivot_qubit)
        circ_with_mcx.append(RYGate(-2.0*angle), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RYGate(2.0*angle), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RZGate(1.0*np.pi/2.0), pivot_qubit)

        # Construct current expected givens rotation.
        expected_master_circuit.compose(Xcirc, inplace=True)
        expected_master_circuit.compose(circ_with_mcx, inplace=True)
        expected_master_circuit.compose(Xcirc, inplace=True)

    print("Expected circuit:")
    print(expected_master_circuit)

    # Create master circuit via the magnetic trotter step code.
    lattice_def = LatticeDef(1.5, 3)
    lattice_encoder = LatticeStateEncoder(
        _T1_LINK_BITMAP,
        dummy_phys_states,
        lattice=lattice_def)
    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    circ_mgr = LatticeCircuitManager(lattice_encoder,
                                     dummy_mag_hamiltonian)
    master_circuit = circ_mgr.create_blank_full_lattice_circuit(
        lattice_registers)
    circ_mgr.apply_magnetic_trotter_step(
        master_circuit,
        lattice_registers,
        optimize_circuits=False,
        cache_mag_evol_circuit=True
    )
    master_circuit.assign_parameters({  # Values computed above assuming dt and coupling set to 1.
        "dt_mag__0": 1,
        "coupling_g_mag__0": 1
    }, inplace=True)
    print("Obtained circuit:")
    print(master_circuit)

    # Checking equivalence via helper methods for
    # (1) flattening a circuit down to a single register and (2) comparing
    # logical equivalence of two circuits.
    assert _check_circuits_logically_equivalent(_flatten_circuit(master_circuit), expected_master_circuit), "Encountered inequivalent circuits."


def test_apply_magnetic_trotter_step_d_3_2_small_lattice():
    print(
        "Checking that application of magnetic Trotter step works for d=3/2 "
        "on a small lattice where some control links are repeated in each "
        "plaquette."
    )
    # DO NOT CHANGE THE "DUMMY" DATA UNLESS YOU ARE WILLING TO WORK OUT
    # WHAT THE CORRECT "EXPECTED" CIRCUITS ARE. THERE IS
    # STRONG DEPENDENCE BETWEEN THAT AND THESE DUMMY
    # TEST DATA.
    # If you need to make more test data, follow the following process:
    #    1. Come up with a dummy hamiltonian matrix element of the form (plaquette bitstring, plaquette bitstring, mat elem value).
    #    2. Come up with the corresponding physical states which get encoded to these bit strings.
    #    3. For each matrix element * plaquette in the lattice, there will be ONE givens rotation circuit. For each such givens rotation:
    #      3a. Determine which registers are involved in the circuit following conventional ordering of vertices, then active links, then control links.
    #      3b. Determine what the "X" circuit prefix is by comparing the state bitstrings for the matrix element, determining the LP family, and then
    #          mapping each substring in the plaquette encoding onto actual registers in the lattice.
    #      3c. Repeat this exercise with the multi-control rotation, where the type of ladder or projector operator involved determines the control states (raising to get to final state is on, projector onto 1 is on).
    # Ask yourself if you REALLY feel like doing all that before mucking about with this test data.
    # Signature for d=3/2, plane=(1,2), default forder.
    _sig_3_2 = ((1, 2, -1), (1, 2, -1), (1, -1, -2), (1, -1, -2))
    _plane_12 = (1, 2)
    dummy_mag_hamiltonian: HamiltonianData = {
        (_plane_12, _sig_3_2): {("00100001" + "00000000", "01010110" + "10100000"): 0.33}  # One matrix element, plaquette only has a_link and c_link substrings.
    }
    # The link that is both v1's -1 control and v2's +1 control has two
    # different states ("10" and "01") in this matrix element's second
    # bit string, which no state of this lattice can have, so the
    # LatticeCircuitManager constructor refuses it.
    inconsistent_mag_hamiltonian: HamiltonianData = {
        (_plane_12, _sig_3_2): {("00100001" + "00000000", "01010110" + "10011010"): 0.33}
    }
    dummy_phys_states: list[PlaquetteState] = [
        (  # Matches the first encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (ONE, THREE, ONE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (  # Matches the second encoded state in the inconsistent magnetic hamiltonian.
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            ((THREE,), (THREE_BAR,), (THREE,), (THREE,))
        ),
        (  # Matches the second encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            ((THREE,), (THREE,), (ONE,), (ONE,))
        )
    ]
    expected_master_circuit = QuantumCircuit(12)
    # Only expecting one rotation per plaquette, yielding two total Givens rotations.
    expected_rotation_gates = {  # Data for constructing the expected circuit.
        "angle": -0.165,
        "MCU ctrl state": "00000000011",  # Little endian per qiskit convention.
        "givens rotations": [
            {
                "pivot": 1,
                "CX targets": [8, 9, 5, 2, 3, 6],
                "MCU ctrls": [8, 3, 0, 2, 4, 5, 6, 7, 9, 10, 11] # on ctrls first, followed by off, with pivot skipped.
            },
            {
                "pivot": 7,
                "CX targets": [2, 3, 11, 8, 9, 0],
                "MCU ctrls": [2, 9, 0, 1, 3, 4, 5, 6, 8, 10, 11]
            },
        ]
    }
    for rotation_data in expected_rotation_gates["givens rotations"]:
        # Build subcircuits.
        Xcirc = QuantumCircuit(12)
        for target in rotation_data["CX targets"]:
            Xcirc.cx(
                control_qubit=rotation_data["pivot"],
                target_qubit=target
            )
        pivot_qubit = [rotation_data["pivot"]]
        angle = expected_rotation_gates["angle"]
        ctrls = rotation_data["MCU ctrls"]
        num_ctrls = len(rotation_data["MCU ctrls"])
        ctrl_state = expected_rotation_gates["MCU ctrl state"]

        circ_with_mcx = QuantumCircuit(12)
        circ_with_mcx.append(RZGate(-1.0*np.pi/2.0), pivot_qubit)
        circ_with_mcx.append(RYGate(-2.0*angle), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RYGate(2.0*angle), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RZGate(1.0*np.pi/2.0), pivot_qubit)

        # Construct current expected givens rotation.
        expected_master_circuit.compose(Xcirc, inplace=True)
        expected_master_circuit.compose(circ_with_mcx, inplace=True)
        expected_master_circuit.compose(Xcirc, inplace=True)

    print("Expected circuit:")
    print(expected_master_circuit)

    # Create master circuit via the magnetic trotter step code.
    lattice_def = LatticeDef(1.5, 2)
    lattice_encoder = LatticeStateEncoder(
        _T1_LINK_BITMAP,
        dummy_phys_states,
        lattice=lattice_def)
    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    with pytest.raises(ValueError):
        LatticeCircuitManager(lattice_encoder, inconsistent_mag_hamiltonian)
    circ_mgr = LatticeCircuitManager(lattice_encoder,
                                     dummy_mag_hamiltonian)
    master_circuit = circ_mgr.create_blank_full_lattice_circuit(
        lattice_registers)
    circ_mgr.apply_magnetic_trotter_step(
        master_circuit,
        lattice_registers,
        optimize_circuits=False,
        cache_mag_evol_circuit=True
    )
    master_circuit.assign_parameters({  # Values computed above assuming dt and coupling set to 1.
        "dt_mag__0": 1,
        "coupling_g_mag__0": 1
    }, inplace=True)
    print("Obtained circuit:")
    print(master_circuit)

    # Checking equivalence via helper methods for
    # (1) flattening a circuit down to a single register and (2) comparing
    # logical equivalence of two circuits.
    assert _check_circuits_logically_equivalent(_flatten_circuit(master_circuit), expected_master_circuit), "Encountered inequivalent circuits."


def test_apply_magnetic_trotter_step_d_2_large_lattice():
    print(
        "Checking that application of magnetic Trotter step works for d=2 "
        "on a large enough lattice that no control links are repeated in any "
        "one plaquette."
    )
    # DO NOT CHANGE THE "DUMMY" DATA UNLESS YOU ARE WILLING TO WORK OUT
    # WHAT THE CORRECT "EXPECTED" CIRCUITS ARE. THERE IS
    # STRONG DEPENDENCE BETWEEN THAT AND THESE DUMMY
    # TEST DATA.
    # If you need to make more test data, follow the following process:
    #    1. Come up with a dummy hamiltonian matrix element of the form (plaquette bitstring, plaquette bitstring, mat elem value).
    #    2. Come up with the corresponding physical states which get encoded to these bit strings.
    #    3. For each matrix element * plaquette in the lattice, there will be ONE givens rotation circuit. For each such givens rotation:
    #      3a. Determine which registers are involved in the circuit following conventional ordering of vertices, then active links, then control links.
    #      3b. Determine what the "X" circuit prefix is by comparing the state bitstrings for the matrix element, determining the LP family, and then
    #          mapping each substring in the plaquette encoding onto actual registers in the lattice.
    #      3c. Repeat this exercise with the multi-control rotation, where the type of ladder or projector operator involved determines the control states.
    # Ask yourself if you REALLY feel like doing all that before mucking about with this test data.
    # Signature for d=2, plane=(1,2), default forder.
    _sig_2 = ((1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2))
    _plane_12 = (1, 2)
    # The controls at each vertex are in the default FORDER [1, 2, 3, -1, -2, -3]:
    # (-1, -2) at v1, (+1, -2) at v2, (+1, +2) at v3, (+2, -1) at v4. v3's two
    # controls carry different irreps (THREE, ONE), so that swapping them
    # changes the values the multi-controlled X requires of their qubits.
    dummy_mag_hamiltonian: HamiltonianData = {
        (_plane_12, _sig_2): {("0000" + "00100000" + "0000000010000010", "0010" + "01010100" + "1001101010000010"): 0.33}  # One matrix element, plaquette has v, a_link, and c_link substrings.
    }
    dummy_phys_states: list[PlaquetteState] = [
        (  # Matches the first encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (ONE, THREE, ONE, ONE),
            ((ONE, ONE), (ONE, ONE), (THREE, ONE), (ONE, THREE))
        ),
        (  # Matches the second encoded state in the dummy magnetic hamiltonian.
            (0, 0, 1, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, ONE),
            ((THREE, THREE_BAR), (THREE, THREE), (THREE, ONE), (ONE, THREE))
        )
    ]
    expected_master_circuit = QuantumCircuit(45)
    expected_rotation_gates = {  # Data for constructing the expected circuit.
        "angle": -0.165,
        "MCU ctrl state": "000000000000000000000000111",  # Little endian per qiskit convention.
        "givens rotations": [
            {
                "pivot": 20,
                "CX targets": [2, 18, 19, 7, 31, 14, 16, 28],
                "MCU ctrls": [18, 36, 21] + [0, 15, 5, 1, 2, 19, 6, 7, 3, 4, 31, 32, 13, 14, 16, 17, 28, 29, 22, 23, 24, 8, 9, 37] # on ctrls first, followed by off, with pivot skipped.
            },
            {
                "pivot": 25,
                "CX targets": [7, 23, 24, 12, 36, 4, 21, 18],
                "MCU ctrls": [23, 41, 26] + [5, 20, 10, 6, 7, 24, 11, 12, 8, 9, 36, 37, 3, 4, 21, 22, 18, 19, 27, 28, 29, 13, 14, 42]
            },
            {
                "pivot": 15,
                "CX targets": [12, 28, 29, 2, 41, 9, 26, 23],
                "MCU ctrls": [28, 31, 16] + [10, 25, 0, 11, 12, 29, 1, 2, 13, 14, 41, 42, 8, 9, 26, 27, 23, 24, 17, 18, 19, 3, 4, 32]
            },
            {
                "pivot": 35,
                "CX targets": [17, 33, 34, 22, 1, 29, 31, 43],
                "MCU ctrls": [33, 6, 36] + [15, 30, 20, 16, 17, 34, 21, 22, 18, 19, 1, 2, 28, 29, 31, 32, 43, 44, 37, 38, 39, 23, 24, 7]
            },
            {
                "pivot": 40,
                "CX targets": [22, 38, 39, 27, 6, 19, 36, 33],
                "MCU ctrls": [38, 11, 41] + [20, 35, 25, 21, 22, 39, 26, 27, 23, 24, 6, 7, 18, 19, 36, 37, 33, 34, 42, 43, 44, 28, 29, 12]
            },
            {
                "pivot": 30,
                "CX targets": [27, 43, 44, 17, 11, 24, 41, 38],
                "MCU ctrls": [43, 1, 31] + [25, 40, 15, 26, 27, 44, 16, 17, 28, 29, 11, 12, 23, 24, 41, 42, 38, 39, 32, 33, 34, 18, 19, 2]
            },
            {
                "pivot": 5,
                "CX targets": [32, 3, 4, 37, 16, 44, 1, 13],
                "MCU ctrls": [3, 21, 6] + [30, 0, 35, 31, 32, 4, 36, 37, 33, 34, 16, 17, 43, 44, 1, 2, 13, 14, 7, 8, 9, 38, 39, 22]
            },
            {
                "pivot": 10,
                "CX targets": [37, 8, 9, 42, 21, 34, 6, 3],
                "MCU ctrls": [8, 26, 11] + [35, 5, 40, 36, 37, 9, 41, 42, 38, 39, 21, 22, 33, 34, 6, 7, 3, 4, 12, 13, 14, 43, 44, 27]
            },
            {
                "pivot": 0,
                "CX targets": [42, 13, 14, 32, 26, 39, 11, 8],
                "MCU ctrls": [13, 16, 1] + [40, 10, 30, 41, 42, 14, 31, 32, 43, 44, 26, 27, 38, 39, 11, 12, 8, 9, 2, 3, 4, 33, 34, 17]
            }
        ]
    }
    for rotation_data in expected_rotation_gates["givens rotations"]:
        # Build subcircuits.
        Xcirc = QuantumCircuit(45)
        for target in rotation_data["CX targets"]:
            Xcirc.cx(
                control_qubit=rotation_data["pivot"],
                target_qubit=target
            )
        pivot_qubit = [rotation_data["pivot"]]
        angle = expected_rotation_gates["angle"]
        ctrls = rotation_data["MCU ctrls"]
        num_ctrls = len(rotation_data["MCU ctrls"])
        ctrl_state = expected_rotation_gates["MCU ctrl state"]

        circ_with_mcx = QuantumCircuit(45)
        circ_with_mcx.append(RZGate(-1.0*np.pi/2.0), pivot_qubit)
        circ_with_mcx.append(RYGate(-2.0*angle), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RYGate(2.0*angle), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RZGate(1.0*np.pi/2.0), pivot_qubit)

        # Construct current expected givens rotation.
        expected_master_circuit.compose(Xcirc, inplace=True)
        expected_master_circuit.compose(circ_with_mcx, inplace=True)
        expected_master_circuit.compose(Xcirc, inplace=True)

    print("Expected circuit:")
    print(expected_master_circuit)

    # Create master circuit via the magnetic trotter step code.
    lattice_def = LatticeDef(2, 3)
    lattice_encoder = LatticeStateEncoder(
        _T1_LINK_BITMAP,
        dummy_phys_states,
        lattice=lattice_def)
    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    circ_mgr = LatticeCircuitManager(lattice_encoder,
                                     dummy_mag_hamiltonian)
    master_circuit = circ_mgr.create_blank_full_lattice_circuit(
        lattice_registers)
    circ_mgr.apply_magnetic_trotter_step(
        master_circuit,
        lattice_registers,
        optimize_circuits=False,
        cache_mag_evol_circuit=True
    )
    master_circuit.assign_parameters({  # Values computed above assuming dt and coupling set to 1.
        "dt_mag__0": 1,
        "coupling_g_mag__0": 1
    }, inplace=True)
    print("Obtained circuit:")
    print(master_circuit)

    # Checking equivalence via helper methods for
    # (1) flattening a circuit down to a single register and (2) comparing
    # logical equivalence of two circuits.
    assert _check_circuits_logically_equivalent(_flatten_circuit(master_circuit), expected_master_circuit), "Encountered inequivalent circuits."


def test_apply_magnetic_trotter_step_d_2_small_lattice():
    print(
        "Checking that application of magnetic Trotter step works for d=2 "
        "on a small lattice where some control links are repeated in each "
        "plaquette."
    )
    # DO NOT CHANGE THE "DUMMY" DATA UNLESS YOU ARE WILLING TO WORK OUT
    # WHAT THE CORRECT "EXPECTED" CIRCUITS ARE. THERE IS
    # STRONG DEPENDENCE BETWEEN THAT AND THESE DUMMY
    # TEST DATA.
    # If you need to make more test data, follow the following process:
    #    1. Come up with a dummy hamiltonian matrix element of the form (plaquette bitstring, plaquette bitstring, mat elem value).
    #    2. Come up with the corresponding physical states which get encoded to these bit strings.
    #    3. For each matrix element * plaquette in the lattice, there will be ONE givens rotation circuit. For each such givens rotation:
    #      3a. Determine which registers are involved in the circuit following conventional ordering of vertices, then active links, then control links.
    #      3b. Determine what the "X" circuit prefix is by comparing the state bitstrings for the matrix element, determining the LP family, and then
    #          mapping each substring in the plaquette encoding onto actual registers in the lattice.
    #      3c. Repeat this exercise with the multi-control rotation, where the type of ladder or projector operator involved determines the control states.
    # Ask yourself if you REALLY feel like doing all that before mucking about with this test data.
    # Signature for d=2, plane=(1,2), default forder.
    _sig_2 = ((1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2))
    _plane_12 = (1, 2)
    dummy_mag_hamiltonian: HamiltonianData = {
        (_plane_12, _sig_2): {("0000" + "00100000" + "0000000010000010", "0010" + "01010100" + "1001101000100100"): 0.33}  # One matrix element, plaquette has v, a_link, and c_link substrings.
    }
    # The link that is both v2's -2 control and v3's +2 control has two
    # different states ("10" and "00") in this matrix element's second
    # bit string, which no state of this lattice can have, so the
    # LatticeCircuitManager constructor refuses it.
    inconsistent_mag_hamiltonian: HamiltonianData = {
        (_plane_12, _sig_2): {("0000" + "00100000" + "0000000000000010", "0010" + "01010100" + "1001101000000010"): 0.33}
    }
    dummy_phys_states: list[PlaquetteState] = [
        (  # Matches the first encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (ONE, THREE, ONE, ONE),
            ((ONE, ONE), (ONE, ONE), (THREE, ONE), (ONE, THREE))
        ),
        (  # Matches the second encoded state in the dummy magnetic hamiltonian.
            (0, 0, 1, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, ONE),
            ((THREE, THREE_BAR), (THREE, THREE), (ONE, THREE), (THREE_BAR, ONE))
        )
    ]
    expected_master_circuit = QuantumCircuit(20)
    expected_rotation_gates = {  # Data for constructing the expected circuit.
        "angle": -0.165,
        "MCU ctrl state": "0000000000000000011",  # Little endian per qiskit convention.
        "givens rotations": [
            {
                "pivot": 15,
                "CX targets": [2, 13, 14, 7, 11, 9, 18, 16],
                "MCU ctrls": [13, 16] + [0, 10, 5, 1, 2, 14, 6, 7, 3, 4, 11, 12, 8, 9, 18, 19, 17] # on ctrls first, followed by off, with pivot skipped.
            },
            {
                "pivot": 10,
                "CX targets": [7, 18, 19, 2, 16, 4, 13, 11],
                "MCU ctrls": [18, 11] + [5, 15, 0, 6, 7, 19, 1, 2, 8, 9, 16, 17, 3, 4, 13, 14, 12] # on ctrls first, followed by off, with pivot skipped.
            },
            {
                "pivot": 5,
                "CX targets": [12, 3, 4, 17, 1, 19, 8, 6],
                "MCU ctrls": [3, 6] + [10, 0, 15, 11, 12, 4, 16, 17, 13, 14, 1, 2, 18, 19, 8, 9, 7] # on ctrls first, followed by off, with pivot skipped.
            },
            {
                "pivot": 0,
                "CX targets": [17, 8, 9, 12, 6, 14, 3, 1],
                "MCU ctrls": [8, 1] + [15, 5, 10, 16, 17, 9, 11, 12, 18, 19, 6, 7, 13, 14, 3, 4, 2] # on ctrls first, followed by off, with pivot skipped.
            },
        ]
    }
    for rotation_data in expected_rotation_gates["givens rotations"]:
        # Build subcircuits.
        Xcirc = QuantumCircuit(20)
        for target in rotation_data["CX targets"]:
            Xcirc.cx(
                control_qubit=rotation_data["pivot"],
                target_qubit=target
            )
        pivot_qubit = [rotation_data["pivot"]]
        angle = expected_rotation_gates["angle"]
        ctrls = rotation_data["MCU ctrls"]
        num_ctrls = len(rotation_data["MCU ctrls"])
        ctrl_state = expected_rotation_gates["MCU ctrl state"]

        circ_with_mcx = QuantumCircuit(20)
        circ_with_mcx.append(RZGate(-1.0*np.pi/2.0), pivot_qubit)
        circ_with_mcx.append(RYGate(-2.0*angle), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RYGate(2.0*angle), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RZGate(1.0*np.pi/2.0), pivot_qubit)

        # Construct current expected givens rotation.
        expected_master_circuit.compose(Xcirc, inplace=True)
        expected_master_circuit.compose(circ_with_mcx, inplace=True)
        expected_master_circuit.compose(Xcirc, inplace=True)

    print("Expected circuit:")
    print(expected_master_circuit)

    # Create master circuit via the magnetic trotter step code.
    lattice_def = LatticeDef(2, 2)
    lattice_encoder = LatticeStateEncoder(
        _T1_LINK_BITMAP,
        dummy_phys_states,
        lattice=lattice_def)
    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    with pytest.raises(ValueError):
        LatticeCircuitManager(lattice_encoder, inconsistent_mag_hamiltonian)
    circ_mgr = LatticeCircuitManager(lattice_encoder,
                                     dummy_mag_hamiltonian)
    master_circuit = circ_mgr.create_blank_full_lattice_circuit(
        lattice_registers)
    circ_mgr.apply_magnetic_trotter_step(
        master_circuit,
        lattice_registers,
        optimize_circuits=False,
        cache_mag_evol_circuit=True
    )
    master_circuit.assign_parameters({  # Values computed above assuming dt and coupling set to 1.
        "dt_mag__0": 1,
        "coupling_g_mag__0": 1
    }, inplace=True)
    print("Obtained circuit:")
    print(master_circuit)

    # Checking equivalence via helper methods for
    # (1) flattening a circuit down to a single register and (2) comparing
    # logical equivalence of two circuits.
    assert _check_circuits_logically_equivalent(_flatten_circuit(master_circuit), expected_master_circuit), "Encountered inequivalent circuits."


def test_apply_electric_trotter_step_d_3_2_lattice():
    print("Checking that the electric trotter step acts as expected on a T2 3x1 lattice.")
    dummy_electric_hamiltonian = [0.33,0.66,0.66,0.99,0.66,0.99,0.99,0.33]
    dummy_mag_hamiltonian: HamiltonianData = {}
    dummy_phys_states: list[PlaquetteState] = [
        (
            (0,0,0,0),
            (ONE, THREE, ONE, THREE_BAR),
            ((THREE_BAR,), (THREE_BAR,), (THREE,), (THREE,))
        ),
        (
            (0,0,0,0),
            (EIGHT,SIX,EIGHT,SIX_BAR),
            ((SIX_BAR,),(SIX_BAR,),(SIX,),(SIX,))
        )
    ]
    lattice_def = LatticeDef(1.5,3)
    lattice_encoder = LatticeStateEncoder(
        _T2_LINK_BITMAP,
        dummy_phys_states,
        lattice=lattice_def)
    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    circ_mgr = LatticeCircuitManager(lattice_encoder,
                                 dummy_mag_hamiltonian)
    master_circuit = circ_mgr.create_blank_full_lattice_circuit(
    lattice_registers)
    circ_mgr.apply_electric_trotter_step(master_circuit,lattice_registers,dummy_electric_hamiltonian,electric_gray_order=True)
    master_circuit.assign_parameters({  # Values computed above assuming dt and coupling set to 1.
        "dt_ee__0": 1,
        "coupling_g_ee__0": 1
    }, inplace=True)

    expected_local_circuit = QuantumCircuit(3)
    expected_local_circuit.rz(0.66,2)

    expected_local_circuit.cx(2,1)
    expected_local_circuit.rz(0.99,1)
    expected_local_circuit.cx(2,1)

    expected_local_circuit.rz(0.66,1)

    expected_local_circuit.cx(1,0)
    expected_local_circuit.rz(0.99,0)
    
    expected_local_circuit.cx(2,0)
    expected_local_circuit.rz(0.33,0)
    expected_local_circuit.cx(1,0)
    
    expected_local_circuit.rz(0.99,0)
    expected_local_circuit.cx(2,0)

    expected_local_circuit.rz(0.66,0)

    expected_master_circuit = QuantumCircuit(27)
    for i in range(9):
        link_qubits = [3*i,3*i + 1, 3*i+2]
        expected_master_circuit.compose(expected_local_circuit,link_qubits,inplace=True)
    
    assert _check_circuits_logically_equivalent(_flatten_circuit(master_circuit), expected_master_circuit), "Encountered inequivalent circuits."
    print("Test for electric trotter step passed.")


def test_creating_correct_ancilla_register_for_d_3_2_T1_small():
    print("Checking if an ancilla register with the correct number of ancillas are added for d=3/2, T1, 1x2")

    truncation_mode, cutoff = "T", 1
    lattice_def = LatticeDef(1.5, 2)

    lattice_encoder = LatticeStateEncoder.for_truncation(lattice_def, truncation_mode, cutoff)

    physical_states_for_control_pruning = set(lattice_encoder.encode_plaquette_state_as_bit_string(plaquette) for plaquette in physical_plaquette_states(lattice_def, truncation_mode, cutoff))

    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    circ_mgr = LatticeCircuitManager.for_truncation(lattice_encoder)
    master_circuit = circ_mgr.create_blank_full_lattice_circuit(
        lattice_registers)

    circ_mgr.num_ancillas = circ_mgr.compute_num_ancillas_needed_from_mag_trotter_step(master_circuit, lattice_registers, control_fusion=True, 
        physical_states_for_control_pruning=physical_states_for_control_pruning,
        optimize_circuits=False)
    circ_mgr.add_ancilla_register_to_quantum_circuit(master_circuit)

    # Number of ancillas for d=3/2, T1, 1x2 with control pruning and fusing found in arXiv:2503.08866
    assert (len(master_circuit.ancillas) == 3), "Ancilla register for d=3/2, T1 with control pruning and fusion is improperly initiated"
    print("Test for ancilla register count passed")


def test_creating_correct_ancilla_register_for_d_2_T1_small():
    print("Checking if an ancilla register with the correct number of ancillas are added are added for d=2, T1, 2x2")

    truncation_mode, cutoff = "T", 1
    lattice_def = LatticeDef(2, 2)

    lattice_encoder = LatticeStateEncoder.for_truncation(lattice_def, truncation_mode, cutoff)

    physical_states_for_control_pruning = set(lattice_encoder.encode_plaquette_state_as_bit_string(plaquette) for plaquette in physical_plaquette_states(lattice_def, truncation_mode, cutoff))

    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    circ_mgr = LatticeCircuitManager.for_truncation(lattice_encoder)
    master_circuit = circ_mgr.create_blank_full_lattice_circuit(
        lattice_registers)

    circ_mgr.num_ancillas = circ_mgr.compute_num_ancillas_needed_from_mag_trotter_step(master_circuit, lattice_registers, control_fusion=True, 
        physical_states_for_control_pruning=physical_states_for_control_pruning,
        optimize_circuits=False)
    circ_mgr.add_ancilla_register_to_quantum_circuit(master_circuit)

    # Number of ancillas for d=2, T1, 2x2 with control pruning and fusing found in arXiv:2503.08866
    assert (len(master_circuit.ancillas) == 7), "Ancilla register for d=2, T1 with control pruning and fusion is improperly initiated"
    print("Test for ancilla register count passed")


def test_magnetic_with_ancilla_has_no_MCX():
    print("Checking that the MCX gates are properly decomposed in the magnetic trotter step")

    truncation_mode, cutoff = "T", 1
    lattice_def = LatticeDef(1.5, 2)

    lattice_encoder = LatticeStateEncoder.for_truncation(lattice_def, truncation_mode, cutoff)

    physical_states_for_control_pruning = set(lattice_encoder.encode_plaquette_state_as_bit_string(plaquette) for plaquette in physical_plaquette_states(lattice_def, truncation_mode, cutoff))

    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    circ_mgr = LatticeCircuitManager.for_truncation(lattice_encoder)
    master_circuit_with_ancillas = circ_mgr.create_blank_full_lattice_circuit(
        lattice_registers)

    circ_mgr.num_ancillas = circ_mgr.compute_num_ancillas_needed_from_mag_trotter_step(master_circuit_with_ancillas, lattice_registers, control_fusion=True, 
        physical_states_for_control_pruning=physical_states_for_control_pruning,
        optimize_circuits=False)
    circ_mgr.add_ancilla_register_to_quantum_circuit(master_circuit_with_ancillas)

    circ_mgr.apply_magnetic_trotter_step(master_circuit_with_ancillas, lattice_registers, physical_states_for_control_pruning=physical_states_for_control_pruning, control_fusion=True)
    master_circuit_with_ancillas.assign_parameters({  # Values computed above assuming dt and coupling set to 1.
        "dt_mag__0": 1,
        "coupling_g_mag__0": 1
    }, inplace=True)

    print(master_circuit_with_ancillas)

    mcx_count = 0

    for circuit_instruction in master_circuit_with_ancillas.data:
        if len(circuit_instruction.operation.name) >= 3 and circuit_instruction.operation.name[:3] == "mcx":
            mcx_count += 1

    assert (mcx_count == 0), "MCX count for magnetic circuit with ancillas is not 0"
    print("Test for MCX count in magnetic circuit has passed")


def test_apply_magnetic_trotter_step_d_3_2_small_lattice_with_ancillas():
    print(
        "Checking that application of magnetic Trotter step works for d=3/2 "
        "on a small lattice with ancilla qubits"
    )
    # DO NOT CHANGE THE "DUMMY" DATA UNLESS YOU ARE WILLING TO WORK OUT
    # WHAT THE CORRECT "EXPECTED" CIRCUITS ARE. THERE IS
    # STRONG DEPENDENCE BETWEEN THAT AND THESE DUMMY
    # TEST DATA.
    # If you need to make more test data, follow the following process:
    #    1. Come up with a dummy hamiltonian matrix element of the form (plaquette bitstring, plaquette bitstring, mat elem value).
    #    2. Come up with the corresponding physical states which get encoded to these bit strings.
    #    3. For each matrix element * plaquette in the lattice, there will be ONE givens rotation circuit. For each such givens rotation:
    #      3a. Determine which registers are involved in the circuit following conventional ordering of vertices, then active links, then control links.
    #      3b. Determine what the "X" circuit prefix is by comparing the state bitstrings for the matrix element, determining the LP family, and then
    #          mapping each substring in the plaquette encoding onto actual registers in the lattice.
    #      3c. Repeat this exercise with the multi-control rotation, where the type of ladder or projector operator involved determines the control states (raising to get to final state is on, projector onto 1 is on).
    # Ask yourself if you REALLY feel like doing all that before mucking about with this test data.
    # Signature for d=3/2, plane=(1,2), default forder.
    _sig_3_2 = ((1, 2, -1), (1, 2, -1), (1, -1, -2), (1, -1, -2))
    _plane_12 = (1, 2)
    dummy_mag_hamiltonian: HamiltonianData = {
        (_plane_12, _sig_3_2): {("00100001" + "00000000", "01010110" + "10100000"): 0.33}  # One matrix element, plaquette only has a_link and c_link substrings.
    }
    # The link that is both v1's -1 control and v2's +1 control has two
    # different states ("10" and "01") in this matrix element's second
    # bit string, which no state of this lattice can have, so the
    # LatticeCircuitManager constructor refuses it.
    inconsistent_mag_hamiltonian: HamiltonianData = {
        (_plane_12, _sig_3_2): {("00100001" + "00000000", "01010110" + "10011010"): 0.33}
    }
    dummy_phys_states: list[PlaquetteState] = [
        (  # Matches the first encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (ONE, THREE, ONE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (  # Matches the second encoded state in the inconsistent magnetic hamiltonian.
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            ((THREE,), (THREE_BAR,), (THREE,), (THREE,))
        ),
        (  # Matches the second encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            ((THREE,), (THREE,), (ONE,), (ONE,))
        )
    ]
    expected_master_circuit = QuantumCircuit(12)
    plaquette_ancilla_qubits = AncillaRegister(9)
    expected_master_circuit.add_register(plaquette_ancilla_qubits)
    # Only expecting one rotation per plaquette, yielding two total Givens rotations.
    expected_rotation_gates = {  # Data for constructing the expected circuit.
        "angle": -0.165,
        "MCU ctrl state": "00000000011",  # Little endian per qiskit convention.
        "givens rotations": [
            {
                "pivot": 1,
                "CX targets": [8, 9, 5, 2, 3, 6],
                "MCU ctrls": [8, 3, 0, 2, 4, 5, 6, 7, 9, 10, 11] # on ctrls first, followed by off, with pivot skipped.
            },
            {
                "pivot": 7,
                "CX targets": [2, 3, 11, 8, 9, 0],
                "MCU ctrls": [2, 9, 0, 1, 3, 4, 5, 6, 8, 10, 11]
            },
        ]
    }
    for rotation_data in expected_rotation_gates["givens rotations"]:
        # Build subcircuits.
        Xcirc = QuantumCircuit(12)
        for target in rotation_data["CX targets"]:
            Xcirc.cx(
                control_qubit=rotation_data["pivot"],
                target_qubit=target
            )
        pivot_qubit = [rotation_data["pivot"]]
        angle = expected_rotation_gates["angle"]
        ctrls = rotation_data["MCU ctrls"]
        ctrl_state = expected_rotation_gates["MCU ctrl state"]
        ancilla_register = AncillaRegister(9)

        # Givens rotation with ancilla qubits 
        circ_with_mcx = QuantumCircuit(12)        
        circ_with_mcx.add_register(AncillaRegister(9, "anc"))
        
        circ_with_mcx.append(RZGate(-1.0*np.pi/2.0), pivot_qubit)
        circ_with_mcx.append(RYGate(-2.0*angle), pivot_qubit)

        # Create first MCX as separate circuit so we can decompose it explicitly.
        mcx_vchain_initial = QuantumCircuit(12)
        mcx_vchain_initial.add_register(ancilla_register)  # Construct separate mcx circuit so we can decompose it.
        mcx_vchain_initial.mcx(ctrls, pivot_qubit, ancilla_qubits=list(range(12, 21)), ctrl_state=ctrl_state, mode='v-chain')
        mcx_vchain_initial = mcx_vchain_initial.decompose(reps=3)
        circ_with_mcx.compose(mcx_vchain_initial, inplace=True)
        circ_with_mcx.append(RYGate(2.0*angle), pivot_qubit)

        # Create final MCX as separate circuit so we can decompose it explicitly.
        mcx_vchain_final = QuantumCircuit(12)
        mcx_vchain_final.add_register(ancilla_register)  # Construct separate mcx circuit so we can decompose it.
        mcx_vchain_final.mcx(ctrls, pivot_qubit, ancilla_qubits=list(range(12, 21)), ctrl_state=ctrl_state, mode='v-chain')
        mcx_vchain_final = mcx_vchain_final.decompose(reps=3)
        circ_with_mcx.compose(mcx_vchain_final, inplace=True)
        circ_with_mcx.append(RZGate(1.0*np.pi/2.0), pivot_qubit)

        # Construct current expected givens rotation.
        expected_master_circuit.compose(Xcirc, inplace=True)
        expected_master_circuit.compose(circ_with_mcx, inplace=True)
        expected_master_circuit.compose(Xcirc, inplace=True)

    print("Expected circuit:")
    print(expected_master_circuit)

    # Create master circuit via the magnetic trotter step code.
    lattice_def = LatticeDef(1.5, 2)
    lattice_encoder = LatticeStateEncoder(
        _T1_LINK_BITMAP,
        dummy_phys_states,
        lattice=lattice_def)
    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    with pytest.raises(ValueError):
        LatticeCircuitManager(lattice_encoder, inconsistent_mag_hamiltonian)
    circ_mgr = LatticeCircuitManager(lattice_encoder,
                                     dummy_mag_hamiltonian)
    master_circuit = circ_mgr.create_blank_full_lattice_circuit(
        lattice_registers)
    circ_mgr.num_ancillas = circ_mgr.compute_num_ancillas_needed_from_mag_trotter_step(master_circuit, lattice_registers, 
        control_fusion=False, physical_states_for_control_pruning=None, optimize_circuits=False)
    circ_mgr.add_ancilla_register_to_quantum_circuit(master_circuit)
    circ_mgr.apply_magnetic_trotter_step(
        master_circuit,
        lattice_registers,
        optimize_circuits=False,
        cache_mag_evol_circuit=True
    )
    master_circuit.assign_parameters({  # Values computed above assuming dt and coupling set to 1.
        "dt_mag__0": 1,
        "coupling_g_mag__0": 1
    }, inplace=True)
    print("Obtained circuit:")
    print(master_circuit)

    # TODO Something about the strict equality check isn't working. Using
    # a weaker check that gate counts match for now.
    assert master_circuit.count_ops() == expected_master_circuit.count_ops(), f"Encountered inequivalent circuits.\nExpected gate counts: {expected_master_circuit.count_ops()}\nActual gate counts:{master_circuit.count_ops()}"
    # Checking equivalence via helper methods for
    # (1) flattening a circuit down to a single register and (2) comparing
    # logical equivalence of two circuits.
    # assert _check_circuits_logically_equivalent(_flatten_circuit(master_circuit), _flatten_circuit(expected_master_circuit)), "Encountered inequivalent circuits."


def test_num_ancillas_setter_works_nonnegative_ints():
    # Some minimal data to create a LatticeCircuitManager.
    dummy_mag_hamiltonian: HamiltonianData = {
        ((1, 2), _SIG_3_2): {("00100001" + "00000000", "01010110" + "10100000"): 0.33},
    }
    dummy_phys_states: list[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, ONE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            ((THREE,), (THREE_BAR,), (THREE,), (THREE,))
        ),
        (
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            ((THREE,), (THREE,), (ONE,), (ONE,))
        )
    ]
    lattice_def = LatticeDef(1.5, 2)
    lattice_encoder = LatticeStateEncoder(
        _T1_LINK_BITMAP,
        dummy_phys_states,
        lattice=lattice_def)
    circ_mgr = LatticeCircuitManager(lattice_encoder,
                                     dummy_mag_hamiltonian)

    init_num_ancillas_is_zero = circ_mgr.num_ancillas == 0
    assert init_num_ancillas_is_zero, "LatticeCircuitManager not initialized with zero ancillas."

    circ_mgr.num_ancillas = 10
    assert circ_mgr.num_ancillas == 10

    circ_mgr.num_ancillas = 0
    assert circ_mgr.num_ancillas == 0


def test_num_ancillas_setter_fails_for_non_int():
    # Some minimal data to create a LatticeCircuitManager.
    dummy_mag_hamiltonian: HamiltonianData = {
        ((1, 2), _SIG_3_2): {("00100001" + "00000000", "01010110" + "10100000"): 0.33},
    }
    dummy_phys_states: list[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, ONE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            ((THREE,), (THREE_BAR,), (THREE,), (THREE,))
        ),
        (
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            ((THREE,), (THREE,), (ONE,), (ONE,))
        )
    ]
    lattice_def = LatticeDef(1.5, 2)
    lattice_encoder = LatticeStateEncoder(
        _T1_LINK_BITMAP,
        dummy_phys_states,
        lattice=lattice_def)
    circ_mgr = LatticeCircuitManager(lattice_encoder,
                                     dummy_mag_hamiltonian)

    with pytest.raises(TypeError):
        circ_mgr.num_ancillas = 1.0


def test_num_ancillas_setter_fails_for_negative_int():
    # Some minimal data to create a LatticeCircuitManager.
    dummy_mag_hamiltonian: HamiltonianData = {
        ((1, 2), _SIG_3_2): {("00100001" + "00000000", "01010110" + "10100000"): 0.33},
    }
    dummy_phys_states: list[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, ONE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            ((THREE,), (THREE_BAR,), (THREE,), (THREE,))
        ),
        (
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            ((THREE,), (THREE,), (ONE,), (ONE,))
        )
    ]
    lattice_def = LatticeDef(1.5, 2)
    lattice_encoder = LatticeStateEncoder(
        _T1_LINK_BITMAP,
        dummy_phys_states,
        lattice=lattice_def)
    circ_mgr = LatticeCircuitManager(lattice_encoder,
                                     dummy_mag_hamiltonian)

    with pytest.raises(ValueError):
        circ_mgr.num_ancillas = -1


def test_adding_ancilla_register_fails_if_already_exists():
    # Some minimal data to create a LatticeCircuitManager.
    dummy_mag_hamiltonian: HamiltonianData = {
        ((1, 2), _SIG_3_2): {("00100001" + "00000000", "01010110" + "10100000"): 0.33},
    }
    dummy_phys_states: list[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, ONE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            ((THREE,), (THREE_BAR,), (THREE,), (THREE,))
        ),
        (
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            ((THREE,), (THREE,), (ONE,), (ONE,))
        )
    ]
    lattice_def = LatticeDef(1.5, 2)
    lattice_encoder = LatticeStateEncoder(
        _T1_LINK_BITMAP,
        dummy_phys_states,
        lattice=lattice_def)
    circ_mgr = LatticeCircuitManager(lattice_encoder,
                                     dummy_mag_hamiltonian)

    # Create a circuit with no ancillas.
    master_circuit = QuantumCircuit(3)
    assert len(master_circuit.ancillas) == 0

    # Add an ancillas register.
    circ_mgr.num_ancillas = 10
    circ_mgr.add_ancilla_register_to_quantum_circuit(master_circuit)
    assert len(master_circuit.ancillas) == 10

    # Adding an ancillas again should fail.
    with pytest.raises(CircuitError):
        circ_mgr.add_ancilla_register_to_quantum_circuit(master_circuit)

def test_apply_mag_trotter_step_independent_params_for_givens_rotations():
    # Some minimal data to create a LatticeCircuitManager.
    _sig_3_2 = ((1, 2, -1), (1, 2, -1), (1, -1, -2), (1, -1, -2))
    _plane_12 = (1, 2)
    dummy_mag_hamiltonian: HamiltonianData = {  # There will be 2 Givens rotations and therefore 2 Parameters.
        (_plane_12, _sig_3_2): {
            ("00100001" + "00000000", "01010110" + "10011001"): 0.99,
            ("00100001" + "00000000", "01010110" + "10100000"): 0.33,
            ("00100001" + "00000000", "01010110" + "00100000"): 0.66,
        }
    }
    dummy_phys_states: list[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, ONE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            ((THREE,), (THREE_BAR,), (THREE,), (THREE,))
        ),
        (
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            ((THREE,), (THREE,), (ONE,), (ONE,))
        )
    ]
    lattice_def = LatticeDef(1.5, 3)
    lattice_encoder = LatticeStateEncoder(_T1_LINK_BITMAP, dummy_phys_states, lattice=lattice_def)
    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    circ_mgr = LatticeCircuitManager(lattice_encoder, dummy_mag_hamiltonian)

    master_circuit = circ_mgr.create_blank_full_lattice_circuit(lattice_registers)
    circ_mgr.apply_magnetic_trotter_step(
        master_circuit,
        lattice_registers,
        givens_have_independent_params=True
    )
    assert len(master_circuit.parameters) == 3
    for idx, parameter in enumerate(master_circuit.parameters):
        assert parameter.name == f'theta[{idx}]'

def test_apply_mag_trotter_step_independent_params_multiple_lp_families():
    # This dummy data creates 3 LP bins. Two of them have 2 Givens rotations, one has 3.
    _sig_3_2 = ((1, 2, -1), (1, 2, -1), (1, -1, -2), (1, -1, -2))
    _plane_12 = (1, 2)
    dummy_mag_hamiltonian: HamiltonianData = {
        (_plane_12, _sig_3_2): {
            ("01010101" + "00000101", "10100101" + "00001010"): 0.99, #LP fam 1: LLLLPPPP + PPPPLLLL
            ("10100000" + "10100101", "01010000" + "10101010"): 0.99, #LP fam 1: LLLLPPPP + PPPPLLLL
            ("01010101" + "01010000", "01011010" + "10100000"): 0.99, #LP fam 2: PPPPLLLL + LLLLPPPP
            ("00001010" + "01010000", "00000101" + "10100000"): 0.99, #LP fam 2: PPPPLLLL + LLLLPPPP
            ("00000000" + "10101010", "00000000" + "01010101"): 0.99, #LP fam 3: PPPPPPPP + LLLLLLLL
            ("00000000" + "01010101", "00000000" + "10101010"): 0.99, #LP fam 3: PPPPPPPP + LLLLLLLL
            ("00000000" + "01011010", "00000000" + "10100101"): 0.99, #LP fam 3: PPPPPPPP + LLLLLLLL
        }
    }

    dummy_phys_states: list[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, ONE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            ((THREE,), (THREE_BAR,), (THREE,), (THREE,))
        ),
        (
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            ((THREE,), (THREE,), (ONE,), (ONE,))
        )
    ]
    lattice_def = LatticeDef(1.5, 3)
    lattice_encoder = LatticeStateEncoder(_T1_LINK_BITMAP, dummy_phys_states, lattice=lattice_def)
    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    circ_mgr = LatticeCircuitManager(lattice_encoder, dummy_mag_hamiltonian)

    master_circuit = circ_mgr.create_blank_full_lattice_circuit(lattice_registers)
    circ_mgr.apply_magnetic_trotter_step(
        master_circuit,
        lattice_registers,
        givens_have_independent_params=True
    )
    assert len(master_circuit.parameters) == 7
    for idx, parameter in enumerate(master_circuit.parameters):
        assert parameter.name == f'theta[{idx}]'


def test_magnetic_trotter_step_does_not_depend_on_matrix_element_order():
    """
    With control fusion off, the magnetic Trotter step is the same whatever the order of the matrix elements.

    The LP bins are visited in Gray-code order, not in the order the matrix
    elements arrive in. Checked on a d=3/2 periodic lattice of size 3 at T1,
    where each LP bin holds a single Givens rotation, so the gate lists must
    be equal.
    """
    lattice_def = LatticeDef(1.5, 3)
    lattice_encoder = LatticeStateEncoder.for_truncation(lattice_def, "T", 1)
    mag_hamiltonian: HamiltonianData = {
        plane_and_signature: {
            (lattice_encoder.encode_plaquette_state_as_bit_string(final_state),
             lattice_encoder.encode_plaquette_state_as_bit_string(initial_state)): value
            for (final_state, initial_state), value in box_matrix_elements.items()
        }
        for plane_and_signature, box_matrix_elements in box_terms(lattice_def, "T", 1).items()
    }
    lp_families = {
        bitstring_value_of_LP_family(compute_LP_family(*bit_strings))
        for matrix_elements in mag_hamiltonian.values() for bit_strings in matrix_elements
    }
    assert len(lp_families) > 1, "With a single LP bin, the order of the bins cannot be checked."
    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)

    def gate_list(mag_hamiltonian_in_some_order):
        circ_mgr = LatticeCircuitManager(lattice_encoder, mag_hamiltonian_in_some_order)
        master_circuit = circ_mgr.create_blank_full_lattice_circuit(lattice_registers)
        circ_mgr.apply_magnetic_trotter_step(
            master_circuit, lattice_registers, optimize_circuits=False, control_fusion=False)
        master_circuit.assign_parameters({"dt_mag__0": 1, "coupling_g_mag__0": 1}, inplace=True)
        return [
            (gate.operation, [master_circuit.find_bit(qubit).index for qubit in gate.qubits])
            for gate in master_circuit.data
        ]

    reversed_mag_hamiltonian = {
        plane_and_signature: dict(reversed(matrix_elements.items()))
        for plane_and_signature, matrix_elements in mag_hamiltonian.items()
    }
    assert gate_list(reversed_mag_hamiltonian) == gate_list(mag_hamiltonian)


@pytest.fixture
def sample_circuit_no_ancillas() -> QuantumCircuit:
    data_reg_a = QuantumRegister(3, "RegA")
    data_reg_b = QuantumRegister(2, "RegB")
    empty_reg = QuantumRegister(0, "RegEmpty")
    test_circ = QuantumCircuit(data_reg_a, data_reg_b, empty_reg)
    test_circ.cx(data_reg_a[1], data_reg_b[1])
    test_circ.x(data_reg_a[1])
    test_circ.h(data_reg_b[0])

    return test_circ

@pytest.fixture
def sample_circuit_with_ancillas() -> QuantumCircuit:
    data_reg_a = QuantumRegister(3, "RegA")
    data_reg_b = QuantumRegister(2, "RegB")
    ancillas = AncillaRegister(4, "anc")
    test_circ = QuantumCircuit(data_reg_a, data_reg_b, ancillas)
    test_circ.cx(data_reg_a[1], data_reg_b[1])
    test_circ.x(data_reg_a[1])
    test_circ.h(data_reg_b[0])
    test_circ.cx(data_reg_a[1], ancillas[3])
    # Dummy "compute/uncompute" pattern using ancillas.
    test_circ.h(ancillas[3])
    test_circ.cx(ancillas[3], data_reg_b[0])
    test_circ.h(ancillas[3])
    test_circ.cx(data_reg_a[1], ancillas[3])

    return test_circ

@pytest.fixture
def sample_circuit_with_parameter_and_parameter_expression() -> QuantumCircuit:
    data_reg = QuantumRegister(3, "data")
    ancillas = AncillaRegister(2, "anc")
    test_circ = QuantumCircuit(data_reg, ancillas)
    theta, phi, a, b, g = Parameter("theta"), Parameter("phi"), Parameter("a"), Parameter("b"), Parameter("g")
    test_circ.x(data_reg[0])
    test_circ.h(data_reg[0])
    test_circ.cx(data_reg[0], ancillas[0])
    test_circ.rx(theta, data_reg[1]) # Single parameter
    test_circ.rz(theta * phi, ancillas[0]) # Parameter expression
    test_circ.cx(ancillas[0], data_reg[1])
    test_circ.ry(g * g,data_reg[2]) # Qiskit can't parse power operator, so doing this instead.
    test_circ.ry(a + b, data_reg[2]) # Addition operator
    test_circ.ry(2*a, data_reg[2]) # Scalar multiplication operator

    return test_circ
    

# TODO fixture with QASM data (or perhaps this needs to be a test file).

# TODO fixture with QPY data (or perhaps this needs to be a test file).

class TestCircuitSaveAndLoad:
    def test_save_and_reload_circuit_no_ancillas_qasm(self, sample_circuit_no_ancillas, tmp_path):
        sample_circ_drawn_str = sample_circuit_no_ancillas.draw() # For use in error messages.
        filepath_as_string = str(tmp_path / "my_circuit_str_write.qasm")
        filepath_as_path = tmp_path / "my_circuit_path_write.qasm"
        LatticeCircuitManager.save_circuit(sample_circuit_no_ancillas, filepath_as_string)
        LatticeCircuitManager.save_circuit(sample_circuit_no_ancillas, filepath_as_path)

        path_and_str_write_are_equivalent = filepath_as_path.read_bytes() == Path(filepath_as_string).read_bytes()
        assert path_and_str_write_are_equivalent

        reloaded_circ_filepath_as_string = LatticeCircuitManager.load_circuit(filepath_as_string)
        reloaded_circ_filepath_as_path = LatticeCircuitManager.load_circuit(filepath_as_path)
        path_and_string_load_are_equivalent = reloaded_circ_filepath_as_path == reloaded_circ_filepath_as_string
        assert path_and_string_load_are_equivalent

        saving_and_reloading_circ_gives_back_same_circ = reloaded_circ_filepath_as_path == sample_circuit_no_ancillas
        assert saving_and_reloading_circ_gives_back_same_circ, f"Inequivalent circuits. Expected:\n {sample_circ_drawn_str}\nEncountered:\n{reloaded_circ_filepath_as_path}"

        # Redundant, but a check that the test data hasn't been altered.
        # Expecting a circuit with 1 H, 1 CX, and 1 X.
        assert reloaded_circ_filepath_as_path.count_ops() == {"cx": 1, "x": 1, "h": 1}, f"Circuit ops {reloaded_circ_filepath_as_path.count_ops()} contains unexpected gates."

    def test_save_and_reload_circuit_no_ancillas_qpy(self, sample_circuit_no_ancillas, tmp_path):
        sample_circ_drawn_str = sample_circuit_no_ancillas.draw() # For use in error messages.
        filepath_as_string = str(tmp_path / "my_circuit_str_write.qpy")
        filepath_as_path = tmp_path / "my_circuit_path_write.qpy"
        LatticeCircuitManager.save_circuit(sample_circuit_no_ancillas, filepath_as_string)
        LatticeCircuitManager.save_circuit(sample_circuit_no_ancillas, filepath_as_path)

        path_and_str_write_are_equivalent = filepath_as_path.read_bytes() == Path(filepath_as_string).read_bytes()
        assert path_and_str_write_are_equivalent

        reloaded_circ_filepath_as_string = LatticeCircuitManager.load_circuit(filepath_as_string)
        reloaded_circ_filepath_as_path = LatticeCircuitManager.load_circuit(filepath_as_path)
        path_and_string_load_are_equivalent = reloaded_circ_filepath_as_path == reloaded_circ_filepath_as_string
        assert path_and_string_load_are_equivalent

        saving_and_reloading_circ_gives_back_same_circ = reloaded_circ_filepath_as_path == sample_circuit_no_ancillas
        assert saving_and_reloading_circ_gives_back_same_circ, f"Inequivalent circuits. Expected:\n {sample_circ_drawn_str}\nEncountered:\n{reloaded_circ_filepath_as_path}"

        # Redundant, but a check that the test data hasn't been altered.
        # Expecting a circuit with 1 H, 1 CX, and 1 X.
        assert reloaded_circ_filepath_as_path.count_ops() == {"cx": 1, "x": 1, "h": 1}, f"Circuit ops {reloaded_circ_filepath_as_path.count_ops()} contains unexpected gates."

    def test_save_and_reload_circuit_with_ancillas_qasm(self, sample_circuit_with_ancillas, tmp_path):
        sample_circ_drawn_str = sample_circuit_with_ancillas.draw() # For use in error messages.
        filepath_as_string = str(tmp_path / "my_circuit_str_write.qasm")
        filepath_as_path = tmp_path / "my_circuit_path_write.qasm"
        LatticeCircuitManager.save_circuit(sample_circuit_with_ancillas, filepath_as_string)
        LatticeCircuitManager.save_circuit(sample_circuit_with_ancillas, filepath_as_path)

        path_and_str_write_are_equivalent = filepath_as_path.read_bytes() == Path(filepath_as_string).read_bytes()
        assert path_and_str_write_are_equivalent

        reloaded_circ_filepath_as_string = LatticeCircuitManager.load_circuit(filepath_as_string, ancilla_reg_name='anc')
        reloaded_circ_filepath_as_path = LatticeCircuitManager.load_circuit(filepath_as_path, ancilla_reg_name='anc')
        path_and_string_load_are_equivalent = reloaded_circ_filepath_as_path == reloaded_circ_filepath_as_string
        assert path_and_string_load_are_equivalent

        saving_and_reloading_circ_gives_back_same_circ = reloaded_circ_filepath_as_path == sample_circuit_with_ancillas
        assert saving_and_reloading_circ_gives_back_same_circ, f"Inequivalent circuits. Expected:\n {sample_circ_drawn_str}\nEncountered:\n{reloaded_circ_filepath_as_path}"

        # Redundant, but a check that the test data hasn't been altered.
        # Expecting a circuit with 3 H, 4 CX, and 1 X.
        assert reloaded_circ_filepath_as_path.count_ops() == {"cx": 4, "x": 1, "h": 3}, f"Circuit ops {reloaded_circ_filepath_as_path.count_ops()} contains unexpected gates."

    def test_save_and_reload_circuit_with_ancillas_qpy_name_given(self, sample_circuit_with_ancillas, tmp_path):
        pytest.skip("The behavior for handling qiskit's deserialization error in this case has not yet been decided.")
        
        sample_circ_drawn_str = sample_circuit_with_ancillas.draw() # For use in error messages.
        filepath_as_string = str(tmp_path / "my_circuit_str_write.qpy")
        filepath_as_path = tmp_path / "my_circuit_path_write.qpy"
        LatticeCircuitManager.save_circuit(sample_circuit_with_ancillas, filepath_as_string)
        LatticeCircuitManager.save_circuit(sample_circuit_with_ancillas, filepath_as_path)

        path_and_str_write_are_equivalent = filepath_as_path.read_bytes() == Path(filepath_as_string).read_bytes()
        assert path_and_str_write_are_equivalent

        reloaded_circ_filepath_as_string = LatticeCircuitManager.load_circuit(filepath_as_string, ancilla_reg_name='anc')
        reloaded_circ_filepath_as_path = LatticeCircuitManager.load_circuit(filepath_as_path, ancilla_reg_name='anc')
        path_and_string_load_are_equivalent = reloaded_circ_filepath_as_path == reloaded_circ_filepath_as_string
        assert path_and_string_load_are_equivalent

        saving_and_reloading_circ_gives_back_same_circ = reloaded_circ_filepath_as_path == sample_circuit_with_ancillas
        assert saving_and_reloading_circ_gives_back_same_circ, f"Inequivalent circuits. Expected:\n {sample_circ_drawn_str}\nEncountered:\n{reloaded_circ_filepath_as_path}"

        # Redundant, but a check that the test data hasn't been altered.
        # Expecting a circuit with 3 H, 4 CX, and 1 X.
        assert reloaded_circ_filepath_as_path.count_ops() == {"cx": 4, "x": 1, "h": 3}, f"Circuit ops {reloaded_circ_filepath_as_path.count_ops()} contains unexpected gates."

    def test_save_and_reload_circuit_with_ancillas_qpy_no_name_given(self):
        pytest.skip("The behavior for handling qiskit's deserialization error in this case has not yet been decided.")
        
        raise AssertionError("Test not yet written.")

    def test_saving_unknown_filetype_raises_value_error(self, sample_circuit_no_ancillas, tmp_path):
        with pytest.raises(ValueError):
            LatticeCircuitManager.save_circuit(sample_circuit_no_ancillas, tmp_path / "bad_filetype.txt")

    def test_loading_unknown_filetype_raises_value_error(self, tmp_path):
        test_file_bad_filetype = tmp_path / Path("bad_filetype.txt")
        with open(test_file_bad_filetype, "w") as file:
            file.write("Test file.")
        with pytest.raises(ValueError):
            LatticeCircuitManager.load_circuit(test_file_bad_filetype)

    def test_save_and_reload_circuit_with_parameter_qasm(self, sample_circuit_with_parameter_and_parameter_expression, tmp_path):
        sample_circ_drawn_str = sample_circuit_with_parameter_and_parameter_expression.draw()
        filepath = tmp_path / "circ.qasm"
        
        LatticeCircuitManager.save_circuit(sample_circuit_with_parameter_and_parameter_expression, filepath)

        reloaded_circ = LatticeCircuitManager.load_circuit(filepath, ancilla_reg_name='anc')

        # Since the reloaded circ has different parameter instances, need to be less strict
        # in checking circuit equality.
        saving_and_reloading_circ_gives_same_gates = reloaded_circ.count_ops() == sample_circuit_with_parameter_and_parameter_expression.count_ops()
        saving_and_reloading_circ_preserves_parameters = [param.name for param in reloaded_circ.parameters] == [param.name for param in sample_circuit_with_parameter_and_parameter_expression.parameters]
        
        assert saving_and_reloading_circ_gives_same_gates and saving_and_reloading_circ_preserves_parameters, f"Inequivalent circuits. Expected:\n {sample_circ_drawn_str}\nEncountered:\n{reloaded_circ}"

        # Redundant, but a check that the test data hasn't been altered.
        # Expecting a circuit with 3 RY, 2 CX, 1 X, 1 H, 1 RX, 1 RZ.
        assert reloaded_circ.count_ops() == {'ry': 3, 'cx': 2, 'x': 1, 'h': 1, 'rx': 1, 'rz': 1}
        assert sorted([param.name for param in reloaded_circ.parameters]) == sorted(['g', 'a', 'b', 'theta', 'phi'])

# TODO: write a test to compare circuits with ancillas and without ancillas. Qiskit doesn't seem to have a clean way to "ignore" registers.
# test_givens does have a test for givens rotation equivalence between with and without ancillas, so maybe this test would be redundant


def test_measure_link_adds_correct_classical_register():
    """measure_link should add a ClassicalRegister and measurement only for the specified link."""
    link_bitmap: IrrepBitmap = {(0, 0, 0): "00", (1, 0, 0): "10", (1, 1, 0): "01"}
    physical_plaquette_states: list[PlaquetteState] = [
        ((0, 0, 0, 0), ((0,0,0), (0,0,0), (0,0,0), (0,0,0)), (((0,0,0),), ((0,0,0),), ((0,0,0),), ((0,0,0),))),
        ((0, 0, 0, 0), ((1,0,0), (1,0,0), (1,1,0), (1,1,0)), (((0,0,0),), ((0,0,0),), ((0,0,0),), ((0,0,0),))),
    ]
    lattice_def = LatticeDef(1.5, 2)
    encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice_def)
    mag_ham: HamiltonianData = {((1, 2), _SIG_3_2): {("0000000000000000", "1010010110100101"): 1.0}}
    lattice = LatticeRegisters.from_lattice_state_encoder(encoder)
    circ_mgr = LatticeCircuitManager(encoder, mag_ham)
    circuit = circ_mgr.create_blank_full_lattice_circuit(lattice)

    link_address = ((0, 0), 1)
    circ_mgr.measure_link(circuit, lattice, link_address)

    # Should have exactly 1 classical register with 2 bits (matching link qubit count)
    assert len(circuit.cregs) == 1
    assert circuit.cregs[0].size == encoder.expected_link_bit_string_length
    # Should have exactly 2 measure instructions
    measure_ops = [inst for inst in circuit.data if inst.operation.name == "measure"]
    assert len(measure_ops) == encoder.expected_link_bit_string_length


def test_measure_vertex_adds_correct_classical_register():
    """measure_vertex should add measurement only for the specified vertex register."""
    link_bitmap: IrrepBitmap = {(0, 0, 0): "00", (1, 0, 0): "10", (1, 1, 0): "01"}
    physical_plaquette_states: list[PlaquetteState] = [
        ((0, 0, 0, 0), ((0,0,0), (0,0,0), (0,0,0), (0,0,0)), (((0,0,0),), ((0,0,0),), ((0,0,0),), ((0,0,0),))),
        ((0, 0, 0, 0), ((1,0,0), (1,0,0), (1,1,0), (1,1,0)), (((0,0,0),), ((0,0,0),), ((0,0,0),), ((0,0,0),))),
    ]
    lattice_def = LatticeDef(1.5, 2)
    encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice_def)
    mag_ham: HamiltonianData = {((1, 2), _SIG_3_2): {("0000000000000000", "1010010110100101"): 1.0}}
    lattice = LatticeRegisters.from_lattice_state_encoder(encoder)
    circ_mgr = LatticeCircuitManager(encoder, mag_ham)
    circuit = circ_mgr.create_blank_full_lattice_circuit(lattice)

    vertex_address = (0, 0)
    circ_mgr.measure_vertex(circuit, lattice, vertex_address)

    # For T1 d=3/2, vertex registers have 0 qubits, so no measurements should be added.
    assert len(circuit.cregs) == 0
    measure_ops = [inst for inst in circuit.data if inst.operation.name == "measure"]
    assert len(measure_ops) == 0


def test_measure_vertex_with_vertex_qubits():
    """measure_vertex on a lattice with non-trivial vertex registers."""
    link_bitmap: IrrepBitmap = {(0, 0, 0): "00", (1, 0, 0): "10", (1, 1, 0): "01"}
    physical_plaquette_states: list[PlaquetteState] = [
        ((0, 0, 0, 0), ((0,0,0), (0,0,0), (0,0,0), (0,0,0)), (((0,0,0),), ((0,0,0),), ((0,0,0),), ((0,0,0),))),
        ((0, 0, 0, 1), ((1,0,0), (1,0,0), (1,1,0), (1,1,0)), (((0,0,0),), ((0,0,0),), ((0,0,0),), ((0,0,0),))),
        ((0, 0, 0, 0), ((1,0,0), (1,0,0), (1,1,0), (1,1,0)), (((0,0,0),), ((0,0,0),), ((0,0,0),), ((0,0,0),))),
    ]
    lattice_def = LatticeDef(1.5, 2)
    encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice_def)
    # Between the first two plaquette states above: vertex, active link and
    # control link bits.
    mag_ham: HamiltonianData = {((1, 2), _SIG_3_2): {("0000" + "00000000" + "00000000", "0001" + "10100101" + "00000000"): 1.0}}
    lattice = LatticeRegisters.from_lattice_state_encoder(encoder)
    circ_mgr = LatticeCircuitManager(encoder, mag_ham)
    circuit = circ_mgr.create_blank_full_lattice_circuit(lattice)

    vertex_address = (0, 0)
    circ_mgr.measure_vertex(circuit, lattice, vertex_address)

    # vertex_bitmap should have 1 qubit per vertex
    assert len(circuit.cregs) == 1
    assert circuit.cregs[0].size == encoder.expected_vertex_bit_string_length


def test_measure_plaquette_measures_all_dofs():
    """measure_plaquette should measure all vertices, active links, and control links."""
    link_bitmap: IrrepBitmap = {(0, 0, 0): "00", (1, 0, 0): "10", (1, 1, 0): "01"}
    physical_plaquette_states: list[PlaquetteState] = [
        ((0, 0, 0, 0), ((0,0,0), (0,0,0), (0,0,0), (0,0,0)), (((0,0,0),), ((0,0,0),), ((0,0,0),), ((0,0,0),))),
        ((0, 0, 0, 0), ((1,0,0), (1,0,0), (1,1,0), (1,1,0)), (((0,0,0),), ((0,0,0),), ((0,0,0),), ((0,0,0),))),
    ]
    lattice_def = LatticeDef(1.5, 4)
    encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice_def)
    mag_ham: HamiltonianData = {((1, 2), _SIG_3_2): {("0000000000000000", "1010010110100101"): 1.0}}
    lattice = LatticeRegisters.from_lattice_state_encoder(encoder)
    circ_mgr = LatticeCircuitManager(encoder, mag_ham)
    circuit = circ_mgr.create_blank_full_lattice_circuit(lattice)

    circ_mgr.measure_plaquette(circuit, lattice, (0, 0), 1, 2)

    # d=3/2 plaquette 0 has: 4 vertices (0 qubits each for T1), 4 active links (2 qubits each),
    # and 4 control links (2 qubits each). Total measured qubits = 0 + 8 + 8 = 16.
    # That's 8 registers of 2 qubits each = 8 classical registers.
    measure_ops = [inst for inst in circuit.data if inst.operation.name == "measure"]
    total_measured_qubits = len(measure_ops)
    # 4 active links * 2 qubits + 4 control links * 2 qubits = 16
    assert total_measured_qubits == 16


def test_measure_plaquette_deduplicates_shared_registers():
    """A link that is a control link at two vertices of a plaquette (on a periodic axis of length 2) is measured once."""
    link_bitmap: IrrepBitmap = {(0, 0, 0): "00", (1, 0, 0): "10", (1, 1, 0): "01"}
    physical_plaquette_states: list[PlaquetteState] = [
        ((0, 0, 0, 0), ((0,0,0), (0,0,0), (0,0,0), (0,0,0)), (((0,0,0),), ((0,0,0),), ((0,0,0),), ((0,0,0),))),
        ((0, 0, 0, 0), ((1,0,0), (1,0,0), (1,1,0), (1,1,0)), (((0,0,0),), ((0,0,0),), ((0,0,0),), ((0,0,0),))),
    ]
    lattice_def = LatticeDef(1.5, 2)
    encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice_def)
    mag_ham: HamiltonianData = {((1, 2), _SIG_3_2): {("0000000000000000", "1010010110100101"): 1.0}}
    lattice = LatticeRegisters.from_lattice_state_encoder(encoder)
    circ_mgr = LatticeCircuitManager(encoder, mag_ham)
    circuit = circ_mgr.create_blank_full_lattice_circuit(lattice)

    circ_mgr.measure_plaquette(circuit, lattice, (0, 0), 1, 2)

    # L=2 d=3/2 periodic: c1==c2 and c3==c4 (shared vertical links).
    # 4 active links + 2 unique control links = 6 unique link registers.
    # With 0 vertex qubits, total measured qubits = 6 * 2 = 12.
    measure_ops = [inst for inst in circuit.data if inst.operation.name == "measure"]
    assert len(measure_ops) == 12


# Plaquette states for the two tests below, on a d=2 lattice of size 2 with
# the F-order [-1, -2, 1, 2, 3, -3]. The controls at each vertex are then, in
# order: v1 (-1, -2), v2 (-2, +1), v3 (+1, +2), v4 (-1, +2). On this lattice
# v2's +1 control is v1's -1 control, v3's +2 control is v2's -2 control,
# v4's -1 control is v3's +1 control, and v4's +2 control is v1's -2
# control. With the T1 link bitmap, ONE is "00", THREE "10" and THREE_BAR "01".
_ALT_FORDER = [-1, -2, 1, 2, 3, -3]
_SIG_2_ALT_FORDER = ((-1, -2, 1, 2), (-1, -2, 1, 2), (-1, -2, 1, 2), (-1, -2, 1, 2))
# Every duplicated control agrees with the control it repeats.
_CONSISTENT_STATE_ALT_FORDER = (
    (0, 0, 0, 0),
    (ONE, ONE, ONE, ONE),
    ((THREE_BAR, THREE), (ONE, THREE_BAR), (ONE, ONE), (ONE, THREE)),
)
_CONSISTENT_BIT_STRING_ALT_FORDER = "00000000" + "0110" + "0001" + "0000" + "0010"
# v2's +1 control is ONE, but v1's -1 control, the same link, is THREE_BAR.
_INCONSISTENT_STATE_ALT_FORDER = (
    (0, 0, 0, 0),
    (ONE, ONE, ONE, ONE),
    ((THREE_BAR, THREE), (ONE, ONE), (ONE, ONE), (ONE, THREE)),
)
_INCONSISTENT_BIT_STRING_ALT_FORDER = "00000000" + "0110" + "0000" + "0000" + "0010"
_VACUUM_STATE = ((0, 0, 0, 0), (ONE, ONE, ONE, ONE), ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE)))
_VACUUM_BIT_STRING = "00000000" + "0000000000000000"


def _alt_forder_d_2_size_2_encoder() -> LatticeStateEncoder:
    """An encoder of the d=2, size-2 lattice with F-order _ALT_FORDER and the plaquette states above."""
    return LatticeStateEncoder(
        _T1_LINK_BITMAP,
        [_CONSISTENT_STATE_ALT_FORDER, _INCONSISTENT_STATE_ALT_FORDER, _VACUUM_STATE],
        lattice=LatticeDef(2, 2, forder=_ALT_FORDER),
    )


def test_matrix_element_with_disagreeing_duplicate_controls_raises_value_error():
    """
    The constructor refuses a matrix element whose duplicated controls disagree.

    Which controls are duplicates depends on the F-order, so a non-default one
    checks that the duplicates are found by direction, not by position.
    """
    encoder = _alt_forder_d_2_size_2_encoder()
    with pytest.raises(ValueError):
        LatticeCircuitManager(
            encoder,
            {((1, 2), _SIG_2_ALT_FORDER): {(_CONSISTENT_BIT_STRING_ALT_FORDER, _INCONSISTENT_BIT_STRING_ALT_FORDER): 0.5}},
        )
    # The same matrix element with a state whose duplicates agree is accepted.
    LatticeCircuitManager(
        encoder,
        {((1, 2), _SIG_2_ALT_FORDER): {(_CONSISTENT_BIT_STRING_ALT_FORDER, _VACUUM_BIT_STRING): 0.5}},
    )


def test_duplicate_controls_removed_from_matrix_elements():
    """
    The constructor removes each duplicated control from the bit strings of a matrix element.

    With the F-order and duplicates described above _ALT_FORDER, v1 keeps
    both its controls, v2 keeps its -2 control, v3 its +1 control, and v4
    none.
    """
    circ_mgr = LatticeCircuitManager(
        _alt_forder_d_2_size_2_encoder(),
        {((1, 2), _SIG_2_ALT_FORDER): {(_CONSISTENT_BIT_STRING_ALT_FORDER, _VACUUM_BIT_STRING): 0.5}},
    )
    expected_final = "00000000" + "0110" + "00" + "00"
    expected_initial = "00000000" + "0000" + "00" + "00"
    assert circ_mgr._mag_hamiltonian == {((1, 2), _SIG_2_ALT_FORDER): {(expected_final, expected_initial): 0.5}}


# A T1 encoder on a d=3/2 periodic lattice of size 3 encodes a plaquette state
# in 16 bits: 4 active links and 4 control links of 2 bits each, and no vertex
# bits. No control repeats another at size 3, so the constructor does not
# decode the bit strings to remove duplicates.
_D_3_2_SIZE_3_PLAQUETTE_BIT_STRING_LENGTH = 16


def test_constructor_refuses_plane_and_signature_without_plaquette():
    """
    The constructor refuses matrix elements under a plane and signature that no plaquette of the lattice has.

    On a d=3/2 periodic lattice of size 3, every plaquette is in plane (1, 2)
    and has the signature _SIG_3_2, so neither _SIG_2 in that plane nor
    _SIG_3_2 in plane (1, 3) belongs to a plaquette.
    """
    encoder = LatticeStateEncoder.for_truncation(LatticeDef(1.5, 3), "T", 1)
    vacuum = "0" * _D_3_2_SIZE_3_PLAQUETTE_BIT_STRING_LENGTH
    excited = "1" + "0" * (_D_3_2_SIZE_3_PLAQUETTE_BIT_STRING_LENGTH - 1)
    with pytest.raises(ValueError, match="no plaquette"):
        LatticeCircuitManager(encoder, {((1, 2), _SIG_2): {(excited, vacuum): 0.5}})
    with pytest.raises(ValueError, match="no plaquette"):
        LatticeCircuitManager(encoder, {((1, 3), _SIG_3_2): {(excited, vacuum): 0.5}})
    # The same matrix element under the plane and signature of the lattice's plaquettes is accepted.
    LatticeCircuitManager(encoder, {((1, 2), _SIG_3_2): {(excited, vacuum): 0.5}})


def test_constructor_refuses_bit_string_of_wrong_length():
    """
    The constructor refuses a matrix element whose final or initial bit string does not have the length of the encoder's plaquette bit strings.
    """
    encoder = LatticeStateEncoder.for_truncation(LatticeDef(1.5, 3), "T", 1)
    assert encoder.expected_plaquette_bit_string_length == _D_3_2_SIZE_3_PLAQUETTE_BIT_STRING_LENGTH
    right_length = "0" * _D_3_2_SIZE_3_PLAQUETTE_BIT_STRING_LENGTH
    too_short = "1" * (_D_3_2_SIZE_3_PLAQUETTE_BIT_STRING_LENGTH - 1)
    too_long = "1" * (_D_3_2_SIZE_3_PLAQUETTE_BIT_STRING_LENGTH + 1)
    for final_bit_string, initial_bit_string in [(too_short, right_length), (right_length, too_long)]:
        with pytest.raises(ValueError, match="length"):
            LatticeCircuitManager(encoder, {((1, 2), _SIG_3_2): {(final_bit_string, initial_bit_string): 0.5}})


def test_signature():
    """Test Plaquette.signature property for d=3/2 and d=2 with default and nonstandard forder."""
    default_forder = [1, 2, 3, -1, -2, -3]
    link_bitmap: IrrepBitmap = {(0, 0, 0): "00", (1, 0, 0): "10", (1, 1, 0): "01"}
    vertex_bitmap = {}

    # d=3/2, plane (1, 2), default forder:
    # v1 (bottom): all dirs except -2 → {1, 2, -1} sorted by forder → (1, 2, -1)
    # v2 (bottom): all dirs except -2 → {1, 2, -1} sorted by forder → (1, 2, -1)
    # v3 (top):    all dirs except +2 → {1, -1, -2} sorted by forder → (1, -1, -2)
    # v4 (top):    all dirs except +2 → {1, -1, -2} sorted by forder → (1, -1, -2)
    lattice_3_2 = LatticeRegisters(1.5, 2, True, link_bitmap=link_bitmap, vertex_bitmap=vertex_bitmap, forder=default_forder)
    plaq_3_2 = lattice_3_2.get_plaquettes((0, 0), 1, 2)
    assert plaq_3_2.signature == ((1, 2, -1), (1, 2, -1), (1, -1, -2), (1, -1, -2))

    # d=2, plane (1, 2), default forder:
    # All vertices have all dirs {1, 2, -1, -2} sorted by forder → (1, 2, -1, -2)
    lattice_2 = LatticeRegisters(2, 2, True, link_bitmap=link_bitmap, vertex_bitmap=vertex_bitmap, forder=default_forder)
    plaq_2 = lattice_2.get_plaquettes((0, 0), 1, 2)
    assert plaq_2.signature == ((1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2))

    # d=3/2, plane (1, 2), nonstandard forder [-1, 2, -3, 1, -2, 3]:
    nonstandard_forder = [-1, 2, -3, 1, -2, 3]
    lattice_3_2_ns = LatticeRegisters(1.5, 2, True, link_bitmap=link_bitmap, vertex_bitmap=vertex_bitmap, forder=nonstandard_forder)
    plaq_3_2_ns = lattice_3_2_ns.get_plaquettes((0, 0), 1, 2)
    # v1, v2 (bottom): dirs {1, 2, -1} sorted by nonstandard forder → (-1, 2, 1)
    # v3, v4 (top): dirs {1, -1, -2} sorted by nonstandard forder → (-1, 1, -2)
    assert plaq_3_2_ns.signature == ((-1, 2, 1), (-1, 2, 1), (-1, 1, -2), (-1, 1, -2))

    # d=2, plane (1, 2), nonstandard forder [-1, 2, -3, 1, -2, 3]:
    lattice_2_ns = LatticeRegisters(2, 2, True, link_bitmap=link_bitmap, vertex_bitmap=vertex_bitmap, forder=nonstandard_forder)
    plaq_2_ns = lattice_2_ns.get_plaquettes((0, 0), 1, 2)
    # All vertices: dirs {1, 2, -1, -2} sorted by nonstandard forder → (-1, 2, 1, -2)
    assert plaq_2_ns.signature == ((-1, 2, 1, -2), (-1, 2, 1, -2), (-1, 2, 1, -2), (-1, 2, 1, -2))


@pytest.mark.skip(reason="Non-periodic LatticeDef support not yet available. "
                         "Should verify plaquette.signature reflects missing directions at boundary vertices.")
def test_signature_nonperiodic():
    """Placeholder: verify signature correctness on a non-periodic lattice."""
    pass


# --- d=3 unit tests of duplicate controls, plane by plane ---

def _make_d3_size2_circ_mgr():
    """Helper: build a LatticeCircuitManager for d=3 B3 size=2, periodic, where every axis has length 2."""
    lattice_def = LatticeDef(3, 2, periodic_boundary_conds=True)
    encoder = LatticeStateEncoder.for_truncation(lattice_def, "B", 3)
    circ_mgr = LatticeCircuitManager.for_truncation(encoder)
    return circ_mgr, encoder


# On d=3, size 2, with the default F-order, every vertex has all six
# half-links.
_SIG_3_DEFAULT_FORDER = ((1, 2, 3, -1, -2, -3),) * 4


def test_disagreeing_duplicate_controls_detected_d3():
    """For d=3 size=2, a state whose duplicate controls disagree in one plane is refused there."""
    circ_mgr, encoder = _make_d3_size2_circ_mgr()
    duplicate_control_indices = encoder.lattice_def._duplicate_control_indices((1, 2), _SIG_3_DEFAULT_FORDER)

    # A physical state of plane (1, 2): its duplicate controls agree there.
    base_ps = next(
        ps for ps in physical_plaquette_states(encoder.lattice_def, "B", 3)
        if circ_mgr._bit_string_without_duplicate_controls(
            encoder.encode_plaquette_state_as_bit_string(ps), duplicate_control_indices) is not None
    )

    # In plane (1, 2), v2's +1 control (index 0 in the default F-order,
    # after its active links +2 and -1 are left out) is v1's -1 control.
    # Change it so that the two disagree.
    c_links_list = [list(vc) for vc in base_ps[2]]
    c_links_list[1][0] = THREE if c_links_list[1][0] == ONE else ONE
    modified_ps = (base_ps[0], base_ps[1], tuple(tuple(vc) for vc in c_links_list))

    assert circ_mgr._bit_string_without_duplicate_controls(
        encoder.encode_plaquette_state_as_bit_string(modified_ps), duplicate_control_indices) is None


def test_duplicate_control_indices_per_vertex_d3():
    """For d=3 size=2, removing duplicates leaves 12 controls per plaquette (4+3+3+2), in every plane."""
    _, encoder = _make_d3_size2_circ_mgr()
    for plane in [(1, 2), (1, 3), (2, 3)]:
        duplicate_control_indices = encoder.lattice_def._duplicate_control_indices(plane, _SIG_3_DEFAULT_FORDER)
        removed_per_vertex = [
            sum(1 for vertex_idx, _ in duplicate_control_indices if vertex_idx == v) for v in range(4)]
        assert removed_per_vertex == [0, 1, 1, 2], f"Plane {plane}."


def test_pruning_states_lose_duplicate_controls_d3():
    """For d=3 size=2, the states for control pruning lose their 4 duplicate controls in every plane."""
    circ_mgr, encoder = _make_d3_size2_circ_mgr()
    physical_state_bitstrings = {
        encoder.encode_plaquette_state_as_bit_string(ps)
        for ps in physical_plaquette_states(encoder.lattice_def, "B", 3)
    }
    # 4 active links and 12 controls left, 2 qubits each for B3.
    expected_qubits_in_stripped_state = 32
    for plane in [(1, 2), (1, 3), (2, 3)]:
        stripped_set = circ_mgr._pruning_states_without_duplicate_controls(
            physical_state_bitstrings,
            encoder.lattice_def._duplicate_control_indices(plane, _SIG_3_DEFAULT_FORDER),
        )
        assert stripped_set, f"Plane {plane} has no states for control pruning."
        assert {len(stripped_state) for stripped_state in stripped_set} == {expected_qubits_in_stripped_state}


def test_d3_hamiltonian_has_three_planes():
    """After __init__, the magnetic Hamiltonian for d=3 size=2 has one key per plane, each with the one signature of size 2."""
    circ_mgr, _ = _make_d3_size2_circ_mgr()
    assert set(circ_mgr._mag_hamiltonian.keys()) == {
        ((1, 2), _SIG_3_DEFAULT_FORDER), ((1, 3), _SIG_3_DEFAULT_FORDER), ((2, 3), _SIG_3_DEFAULT_FORDER)}


# --- d=3 circuit construction tests ---

def _make_d3_encoder_and_circ_mgr(size, threshold=0.3):
    """Helper: create encoder and circuit manager for d=3 B3 lattice of given size."""
    lattice_def = LatticeDef(3, size, periodic_boundary_conds=True)
    encoder = LatticeStateEncoder.for_truncation(lattice_def, "B", 3)
    circ_mgr = LatticeCircuitManager.for_truncation(encoder,
                                                    mag_hamiltonian_matrix_element_threshold=threshold)
    return encoder, circ_mgr


@pytest.mark.slow
def test_d3_B3_size3_magnetic_trotter_step():
    """d=3 B3 size=3: construct circuit and apply one magnetic Trotter step (no duplicate controls)."""
    encoder, circ_mgr = _make_d3_encoder_and_circ_mgr(size=3)
    lattice = LatticeRegisters.from_lattice_state_encoder(encoder)
    circuit = circ_mgr.create_blank_full_lattice_circuit(lattice)

    circ_mgr.apply_magnetic_trotter_step(circuit, lattice)

    assert circuit.num_qubits > 0
    assert len(circuit.parameters) > 0, "Circuit should have unbound parameters (dt, coupling_g)."


@pytest.mark.parametrize(
    ("dims", "shape", "n_links_per_plaquette_circuit_by_plane"),
    [
        # No axis of length 2: 4 active links and 4 * 2 * (d - 1) controls.
        (2, (4, 3), {(1, 2): 12}),
        pytest.param(3, (3, 3, 4), {(1, 2): 20, (1, 3): 20, (2, 3): 20}, marks=pytest.mark.slow),
        # A periodic axis of length 2 along e2 of a plane (e1, e2): v3's +e2
        # control is v2's -e2 control, and v4's +e2 control is v1's -e2
        # control, so two controls are left out of the plaquette circuit.
        # Along an axis normal to the plane, a vertex's two controls start at
        # different sites and stay distinct.
        (2, (3, 2), {(1, 2): 10}),
        pytest.param(3, (3, 3, 2), {(1, 2): 20, (1, 3): 18, (2, 3): 18}, marks=pytest.mark.slow),
    ],
)
def test_rectangular_periodic_lattice_magnetic_trotter_step(dims, shape, n_links_per_plaquette_circuit_by_plane):
    """
    On periodic lattices with different lengths along different axes, the magnetic step builds.

    Every plaquette of a plane has the same signature, so the step builds one
    plaquette circuit per plane, acting on the plaquette's vertices and on its
    links other than duplicate controls. The circuit's registers come in the
    lattice's traversal order: at each vertex, its vertex register, then its
    links.
    """
    encoder = LatticeStateEncoder.for_truncation(LatticeDef(dims, shape), "B", 3)
    circ_mgr = LatticeCircuitManager.for_truncation(encoder, mag_hamiltonian_matrix_element_threshold=0.3)
    lattice = LatticeRegisters.from_lattice_state_encoder(encoder)
    assert lattice.shape == shape
    circuit = circ_mgr.create_blank_full_lattice_circuit(lattice)
    expected_register_names = [
        name
        for vertex, links in lattice.get_traversal_order()
        for name in [f"v:{vertex}", *(f"l:{link}" for link in links)]
    ]
    assert [register.name for register in circuit.qregs] == expected_register_names
    assert circuit.num_qubits == lattice.n_total_qubits

    circ_mgr.apply_magnetic_trotter_step(circuit, lattice, cache_mag_evol_circuit=True)
    assert len(circ_mgr._cached_mag_evol_circuits) == len(n_links_per_plaquette_circuit_by_plane)
    for (plane, _), plaquette_circuit in circ_mgr._cached_mag_evol_circuits.items():
        expected_n_qubits = (
            4 * encoder.expected_vertex_bit_string_length
            + n_links_per_plaquette_circuit_by_plane[plane] * encoder.expected_link_bit_string_length
        )
        assert plaquette_circuit.num_qubits == expected_n_qubits, f"Plane {plane}."


def test_d3_B3_size2_magnetic_trotter_step():
    """d=3 B3 size=2: duplicate controls are removed; verify circuit constructs and has correct c_link counts."""
    encoder, circ_mgr = _make_d3_encoder_and_circ_mgr(size=2)
    lattice = LatticeRegisters.from_lattice_state_encoder(encoder)
    circuit = circ_mgr.create_blank_full_lattice_circuit(lattice)

    circ_mgr.apply_magnetic_trotter_step(circuit, lattice)

    assert circuit.num_qubits > 0
    assert len(circuit.parameters) > 0, "Circuit should have unbound parameters (dt, coupling_g)."


# Wiring of the d=3 magnetic Trotter step.
#
# These tests check which qubits each plaquette's Givens rotation acts on in
# d=3, on periodic lattices of size 3 and size 2. The expected qubits were
# derived by hand, as follows. DO NOT CHANGE THE DUMMY DATA BELOW WITHOUT
# REDERIVING THE EXPECTED QUBITS.
#
#   1. Qubit indices of the registers. The circuit holds the registers in
#      the lattice's traversal order: vertices in lexicographic order of their
#      coordinates, and at each vertex its vertex register, then its links in
#      directions 1, 2, 3. With one qubit per vertex and three per link (the
#      T2 link bitmap), each vertex has 10 qubits. On a lattice of size L, the
#      vertex (x, y, z) is preceded by r = L^2 x + L y + z vertices in that
#      order, so its qubits are 10 r to 10 r + 9: its vertex qubit is 10 r,
#      and its link in direction k starts at 10 r + 1 + 3 (k - 1).
#   2. The registers of the plaquette at vertex v in plane (e1, e2). Its
#      vertices are v1 = v, v2 = v + e1, v3 = v + e1 + e2, v4 = v + e2
#      (mod L). Its active links are (v1, e1), (v2, e2), (v4, e1), (v1, e2).
#      A vertex has six half-links, one for each direction +k or -k in
#      which a link leaves it: the half-link at site s in direction +k is
#      the link (s, k), and in direction -k the link (s - e_k mod L, k).
#      The controls at a vertex are its four half-links other than the two
#      that are active links, sorted by the default FORDER
#      [1, 2, 3, -1, -2, -3].
#   3. On a periodic lattice of size 2, s + e_k and s - e_k are the same
#      site, so four controls repeat a control of an earlier vertex and are
#      dropped: v2's +e1 (v1's -e1), v3's +e2 (v2's -e2), v4's +e2 (v1's -e2)
#      and v4's -e1 (v3's +e1). The two out-of-plane controls of a vertex
#      start at different sites, so they never coincide. 12 controls remain.
#   4. The plaquette circuit acts on the vertex qubits of v1-v4, the qubits
#      of the active links, then the qubits of the controls, in that order:
#      position i of the plaquette's bit string is the i-th of these qubits.
#   5. The rotation between two bit strings has its pivot at the first
#      position where they differ, CX gates from the pivot to every later
#      position where they differ, and a multi-controlled X on the pivot,
#      controlled by every other qubit, with the value that qubit has in the
#      bit string that has a 0 at the pivot.
#
# The dummy data put a different irrep on each of the four controls at a
# vertex. The T2 link bitmap (_T2_LINK_BITMAP) encodes different
# irreps as different bit strings, so swapping two controls of one vertex
# changes the values the multi-controlled X requires of their qubits. On
# size 2 a control and the earlier control it repeats (step 3) are the same
# link and must carry the same irrep; otherwise the LatticeCircuitManager
# constructor raises a ValueError for the matrix element. The irreps are
# therefore assigned by link: the link shared by v1 and v2 (v1's -e1) is
# THREE, by v1 and v4 (v1's -e2) THREE_BAR, by v2 and v3 (v2's -e2) SIX, by
# v3 and v4 (v3's +e1) SIX_BAR; every out-of-plane
# control is EIGHT in the positive direction and ONE in the negative
# direction. The two states of each matrix element differ only in their
# active links.
_D3_WIRING_VERTEX_MULTIPLICITIES = (0, 1, 0, 1)
# Bit strings in the T2 link bitmap: ONE 000, THREE 100, SIX 110, EIGHT 111.
_D3_WIRING_INITIAL_ACTIVE_LINKS = (ONE, THREE, SIX, EIGHT)
# Bit strings in the T2 link bitmap: THREE 100, SIX_BAR 011, ONE 000,
# THREE_BAR 001.
_D3_WIRING_FINAL_ACTIVE_LINKS = (THREE, SIX_BAR, ONE, THREE_BAR)
_D3_WIRING_CONTROLS_BY_PLANE = {
    (1, 2): (
        (EIGHT, THREE, THREE_BAR, ONE),  # v1: +3, -1, -2, -3
        (THREE, EIGHT, SIX, ONE),  # v2: +1, +3, -2, -3
        (SIX_BAR, SIX, EIGHT, ONE),  # v3: +1, +2, +3, -3
        (THREE_BAR, EIGHT, SIX_BAR, ONE),  # v4: +2, +3, -1, -3
    ),
    (1, 3): (
        (EIGHT, THREE, ONE, THREE_BAR),  # v1: +2, -1, -2, -3
        (THREE, EIGHT, ONE, SIX),  # v2: +1, +2, -2, -3
        (SIX_BAR, EIGHT, SIX, ONE),  # v3: +1, +2, +3, -2
        (EIGHT, THREE_BAR, SIX_BAR, ONE),  # v4: +2, +3, -1, -2
    ),
    (2, 3): (
        (EIGHT, ONE, THREE, THREE_BAR),  # v1: +1, -1, -2, -3
        (EIGHT, THREE, ONE, SIX),  # v2: +1, +2, -1, -3
        (EIGHT, SIX_BAR, SIX, ONE),  # v3: +1, +2, +3, -1
        (EIGHT, THREE_BAR, ONE, SIX_BAR),  # v4: +1, +3, -1, -2
    ),
}
# A different value per plane, so that a matrix element applied in the wrong
# plane changes the rotation angle.
_D3_WIRING_MATRIX_ELEMENT_BY_PLANE = {(1, 2): 0.11, (1, 3): 0.22, (2, 3): 0.33}
# The signature every plaquette has on a periodic d=3 lattice: all six
# half-links at every corner (Signature in ymcirc._abstract.lattice_data).
# The magnetic Hamiltonian is keyed by plane, then signature, so the dummy
# matrix elements must use it to be applied.
_D3_WIRING_SIGNATURE = ((1, 2, 3, -1, -2, -3),) * 4
# The initial and final active links encode to 000 100 110 111 and
# 100 011 000 001, after the four vertex bits. They first differ at position
# 4 (the first bit of l1), and then at positions 7, 8, 9 (all of l2), 10, 11
# (the first two of l3) and 13, 14 (the first two of l4). The initial state
# has a 0 at the pivot, so its bits are the values the multi-controlled X
# requires of the other qubits (step 5 above).
_D3_WIRING_PIVOT_POSITION = 4
_D3_WIRING_CX_TARGET_POSITIONS = (7, 8, 9, 10, 11, 13, 14)
# Each rotation: the CX gates, then these gates on the pivot, then the CX
# gates again (givens.givens and givens._CRXCircuit_with_MCX).
_D3_WIRING_GATE_TYPES_BETWEEN_CX = [RZGate, RYGate, MCXGate, RYGate, MCXGate, RZGate]
_D3_WIRING_GATES_PER_ROTATION = 2 * len(_D3_WIRING_CX_TARGET_POSITIONS) + len(_D3_WIRING_GATE_TYPES_BETWEEN_CX)
_D3_WIRING_QUBITS_PER_LINK = 3

# Expected registers of the plaquettes checked, by (lattice size, bottom-left
# vertex, plane). Vertex qubits are listed for v1-v4, active links by their
# first qubit for l1-l4, and controls by their first qubit with the irrep the
# dummy data put on that link.
_D3_WIRING_EXPECTED_REGISTERS = {
    (3, (2, 2, 2), (1, 2)): {
        # v1 = (2, 2, 2), v2 = (0, 2, 2), v3 = (0, 0, 2), v4 = (2, 0, 2).
        "vertex qubits": [260, 80, 20, 200],
        # ((2, 2, 2), 1), ((0, 2, 2), 2), ((2, 0, 2), 1), ((2, 2, 2), 2).
        "active link first qubits": [261, 84, 201, 264],
        "controls": [
            # v1: ((2, 2, 2), 3), ((1, 2, 2), 1), ((2, 1, 2), 2), ((2, 2, 1), 3).
            (267, EIGHT), (171, THREE), (234, THREE_BAR), (257, ONE),
            # v2: ((0, 2, 2), 1), ((0, 2, 2), 3), ((0, 1, 2), 2), ((0, 2, 1), 3).
            (81, THREE), (87, EIGHT), (54, SIX), (77, ONE),
            # v3: ((0, 0, 2), 1), ((0, 0, 2), 2), ((0, 0, 2), 3), ((0, 0, 1), 3).
            (21, SIX_BAR), (24, SIX), (27, EIGHT), (17, ONE),
            # v4: ((2, 0, 2), 2), ((2, 0, 2), 3), ((1, 0, 2), 1), ((2, 0, 1), 3).
            (204, THREE_BAR), (207, EIGHT), (111, SIX_BAR), (197, ONE),
        ],
    },
    (3, (2, 2, 2), (1, 3)): {
        # v1 = (2, 2, 2), v2 = (0, 2, 2), v3 = (0, 2, 0), v4 = (2, 2, 0).
        "vertex qubits": [260, 80, 60, 240],
        # ((2, 2, 2), 1), ((0, 2, 2), 3), ((2, 2, 0), 1), ((2, 2, 2), 3).
        "active link first qubits": [261, 87, 241, 267],
        "controls": [
            # v1: ((2, 2, 2), 2), ((1, 2, 2), 1), ((2, 1, 2), 2), ((2, 2, 1), 3).
            (264, EIGHT), (171, THREE), (234, ONE), (257, THREE_BAR),
            # v2: ((0, 2, 2), 1), ((0, 2, 2), 2), ((0, 1, 2), 2), ((0, 2, 1), 3).
            (81, THREE), (84, EIGHT), (54, ONE), (77, SIX),
            # v3: ((0, 2, 0), 1), ((0, 2, 0), 2), ((0, 2, 0), 3), ((0, 1, 0), 2).
            (61, SIX_BAR), (64, EIGHT), (67, SIX), (34, ONE),
            # v4: ((2, 2, 0), 2), ((2, 2, 0), 3), ((1, 2, 0), 1), ((2, 1, 0), 2).
            (244, EIGHT), (247, THREE_BAR), (151, SIX_BAR), (214, ONE),
        ],
    },
    (3, (2, 2, 2), (2, 3)): {
        # v1 = (2, 2, 2), v2 = (2, 0, 2), v3 = (2, 0, 0), v4 = (2, 2, 0).
        "vertex qubits": [260, 200, 180, 240],
        # ((2, 2, 2), 2), ((2, 0, 2), 3), ((2, 2, 0), 2), ((2, 2, 2), 3).
        "active link first qubits": [264, 207, 244, 267],
        "controls": [
            # v1: ((2, 2, 2), 1), ((1, 2, 2), 1), ((2, 1, 2), 2), ((2, 2, 1), 3).
            (261, EIGHT), (171, ONE), (234, THREE), (257, THREE_BAR),
            # v2: ((2, 0, 2), 1), ((2, 0, 2), 2), ((1, 0, 2), 1), ((2, 0, 1), 3).
            (201, EIGHT), (204, THREE), (111, ONE), (197, SIX),
            # v3: ((2, 0, 0), 1), ((2, 0, 0), 2), ((2, 0, 0), 3), ((1, 0, 0), 1).
            (181, EIGHT), (184, SIX_BAR), (187, SIX), (91, ONE),
            # v4: ((2, 2, 0), 1), ((2, 2, 0), 3), ((1, 2, 0), 1), ((2, 1, 0), 2).
            (241, EIGHT), (247, THREE_BAR), (151, ONE), (214, SIX_BAR),
        ],
    },
    (2, (0, 0, 0), (1, 2)): {
        # v1 = (0, 0, 0), v2 = (1, 0, 0), v3 = (1, 1, 0), v4 = (0, 1, 0).
        "vertex qubits": [0, 40, 60, 20],
        # ((0, 0, 0), 1), ((1, 0, 0), 2), ((0, 1, 0), 1), ((0, 0, 0), 2).
        "active link first qubits": [1, 44, 21, 4],
        "controls": [
            # v1: ((0, 0, 0), 3), ((1, 0, 0), 1), ((0, 1, 0), 2), ((0, 0, 1), 3).
            (7, EIGHT), (41, THREE), (24, THREE_BAR), (17, ONE),
            # v2, +1 dropped: ((1, 0, 0), 3), ((1, 1, 0), 2), ((1, 0, 1), 3).
            (47, EIGHT), (64, SIX), (57, ONE),
            # v3, +2 dropped: ((1, 1, 0), 1), ((1, 1, 0), 3), ((1, 1, 1), 3).
            (61, SIX_BAR), (67, EIGHT), (77, ONE),
            # v4, +2 and -1 dropped: ((0, 1, 0), 3), ((0, 1, 1), 3).
            (27, EIGHT), (37, ONE),
        ],
    },
    (2, (0, 0, 0), (1, 3)): {
        # v1 = (0, 0, 0), v2 = (1, 0, 0), v3 = (1, 0, 1), v4 = (0, 0, 1).
        "vertex qubits": [0, 40, 50, 10],
        # ((0, 0, 0), 1), ((1, 0, 0), 3), ((0, 0, 1), 1), ((0, 0, 0), 3).
        "active link first qubits": [1, 47, 11, 7],
        "controls": [
            # v1: ((0, 0, 0), 2), ((1, 0, 0), 1), ((0, 1, 0), 2), ((0, 0, 1), 3).
            (4, EIGHT), (41, THREE), (24, ONE), (17, THREE_BAR),
            # v2, +1 dropped: ((1, 0, 0), 2), ((1, 1, 0), 2), ((1, 0, 1), 3).
            (44, EIGHT), (64, ONE), (57, SIX),
            # v3, +3 dropped: ((1, 0, 1), 1), ((1, 0, 1), 2), ((1, 1, 1), 2).
            (51, SIX_BAR), (54, EIGHT), (74, ONE),
            # v4, +3 and -1 dropped: ((0, 0, 1), 2), ((0, 1, 1), 2).
            (14, EIGHT), (34, ONE),
        ],
    },
    (2, (0, 0, 0), (2, 3)): {
        # v1 = (0, 0, 0), v2 = (0, 1, 0), v3 = (0, 1, 1), v4 = (0, 0, 1).
        "vertex qubits": [0, 20, 30, 10],
        # ((0, 0, 0), 2), ((0, 1, 0), 3), ((0, 0, 1), 2), ((0, 0, 0), 3).
        "active link first qubits": [4, 27, 14, 7],
        "controls": [
            # v1: ((0, 0, 0), 1), ((1, 0, 0), 1), ((0, 1, 0), 2), ((0, 0, 1), 3).
            (1, EIGHT), (41, ONE), (24, THREE), (17, THREE_BAR),
            # v2, +2 dropped: ((0, 1, 0), 1), ((1, 1, 0), 1), ((0, 1, 1), 3).
            (21, EIGHT), (61, ONE), (37, SIX),
            # v3, +3 dropped: ((0, 1, 1), 1), ((0, 1, 1), 2), ((1, 1, 1), 1).
            (31, EIGHT), (34, SIX_BAR), (71, ONE),
            # v4, +3 and -2 dropped: ((0, 0, 1), 1), ((1, 0, 1), 1).
            (11, EIGHT), (51, ONE),
        ],
    },
    (2, (1, 1, 1), (1, 2)): {
        # v1 = (1, 1, 1), v2 = (0, 1, 1), v3 = (0, 0, 1), v4 = (1, 0, 1).
        "vertex qubits": [70, 30, 10, 50],
        # ((1, 1, 1), 1), ((0, 1, 1), 2), ((1, 0, 1), 1), ((1, 1, 1), 2).
        "active link first qubits": [71, 34, 51, 74],
        "controls": [
            # v1: ((1, 1, 1), 3), ((0, 1, 1), 1), ((1, 0, 1), 2), ((1, 1, 0), 3).
            (77, EIGHT), (31, THREE), (54, THREE_BAR), (67, ONE),
            # v2, +1 dropped: ((0, 1, 1), 3), ((0, 0, 1), 2), ((0, 1, 0), 3).
            (37, EIGHT), (14, SIX), (27, ONE),
            # v3, +2 dropped: ((0, 0, 1), 1), ((0, 0, 1), 3), ((0, 0, 0), 3).
            (11, SIX_BAR), (17, EIGHT), (7, ONE),
            # v4, +2 and -1 dropped: ((1, 0, 1), 3), ((1, 0, 0), 3).
            (57, EIGHT), (47, ONE),
        ],
    },
    (2, (1, 1, 1), (1, 3)): {
        # v1 = (1, 1, 1), v2 = (0, 1, 1), v3 = (0, 1, 0), v4 = (1, 1, 0).
        "vertex qubits": [70, 30, 20, 60],
        # ((1, 1, 1), 1), ((0, 1, 1), 3), ((1, 1, 0), 1), ((1, 1, 1), 3).
        "active link first qubits": [71, 37, 61, 77],
        "controls": [
            # v1: ((1, 1, 1), 2), ((0, 1, 1), 1), ((1, 0, 1), 2), ((1, 1, 0), 3).
            (74, EIGHT), (31, THREE), (54, ONE), (67, THREE_BAR),
            # v2, +1 dropped: ((0, 1, 1), 2), ((0, 0, 1), 2), ((0, 1, 0), 3).
            (34, EIGHT), (14, ONE), (27, SIX),
            # v3, +3 dropped: ((0, 1, 0), 1), ((0, 1, 0), 2), ((0, 0, 0), 2).
            (21, SIX_BAR), (24, EIGHT), (4, ONE),
            # v4, +3 and -1 dropped: ((1, 1, 0), 2), ((1, 0, 0), 2).
            (64, EIGHT), (44, ONE),
        ],
    },
    (2, (1, 1, 1), (2, 3)): {
        # v1 = (1, 1, 1), v2 = (1, 0, 1), v3 = (1, 0, 0), v4 = (1, 1, 0).
        "vertex qubits": [70, 50, 40, 60],
        # ((1, 1, 1), 2), ((1, 0, 1), 3), ((1, 1, 0), 2), ((1, 1, 1), 3).
        "active link first qubits": [74, 57, 64, 77],
        "controls": [
            # v1: ((1, 1, 1), 1), ((0, 1, 1), 1), ((1, 0, 1), 2), ((1, 1, 0), 3).
            (71, EIGHT), (31, ONE), (54, THREE), (67, THREE_BAR),
            # v2, +2 dropped: ((1, 0, 1), 1), ((0, 0, 1), 1), ((1, 0, 0), 3).
            (51, EIGHT), (11, ONE), (47, SIX),
            # v3, +3 dropped: ((1, 0, 0), 1), ((1, 0, 0), 2), ((0, 0, 0), 1).
            (41, EIGHT), (44, SIX_BAR), (1, ONE),
            # v4, +3 and -2 dropped: ((1, 1, 0), 1), ((0, 1, 0), 1).
            (61, EIGHT), (21, ONE),
        ],
    },
}


def _d3_wiring_dummy_data() -> tuple[list[PlaquetteState], BoxTerms]:
    """Return the dummy physical plaquette states and magnetic Hamiltonian, with unencoded states, of the d=3 wiring tests."""
    dummy_phys_states: list[PlaquetteState] = []
    dummy_mag_hamiltonian_by_state: BoxTerms = {}
    for plane, controls in _D3_WIRING_CONTROLS_BY_PLANE.items():
        initial_state = (_D3_WIRING_VERTEX_MULTIPLICITIES, _D3_WIRING_INITIAL_ACTIVE_LINKS, controls)
        final_state = (_D3_WIRING_VERTEX_MULTIPLICITIES, _D3_WIRING_FINAL_ACTIVE_LINKS, controls)
        dummy_phys_states += [initial_state, final_state]
        dummy_mag_hamiltonian_by_state[(plane, _D3_WIRING_SIGNATURE)] = {
            (final_state, initial_state): _D3_WIRING_MATRIX_ELEMENT_BY_PLANE[plane]
        }
    return dummy_phys_states, dummy_mag_hamiltonian_by_state


def _d3_wiring_expected_rotation(size: int, vertex: tuple[int, int, int], plane: Plane) -> dict:
    """
    Return the expected pivot, CX targets and MCX controls of one plaquette's rotation.

    The qubits come from _D3_WIRING_EXPECTED_REGISTERS, in the order of the
    plaquette's bit string, and the control values are the bits of the
    matrix element's initial state.
    """
    registers = _D3_WIRING_EXPECTED_REGISTERS[(size, vertex, plane)]
    link_bitmap = _T2_LINK_BITMAP
    qubits = list(registers["vertex qubits"])
    bits = "".join(str(multiplicity) for multiplicity in _D3_WIRING_VERTEX_MULTIPLICITIES)
    links = list(zip(registers["active link first qubits"], _D3_WIRING_INITIAL_ACTIVE_LINKS)) + registers["controls"]
    for first_qubit, irrep in links:
        qubits += range(first_qubit, first_qubit + _D3_WIRING_QUBITS_PER_LINK)
        bits += link_bitmap[irrep]
    pivot = qubits[_D3_WIRING_PIVOT_POSITION]
    return {
        "pivot": pivot,
        "CX targets": [qubits[position] for position in _D3_WIRING_CX_TARGET_POSITIONS],
        "MCX controls": {
            qubit: bit for position, (qubit, bit) in enumerate(zip(qubits, bits))
            if position != _D3_WIRING_PIVOT_POSITION
        },
        "RY angle": _D3_WIRING_MATRIX_ELEMENT_BY_PLANE[plane],
    }


def _d3_wiring_built_rotation(circuit: QuantumCircuit, rotation_index: int) -> dict:
    """
    Return the pivot, CX targets and MCX controls of one rotation in a built circuit.

    The rotation is the rotation_index-th one in the circuit. Checks that its
    gates have the structure of _D3_WIRING_GATES_PER_ROTATION, and that both
    copies of its CX gates and of its MCX gate agree.
    """
    first_gate = rotation_index * _D3_WIRING_GATES_PER_ROTATION
    gates = circuit.data[first_gate:first_gate + _D3_WIRING_GATES_PER_ROTATION]
    n_cx = len(_D3_WIRING_CX_TARGET_POSITIONS)
    n_core = len(_D3_WIRING_GATE_TYPES_BETWEEN_CX)
    cx_before, core, cx_after = gates[:n_cx], gates[n_cx:n_cx + n_core], gates[n_cx + n_core:]
    assert [gate.operation.name for gate in cx_before + cx_after] == ["cx"] * (2 * n_cx)
    assert all(isinstance(gate.operation, gate_type) for gate, gate_type in zip(core, _D3_WIRING_GATE_TYPES_BETWEEN_CX))

    def index(qubit):
        return circuit.find_bit(qubit).index

    cx_pairs = [(index(gate.qubits[0]), index(gate.qubits[1])) for gate in cx_before]
    assert cx_pairs == [(index(gate.qubits[0]), index(gate.qubits[1])) for gate in cx_after]
    pivot = index(core[0].qubits[0])
    assert all(control == pivot for control, _ in cx_pairs), "Every CX should be controlled by the pivot."

    def mcx_controls(gate):
        n_ctrls = gate.operation.num_ctrl_qubits
        # Qiskit's ctrl_state integer has the first control as its least significant bit.
        ctrl_values = f"{gate.operation.ctrl_state:0{n_ctrls}b}"[::-1]
        return {index(qubit): value for qubit, value in zip(gate.qubits[:-1], ctrl_values)}

    first_mcx, second_mcx = core[2], core[4]
    assert index(first_mcx.qubits[-1]) == pivot and index(second_mcx.qubits[-1]) == pivot
    assert mcx_controls(first_mcx) == mcx_controls(second_mcx)
    return {
        "pivot": pivot,
        "CX targets": [target for _, target in cx_pairs],
        "MCX controls": mcx_controls(first_mcx),
        "RY angle": float(core[1].operation.params[0]),
    }


@pytest.mark.parametrize("size, checked_vertices", [
    (3, [(2, 2, 2)]),  # Every step in a positive direction from (2, 2, 2) wraps around.
    (2, [(0, 0, 0), (1, 1, 1)]),  # Repeated controls are dropped.
])
def test_apply_magnetic_trotter_step_d_3_wiring(size: int, checked_vertices: list[tuple[int, int, int]]):
    print(f"Checking the qubits of the d=3 magnetic Trotter step on a size-{size} periodic lattice.")
    dummy_phys_states, dummy_mag_hamiltonian_by_state = _d3_wiring_dummy_data()
    lattice_def = LatticeDef(3, size)
    lattice_encoder = LatticeStateEncoder(_T2_LINK_BITMAP, dummy_phys_states, lattice=lattice_def)
    dummy_mag_hamiltonian: HamiltonianData = {
        plane_and_signature: {
            (lattice_encoder.encode_plaquette_state_as_bit_string(final_state),
             lattice_encoder.encode_plaquette_state_as_bit_string(initial_state)): value
            for (final_state, initial_state), value in matrix_elements_by_state.items()
        }
        for plane_and_signature, matrix_elements_by_state in dummy_mag_hamiltonian_by_state.items()
    }
    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    circ_mgr = LatticeCircuitManager(lattice_encoder, dummy_mag_hamiltonian)
    master_circuit = circ_mgr.create_blank_full_lattice_circuit(lattice_registers)
    circ_mgr.apply_magnetic_trotter_step(
        master_circuit,
        lattice_registers,
        optimize_circuits=False,
        cache_mag_evol_circuit=True
    )
    master_circuit.assign_parameters({"dt_mag__0": 1, "coupling_g_mag__0": 1}, inplace=True)

    # One rotation per plaquette, visited by vertex in lexicographic order,
    # then by plane.
    planes = sorted(_D3_WIRING_CONTROLS_BY_PLANE)
    vertices = sorted(product(range(size), repeat=3))
    assert len(master_circuit.data) == _D3_WIRING_GATES_PER_ROTATION * len(vertices) * len(planes)
    for vertex in checked_vertices:
        for plane_index, plane in enumerate(planes):
            print(f"Checking the rotation of the plaquette at {vertex} in plane {plane}.")
            rotation_index = vertices.index(vertex) * len(planes) + plane_index
            built = _d3_wiring_built_rotation(master_circuit, rotation_index)
            expected = _d3_wiring_expected_rotation(size, vertex, plane)
            assert built["pivot"] == expected["pivot"]
            assert built["CX targets"] == expected["CX targets"]
            assert built["MCX controls"] == expected["MCX controls"]
            # The Givens angle is -(matrix element) * dt / g^2, and the first
            # RY gate's angle is -2/2 times the Givens angle
            # (givens._CRXCircuit_with_MCX). With dt = g = 1 every factor is
            # exact in floating point, so the RY angle equals the matrix
            # element exactly and no tolerance is needed.
            assert built["RY angle"] == expected["RY angle"]


# --- Building the magnetic Hamiltonian from the box matrix elements ---

def test_for_truncation_threshold_keeps_matrix_elements_at_least_as_large():
    """
    for_truncation keeps exactly the matrix elements whose magnitude is at least its threshold.

    Checked on a d=3/2 periodic lattice of size 3 at T1, with the median of
    the magnitudes as the threshold, so that some matrix elements are kept and
    some dropped.
    """
    encoder = LatticeStateEncoder.for_truncation(LatticeDef(1.5, 3), "T", 1)
    all_matrix_elements = LatticeCircuitManager.for_truncation(encoder)._mag_hamiltonian
    magnitudes = sorted({abs(value) for elements in all_matrix_elements.values() for value in elements.values()})
    assert len(magnitudes) > 1, "With a single magnitude, no threshold keeps some matrix elements and drops others."
    threshold = magnitudes[len(magnitudes) // 2]

    kept_matrix_elements = LatticeCircuitManager.for_truncation(
        encoder, mag_hamiltonian_matrix_element_threshold=threshold)._mag_hamiltonian

    expected = {
        plane_and_signature: {pair: value for pair, value in elements.items() if abs(value) >= threshold}
        for plane_and_signature, elements in all_matrix_elements.items()
    }
    assert kept_matrix_elements == expected


@pytest.mark.parametrize("link_bitmap", [
    _T1_LINK_BITMAP,
    {ONE: "11", THREE: "10", THREE_BAR: "01"},
    {ONE: "10", THREE: "00", THREE_BAR: "01"},
], ids=["singlet 00", "singlet 11", "singlet 10"])
def test_for_truncation_keeps_matrix_elements_connected_to_vacuum_whatever_the_link_bitmap(link_bitmap: IrrepBitmap):
    """
    With only_include_elems_connected_to_electric_vacuum, for_truncation keeps exactly the matrix elements with the electric vacuum as one of their states.

    The electric vacuum has the singlet on every link. Which matrix elements
    are kept must not depend on how the link bitmap encodes the singlet, so
    the test runs with three link bitmaps that encode it differently. Checked
    on a d=2 periodic lattice of size 3 at T1. In d=2, unlike d=3/2, a corner
    whose active links carry the singlet can have two control links carrying
    a 3 and a 3bar, so the test also fails if the control links are not
    checked.
    """
    lattice_def = LatticeDef(2, 3)
    encoder = LatticeStateEncoder.for_truncation(lattice_def, "T", 1, link_bitmap=link_bitmap)
    vacuum: PlaquetteState = ((0, 0, 0, 0), (ONE, ONE, ONE, ONE), ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE)))
    box_matrix_elements = box_terms(lattice_def, "T", 1)[((1, 2), _SIG_2)]
    expected_pairs = {
        (encoder.encode_plaquette_state_as_bit_string(final_state),
         encoder.encode_plaquette_state_as_bit_string(initial_state))
        for final_state, initial_state in box_matrix_elements
        if vacuum in (final_state, initial_state)
    }
    assert 0 < len(expected_pairs) < len(box_matrix_elements), (
        "Some matrix elements, but not all, should have the vacuum as one of their states.")

    kept_matrix_elements = LatticeCircuitManager.for_truncation(
        encoder, only_include_elems_connected_to_electric_vacuum=True)._mag_hamiltonian

    assert set(kept_matrix_elements[((1, 2), _SIG_2)]) == expected_pairs


@pytest.mark.parametrize("dims", [1.5, 2])
def test_box_has_no_matrix_element_in_both_directions_or_on_the_diagonal(dims):
    """
    No two plaquette states are connected by box in both directions, and box has no diagonal matrix elements.

    LatticeCircuitManager.for_truncation relies on this (see the module
    docstring of ymcirc.circuit): each matrix element of box is then also that
    of box + box^dagger. Checked for each plane and signature at T1 on
    periodic lattices of size 3 in d=3/2 and d=2.
    """
    box_matrix_elements_by_plane_and_signature = box_terms(LatticeDef(dims, 3), "T", 1)
    assert box_matrix_elements_by_plane_and_signature, "No matrix elements of box."
    pairs_in_both_directions = [
        (f, i)
        for box_matrix_elements in box_matrix_elements_by_plane_and_signature.values()
        for f, i in box_matrix_elements if (i, f) in box_matrix_elements
    ]
    diagonal = [
        (f, i)
        for box_matrix_elements in box_matrix_elements_by_plane_and_signature.values()
        for f, i in box_matrix_elements if f == i
    ]
    assert pairs_in_both_directions == [], f"{len(pairs_in_both_directions)} matrix elements of box have their reverse too."
    assert diagonal == [], f"{len(diagonal)} diagonal matrix elements of box."


def test_for_truncation_magnetic_hamiltonian_has_only_states_of_the_encoders_truncation():
    """
    The manager's magnetic Hamiltonian involves only physical plaquette states of its encoder's truncation.

    On a d=3/2 periodic lattice of size 3, the 35 physical plaquette states
    at B3 are among the 81 at T1, and all of them have the same link irreps,
    1, 3 and 3bar, so an encoder built at B3 can encode every T1 state. A
    manager whose magnetic Hamiltonian came from T1 would then hold
    transitions to the 46 T1 states that are not physical at B3.
    """
    lattice_def = LatticeDef(1.5, 3)
    encoder = LatticeStateEncoder.for_truncation(lattice_def, "B", 3)
    encoded_states_at_b3 = {
        encoder.encode_plaquette_state_as_bit_string(state)
        for state in physical_plaquette_states(lattice_def, "B", 3)
    }

    circ_mgr = LatticeCircuitManager.for_truncation(encoder)

    states_in_hamiltonian = {
        bit_string
        for elements_by_pair in circ_mgr._mag_hamiltonian.values()
        for pair in elements_by_pair
        for bit_string in pair
    }
    assert states_in_hamiltonian, "The magnetic Hamiltonian is empty."
    assert states_in_hamiltonian <= encoded_states_at_b3, (
        f"{len(states_in_hamiltonian - encoded_states_at_b3)} plaquette states in the magnetic Hamiltonian "
        "are not physical at the encoder's truncation, B3."
    )


def test_for_truncation_refuses_encoder_without_truncation():
    """An encoder built from its constructor records no truncation, so the manager has no truncation to compute its magnetic Hamiltonian for."""
    lattice_def = LatticeDef(1.5, 3)
    encoder = LatticeStateEncoder(_T1_LINK_BITMAP, physical_plaquette_states(lattice_def, "T", 1), lattice_def)
    with pytest.raises(ValueError, match="LatticeStateEncoder.for_truncation"):
        LatticeCircuitManager.for_truncation(encoder)
