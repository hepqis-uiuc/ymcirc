from ymcirc._abstract import LatticeDef
from ymcirc.circuit import LatticeCircuitManager
from ymcirc.conventions import LatticeStateEncoder, ONE, THREE, THREE_BAR, IRREP_TRUNCATION_DICT_1_3_3BAR
from ymcirc.conventions import SIX, SIX_BAR, EIGHT, IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8, PHYSICAL_PLAQUETTE_STATES, load_magnetic_hamiltonian
from ymcirc.lattice_registers import LatticeRegisters
from ymcirc.utilities import _flatten_circuit, _check_circuits_logically_equivalent
from qiskit.circuit import QuantumCircuit, AncillaRegister
from qiskit.circuit.library.standard_gates import RXGate, RZGate, RYGate, MCXGate
from qiskit.quantum_info import Operator, Statevector, DensityMatrix, partial_trace
import numpy as np


def test_create_blank_full_lattice_circuit_has_promised_register_order():
    """Check in some cases that we get the ordering promised in the method docstring."""
    # Creating test data.
    # Not physically meaningful, but has the right format.
    iweight_one = (0, 0, 0)
    iweight_three = (1, 0, 0)
    irrep_bitmap = {
        iweight_one: "0",
        iweight_three: "1"
    }
    physical_plaquette_states_3halves_no_vertices_needed = [
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            (iweight_one, iweight_one, iweight_one, iweight_one)
        ),
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_three, iweight_three, iweight_three),
            (iweight_three, iweight_one, iweight_one, iweight_one)
        )
    ]
    physical_plaquette_states_3halves = [
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            (iweight_one, iweight_one, iweight_one, iweight_one)
        ),
        (
            (1, 1, 1, 1),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            (iweight_one, iweight_one, iweight_one, iweight_one)
        ),
        (
            (2, 2, 2, 2),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            (iweight_one, iweight_one, iweight_one, iweight_one)
        ),
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_three, iweight_three, iweight_three),
            (iweight_three, iweight_one, iweight_one, iweight_one)
        )
    ]
    physical_plaquette_states_2d = [
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            (iweight_one, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one)
        ),
        (
            (1, 1, 1, 1),
            (iweight_one, iweight_one, iweight_three, iweight_one),
            (iweight_one, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one)
        ),
        (
            (0, 0, 0, 0),
            (iweight_one, iweight_three, iweight_three, iweight_three),
            (iweight_three, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one, iweight_one)
        )
    ]
    # Hamiltonian bitstrings take the form vertex_bits + active link bits + c link bits.
    # For the "no_vertices" data, vertex_bits is the empty string. The numbers of
    # Vertex bits and link bits can be inferred from the test data (encode integer in bitstring, use link bitmap).
    mag_hamiltonian_2d = [("1110111100000000", "0001000011111111", -0.33), ("0000111100000000", "1111000011111111", 1.0)]
    mag_hamiltonian_3halves = [("1010010111110000", "0000000011110000", 1.0), ("0000000010100101", "1010101000000001", 1.0)]
    mag_hamiltonian_3halves_no_vertices = [("10101111", "11110010", 1.0), ("10010000", "10000001", 1.0), ("11111101", "00000101", 1.0)]
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
    # Registers for a lattice ith size 2 (small enough for the same link to control multiple vertices in a single plaquette).
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
    for expected_register_names_ordered, link_bitmap, physical_plaquette_states, dims, size, hamiltonian in test_cases:
        # Initialize registers and create circuit.
        lattice_encoder = LatticeStateEncoder(
            link_bitmap, physical_plaquette_states, LatticeDef(dims, size))
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
    dummy_mag_hamiltonian = [
        ("00100000" + "00000000", "01010100" + "10011010", 0.33)  # One matrix element, plaquette only has a_link and c_link substrings.
    ]
    dummy_phys_states = [
        (  # Matches the first encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (ONE, THREE, ONE, ONE),
            (ONE, ONE, ONE, ONE)
        ),
        (  # Matches the second encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, ONE),
            (THREE, THREE_BAR, THREE, THREE)
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
        circ_with_mcx.append(RYGate(-1.0*angle), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RYGate(1.0*angle), pivot_qubit)
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
        IRREP_TRUNCATION_DICT_1_3_3BAR,
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
        optimize_circuits=False
    )
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
    dummy_mag_hamiltonian = [
        ("00100001" + "00000000", "01010110" + "10011010", 0.33),  # One matrix element, plaquette only has a_link and c_link substrings. Should get filtered out based on c_link consistency.
        ("00100001" + "00000000", "01010110" + "10100000", 0.33)  # One matrix element, plaquette only has a_link and c_link substrings. Should not get filtered out based on c_link consistency.
    ]
    dummy_phys_states = [
        (  # Matches the first encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (ONE, THREE, ONE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (  # Matches the second encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            (THREE, THREE_BAR, THREE, THREE)
        ),
        (  # Matches the third encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            (THREE, THREE, ONE, ONE)
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
        circ_with_mcx.append(RYGate(-1.0*angle), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RYGate(1.0*angle), pivot_qubit)
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
        IRREP_TRUNCATION_DICT_1_3_3BAR,
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
        optimize_circuits=False
    )
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
    dummy_mag_hamiltonian = [
        ("0000" + "00100000" + "0000000000000010", "0010" + "01010100" + "1001101000000010", 0.33)  # One matrix element, plaquette has v, a_link, and c_link substrings.
    ]
    dummy_phys_states = [
        (  # Matches the first encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (ONE, THREE, ONE, ONE),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, THREE)
        ),
        (  # Matches the second encoded state in the dummy magnetic hamiltonian.
            (0, 0, 1, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, ONE),
            (THREE, THREE_BAR, THREE, THREE, ONE, ONE, ONE, THREE)
        )
    ]
    expected_master_circuit = QuantumCircuit(45)
    expected_rotation_gates = {  # Data for constructing the expected circuit.
        "angle": -0.165,
        "MCU ctrl state": "000000000000000000000000011",  # Little endian per qiskit convention.
        "givens rotations": [
            {
                "pivot": 20,
                "CX targets": [2, 18, 19, 7, 31, 14, 28, 16],
                "MCU ctrls": [18, 36] + [0, 15, 5, 1, 2, 19, 6, 7, 3, 4, 31, 32, 13, 14, 28, 29, 16, 17, 21, 22, 23, 24, 8, 9, 37] # on ctrls first, followed by off, with pivot skipped.
            },
            {
                "pivot": 25,
                "CX targets": [7, 23, 24, 12, 36, 4, 18, 21],
                "MCU ctrls": [23, 41] + [5, 20, 10, 6, 7, 24, 11, 12, 8, 9, 36, 37, 3, 4, 18, 19, 21, 22, 26, 27, 28, 29, 13, 14, 42]
            },
            {
                "pivot": 15,
                "CX targets": [12, 28, 29, 2, 41, 9, 23, 26],
                "MCU ctrls": [28, 31] + [10, 25, 0, 11, 12, 29, 1, 2, 13, 14, 41, 42, 8, 9, 23, 24, 26, 27, 16, 17, 18, 19, 3, 4, 32]
            },
            {
                "pivot": 35,
                "CX targets": [17, 33, 34, 22, 1, 29, 43, 31],
                "MCU ctrls": [33, 6] + [15, 30, 20, 16, 17, 34, 21, 22, 18, 19, 1, 2, 28, 29, 43, 44, 31, 32, 36, 37, 38, 39, 23, 24, 7]
            },
            {
                "pivot": 40,
                "CX targets": [22, 38, 39, 27, 6, 19, 33, 36],
                "MCU ctrls": [38, 11] + [20, 35, 25, 21, 22, 39, 26, 27, 23, 24, 6, 7, 18, 19, 33, 34, 36, 37, 41, 42, 43, 44, 28, 29, 12]
            },
            {
                "pivot": 30,
                "CX targets": [27, 43, 44, 17, 11, 24, 38, 41],
                "MCU ctrls": [43, 1] + [25, 40, 15, 26, 27, 44, 16, 17, 28, 29, 11, 12, 23, 24, 38, 39, 41, 42, 31, 32, 33, 34, 18, 19, 2]
            },
            {
                "pivot": 5,
                "CX targets": [32, 3, 4, 37, 16, 44, 13, 1],
                "MCU ctrls": [3, 21] + [30, 0, 35, 31, 32, 4, 36, 37, 33, 34, 16, 17, 43, 44, 13, 14, 1, 2, 6, 7, 8, 9, 38, 39, 22]
            },
            {
                "pivot": 10,
                "CX targets": [37, 8, 9, 42, 21, 34, 3, 6],
                "MCU ctrls": [8, 26] + [35, 5, 40, 36, 37, 9, 41, 42, 38, 39, 21, 22, 33, 34, 3, 4, 6, 7, 11, 12, 13, 14, 43, 44, 27]
            },
            {
                "pivot": 0,
                "CX targets": [42, 13, 14, 32, 26, 39, 8, 11],
                "MCU ctrls": [13, 16] + [40, 10, 30, 41, 42, 14, 31, 32, 43, 44, 26, 27, 38, 39, 8, 9, 11, 12, 1, 2, 3, 4, 33, 34, 17]
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
        circ_with_mcx.append(RYGate(-1.0*angle), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RYGate(1.0*angle), pivot_qubit)
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
        IRREP_TRUNCATION_DICT_1_3_3BAR,
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
        optimize_circuits=False
    )
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
    dummy_mag_hamiltonian = [
        ("0000" + "00100000" + "0000000000000010", "0010" + "01010100" + "1001101000000010", 0.33), # One matrix element, plaquette has v, a_link, and c_link substrings. Should get filtered out based on c_link consistency.
        ("0000" + "00100000" + "0000000010000010", "0010" + "01010100" + "1001101000100100", 0.33)  # One matrix element, plaquette has v, a_link, and c_link substrings. Should not get filtered out based on c_link consistency.
    ]
    dummy_phys_states = [
        (  # Matches the first encoded state in the dummy magnetic hamiltonian that isn't discarded.
            (0, 0, 0, 0),
            (ONE, THREE, ONE, ONE),
            (ONE, ONE, ONE, ONE, THREE, ONE, ONE, THREE)
        ),
        (  # Matches the second encoded state in the dummy magnetic hamiltonian that isn't discarded.
            (0, 0, 1, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, ONE),
            (THREE, THREE_BAR, THREE, THREE, ONE, THREE, THREE_BAR, ONE)
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
        circ_with_mcx.append(RYGate(-1.0*angle), pivot_qubit)
        circ_with_mcx.append(MCXGate(num_ctrl_qubits=num_ctrls, ctrl_state=ctrl_state), ctrls + pivot_qubit)
        circ_with_mcx.append(RYGate(1.0*angle), pivot_qubit)
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
        IRREP_TRUNCATION_DICT_1_3_3BAR,
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
        optimize_circuits=False
    )
    print("Obtained circuit:")
    print(master_circuit)

    # Checking equivalence via helper methods for
    # (1) flattening a circuit down to a single register and (2) comparing
    # logical equivalence of two circuits.
    assert _check_circuits_logically_equivalent(_flatten_circuit(master_circuit), expected_master_circuit), "Encountered inequivalent circuits."


def test_apply_electric_trotter_step_d_3_2_lattice():
    print("Checking that the electric trotter step acts as expected on a T2 3x1 lattice.")
    dummy_electric_hamiltonian = [0.33,0.66,0.66,0.99,0.66,0.99,0.99,0.33]
    dummy_mag_hamiltonian = []
    dummy_phys_states = [
        (
            (0,0,0,0),
            (ONE, THREE, ONE, THREE_BAR),
            (THREE_BAR, THREE_BAR, THREE, THREE)
        ),
        (
            (0,0,0,0),
            (EIGHT,SIX,EIGHT,SIX_BAR),
            (SIX_BAR,SIX_BAR,SIX,SIX)
        )
    ]
    lattice_def = LatticeDef(1.5,3)
    lattice_encoder = LatticeStateEncoder(
    IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8,
    dummy_phys_states,
    lattice=lattice_def)
    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    circ_mgr = LatticeCircuitManager(lattice_encoder,
                                 dummy_mag_hamiltonian)
    master_circuit = circ_mgr.create_blank_full_lattice_circuit(
    lattice_registers)
    circ_mgr.apply_electric_trotter_step(master_circuit,lattice_registers,dummy_electric_hamiltonian,electric_gray_order=True)

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

    dimensionality_and_truncation_string = "d=3/2, T1"
    lattice_def = LatticeDef(1.5, 2)

    lattice_encoder = LatticeStateEncoder(
        IRREP_TRUNCATION_DICT_1_3_3BAR,
        PHYSICAL_PLAQUETTE_STATES[dimensionality_and_truncation_string],
        lattice=lattice_def)

    physical_plaquette_states = set(lattice_encoder.encode_plaquette_state_as_bit_string(plaquette) for plaquette in PHYSICAL_PLAQUETTE_STATES[dimensionality_and_truncation_string])

    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    magnetic_hamiltonian = load_magnetic_hamiltonian(
        dimensionality_and_truncation_string,
        lattice_encoder)

    circ_mgr = LatticeCircuitManager(lattice_encoder, magnetic_hamiltonian)
    master_circuit = circ_mgr.create_blank_full_lattice_circuit(
        lattice_registers)

    circ_mgr.num_ancillas = circ_mgr.compute_num_ancillas_needed_from_mag_trotter_step(master_circuit, lattice_registers, control_fusion=True, 
        physical_states_for_control_pruning=physical_plaquette_states,
        optimize_circuits=False)
    circ_mgr.add_ancilla_register_to_quantum_circuit(master_circuit)

    # Number of ancillas for d=3/2, T1, 1x2 with control pruning and fusing found in arXiv:2503.08866
    assert (len(master_circuit.ancillas) == 3), "Ancilla register for d=3/2, T1 with control pruning and fusion is improperly initiated"
    print("Test for ancilla register count passed")


def test_creating_correct_ancilla_register_for_d_2_T1_small():
    print("Checking if an ancilla register with the correct number of ancillas are added are added for d=2, T1, 2x2")

    dimensionality_and_truncation_string = "d=2, T1"
    lattice_def = LatticeDef(2, 2)

    lattice_encoder = LatticeStateEncoder(
        IRREP_TRUNCATION_DICT_1_3_3BAR,
        PHYSICAL_PLAQUETTE_STATES[dimensionality_and_truncation_string],
        lattice=lattice_def)

    physical_plaquette_states = set(lattice_encoder.encode_plaquette_state_as_bit_string(plaquette) for plaquette in PHYSICAL_PLAQUETTE_STATES[dimensionality_and_truncation_string])

    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    magnetic_hamiltonian = load_magnetic_hamiltonian(
        dimensionality_and_truncation_string,
        lattice_encoder)

    circ_mgr = LatticeCircuitManager(lattice_encoder, magnetic_hamiltonian)
    master_circuit = circ_mgr.create_blank_full_lattice_circuit(
        lattice_registers)

    circ_mgr.num_ancillas = circ_mgr.compute_num_ancillas_needed_from_mag_trotter_step(master_circuit, lattice_registers, control_fusion=True, 
        physical_states_for_control_pruning=physical_plaquette_states,
        optimize_circuits=False)
    circ_mgr.add_ancilla_register_to_quantum_circuit(master_circuit)

    # Number of ancillas for d=2, T1, 2x2 with control pruning and fusing found in arXiv:2503.08866
    assert (len(master_circuit.ancillas) == 7), "Ancilla register for d=2, T1 with control pruning and fusion is improperly initiated"
    print("Test for ancilla register count passed")


def test_magnetic_with_ancilla_has_no_MCX():
    print("Checking that the MCX gates are properly decomposed in the magnetic trotter step")

    dimensionality_and_truncation_string = "d=3/2, T1"
    lattice_def = LatticeDef(1.5, 2)

    lattice_encoder = LatticeStateEncoder(
        IRREP_TRUNCATION_DICT_1_3_3BAR,
        PHYSICAL_PLAQUETTE_STATES[dimensionality_and_truncation_string],
        lattice=lattice_def)

    physical_plaquette_states = set(lattice_encoder.encode_plaquette_state_as_bit_string(plaquette) for plaquette in PHYSICAL_PLAQUETTE_STATES[dimensionality_and_truncation_string])

    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
    magnetic_hamiltonian = load_magnetic_hamiltonian(
        dimensionality_and_truncation_string,
        lattice_encoder)

    circ_mgr = LatticeCircuitManager(lattice_encoder, magnetic_hamiltonian)
    master_circuit_with_ancillas = circ_mgr.create_blank_full_lattice_circuit(
        lattice_registers)

    circ_mgr.num_ancillas = circ_mgr.compute_num_ancillas_needed_from_mag_trotter_step(master_circuit_with_ancillas, lattice_registers, control_fusion=True, 
        physical_states_for_control_pruning=physical_plaquette_states,
        optimize_circuits=False)
    circ_mgr.add_ancilla_register_to_quantum_circuit(master_circuit_with_ancillas)

    circ_mgr.apply_magnetic_trotter_step(master_circuit_with_ancillas, lattice_registers, physical_states_for_control_pruning=physical_plaquette_states, control_fusion=True)

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
    dummy_mag_hamiltonian = [
        ("00100001" + "00000000", "01010110" + "10011010", 0.33),  # One matrix element, plaquette only has a_link and c_link substrings. Should get filtered out based on c_link consistency.
        ("00100001" + "00000000", "01010110" + "10100000", 0.33)  # One matrix element, plaquette only has a_link and c_link substrings. Should not get filtered out based on c_link consistency.
    ]
    dummy_phys_states = [
        (  # Matches the first encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (ONE, THREE, ONE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (  # Matches the second encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            (THREE, THREE_BAR, THREE, THREE)
        ),
        (  # Matches the third encoded state in the dummy magnetic hamiltonian.
            (0, 0, 0, 0),
            (THREE_BAR, THREE_BAR, THREE_BAR, THREE),
            (THREE, THREE, ONE, ONE)
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
        num_ctrls = len(rotation_data["MCU ctrls"])
        ctrl_state = expected_rotation_gates["MCU ctrl state"]

        # Givens rotation with ancilla qubits 
        circ_with_mcx = QuantumCircuit(12);
        circ_with_mcx.add_register(AncillaRegister(9, "anc"))
        circ_with_mcx.append(RZGate(-1.0*np.pi/2.0), pivot_qubit)
        circ_with_mcx.append(RYGate(-1.0*angle), pivot_qubit)
        circ_with_mcx.mcx(ctrls, pivot_qubit, ancilla_qubits=list(range(12,21)), ctrl_state = ctrl_state, mode='v-chain')
        circ_with_mcx.append(RYGate(1.0*angle), pivot_qubit)
        circ_with_mcx.mcx(ctrls, pivot_qubit, ancilla_qubits=list(range(12,21)), ctrl_state = ctrl_state, mode='v-chain')
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
        IRREP_TRUNCATION_DICT_1_3_3BAR,
        dummy_phys_states,
        lattice=lattice_def)
    lattice_registers = LatticeRegisters.from_lattice_state_encoder(lattice_encoder)
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
        optimize_circuits=False
    )
    print("Obtained circuit:")
    print(master_circuit)

    # Checking equivalence via helper methods for
    # (1) flattening a circuit down to a single register and (2) comparing
    # logical equivalence of two circuits.
    assert _check_circuits_logically_equivalent(_flatten_circuit(master_circuit), _flatten_circuit(expected_master_circuit)), "Encountered inequivalent circuits."
# TODO: write a test to compare circuits with ancillas and without ancillas. Qiskit doesn't seem to have a clean way to "ignore" registers. 
# test_givens does have a test for givens rotation equivalence between with and without ancillas, so maybe this test would be redundant
