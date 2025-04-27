from ymcirc.circuit import LatticeCircuitManager
from ymcirc.conventions import LatticeStateEncoder
from ymcirc.lattice_registers import LatticeRegisters


def test_create_blank_full_lattice_circuit_has_promised_register_order():
    """Check in some cases that we get the ordering promised in the method docstring."""
    # Creating test data.
    # Not physically meaningful, but has the right format.
    irrep_bitmap = {(0, 0, 0): "0", (1, 0, 0): "1"}
    singlet_bitmap_2d = {
        (((0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)), 1): "00",
        (((0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)), 2): "01",
        (((1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)), 1): "10",
    }
    singlet_bitmap_3halves = {
        (((0, 0, 0), (1, 0, 0), (1, 0, 0)), 1): "0",
        (((0, 0, 0), (1, 0, 0), (1, 0, 0)), 2): "1",
    }
    singlet_bitmap_3halves_no_vertices = {}
    mag_hamiltonian_2d = [
        ("000110000001", "000110000010", 1.0),
        ("010110000001", "100110000010", 1.0),
    ]
    mag_hamiltonian_3halves = [
        ("00001111", "11110000", 1.0),
        ("10100101", "00000001", 1.0),
    ]
    mag_hamiltonian_3halves_no_vertices = [
        ("1111", "000", 1.0),
        ("1001", "0001", 1.0),
        ("1101", "0101", 1.0),
    ]
    expected_register_order_2d = [
        "v:(0, 0)",
        "l:((0, 0), 1)",
        "l:((0, 0), 2)",
        "v:(0, 1)",
        "l:((0, 1), 1)",
        "l:((0, 1), 2)",
        "v:(1, 0)",
        "l:((1, 0), 1)",
        "l:((1, 0), 2)",
        "v:(1, 1)",
        "l:((1, 1), 1)",
        "l:((1, 1), 2)",
    ]
    expected_register_order_3halves = [
        "v:(0, 0)",
        "l:((0, 0), 1)",
        "l:((0, 0), 2)",
        "v:(0, 1)",
        "l:((0, 1), 1)",
        "v:(1, 0)",
        "l:((1, 0), 1)",
        "l:((1, 0), 2)",
        "v:(1, 1)",
        "l:((1, 1), 1)",
    ]
    expected_register_order_3halves_no_vertices = [
        "l:((0, 0), 1)",
        "l:((0, 0), 2)",
        "l:((0, 1), 1)",
        "l:((1, 0), 1)",
        "l:((1, 0), 2)",
        "l:((1, 1), 1)",
    ]
    test_cases = [
        (
            expected_register_order_2d,
            irrep_bitmap,
            singlet_bitmap_2d,
            2,
            mag_hamiltonian_2d,
        ),
        (
            expected_register_order_3halves,
            irrep_bitmap,
            singlet_bitmap_3halves,
            1.5,
            mag_hamiltonian_3halves,
        ),
        (
            expected_register_order_3halves_no_vertices,
            irrep_bitmap,
            singlet_bitmap_3halves_no_vertices,
            1.5,
            mag_hamiltonian_3halves_no_vertices,
        ),
    ]

    # Iterate over all test cases.
    for (
        expected_register_names_ordered,
        link_bitmap,
        vertex_bitmap,
        dims,
        hamiltonian,
    ) in test_cases:
        print(
            f"Checking register order in a circuit constructed from a {dims}-dimensional lattice."
        )
        print(f"Link bitmap: {link_bitmap}\nVertex bitmap: {vertex_bitmap}")
        print(f"Expected register ordering: {expected_register_names_ordered}")

        # Create circuit.
        lattice = LatticeRegisters(
            dimensions=dims,
            size=2,
            link_truncation_dict=link_bitmap,
            vertex_singlet_dict=vertex_bitmap,
        )
        circ_mgr = LatticeCircuitManager(
            lattice_encoder=LatticeStateEncoder(link_bitmap, vertex_bitmap),
            mag_hamiltonian=hamiltonian,
        )
        master_circuit = circ_mgr.create_blank_full_lattice_circuit(lattice)
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

    print("Register order tests passed.")
