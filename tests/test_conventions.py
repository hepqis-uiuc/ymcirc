import pytest
from ymcirc._abstract import LatticeDef
from ymcirc.conventions import (
    PHYSICAL_PLAQUETTE_STATES, IRREP_TRUNCATION_DICT_1_3_3BAR,
    IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8, ONE, THREE,
    THREE_BAR, SIX, SIX_BAR, EIGHT,
    LatticeStateEncoder, HAMILTONIAN_BOX_TERMS
)


def test_no_duplicate_physical_plaquette_states():
    print("Checking that none of the physical plaquette states data contain duplicates.")
    for dim_trunc_case in PHYSICAL_PLAQUETTE_STATES.keys():
        print(f"Checking {dim_trunc_case}...")
        num_duplicates = len(PHYSICAL_PLAQUETTE_STATES[dim_trunc_case]) \
            - len(set(PHYSICAL_PLAQUETTE_STATES[dim_trunc_case]))
        has_no_duplicates = num_duplicates == 0
        assert has_no_duplicates, f"Detected {num_duplicates} duplicate entries."


def test_no_duplicate_matrix_elements():
    print("Checking that none of the matrix element data contain duplicates.")
    for dim_trunc_case in HAMILTONIAN_BOX_TERMS.keys():
        print(f"Checking {dim_trunc_case}...")
        # list of tuples (final state, initial state) that index matrix elements.
        state_indices = list(HAMILTONIAN_BOX_TERMS[dim_trunc_case].keys())
        num_duplicates = len(state_indices) - len(set(state_indices))
        has_no_duplicates = num_duplicates == 0
        assert has_no_duplicates, f"Detected {num_duplicates} duplicate entries."


def test_physical_plaquette_state_data_are_valid():
    print("Checking that physical state data are valid.")
    expected_num_vertices = 4
    expected_num_a_links = 4
    expected_iweight_length = 3
    for dim_trunc_case in PHYSICAL_PLAQUETTE_STATES.keys():
        print(f"Case: {dim_trunc_case}")
        dim, trunc = dim_trunc_case.split(",")
        trunc = trunc.strip()
        # Figure out number of control links based on dimension.
        match dim:
            case "d=3/2":
                expected_num_c_links = 4
            case "d=2":
                expected_num_c_links = 8
            case _:
                raise NotImplementedError(f"Test not implemented for dimension {dim}.")
        for state in PHYSICAL_PLAQUETTE_STATES[dim_trunc_case]:
            assert len(state) == 3, \
                "States should be length-3 tuples of tuples (vertices, a-links, c-links)." \
                f" Encountered state: {state}"
            vertices = state[0]
            a_links = state[1]
            c_links = state[2]

            vertices_are_valid = len(vertices) == expected_num_vertices \
                and all(isinstance(vertex, int) for vertex in vertices)
            assert vertices_are_valid is True, f"Encountered state with invalid vertices: {vertices}."

            a_links_are_valid = len(a_links) == expected_num_a_links \
                and all(isinstance(a_link, tuple) and len(a_link) == expected_iweight_length for a_link in a_links) \
                and all(isinstance(iweight_elem, int) for a_link in a_links for iweight_elem in a_link)
            assert a_links_are_valid is True, f"Encountered state with invalid active links: {a_links}."

            c_links_are_valid = len(c_links) == expected_num_c_links \
                and all(isinstance(c_link, tuple) and len(c_link) == expected_iweight_length for c_link in c_links) \
                and all(isinstance(iweight_elem, int) for c_link in c_links for iweight_elem in c_link)
            assert c_links_are_valid is True, f"Encountered state with invalid control links: {c_links}."


# TODO: handle slow tests differently.
def test_matrix_element_data_are_valid():
    print("Checking that matrix element data are valid. WARNING: this can be slow.")
    for dim_trunc_case in HAMILTONIAN_BOX_TERMS.keys():
        print(f"Case: {dim_trunc_case}")
        dim, trunc = dim_trunc_case.split(",")
        trunc = trunc.strip()
        current_iter = 0
        for (state_f, state_i), mat_elem_val in HAMILTONIAN_BOX_TERMS[dim_trunc_case].items():
            current_iter += 1
            percent_done = current_iter/len(HAMILTONIAN_BOX_TERMS[dim_trunc_case])
            # These cases are very slow because the data are large.
            if dim_trunc_case == "d=3/2, T2" or dim_trunc_case == "d=2, T1":
                print("Skipping slow test.")
                break
            print(f"Current status: {percent_done:.4%}", end='\r')
            assert state_f in PHYSICAL_PLAQUETTE_STATES[dim_trunc_case], "Encountered state not in physical" \
                f" plaquette state list: {state_f}."
            assert state_i in PHYSICAL_PLAQUETTE_STATES[dim_trunc_case], "Encountered state not in physical" \
                f" plaquette state list: {state_i}."
            assert isinstance(mat_elem_val, (float, int)), f"Non-numeric matrix element: {mat_elem_val}."

        if dim_trunc_case != "d=3/2, T2" and dim_trunc_case != "d=2, T1":
            print("\nTest passed.")  # TODO this isn't the right way to handle this case.


def test_lattice_encoder_type_error_for_bad_lattice_arg():
    print("Check that LatticeStateEncoder raises a TypeError if lattice isn't a LatticeDef.")
    link_bitmap: IrrepBitmap = {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "01"
    }
    physical_states: List[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE_BAR, THREE, THREE_BAR),
            (THREE, ONE, ONE, ONE)
        )
    ]
    bad_lattice_arg = None
    with pytest.raises(TypeError) as e_info:
        LatticeStateEncoder(
            link_bitmap=link_bitmap,
            physical_plaquette_states=physical_states,
            lattice=bad_lattice_arg)


def test_lattice_encoder_fails_if_plaquette_states_have_wrong_number_of_controls():
    print(
        "Raise a ValueError if the number of controls in the plaquette state data"
        " is inconsistent with the lattice geometry."
    )
    link_bitmap: IrrepBitmap = {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "01"
    }
    physical_states: List[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE_BAR, THREE, THREE_BAR),
            (THREE, ONE, ONE, ONE)
        )
    ]
    lattice_d_2 = LatticeDef(2, 3)
    assert lattice_d_2.n_control_links_per_plaquette == 8

    with pytest.raises(ValueError) as e_info:
        LatticeStateEncoder(link_bitmap, physical_states, lattice_d_2)


def test_lattice_encoder_infers_correct_vertex_bitmaps():
    print(
        "Check that LatticeStateEncoder correctly infers (1) "
        "the number of bits needed to encode vertices and (2) the right bitmaps."
    )

    # Case 1, d=3/2
    lattice_d_3_2 = LatticeDef(1.5, 3)
    good_link_bitmap: IrrepBitmap = {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "01"
    }
    good_physical_states: List[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE)
        ),
        (
            (0, 0, 0, 1),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        )
    ]
    expected_num_vertex_bits = 1
    expected_vertex_bitmap = {
        0: "0",
        1: "1"
    }
    plaquette_states_str_rep = "\n\t".join(f"{item}" for item in good_physical_states)
    print("\nCase:\n"
          f"link bitmap = {good_link_bitmap}\n"
          f"plaquette states =\n\t{plaquette_states_str_rep}\n"
          f"expected num vertex bits = {expected_num_vertex_bits}\n"
          f"expected vertex bitmap = {expected_vertex_bitmap}\n")
    lattice_encoder = LatticeStateEncoder(
        link_bitmap=good_link_bitmap,
        physical_plaquette_states=good_physical_states,
        lattice=lattice_d_3_2
    )

    assert lattice_encoder.expected_vertex_bit_string_length == expected_num_vertex_bits, f"Actual num vertex bits = {lattice_encoder.expected_vertex_bit_string_length}"
    assert lattice_encoder.vertex_bitmap == expected_vertex_bitmap, f"Actual vertex bitmap = {lattice_encoder.vertex_bitmap}"

    # Case 2, d=2
    lattice_d_2 = LatticeDef(2, 2)
    good_link_bitmap: IrrepBitmap = {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "01"
    }
    good_physical_states: List[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 1, 1),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 2, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        )
    ]
    expected_num_vertex_bits = 2
    expected_vertex_bitmap = {
        0: "00",
        1: "01",
        2: "10"
    }
    plaquette_states_str_rep = "\n\t".join(f"{item}" for item in good_physical_states)
    print("\nCase:\n"
          f"link bitmap = {good_link_bitmap}\n"
          f"plaquette states =\n\t{plaquette_states_str_rep}\n"
          f"expected num vertex bits = {expected_num_vertex_bits}\n"
          f"expected vertex bitmap = {expected_vertex_bitmap}\n")
    lattice_encoder = LatticeStateEncoder(
        link_bitmap=good_link_bitmap,
        physical_plaquette_states=good_physical_states,
        lattice=lattice_d_2
    )

    assert lattice_encoder.expected_vertex_bit_string_length == expected_num_vertex_bits, f"Actual num vertex bits = {lattice_encoder.expected_vertex_bit_string_length}"
    assert lattice_encoder.vertex_bitmap == expected_vertex_bitmap, f"Actual vertex bitmap = {lattice_encoder.vertex_bitmap}"

    # Case 3, d=2
    good_link_bitmap: IrrepBitmap = {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "01"
    }
    good_physical_states: List[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 1, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 2, 1),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 3, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 4, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 5, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 6, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 7, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 8, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        )
    ]
    expected_num_vertex_bits = 4
    expected_vertex_bitmap = {
        0: "0000",
        1: "0001",
        2: "0010",
        3: "0011",
        4: "0100",
        5: "0101",
        6: "0110",
        7: "0111",
        8: "1000"
    }
    plaquette_states_str_rep = "\n\t".join(f"{item}" for item in good_physical_states)
    print("\nCase:\n"
          f"link bitmap = {good_link_bitmap}\n"
          f"plaquette states =\n\t{plaquette_states_str_rep}\n"
          f"expected num vertex bits = {expected_num_vertex_bits}\n"
          f"expected vertex bitmap = {expected_vertex_bitmap}\n")
    lattice_encoder = LatticeStateEncoder(
        link_bitmap=good_link_bitmap,
        physical_plaquette_states=good_physical_states,
        lattice=lattice_d_2
    )

    assert lattice_encoder.expected_vertex_bit_string_length == expected_num_vertex_bits, f"Actual num vertex bits = {lattice_encoder.expected_vertex_bit_string_length}"
    assert lattice_encoder.vertex_bitmap == expected_vertex_bitmap, f"Actual vertex bitmap = {lattice_encoder.vertex_bitmap}"

    # Case 4, d=3/2,
    # since all vertex singlets are trivial.
    good_link_bitmap: IrrepBitmap = {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "01"
    }
    good_physical_states: List[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE_BAR, THREE, THREE_BAR),
            (THREE, ONE, ONE, ONE)
        )
    ]
    expected_num_vertex_bits = 0
    expected_vertex_bitmap = {}
    plaquette_states_str_rep = "\n\t".join(f"{item}" for item in good_physical_states)
    print("\nCase:\n"
          f"link bitmap = {good_link_bitmap}\n"
          f"plaquette states =\n\t{plaquette_states_str_rep}\n"
          f"expected num vertex bits = {expected_num_vertex_bits}\n"
          f"expected vertex bitmap = {expected_vertex_bitmap}\n")
    lattice_encoder = LatticeStateEncoder(
        link_bitmap=good_link_bitmap,
        physical_plaquette_states=good_physical_states,
        lattice=lattice_d_3_2
    )

    assert lattice_encoder.expected_vertex_bit_string_length == expected_num_vertex_bits, f"Actual num vertex bits = {lattice_encoder.expected_vertex_bit_string_length}"
    assert lattice_encoder.vertex_bitmap == expected_vertex_bitmap, f"Actual vertex bitmap = {lattice_encoder.vertex_bitmap}"


def test_lattice_encoder_infers_correct_plaquette_length():
    print(
        "Check that LatticeStateEncoder correctly infers the number of qubits per plaquette."
    )
    expected_num_controls_dict = {
        "d=3/2": 1*4,
        "d=2": 2*4,
        "d=3": 4*4
    }
    # Case: d=3/2, no trivial multiplicites
    link_bitmap: IrrepBitmap = {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "01"
    }
    physical_states: List[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE_BAR, THREE, THREE_BAR),
            (THREE, ONE, ONE, ONE)
        )
    ]
    lattice_d_3_2 = LatticeDef(1.5, 3)
    # N_qubits_per_link * (N_control links + N active links) + 4*N_qubits_per_vertex
    expected_qubits_per_plaquette = (2 * (4 + 4)) + (4 * 0)

    plaquette_states_str_rep = "\n\t".join(f"{item}" for item in physical_states)
    print("\nCase:\n"
          f"link bitmap = {link_bitmap}\n"
          f"plaquette states =\n\t{plaquette_states_str_rep}\n"
          f"expected num plaquette qubits = {expected_qubits_per_plaquette}\n")
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_states, lattice_d_3_2)
    assert lattice_encoder.expected_plaquette_bit_string_length == expected_qubits_per_plaquette, \
        f"Wrong actual plaquette qubit count inferred: {lattice_encoder.expected_plaquette_bit_string_length}"

    # Case: d=3/2, nontrivial multiplicities
    physical_states = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 1),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE)
        ),
        (
            (0, 1, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE)
        ),
        (
            (0, 2, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE_BAR, THREE, THREE_BAR),
            (THREE, ONE, ONE, ONE)
        )
    ]
    # N_qubits_per_link * (N_control links + N active links) + 4*N_qubits_per_vertex
    expected_qubits_per_plaquette = (2 * (4 + 4)) + (4 * 2)

    plaquette_states_str_rep = "\n\t".join(f"{item}" for item in physical_states)
    print("\nCase:\n"
          f"link bitmap = {link_bitmap}\n"
          f"plaquette states =\n\t{plaquette_states_str_rep}\n"
          f"expected num plaquette qubits = {expected_qubits_per_plaquette}\n")
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_states, lattice_d_3_2)
    assert lattice_encoder.expected_plaquette_bit_string_length == expected_qubits_per_plaquette, \
        f"Wrong actual plaquette qubit count inferred: {lattice_encoder.expected_plaquette_bit_string_length}"

    # Case: d=2, nontrivial multiplicites
    physical_states = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 1),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 2),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 3),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 4),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 1, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE, THREE_BAR, ONE, ONE, ONE)
        ),
        (
            (0, 2, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE, THREE_BAR, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE_BAR, THREE, THREE_BAR),
            (ONE, THREE, ONE, ONE, THREE_BAR, ONE, ONE, ONE)
        )
    ]
    lattice_d_2 = LatticeDef(2, 3)
    # N_qubits_per_link * (N_control links + N active links) + 4*N_qubits_per_vertex
    expected_qubits_per_plaquette = (2 * (8 + 4)) + (4 * 3)

    plaquette_states_str_rep = "\n\t".join(f"{item}" for item in physical_states)
    print("\nCase:\n"
          f"link bitmap = {link_bitmap}\n"
          f"plaquette states =\n\t{plaquette_states_str_rep}\n"
          f"expected num plaquette qubits = {expected_qubits_per_plaquette}\n")
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_states, lattice_d_2)
    assert lattice_encoder.expected_plaquette_bit_string_length == expected_qubits_per_plaquette, \
        f"Wrong actual plaquette qubit count inferred: {lattice_encoder.expected_plaquette_bit_string_length}"

    # Case: d=3, nontrivial multiplicities
    physical_states = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE,) * 16
        ),
        (
            (0, 0, 0, 1),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE,) * 16
        ),
        (
            (0, 0, 0, 2),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE,) * 16
        ),
        (
            (0, 0, 0, 3),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE,) * 16
        ),
        (
            (0, 0, 0, 4),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE,) * 16
        ),
        (
            (0, 1, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE,) * 16
        ),
        (
            (0, 2, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE,) * 16
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE_BAR, THREE, THREE_BAR),
            (ONE,) * 16
        )
    ]
    lattice_d_3 = LatticeDef(3, 3)
    # N_qubits_per_link * (N_control links + N active links) + 4*N_qubits_per_vertex
    expected_qubits_per_plaquette = (2 * (16 + 4)) + (4 * 3)

    plaquette_states_str_rep = "\n\t".join(f"{item}" for item in physical_states)
    print("\nCase:\n"
          f"link bitmap = {link_bitmap}\n"
          f"plaquette states =\n\t{plaquette_states_str_rep}\n"
          f"expected num plaquette qubits = {expected_qubits_per_plaquette}\n")
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_states, lattice_d_3)
    assert lattice_encoder.expected_plaquette_bit_string_length == expected_qubits_per_plaquette, \
        f"Wrong actual plaquette qubit count inferred: {lattice_encoder.expected_plaquette_bit_string_length}"


def test_lattice_encoder_fails_on_bad_creation_args():
    print("Checking that plaquette states list with differing lengths of controls causes ValueError.")
    good_link_bitmap: IrrepBitmap = {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "01"
    }
    physical_states_inconsistent_control_lengths =  [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, THREE)
        ),
        (
            (0, 0, 0, 1),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, THREE, THREE, THREE, ONE, THREE_BAR)
        )
    ]
    lattice_d_3_2 = LatticeDef(1.5, 2)
    with pytest.raises(ValueError) as e_info:
        LatticeStateEncoder(good_link_bitmap, physical_states_inconsistent_control_lengths, lattice_d_3_2)


    print("Checking that repeated physical plaquette states cause a ValueError.")
    non_unique_physical_states: List[PlaquetteState] = [
        (
            (1, 1, 1, 1),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (1, 1, 1, 1),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE)
        ),
        (
            (1, 1, 1, 1),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        )
    ]
    with pytest.raises(ValueError) as e_info:
        LatticeStateEncoder(
            link_bitmap=good_link_bitmap,
            physical_plaquette_states=non_unique_physical_states,
            lattice=lattice_d_3_2
        )

    print("Checking that a link bitmap with non-unique bit string values causes ValueError.")
    non_unique_link_bitmap: IrrepBitmap = {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "10"
    }
    good_physical_states: List[PlaquetteState] = [
        (
            (1, 1, 1, 1),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (1, 1, 1, 1),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE)
        ),
        (
            (1, 1, 1, 2),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        )
    ]
    with pytest.raises(ValueError) as e_info:
        LatticeStateEncoder(non_unique_link_bitmap, good_physical_states, lattice_d_3_2)

    print("Checking that a link bitmap with bit strings of different lengths causes ValueError.")
    link_bitmap_different_string_lengths = {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "010"
    }
    with pytest.raises(ValueError) as e_info:
        LatticeStateEncoder(link_bitmap_different_string_lengths, good_physical_states, lattice_d_3_2)


def test_encode_decode_various_links():
    lattice_d_3_2 = LatticeDef("3/2", 3)
    link_bitmap = {
        ONE: "00",
        THREE: "10",
        EIGHT: "11"
    }
    # Test data, not physically meaningful but has right format for creating creating an encoder.
    physical_plaquette_states = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE)
        )
    ]
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice_d_3_2)

    print("Checking that the following link_bitmap is used to correctly encode/decode links:")
    print(link_bitmap)
    for link_state, bit_string_encoding in link_bitmap.items():
        result_encoding = lattice_encoder.encode_link_state_as_bit_string(link_state)
        result_decoding = lattice_encoder.decode_bit_string_to_link_state(bit_string_encoding)
        assert result_encoding == bit_string_encoding, f"(result != expected): {result_encoding} != {bit_string_encoding}"
        assert result_decoding == link_state, f"(result != expected): {result_decoding} != {link_state}"
        print(f"Validated {bit_string_encoding} <-> {link_state}")

    print("Verifying that unknown bit string is decoded to None.")
    assert lattice_encoder.decode_bit_string_to_link_state("01") is None


def test_encode_decode_various_vertices():
    lattice_d_3_2 = LatticeDef("3/2", 3)
    link_bitmap = {
        ONE: "00",
        THREE: "10",
        EIGHT: "11"
    }
    # Should generate a bitmap to length-2 bitstrings for vertices.
    physical_plaquette_states = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 1, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 2, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE)
        )
    ]
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice_d_3_2)
    expected_vertex_bitmap = {
        0: "00",
        1: "01",
        2: "10"
    }

    print(f"Checking that the following vertex_bitmap is generated for encoding/decoding vertices:\n{expected_vertex_bitmap}")

    for multiplicity_index, bit_string_encoding in expected_vertex_bitmap.items():
        result_encoding = lattice_encoder.encode_vertex_state_as_bit_string(multiplicity_index)
        result_decoding = lattice_encoder.decode_bit_string_to_vertex_state(bit_string_encoding)
        assert result_encoding == bit_string_encoding, f"(result != expected): {result_encoding} != {bit_string_encoding}"
        assert result_decoding == multiplicity_index, f"(result != expected): {result_decoding} != {multiplicity_index}"
        print(f"Validated {bit_string_encoding} <-> {multiplicity_index}")

    print("Verifying that unknown bit string is decoded to None.")
    assert lattice_encoder.decode_bit_string_to_vertex_state("11") is None


def test_encoding_malformed_plaquette_fails():
    lattice_d_3_2 = LatticeDef(1.5, 4)
    lattice_encoder = LatticeStateEncoder(
        link_bitmap=IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8,
        physical_plaquette_states=PHYSICAL_PLAQUETTE_STATES["d=3/2, T2"],
        lattice=lattice_d_3_2
    )

    print("Checking that encoding a wrong-length plaquette fails.")
    plaquette_wrong_length: PlaquetteState = (
        (0, 0, 0, 0),
        (ONE, ONE, ONE, ONE),
        (ONE, ONE, ONE, ONE),
        (ONE, ONE, ONE, SIX)
    )
    with pytest.raises(ValueError) as e_info:
        lattice_encoder.encode_plaquette_state_as_bit_string(plaquette_wrong_length)

    print("Checking that encoding a plaquette with the wrong data types for vertices, active links, or control links fails.")
    plaquette_bad_vertex_data: PlaquetteState = (
        (0, 0, "zero", 0),
        (ONE, ONE, ONE, ONE),
        (ONE, ONE, ONE, ONE),
    )
    plaquette_bad_a_link_data: PlaquetteState = (
        (0, 0, 0, 0),
        (ONE, ONE, (0, 0), ONE),
        (ONE, ONE, ONE, ONE),
    )
    plaquette_bad_c_link_data: PlaquetteState = (
        (0, 0, 0, 0),
        (ONE, ONE, ONE, ONE),
        (ONE, ONE, (0, 0), ONE),
    )
    cases = {
        "Vertex test": plaquette_bad_vertex_data,
        "Active link test": plaquette_bad_a_link_data,
        "Control link test": plaquette_bad_c_link_data
    }
    for test_name, test_data in cases.items():
        with pytest.raises(ValueError) as e_info:
            lattice_encoder.encode_plaquette_state_as_bit_string(test_data)

    print("Checking that encoding a plaquette which has too many vertices, active links, or control links fails.")
    plaquette_too_many_vertices: PlaquetteState = (
        (0, 0, 0, 0, 1),
        (ONE, ONE, ONE, ONE),
        (SIX, SIX, SIX, SIX_BAR)
    )
    plaquette_too_many_a_links: PlaquetteState = (
        (0, 0, 0, 1),
        (ONE, ONE, ONE, ONE, THREE),
        (SIX, SIX, SIX, SIX_BAR)
    )
    plaquette_too_many_c_links: PlaquetteState = (
        (0, 0, 1, 0),
        (ONE, ONE, ONE, ONE),
        (SIX, SIX_BAR, EIGHT)
    )
    cases = {
        "Vertex test": plaquette_too_many_vertices,
        "Active link test": plaquette_too_many_a_links,
        "Control link test": plaquette_too_many_c_links
    }
    for test_name, test_data in cases.items():
        with pytest.raises(ValueError) as e_info:
            lattice_encoder.encode_plaquette_state_as_bit_string(test_data)


def test_encoding_good_plaquette():
    print("Check that the encoding some known plaquette yields the expected results.")

    # Set up needed encoders. TODO implement d=3 test.
    # d=3/2 T2 and d=2 T1 should have a single bit for multiplicity encoding.
    lattice_d_3_2 = LatticeDef(1.5, 3)
    lattice_d_2 = LatticeDef(2, 2)
    l_encoder_d_3_2_T1 = LatticeStateEncoder(IRREP_TRUNCATION_DICT_1_3_3BAR, PHYSICAL_PLAQUETTE_STATES["d=3/2, T1"], lattice_d_3_2)
    l_encoder_d_3_2_T2 = LatticeStateEncoder(IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8, PHYSICAL_PLAQUETTE_STATES["d=3/2, T2"], lattice_d_3_2)
    l_encoder_d_2_T1 = LatticeStateEncoder(IRREP_TRUNCATION_DICT_1_3_3BAR, PHYSICAL_PLAQUETTE_STATES["d=2, T1"], lattice_d_2)

    # Construct test data.
    plaquette_d_3_2_T1 = (
        (0, 0, 0, 0),
        (ONE, THREE, THREE_BAR, ONE),
        (ONE, ONE, ONE, THREE_BAR)
    )
    expected_plaquette_d_3_2_T1_bit_string = "00100100" + "00000001" # a links + c links
    plaquette_d_3_2_T2 = (
        (0, 1, 0, 0),
        (SIX, EIGHT, THREE_BAR, ONE),
        (ONE, ONE, SIX_BAR, THREE_BAR)
    )
    expected_plaquette_d_3_2_T2_bit_string = "0100" + "110111001000" + "000000011001" # v + a links + c links
    plaquette_d_2_T1 = (
        (0, 0, 1, 0),
        (THREE, THREE, THREE_BAR, THREE_BAR),
        (ONE, ONE, THREE_BAR, THREE_BAR, THREE, THREE, THREE, ONE)
    )
    expected_plaquette_d_2_T1_bit_string = "0010" + "10100101" + "0000010110101000" # v + a links + c links
    cases = [
        (l_encoder_d_3_2_T1, plaquette_d_3_2_T1, expected_plaquette_d_3_2_T1_bit_string),
        (l_encoder_d_3_2_T2, plaquette_d_3_2_T2, expected_plaquette_d_3_2_T2_bit_string),
        (l_encoder_d_2_T1, plaquette_d_2_T1, expected_plaquette_d_2_T1_bit_string)
    ]
    for lattice_encoder, input_plaquette_state, expected_bit_string_encoding in cases:
        print(f"Encoding state {input_plaquette_state}.")
        actual_bit_string_encoding = lattice_encoder.encode_plaquette_state_as_bit_string(input_plaquette_state)
        assert actual_bit_string_encoding == expected_bit_string_encoding, "Encoding error. " \
            f"The state {input_plaquette_state} should encode as {expected_bit_string_encoding}. " \
            f"Instead obtained: {actual_bit_string_encoding}."


# This test is a little bit slow.
def test_all_mag_hamiltonian_plaquette_states_have_unique_bit_string_encoding():
    """
    Check that all plaquette states have unique bitstring encodings.

    Attempts encoding the following cases:
    - d=3/2, T1
    - d=3/2, T1p
    - d=3/2, T2
    - d=2, T1
    """
    # dim_trunc_str, expected_plaquette_bit_length, LatticeDef
    # Each expected_bitlength = 4 * (2 * (dim - 1) + 1) * n_link_qubits + 4 * n_vertex_qubits for the corresponding case. This is just (n_control_links + n_active_links) * n_link_qubits + n_vertices * n_vertex_qubits.
    cases = [
        ("d=3/2, T1", (4 * (2 * (3/2 - 1) + 1) * 2) + 4*0, LatticeDef(1.5, 3)),
        ("d=3/2, T2", (4 * (2 * (3/2 - 1) + 1) * 3) + 4*1, LatticeDef(1.5, 3)),
        ("d=2, T1", (4 * (2 * (2 - 1) + 1) * 2) + 4*1, LatticeDef(2, 3))
    ]
    print(
        "Checking that there is a unique bit string encoding available for all "
        "the plaquette states appearing in all the matrix elements for the "
        f"following cases:\n{cases}."
    )
    for current_dim_trunc_str, current_expected_bitlength, current_lattice in cases:
        dim_str, trunc_str = current_dim_trunc_str.split(",")
        dim_str = dim_str.strip()
        trunc_str = trunc_str.strip()
        # Make encoder instance.
        link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR if \
                trunc_str == "T1" else IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8
        lattice_encoder = LatticeStateEncoder(
            link_bitmap, PHYSICAL_PLAQUETTE_STATES[current_dim_trunc_str], lattice=current_lattice)

        print(f"Case {current_dim_trunc_str}.\nConfirming all initial and final states "
              "appearing in the physical plaquette states list can be succesfully encoded. Using the bitmaps:\n"
              f"Link bitmap =  {lattice_encoder.link_bitmap}\n"
              f"Vertex bitmap = {lattice_encoder.vertex_bitmap}")

        # Get the set of unique plaquette states.
        all_plaquette_states = set([
            final_and_initial_state_tuple[0] for final_and_initial_state_tuple in HAMILTONIAN_BOX_TERMS[current_dim_trunc_str].keys()] + [
                final_and_initial_state_tuple[1] for final_and_initial_state_tuple in HAMILTONIAN_BOX_TERMS[current_dim_trunc_str].keys()
            ])

        # Attempt encodings and check for uniqueness.
        all_encoded_plaquette_bit_strings = []
        for plaquette_state in all_plaquette_states:
            plaquette_state_bit_string = lattice_encoder.encode_plaquette_state_as_bit_string(plaquette_state)
            assert len(plaquette_state_bit_string) == current_expected_bitlength, f"len(plaquette_state_bit_string) == {len(plaquette_state_bit_string)}; expected len == {current_expected_bitlength}."
            all_encoded_plaquette_bit_strings.append(plaquette_state_bit_string)
        n_unique_plaquette_encodings = len(set(all_encoded_plaquette_bit_strings))
        assert n_unique_plaquette_encodings == len(all_plaquette_states), f"Encountered {n_unique_plaquette_encodings} unique bit strings encoding {len(all_plaquette_states)} unique plaquette states."


def test_bit_string_decoding_to_plaquette():
    # Check that decoding of bit strings is as expected.
    # Case data tuple format:
    # case_name, encoded_plaquette, vertex_bitmap, link_bitmap, expected_decoded_plaquette.
    # Note that the data were manually constructed by irrep encoding bitmaps
    # with data in vertex singlet json files.
    cases = [
        (
            "d=3/2, T1",
            "10101001" + "00001110", # active links + control links (one of which is in a garbage state)
            LatticeDef(3/2, 3),
            IRREP_TRUNCATION_DICT_1_3_3BAR,
            (
                (None, None, None, None),  # When no vertex bitmap needed, should get back None for decoded vertices.
                (THREE, THREE, THREE, THREE_BAR),
                (ONE, ONE, None, THREE)  # Garbage control link should decode to None
            )
        ),
        (
            "d=3/2, T2",
            "0001" + "110111000001" + "000000111011",  # vertex multiplicities + active links + control links
            LatticeDef(3/2, 3),
            IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8,
            (
                (0, 0, 0, 1),
                (SIX, EIGHT, ONE, THREE_BAR),
                (ONE, ONE, EIGHT, SIX_BAR)
            )
        ),
        (
            "d=2, T1",
            "1011" + "00000010" + "0101011010000100",  # vertex multiplicities + active links + control links
            LatticeDef(2, 2),
            IRREP_TRUNCATION_DICT_1_3_3BAR,
            (
                (1, 0, 1, 1),
                (ONE, ONE, ONE, THREE),
                (THREE_BAR, THREE_BAR, THREE_BAR, THREE, THREE, ONE, THREE_BAR, ONE)
            )
        )
    ]

    print("Checking decoding of bit strings corresponding to gauge-invariant plaquette states.")

    for current_dim_trunc_str, encoded_plaquette, current_lattice, link_bitmap, expected_decoded_plaquette in cases:
        print(f"Checking plaquette bit string decoding for a {current_dim_trunc_str} plaquette...")
        lattice_encoder = LatticeStateEncoder(
            link_bitmap,
            PHYSICAL_PLAQUETTE_STATES[current_dim_trunc_str],
            current_lattice
        )
        resulting_decoded_plaquette = lattice_encoder.decode_bit_string_to_plaquette_state(encoded_plaquette)
        assert resulting_decoded_plaquette == expected_decoded_plaquette, f"Expected: {expected_decoded_plaquette}\nEncountered: {resulting_decoded_plaquette}"
        print(f"\n{encoded_plaquette} successfully decoded to {resulting_decoded_plaquette}.")


def test_decoding_garbage_bit_strings_result_in_none():
    print("Checking that various garbage bit string correctly decode to 'None'.")
    # This should map to something since it's possible for noise to result in such states.
    link_bitmap = {
        ONE: "000",
        THREE: "100",
        EIGHT: "101",
        SIX: "111",
        SIX_BAR: "010"
    }
    physical_states = [
        (
            (0, 0, 0, 0),
            (ONE, ONE, ONE, ONE),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 1, 0),
            (ONE, ONE, ONE, ONE),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 2, 0),
            (ONE, ONE, ONE, ONE),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (THREE, ONE, EIGHT, ONE),
            (ONE, SIX_BAR, ONE, ONE)
        )
    ]
    lattice = LatticeDef(1.5, 4)
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_states, lattice)

    # Link, vertex, and plaquette test data.
    bad_encoded_link = "001"
    bad_encoded_vertex = "11"   # Max multiplicity in test data is 2 -> 10 in binary.
    vertex_bitstring = "00" + "00" + bad_encoded_vertex + "10"
    a_link_bitstring = "000" + "000" + bad_encoded_link + "100"
    c_link_bitstring = bad_encoded_link + "000" + "000" + "010"
    encoded_plaquette_some_links_and_vertices_good_others_bad = \
        vertex_bitstring + a_link_bitstring + c_link_bitstring
    expected_decoded_plaquette_some_links_and_vertices_good_others_bad = (
        (0, 0, None, 2),
        (ONE, ONE, None, THREE),
        (None, ONE, ONE, SIX_BAR)
    )

    print(f"Checking {bad_encoded_link} decodes to None using the link bitmap: {link_bitmap}")
    assert lattice_encoder.decode_bit_string_to_link_state(bad_encoded_link) is None

    print(f"Checking {bad_encoded_vertex} decodes to None using the vertex bitmap: {lattice_encoder.vertex_bitmap}")
    assert lattice_encoder.decode_bit_string_to_vertex_state(bad_encoded_vertex) is None

    print(f"Checking {encoded_plaquette_some_links_and_vertices_good_others_bad} decodes to the plaquette:\n {expected_decoded_plaquette_some_links_and_vertices_good_others_bad}")
    decoded_plaquette = lattice_encoder.decode_bit_string_to_plaquette_state(encoded_plaquette_some_links_and_vertices_good_others_bad)
    assert decoded_plaquette == expected_decoded_plaquette_some_links_and_vertices_good_others_bad, f"(decoded != expected): {decoded_plaquette}\n!=\n{expected_decoded_plaquette_some_links_and_vertices_good_others_bad}"


def test_decoding_fails_when_len_bit_string_doesnt_match_bitmaps():
    # This should map to something since it's possible for noise to result in such states.
    link_bitmap = {
        ONE: "000",
        THREE: "100",
        EIGHT: "101",
        SIX: "111",
        SIX_BAR: "010"
    }
    # Test data, not physically meaningful but has right format for creating creating an encoder.
    physical_states = [
        (
            (0, 0, 0, 0),
            (ONE, ONE, ONE, ONE),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 1, 0),
            (ONE, ONE, ONE, ONE),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 2, 0),
            (ONE, ONE, ONE, ONE),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (THREE, ONE, EIGHT, ONE),
            (ONE, SIX_BAR, ONE, ONE, ONE, ONE, ONE, ONE)
        )
    ]
    lattice = LatticeDef(2, 2)
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_states, lattice)

    print("Testing that decoding links/vertices/plaquettes fails with wrong length bit string (ValueError).")
    print(f"Using link bitmap: {link_bitmap}")
    print(f"Using vertex bitmap: {lattice_encoder.vertex_bitmap}")

    # Set up test data.
    expected_link_bit_string_length = 3  # Based on test data.
    expected_vertex_bit_string_length = 2  # Based on test data.
    expected_plaquette_bit_string_length = 4 * expected_vertex_bit_string_length + 12 * expected_link_bit_string_length  # Should equal 44 in d = 2 with the above link encoding and physical states.
    bad_length_link_bit_string = "10"
    bad_length_vertex_bit_string = "101"
    bad_length_plaquette_bit_string = "000000010000100011101"
    assert len(bad_length_link_bit_string) != expected_link_bit_string_length
    assert len(bad_length_vertex_bit_string) != expected_vertex_bit_string_length
    assert len(bad_length_plaquette_bit_string) != expected_plaquette_bit_string_length

    print(f"Checking link bit string {bad_length_link_bit_string} fails to decode.")
    with pytest.raises(ValueError) as e_info:
        lattice_encoder.decode_bit_string_to_link_state(bad_length_link_bit_string)

    print(f"Checking vertex bit string {bad_length_vertex_bit_string} fails to decode.")
    with pytest.raises(ValueError) as e_info:
        lattice_encoder.decode_bit_string_to_vertex_state(bad_length_vertex_bit_string)

    print(f"Checking plaquette bit string {bad_length_plaquette_bit_string} fails to decode.")
    with pytest.raises(ValueError) as e_info:
        lattice_encoder.decode_bit_string_to_plaquette_state(bad_length_plaquette_bit_string)
