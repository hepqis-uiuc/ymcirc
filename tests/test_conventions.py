import pytest
from ymcirc._abstract import LatticeDef
from ymcirc.conventions import (
    ONE, THREE,
    THREE_BAR, SIX, SIX_BAR, EIGHT,
    LatticeStateEncoder, physical_plaquette_states,
    IrrepBitmap, PlaquetteState, FIFTEEN, FIFTEEN_BAR
)
from ymcirc._pyclebsch_adapter import (
    N_COLORS, box_terms, plaq_state_pyclebsch_to_ymcirc_format, plaq_state_ymcirc_to_pyclebsch_format
)
from pyclebsch.matrix_elements.lattice_data import (
    LatticeDef as PyclebschLatticeDef, physical_plaquette_states_by_signature
)

# The link bitmaps that ymcirc used for the truncations its data files
# covered, under the names "T1", "T2" and "B10_d=3/2" before the bitmaps were
# keyed by sets of irreps.
_T1_LINK_BITMAP = {ONE: "00", THREE: "10", THREE_BAR: "01"}
_T2_LINK_BITMAP = {ONE: "000", THREE: "100", THREE_BAR: "001", SIX: "110", SIX_BAR: "011", EIGHT: "111"}
_B10_LINK_BITMAP = _T2_LINK_BITMAP | {FIFTEEN: "101", FIFTEEN_BAR: "010"}


def test_lattice_encoder_type_error_for_bad_lattice_arg():
    print("Check that LatticeStateEncoder raises a TypeError if lattice isn't a LatticeDef.")
    link_bitmap: IrrepBitmap = {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "01"
    }
    physical_states: list[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            ((ONE,), (THREE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE_BAR, THREE, THREE_BAR),
            ((THREE,), (ONE,), (ONE,), (ONE,))
        )
    ]
    bad_lattice_arg = None
    with pytest.raises(TypeError):
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
    physical_states: list[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            ((ONE,), (THREE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE_BAR, THREE, THREE_BAR),
            ((THREE,), (ONE,), (ONE,), (ONE,))
        )
    ]
    lattice_d_2 = LatticeDef(2, 3)
    assert lattice_d_2.n_control_links_per_plaquette == 8

    with pytest.raises(ValueError):
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
    good_physical_states: list[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            ((ONE,), (THREE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 1),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
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
    good_physical_states: list[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            ((ONE, THREE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 1, 1),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 2, 0),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
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
    good_physical_states: list[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 1, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            ((ONE, THREE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 2, 1),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 3, 0),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 4, 0),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 5, 0),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 6, 0),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 7, 0),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 8, 0),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
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
    good_physical_states: list[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            ((ONE,), (THREE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE_BAR, THREE, THREE_BAR),
            ((THREE,), (ONE,), (ONE,), (ONE,))
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
    physical_states: list[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            ((ONE,), (THREE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE_BAR, THREE, THREE_BAR),
            ((THREE,), (ONE,), (ONE,), (ONE,))
        )
    ]
    lattice_d_3_2 = LatticeDef(1.5, 3)
    # N_qubits_per_link * (N_control links + N active links) + 4*N_qubits_per_vertex
    expected_qubits_per_plaquette = (2 * (expected_num_controls_dict["d=3/2"] + 4)) + (4 * 0)

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
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 1),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            ((ONE,), (THREE,), (ONE,), (ONE,))
        ),
        (
            (0, 1, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            ((ONE,), (THREE,), (ONE,), (ONE,))
        ),
        (
            (0, 2, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            ((ONE,), (THREE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE_BAR, THREE, THREE_BAR),
            ((THREE,), (ONE,), (ONE,), (ONE,))
        )
    ]
    # N_qubits_per_link * (N_control links + N active links) + 4*N_qubits_per_vertex
    expected_qubits_per_plaquette = (2 * (expected_num_controls_dict["d=3/2"] + 4)) + (4 * 2)

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
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 0, 1),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 0, 2),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 0, 3),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 0, 4),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 1, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            ((ONE, THREE), (ONE, ONE), (THREE_BAR, ONE), (ONE, ONE))
        ),
        (
            (0, 2, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            ((ONE, THREE), (ONE, ONE), (THREE_BAR, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE_BAR, THREE, THREE_BAR),
            ((ONE, THREE), (ONE, ONE), (THREE_BAR, ONE), (ONE, ONE))
        )
    ]
    lattice_d_2 = LatticeDef(2, 3)
    # N_qubits_per_link * (N_control links + N active links) + 4*N_qubits_per_vertex
    expected_qubits_per_plaquette = (2 * (expected_num_controls_dict["d=2"] + 4)) + (4 * 3)

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
            ((ONE,) * 4, (ONE,) * 4, (ONE,) * 4, (ONE,) * 4)
        ),
        (
            (0, 0, 0, 1),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE,) * 4, (ONE,) * 4, (ONE,) * 4, (ONE,) * 4)
        ),
        (
            (0, 0, 0, 2),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE,) * 4, (ONE,) * 4, (ONE,) * 4, (ONE,) * 4)
        ),
        (
            (0, 0, 0, 3),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE,) * 4, (ONE,) * 4, (ONE,) * 4, (ONE,) * 4)
        ),
        (
            (0, 0, 0, 4),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE,) * 4, (ONE,) * 4, (ONE,) * 4, (ONE,) * 4)
        ),
        (
            (0, 1, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            ((ONE,) * 4, (ONE,) * 4, (ONE,) * 4, (ONE,) * 4)
        ),
        (
            (0, 2, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            ((ONE,) * 4, (ONE,) * 4, (ONE,) * 4, (ONE,) * 4)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE_BAR, THREE, THREE_BAR),
            ((ONE,) * 4, (ONE,) * 4, (ONE,) * 4, (ONE,) * 4)
        )
    ]
    lattice_d_3 = LatticeDef(3, 3)
    # N_qubits_per_link * (N_control links + N active links) + 4*N_qubits_per_vertex
    expected_qubits_per_plaquette = (2 * (expected_num_controls_dict["d=3"] + 4)) + (4 * 3)

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
            ((ONE,), (ONE,), (ONE,), (THREE,))
        ),
        (
            (0, 0, 0, 1),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE, ONE), (ONE, THREE), (THREE, THREE), (ONE, THREE_BAR))
        )
    ]
    lattice_d_3_2 = LatticeDef(1.5, 2)
    with pytest.raises(ValueError):
        LatticeStateEncoder(good_link_bitmap, physical_states_inconsistent_control_lengths, lattice_d_3_2)


    print("Checking that repeated physical plaquette states cause a ValueError.")
    non_unique_physical_states: list[PlaquetteState] = [
        (
            (1, 1, 1, 1),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (1, 1, 1, 1),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            ((ONE,), (THREE,), (ONE,), (ONE,))
        ),
        (
            (1, 1, 1, 1),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        )
    ]
    with pytest.raises(ValueError):
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
    good_physical_states: list[PlaquetteState] = [
        (
            (1, 1, 1, 1),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (1, 1, 1, 1),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            ((ONE,), (THREE,), (ONE,), (ONE,))
        ),
        (
            (1, 1, 1, 2),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        )
    ]
    with pytest.raises(ValueError):
        LatticeStateEncoder(non_unique_link_bitmap, good_physical_states, lattice_d_3_2)

    print("Checking that a link bitmap with bit strings of different lengths causes ValueError.")
    link_bitmap_different_string_lengths = {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "010"
    }
    with pytest.raises(ValueError):
        LatticeStateEncoder(link_bitmap_different_string_lengths, good_physical_states, lattice_d_3_2)


def test_encode_decode_various_links():
    lattice_d_3_2 = LatticeDef("3/2", 3)
    link_bitmap = {
        ONE: "00",
        THREE: "10",
        EIGHT: "11"
    }
    # Test data, not physically meaningful but has right format for creating creating an encoder.
    # Its links carry only irreps that link_bitmap has codes for.
    physical_plaquette_states = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, EIGHT, EIGHT),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, EIGHT, EIGHT),
            ((ONE,), (THREE,), (ONE,), (ONE,))
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
    # Should generate a bitmap to length-2 bitstrings for vertices. The links
    # carry only irreps that link_bitmap has codes for.
    physical_plaquette_states = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, EIGHT, EIGHT),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 1, 0),
            (ONE, THREE, EIGHT, EIGHT),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 2, 0),
            (ONE, THREE, EIGHT, EIGHT),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, EIGHT, EIGHT),
            ((ONE,), (THREE,), (ONE,), (ONE,))
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
        link_bitmap=_T2_LINK_BITMAP,
        physical_plaquette_states=physical_plaquette_states(lattice_d_3_2, "T", 2),
        lattice=lattice_d_3_2
    )

    print("Checking that encoding a wrong-length plaquette fails.")
    plaquette_wrong_length: PlaquetteState = (
        (0, 0, 0, 0),
        (ONE, ONE, ONE, ONE),
        ((ONE,), (ONE,), (ONE,), (ONE,)),
        ((ONE,), (ONE,), (ONE,), (SIX,))
    )
    with pytest.raises(ValueError):
        lattice_encoder.encode_plaquette_state_as_bit_string(plaquette_wrong_length)

    print("Checking that encoding a plaquette with the wrong data types for vertices, active links, or control links fails.")
    plaquette_bad_vertex_data: PlaquetteState = (
        (0, 0, "zero", 0),
        (ONE, ONE, ONE, ONE),
        ((ONE,), (ONE,), (ONE,), (ONE,)),
    )
    plaquette_bad_a_link_data: PlaquetteState = (
        (0, 0, 0, 0),
        (ONE, ONE, (0, 0), ONE),
        ((ONE,), (ONE,), (ONE,), (ONE,)),
    )
    plaquette_bad_c_link_data: PlaquetteState = (
        (0, 0, 0, 0),
        (ONE, ONE, ONE, ONE),
        ((ONE,), (ONE,), ((0, 0),), (ONE,)),
    )
    cases = {
        "Vertex test": plaquette_bad_vertex_data,
        "Active link test": plaquette_bad_a_link_data,
        "Control link test": plaquette_bad_c_link_data
    }
    for test_name, test_data in cases.items():
        with pytest.raises(ValueError):
            lattice_encoder.encode_plaquette_state_as_bit_string(test_data)

    print("Checking that encoding a plaquette which has too many vertices, active links, or control links fails.")
    plaquette_too_many_vertices: PlaquetteState = (
        (0, 0, 0, 0, 1),
        (ONE, ONE, ONE, ONE),
        ((SIX,), (SIX,), (SIX,), (SIX_BAR,))
    )
    plaquette_too_many_a_links: PlaquetteState = (
        (0, 0, 0, 1),
        (ONE, ONE, ONE, ONE, THREE),
        ((SIX,), (SIX,), (SIX,), (SIX_BAR,))
    )
    plaquette_too_many_c_links: PlaquetteState = (
        (0, 0, 1, 0),
        (ONE, ONE, ONE, ONE),
        ((SIX,), (SIX_BAR,), (EIGHT,))
    )
    cases = {
        "Vertex test": plaquette_too_many_vertices,
        "Active link test": plaquette_too_many_a_links,
        "Control link test": plaquette_too_many_c_links
    }
    for test_name, test_data in cases.items():
        with pytest.raises(ValueError):
            lattice_encoder.encode_plaquette_state_as_bit_string(test_data)


def test_encoding_good_plaquette():
    print("Check that the encoding some known plaquette yields the expected results.")

    # Set up needed encoders. TODO implement d=3 test.
    # d=3/2 T2 and d=2 T1 should have a single bit for multiplicity encoding.
    lattice_d_3_2 = LatticeDef(1.5, 3)
    lattice_d_2 = LatticeDef(2, 2)
    l_encoder_d_3_2_T1 = LatticeStateEncoder(_T1_LINK_BITMAP, physical_plaquette_states(lattice_d_3_2, "T", 1), lattice_d_3_2)
    l_encoder_d_3_2_T2 = LatticeStateEncoder(_T2_LINK_BITMAP, physical_plaquette_states(lattice_d_3_2, "T", 2), lattice_d_3_2)
    l_encoder_d_2_T1 = LatticeStateEncoder(_T1_LINK_BITMAP, physical_plaquette_states(lattice_d_2, "T", 1), lattice_d_2)

    # Construct test data.
    plaquette_d_3_2_T1 = (
        (0, 0, 0, 0),
        (ONE, THREE, THREE_BAR, ONE),
        ((ONE,), (ONE,), (ONE,), (THREE_BAR,))
    )
    expected_plaquette_d_3_2_T1_bit_string = "00100100" + "00000001" # a links + c links
    plaquette_d_3_2_T2 = (
        (0, 1, 0, 0),
        (SIX, EIGHT, THREE_BAR, ONE),
        ((ONE,), (ONE,), (SIX_BAR,), (THREE_BAR,))
    )
    expected_plaquette_d_3_2_T2_bit_string = "0100" + "110111001000" + "000000011001" # v + a links + c links
    plaquette_d_2_T1 = (
        (0, 0, 1, 0),
        (THREE, THREE, THREE_BAR, THREE_BAR),
        ((ONE, ONE), (THREE_BAR, THREE_BAR), (THREE, THREE), (THREE, ONE))
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


# TODO decide whether this test is slow enough to merit skipping by default.
# truncation mode, cutoff, expected_plaquette_bit_length, LatticeDef
# Each expected_bitlength = 4 * (2 * (dim - 1) + 1) * n_link_qubits + 4 * n_vertex_qubits for the corresponding case. This is just (n_control_links + n_active_links) * n_link_qubits + n_vertices * n_vertex_qubits.
@pytest.mark.parametrize(
    "truncation_mode,cutoff,current_expected_bitlength,current_lattice",
    [
        ("T", 1, (4 * (2 * (3/2 - 1) + 1) * 2) + 4*0, LatticeDef(1.5, 3)),
        ("T", 2, (4 * (2 * (3/2 - 1) + 1) * 3) + 4*1, LatticeDef(1.5, 3)),
        ("T", 1, (4 * (2 * (2 - 1) + 1) * 2) + 4*1, LatticeDef(2, 3)),
        ("B", 3, (4 * (2 * (3 - 1) + 1) * 2) + 4*0, LatticeDef(3, 3))
    ]
)
def test_all_mag_hamiltonian_plaquette_states_have_unique_bit_string_encoding(
        truncation_mode, cutoff, current_expected_bitlength, current_lattice):
    """
    Check that all plaquette states have unique bitstring encodings.

    Attempts encoding the following cases:
    - d=3/2, T1
    - d=3/2, T2
    - d=2, T1
    - d=3, B3
    """
    # Make encoder instance.
    lattice_encoder = LatticeStateEncoder.for_truncation(current_lattice, truncation_mode, cutoff)

    print(f"Case {current_lattice}, {truncation_mode}{cutoff}.\nConfirming all initial and final states "
          "appearing in the physical plaquette states list can be succesfully encoded. Using the bitmaps:\n"
          f"Link bitmap =  {lattice_encoder.link_bitmap}\n"
          f"Vertex bitmap = {lattice_encoder.vertex_bitmap}")

    # Get the set of unique plaquette states.
    transitions = box_terms(current_lattice, truncation_mode, cutoff).keys()
    all_plaquette_states = set([
        final_and_initial_state_tuple[0] for final_and_initial_state_tuple in transitions] + [
            final_and_initial_state_tuple[1] for final_and_initial_state_tuple in transitions
        ])

    # Attempt encodings and check for uniqueness.
    all_encoded_plaquette_bit_strings = []
    current_iter = 0
    for plaquette_state in all_plaquette_states:
        # Log test progress
        current_iter += 1
        percent_done = current_iter/len(all_plaquette_states)
        print(f"Current status: {percent_done:.4%}", end='\r')

        plaquette_state_bit_string = lattice_encoder.encode_plaquette_state_as_bit_string(plaquette_state)
        assert len(plaquette_state_bit_string) == current_expected_bitlength, f"len(plaquette_state_bit_string) == {len(plaquette_state_bit_string)}; expected len == {current_expected_bitlength}."
        all_encoded_plaquette_bit_strings.append(plaquette_state_bit_string)
    n_unique_plaquette_encodings = len(set(all_encoded_plaquette_bit_strings))
    assert n_unique_plaquette_encodings == len(all_plaquette_states), f"Encountered {n_unique_plaquette_encodings} unique bit strings encoding {len(all_plaquette_states)} unique plaquette states."


# See arxiv 2509.25865 for tables of max multiplicity per B truncation. Guides choice of vertex values in test data.
# Note that the data were manually constructed by irrep encoding bitmaps
# with data in vertex singlet json files.
@pytest.mark.parametrize("truncation_mode,cutoff,encoded_plaquette,current_lattice,link_bitmap,expected_decoded_plaquette", [
    (
        "T",
        1,
        "10001001" + "00000110", # active links + control links
        LatticeDef(3/2, 3),
        _T1_LINK_BITMAP,
        (
            (None, None, None, None),  # When no vertex bitmap needed, should get back None for decoded vertices.
            (THREE, ONE, THREE, THREE_BAR),
            ((ONE,), (ONE,), (THREE_BAR,), (THREE,))
        )
    ),
    (
        "B",
        10,
        "0110" + "100101100001" + "000010110100", # vertex multiplicites + active links + control links
        LatticeDef(3/2, 3),
        _B10_LINK_BITMAP,
        (
            (0, 1, 1, 0),  
            (THREE, FIFTEEN, THREE, THREE_BAR),
            ((ONE,), (FIFTEEN_BAR,), (SIX,), (THREE,))
        )
    ),
    (
        "T",
        2,
        "0001" + "110111000001" + "000000111011",  # vertex multiplicities + active links + control links
        LatticeDef(3/2, 3),
        _T2_LINK_BITMAP,
        (
            (0, 0, 0, 1),
            (SIX, EIGHT, ONE, THREE_BAR),
            ((ONE,), (ONE,), (EIGHT,), (SIX_BAR,))
        )
    ),
    (
        "T",
        1,
        "1011" + "00000010" + "0101011010000111",  # vertex multiplicities + active links + control links (one of which is in a garbage state)
        LatticeDef(2, 2),
        _T1_LINK_BITMAP,
        (
            (1, 0, 1, 1),
            (ONE, ONE, ONE, THREE),
            ((THREE_BAR, THREE_BAR), (THREE_BAR, THREE), (THREE, ONE), (THREE_BAR, None)) # Garbage control link should decode to None
        )
    ),
    (
        "B",
        3,
        "00000010" + "01000001010000101000000001000000",  # active links + control links
        LatticeDef(3, 3),
        _T1_LINK_BITMAP,
        (
            (None, None, None, None), # No nontrivial multiplicites in this case.
            (ONE, ONE, ONE, THREE),
            ((THREE_BAR, ONE, ONE, THREE_BAR), (THREE_BAR, ONE, ONE, THREE), (THREE, ONE, ONE, ONE), (THREE_BAR, ONE, ONE, ONE))
        )
    )
])
def test_bit_string_decoding_to_plaquette(truncation_mode, cutoff, encoded_plaquette, current_lattice, link_bitmap, expected_decoded_plaquette):
    print("Checking decoding of bit strings corresponding to gauge-invariant plaquette states.")

    lattice_encoder = LatticeStateEncoder(
        link_bitmap,
        physical_plaquette_states(current_lattice, truncation_mode, cutoff),
        current_lattice
    )
    resulting_decoded_plaquette = lattice_encoder.decode_bit_string_to_plaquette_state(encoded_plaquette)
    assert resulting_decoded_plaquette == expected_decoded_plaquette, f"Expected: {expected_decoded_plaquette}\nEncountered: {resulting_decoded_plaquette}"


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
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 1, 0),
            (ONE, ONE, ONE, ONE),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 2, 0),
            (ONE, ONE, ONE, ONE),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (THREE, ONE, EIGHT, ONE),
            ((ONE,), (SIX_BAR,), (ONE,), (ONE,))
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
        ((None,), (ONE,), (ONE,), (SIX_BAR,))
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
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 1, 0),
            (ONE, ONE, ONE, ONE),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 2, 0),
            (ONE, ONE, ONE, ONE),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        ),
        (
            (0, 0, 0, 0),
            (THREE, ONE, EIGHT, ONE),
            ((ONE, SIX_BAR), (ONE, ONE), (ONE, ONE), (ONE, ONE))
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
    with pytest.raises(ValueError):
        lattice_encoder.decode_bit_string_to_link_state(bad_length_link_bit_string)

    print(f"Checking vertex bit string {bad_length_vertex_bit_string} fails to decode.")
    with pytest.raises(ValueError):
        lattice_encoder.decode_bit_string_to_vertex_state(bad_length_vertex_bit_string)

    print(f"Checking plaquette bit string {bad_length_plaquette_bit_string} fails to decode.")
    with pytest.raises(ValueError):
        lattice_encoder.decode_bit_string_to_plaquette_state(bad_length_plaquette_bit_string)


def test_non_default_forder_plaquette_encode_decode_round_trip():
    """
    Check that encode/decode round-trips correctly with non-default forder.

    Also verifies that the same physical state (same irreps on same physical links)
    produces a different bit string when represented in a different forder convention,
    because the within-vertex control link ordering changes.
    """
    print("Checking the encoder's forder and encode/decode round-trip with non-default forder.")
    alt_forder = [-1, -2, 1, 2, 3, -3]

    # Create encoders with default and alt forder.
    default_lattice_def = LatticeDef(2, 2)
    alt_lattice_def = LatticeDef(2, 2, forder=alt_forder)

    default_encoder = LatticeStateEncoder.for_truncation(default_lattice_def, "T", 1)
    alt_encoder = LatticeStateEncoder.for_truncation(alt_lattice_def, "T", 1)

    # Verify that each encoder keeps its lattice's forder.
    assert default_encoder.lattice_def.forder == (1, 2, 3, -1, -2, -3), "Default forder incorrect."
    assert alt_encoder.lattice_def.forder == (-1, -2, 1, 2, 3, -3), "Alt forder incorrect."

    # A test plaquette state where v2 has distinct ctrl link irreps.
    # Default forder v2 ctrl dirs: (+1, -2) → tuple position 0 = dir+1, position 1 = dir-2.
    # Physical state: v2 dir+1 = THREE, v2 dir-2 = THREE_BAR.
    default_state = (
        (0, 0, 0, 0),
        (ONE, ONE, ONE, ONE),
        ((ONE, ONE), (THREE, THREE_BAR), (ONE, ONE), (ONE, ONE))
    )

    # Same physical state in alt forder representation.
    # Alt forder v2 ctrl dirs: (-2, +1) → tuple position 0 = dir-2, position 1 = dir+1.
    # So v2 tuple becomes (THREE_BAR, THREE) — swapped.
    alt_state = (
        (0, 0, 0, 0),
        (ONE, ONE, ONE, ONE),
        ((ONE, ONE), (THREE_BAR, THREE), (ONE, ONE), (ONE, ONE))
    )

    # Encode each representation with its matching encoder.
    default_bitstring = default_encoder.encode_plaquette_state_as_bit_string(default_state)
    alt_bitstring = alt_encoder.encode_plaquette_state_as_bit_string(alt_state)

    print(f"  Default forder bit string: {default_bitstring}")
    print(f"  Alt forder bit string:     {alt_bitstring}")

    # Same physical state, different forder → different bit strings.
    assert default_bitstring != alt_bitstring, (
        "Expected different bit strings for same physical state under different forders."
    )

    # Round-trip: decode each bit string back and verify we get the original state.
    default_round_trip = default_encoder.decode_bit_string_to_plaquette_state(default_bitstring)
    alt_round_trip = alt_encoder.decode_bit_string_to_plaquette_state(alt_bitstring)

    assert default_round_trip == default_state, "Default forder round-trip failed."
    assert alt_round_trip == alt_state, "Alt forder round-trip failed."


# Each data file was computed on the lattice in its metadata: d=3/2 of size
# 3, d=2 3x3 and d=3 3x3x3, all periodic.
@pytest.mark.parametrize("dims,truncation_mode,cutoff,expected_link_bitmap", [
    (1.5, "T", 1, _T1_LINK_BITMAP),
    (1.5, "T", 2, _T2_LINK_BITMAP),
    (1.5, "B", 3, _T1_LINK_BITMAP),
    (1.5, "B", 5, _T1_LINK_BITMAP),
    (1.5, "B", 6, _T2_LINK_BITMAP),
    (1.5, "B", 7, _T2_LINK_BITMAP),
    (1.5, "B", 8, _T2_LINK_BITMAP),
    (1.5, "B", 9, _T2_LINK_BITMAP),
    (1.5, "B", 10, _B10_LINK_BITMAP),
    (2, "T", 1, _T1_LINK_BITMAP),
    (2, "B", 3, _T1_LINK_BITMAP),
    (2, "B", 4, _T1_LINK_BITMAP),
    (2, "B", 7, _T2_LINK_BITMAP),
    (3, "B", 3, _T1_LINK_BITMAP),
    (3, "B", 4, _T1_LINK_BITMAP),
])
def test_for_truncation_uses_registered_bitmap(dims, truncation_mode, cutoff, expected_link_bitmap):
    """On each data file's lattice, the encoder uses the bitmap ymcirc used for that truncation."""
    lattice = LatticeDef(dims, 3)
    encoder = LatticeStateEncoder.for_truncation(lattice, truncation_mode, cutoff)
    assert encoder.link_bitmap == expected_link_bitmap


def test_for_truncation_records_truncation():
    """The encoder records the truncation it was created for; one created from its constructor records none."""
    lattice = LatticeDef(1.5, 3)
    assert LatticeStateEncoder.for_truncation(lattice, "B", 3).truncation == ("B", 3)
    assert LatticeStateEncoder.for_truncation(lattice, "T", 1).truncation == ("T", 1)
    encoder = LatticeStateEncoder(_T1_LINK_BITMAP, physical_plaquette_states(lattice, "T", 1), lattice)
    assert encoder.truncation is None


def test_encoder_truncation_cannot_be_reassigned():
    """The recorded truncation cannot be replaced, so it always describes the encoder's plaquette states."""
    encoder = LatticeStateEncoder.for_truncation(LatticeDef(1.5, 3), "B", 3)
    with pytest.raises(AttributeError):
        # A deliberate assignment to the read-only property.
        encoder.truncation = ("T", 1)  # ty: ignore[invalid-assignment]
    assert encoder.truncation == ("B", 3)


def test_for_truncation_unregistered_irrep_set_raises_value_error():
    """d=3/2 at B11 has 10 and 10bar on its links, for which no bitmap is registered."""
    lattice = LatticeDef(1.5, 3)
    with pytest.raises(ValueError, match=r"\(3, 0, 0\)") as e_info:
        LatticeStateEncoder.for_truncation(lattice, "B", 11)
    assert "(3, 3, 0)" in str(e_info.value)
    assert "link_bitmap" in str(e_info.value)


def test_for_truncation_with_explicit_bitmap():
    """An explicit bitmap is used if it has a code for every link irrep; codes for other irreps are allowed."""
    lattice = LatticeDef(1.5, 3)
    t1_bitmap = {ONE: "00", THREE: "10", THREE_BAR: "01"}
    encoder = LatticeStateEncoder.for_truncation(lattice, "T", 1, link_bitmap=t1_bitmap)
    assert encoder.link_bitmap == t1_bitmap

    extra_six = {ONE: "00", THREE: "10", THREE_BAR: "01", SIX: "11"}
    encoder = LatticeStateEncoder.for_truncation(lattice, "T", 1, link_bitmap=extra_six)
    assert encoder.link_bitmap == extra_six

    missing_three_bar = {ONE: "00", THREE: "10"}
    with pytest.raises(ValueError, match=r"no code for the irreps \[\(1, 1, 0\)\]"):
        LatticeStateEncoder.for_truncation(lattice, "T", 1, link_bitmap=missing_three_bar)


def test_lattice_encoder_raises_value_error_for_irrep_without_code():
    """The encoder refuses a bitmap without a code for an irrep on a link of its plaquette states."""
    physical_states: list[PlaquetteState] = [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        ),
        (
            (0, 0, 0, 0),
            (ONE, ONE, ONE, ONE),
            ((ONE,), (SIX,), (ONE,), (ONE,))
        ),
    ]
    link_bitmap_without_six = {ONE: "00", THREE: "10", THREE_BAR: "01"}
    with pytest.raises(ValueError, match=r"no code for the irreps \[\(2, 0, 0\)\]"):
        LatticeStateEncoder(link_bitmap_without_six, physical_states, LatticeDef(1.5, 3))


def test_plaquette_state_format_conversion_on_a_hand_written_state():
    """The conversion regroups pyclebsch's flat state into ymcirc's (multiplicities, active links, controls).

    Every entry differs from every other, so exchanging any two positions or
    groups changes the result. The state need not be physical.
    """
    pyclebsch_state = (
        THREE, THREE_BAR, ONE, EIGHT,  # active links at the four corners
        (ONE,), (THREE,), (SIX, SIX_BAR), (FIFTEEN,),  # each corner's control links
        0, 1, 3, 2,  # each corner's singlet multiplicity index
    )
    ymcirc_state = (
        (0, 1, 3, 2),
        (THREE, THREE_BAR, ONE, EIGHT),
        ((ONE,), (THREE,), (SIX, SIX_BAR), (FIFTEEN,)),
    )
    assert plaq_state_pyclebsch_to_ymcirc_format(pyclebsch_state) == ymcirc_state
    assert plaq_state_ymcirc_to_pyclebsch_format(ymcirc_state) == pyclebsch_state


@pytest.mark.parametrize("pyclebsch_lattice,truncation_mode,cutoff", [
    (((3, 3, 3), (True, True, True)), "B", 3),
    (((3, 2, 1), (True, False, False)), "T", 2),
])
def test_plaquette_state_format_conversion_round_trips(pyclebsch_lattice, truncation_mode, cutoff):
    """pyclebsch's plaquette-state format to ymcirc's and back is the identity on every physical state."""
    num_sites, pbcs = pyclebsch_lattice
    states_by_signature = physical_plaquette_states_by_signature(
        PyclebschLatticeDef(num_sites, pbcs, (1, 2, 3, -1, -2, -3)), N_COLORS, truncation_mode, cutoff)
    for signature_states in states_by_signature.values():
        for state in signature_states:
            ymcirc_state = plaq_state_pyclebsch_to_ymcirc_format(state)
            assert plaq_state_ymcirc_to_pyclebsch_format(ymcirc_state) == tuple(state)
