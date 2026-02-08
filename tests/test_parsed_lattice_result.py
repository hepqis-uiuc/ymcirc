import copy
import pytest
from typing import Dict, List
from ymcirc._abstract import LatticeDef
from ymcirc.parsed_lattice_result import ParsedLatticeResult
from ymcirc.conventions import LatticeStateEncoder, IrrepWeight, PlaquetteState, ONE, THREE, THREE_BAR, SIX, SIX_BAR, EIGHT


# Create some test data.
@pytest.fixture
def T1_link_bitmap() -> Dict[IrrepWeight, str]:
    return {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "01"
    }


@pytest.fixture
def T2_link_bitmap() -> Dict[IrrepWeight, str]:
    return {
        ONE: "000",
        THREE: "100",
        THREE_BAR: "001",
        SIX: "110",
        SIX_BAR: "011",
        EIGHT: "111"
    }


@pytest.fixture
def good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed() -> List[PlaquetteState]:
    return [
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
            (ONE, ONE, ONE, ONE),
            (ONE, ONE, ONE, ONE)
        )
    ]


@pytest.fixture
def good_physical_plaquette_states_d_3_2_T1_vertex_data_needed() -> List[PlaquetteState]:
    return [
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
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, ONE, ONE, ONE),
            (ONE, ONE, ONE, ONE)
        )
    ]


@pytest.fixture
def good_physical_plaquette_states_d_3_2_T2_two_vertex_qubits_needed() -> List[PlaquetteState]:
    return [
        (
            (0, 0, 0, 0),
            (ONE, EIGHT, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (0, 1, 0, 0),
            (ONE, EIGHT, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (0, 2, 0, 0),
            (ONE, EIGHT, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE)
        ),
        (
            (0, 0, 0, 1),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, ONE, SIX, ONE),
            (ONE, SIX_BAR, ONE, ONE)
        )
    ]


@pytest.fixture
def good_physical_plaquette_states_d_2_T1_one_vertex_qubit() -> List[PlaquetteState]:
    return [
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE, THREE_BAR),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE, THREE, THREE, THREE, ONE)
        ),
        (
            (0, 0, 0, 1),
            (ONE, THREE, THREE_BAR, THREE_BAR),
            (ONE, THREE, ONE, ONE, THREE, THREE, THREE, ONE)
        ),
        (
            (0, 0, 0, 0),
            (ONE, ONE, ONE, ONE),
            (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
        )
    ]


# Tests begin.
def test_parse_d_3_2_lattice_no_vertex_data(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed):
    """Check creation of parsed lattice measurement d=3/2 when no vertex data present."""
    # Prepare test data.
    link_bitmap = T1_link_bitmap
    physical_plaquette_states = good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed
    lattice = LatticeDef(1.5, 2)
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)
    global_meas_bit_string = "110001101000"
    expected_decoded_lattice_result_dict = {
        (0, 0): None,
        ((0, 0), 1): None,  # Because junk substring.
        ((0, 0), 2): ONE,
        (0, 1): None,
        ((0, 1), 1): THREE_BAR,
        (1, 0): None,
        ((1, 0), 1): THREE,
        ((1, 0), 2): THREE,
        (1, 1): None,
        ((1, 1), 1): ONE,
    }
    expected_bit_string_lattice_result_dict = {
        (0, 0): "",             # Because no vertex data.
        ((0, 0), 1): "11",
        ((0, 0), 2): "00",
        (0, 1): "",
        ((0, 1), 1): "01",
        (1, 0): "",
        ((1, 0), 1): "10",
        ((1, 0), 2): "10",
        (1, 1): "",
        ((1, 1), 1): "00",
    }
    expected_decoded_plaquette_result_dict = {
        (0, 0): {
            "a_links": (None, THREE, THREE_BAR, ONE),
            "c_links_ordered": (THREE, THREE, ONE, ONE),
            "vertices": (None, None, None, None)
        },
        (1, 0): {
            "a_links": (THREE, ONE, ONE, THREE),
            "c_links_ordered": (None, None, THREE_BAR, THREE_BAR),
            "vertices": (None, None, None, None)
        }
    }
    expected_bit_string_plaquette_result_dict = {
        (0, 0): {
            "a_links": ("11", "10", "01", "00"),
            "c_links_ordered": ("10", "10", "00", "00"),
            "vertices": ("", "", "", "")
        },
        (1, 0): {
            "a_links": ("10", "00", "00", "10"),
            "c_links_ordered": ("11", "11", "01", "01"),
            "vertices": ("", "", "", "")
        },
    }

    # Test proper begins here.
    parsed_lattice_meas = ParsedLatticeResult(
        dimensions=1.5,
        size=2,
        global_lattice_measurement_bit_string=global_meas_bit_string,
        lattice_encoder=lattice_encoder)

    for (current_vertex, current_link_list) in parsed_lattice_meas.get_traversal_order():
        print("On vertex: ", current_vertex)
        # Check vertex data.
        assert parsed_lattice_meas.get_vertex(lattice_vector=current_vertex) == expected_decoded_lattice_result_dict[current_vertex]
        assert parsed_lattice_meas.get_vertex(lattice_vector=current_vertex, get_bit_string=True) == expected_bit_string_lattice_result_dict[current_vertex]

        # Check link data.
        for current_link_address in current_link_list:
            assert parsed_lattice_meas.get_link(link_address=current_link_address) == expected_decoded_lattice_result_dict[current_link_address]
            assert parsed_lattice_meas.get_link(link_address=current_link_address, get_bit_string=True) == expected_bit_string_lattice_result_dict[current_link_address]

        # Check plaquette data
        # (if not on bottom rung of d=3/2 lattice, KeyError should occur, so just skip getting plaquettes for current vertex).
        try:
            current_plaquette_decoded_result = parsed_lattice_meas.get_plaquettes(current_vertex)
            current_plaquette_bit_string_result = parsed_lattice_meas.get_plaquettes(current_vertex, get_bit_string=True)
        except KeyError:
            continue
        # Decoded plaquette.
        assert current_plaquette_decoded_result.active_links == expected_decoded_plaquette_result_dict[current_vertex]["a_links"]
        assert current_plaquette_decoded_result.control_links_ordered == expected_decoded_plaquette_result_dict[current_vertex]["c_links_ordered"]
        assert current_plaquette_decoded_result.vertices == expected_decoded_plaquette_result_dict[current_vertex]["vertices"]
        # Bit string plaquette.
        assert current_plaquette_bit_string_result.active_links == expected_bit_string_plaquette_result_dict[current_vertex]["a_links"]
        assert current_plaquette_bit_string_result.control_links_ordered == expected_bit_string_plaquette_result_dict[current_vertex]["c_links_ordered"]
        assert current_plaquette_bit_string_result.vertices == expected_bit_string_plaquette_result_dict[current_vertex]["vertices"]


def test_parse_d_3_2_lattice_with_vertex_data(
        T2_link_bitmap,
        good_physical_plaquette_states_d_3_2_T2_two_vertex_qubits_needed):
    """Check creation of parsed lattice measurement d=3/2 when two vertex qubits were present."""
    # Prepare test data.
    link_bitmap = T2_link_bitmap
    physical_plaquette_states = good_physical_plaquette_states_d_3_2_T2_two_vertex_qubits_needed
    lattice = LatticeDef(1.5, 3)
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)
    global_meas_bit_string = "00" + "000000" + "01" + "111" + "10" + "110011" + "11" + "000" + "00" + "100001" "10" + "010"
    expected_decoded_lattice_result_dict = {
        (0, 0): 0,
        ((0, 0), 1): ONE,
        ((0, 0), 2): ONE,
        (0, 1): 1,
        ((0, 1), 1): EIGHT,
        (1, 0): 2,
        ((1, 0), 1): SIX,
        ((1, 0), 2): SIX_BAR,
        (1, 1): None,           # Because junk substring
        ((1, 1), 1): ONE,
        (2, 0): 0,
        ((2, 0), 1): THREE,
        ((2, 0), 2): THREE_BAR,
        (2, 1): 2,
        ((2, 1), 1): None,      # Because junk substring.
    }
    expected_bit_string_lattice_result_dict = {
        (0, 0): "00",
        ((0, 0), 1): "000",
        ((0, 0), 2): "000",
        (0, 1): "01",
        ((0, 1), 1): "111",
        (1, 0): "10",
        ((1, 0), 1): "110",
        ((1, 0), 2): "011",
        (1, 1): "11",
        ((1, 1), 1): "000",
        (2, 0): "00",
        ((2, 0), 1): "100",
        ((2, 0), 2): "001",
        (2, 1): "10",
        ((2, 1), 1): "010",
    }
    expected_decoded_plaquette_result_dict = {
        (0, 0): {
            "a_links": (ONE, SIX_BAR, EIGHT, ONE),
            "c_links_ordered": (THREE, SIX, ONE, None),
            "vertices": (0, 2, None, 1)
        },
        (1, 0): {
            "a_links": (SIX, THREE_BAR, ONE, SIX_BAR),
            "c_links_ordered": (ONE, THREE, None, EIGHT),
            "vertices": (2, 0, 2, None)
        },
        (2, 0): {
            "a_links": (THREE, ONE, None, THREE_BAR),
            "c_links_ordered": (SIX, ONE, EIGHT, ONE),
            "vertices": (0, 0, 1, 2)
        }
    }
    expected_bit_string_plaquette_result_dict = {
        (0, 0): {
            "a_links": ("000", "011", "111", "000"),
            "c_links_ordered": ("100", "110", "000", "010"),
            "vertices": ("00", "10", "11", "01")
        },
        (1, 0): {
            "a_links": ("110", "001", "000", "011"),
            "c_links_ordered": ("000", "100", "010", "111"),
            "vertices": ("10", "00", "10", "11")
        },
        (2, 0): {
            "a_links": ("100", "000", "010", "001"),
            "c_links_ordered": ("110", "000", "111", "000"),
            "vertices": ("00", "00", "01", "10")
        },
    }

    # Test proper begins here.
    parsed_lattice_meas = ParsedLatticeResult(
        dimensions=1.5,
        size=3,
        global_lattice_measurement_bit_string=global_meas_bit_string,
        lattice_encoder=lattice_encoder)

    for (current_vertex, current_link_list) in parsed_lattice_meas.get_traversal_order():
        print("On vertex: ", current_vertex)
        # Check vertex data.
        assert parsed_lattice_meas.get_vertex(lattice_vector=current_vertex) == expected_decoded_lattice_result_dict[current_vertex]
        assert parsed_lattice_meas.get_vertex(lattice_vector=current_vertex, get_bit_string=True) == expected_bit_string_lattice_result_dict[current_vertex]

        # Check link data.
        for current_link_address in current_link_list:
            assert parsed_lattice_meas.get_link(link_address=current_link_address) == expected_decoded_lattice_result_dict[current_link_address]
            assert parsed_lattice_meas.get_link(link_address=current_link_address, get_bit_string=True) == expected_bit_string_lattice_result_dict[current_link_address]

        # Check plaquette data
        # (if not on bottom rung of d=3/2 lattice, KeyError should occur, so just skip getting plaquettes for current vertex).
        try:
            current_plaquette_decoded_result = parsed_lattice_meas.get_plaquettes(current_vertex)
            current_plaquette_bit_string_result = parsed_lattice_meas.get_plaquettes(current_vertex, get_bit_string=True)
        except KeyError:
            continue
        # Decoded plaquette.
        assert current_plaquette_decoded_result.active_links == expected_decoded_plaquette_result_dict[current_vertex]["a_links"]
        assert current_plaquette_decoded_result.control_links_ordered == expected_decoded_plaquette_result_dict[current_vertex]["c_links_ordered"]
        assert current_plaquette_decoded_result.vertices == expected_decoded_plaquette_result_dict[current_vertex]["vertices"]
        # Bit string plaquette.
        assert current_plaquette_bit_string_result.active_links == expected_bit_string_plaquette_result_dict[current_vertex]["a_links"]
        assert current_plaquette_bit_string_result.control_links_ordered == expected_bit_string_plaquette_result_dict[current_vertex]["c_links_ordered"]
        assert current_plaquette_bit_string_result.vertices == expected_bit_string_plaquette_result_dict[current_vertex]["vertices"]


def test_full_lattice_bit_string_matches_reconstructed_bit_string(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_vertex_data_needed):
    # Prepare test data.
    link_bitmap = T1_link_bitmap
    physical_plaquette_states = good_physical_plaquette_states_d_3_2_T1_vertex_data_needed
    lattice = LatticeDef(1.5, 4)
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)
    global_meas_bit_string = "0" + "1100" + "1" + "01" + "0" + "1000" + "0" + "00" + "1" + "1111" + "1" + "11" + "0" + "0000" + "0" + "00"

    # Test begins.
    parsed_lattice_meas = ParsedLatticeResult(
        dimensions=1.5,
        size=4,
        global_lattice_measurement_bit_string=global_meas_bit_string,
        lattice_encoder=lattice_encoder)

    reconstructed_global_bitstring = ""
    for (current_vertex_address, current_connected_links) in parsed_lattice_meas.get_traversal_order():
        reconstructed_global_bitstring += parsed_lattice_meas.get_vertex(current_vertex_address, get_bit_string=True)
        for current_link_address in current_connected_links:
            reconstructed_global_bitstring += parsed_lattice_meas.get_link(current_link_address, get_bit_string=True)

    assert reconstructed_global_bitstring == global_meas_bit_string
    assert parsed_lattice_meas.global_lattice_measurement_bit_string == global_meas_bit_string


def test_parsed_lattice_result_has_accurate_copy_of_encoders_lattice_def(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed):
    # Prepare test data.
    link_bitmap = T1_link_bitmap
    physical_plaquette_states = good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed
    lattice = LatticeDef(1.5, 2)
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)
    global_meas_bit_string = "000000000000"

    # Test begins.
    parsed_lattice_meas = ParsedLatticeResult(
        dimensions=1.5,
        size=2,
        global_lattice_measurement_bit_string=global_meas_bit_string,
        lattice_encoder=lattice_encoder)

    assert parsed_lattice_meas.lattice_def == lattice_encoder.lattice_def # Check that same value.
    assert parsed_lattice_meas.lattice_def is not lattice_encoder.lattice_def # Check that not the same instance.


def test_parse_d_2_lattice_with_vertex_data(
        T1_link_bitmap, good_physical_plaquette_states_d_2_T1_one_vertex_qubit):
    """Check creation of parsed lattice measurement d=2 when one vertex qubit was present."""
    # Prepare test data.
    link_bitmap = T1_link_bitmap
    physical_plaquette_states = good_physical_plaquette_states_d_2_T1_one_vertex_qubit
    lattice = LatticeDef(2, 2)
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)
    global_meas_bit_string = "0" + "0000" + "1" + "0011" + "0" + "0110" + "0" + "1010"
    expected_decoded_lattice_result_dict = {
        (0, 0): 0,
        ((0, 0), 1): ONE,
        ((0, 0), 2): ONE,
        (0, 1): 1,
        ((0, 1), 1): ONE,
        ((0, 1), 2): None,      # Junk bit string.
        (1, 0): 0,
        ((1, 0), 1): THREE_BAR,
        ((1, 0), 2): THREE,
        (1, 1): 0,
        ((1, 1), 1): THREE,
        ((1, 1), 2): THREE
    }
    expected_bit_string_lattice_result_dict = {
        (0, 0): "0",
        ((0, 0), 1): "00",
        ((0, 0), 2): "00",
        (0, 1): "1",
        ((0, 1), 1): "00",
        ((0, 1), 2): "11",
        (1, 0): "0",
        ((1, 0), 1): "01",
        ((1, 0), 2): "10",
        (1, 1): "0",
        ((1, 1), 1): "10",
        ((1, 1), 2): "10",
    }
    expected_decoded_plaquette_result_dict = {
        (0, 0): {
            "a_links": (ONE, THREE, ONE, ONE),
            "c_links_ordered": (THREE_BAR, None, THREE, THREE_BAR, THREE, THREE, None, THREE),
            "vertices": (0, 0, 0, 1)
        },
        (0, 1): {
            "a_links": (ONE, THREE, ONE, None),
            "c_links_ordered": (THREE, ONE, THREE, THREE, THREE_BAR, THREE, ONE, THREE_BAR),
            "vertices": (1, 0, 0, 0)
        },
        (1, 0): {
            "a_links": (THREE_BAR, ONE, THREE, THREE),
            "c_links_ordered": (ONE, THREE, None, ONE, ONE, None, THREE, ONE),
            "vertices": (0, 0, 1, 0)
        },
        (1, 1): {
            "a_links": (THREE, None, THREE_BAR, THREE),
            "c_links_ordered": (ONE, THREE, ONE, ONE, ONE, ONE, THREE, ONE),
            "vertices": (0, 1, 0, 0)
        }
    }
    expected_bit_string_plaquette_result_dict = {
        (0, 0): {
            "a_links": ("00", "10", "00", "00"),
            "c_links_ordered": ("01", "11", "10", "01", "10", "10", "11", "10"),
            "vertices": ("0", "0", "0", "1")
        },
        (0, 1): {
            "a_links": ("00", "10", "00", "11"),
            "c_links_ordered": ("10", "00", "10", "10", "01", "10", "00", "01"),
            "vertices": ("1", "0", "0", "0")
        },
        (1, 0): {
            "a_links": ("01", "00", "10", "10"),
            "c_links_ordered": ("00", "10", "11", "00", "00", "11", "10", "00"),
            "vertices": ("0", "0", "1", "0")
        },
        (1, 1): {
            "a_links": ("10", "11", "01", "10"),
            "c_links_ordered": ("00", "10", "00", "00", "00", "00", "10", "00"),
            "vertices": ("0", "1", "0", "0")
        },
    }

    # Test proper begins here.
    parsed_lattice_meas = ParsedLatticeResult(
        dimensions=2,
        size=2,
        global_lattice_measurement_bit_string=global_meas_bit_string,
        lattice_encoder=lattice_encoder)

    for (current_vertex, current_link_list) in parsed_lattice_meas.get_traversal_order():
        print("On vertex: ", current_vertex)
        # Check vertex data.
        assert parsed_lattice_meas.get_vertex(lattice_vector=current_vertex) == expected_decoded_lattice_result_dict[current_vertex]
        assert parsed_lattice_meas.get_vertex(lattice_vector=current_vertex, get_bit_string=True) == expected_bit_string_lattice_result_dict[current_vertex]

        # Check link data.
        for current_link_address in current_link_list:
            assert parsed_lattice_meas.get_link(link_address=current_link_address) == expected_decoded_lattice_result_dict[current_link_address]
            assert parsed_lattice_meas.get_link(link_address=current_link_address, get_bit_string=True) == expected_bit_string_lattice_result_dict[current_link_address]

        # Check plaquette data.
        current_plaquette_decoded_result = parsed_lattice_meas.get_plaquettes(current_vertex)
        current_plaquette_bit_string_result = parsed_lattice_meas.get_plaquettes(current_vertex, get_bit_string=True)
        # Decoded plaquette.
        assert current_plaquette_decoded_result.active_links == expected_decoded_plaquette_result_dict[current_vertex]["a_links"]
        assert current_plaquette_decoded_result.control_links_ordered == expected_decoded_plaquette_result_dict[current_vertex]["c_links_ordered"]
        assert current_plaquette_decoded_result.vertices == expected_decoded_plaquette_result_dict[current_vertex]["vertices"]
        # Bit string plaquette.
        assert current_plaquette_bit_string_result.active_links == expected_bit_string_plaquette_result_dict[current_vertex]["a_links"]
        assert current_plaquette_bit_string_result.control_links_ordered == expected_bit_string_plaquette_result_dict[current_vertex]["c_links_ordered"]
        assert current_plaquette_bit_string_result.vertices == expected_bit_string_plaquette_result_dict[current_vertex]["vertices"]


def test_parsed_lattice_result_works_as_dict_key(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed):
    # Prepare test data.
    link_bitmap = T1_link_bitmap
    physical_plaquette_states = good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed
    lattice = LatticeDef(1.5, 2)
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)
    global_meas_bit_string = "110001101000"

    # Test begins.
    parsed_lattice_meas = ParsedLatticeResult(
        dimensions=1.5,
        size=2,
        global_lattice_measurement_bit_string=global_meas_bit_string,
        lattice_encoder=lattice_encoder)

    assert {parsed_lattice_meas: 4}[parsed_lattice_meas] == 4


def test_global_bit_string_too_short_for_lattice_size(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_vertex_data_needed):
    # Prepare test data.
    link_bitmap = T1_link_bitmap
    physical_plaquette_states = good_physical_plaquette_states_d_3_2_T1_vertex_data_needed
    lattice = LatticeDef(1.5, 4)
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)
    global_meas_bit_string = "10"  # This is too short.

    with pytest.raises(ValueError) as e_info:
        ParsedLatticeResult(
        dimensions=1.5,
        size=4,
        global_lattice_measurement_bit_string=global_meas_bit_string,
        lattice_encoder=lattice_encoder)


def test_inconsistent_dim_arg(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed,
        good_physical_plaquette_states_d_2_T1_one_vertex_qubit
):
    print("Checking case where ParsedLatticeResult is initialized with wrong dim...")
    link_bitmap = T1_link_bitmap
    physical_plaquette_states = good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed
    lattice = LatticeDef(1.5, 2)
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)
    global_meas_bit_string = "110001101000"

    with pytest.raises(ValueError) as e_info:
        ParsedLatticeResult(
            dimensions=2,
            size=2,
            global_lattice_measurement_bit_string=global_meas_bit_string,
            lattice_encoder=lattice_encoder)

    print("Checking case where LatticeStateEncoder is initialized with wrong dim...")
    link_bitmap = T1_link_bitmap
    physical_plaquette_states = good_physical_plaquette_states_d_2_T1_one_vertex_qubit
    lattice = LatticeDef(2, 2)
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)
    global_meas_bit_string = "1100011010001100"

    with pytest.raises(ValueError) as e_info:
        ParsedLatticeResult(
            dimensions=1.5,
            size=2,
            global_lattice_measurement_bit_string=global_meas_bit_string,
            lattice_encoder=lattice_encoder)


def test_inconsistent_size_arg(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed
):
    print("Checking case where ParsedLatticeResult is initialized with wrong size...")
    link_bitmap = T1_link_bitmap
    physical_plaquette_states = good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed
    lattice = LatticeDef(1.5, 2)
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)
    global_meas_bit_string = "110001101000000011110000"

    with pytest.raises(ValueError) as e_info:
        ParsedLatticeResult(
            dimensions=1.5,
            size=4,
            global_lattice_measurement_bit_string=global_meas_bit_string,
            lattice_encoder=lattice_encoder)

    print("Checking case where LatticeStateEncoder is initialized with wrong size...")
    link_bitmap = T1_link_bitmap
    physical_plaquette_states = good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed
    lattice = LatticeDef(1.5, 4)
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)
    global_meas_bit_string = "110001101000"

    with pytest.raises(ValueError) as e_info:
        ParsedLatticeResult(
            dimensions=1.5,
            size=2,
            global_lattice_measurement_bit_string=global_meas_bit_string,
            lattice_encoder=lattice_encoder)


# TODO disable skip once nonperiodic boundary conditions are implemented.
@pytest.mark.skip
def test_inconsistent_boundary_conds_arg(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed
):
    print("Checking case where ParsedLatticeResult is initialized with wrong bcs...")
    link_bitmap = T1_link_bitmap
    physical_plaquette_states = good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed
    lattice = LatticeDef(1.5, 2, periodic_boundary_conds=False)
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)
    global_meas_bit_string = "110001101000"

    with pytest.raises(ValueError) as e_info:
        ParsedLatticeResult(
            dimensions=1.5,
            size=2,
            global_lattice_measurement_bit_string=global_meas_bit_string,
            lattice_encoder=lattice_encoder)

    print("Checking case where LatticeStateEncoder is initialized with wrong bcs...")
    link_bitmap = T1_link_bitmap
    physical_plaquette_states = good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed
    lattice = LatticeDef(1.5, 2)
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)
    global_meas_bit_string = "110001101000"

    with pytest.raises(ValueError) as e_info:
        ParsedLatticeResult(
            dimensions=1.5,
            size=2,
            global_lattice_measurement_bit_string=global_meas_bit_string,
            lattice_encoder=lattice_encoder,
            periodic_boundary_conds=False
        )


def test_global_bit_string_has_bad_chars(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed):
    # Prepare test data.
    link_bitmap = T1_link_bitmap
    physical_plaquette_states = good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed
    lattice = LatticeDef(1.5, 2)
    lattice_encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)
    global_meas_bit_string = "1100011a1000"

    with pytest.raises(TypeError) as e_info:
        ParsedLatticeResult(
            dimensions=1.5,
            size=2,
            global_lattice_measurement_bit_string=global_meas_bit_string,
            lattice_encoder=lattice_encoder)


def test_parsed_lattice_result_equality(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed):
    """Two ParsedLatticeResult instances with the same data should be equal."""
    link_bitmap = T1_link_bitmap
    physical_plaquette_states = good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed
    lattice = LatticeDef(1.5, 2)
    encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)
    bitstring = "000000000000"

    plr1 = ParsedLatticeResult(1.5, 2, bitstring, encoder)
    plr2 = ParsedLatticeResult(1.5, 2, bitstring, encoder)

    assert plr1 == plr2
    assert plr1 is not plr2
    assert hash(plr1) == hash(plr2)

    # Different bitstring -> not equal
    plr3 = ParsedLatticeResult(1.5, 2, "100000000000", encoder)
    assert plr1 != plr3


def test_from_links_and_vertices_basic(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed):
    """from_links_and_vertices should create a ParsedLatticeResult from decoded data."""
    link_bitmap = T1_link_bitmap
    physical_plaquette_states = good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed
    lattice = LatticeDef(1.5, 2)
    encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)

    links_dict = {
        ((0, 0), 1): THREE,
        ((0, 0), 2): ONE,
        ((1, 0), 1): THREE_BAR,
    }

    plr = ParsedLatticeResult.from_links_and_vertices(
        links_dict=links_dict, encoder=encoder
    )

    # Provided links should decode correctly
    assert plr.get_link(((0, 0), 1)) == THREE
    assert plr.get_link(((0, 0), 2)) == ONE
    assert plr.get_link(((1, 0), 1)) == THREE_BAR

    # Bitstrings for provided links should be correct
    assert plr.get_link(((0, 0), 1), get_bit_string=True) == "10"
    assert plr.get_link(((0, 0), 2), get_bit_string=True) == "00"

    # Unprovided links should return None (decoded) and "XX" (bitstring)
    assert plr.get_link(((1, 0), 2)) is None
    assert plr.get_link(((1, 0), 2), get_bit_string=True) == "XX"

    # Unprovided vertices (no vertex qubits in T1 d=3/2): decoded=None, bitstring=""
    assert plr.get_vertex((0, 0)) is None
    assert plr.get_vertex((0, 0), get_bit_string=True) == ""

    # Lattice geometry should be correct
    assert plr.dim == 1.5
    assert plr.shape == (2, 2)


def test_from_links_and_vertices_with_vertices(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_vertex_data_needed):
    """from_links_and_vertices with explicit vertex data."""
    link_bitmap = T1_link_bitmap
    physical_plaquette_states = good_physical_plaquette_states_d_3_2_T1_vertex_data_needed
    lattice = LatticeDef(1.5, 2)
    encoder = LatticeStateEncoder(link_bitmap, physical_plaquette_states, lattice)

    links_dict = {((0, 0), 1): ONE}
    vertices_dict = {(0, 0): 0, (1, 0): 1}

    plr = ParsedLatticeResult.from_links_and_vertices(
        links_dict=links_dict, vertices_dict=vertices_dict, encoder=encoder
    )

    assert plr.get_vertex((0, 0)) == 0
    assert plr.get_vertex((1, 0)) == 1
    assert plr.get_vertex((0, 1)) is None  # Not provided
    assert plr.get_link(((0, 0), 1)) == ONE


def test_from_partial_measurement_link(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed):
    """from_partial_measurement with a single link measurement."""
    encoder = LatticeStateEncoder(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed,
        LatticeDef(1.5, 2))

    measurements = [(((0, 0), 1), "10")]  # Link ((0,0),1) measured as "10" = THREE

    plr = ParsedLatticeResult.from_partial_measurement(measurements, encoder)

    assert plr.get_link(((0, 0), 1)) == THREE
    assert plr.get_link(((0, 0), 1), get_bit_string=True) == "10"
    assert plr.get_link(((0, 0), 2)) is None  # Not measured
    assert plr.get_link(((0, 0), 2), get_bit_string=True) == "XX"


def test_from_partial_measurement_vertex(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_vertex_data_needed):
    """from_partial_measurement with a vertex measurement."""
    encoder = LatticeStateEncoder(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_vertex_data_needed,
        LatticeDef(1.5, 2))

    measurements = [((0, 0), "0")]  # Vertex (0,0) measured as "0" = multiplicity 0

    plr = ParsedLatticeResult.from_partial_measurement(measurements, encoder)

    assert plr.get_vertex((0, 0)) == 0
    assert plr.get_vertex((0, 0), get_bit_string=True) == "0"
    assert plr.get_vertex((1, 0)) is None  # Not measured


def test_from_partial_measurement_plaquette(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed):
    """from_partial_measurement with a plaquette measurement."""
    encoder = LatticeStateEncoder(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed,
        LatticeDef(1.5, 4))

    # Plaquette 0 in vacuum: all links = ONE = "00". No vertex qubits.
    # Active links: 4 * "00" = "00000000", control links: 4 * "00" = "00000000"
    # Total plaquette bitstring: "0000000000000000"
    plaq_bitstring = "0000000000000000"
    measurements = [(((0, 0), 1, 2), plaq_bitstring)]

    plr = ParsedLatticeResult.from_partial_measurement(measurements, encoder)

    # All 4 active links of plaquette 0 should be ONE
    assert plr.get_link(((0, 0), 1)) == ONE   # l1
    assert plr.get_link(((1, 0), 2)) == ONE   # l2
    assert plr.get_link(((0, 1), 1)) == ONE   # l3
    assert plr.get_link(((0, 0), 2)) == ONE   # l4

    # Links not in this plaquette should be None
    assert plr.get_link(((2, 0), 1)) is None


def test_global_bitstring_reconstructed_for_partial(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed):
    """global_lattice_measurement_bit_string should reconstruct from partial data."""
    encoder = LatticeStateEncoder(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed,
        LatticeDef(1.5, 2))

    # Only measure link ((0,0),1) = THREE = "10"
    links_dict = {((0, 0), 1): THREE}
    plr = ParsedLatticeResult.from_links_and_vertices(links_dict=links_dict, encoder=encoder)

    bitstring = plr.global_lattice_measurement_bit_string
    # d=3/2, L=2, T1: 6 links * 2 qubits = 12 total data qubits
    assert len(bitstring) == 12
    # First 2 chars should be "10" (link ((0,0),1) = THREE)
    assert bitstring[:2] == "10"
    # Remaining should be "XX" placeholders
    assert all(c == "X" for c in bitstring[2:])


def test_global_bitstring_unchanged_for_full_measurement(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed):
    """Full-measurement ParsedLatticeResult should return the original bitstring."""
    encoder = LatticeStateEncoder(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed,
        LatticeDef(1.5, 2))

    original = "110001101000"
    plr = ParsedLatticeResult(1.5, 2, original, encoder)
    assert plr.global_lattice_measurement_bit_string == original


from ymcirc.electric_helper import gt_pattern_iweight_to_casimir

def test_get_link_electric_energy(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed):
    """get_link_electric_energy returns Casimir for measured links, None for unmeasured."""
    encoder = LatticeStateEncoder(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed,
        LatticeDef(1.5, 2))

    links_dict = {
        ((0, 0), 1): THREE,      # C_2 = 4/3
        ((0, 0), 2): ONE,        # C_2 = 0
    }
    plr = ParsedLatticeResult.from_links_and_vertices(links_dict=links_dict, encoder=encoder)

    assert plr.get_link_electric_energy(((0, 0), 1)) == pytest.approx(4.0 / 3.0)
    assert plr.get_link_electric_energy(((0, 0), 2)) == pytest.approx(0.0)
    assert plr.get_link_electric_energy(((1, 0), 1)) is None  # Not measured


import warnings

def test_get_lattice_electric_energy_total(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed):
    """get_lattice_electric_energy with average_result=False returns total energy."""
    encoder = LatticeStateEncoder(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed,
        LatticeDef(1.5, 2))

    # All links vacuum = all C_2 = 0
    plr_vacuum = ParsedLatticeResult(1.5, 2, "000000000000", encoder)
    assert plr_vacuum.get_lattice_electric_energy(average_result=False) == pytest.approx(0.0)

    # Set all 6 links to THREE (C_2=4/3): total = 6 * 4/3 = 8.0
    all_three = {addr: THREE for addr in encoder.lattice_def.link_addresses}
    plr_three = ParsedLatticeResult.from_links_and_vertices(links_dict=all_three, encoder=encoder)
    assert plr_three.get_lattice_electric_energy(average_result=False) == pytest.approx(8.0)


def test_get_lattice_electric_energy_average(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed):
    """get_lattice_electric_energy with average_result=True divides by number of links."""
    encoder = LatticeStateEncoder(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed,
        LatticeDef(1.5, 2))

    all_three = {addr: THREE for addr in encoder.lattice_def.link_addresses}
    plr = ParsedLatticeResult.from_links_and_vertices(links_dict=all_three, encoder=encoder)
    # 6 links, each C_2=4/3; average = 4/3
    assert plr.get_lattice_electric_energy(average_result=True) == pytest.approx(4.0 / 3.0)


def test_get_lattice_electric_energy_warns_on_none(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed):
    """get_lattice_electric_energy warns when unmeasured links are encountered."""
    encoder = LatticeStateEncoder(
        T1_link_bitmap,
        good_physical_plaquette_states_d_3_2_T1_no_vertex_data_needed,
        LatticeDef(1.5, 2))

    # Only measure one link
    plr = ParsedLatticeResult.from_links_and_vertices(
        links_dict={((0, 0), 1): THREE}, encoder=encoder)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        energy = plr.get_lattice_electric_energy(average_result=False)
        assert len(w) >= 1
        assert "None" in str(w[0].message) or "unmeasured" in str(w[0].message).lower()
