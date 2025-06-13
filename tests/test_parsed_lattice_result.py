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
