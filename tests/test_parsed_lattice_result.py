from typing import List
from ymcirc._abstract import LatticeDef
from ymcirc.parsed_lattice_result import ParsedLatticeResult
from ymcirc.conventions import LatticeStateEncoder, PlaquetteState, ONE, THREE, THREE_BAR, SIX, SIX_BAR, EIGHT


def test_parse_d_3_2_lattice_no_vertex_data():
    """Check creation of parsed lattice measurement d=3/2 when no vertex data present."""
    # Prepare test data.
    link_bitmap = {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "01"
    }
    physical_plaquette_states: List[PlaquetteState] = [
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


def test_parse_d_3_2_lattice_with_vertex_data():
    raise NotImplementedError("Test not yet written.")


# TODO test a global_bitstring method and compare it to reconstructing by iterating over the lattice.

# TODO test d = 2 and maybe d = 3.

# TODO test hashability of class.

# TODO some kind of test for a method that returns unphysical vertices/plaquettes? Both the Nones, and the inconsistent ones where we don't get a singlet or the right singlet.

# TODO a few tests that try throwing in bad input to make sure there are no silent failure. (such as getting a plaquette off a top vertex in d=3/2?)
