"""
This module provides tools for working with magnetic Hamiltonian matrix elements.

It is a work in progress.
"""
import ast
from lattice_tools.conventions import (VertexBag, IrrepWeight, VertexMultiplicityBitmap,
                                       IrrepBitmap, ONE, THREE, THREE_BAR, SIX, SIX_BAR, EIGHT,
                                       VERTEX_SINGLET_BITMAPS, IRREP_TRUNCATION_DICT_1_3_3BAR,
                                       IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8)
from pathlib import Path
from typing import Tuple
import json
from typing import Dict

# Filesystem stuff.
_PROJECT_ROOT = Path(__file__).parent
_DATA_DIR = _PROJECT_ROOT / "lattice_tools_data/magnetic-hamiltonian-matrix-elements/"
_HAMILTONIAN_DATA_FILE_PATHS: Dict[str, Path] = {
    "d=3/2, T1": _DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(3_2)_magnetic_hamiltonian.json",
    "d=3/2, T2": _DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0)]_dim(3_2)_magnetic_hamiltonian.json",
    "d=2, T1": _DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(2)_magnetic_hamiltonian.json"
}

# TODO some kind of class hierarchy for loading plaquette data + doing validation?
# For now, just a type alias as a placeholder.
VertexState = VertexBag
LinkState = IrrepWeight
PlaquetteState = Tuple[
    VertexState, VertexState, VertexState, VertexState,
    LinkState, LinkState, LinkState, LinkState]

# Load the magnetic Hamiltonian data from precomputed json files.
# Safer to use ast.literal_eval than eval to convert data keys to tuples.
# The latter can execute arbitrary potentially malicious code while
# the worst case attack vector for litera_eval would be to crash
# the python process.
# See https://docs.python.org/3/library/ast.html#ast.literal_eval
# for more information.
MAGNETIC_HAMILTONIANS: Dict[str, Dict[PlaquetteState, float]] = {}
for dim_trunc_case, file_path in _HAMILTONIAN_DATA_FILE_PATHS.items():
    with file_path.open('r') as json_data:
        d = json.load(json_data)
        MAGNETIC_HAMILTONIANS[dim_trunc_case] = {ast.literal_eval(key): value for key, value in d.items()}


def encode_plaquette_state_as_bitstring(
        plaquette: PlaquetteState,
        link_bitmap: IrrepBitmap,
        vertex_bitmap: VertexMultiplicityBitmap
) -> str:
    """
    Convert plaquette to a bitstring encoding.

    Assumes len(plaquette) = 8, where the first four elements are vertex bag states
    and the last four elements are the link states. Assumes the ordering convention:

    |v1 v2 v3 v4 l1 l2 l3 l4>

    according to the layout:

    v4 ----l3--- v3
    |            |
    |            |
    l4           l2
    |            |
    |            |
    v1 ----l1--- v2

    If vertex_bitmap is empty, it is assumed that vertex degrees of freedom
    are redundant, and they are treated as the "empty" string. Operationally,
    this translates to encoding

    |l1 l2 l3 l4>

    in a bitstring instead.
    """
    if len(plaquette) != 8:
        raise ValueError(
            "Plaquette states should be a tuple of four vertices and four links. "
            f"Encountered:\nlen(plaquette) = {len(plaquette)}.")

    vertices = [item for item in plaquette[:4]]
    links = [item for item in plaquette[4:]]
    bitstring_encoding = ""

    if not len(vertex_bitmap) == 0:
        for vertex in vertices:
            if not (len(vertex) == 2 and all(isinstance(irrep, tuple) and len(irrep) == 3 for irrep in vertex[0]) and isinstance(vertex[1], int)):
                raise ValueError("Vertex data must take the form of a length-2 tuple. "
                                 "The first element should be a tuple of irreps, "
                                 "and the second element should be an int indicating "
                                 f"multiplicity.\nEncountered: {vertex}.")
            bitstring_encoding += vertex_bitmap[vertex]
    for link in links:
        if not (len(link) == 3 and all(isinstance(elem, int) for elem in link)):
            raise ValueError("Link data must take the form of an SU(3) i-Weight. "
                             "They should be length-3 tuples of ints. "
                             f"Encountered:\n{link}.")
        bitstring_encoding += link_bitmap[link]

    return bitstring_encoding


def decode_bitstring_to_plaquette_state(
        bitstring: str,
        link_bitmap: IrrepBitmap,
        vertex_bitmap: VertexMultiplicityBitmap
) -> PlaquetteState:
    """
    Decode bitstring to a plaquette state in terms of iWeights.

    State ordering convetion starts at bottom left vertex and goes

    |v1 v2 v3 v4 l1 l2 l3 l4>

    according to the layout:

    v4 ----l3--- v3
    |            |
    |            |
    l4           l2
    |            |
    |            |
    v1 ----l1--- v2
    """
    raise NotImplementedError()

# Tests
def _test_bitstring_encoding_of_plaquette():
    v1: VertexBag = ((ONE, THREE, THREE_BAR), 1)
    v2: VertexBag = ((ONE, THREE, THREE_BAR), 1)
    v3: VertexBag = ((ONE, THREE, THREE_BAR), 1)
    v4: VertexBag = ((ONE, ONE, ONE), 1)
    l1: IrrepWeight = THREE
    l2: IrrepWeight = THREE_BAR
    l3: IrrepWeight = ONE
    l4: IrrepWeight = ONE
    plaquette: PlaquetteState = (v1, v2, v3, v4, l1, l2, l3, l4)
    expected_bitstring = "01010100" + "10010000"  # vertices encoding + link encoding
    print(
        "Checking that that the plaquette:\n"
        f"{plaquette}"
        f"is encoded in the bitstring {expected_bitstring}."
    )
    resulting_bitstring = encode_plaquette_state_as_bitstring(
        plaquette,
        link_bitmap=IRREP_TRUNCATION_DICT_1_3_3BAR,
        vertex_bitmap=VERTEX_SINGLET_BITMAPS["d=3/2, T1"])
    assert expected_bitstring == resulting_bitstring, f"Test failed, resulting_bitstring == {resulting_bitstring}."
    print("Test passed.")


def _test_bad_plaquette_input_fails():
    v1: VertexBag = ((ONE, THREE, THREE_BAR), 1)
    v2: VertexBag = ((ONE, THREE, THREE_BAR), 1)
    v3: VertexBag = ((ONE, THREE, THREE_BAR), 1)
    v4: VertexBag = ((ONE, ONE, ONE), 1)
    l1: IrrepWeight = THREE
    l2: IrrepWeight = THREE_BAR
    l3: IrrepWeight = ONE
    l4: IrrepWeight = ONE

    print("Checking that a wrong-length plaquette input fails to be encoded.")
    plaquette_wrong_length: PlaquetteState = (v1, v2, v3, l1, l2, l3)
    try:
        encode_plaquette_state_as_bitstring(
            plaquette_wrong_length,
            link_bitmap=IRREP_TRUNCATION_DICT_1_3_3BAR,
            vertex_bitmap=VERTEX_SINGLET_BITMAPS["d=3/2, T1"])
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("Test failed. No ValueError raised.")

    print("Checking that a plaquette which isn't ordered with vertices first followed by links fails.")
    plaquette_wrong_ordering = (l1, l2, l3, l4, v1, v2, v3, v4)
    try:
        encode_plaquette_state_as_bitstring(
            plaquette_wrong_ordering,
            link_bitmap=IRREP_TRUNCATION_DICT_1_3_3BAR,
            vertex_bitmap=VERTEX_SINGLET_BITMAPS["d=3/2, T1"])
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("Test failed. No ValueError raised.")

    print("Checking that a plaquette which has too many vertices fails.")
    plaquette_too_many_vertices = (v1, v2, v3, v4, v4, l1, l2, l3)
    try:
        encode_plaquette_state_as_bitstring(
            plaquette_too_many_vertices,
            link_bitmap=IRREP_TRUNCATION_DICT_1_3_3BAR,
            vertex_bitmap=VERTEX_SINGLET_BITMAPS["d=3/2, T1"])
    except ValueError as e:
        print(f"Test passed. Raise ValueError: {e}")
    else:
        raise AssertionError("Test failed. No ValueError raised.")


def _test_all_plaquette_states_have_unique_bitstring_encoding():
    """
    Check that data loaded from conventions.py can be encoded in unique bitstrings.

    Attempts encoding on the following cases:
    - d=3/2, T1
    - d=3/2, T2
    - d=2, T1
    """
    cases = ["d=3/2, T1", "d=3/2, T2", "d=2, T1"]
    # expected_bitlength = 4 * (n_link_qubits + n_vertex_qubits)
    expected_bitlength = [4*(2 + 0) , 4*(4 + 3), 4*(3 + 2)]
    print(
        "Checking that there is a unique bitstring encoding available for all the plaquette states "
        f"appearing in all the matrix elements for the following cases:\n{cases}."
    )
    for current_expected_bitlength, current_case in zip(expected_bitlength, cases):
        link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR if \
                current_case[-2:] == "T1" else IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8
        # No vertices needed for T1 and d=3/2
        vertex_bitmap = {} if current_case[-2:] == "T1" \
                and current_case[:5] == "d=3/2" else VERTEX_SINGLET_BITMAPS[current_case]
        print(f"Case {current_case}: confirming all initial and final states appearing the "
              "magnetic Hamiltonian can be succesfully encoded.\n"
              f"Link bitmap: {link_bitmap}\n"
              f"Vertex bitmap: {vertex_bitmap}")

        # Get the set of unique plaquette states.
        all_plaquette_states = set([
            final_and_initial_state_tuple[0] for final_and_initial_state_tuple in MAGNETIC_HAMILTONIANS[current_case].keys()] + [
                final_and_initial_state_tuple[1] for final_and_initial_state_tuple in MAGNETIC_HAMILTONIANS[current_case].keys()
            ])
        all_encoded_plaquette_bitstrings = []
        # Unpack keys containing both final and initial states.
        #all_plaquette_states = set(key[0], key[1] for key in MAGNETIC_HAMILTONIANS[current_case].keys())
        #print("Here are the states:", all_plaquette_states)
        #assert False
        for plaquette_state in all_plaquette_states:
            plaquette_state_bitstring = encode_plaquette_state_as_bitstring(
                plaquette_state,
                link_bitmap,
                vertex_bitmap
            )
            assert len(plaquette_state_bitstring) == current_expected_bitlength, f"len(plaquette_state_bitstring) == {len(plaquette_state_bitstring)}; expected len == {current_expected_bitlength}."
            # assert len(final_state_bitstring) == current_expected_bitlength, f"len(final_state_bitstring) == {len(final_state_bitstring)}; expected len == {current_expected_bitlength}."
            # assert len(initial_state_bitstring) == current_expected_bitlength, f"len(initial_state_bitstring) == {len(initial_state_bitstring)}; expected len == {current_expected_bitlength}."
            # all_encoded_plaquette_bitstrings.append(final_state_bitstring)
            # all_encoded_plaquette_bitstrings.append(initial_state_bitstring)
            all_encoded_plaquette_bitstrings.append(plaquette_state_bitstring)

        n_unique_plaquette_encodings = len(set(all_encoded_plaquette_bitstrings))
        assert n_unique_plaquette_encodings == len(all_plaquette_states), f"Encountered {n_unique_plaquette_encodings} unique bitstrings encoding {len(all_plaquette_states)} unique plaquette states."
        print("Test passed.")


def _run_tests():
    print("Checking that conventions.py contains expected data.")
    # Check that conventions are as expected.
    assert ONE == (0, 0, 0)
    assert THREE == (1, 0, 0)
    assert THREE_BAR == (1, 1, 0)
    assert SIX == (2, 0, 0)
    assert SIX_BAR == (2, 2, 0)
    assert EIGHT == (2, 1, 0)
    assert IRREP_TRUNCATION_DICT_1_3_3BAR == {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "01"
    }
    assert VERTEX_SINGLET_BITMAPS["d=3/2, T1"] == {
        ((ONE, ONE, ONE), 1): "00",
        ((ONE, THREE, THREE_BAR), 1): "01",
        ((THREE, THREE, THREE), 1): "10",
        ((THREE_BAR, THREE_BAR, THREE_BAR), 1): "11"
    }
    print("Confirmed conventions.py is as expected.")

    _test_bitstring_encoding_of_plaquette()
    _test_bad_plaquette_input_fails()
    _test_all_plaquette_states_have_unique_bitstring_encoding()

    print("All tests passed.")


if __name__ == "__main__":
    #print(MAGNETIC_HAMILTONIANS)
    
    _run_tests()
