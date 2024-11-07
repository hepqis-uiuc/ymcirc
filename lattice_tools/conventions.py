"""
This module is a canonical source for project conventions, including:

- Bit string encodings.
- Magnetic Hamiltonian data.

########## Irrep link state bit string encodings ##########

There are two encodings of irrep link states:

- IRREP_TRUNCATION_DICT_1_3_3BAR
- IRREP_TRUNCATION_DICT_1_3_3BAR_6_6_BAR_8

These are dictionaries which map length-3 tuples to unique bit strings.
The tuples represent "i-Weights", which are a way of uniquely labeling
SU(N) irreps which come from working with the Gelfand-Tsetlin pattern calculus.
See Arne et al for more details (https://doi.org/10.1063/1.3521562).
As a convenience, the following constants are defined which take on the correct
i-Weights values:

- ONE = (0, 0, 0)
- THREE = (1, 0, 0)
- THREE_BAR = (1, 1, 0)
- SIX = (2, 0, 0)
- SIX_BAR = (2, 2, 0)
- EIGHT = (2, 1, 0)

There are generically "leftover" states when encoding a particular truncation
into a set of qubits (i.e. the state space of a link in a particular truncation
is usually not of dimension 2^n). These are omitted from the irrep truncation
dicts.

The "bag states" at vertices encode the information needed to establish which
singlet state the vertex is in. This is ambiguous in general because there are
multiple ways to create singlets from link assignments in arbitrary dimension,
and for arbitrary irrep truncations, there will be complete link assignments
which yield various "multiplicites" of singlets.

########## Vertex singlet bit string encodings ##########

The dict VERTEX_SINGLET_BITMAPS can be indexed with strings to obtain bitmaps
for the following cases:

- "d=3/2, T1"
- "d=3/2, T2"
- "d=2, T1"
- "d=2, T2"
- "d=3, T1"
- "d=3, T2"

T1 refers to the ONE, THREE, THREE_BAR truncation, while T2 includes the
additional states SIX, SIX_BAR, and EIGHT. To get the bitmap for a particular
case, use the following syntax:

VERTEX_SINGLET_BITMAPS["d=2, T2"]

Since the key in this dict is a string, spacing and capitalization is
necessary. Once a particular bitmap has been accessed, that will
yield a dict containing keys which are tuples of iweights with a
multiplicity index, and a bit string. For example:

VERTEX_SINGLET_BITMAPS["d=3/2, T1"] = {
    (((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1): '00',
    (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1): '01',
    (((1, 0, 0), (1, 0, 0), (1, 0, 0)), 1): '10',
    (((1, 1, 0), (1, 1, 0), (1, 1, 0)), 1): '11'
}

The integer after the tuple of i-Weights tells us that a vertex with these
three irreps on one of the links will have only one singlet. Compare this
with the following case:

VERTEX_SINGLET_BITMAPS["d=2, T1"] = {
    (((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), 1): '000',
    (((0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 1, 0)), 1): '001',
    (((0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)), 1): '010',
    (((0, 0, 0), (1, 1, 0), (1, 1, 0), (1, 1, 0)), 1): '011',
    (((1, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0)), 1): '100',
    (((1, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0)), 2): '101'
}

Observe that there are two multiplicities of the
((1, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0)) set of irreps coming into a
particular vertex; these get assigned to distinct bit strings.

Finally, it is a special quirk of d=3/2, T1 that it is possible to fully deduce
the singlet state with knowledge of only two links. Therefore, the following
bitmap is provided since vertex degrees of freedom are unnecessary in that
case:

VERTEX_SINGLET_DICT_D_3HALVES_133BAR_NO_VERTEX_DATA = {}

########## Convenience methods for encoding/decoding ##########

The following methods are provided to facilitate encoding plaquette
states into bit strings, and decoding bit strings into plaquettes:

encode_plaquette_state_as_bitstring
decode_bitstring_to_plaquette_state

See the documentation on those specific methods for more information
about their usage.

########## Magnetic Hamiltonian data ##########

Magnetic Hamiltonian data is loaded into the dict
MAGNETIC_HAMILTONIANS. This dict follows an indexing pattern similar
to thar of VERTEX_SINGLET_BITMAPS, where strings of the form

"d=3/2, T1"

are used to access the matrix elements of a particular lattice dimension
in a particular irrep truncation. Once a particular case of dimension and
truncation has been chosed, the actual matrix element data takes the form of a
dictionary whose keys are tuples (final_plaquette_state, initial_plaquette_state)
and whose values are floats. The plaquette state data consists of nested tuples conveying
vertex bag states and link states. As an example,

MAGNETIC_HAMILTONIANS["d=3/2, T1"] = {
    (((((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (1, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0)), ((((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1), (((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1), (((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1), (((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))): 0.9999999999999994,
    (((((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (1, 0, 0), (1, 0, 0), (0, 0, 0), (1, 1, 0)), ((((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1), (((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1), (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 0, 0))): 0.33333333333333304,
    ...

See the definition of MAGNETIC_HAMILTONIANS below for a complete listing of all available
combinations of dimension and truncation data.
}
"""
from __future__ import annotations
import ast
import copy
from pathlib import Path
from typing import Tuple, Dict, Union
import json

# Filesystem stuff.
_PROJECT_ROOT = Path(__file__).parent
_SINGLET_DATA_DIR = _PROJECT_ROOT / "lattice_tools_data/singlet-bitmaps/"
_HAMILTONIAN_DATA_DIR = _PROJECT_ROOT / "lattice_tools_data/magnetic-hamiltonian-matrix-elements/"
_SINGLET_DATA_FILE_PATHS: Dict[str, Path] = {
    "d=3/2, T1": _SINGLET_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(3_2)_singlet_bitmaps.json",
    "d=3/2, T2": _SINGLET_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0)]_dim(3_2)_singlet_bitmaps.json",
    "d=2, T1": _SINGLET_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(2)_singlet_bitmaps.json",
    "d=2, T2": _SINGLET_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0)]_dim(2)_singlet_bitmaps.json",
    "d=3, T1": _SINGLET_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(3)_singlet_bitmaps.json",
    "d=3, T2": _SINGLET_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0)]_dim(3)_singlet_bitmaps.json",
}
_HAMILTONIAN_DATA_FILE_PATHS: Dict[str, Path] = {
    "d=3/2, T1": _HAMILTONIAN_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(3_2)_magnetic_hamiltonian.json",
    "d=3/2, T2": _HAMILTONIAN_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0)]_dim(3_2)_magnetic_hamiltonian.json",
    "d=2, T1": _HAMILTONIAN_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(2)_magnetic_hamiltonian.json"
}

# Useful type aliases.
IrrepWeight = Tuple[int, int, int]
LinkState = IrrepWeight  # A useful semantic alias
BitString = str
MultiplicityIndex = int
IrrepBitmap = Dict[IrrepWeight, BitString]
SingletsDef = Tuple[Tuple[IrrepWeight, ...], Tuple[MultiplicityIndex, ...]]
VertexBag = Tuple[Tuple[IrrepWeight, ...], MultiplicityIndex]
VertexState = VertexBag  # Another useful semantic alias
VertexMultiplicityBitmap = Dict[VertexBag, BitString]
PlaquetteState = Union[
    Tuple[
        VertexState, VertexState, VertexState, VertexState,
        LinkState, LinkState, LinkState, LinkState],
    Tuple[
        None, None, None, None,
        LinkState, LinkState, LinkState, LinkState]
]

# Irrep iweights (top row of GT pattern).
ONE: IrrepWeight = (0, 0, 0)
THREE: IrrepWeight = (1, 0, 0)
THREE_BAR: IrrepWeight = (1, 1, 0)
SIX: IrrepWeight = (2, 0, 0)
SIX_BAR: IrrepWeight = (2, 2, 0)
EIGHT: IrrepWeight = (2, 1, 0)

# Irrep encoding bitmaps.
IRREP_TRUNCATION_DICT_1_3_3BAR: IrrepBitmap = {
    ONE: "00",
    THREE: "10",
    THREE_BAR: "01"
}
IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8: IrrepBitmap = {
    ONE: "000",
    THREE: "100",
    THREE_BAR: "001",
    SIX: "110",
    SIX_BAR: "011",
    EIGHT: "111"
}

# Load vertex singlet data from precomputed json files.
# Safer to use ast.literal_eval than eval to convert data keys to tuples.
# The latter can execute arbitrary potentially malicious code while
# the worst case attack vector for literal_eval would be to crash
# the python process.
# See https://docs.python.org/3/library/ast.html#ast.literal_eval
# for more information.
VERTEX_SINGLET_BITMAPS: Dict[str, VertexMultiplicityBitmap] = {}
# Load singlet bag state encoding bitmaps.
for dim_trunc_case, file_path in _SINGLET_DATA_FILE_PATHS.items():
    with file_path.open('r') as json_data:
        d = json.load(json_data)
        VERTEX_SINGLET_BITMAPS[dim_trunc_case] = {ast.literal_eval(key): value for key, value in d.items()}  

# This is a special case where it isn't strictly necessary
# to disambiguate singlets via additional vertex data.
# Essentially, with 3 irreps and 3 links per vertex,
# Knowledge of two link irrep states is sufficient to unambiguously
# determine that of the third link.
VERTEX_SINGLET_DICT_D_3HALVES_133BAR_NO_VERTEX_DATA: VertexMultiplicityBitmap = {}

# The following magnetic Hamiltonian data is available:
# d=3/2, T1
# d=3/2, T2
# d=2, T1
MAGNETIC_HAMILTONIANS: Dict[str, Dict[PlaquetteState, float]] = {}
for dim_trunc_case, file_path in _HAMILTONIAN_DATA_FILE_PATHS.items():
    with file_path.open('r') as json_data:
        d = json.load(json_data)
        MAGNETIC_HAMILTONIANS[dim_trunc_case] = {ast.literal_eval(key): value for key, value in d.items()}


class LatticeStateEncoder:
    """
    Class for encoding/decode lattice state data.

    Provides mapping functionality between the physical states of links,
    vertices, and plaquettes into bit strings, and vice versa.
    """

    def __init__(
            self,
            link_bitmap: IrrepBitmap,
            vertex_bitmap: VertexMultiplicityBitmap):
        """
        Create a LatticeStateEncoder using the provided bitmaps.

        Inputs:
        - link_bitmap: a dictionary of link iWeights mapping to bit strings.
        - vertex_bitmap: a dictionary of vertex bags with a multiplicity index
                         mapping to bit strings.

        Bitmaps are assumed to have unique bit strings as values,
        and the desired link iWeight tuple or vertex bag tuple
        as keys. Inverting of the bitmaps is handled internally when decoding.

        ValueError raised if there are inconsistent length bit strings among the values
        of the bitmap dictionaries.
        """
        # TODO: Validate the bitmaps.

        # Extract expected lengths of various kinds of bit strings.
        try:
            self._expected_vertex_bit_string_length = len(list(vertex_bitmap.values())[0])
        except IndexError:  # Allowed to have no vertices for d=3/2, T1 case.
            self._expected_vertex_bit_string_length = 0
        self._expected_link_bit_string_length = len(list(link_bitmap.values())[0])
        self._expected_plaquette_bit_string_length = (self._expected_vertex_bit_string_length + self._expected_link_bit_string_length) * 4

        # Set the internal bitmaps.
        self._link_bitmap = copy.deepcopy(link_bitmap)
        self._vertex_bitmap = copy.deepcopy(vertex_bitmap)
        self._bit_string_to_vertex_map = {bit_string: vertex for vertex, bit_string in vertex_bitmap.items()}
        self._bit_string_to_link_map = {bit_string: link for link, bit_string in link_bitmap.items()}

    @property
    def vertex_bitmap(self) -> VertexMultiplicityBitmap:
        """Return a copy of the vertex bitmap in use."""
        return copy.deepcopy(self._vertex_bitmap)

    @property
    def link_bitmap(self) -> IrrepBitmap:
        """Return a copy of the link_bitmap in use."""
        return copy.deepcopy(self._link_bitmap)

    @property
    def expected_plaquette_bit_string_length(self) -> int:
        """Return the length of plaquette bit strings the encoder expects."""
        return self._expected_plaquette_bit_string_length

    @property
    def expected_link_bit_string_length(self) -> int:
        """Return the length of link bit strings the encoder expects."""
        return self._expected_link_bit_string_length

    @property
    def expected_vertex_bit_string_length(self) -> int:
        """Return the length of vertex bit strings the encoder expects."""
        return self._expected_vertex_bit_string_length

    def encode_link_state_as_bit_string(self, link: IrrepWeight) -> str:
        """Encode an i-Weight tuple as a bit string."""
        raise NotImplementedError()

    def decode_link_state_as_bit_string(self, encoded_link: str) -> Union[IrrepWeight, None]:
        """
        Decode a bit string to an i-Weight tuple.

        If the bit string doesn't have a valid decoding, returns None.
        """
        raise NotImplementedError()

    def encode_vertex_state_as_bitstring(self, vertex: VertexBag) -> str:
        """Encode a vertex bag as a bit string."""
        raise NotImplementedError()

    def decode_vertex_state_as_bitstring(self, encoded_vertex: str) -> Union[VertexBag, None]:
        """
        Decode a bit string to a vertex bag tuple.

        If the bit string doesn't have a valid decoding, returns None.
        """
        raise NotImplementedError()

    def encode_plaquette_state_as_bitstring(
            self,
            plaquette: PlaquetteState) -> str:
        """
        Convert plaquette to a bitstring encoding.

        Assumes len(plaquette) = 8, where the first four elements are vertex
        bag states and the last four elements are the link states. Assumes the
        ordering convention:

        |v1 v2 v3 v4 l1 l2 l3 l4>

        according to the layout:

        v4 ----l3--- v3
        |            |
        |            |
        l4           l2
        |            |
        |            |
        v1 ----l1--- v2

        If self.vertex_bitmap is empty, it is assumed that vertex degrees of freedom
        are redundant, and they are treated as the "empty" string.
        Operationally, this translates to encoding

        |l1 l2 l3 l4>

        in a bit string instead.
        """
        if len(plaquette) != 8:
            raise ValueError(
                "Plaquette states should be a tuple of four vertices and four"
                f" links. Encountered:\nlen(plaquette) = {len(plaquette)}.")

        vertices = [item for item in plaquette[:4]]
        links = [item for item in plaquette[4:]]
        bitstring_encoding = ""

        if not len(self._vertex_bitmap) == 0:
            for vertex in vertices:
                if not (len(vertex) == 2 and all(isinstance(irrep, tuple) and len(irrep) == 3 for irrep in vertex[0]) and isinstance(vertex[1], int)):
                    raise ValueError("Vertex data must take the form of a length-2 tuple. "
                                     "The first element should be a tuple of irreps, "
                                     "and the second element should be an int indicating "
                                     f"multiplicity.\nEncountered: {vertex}.")
                bitstring_encoding += self._vertex_bitmap[vertex]
        for link in links:
            if not (len(link) == 3 and all(isinstance(elem, int) for elem in link)):
                raise ValueError("Link data must take the form of an SU(3) i-Weight. "
                                 "They should be length-3 tuples of ints. "
                                 f"Encountered:\n{link}.")
            bitstring_encoding += self._link_bitmap[link]

        return bitstring_encoding

    def decode_bitstring_to_plaquette_state(self, bit_string: str) -> PlaquetteState:
        """
        Decode bit string to a plaquette state in terms of iWeights.

        State ordering convention starts at bottom left vertex and goes

        |v1 v2 v3 v4 l1 l2 l3 l4>

        according to the layout:

        v4 ----l3--- v3
        |            |
        |            |
        l4           l2
        |            |
        |            |
        v1 ----l1--- v2

        If using an empty vertex bitmap, the input bitstring is assumed to consist
        exclusively of encoded links, and dummy "decoded" vertices will take the
        value None.
        """
        if self._expected_plaquette_bit_string_length != len(bit_string):
            raise ValueError("Vertex and link bitmaps are inconsistent with length of\n"
                             f"bit string {bit_string}. Expected n_bits: {self._expected_plaquette_bit_string_length}; encountered n_bits: {len(bit_string)}.")
        idx_first_link_bit = 4 * self._expected_vertex_bit_string_length
        vertices_substring = bit_string[:idx_first_link_bit]
        links_substring = bit_string[idx_first_link_bit:]

        # Decode plaquette.
        decoded_plaquette = []
        # Vertex decoding.
        if self._expected_vertex_bit_string_length != 0:
            for encoded_vertex in LatticeStateEncoder._split_string_evenly(
                    vertices_substring, self._expected_vertex_bit_string_length):
                decoded_vertex = self._bit_string_to_vertex_map[encoded_vertex]
                decoded_plaquette.append(decoded_vertex)
        # Link decoding.
        links_substring = bit_string[idx_first_link_bit:]
        for encoded_link in LatticeStateEncoder._split_string_evenly(
                links_substring, self._expected_link_bit_string_length):
            decoded_link = self._bit_string_to_link_map[encoded_link]
            decoded_plaquette.append(decoded_link)

        return tuple(decoded_plaquette)

    @staticmethod
    def _split_string_evenly(string, split_length) -> List[str]:
        """Helper to break up a string into parts of equal length."""
        return [string[i:i+split_length] for i in range(0, len(string), split_length)]

def _test_singlet_bitmaps():
    print("Testing singlet bitmaps...")
    # Check that there are 6 vertex singlet bitmaps (3 dimensionalities * 2 irrep truncations).
    there_are_six_singlet_bitmaps = len(VERTEX_SINGLET_BITMAPS) == 6
    print(f"\nlen(VERTEX_SINGLET_BITMAPS) == 6? {there_are_six_singlet_bitmaps}.")
    assert there_are_six_singlet_bitmaps


def _test_vertex_bitmaps_have_right_amount_of_singlets():
    # Check that all vertex bitmaps have the right amount of distinct singlets.
    # The values for expected_nums come from classical precomputation.
    cases = ["d=3/2, T1", "d=3/2, T2", "d=2, T1", "d=2, T2", "d=3, T1", "d=3, T2"]
    expected_nums = [4, 16, 6, 66, 28, 1646]
    for current_case, expected_num_singlets in zip(cases, expected_nums):
        # Number check.
        test_result_num_singlets = len(VERTEX_SINGLET_BITMAPS[current_case]) == expected_num_singlets
        print(f'\nlen(VERTEX_SINGLET_BITMAPS["{current_case}"]) == {expected_num_singlets}? {test_result_num_singlets}.')
        if test_result_num_singlets is False:
            print(f"Encountered {len(VERTEX_SINGLET_BITMAPS[current_case])} singlets.")

        assert test_result_num_singlets

        # Uniqueness of encoding check.
        test_result_singlets_have_unique_bitstring = expected_num_singlets == len(set(VERTEX_SINGLET_BITMAPS[current_case].values()))
        print(f'# of distinct bit string encodings for VERTEX_SINGLET_BITMAPS["{current_case}"] == {expected_num_singlets}? {test_result_singlets_have_unique_bitstring}.')
        if test_result_singlets_have_unique_bitstring is False:
            print(f"Encountered {len(set(VERTEX_SINGLET_BITMAPS[current_case].values()))} distinct bit strings.")

        assert test_result_singlets_have_unique_bitstring


def _test_bad_plaquette_input_fails():
    v1: VertexBag = ((ONE, THREE, THREE_BAR), 1)
    v2: VertexBag = ((ONE, THREE, THREE_BAR), 1)
    v3: VertexBag = ((ONE, THREE, THREE_BAR), 1)
    v4: VertexBag = ((ONE, ONE, ONE), 1)
    l1: IrrepWeight = THREE
    l2: IrrepWeight = THREE_BAR
    l3: IrrepWeight = ONE
    l4: IrrepWeight = ONE

    lattice_encoder = LatticeStateEncoder(
        link_bitmap=IRREP_TRUNCATION_DICT_1_3_3BAR,
        vertex_bitmap=VERTEX_SINGLET_BITMAPS["d=3/2, T1"])

    print("Checking that a wrong-length plaquette input fails to be encoded.")
    plaquette_wrong_length: PlaquetteState = (v1, v2, v3, l1, l2, l3)
    try:
        lattice_encoder.encode_plaquette_state_as_bitstring(plaquette_wrong_length)
    except ValueError as e:
        print(f"Test passed. Raised ValueError:\n{e}")
    else:
        raise AssertionError("Test failed. No ValueError raised.")

    print("Checking that a plaquette which isn't ordered with vertices first "
          "followed by links fails.")
    plaquette_wrong_ordering = (l1, l2, l3, l4, v1, v2, v3, v4)
    try:
        lattice_encoder.encode_plaquette_state_as_bitstring(plaquette_wrong_ordering)
    except ValueError as e:
        print(f"Test passed. Raised ValueError:\n{e}")
    else:
        raise AssertionError("Test failed. No ValueError raised.")

    print("Checking that a plaquette which has too many vertices fails.")
    plaquette_too_many_vertices = (v1, v2, v3, v4, v4, l1, l2, l3)
    try:
        lattice_encoder.encode_plaquette_state_as_bitstring(plaquette_too_many_vertices)
    except ValueError as e:
        print(f"Test passed. Raise ValueError:\n{e}")
    else:
        raise AssertionError("Test failed. No ValueError raised.")


def _test_all_plaquette_states_have_unique_bitstring_encoding():
    """
    Check that all plaquette states appearing in the magnetic
    Hamiltonian can be encoded in unique bit strings.

    Attempts encoding on the following cases:
    - d=3/2, T1
    - d=3/2, T2
    - d=2, T1
    """
    cases = ["d=3/2, T1", "d=3/2, T2", "d=2, T1"]
    # expected_bitlength = 4 * (n_link_qubits + n_vertex_qubits)
    expected_bitlength = [4*(2 + 0) , 4*(4 + 3), 4*(3 + 2)]
    print(
        "Checking that there is a unique bit string encoding available for all "
        "the plaquette states appearing in all the matrix elements for the "
        f"following cases:\n{cases}."
    )
    for current_expected_bitlength, current_case in zip(expected_bitlength, cases):
        # Make encoder instance.
        link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR if \
                current_case[-2:] == "T1" else IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8
        # No vertices needed for T1 and d=3/2
        vertex_bitmap = {} if current_case[-2:] == "T1" \
                and current_case[:5] == "d=3/2" else VERTEX_SINGLET_BITMAPS[current_case]
        lattice_encoder = LatticeStateEncoder(link_bitmap, vertex_bitmap)
        
        print(f"Case {current_case}: confirming all initial and final states "
              "appearing in the magnetic Hamiltonian can be succesfully encoded.\n"
              f"Link bitmap: {link_bitmap}\n"
              f"Vertex bitmap: {vertex_bitmap}")

        # Get the set of unique plaquette states.
        all_plaquette_states = set([
            final_and_initial_state_tuple[0] for final_and_initial_state_tuple in MAGNETIC_HAMILTONIANS[current_case].keys()] + [
                final_and_initial_state_tuple[1] for final_and_initial_state_tuple in MAGNETIC_HAMILTONIANS[current_case].keys()
            ])

        # Attempt encodings and check for uniqueness.
        all_encoded_plaquette_bitstrings = []
        for plaquette_state in all_plaquette_states:
            plaquette_state_bitstring = lattice_encoder.encode_plaquette_state_as_bitstring(plaquette_state)
            assert len(plaquette_state_bitstring) == current_expected_bitlength, f"len(plaquette_state_bitstring) == {len(plaquette_state_bitstring)}; expected len == {current_expected_bitlength}."
            all_encoded_plaquette_bitstrings.append(plaquette_state_bitstring)
        n_unique_plaquette_encodings = len(set(all_encoded_plaquette_bitstrings))
        assert n_unique_plaquette_encodings == len(all_plaquette_states), f"Encountered {n_unique_plaquette_encodings} unique bit strings encoding {len(all_plaquette_states)} unique plaquette states."

        print("Test passed.")

def _test_bitstring_encoding_of_plaquette():
    # Check that the encoding is as expected.
    assert VERTEX_SINGLET_BITMAPS["d=3/2, T1"] == {
        ((ONE, ONE, ONE), 1): "00",
        ((ONE, THREE, THREE_BAR), 1): "01",
        ((THREE, THREE, THREE), 1): "10",
        ((THREE_BAR, THREE_BAR, THREE_BAR), 1): "11"
    }

    # Create test data and a LatticeEncoder.
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
    lattice_encoder = LatticeStateEncoder(IRREP_TRUNCATION_DICT_1_3_3BAR, VERTEX_SINGLET_BITMAPS["d=3/2, T1"])
    print(
        "Checking that that the following plaquette is encoded in the bit "
        f"string {expected_bitstring}:\n"
        f"{plaquette}"
    )
    resulting_bitstring = lattice_encoder.encode_plaquette_state_as_bitstring(plaquette)
    assert expected_bitstring == resulting_bitstring, f"Test failed, resulting_bitstring == {resulting_bitstring}."
    print("Test passed.")


def _test_bitstring_decoding_to_plaquette():
    # Check that decoding of bit strings is as expected.
    # Case data tuple format:
    # case_name, encoded_plaquette, vertex_bitmap, link_bitmap, expected_decoded_plaquette.
    # Note that the data were manually constructed by irrep encoding bitmaps
    # with data in vertex singlet json files.
    cases = [
        (
            "d=3/2, T1",
            "10101001",
            {},  # No vertex data needed  for d=3/2, T1.
            IRREP_TRUNCATION_DICT_1_3_3BAR,
            (THREE, THREE, THREE, THREE_BAR)
        ),
        (
            "d=3/2, T2",
            "1001" + "1100" + "0011" + "0001" + "110111000001",  # Vertex strings + link string
            VERTEX_SINGLET_BITMAPS["d=3/2, T2"],
            IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8,
            (
                ((THREE_BAR, THREE_BAR, SIX), 1),
                ((SIX, EIGHT, SIX_BAR), 1),
                ((ONE, EIGHT, EIGHT), 1),
                ((ONE, THREE, THREE_BAR), 1),
                SIX, EIGHT, ONE, THREE_BAR
            )
        ),
        (
            "d=2, T1",
            "101" + "100" + "010" + "001" + "10101001",  # Vertex strings + link string
            VERTEX_SINGLET_BITMAPS["d=2, T1"],
            IRREP_TRUNCATION_DICT_1_3_3BAR,
            (
                ((THREE, THREE, THREE_BAR, THREE_BAR), 2),
                ((THREE, THREE, THREE_BAR, THREE_BAR), 1),
                ((ONE, THREE, THREE, THREE), 1),
                ((ONE, ONE, THREE, THREE_BAR), 1),
                THREE, THREE, THREE, THREE_BAR
            )
        )
    ]

    print("Checking decoding of bit strings corresponding to gauge-invariant plaquette states.")

    for current_case, encoded_plaquette, vertex_bitmap, link_bitmap, expected_decoded_plaquette in cases:
        print(f"Checking plaquette bit string decoding for a {current_case} plaquette...")
        lattice_encoder = LatticeStateEncoder(link_bitmap, vertex_bitmap)
        resulting_decoded_plaquette = lattice_encoder.decode_bitstring_to_plaquette_state(encoded_plaquette)
        assert resulting_decoded_plaquette == expected_decoded_plaquette, f"Expected: {expected_decoded_plaquette}\nEncountered: {resulting_decoded_plaquette}"
        print(f"Test passed.\n{encoded_plaquette} successfully decoded to {resulting_decoded_plaquette}.")


def _test_decoding_non_gauge_invariant_bitstring():
    print("Checking decoding of a bit string corresponding to a non-gauge-invariant plaquette state.")
    # Configure test data and encoder.
    lattice_encoder = LatticeStateEncoder(
        IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8,
        VERTEX_SINGLET_BITMAPS["d=3/2, T2"])
    # This should still succeed since it's possible for noise to result in such states.
    # In d=3/2, T2, it is unphysical to have all ONE link states with vertex bags all
    # in EIGHTs. But physical qubit errors could causes such a measurement outcome.
    expected_plaquette_broken_gauge_invariance = (
        ((EIGHT, EIGHT, EIGHT), 1),
        ((EIGHT, EIGHT, EIGHT), 1),
        ((EIGHT, EIGHT, EIGHT), 1),
        ((EIGHT, EIGHT, EIGHT), 1),
        ONE, ONE, ONE, ONE
    )
    encoded_plaquette = "1101" + "1101" + "1101" + "1101" + "000000000000" # Vertex strings + link string
    decoded_plaquette = lattice_encoder.decode_bitstring_to_plaquette_state(encoded_plaquette)

    assert decoded_plaquette == expected_plaquette_broken_gauge_invariance, f"Expected: {expected_plaquette_broken_gauge_invariance}\nEncountered: {decoded_plaquette}"
    print(f"Test passed.\n{encoded_plaquette} successfully decoded to {expected_plaquette_broken_gauge_invariance}.")


def _test_decoding_garbage_bitstring():
    # This should map to something since it's possible for noise to result in such states.
    raise NotImplementedError()


def _test_decoding_fails_when_len_bitstring_doesnt_match_bitmaps():
    raise NotImplementedError()


def _test_decoding_fails_when_bitmaps_have_inconsistent_lengths():
    raise NotImplementedError()


def _run_tests():
    _test_singlet_bitmaps()
    print()
    _test_vertex_bitmaps_have_right_amount_of_singlets()
    print()
    _test_bad_plaquette_input_fails()
    print()
    _test_bitstring_encoding_of_plaquette()
    print()
    _test_all_plaquette_states_have_unique_bitstring_encoding()
    print()
    _test_bitstring_decoding_to_plaquette()
    print()
    _test_decoding_non_gauge_invariant_bitstring()
    print()
    _test_decoding_garbage_bitstring()
    print()
    _test_decoding_fails_when_len_bitstring_doesnt_match_bitmaps()
    print()
    _test_decoding_fails_when_bitmaps_have_inconsistent_lengths()
    print()


if __name__ == "__main__":
    _run_tests()

    print("All tests passed.")
