"""
This module is a canonical source for project conventions, including:

- Bit string encodings.
- Magnetic Hamiltonian box term data.

This module loads data for both of these from package-local json files.
The module also provides the class LatticeStateEncoder for converting
link, vertex, and plaquette states to and from bit string encodings.
See the documentation on the class itself for more information.

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

The (lazy-loaded) dict VERTEX_SINGLET_BITMAPS can be indexed with strings to obtain bitmaps
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

########## Magnetic Hamiltonian box term data ##########

Data on the box term of the magnetic Hamiltonian is (lazy) loaded into the dict
HAMILTONIAN_BOX_TERMS. This dict follows an indexing pattern similar
to thar of VERTEX_SINGLET_BITMAPS, where strings of the form

"d=3/2, T1"

are used to access the matrix elements of a particular lattice dimension
in a particular irrep truncation. Once a particular case of dimension and
truncation has been chosed, the actual matrix element data takes the form of a
dictionary whose keys are tuples (final_plaquette_state, initial_plaquette_state)
and whose values are floats. The plaquette state data consists of nested tuples conveying
vertex bag states and link states. As an example,

HAMILTONIAN_BOX_TERMS["d=3/2, T1"] = {
    (((((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (1, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0)), ((((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1), (((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1), (((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1), (((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))): 0.9999999999999994,
    (((((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (1, 0, 0), (1, 0, 0), (0, 0, 0), (1, 1, 0)), ((((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1), (((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1), (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1), (0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 0, 0))): 0.33333333333333304,
    ...

See the definition of HAMILTONIAN_BOX_TERMS below for a complete listing of all available
combinations of dimension and truncation data.
}
"""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Tuple, Dict, Union, List
from ymcirc.utilities import LazyDict, json_dict_loader

# Filesystem stuff.
_PROJECT_ROOT = Path(__file__).parent
_SINGLET_DATA_DIR = _PROJECT_ROOT / "ymcirc_data/singlet-bitmaps/"
_HAMILTONIAN_DATA_DIR = _PROJECT_ROOT / "ymcirc_data/magnetic-hamiltonian-box-term-matrix-elements/"
_SINGLET_DATA_FILE_PATHS: Dict[str, Path] = {
    "d=3/2, T1": _SINGLET_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(3_2)_singlet_bitmaps.json",
    "d=3/2, T2": _SINGLET_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0)]_dim(3_2)_singlet_bitmaps.json",
    "d=2, T1": _SINGLET_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(2)_singlet_bitmaps.json",
    "d=2, T2": _SINGLET_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0)]_dim(2)_singlet_bitmaps.json",
    "d=3, T1": _SINGLET_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(3)_singlet_bitmaps.json",
    "d=3, T2": _SINGLET_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0)]_dim(3)_singlet_bitmaps.json",
    "d=3/2, T1p": _SINGLET_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(3_2)_singlet_prime_bitmaps.json",
    "d=2, T1p": _SINGLET_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(2)_singlet_prime_bitmaps.json"
}
_HAMILTONIAN_DATA_FILE_PATHS: Dict[str, Path] = {
    "d=3/2, T1": _HAMILTONIAN_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(3_2)_magnetic_hamiltonian.json",
    "d=3/2, T2": _HAMILTONIAN_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0)]_dim(3_2)_magnetic_hamiltonian.json",
    "d=2, T1": _HAMILTONIAN_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(2)_magnetic_hamiltonian.json",
    "d=3, T1": _HAMILTONIAN_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(3)_magnetic_hamiltonian.json",
    "d=3/2, T1p": _HAMILTONIAN_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(3_2)_prime_magnetic_hamiltonian.json",
    "d=2, T1p": _HAMILTONIAN_DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(2)_prime_magnetic_hamiltonian.json"
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

# Lazy-load vertex singlet data from precomputed json files.
VERTEX_SINGLET_BITMAPS: LazyDict = LazyDict({
    dim_trunc_case: (json_dict_loader, file_path)
    for dim_trunc_case, file_path in _SINGLET_DATA_FILE_PATHS.items()
})

# This is a special case where it isn't strictly necessary
# to disambiguate singlets via additional vertex data.
# Essentially, with 3 irreps and 3 links per vertex,
# Knowledge of two link irrep states is sufficient to unambiguously
# determine that of the third link.
VERTEX_SINGLET_DICT_D_3HALVES_133BAR_NO_VERTEX_DATA: VertexMultiplicityBitmap = {}

# Lazy-load magnetic Hamiltonian box terms from precomputed json files.
# The following magnetic Hamiltonian box term data is available:
# d=3/2, T1
# d=3/2, T2
# d=2, T1
# d=3, T1
HAMILTONIAN_BOX_TERMS: LazyDict = LazyDict({
    dim_trunc_case: (json_dict_loader, file_path)
    for dim_trunc_case, file_path in _HAMILTONIAN_DATA_FILE_PATHS.items()
})


def load_magnetic_hamiltonian(
        dimensionality_and_truncation_string: str,
        lattice_encoder: LatticeStateEncoder,
        mag_hamiltonian_matrix_element_threshold: float = 0,
        only_include_elems_connected_to_electric_vacuum: bool = False,
        use_2box_hack: bool = False
) -> List[Tuple[str, str, float]]:
    """
    Compute box + box^dagger as a list.

    This is a convenince method to obtain the magnetic Hamiltonian terms in a format
    which facilitates the construction of rotation cicuits.
    The return list consists of tuples whose first two elements
    are encoded plaquette states representing a matrix element of
    box + box dagger. The third element of each tuple is the numerical
    value of the matrix element.

    Necessary arguments:
      - dimensionality_and_truncation_string: a string of the form "d=3/2, T2" which specifies
        the particular matrix elements for a given dimensionality and irrep truncation. See
        the module docstring on ymcirc.conventions for more information.
      - lattice_encoder: A LatticeStateEncoder instance that allows en/decoding lattices states
        as bit strings.

    Optional arguments:
      - mag_hamiltonian_matrix_element_threshold: Only include matrix elements greater than this value.
      - only_include_elems_connected_to_electric_vacuum: Drop any matrix elements which aren't connected to the electric vacuum state.
      - use_2box_hack: Don't include box^dagger, and double the values of each matrix element to compensate.
    """
    mag_hamiltonian: List[Tuple[str, str, float]] = []
    if use_2box_hack is False:
        box_term: List[Tuple[str, str, float]] = []
        box_dagger_term: List[Tuple[str, str, float]] = []
    for (final_plaquette_state, initial_plaquette_state), matrix_element_value in HAMILTONIAN_BOX_TERMS[dimensionality_and_truncation_string].items():
        if abs(matrix_element_value) < mag_hamiltonian_matrix_element_threshold:
            continue
        final_state_bitstring = lattice_encoder.encode_plaquette_state_as_bit_string(final_plaquette_state)
        initial_state_bitstring = lattice_encoder.encode_plaquette_state_as_bit_string(initial_plaquette_state)
        if only_include_elems_connected_to_electric_vacuum and ('1' in final_state_bitstring) and ('1' in initial_state_bitstring):
            continue
        if use_2box_hack is False:
            box_term.append((final_state_bitstring, initial_state_bitstring, matrix_element_value))
            box_dagger_term.append((initial_state_bitstring, final_state_bitstring, matrix_element_value))
        else:
            mag_hamiltonian.append((final_state_bitstring, initial_state_bitstring, 2*matrix_element_value))

    if use_2box_hack is False:
        mag_hamiltonian = box_term + box_dagger_term

    return mag_hamiltonian


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
        # Check bitmaps for uniqueness of bit string encodings.
        vertex_bitmap_has_unique_values = len(set(bit_string for bit_string in vertex_bitmap.values())) \
            == len([bit_string for bit_string in vertex_bitmap.values()])
        link_bitmap_has_unique_values = len(set(bit_string for bit_string in link_bitmap.values())) \
            == len([bit_string for bit_string in link_bitmap.values()])
        if vertex_bitmap_has_unique_values is False:
            raise ValueError("Argument vertex_bitmap must be a dict with unique values. "
                             f"Encountered: {vertex_bitmap}")
        if link_bitmap_has_unique_values is False:
            raise ValueError("Argument link_bitmap must be a dict with unique values. "
                             f"Encountered: {link_bitmap}")

        # Extract expected lengths of various kinds of bit strings.
        try:
            self._expected_vertex_bit_string_length = len(list(vertex_bitmap.values())[0])
        except IndexError:  # Allowed to have no vertices for d=3/2, T1 case.
            self._expected_vertex_bit_string_length = 0
        self._expected_link_bit_string_length = len(list(link_bitmap.values())[0])
        self._expected_plaquette_bit_string_length = (self._expected_vertex_bit_string_length + self._expected_link_bit_string_length) * 4

        # Check bitmaps for consistency of length of bit string encodings.
        if any(len(bit_string) != self._expected_vertex_bit_string_length for bit_string in vertex_bitmap.values()):
            raise ValueError(f"Expecting length {self._expected_vertex_bit_string_length} vertex bit strings. Encountered: {list(vertex_bitmap.values())}.")
        if any(len(bit_string) != self._expected_link_bit_string_length for bit_string in link_bitmap.values()):
            raise ValueError(f"Expecting length {self._expected_link_bit_string_length} link bit strings. Encountered: {list(link_bitmap.values())}.")

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
        return self._link_bitmap[link]

    def decode_bit_string_to_link_state(self, encoded_link: str) -> Union[IrrepWeight, None]:
        """
        Decode a bit string to a linke state i-Weight tuple.

        If the bit string doesn't have a valid decoding, returns None.
        """
        if len(encoded_link) != self.expected_link_bit_string_length:
            raise ValueError(
                f"Tried to decode a length-{len(encoded_link)} bit string. "
                f"Expecting length-{self.expected_link_bit_string_length} bit strings."
            )
        try:
            return self._bit_string_to_link_map[encoded_link]
        except KeyError:
            return None

    def encode_vertex_state_as_bit_string(self, vertex: VertexBag) -> str:
        """Encode a vertex bag state tuple as a bit string."""
        return self._vertex_bitmap[vertex]

    def decode_bit_string_to_vertex_state(self, encoded_vertex: str) -> Union[VertexBag, None]:
        """
        Decode a bit string to a vertex bag state tuple.

        If the bit string doesn't have a valid decoding, returns None.
        """
        if len(encoded_vertex) != self.expected_vertex_bit_string_length:
            raise ValueError(
                f"Tried to decode a length-{len(encoded_vertex)} bit string. "
                f"Expecting length-{self.expected_vertex_bit_string_length} bit strings."
            )
        try:
            return self._bit_string_to_vertex_map[encoded_vertex]
        except KeyError:
            return None

    def encode_plaquette_state_as_bit_string(
            self,
            plaquette: PlaquetteState) -> str:
        """
        Convert plaquette to a bit string encoding.

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
        bit_string_encoding = ""

        if not len(self._vertex_bitmap) == 0:
            for vertex in vertices:
                if not (len(vertex) == 2 and all(isinstance(irrep, tuple) and len(irrep) == 3 for irrep in vertex[0]) and isinstance(vertex[1], int)):
                    raise ValueError("Vertex data must take the form of a length-2 tuple. "
                                     "The first element should be a tuple of irreps, "
                                     "and the second element should be an int indicating "
                                     f"multiplicity.\nEncountered: {vertex}.")
                bit_string_encoding += self._vertex_bitmap[vertex]
        for link in links:
            if not (len(link) == 3 and all(isinstance(elem, int) for elem in link)):
                raise ValueError("Link data must take the form of an SU(3) i-Weight. "
                                 "They should be length-3 tuples of ints. "
                                 f"Encountered:\n{link}.")
            bit_string_encoding += self._link_bitmap[link]

        return bit_string_encoding

    def decode_bit_string_to_plaquette_state(self, bit_string: str) -> PlaquetteState:
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

        If using an empty vertex bitmap, the input bit string is assumed to consist
        exclusively of encoded links, and dummy "decoded" vertices will take the
        value None.

        Any link or vertex bit string which doesn't correspond to a known
        link or vertex state will also be decoded as None.
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
                decoded_vertex = self.decode_bit_string_to_vertex_state(encoded_vertex)
                decoded_plaquette.append(decoded_vertex)
        else:  # Set all decoded vertices to None.
            decoded_plaquette += [None,] * 4
        # Link decoding.
        links_substring = bit_string[idx_first_link_bit:]
        for encoded_link in LatticeStateEncoder._split_string_evenly(
                links_substring, self._expected_link_bit_string_length):
            decoded_link = self.decode_bit_string_to_link_state(encoded_link)
            decoded_plaquette.append(decoded_link)

        return tuple(decoded_plaquette)

    @staticmethod
    def _split_string_evenly(string, split_length) -> List[str]:
        """Helper to break up a string into parts of equal length."""
        return [string[i:i+split_length] for i in range(0, len(string), split_length)]

def _test_singlet_bitmaps():
    print("Testing singlet bitmaps...")
    # Check that there are 8 vertex singlet bitmaps (3 dimensionalities * 2 irrep truncations + T1p in two dimensions).
    there_are_six_singlet_bitmaps = len(VERTEX_SINGLET_BITMAPS) == 8
    print(f"\nlen(VERTEX_SINGLET_BITMAPS) == 8? {there_are_six_singlet_bitmaps}.")
    assert there_are_six_singlet_bitmaps, f"Encountered {len(VERTEX_SINGLET_BITMAPS)} bitmaps."


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
        test_result_singlets_have_unique_bit_string = expected_num_singlets == len(set(VERTEX_SINGLET_BITMAPS[current_case].values()))
        print(f'# of distinct bit string encodings for VERTEX_SINGLET_BITMAPS["{current_case}"] == {expected_num_singlets}? {test_result_singlets_have_unique_bit_string}.')
        if test_result_singlets_have_unique_bit_string is False:
            print(f"Encountered {len(set(VERTEX_SINGLET_BITMAPS[current_case].values()))} distinct bit strings.")

        assert test_result_singlets_have_unique_bit_string


def _test_lattice_encoder_fails_on_bad_bitmaps():
    print("Checking that a vertex bitmap with non-unique bit string values causes ValueError.")
    good_link_bitmap = {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "01"
    }
    non_unique_vertex_bitmap = {
        ((ONE, THREE, THREE), 1): "00",
        ((ONE, THREE, THREE_BAR), 1): "00",
        ((ONE, ONE, ONE), 1): "01"
    }
    try:
        LatticeStateEncoder(good_link_bitmap, non_unique_vertex_bitmap)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("No ValueError raised.")

    print("Checking that a link bitmap with non-unique bit string values causes ValueError.")
    non_unique_link_bitmap = {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "10"
    }
    good_vertex_bitmap = {
        ((ONE, THREE, THREE), 1): "00",
        ((ONE, THREE, THREE_BAR), 1): "01",
        ((ONE, ONE, ONE), 1): "10"
    }
    try:
        LatticeStateEncoder(non_unique_link_bitmap, good_vertex_bitmap)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("No ValueError raised.")

    print("Checking that a vertex bitmap with bit strings of different lengths causes ValueError.")
    vertex_bitmap_different_string_lengths = {
        ((ONE, THREE, THREE), 1): "00",
        ((ONE, THREE, THREE_BAR), 1): "001",
        ((ONE, ONE, ONE), 1): "10"
    }
    try:
        LatticeStateEncoder(good_link_bitmap, vertex_bitmap_different_string_lengths)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("No ValueError raised.")

    print("Checking that a link bitmap with bit strings of different lengths causes ValueError.")
    link_bitmap_different_string_lengths = {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "010"
    }
    try:
        LatticeStateEncoder(link_bitmap_different_string_lengths, good_vertex_bitmap)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("No ValueError raised.")


def _test_encode_decode_various_links():
    link_bitmap = {
        ONE: "00",
        THREE: "10",
        EIGHT: "11"
    }
    # Test data, not physically meaningful but has right format for creating creating an encoder.
    vertex_bitmap = {
        ((ONE, ONE, ONE), 1): "00",
        ((THREE, THREE, THREE), 1): "01",
        ((THREE, THREE, THREE), 2): "10"
    }
    lattice_encoder = LatticeStateEncoder(link_bitmap, vertex_bitmap)

    print("Checking that the following link_bitmap is used to correctly encode/decode links:")
    print(link_bitmap)
    for link_state, bit_string_encoding in link_bitmap.items():
        result_encoding = lattice_encoder.encode_link_state_as_bit_string(link_state)
        result_decoding = lattice_encoder.decode_bit_string_to_link_state(bit_string_encoding)
        assert result_encoding == bit_string_encoding, f"(result != expected): {result_encoding} != {bit_string_encoding}"
        assert result_decoding == link_state, f"(result != expected): {result_decoding} != {link_state}"
        print(f"Test passed. Validated {bit_string_encoding} <-> {link_state}")

    print("Verifying that unknown bit string is decoded to None.")
    assert lattice_encoder.decode_bit_string_to_link_state("01") is None
    print("Test passed.")


def _test_encode_decode_various_vertices():
    link_bitmap = {
        ONE: "00",
        THREE: "10",
        EIGHT: "11"
    }
    # Test data, not physically meaningful but has right format for creating creating an encoder.
    vertex_bitmap = {
        ((ONE, ONE, ONE), 1): "00",
        ((THREE, THREE, THREE), 1): "01",
        ((THREE, THREE, THREE), 2): "10"
    }
    lattice_encoder = LatticeStateEncoder(link_bitmap, vertex_bitmap)

    print("Checking that the following vertex_bitmap is used to correctly encode/decode vertices:")
    print(vertex_bitmap)
    for vertex_state, bit_string_encoding in vertex_bitmap.items():
        result_encoding = lattice_encoder.encode_vertex_state_as_bit_string(vertex_state)
        result_decoding = lattice_encoder.decode_bit_string_to_vertex_state(bit_string_encoding)
        assert result_encoding == bit_string_encoding, f"(result != expected): {result_encoding} != {bit_string_encoding}"
        assert result_decoding == vertex_state, f"(result != expected): {result_decoding} != {vertex_state}"
        print(f"Test passed. Validated {bit_string_encoding} <-> {vertex_state}")

    print("Verifying that unknown bit string is decoded to None.")
    assert lattice_encoder.decode_bit_string_to_vertex_state("11") is None
    print("Test passed.")
    

def _test_encoding_malformed_plaquette_fails():
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
        lattice_encoder.encode_plaquette_state_as_bit_string(plaquette_wrong_length)
    except ValueError as e:
        print(f"Test passed. Raised ValueError:\n{e}")
    else:
        raise AssertionError("Test failed. No ValueError raised.")

    print("Checking that a plaquette which isn't ordered with vertices first "
          "followed by links fails.")
    plaquette_wrong_ordering = (l1, l2, l3, l4, v1, v2, v3, v4)
    try:
        lattice_encoder.encode_plaquette_state_as_bit_string(plaquette_wrong_ordering)
    except ValueError as e:
        print(f"Test passed. Raised ValueError:\n{e}")
    else:
        raise AssertionError("Test failed. No ValueError raised.")

    print("Checking that a plaquette which has too many vertices fails.")
    plaquette_too_many_vertices = (v1, v2, v3, v4, v4, l1, l2, l3)
    try:
        lattice_encoder.encode_plaquette_state_as_bit_string(plaquette_too_many_vertices)
    except ValueError as e:
        print(f"Test passed. Raise ValueError:\n{e}")
    else:
        raise AssertionError("Test failed. No ValueError raised.")


def _test_all_mag_hamiltonian_plaquette_states_have_unique_bit_string_encoding():
    """
    Check that all plaquette states appearing in the magnetic
    Hamiltonian box term can be encoded in unique bit strings.

    Attempts encoding on the following cases:
    - d=3/2, T1
    - d=3/2, T1p
    - d=3/2, T2
    - d=2, T1
    - d=2, T1p
    - d=3, T1
    """
    cases = ["d=3/2, T1", "d=3/2, T1p", "d=3/2, T2", "d=2, T1", "d=2, T1p", "d=3, T1"]
    # Each expected_bitlength = 4 * (n_link_qubits + n_vertex_qubits) for the corresponding case.
    expected_bitlength = [
        4*(2 + 0),
        4*(2 + 0),
        4*(4 + 3),
        4*(2 + 3),
        4*(2 + 1),
        4*(2 + 5)
    ]
    print(
        "Checking that there is a unique bit string encoding available for all "
        "the plaquette states appearing in all the matrix elements for the "
        f"following cases:\n{cases}."
    )
    for current_expected_bitlength, current_case in zip(expected_bitlength, cases):
        dim_string, trunc_string = current_case.split(",")
        dim_string = dim_string.strip()
        trunc_string = trunc_string.strip()
        # Make encoder instance.
        if trunc_string in ["T1", "T1p"]:
            link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR
        elif trunc_string in ["T2"]:
            link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8
        else:
            raise ValueError(f"Unknown irrep truncation: '{trunc_string}'.")
        # No vertices needed for T1 and d=3/2
        vertex_bitmap = {} if current_case in ["d=3/2, T1", "d=3/2, T1p"] else VERTEX_SINGLET_BITMAPS[current_case]
        lattice_encoder = LatticeStateEncoder(link_bitmap, vertex_bitmap)

        print(f"Case {current_case}: confirming all initial and final states "
              "appearing in the magnetic Hamiltonian box term can be succesfully encoded.\n"
              f"Link bitmap: {link_bitmap}\n"
              f"Vertex bitmap: {vertex_bitmap}")

        # Get the set of unique plaquette states.
        all_plaquette_states = set([
            final_and_initial_state_tuple[0] for final_and_initial_state_tuple in HAMILTONIAN_BOX_TERMS[current_case].keys()] + [
                final_and_initial_state_tuple[1] for final_and_initial_state_tuple in HAMILTONIAN_BOX_TERMS[current_case].keys()
            ])

        # Attempt encodings and check for uniqueness.
        all_encoded_plaquette_bit_strings = []
        for plaquette_state in all_plaquette_states:
            plaquette_state_bit_string = lattice_encoder.encode_plaquette_state_as_bit_string(plaquette_state)
            assert len(plaquette_state_bit_string) == current_expected_bitlength, f"len(plaquette_state_bit_string) == {len(plaquette_state_bit_string)}; expected len == {current_expected_bitlength}."
            all_encoded_plaquette_bit_strings.append(plaquette_state_bit_string)
        n_unique_plaquette_encodings = len(set(all_encoded_plaquette_bit_strings))
        assert n_unique_plaquette_encodings == len(all_plaquette_states), f"Encountered {n_unique_plaquette_encodings} unique bit strings encoding {len(all_plaquette_states)} unique plaquette states."

        print("Test passed.")


def _test_encoding_good_plaquette():
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
    expected_bit_string = "01010100" + "10010000"  # vertices encoding + link encoding
    lattice_encoder = LatticeStateEncoder(IRREP_TRUNCATION_DICT_1_3_3BAR, VERTEX_SINGLET_BITMAPS["d=3/2, T1"])
    print(
        "Checking that that the following plaquette is encoded in the bit "
        f"string {expected_bit_string}:\n"
        f"{plaquette}"
    )
    resulting_bit_string = lattice_encoder.encode_plaquette_state_as_bit_string(plaquette)
    assert expected_bit_string == resulting_bit_string, f"Test failed, resulting_bit_string == {resulting_bit_string}."
    print("Test passed.")


def _test_bit_string_decoding_to_plaquette():
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
            (
                None, None, None, None,  # When using empty vertex bitmap, should get back None for decoded vertices.
                THREE, THREE, THREE, THREE_BAR
            )
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
        resulting_decoded_plaquette = lattice_encoder.decode_bit_string_to_plaquette_state(encoded_plaquette)
        assert resulting_decoded_plaquette == expected_decoded_plaquette, f"Expected: {expected_decoded_plaquette}\nEncountered: {resulting_decoded_plaquette}"
        print(f"Test passed.\n{encoded_plaquette} successfully decoded to {resulting_decoded_plaquette}.")


def _test_decoding_non_gauge_invariant_bit_string():
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
    decoded_plaquette = lattice_encoder.decode_bit_string_to_plaquette_state(encoded_plaquette)

    assert decoded_plaquette == expected_plaquette_broken_gauge_invariance, f"Expected: {expected_plaquette_broken_gauge_invariance}\nEncountered: {decoded_plaquette}"
    print(f"Test passed.\n{encoded_plaquette} successfully decoded to {expected_plaquette_broken_gauge_invariance}.")


def _test_decoding_garbage_bit_strings_result_in_none():
    # This should map to something since it's possible for noise to result in such states.
    link_bitmap = {
        ONE: "000",
        THREE: "100",
        EIGHT: "101",
        SIX: "111",
        SIX_BAR: "010"
    }
    # Test data, not physically meaningful but has right format for creating creating an encoder.
    vertex_bitmap = {
        ((ONE, ONE, ONE, ONE), 1): "000",
        ((ONE, ONE, ONE, THREE), 1): "001",
        ((THREE, THREE, THREE, EIGHT), 1): "010",
        ((THREE, THREE, THREE, EIGHT), 2): "100",
        ((SIX, EIGHT, EIGHT, SIX_BAR), 1): "110"
    }
    lattice_encoder = LatticeStateEncoder(link_bitmap, vertex_bitmap)

    # Link, vertex, and plaquette test data.
    bad_encoded_link = "001"
    bad_encoded_vertex = "111"
    encoded_plaquette_some_links_and_vertices_good_others_bad = \
        "000" + "000" + bad_encoded_vertex + "110" + "000" + "000" + bad_encoded_link + "100"
    expected_decoded_plaquette_some_links_and_vertices_good_others_bad = (
         ((ONE, ONE, ONE, ONE), 1),  ((ONE, ONE, ONE, ONE), 1), None, ((SIX, EIGHT, EIGHT, SIX_BAR), 1),
        ONE, ONE, None, THREE
    )

    print(f"Checking {bad_encoded_link} decodes to None using the link bitmap: {link_bitmap}")
    assert lattice_encoder.decode_bit_string_to_link_state("001") is None
    print("Test passed.")

    print(f"Checking {bad_encoded_vertex} decodes to None using the link bitmap: {vertex_bitmap}")
    assert lattice_encoder.decode_bit_string_to_vertex_state("111") is None
    print("Test passed.")

    print(f"Checking {encoded_plaquette_some_links_and_vertices_good_others_bad} decodes to the plaquette:\n {expected_decoded_plaquette_some_links_and_vertices_good_others_bad}")
    decoded_plaquette = lattice_encoder.decode_bit_string_to_plaquette_state(encoded_plaquette_some_links_and_vertices_good_others_bad)
    assert decoded_plaquette  == expected_decoded_plaquette_some_links_and_vertices_good_others_bad, f"(decoded != expected): {decoded_plaquette}\n!=\n{expected_decoded_plaquette_some_links_and_vertices_good_others_bad}"
    print("Test passed.")


def _test_decoding_fails_when_len_bit_string_doesnt_match_bitmaps():
    # This should map to something since it's possible for noise to result in such states.
    link_bitmap = {
        ONE: "000",
        THREE: "100",
        EIGHT: "101",
        SIX: "111",
        SIX_BAR: "010"
    }
    # Test data, not physically meaningful but has right format for creating creating an encoder.
    vertex_bitmap = {
        ((ONE, ONE, ONE, ONE), 1): "0000",
        ((ONE, ONE, ONE, THREE), 1): "0001",
        ((THREE, THREE, THREE, EIGHT), 1): "0010",
        ((THREE, THREE, THREE, EIGHT), 2): "1000",
        ((SIX, EIGHT, EIGHT, SIX_BAR), 1): "1100"
    }
    lattice_encoder = LatticeStateEncoder(link_bitmap, vertex_bitmap)

    print("Testing that decoding links/vertices/plaquettes fails with wrong length bit string.")
    print(f"Using link bitmap: {link_bitmap}")
    print(f"Using vertex bitmap: {vertex_bitmap}")

    # Set up test data.
    expected_link_bit_string_length = len(list(link_bitmap.keys())[0])
    expected_vertex_bit_string_length = len(list(vertex_bitmap.keys())[0])
    expected_plaquette_bit_string_length = 4 * (expected_link_bit_string_length + expected_vertex_bit_string_length)
    bad_length_link_bit_string = "10"
    bad_length_vertex_bit_string = "101"
    bad_length_plaquette_bit_string = "00000000000000010000100011101"
    assert len(bad_length_link_bit_string) != expected_link_bit_string_length
    assert len(bad_length_vertex_bit_string) != expected_vertex_bit_string_length
    assert len(bad_length_plaquette_bit_string) != expected_plaquette_bit_string_length

    print(f"Checking link bit string {bad_length_link_bit_string} fails to decode.")
    try:
        lattice_encoder.decode_bit_string_to_link_state(bad_length_link_bit_string)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("ValueError not raised.")

    print(f"Checking vertex bit string {bad_length_vertex_bit_string} fails to decode.")
    try:
        lattice_encoder.decode_bit_string_to_vertex_state(bad_length_vertex_bit_string)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("ValueError not raised.")

    print(f"Checking plaquette bit string {bad_length_plaquette_bit_string} fails to decode.")
    try:
        lattice_encoder.decode_bit_string_to_plaquette_state(bad_length_plaquette_bit_string)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("ValueError not raised.")


def _run_tests():
    _test_singlet_bitmaps()
    print()
    _test_vertex_bitmaps_have_right_amount_of_singlets()
    print()
    _test_lattice_encoder_fails_on_bad_bitmaps()
    print()
    _test_encode_decode_various_links()
    print()
    _test_encode_decode_various_vertices()
    print()
    _test_encoding_malformed_plaquette_fails()
    print()
    _test_encoding_good_plaquette()
    print()
    _test_all_mag_hamiltonian_plaquette_states_have_unique_bit_string_encoding()
    print()
    _test_bit_string_decoding_to_plaquette()
    print()
    _test_decoding_non_gauge_invariant_bit_string()
    print()
    _test_decoding_garbage_bit_strings_result_in_none()
    print()
    _test_decoding_fails_when_len_bit_string_doesnt_match_bitmaps()
    print()


if __name__ == "__main__":
    _run_tests()

    print("All tests passed.")
