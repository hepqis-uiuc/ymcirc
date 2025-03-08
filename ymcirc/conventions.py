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
