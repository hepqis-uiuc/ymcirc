"""
This module is a canonical source for project conventions, including:

- Bit string encodings.
- Magnetic Hamiltonian box term data.

This module loads data for both of these from package-local json files.
The module also provides the class LatticeStateEncoder for converting
link, vertex multiplicity, and plaquette states to and from bit string encodings.
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

There is also generically a multiplicity index associated with each lattice vertex
denoting which singlet state the vertex is in. The multiplicity index is needed because
there are generically multiple ways to create singlets from link assignments in arbitrary dimension,
and for arbitrary irrep truncations, there will be complete link assignments
which yield various "multiplicites" of singlets.

########## Physical plaquette states, and singlet multiplicities ##########

The (lazy-loaded) dict PHYSICAL_PLAQUETTE_STATES consists of all the single-plaquette
gauge-invariant states in a particular lattice geometry and truncation. The dict
contains data for the following cases

- "d=3/2, T1"
- "d=3/2, T2"
- "d=2, T1"

T1 refers to the ONE, THREE, THREE_BAR truncation, while T2 includes the
additional states SIX, SIX_BAR, and EIGHT. To get the physical states for a particular
case, use the following syntax:

PHYSICAL_PLAQUETTE_STATES["d=2, T2"]

Since the key in this dict is a string, spacing and capitalization is
necessary. The entries in the physical states dict consist of lists of all the
single-plaquete gauge-invariant states in that particular dimensionality and truncation.
The plaquette data takes the form a tuple of tuples:

plaquette = (
    vertex_multiplicities,
    a_links,
    c_links
)

The first two elements of plaquette are length-4 tuples consisting of multiplicity integers
(zero-indexed) and the four "active" link i-Weights for the plaquette state. For example:

vertex_multiplicities = (0, 0, 0, 2)
a_links = (ONE, ONE, THREE, THREE)

The final element of a plaquette is a tuple of i-Weights whose length depends on the
dimensionality of the lattice. These i-Weights give the states of the "control" links
to the plaquette, which are defined as all links connected to a vertex which aren't an
active link. The number of controls per-vertex is 2*(d - 1). We use a convention where
control links are ordered counter-clockwise according to vertex.

In d=3/2:

c_links = (c1, c2, c3, c4)

 c4 ---- v4 ----l3--- v3 ---- c3
         |            |
         |            |
         l4           l2
         |            |
         |            |
 c1 ---- v1 ----l1--- v2 ---- c2

In d=2:

c_links = (c1, c2, c3, c4, c5, c6, c7, c8)

         c7           c6
         |            |
         |            |
         |            |
 c8 ---- v4 ----l3--- v3 ---- c5
         |            |
         |            |
         l4           l2
         |            |
         |            |
 c1 ---- v1 ----l1--- v2 ---- c4
         |            |
         |            |
         |            |
         c2           c3

In d=3, c1 through c8 match d=2. c9 through c12 are the four links "above" the plaquette as
determined by the right-hand rule, ordered counter-clockwise starting from v1. c13 through c16
are similarly the four links "below" the plaquette.

Finally, it is a special quirk of d=3/2, T1 that there are no nontrivial singlet multiplicities,
and those data can be ignored.

########## Magnetic Hamiltonian box term data ##########

Data on the box term of the magnetic Hamiltonian is (lazy) loaded into the dict
HAMILTONIAN_BOX_TERMS. This dict follows an indexing pattern similar
to that of VERTEX_SINGLET_BITMAPS, where strings of the form

"d=3/2, T1"

are used to access the matrix elements of a particular lattice dimension
in a particular irrep truncation. Once a particular case of dimension and
truncation has been chosed, the actual matrix element data takes the form of a
dictionary whose keys are tuples (final_plaquette_state, initial_plaquette_state)
and whose values are floats. The plaquette state data consists of nested tuples conveying
vertex bag states and link states. As an example,

HAMILTONIAN_BOX_TERMS["d=3/2, T1"] = {
    (plaq_1, plaq_2): 0.9999999999999994,
    (plaq_3, plaq_4): 0.33333333333333304,
    ...
}
with plaq_1 through plaq_4 taking the form described above for plaquette states.
See the definition of HAMILTONIAN_BOX_TERMS below for a complete listing of all available
combinations of dimension and truncation data.

For small, periodic lattices, the same physical lattice link can be a control on
distinct vertices in the same plaquette. For example, a periodic d=3/2 lattice
with size 2 (2 plaquettes) will have c1 == c2 and c3 == c4. When circuits are constructed,
all plaquette states and magnetic Hamiltonian terms for which the control links do
not satisfy these equality constraints will be discarded. This logic is handled in
the ymcirc.circuit module.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from typing import Tuple, Dict, Union, List
from ymcirc._abstract import LatticeDef, Plaquette
from ymcirc.utilities import LazyDict, json_loader

# Filesystem stuff.
_PROJECT_ROOT = Path(__file__).parent
_HAMILTONIAN_DATA_DIR = _PROJECT_ROOT / "_ymcirc_data/magnetic-hamiltonian-box-term-matrix-elements/"
_PLAQUETTE_STATES_DATA_DIR = _PROJECT_ROOT / "_ymcirc_data/plaquette-states/"
_HAMILTONIAN_DATA_FILE_PATHS: Dict[str, Path] = {
    "d=3/2, T1": _HAMILTONIAN_DATA_DIR / "T1_dim(3_2)_magnetic_hamiltonian.json",
    "d=3/2, T2": _HAMILTONIAN_DATA_DIR / "T2_dim(3_2)_magnetic_hamiltonian.json",
    "d=2, T1": _HAMILTONIAN_DATA_DIR / "T1_dim(2)_magnetic_hamiltonian.json",
}
_PLAQUETTE_STATES_DATA_FILE_PATHS: Dict[str, Path] = {
    "d=3/2, T1": _PLAQUETTE_STATES_DATA_DIR / "T1_dim(3_2)_plaquette_states.json",
    "d=3/2, T2": _PLAQUETTE_STATES_DATA_DIR / "T2_dim(3_2)_plaquette_states.json",
    "d=2, T1": _PLAQUETTE_STATES_DATA_DIR / "T1_dim(2)_plaquette_states.json",
}

# Useful type aliases.
IrrepWeight = Tuple[int, int, int]
LinkState = IrrepWeight  # A useful semantic alias
BitString = str
MultiplicityIndex = int
IrrepBitmap = Dict[IrrepWeight, BitString]
VertexMultiplicityBitmap = Dict[MultiplicityIndex, BitString]
# Tuple of 4 vertex multiplicites, tuple of 4 "active links", tuple of arbitrary num "control links".
PlaquetteState = Union[
    Tuple[
        Tuple[MultiplicityIndex, MultiplicityIndex, MultiplicityIndex, MultiplicityIndex],
        Tuple[LinkState, LinkState, LinkState, LinkState],
        Tuple[LinkState, ...]
    ],
    Tuple[
        Tuple[None, None, None, None],
        Tuple[LinkState, LinkState, LinkState, LinkState],
        Tuple[LinkState, ...]]
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

# Lazy-load vertex physical plaquette states from precomputed json files.
# Entries has the format s, a, c,
# where s is a tuple of 4 site multiplicity indices,
# a is a tuple of 4 "active link" iweights,
# and c is a variable-length tuple of "control_link" iweights.
PHYSICAL_PLAQUETTE_STATES: LazyDict = LazyDict({
    dim_trunc_case: (json_loader, file_path)
    for dim_trunc_case, file_path in _PLAQUETTE_STATES_DATA_FILE_PATHS.items()
})

# Lazy-load magnetic Hamiltonian box terms from precomputed json files.
# The following magnetic Hamiltonian box term data is available:
# d=3/2, T1
# d=3/2, T2
# d=2, T1
HAMILTONIAN_BOX_TERMS: LazyDict = LazyDict({
    dim_trunc_case: (json_loader, file_path)
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
            physical_plaquette_states: List[PlaquetteState],
            lattice: LatticeDef
    ):
        """
        Create a LatticeStateEncoder for states on lattice using link_bitmap and list of physical plaquette states.

        Inputs:
        - link_bitmap: a dictionary of link iWeights mapping to bit strings.
        - physical_plaquette_states: a list of physical plaquette states.
        - lattice: a LatticeDef or child class. This provides information
            about the size of the lattice, boundary conditions, dimensionality, etc.

        The link bitmap is assumed to have unique bit strings as values,
        and the desired link iWeight tuple as keys as keys. A multiplicity bitmap is
        automatically constructed from physical_plaquette_states by examining the largest
        vertex multiplicity integer n appearing among the physical states, and allocating
        a length l = floor(log2(n) + 1) bitstring to store the binary encoding of each integer.
        Inverting of the bitmaps is handled internally when decoding.

        ValueError raised if there are inconsistent length bit strings among the values
        of the link bitmap dictionary, or there are duplicate entries in the
        list of physical plaquette states.
        """
        # Validation of input.
        # Check link bitmaps for uniqueness of bit string encodings.
        # Also check physical states for uniqueness (no dupes allowed).
        link_bitmap_has_unique_values = len(set(bit_string for bit_string in link_bitmap.values())) \
            == len([bit_string for bit_string in link_bitmap.values()])
        plaquette_states_are_unique = len(physical_plaquette_states) == len(set(physical_plaquette_states))
        consistent_num_controls_in_plaquette_states = all([len(plaquette[2]) == len(physical_plaquette_states[0][2]) for plaquette in physical_plaquette_states])
        if link_bitmap_has_unique_values is False:
            raise ValueError("Argument link_bitmap must be a dict with unique values. "
                             f"Encountered: {link_bitmap}")
        if plaquette_states_are_unique is False:
            raise ValueError("Argument physical_plaquette_states must be a list"
                             f" with unique values. Encountered: {physical_plaquette_states}.")
        if not isinstance(lattice, LatticeDef):
            raise TypeError(f"The lattice argument must be an instance of {LatticeDef.__name__}. Received: {type(lattice)}.")
        if consistent_num_controls_in_plaquette_states is not True:
            raise ValueError("All physical plaquette states must have the same number of controls.")

        # Now construct the vertex multiplicity bitmap, and infer qubit counts per DoFs.
        max_zero_indexed_multiplicity = max([current_vertex for vertices, a_links, c_links in physical_plaquette_states for current_vertex in vertices])
        if max_zero_indexed_multiplicity == 0:
            # All multiplicities in the lattice are trivial. No need to track vertex DoFs.
            self._expected_vertex_bit_string_length = 0
            vertex_bitmap = {}
        else:
            self._expected_vertex_bit_string_length = int(np.floor(np.log2(max_zero_indexed_multiplicity) + 1))
            zero_pad_int_to_bin = lambda some_int, total_len: f"{bin(some_int)[2:]:0>{total_len}s}"
            vertex_bitmap = {
                multiplicity_index: f"{zero_pad_int_to_bin(multiplicity_index, self._expected_vertex_bit_string_length)}"
                for multiplicity_index in range(max_zero_indexed_multiplicity + 1)
            }
        self._expected_link_bit_string_length = len(list(link_bitmap.values())[0])
        n_total_control_links = len(physical_plaquette_states[0][2])
        if not n_total_control_links == lattice.n_control_links_per_plaquette:
            raise ValueError(f"Expected {lattice.n_control_links_per_plaquette} total plaquette controls from lattice. Encountered a plaquette states with {n_total_control_links} total controls.")
        self._expected_plaquette_bit_string_length = \
            (4 * (self._expected_vertex_bit_string_length + self._expected_link_bit_string_length)) \
            + (n_total_control_links * self._expected_link_bit_string_length)

        # Check all bitmaps for consistency of length of bit string encodings.
        if any(len(bit_string) != self._expected_vertex_bit_string_length for bit_string in vertex_bitmap.values()):
            raise ValueError(f"Expecting length {self._expected_vertex_bit_string_length} vertex bit strings. Encountered: {list(vertex_bitmap.values())}.")
        if any(len(bit_string) != self._expected_link_bit_string_length for bit_string in link_bitmap.values()):
            raise ValueError(f"Expecting length {self._expected_link_bit_string_length} link bit strings. Encountered: {list(link_bitmap.values())}.")

        # Set the internal bitmaps now that they're validated.
        self._link_bitmap = copy.deepcopy(link_bitmap)
        self._vertex_bitmap = vertex_bitmap
        self._bit_string_to_vertex_map = {bit_string: vertex for vertex, bit_string in vertex_bitmap.items()}
        self._bit_string_to_link_map = {bit_string: link for link, bit_string in link_bitmap.items()}
        # TODO The following conditional is a placeholder to deal with LatticeDef not yet supporting size tuples, remove eventually.
        if lattice.dim == 1.5:
            self._lattice = LatticeDef(lattice.dim, lattice.shape[0], lattice.periodic_boundary_conds)  # Make a new instance to avoid mutating user data!
        elif not all([axis_length == lattice.shape[0] for axis_length in lattice.shape]):  
            raise NotImplementedError("Lattices with different lengths along different dimensions not yet supported.")
        else:
            self._lattice = LatticeDef(lattice.dim, lattice.shape[0], lattice.periodic_boundary_conds)  # Make a new instance to avoid mutating user data!

    @property
    def vertex_bitmap(self) -> VertexMultiplicityBitmap:
        """Return a copy of the vertex multiplicity bitmap in use."""
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

    @property
    def lattice_def(self) -> LatticeDef:
        """Return copy of the LatticeDef instance used to create the encoder instance."""
        return copy.deepcopy(self._lattice)

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

    def encode_vertex_state_as_bit_string(self, vertex: MultiplicityIndex) -> str:
        """Encode a vertex multiplicity integer as a bit string."""
        return self._vertex_bitmap[vertex]

    def decode_bit_string_to_vertex_state(self, encoded_vertex: str) -> Union[VertexBag, None]:
        """
        Decode a bit string to a vertex multiplicity index.

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
            plaquette: PlaquetteState,
            override_n_c_links_validation: bool = False
    ) -> str:
        """
        Convert plaquette to a bit string encoding.

        Plaquette states take the form of a length-3 tuple. The first element of
        this tuple is a length-4 tuple of integers representing multiplicity indices
        associated to each vertex. The second element is a length-4 tuple of
        i-weights which label irreps on the "active" links of the plaquette.
        The third element is a tuple of i-weights labelling the irreps on "control"
        links to the plaquettes, where the number of controls depends on the lattice
        geometry and boundary conditions.

        Assumes the ordering convention:

        |v1 v2 v3 v4 l1 l2 l3 l4 c1 c2 c3 c4>

        according to the layout:

        v4 ----l3--- v3
        |            |
        |            |
        l4           l2
        |            |
        |            |
        v1 ----l1--- v2

        where cj is shorthand for the list of controls connected to vertex vj.

        If self.vertex_bitmap is empty, it is assumed that vertex degrees of freedom
        are redundant, and they are treated as the "empty" string.
        Operationally, this translates to encoding

        |l1 l2 l3 l4 controls>

        in a bit string instead.

        Finally, if the argument override_n_c_links_validation is True, then no validation on the total
        number of control links will be performed. This is useful (for example) when working on small
        lattices with periodic boundary conditions, where the same physical control link can be
        connected to more than one vertex in a plaquette.
        """
        if len(plaquette) != 3:
            raise ValueError(
                "Plaquette states should be a tuple of three tuples corresponding to vertices, active links, and control links."
                f" Encountered a plaquette with len(plaquette) = {len(plaquette)}.")

        vertices = plaquette[0]
        a_links = plaquette[1]
        c_links = plaquette[2]
        if len(vertices) != 4:
            raise ValueError(f"Encountered {len(vertices)} vertex multiplicities instead of 4.")
        if len(a_links) != 4:
            raise ValueError(f"Encountered {len(a_links)} active links instead of 4.")
        if len(c_links) != self._lattice.n_control_links_per_plaquette and (override_n_c_links_validation is False):
            raise ValueError(f"Encountered {len(c_links)} control links instead of {self._lattice.n_control_links_per_plaquette}.")
        bit_string_encoding = ""

        if not len(self._vertex_bitmap) == 0:
            for multiplicity in vertices:
                if not (isinstance(multiplicity, int) and multiplicity >= 0):
                    raise ValueError(f"Encountered multiplicity index '{multiplicity}' Should be a nonnegative integer")
                bit_string_encoding += self._vertex_bitmap[multiplicity]
        for a_link in a_links:
            if not (len(a_link) == 3 and all(isinstance(elem, int) for elem in a_link)):
                raise ValueError("Link data must take the form of an SU(3) i-Weight. "
                                 "They should be length-3 tuples of ints. "
                                 f"Encountered:\n{a_link}.")
            bit_string_encoding += self._link_bitmap[a_link]
        for c_link in c_links:
            if not (len(c_link) == 3 and all(isinstance(elem, int) for elem in c_link)):
                raise ValueError("Link data must take the form of an SU(3) i-Weight. "
                                 "They should be length-3 tuples of ints. "
                                 f"Encountered:\n{c_link}.")
            bit_string_encoding += self._link_bitmap[c_link]

        return bit_string_encoding

    def decode_bit_string_to_plaquette_state(self, bit_string: str) -> PlaquetteState:
        """
        Decode bit string to a plaquette state in terms of iWeights.

        State ordering convention starts at bottom left vertex and goes

        |v1 v2 v3 v4 l1 l2 l3 l4 controls>

        according to the layout:

        v4 ----l3--- v3
        |            |
        |            |
        l4           l2
        |            |
        |            |
        v1 ----l1--- v2

        The controls are ordered according to those attached to v1, v2, etc. If unable to decode
        to physical state data, returns None for that degree of freedom.
        """
        # Validate input.
        if self._expected_plaquette_bit_string_length != len(bit_string):
            raise ValueError("Vertex and link bitmaps are inconsistent with length of\n"
                             f"bit string {bit_string}. Expected n_bits: {self._expected_plaquette_bit_string_length}; encountered n_bits: {len(bit_string)}.")

        # Parse input into vertex, active link, and control link substrings.
        idx_first_a_link_bit = 4 * self._expected_vertex_bit_string_length
        idx_first_c_link_bit = idx_first_a_link_bit + (4 * self._expected_link_bit_string_length)
        vertices_substring = bit_string[:idx_first_a_link_bit]
        a_links_substring = bit_string[idx_first_a_link_bit:idx_first_c_link_bit]
        c_links_substring = bit_string[idx_first_c_link_bit:]

        # Decode plaquette.
        decoded_plaquette = []
        decoded_vertices = tuple(
            self.decode_bit_string_to_vertex_state(encoded_vertex) for encoded_vertex in
            LatticeStateEncoder._split_string_evenly(
                vertices_substring, self._expected_vertex_bit_string_length)
        ) if len(vertices_substring) > 0 else (None,) * 4
        decoded_a_links = tuple(
            self.decode_bit_string_to_link_state(encoded_link) for encoded_link in
            LatticeStateEncoder._split_string_evenly(
                a_links_substring, self._expected_link_bit_string_length)
        )
        decoded_c_links = tuple(
            self.decode_bit_string_to_link_state(encoded_link) for encoded_link in
            LatticeStateEncoder._split_string_evenly(
                c_links_substring, self._expected_link_bit_string_length)
        )
        decoded_plaquette = (
            decoded_vertices,
            decoded_a_links,
            decoded_c_links
        )
        return decoded_plaquette

    @staticmethod
    def _split_string_evenly(string, split_length) -> List[str]:
        """Break up a string into parts of equal length."""
        return [string[i:i+split_length] for i in range(0, len(string), split_length)]


def _test_no_duplicate_physical_plaquette_states():
    print("Checking that none of the physical plaquette states data contain duplicates.")
    for dim_trunc_case in PHYSICAL_PLAQUETTE_STATES.keys():
        print(f"Checking {dim_trunc_case}...")
        num_duplicates = len(PHYSICAL_PLAQUETTE_STATES[dim_trunc_case]) \
            - len(set(PHYSICAL_PLAQUETTE_STATES[dim_trunc_case]))
        has_no_duplicates = num_duplicates == 0
        assert has_no_duplicates, f"Detected {num_duplicates} duplicate entries."
        print("Test passed.")


def _test_no_duplicate_matrix_elements():
    print("Checking that none of the matrix element data contain duplicates.")
    for dim_trunc_case in HAMILTONIAN_BOX_TERMS.keys():
        print(f"Checking {dim_trunc_case}...")
        # list of tuples (final state, initial state) that index matrix elements.
        state_indices = list(HAMILTONIAN_BOX_TERMS[dim_trunc_case].keys())
        num_duplicates = len(state_indices) - len(set(state_indices))
        has_no_duplicates = num_duplicates == 0
        assert has_no_duplicates, f"Detected {num_duplicates} duplicate entries."
        print("Test passed.")


def _test_physical_plaquette_state_data_are_valid():
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
            assert len(state) == 3,\
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

        print("Test passed.")


def _test_matrix_element_data_are_valid():
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
            print("\nTest passed.")


def _test_lattice_encoder_type_error_for_bad_lattice_arg():
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
    try:
        LatticeStateEncoder(
            link_bitmap=link_bitmap,
            physical_plaquette_states=physical_states,
            lattice=bad_lattice_arg)
    except TypeError as e:
        print(f"Test passed. Raised TypeError: {e}")
        pass
    else:
        raise AssertionError("Failed to raise TypeError.")


def _test_lattice_encoder_fails_if_plaquette_states_have_wrong_number_of_controls():
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
    try:
        LatticeStateEncoder(link_bitmap, physical_states, lattice_d_2)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
        pass
    else:
        raise AssertionError("Failed to raise ValueError")


def _test_lattice_encoder_infers_correct_vertex_bitmaps():
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
    print("Test passed.")

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
    print("Test passed.")

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
    print("Test passed.")

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
    print("Test passed.")


def _test_lattice_encoder_infers_correct_plaquette_length():
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
    print("Test passed.")

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
    print("Test passed.")

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
    print("Test passed.")

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
    print("Test passed.")


def _test_lattice_encoder_fails_on_bad_creation_args():
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
    try:
        LatticeStateEncoder(good_link_bitmap, physical_states_inconsistent_control_lengths, lattice_d_3_2)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("No ValueError raised.")

    
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
    try:
        LatticeStateEncoder(
            link_bitmap=good_link_bitmap,
            physical_plaquette_states=non_unique_physical_states,
            lattice=lattice_d_3_2
        )
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("No ValueError raised.")

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
    try:
        LatticeStateEncoder(non_unique_link_bitmap, good_physical_states, lattice_d_3_2)
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
        LatticeStateEncoder(link_bitmap_different_string_lengths, good_physical_states, lattice_d_3_2)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("No ValueError raised.")


def _test_encode_decode_various_links():
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
        print(f"Test passed. Validated {bit_string_encoding} <-> {link_state}")

    print("Verifying that unknown bit string is decoded to None.")
    assert lattice_encoder.decode_bit_string_to_link_state("01") is None
    print("Test passed.")


def _test_encode_decode_various_vertices():
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
        print(f"Test passed. Validated {bit_string_encoding} <-> {multiplicity_index}")

    print("Verifying that unknown bit string is decoded to None.")
    assert lattice_encoder.decode_bit_string_to_vertex_state("11") is None
    print("Test passed.")


def _test_encoding_malformed_plaquette_fails():
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
    try:
        lattice_encoder.encode_plaquette_state_as_bit_string(plaquette_wrong_length)
    except ValueError as e:
        print(f"Test passed. Raised ValueError:\n{e}")
    else:
        raise AssertionError("Test failed. No ValueError raised.")

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
        try:
            lattice_encoder.encode_plaquette_state_as_bit_string(test_data)
        except ValueError as e:
            print(f"{test_name} passed. Raised ValueError:\n{e}")
        else:
            raise AssertionError("Test failed. No ValueError raised.")

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
        try:
            lattice_encoder.encode_plaquette_state_as_bit_string(test_data)
        except ValueError as e:
            print(f"{test_name} passed. Raised ValueError:\n{e}")
        else:
            raise AssertionError("Test failed. No ValueError raised.")


def _test_encoding_good_plaquette():
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
        print("Test passed.")


# This test is a little bit slow.
def _test_all_mag_hamiltonian_plaquette_states_have_unique_bit_string_encoding():
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
        print(f"Test passed.\n{encoded_plaquette} successfully decoded to {resulting_decoded_plaquette}.")


def _test_decoding_garbage_bit_strings_result_in_none():
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
    print("Test passed.")

    print(f"Checking {bad_encoded_vertex} decodes to None using the vertex bitmap: {lattice_encoder.vertex_bitmap}")
    assert lattice_encoder.decode_bit_string_to_vertex_state(bad_encoded_vertex) is None
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

    print("Testing that decoding links/vertices/plaquettes fails with wrong length bit string.")
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
    _test_no_duplicate_physical_plaquette_states()
    print()
    _test_no_duplicate_matrix_elements()
    print()
    _test_physical_plaquette_state_data_are_valid()
    print()
    _test_matrix_element_data_are_valid()
    print()
    _test_lattice_encoder_type_error_for_bad_lattice_arg()
    print()
    _test_lattice_encoder_fails_if_plaquette_states_have_wrong_number_of_controls()
    print()
    _test_lattice_encoder_infers_correct_vertex_bitmaps()
    print()
    _test_lattice_encoder_infers_correct_plaquette_length()
    print()
    _test_lattice_encoder_fails_on_bad_creation_args()
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
    _test_decoding_garbage_bit_strings_result_in_none()
    print()
    _test_decoding_fails_when_len_bit_string_doesnt_match_bitmaps()
    print()


if __name__ == "__main__":
    _run_tests()

    print("All tests passed.")
