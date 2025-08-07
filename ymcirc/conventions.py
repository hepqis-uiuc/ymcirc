"""
This module is a canonical source for project conventions, including:

- Bit string encodings.
- Magnetic Hamiltonian box term data.

This module loads data for both of these from package-local json files.
The module also provides the class LatticeStateEncoder for converting
link, vertex multiplicity, and plaquette states to and from bit string encodings.
See the documentation on the class itself for more information.

########## Irrep link state bit string encodings ##########

The constant IRREP_TRUNCATIONS is a dict of possible choices of link irrep states
to include in a simulation. A particular choice of such states is referred to
as a truncation throughout this codebase. The keys of IRREP_TRUNCATIONS correspond
to particular choices of truncation, and the values are the actual bit string encodings.
Currently supported truncations:

- T1: contains 1, 3, 3bar
- T2: contains T1 along with 6, 6bar, and 8

Each truncation is a dictionary which map length-3 tuples to unique bit strings.
The tuples represent "i-Weights", which are a way of uniquely labeling
SU(N) irreps which come from working with the Gelfand-Tsetlin pattern calculus.
See Arne et al for more details (https://doi.org/10.1063/1.3521562).
As a convenience, the following constants are defined which take on the correct
i-Weight values:

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

The dict of (lazy-loaded) dicts PHYSICAL_PLAQUETTE_STATES consists of all the single-plaquette
gauge-invariant states in a particular lattice geometry and truncation. The dict
contains data for the following cases

- "d=3/2, T1"
- "d=3/2, T2"
- "d=2, T1"

T1 refers to the ONE, THREE, THREE_BAR truncation, while T2 includes the
additional states SIX, SIX_BAR, and EIGHT. To get the physical states for a particular
case, use the following syntax:

PHYSICAL_PLAQUETTE_STATES["d=2"]["T2"]

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
HAMILTONIAN_BOX_TERMS. This dict follows an indexing pattern similar to
that of PHYSICAL_PLAQUETTE_STATES, and contains the same cases.

Once a particular case of dimension and truncation has been chosen,
the actual matrix element data takes the form of a
dictionary whose keys are tuples (final_plaquette_state, initial_plaquette_state)
and whose values are floats. The plaquette state data consists of nested tuples conveying
vertex bag states and link states. As an example,

HAMILTONIAN_BOX_TERMS["d=3/2"]["T1"] = {
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
from ymcirc._abstract import LatticeDef
from ymcirc.utilities import LazyDict, json_loader

# Filesystem stuff.
_PROJECT_ROOT = Path(__file__).parent
_HAMILTONIAN_DATA_DIR = _PROJECT_ROOT / "_ymcirc_data/magnetic-hamiltonian-box-term-matrix-elements/"
_PLAQUETTE_STATES_DATA_DIR = _PROJECT_ROOT / "_ymcirc_data/plaquette-states/"
_HAMILTONIAN_DATA_FILE_PATHS: Dict[str, Dict[str, Path]] = {
    "d=3/2": {
        "T1": _HAMILTONIAN_DATA_DIR / "T1_dim(3_2)_magnetic_hamiltonian.json",
        "T2": _HAMILTONIAN_DATA_DIR / "T2_dim(3_2)_magnetic_hamiltonian.json"},
    "d=2": {
        "T1": _HAMILTONIAN_DATA_DIR / "T1_dim(2)_magnetic_hamiltonian.json"
    }
}
_PLAQUETTE_STATES_DATA_FILE_PATHS: Dict[str, Dict[str, Path]] = {
    "d=3/2": {
        "T1": _PLAQUETTE_STATES_DATA_DIR / "T1_dim(3_2)_plaquette_states.json",
        "T2": _PLAQUETTE_STATES_DATA_DIR / "T2_dim(3_2)_plaquette_states.json"
    },
    "d=2": {
        "T1": _PLAQUETTE_STATES_DATA_DIR / "T1_dim(2)_plaquette_states.json"
    }
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
IRREP_TRUNCATIONS: Dict[str, IrrepBitmap] = {
    "T1": {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "01"
    },
    "T2": {
        ONE: "000",
        THREE: "100",
        THREE_BAR: "001",
        SIX: "110",
        SIX_BAR: "011",
        EIGHT: "111"
    }
}

# Lazy-load vertex physical plaquette states from precomputed json files.
# Entries has the format s, a, c,
# where s is a tuple of 4 site multiplicity indices,
# a is a tuple of 4 "active link" iweights,
# and c is a variable-length tuple of "control_link" iweights.
PHYSICAL_PLAQUETTE_STATES: Dict[str, LazyDict] = {
    dim_string: LazyDict({trunc_string: (json_loader, file_path) for trunc_string, file_path in _PLAQUETTE_STATES_DATA_FILE_PATHS[dim_string].items()})
    for dim_string in _PLAQUETTE_STATES_DATA_FILE_PATHS.keys()
}

# Lazy-load magnetic Hamiltonian box terms from precomputed json files.
# The following magnetic Hamiltonian box term data is available:
# d=3/2, T1
# d=3/2, T2
# d=2, T1
HAMILTONIAN_BOX_TERMS:  Dict[str, LazyDict] = {
    dim_string: LazyDict({trunc_string: (json_loader, file_path) for trunc_string, file_path in _HAMILTONIAN_DATA_FILE_PATHS[dim_string].items()})
    for dim_string in _HAMILTONIAN_DATA_FILE_PATHS.keys()
}


def load_magnetic_hamiltonian(
        dim_string: str,
        trunc_string: str,
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
      - dim_string: a string of the form "d=3/2" which specifies what
        dimensionality lattice to assume when loading magnetic Hamiltonian data. See
        the module docstring on ymcirc.conventions for more information.
      - trunc_string: a string of the form "T1" which specifies
        the particular link irrep truncation type to assume when loading magnetic Hamiltonian data. See
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
    for (final_plaquette_state, initial_plaquette_state), matrix_element_value in HAMILTONIAN_BOX_TERMS[dim_string][trunc_string].items():
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
