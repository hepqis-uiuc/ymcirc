"""
This module holds the encoding of lattice states as bit strings.

It provides the irreps' i-weights, the link bitmaps (IRREP_TRUNCATIONS), and
the class LatticeStateEncoder for converting link, vertex multiplicity, and
plaquette states to and from bit string encodings. See the documentation on
the class itself for more information. It also provides
physical_plaquette_states, which computes a lattice's physical plaquette
states with pyclebsch (https://github.com/hepqis-uiuc/pyclebsch).

########## Irrep link state bit string encodings ##########

A truncation is a choice of which link irrep states to include in a
simulation. Like pyclebsch, ymcirc specifies a truncation by a truncation
mode and a cutoff c. The modes are pyclebsch's:

- "T": a link can carry any irrep with Dynkin labels (p, q) such that
  p + q <= c, and every singlet of those irreps is allowed at a vertex. T1
  (mode "T", cutoff 1) has 1, 3 and 3bar; T2 adds 6, 6bar and 8.
- "C": a link can carry any irrep whose quadratic Casimir is at most c, and
  every singlet of those irreps is allowed at a vertex.
- "B": a singlet is allowed at a vertex only if the quadratic Casimirs of its
  links sum to at most c. B3 (mode "B", cutoff 3), for example, has 1, 3 and
  3bar on its links.

Quadratic Casimirs are normalized so that the Casimir of the 3 is 4/3.

The constant IRREP_TRUNCATIONS holds the link bitmaps: the bit string
encodings of the irreps on a lattice's links. It is keyed by the set of those
irreps, as a frozenset (a plain set cannot be a dict key). For example, the
bitmap for 1, 3 and 3bar (whose constants are defined below) is

IRREP_TRUNCATIONS[frozenset({ONE, THREE, THREE_BAR})]

LatticeStateEncoder.for_truncation(lattice, truncation_mode, cutoff) computes
the lattice's physical plaquette states, and uses the bitmap for the set of
irreps on their links. The comment above
IRREP_TRUNCATIONS lists which truncations each entry covers. A truncation
whose set of irreps has no entry needs a link bitmap passed to
for_truncation.

Each link bitmap is a dictionary which maps 3-tuples to unique bit strings.
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
- FIFTEEN = (3, 1, 0)
- FIFTEEN_BAR = (3, 2, 0)

There are generically "leftover" states when encoding a particular truncation
into a set of qubits (i.e. the state space of a link in a particular truncation
is usually not of dimension 2^n). These are omitted from the link bitmaps.

There is also generically a multiplicity index associated with each lattice
vertex denoting which singlet state the vertex is in. The multiplicity index is
needed because there are generically multiple ways to create singlets from link
assignments in arbitrary dimension, and for arbitrary irrep truncations, there
will be complete link assignments which yield various "multiplicities" of
singlets.

########## Physical plaquette states, and singlet multiplicities ##########

The function physical_plaquette_states(lattice_def, truncation_mode, cutoff)
returns all the single-plaquette gauge-invariant states of a lattice at a
truncation, computed by pyclebsch. The states are grouped by the plane and the
signature of the plaquettes they belong to. A plaquette's signature
(ymcirc._abstract.Signature, given by Plaquette.signature) lists the half-link
directions at each of its vertices. It fixes how many control links each vertex
has, so plaquettes at an open boundary have signatures of their own. On a
periodic lattice, every plaquette of a plane has the same signature. For
example, a periodic d=2 lattice of size 3 has one plane and one signature, so

physical_plaquette_states(LatticeDef(2, 3), "T", 1)

returns {((1, 2), ((1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2),
(1, 2, -1, -2))): states}, with states the set of the plaquette states
described below.

Each plaquette state takes the form of a tuple of tuples:

plaquette = (
    vertex_multiplicities,
    a_links,
    c_links
)

The first two elements of plaquette are 4-tuples consisting of multiplicity
integers (zero-indexed) and the four "active" link i-Weights for the plaquette
state. For example:

vertex_multiplicities = (0, 0, 0, 2)
a_links = (ONE, ONE, THREE, THREE)

The final element of a plaquette is a tuple of per-vertex "control" link tuples.
A vertex's control links are its half-links other than the plaquette's two
active links there. Each inner tuple contains the i-Weights for the control
links at that vertex, in the lattice's F-order. On a periodic lattice, each
vertex has 2*(d - 1) controls. A vertex on an open boundary has one control
fewer for each half-link it lacks. The plaquette's signature gives the number
of controls at each vertex.

The names "active" and "control" come from the plaquette operator, a Wilson
loop in the fundamental representation. The plaquette operator changes the
states of the four active links only. Which changes satisfy Gauss's law at the
plaquette's vertices depends on the states of the control links, which the
plaquette operator leaves unchanged. A plaquette's Givens rotations are
therefore controlled on the qubits of its control links.

In d=3/2 (1 control per vertex):

c_links = ((c_v1,), (c_v2,), (c_v3,), (c_v4,))

 c_v4 --- v4 ----l3--- v3 --- c_v3
          |            |
          |            |
          l4           l2
          |            |
          |            |
 c_v1 --- v1 ----l1--- v2 --- c_v2

In d=2 (2 controls per vertex, with the default F-order (1, 2, 3, -1, -2, -3)):

c_links = (
    (c_v1_dir-1, c_v1_dir-2),   # v1: dirs (-1, -2)
    (c_v2_dir+1, c_v2_dir-2),   # v2: dirs (+1, -2)
    (c_v3_dir+1, c_v3_dir+2),   # v3: dirs (+1, +2)
    (c_v4_dir+2, c_v4_dir-1)    # v4: dirs (+2, -1)
)

              c_v4_dir+2   c_v3_dir+2
                   |            |
c_v4_dir-1 ---- v4 ----l3--- v3 ---- c_v3_dir+1
                   |            |
                   l4           l2
                   |            |
c_v1_dir-1 ---- v1 ----l1--- v2 ---- c_v2_dir+1
                   |            |
              c_v1_dir-2   c_v2_dir-2

For some lattices and truncations (d=3/2 at T1, for example), no vertex has a
singlet multiplicity greater than one, so every multiplicity index is 0 and
those data can be ignored; the encoder's vertex bitmap is then empty.

Some control links of a plaquette repeat another of its control links. The
encoder leaves these repeats out of the plaquette's bit string. Such repeats
occur on a periodic axis of length 2, where one link can be a control link at
two vertices of a plaquette. For example, on a periodic d=3/2 lattice of size
2 (2 plaquettes), the control link of v2 is that of v1, and the control link
of v4 is that of v3. Which controls repeat depends on the plaquette's plane as
well as its signature. Encoding and decoding therefore take both. A plaquette
state lists all its controls, repeats included. Encoding raises a ValueError
for a state in which two copies of one link carry different irreps. Decoding
puts the repeats back.

The matrix elements of the magnetic Hamiltonian between these states are
computed, for the same lattice and truncation, by
LatticeCircuitManager.for_truncation in ymcirc.circuit, whose constructor
documents their format.
"""
from __future__ import annotations
import copy
import logging
import numpy as np
from pyclebsch.matrix_elements.lattice_data import SiteMultiplicityIndex
from pyclebsch.su_n_operators import IrrepWeight
from ymcirc._abstract import DecodedPlaquetteState, LatticeDef, Plane, PlaquetteState, Signature
from ymcirc._abstract.lattice_data import DuplicateControlIndices, PlaquetteControlIndices
from ymcirc._pyclebsch_adapter import physical_plaquette_states

# IrrepWeight (pyclebsch's) and physical_plaquette_states
# (ymcirc._pyclebsch_adapter's) are defined elsewhere and available from
# here.
__all__ = [
    "EIGHT",
    "FIFTEEN",
    "FIFTEEN_BAR",
    "IRREP_TRUNCATIONS",
    "ONE",
    "SIX",
    "SIX_BAR",
    "THREE",
    "THREE_BAR",
    "BitString",
    "IrrepBitmap",
    "IrrepWeight",
    "LatticeStateEncoder",
    "VertexMultiplicityBitmap",
    "physical_plaquette_states",
]

# Set up module-specific logger
logger = logging.getLogger(__name__)

# A plaquette's two active links meet at each of its vertices.
_N_ACTIVE_LINKS_PER_VERTEX = 2


# Useful type aliases. IrrepWeight and SiteMultiplicityIndex are pyclebsch's,
# and PlaquetteState is defined in ymcirc._abstract.lattice_data.
type BitString = str
type IrrepBitmap = dict[IrrepWeight, BitString]
type VertexMultiplicityBitmap = dict[SiteMultiplicityIndex, BitString]

# Irrep iweights (top row of GT pattern).
# Note: related to (p, q) scheme via (p + q, q, 0).
ONE: IrrepWeight = (0, 0, 0)
THREE: IrrepWeight = (1, 0, 0)
THREE_BAR: IrrepWeight = (1, 1, 0)
SIX: IrrepWeight = (2, 0, 0)
SIX_BAR: IrrepWeight = (2, 2, 0)
EIGHT: IrrepWeight = (2, 1, 0)
FIFTEEN: IrrepWeight = (3, 1, 0)
FIFTEEN_BAR: IrrepWeight = (3, 2, 0)

# Irrep encoding bitmaps, keyed by the set of irreps on a lattice's links.
# The codes are chosen by hand: they decide which Givens rotations share an LP
# family. LatticeStateEncoder.for_truncation looks up the entry for a lattice
# and truncation.
#
# Each entry is the set of link irreps that some supported truncation produces.
# On periodic lattices of size 3:
# - {1, 3, 3bar}: T1 and B3 in d=3/2 and d=2; B5 in d=3/2; B3 and B4 in d=3;
#   B4 in d=2.
# - {1, 3, 3bar, 6, 6bar, 8}: T2 and B6-B9 in d=3/2; B7 in d=2.
# - the same plus 15 and 15bar: B10 in d=3/2.
# A B truncation with cutoff c keeps the singlets at a site whose links'
# quadratic Casimirs sum to at most c (with C2(3) = 4/3, so C2(15) = 16/3 and
# C2(10) = 6). In d=3/2 every site has three links, and the cheapest singlet
# there that contains a 10 is 10 x 6bar x 3bar, whose Casimirs sum to 32/3. So
# at B10, 10 and 10bar appear on no link, and the third entry needs no codes
# for them; it fills all eight codes of three bits. They first appear at
# B11, whose set has no entry: LatticeStateEncoder.for_truncation raises a
# ValueError for it unless given codes to use as link_bitmap, and making codes
# the default means adding an entry here, with codes chosen by hand.
IRREP_TRUNCATIONS: dict[frozenset[IrrepWeight], IrrepBitmap] = {
    frozenset({ONE, THREE, THREE_BAR}): {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "01"
    },
    frozenset({ONE, THREE, THREE_BAR, SIX, SIX_BAR, EIGHT}): {
        ONE: "000",
        THREE: "100",
        THREE_BAR: "001",
        SIX: "110",
        SIX_BAR: "011",
        EIGHT: "111"
    },
    frozenset({ONE, THREE, THREE_BAR, SIX, SIX_BAR, EIGHT, FIFTEEN, FIFTEEN_BAR}): {
        ONE: "000",
        THREE: "100",
        THREE_BAR: "001",
        SIX: "110",
        SIX_BAR: "011",
        EIGHT: "111",
        FIFTEEN: "101",
        FIFTEEN_BAR: "010"
    },
}


def _link_irreps(physical_plaquette_states: list[PlaquetteState]) -> frozenset[IrrepWeight]:
    """Return the irreps that appear on any link, active or control, of any of the plaquette states."""
    return frozenset(
        link
        for _, a_links, c_links in physical_plaquette_states
        for link in (*a_links, *(c_link for vertex in c_links for c_link in vertex))
    )


class LatticeStateEncoder:
    """
    Class for encoding/decode lattice state data.

    Provides mapping functionality between the physical states of links,
    vertices, and plaquettes into bit strings, and vice versa.
    """

    def __init__(
            self,
            link_bitmap: IrrepBitmap,
            physical_plaquette_states: list[PlaquetteState],
            lattice: LatticeDef,
    ):
        """
        Create a LatticeStateEncoder for states on lattice using link_bitmap and list of physical plaquette states.

        Inputs:
        - link_bitmap: a dictionary of link iWeights mapping to bit strings.
        - physical_plaquette_states: a list of physical plaquette states.
        - lattice: a ymcirc LatticeDef or child class. This provides information
            about the size of the lattice, boundary conditions, dimensionality, etc.
            The encoder takes the F-order, the order of the half-links at a
            vertex, from lattice.forder. Plaquette states list each vertex's
            control links in that order, and bit strings keep it
            (encode_plaquette_state_as_bit_string).

        The link bitmap is assumed to have unique bit strings as values,
        and the desired link iWeight tuple as keys as keys. A multiplicity bitmap is
        automatically constructed from physical_plaquette_states by examining the largest
        vertex multiplicity integer n appearing among the physical states, and allocating
        a length l = floor(log2(n) + 1) bitstring to store the binary encoding of each integer.
        Inverting of the bitmaps is handled internally when decoding.

        Control links in plaquette states use a per-vertex nested format:
        c_links = ((c_at_v1, ...), (c_at_v2, ...), (c_at_v3, ...), (c_at_v4, ...))

        A ValueError is raised if there are inconsistent length bit strings among the
        values of the link bitmap dictionary, if there are duplicate entries in the
        list of physical plaquette states, if a physical plaquette state has
        numbers of control links at its vertices that no plaquette of the
        lattice has, or if an irrep on a link of a
        physical plaquette state has no code in the link bitmap (the error
        names those irreps). The link bitmap may have codes for irreps that
        appear on no link.

        The physical plaquette states of all the lattice's planes and
        signatures may be passed together. Encoding and decoding take a
        plaquette's plane and signature, which fix the layout of its bit
        string (encode_plaquette_state_as_bit_string).

        LatticeStateEncoder.for_truncation creates the encoder for a lattice
        at an irrep truncation, using pyclebsch to either compute
        the physical plaquette states or read them from memory if already
        computed. An encoder created from the constructor has no truncation
        (its truncation property is None).
        """
        # Validation of input.
        # Check link bitmaps for uniqueness of bit string encodings.
        # Also check physical states for uniqueness (no dupes allowed).
        link_bitmap_has_unique_values = len(set(bit_string for bit_string in link_bitmap.values())) \
            == len([bit_string for bit_string in link_bitmap.values()])
        plaquette_states_are_unique = len(physical_plaquette_states) == len(set(physical_plaquette_states))
        if link_bitmap_has_unique_values is False:
            raise ValueError("Argument link_bitmap must be a dict with unique values. "
                             f"Encountered: {link_bitmap}")
        if plaquette_states_are_unique is False:
            raise ValueError("Argument physical_plaquette_states must be a list"
                             f" with unique values. Encountered: {physical_plaquette_states}.")
        if not isinstance(lattice, LatticeDef):
            raise TypeError(f"The lattice argument must be an instance of {LatticeDef.__name__}. Received: {type(lattice)}.")
        # The duplicate controls of each plane and signature of the lattice's
        # plaquettes. Its keys are the lattice's planes and signatures.
        self._duplicate_control_indices_by_plane_and_signature: dict[
            tuple[Plane, Signature], DuplicateControlIndices
        ] = {
            (plane, signature): lattice._duplicate_control_indices(plane, signature)
            for plane, signature in lattice._planes_and_signatures()
        }
        # The constructor's states don't say which signature they belong to, so each
        # state's numbers of control links at its vertices are checked against those
        # of every signature of the lattice. Encoding checks a state against its own
        # signature.
        n_controls_at_vertices_of_each_lattice_plaquette = {
            LatticeStateEncoder._n_controls_at_plaquette_vertices(signature)
            for _, signature in self._duplicate_control_indices_by_plane_and_signature
        }
        for plaquette in physical_plaquette_states:
            _, _, c_links = plaquette
            if tuple(len(vertex_controls) for vertex_controls in c_links) not in n_controls_at_vertices_of_each_lattice_plaquette:
                raise ValueError(
                    f"The plaquette state {plaquette} has a number of control links at its vertices that no plaquette "
                    f"of {lattice} has. The plaquettes there have {sorted(n_controls_at_vertices_of_each_lattice_plaquette)}.")
        irreps_without_code = _link_irreps(physical_plaquette_states) - link_bitmap.keys()
        if irreps_without_code:
            raise ValueError(
                f"Argument link_bitmap has no code for the irreps {sorted(irreps_without_code)}, which appear on "
                "the links of physical plaquette states.")

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
        self._lattice = LatticeDef(lattice.dim, lattice.size, lattice.periodic_boundary_conds, forder=lattice.forder)

        # The repr gives the number of physical plaquette states, not the
        # states, of which there can be millions.
        self._n_physical_plaquette_states = len(physical_plaquette_states)
        # Set by for_truncation; an encoder created from its constructor was
        # given its physical plaquette states directly, for no truncation.
        self._truncation: tuple[str, int] | None = None

        logger.info(f"Created {self}.")

    @classmethod
    def for_truncation(
            cls,
            lattice: LatticeDef,
            truncation_mode: str,
            cutoff: int,
            link_bitmap: IrrepBitmap | None = None,
    ) -> LatticeStateEncoder:
        """
        Create the encoder for lattice at an irrep truncation.

        Arguments:
        - lattice: a ymcirc LatticeDef or child class: the lattice whose
            plaquette states the encoder encodes. The encoder takes the
            dimensionality, size, boundary conditions and F-order
            (lattice.forder) from lattice.
        - truncation_mode, cutoff: the irrep truncation, as pyclebsch's
            irreps_and_singlets takes it, for example ("T", 1) or ("B", 3).
            pyclebsch validates them.
        - link_bitmap: optional. The codes of the link irreps. By default,
            the entry of IRREP_TRUNCATIONS for exactly the set of irreps that
            appear on the links of the lattice's physical plaquette states;
            provide this argument if working with a set of irreps
            not present in IRREP_TRUNCATIONS.

        The physical plaquette states are computed/cached in memory by pyclebsch
        (physical_plaquette_states). The link bitmap in use can be read from
        the encoder's link_bitmap. The encoder records truncation_mode and
        cutoff, readable as its truncation property, which
        LatticeCircuitManager.for_truncation uses.

        There are two scenarios where a ValueError is raised to aid the user:
          1. link_bitmap is not provided, and IRREP_TRUNCATIONS has no entry for
             the set of irreps occurring on the links of lattice with the given
             truncation_mode and cutoff. In this case, the error message states
             what the set of irreps appearing on the links is, and a custom
             link_bitmap which contains them as a subset may be used.
          2. link_bitmap is provided, but is missing an irrep which appears
             on the provided lattice with the given truncation_mode and cutoff.
             The irrep(s) missing from link_bitmap appear(s) in the error message.
        """
        # The states of every plane and signature together, each state once.
        states = list(set().union(*physical_plaquette_states(lattice, truncation_mode, cutoff).values()))
        if link_bitmap is None:
            link_irreps = _link_irreps(states)
            if link_irreps not in IRREP_TRUNCATIONS:
                raise ValueError(
                    f"No link bitmap is registered for the irreps {sorted(link_irreps)}, which appear on the links of "
                    f"{lattice} at truncation mode {truncation_mode!r} and cutoff {cutoff}. Pass your own bitmap codes as "
                    "link_bitmap."
                )
            link_bitmap = IRREP_TRUNCATIONS[link_irreps]
        encoder = cls(link_bitmap, states, lattice)
        encoder._truncation = (truncation_mode, cutoff)
        return encoder

    def __repr__(self):
        """
        Return a summary of the encoder, not an expression that recreates it.

        It gives the link bitmap, the lattice and the number of physical
        plaquette states, because the number of physical plaquette states
        can be extremely large.
        """
        class_name = type(self).__name__
        return (
            f"<{class_name} link_bitmap={self._link_bitmap!r}, lattice={self._lattice!r}, "
            f"{self._n_physical_plaquette_states} physical plaquette states>"
        )

    def __str__(self):
        return f"Lattice encoder:\nlink bitmap = {self.link_bitmap}\nvertex bitmap = {self.vertex_bitmap}\nlattice geometry = {self._lattice}"

    @property
    def vertex_bitmap(self) -> VertexMultiplicityBitmap:
        """Return a copy of the vertex multiplicity bitmap in use."""
        return copy.deepcopy(self._vertex_bitmap)

    @property
    def truncation(self) -> tuple[str, int] | None:
        """
        Return the irrep truncation the encoder was created for, as (truncation_mode, cutoff).

        None if the encoder was created from its constructor, with physical
        plaquette states given directly; LatticeStateEncoder.for_truncation
        records the truncation. It cannot be reassigned.
        """
        return self._truncation

    @property
    def link_bitmap(self) -> IrrepBitmap:
        """Return a copy of the link_bitmap in use."""
        return copy.deepcopy(self._link_bitmap)

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

    def decode_bit_string_to_link_state(self, encoded_link: str) -> IrrepWeight | None:
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

    def encode_vertex_state_as_bit_string(self, vertex: SiteMultiplicityIndex) -> str:
        """Encode a vertex multiplicity integer as a bit string."""
        return self._vertex_bitmap[vertex]

    def decode_bit_string_to_vertex_state(self, encoded_vertex: str) -> SiteMultiplicityIndex | None:
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
            plaquette_state: PlaquetteState,
            plane: Plane,
            signature: Signature,
    ) -> str:
        """
        Encode plaquette_state, a state of a plaquette with this plane and signature, as a bit string.

        Plaquette states take the form of a 3-tuple. The first element of
        this tuple is a 4-tuple of integers representing multiplicity indices
        associated to each vertex. The second element is a 4-tuple of
        i-weights which label irreps on the active links of the plaquette.
        The third element is a 4-tuple of per-vertex control tuples. Each lists
        the i-weights of the control links at that vertex in the F-order of
        the encoder's lattice (self.lattice_def.forder), as pyclebsch's
        states do. The encoder writes the controls into the bit string in the
        order given. It cannot check that order, so the caller must list them
        in the F-order of the encoder's lattice (see PlaquetteState in
        ymcirc._abstract). The F-order also fixes the order of half-links in
        the signature and the positions of the duplicate controls.

        The signature (Plaquette.signature) fixes the number of control links
        at each vertex. On a periodic axis of length 2, one link can be a
        control link at two vertices of a plaquette, depending on the plane as
        well as the signature (LatticeDef._duplicate_control_indices).
        plaquette_state gives every control at every vertex, duplicates
        included. The bit string keeps the first instance of each link and
        leaves out the duplicates.

        Assumes the ordering convention:

        |v1 v2 v3 v4 l1 l2 l3 l4 c_v1... c_v2... c_v3... c_v4...>

        according to the layout:

        v4 ----l3--- v3
        |            |
        |            |
        l4           l2
        |            |
        |            |
        v1 ----l1--- v2

        where c_vj is shorthand for the per-vertex control links at vertex vj.

        If self.vertex_bitmap is empty, it is assumed that vertex degrees of freedom
        are redundant, and they are treated as the "empty" string.
        Operationally, this translates to encoding

        |l1 l2 l3 l4 controls>

        in a bit string instead.

        Raises a ValueError if the lattice has no plaquette with this plane
        and signature, if a vertex of plaquette_state has a different number of
        control links from the one the signature gives, or if a duplicate
        control carries a different irrep from the control it repeats (no
        state of the lattice does that).
        """
        if len(plaquette_state) != 3:
            raise ValueError(
                "Plaquette states should be a tuple of three tuples corresponding to vertices, active links, and control links."
                f" Encountered a plaquette state with len(plaquette_state) = {len(plaquette_state)}.")

        vertices, a_links, c_links = plaquette_state
        if len(vertices) != 4:
            raise ValueError(f"Encountered {len(vertices)} vertex multiplicities instead of 4.")
        if len(a_links) != 4:
            raise ValueError(f"Encountered {len(a_links)} active links instead of 4.")
        duplicate_control_indices = self._duplicate_control_indices(plane, signature)
        n_controls_at_plaquette_vertices = LatticeStateEncoder._n_controls_at_plaquette_vertices(signature)
        if tuple(len(vertex_controls) for vertex_controls in c_links) != n_controls_at_plaquette_vertices:
            raise ValueError(
                f"The plaquette state {plaquette_state} has {tuple(len(vertex_controls) for vertex_controls in c_links)} "
                f"control links at its vertices; plaquettes in plane {plane} with signature {signature} have "
                f"{n_controls_at_plaquette_vertices}.")
        for (duplicate_vertex_idx, duplicate_control_idx), (original_vertex_idx, original_control_idx) \
                in duplicate_control_indices.items():
            if c_links[duplicate_vertex_idx][duplicate_control_idx] != c_links[original_vertex_idx][original_control_idx]:
                raise ValueError(
                    f"In the plaquette state {plaquette_state}, the control link at vertex {duplicate_vertex_idx + 1}, "
                    f"position {duplicate_control_idx}, is the same link as that at vertex {original_vertex_idx + 1}, "
                    f"position {original_control_idx}, in plane {plane}, but carries a different irrep.")

        bit_string_encoding = ""

        if not len(self._vertex_bitmap) == 0:
            for multiplicity in vertices:
                if not (isinstance(multiplicity, int) and multiplicity >= 0):
                    raise ValueError(f"Encountered multiplicity index '{multiplicity}' Should be a nonnegative integer")
                bit_string_encoding += self._vertex_bitmap[multiplicity]
        for a_link in a_links:
            if not (len(a_link) == 3 and all(isinstance(elem, int) for elem in a_link)):
                raise ValueError("Link data must take the form of an SU(3) i-Weight. "
                                 "They should be 3-tuples of ints. "
                                 f"Encountered:\n{a_link}.")
            bit_string_encoding += self._link_bitmap[a_link]
        for vertex_idx, vertex_controls in enumerate(c_links):
            for control_idx, c_link in enumerate(vertex_controls):
                if not (len(c_link) == 3 and all(isinstance(elem, int) for elem in c_link)):
                    raise ValueError("Link data must take the form of an SU(3) i-Weight. "
                                     "They should be 3-tuples of ints. "
                                     f"Encountered:\n{c_link}.")
                if (vertex_idx, control_idx) in duplicate_control_indices:
                    continue
                bit_string_encoding += self._link_bitmap[c_link]

        logging.debug(f"Extracted plaquette data. Proceeding to encode.")

        return bit_string_encoding

    def decode_bit_string_to_plaquette_state(
            self, bit_string: str, plane: Plane, signature: Signature) -> DecodedPlaquetteState:
        """
        Decode bit_string, the encoding of a state of a plaquette with this plane and signature, as a plaquette state.

        bit_string must have been encoded by an encoder that matches this
        one: the same lattice, F-order included, the same link bitmap and the
        same vertex bitmap. Such an encoder is, for example, the one used to
        build the circuit that measured bit_string. The decoder cannot check
        this. Decoding saved measurement results, for example, with an
        encoder rebuilt with a different F-order or link bitmap reads each
        bit string as a different state, with no error unless the lengths
        differ.

        State ordering convention starts at bottom left vertex and goes

        |v1 v2 v3 v4 l1 l2 l3 l4 c_v1... c_v2... c_v3... c_v4...>

        according to the layout:

        v4 ----l3--- v3
        |            |
        |            |
        l4           l2
        |            |
        |            |
        v1 ----l1--- v2

        The controls are grouped per vertex, each vertex's in the F-order of
        the encoder's lattice (self.lattice_def.forder), the order in which
        encode_plaquette_state_as_bit_string writes them. If unable to decode
        to physical state data, returns None for that degree of freedom. If
        the encoder has no vertex bits, every multiplicity index is returned
        as None. The result is a DecodedPlaquetteState (ymcirc._abstract),
        which allows these None values.

        This is the inverse of encode_plaquette_state_as_bit_string. Controls
        that the bit string leaves out because they repeat an earlier control
        of the plaquette (on a periodic axis of length 2) are put back, with
        the irrep of the control they repeat.

        Raises a ValueError if the lattice has no plaquette with this plane
        and signature, or if bit_string does not have the length of a state
        of those plaquettes.
        """
        duplicate_control_indices = self._duplicate_control_indices(plane, signature)
        n_controls_at_plaquette_vertices = LatticeStateEncoder._n_controls_at_plaquette_vertices(signature)
        expected_length = self._plaquette_bit_string_length(plane, signature)
        if len(bit_string) != expected_length:
            raise ValueError(f"The bit string {bit_string} has length {len(bit_string)}; states of plaquettes in "
                             f"plane {plane} with signature {signature} have length {expected_length}.")

        # Parse input into vertex, active link, and control link substrings.
        idx_first_a_link_bit = 4 * self._expected_vertex_bit_string_length
        idx_first_c_link_bit = idx_first_a_link_bit + (4 * self._expected_link_bit_string_length)
        vertices_substring = bit_string[:idx_first_a_link_bit]
        a_links_substring = bit_string[idx_first_a_link_bit:idx_first_c_link_bit]
        c_links_substring = bit_string[idx_first_c_link_bit:]

        # Decode vertices and active links.
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

        # Decode control links, then put back the duplicates the bit string
        # leaves out, and group the controls per vertex.
        decoded_c_links_in_bit_string = [
            self.decode_bit_string_to_link_state(encoded_link) for encoded_link in
            LatticeStateEncoder._split_string_evenly(
                c_links_substring, self._expected_link_bit_string_length)
        ]
        # Every control link of the plaquette, as (vertex index, control index), in
        # the order of the bit string.
        control_indices: list[PlaquetteControlIndices] = [
            (vertex_idx, control_idx)
            for vertex_idx, n_controls in enumerate(n_controls_at_plaquette_vertices)
            for control_idx in range(n_controls)
        ]
        # The bit string holds the controls that are not duplicates, in this order.
        non_duplicate_control_indices = [
            indices for indices in control_indices if indices not in duplicate_control_indices]
        control_link_irreps: dict[PlaquetteControlIndices, IrrepWeight | None] = dict(
            zip(non_duplicate_control_indices, decoded_c_links_in_bit_string, strict=True))
        # Each duplicate is the same link as the control it repeats.
        for duplicate, original in duplicate_control_indices.items():
            control_link_irreps[duplicate] = control_link_irreps[original]
        decoded_c_links = tuple(
            tuple(control_link_irreps[(vertex_idx, control_idx)] for control_idx in range(n_controls))
            for vertex_idx, n_controls in enumerate(n_controls_at_plaquette_vertices)
        )

        decoded_plaquette = (
            decoded_vertices,
            decoded_a_links,
            decoded_c_links
        )

        return decoded_plaquette

    def _duplicate_control_indices(
            self, plane: Plane, signature: Signature
    ) -> DuplicateControlIndices:
        """
        The duplicate controls of the plaquettes with this plane and signature, as LatticeDef._duplicate_control_indices gives them.

        Raises a ValueError if the lattice has no plaquette with this plane
        and signature.
        """
        try:
            return self._duplicate_control_indices_by_plane_and_signature[(plane, signature)]
        except KeyError:
            raise ValueError(
                f"The lattice {self._lattice} has no plaquette in plane {plane} with signature {signature}."
            ) from None

    @staticmethod
    def _n_controls_at_plaquette_vertices(signature: Signature) -> tuple[int, int, int, int]:
        """
        The number of control links at each vertex of a plaquette with this signature.

        A vertex's control links are its half-links other than its active links.
        """
        v1, v2, v3, v4 = signature
        return (len(v1) - _N_ACTIVE_LINKS_PER_VERTEX, len(v2) - _N_ACTIVE_LINKS_PER_VERTEX,
                len(v3) - _N_ACTIVE_LINKS_PER_VERTEX, len(v4) - _N_ACTIVE_LINKS_PER_VERTEX)

    def _plaquette_bit_string_length(self, plane: Plane, signature: Signature) -> int:
        """The length of the bit strings of states of the plaquettes with this plane and signature."""
        duplicate_control_indices = self._duplicate_control_indices(plane, signature)
        n_controls_at_plaquette_vertices = LatticeStateEncoder._n_controls_at_plaquette_vertices(signature)
        n_controls_in_bit_string = sum(n_controls_at_plaquette_vertices) - len(duplicate_control_indices)
        return (4 * (self._expected_vertex_bit_string_length + self._expected_link_bit_string_length)
                + n_controls_in_bit_string * self._expected_link_bit_string_length)

    @staticmethod
    def _split_string_evenly(string, split_length) -> list[str]:
        """Break up a string into parts of equal length."""
        return [string[i:i+split_length] for i in range(0, len(string), split_length)]
