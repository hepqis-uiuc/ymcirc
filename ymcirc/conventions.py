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

Each link bitmap is a dictionary which maps length-3 tuples to unique bit strings.
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

There is also generically a multiplicity index associated with each lattice vertex
denoting which singlet state the vertex is in. The multiplicity index is needed because
there are generically multiple ways to create singlets from link assignments in arbitrary dimension,
and for arbitrary irrep truncations, there will be complete link assignments
which yield various "multiplicites" of singlets.

########## Physical plaquette states, and singlet multiplicities ##########

The function physical_plaquette_states(lattice_def, truncation_mode, cutoff)
returns a list of all the single-plaquette gauge-invariant states of a
lattice at a truncation, computed by pyclebsch. For example, for a periodic
d=2 lattice of size 3 at T1:

physical_plaquette_states(LatticeDef(2, 3), "T", 1)

Each plaquette state takes the form of a tuple of tuples:

plaquette = (
    vertex_multiplicities,
    a_links,
    c_links
)

The first two elements of plaquette are length-4 tuples consisting of multiplicity integers
(zero-indexed) and the four "active" link i-Weights for the plaquette state. For example:

vertex_multiplicities = (0, 0, 0, 2)
a_links = (ONE, ONE, THREE, THREE)

The final element of a plaquette is a tuple of per-vertex control link tuples. Each
inner tuple contains the i-Weights for the control links at that vertex, sorted by
the FORDER convention. The number of controls per-vertex is 2*(d - 1).

In d=3/2 (1 control per vertex):

c_links = ((c_v1,), (c_v2,), (c_v3,), (c_v4,))

 c_v4 --- v4 ----l3--- v3 --- c_v3
          |            |
          |            |
          l4           l2
          |            |
          |            |
 c_v1 --- v1 ----l1--- v2 --- c_v2

In d=2 (2 controls per vertex, with default FORDER [1,2,3,-1,-2,-3]):

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

On a periodic axis of length 2, one link can be a control link at two vertices
of a plaquette. For example, on a periodic d=3/2 lattice of size 2 (2
plaquettes), the control link of v2 is that of v1, and the control link of v4
is that of v3. The encoder encodes plaquette states with all their controls.
ymcirc.circuit's LatticeCircuitManager removes the duplicates from the bit
strings: a matrix element in which two copies of one link have different link
states raises a ValueError, and a state for control pruning in which they do is
left out.

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
from ymcirc._abstract import LatticeDef
from ymcirc._pyclebsch_adapter import (
    PlaquetteState,
    physical_plaquette_states,
)

# IrrepWeight (pyclebsch's), PlaquetteState and physical_plaquette_states
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
    "PlaquetteState",
    "VertexMultiplicityBitmap",
    "physical_plaquette_states",
]

# Set up module-specific logger
logger = logging.getLogger(__name__)


# Useful type aliases. IrrepWeight and SiteMultiplicityIndex are pyclebsch's,
# and PlaquetteState is defined in ymcirc._pyclebsch_adapter, which produces
# data in that format.
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
        for link in (*a_links, *(c_link for corner in c_links for c_link in corner))
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
            The F-order convention is inherited from lattice.forder.

        The link bitmap is assumed to have unique bit strings as values,
        and the desired link iWeight tuple as keys as keys. A multiplicity bitmap is
        automatically constructed from physical_plaquette_states by examining the largest
        vertex multiplicity integer n appearing among the physical states, and allocating
        a length l = floor(log2(n) + 1) bitstring to store the binary encoding of each integer.
        Inverting of the bitmaps is handled internally when decoding.

        Control links in plaquette states use a per-vertex nested format:
        c_links = ((c_at_v1, ...), (c_at_v2, ...), (c_at_v3, ...), (c_at_v4, ...))

        A ValueError is raised if there are inconsistent length bit strings among the values
        of the link bitmap dictionary, if there are duplicate entries in the
        list of physical plaquette states, or if an irrep on a link of a
        physical plaquette state has no code in the link bitmap (the error
        names those irreps). The link bitmap may have codes for irreps that
        appear on no link.

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
        # Count total controls across all vertices in the nested format.
        _count_total_controls = lambda c_links: sum(len(vc) for vc in c_links)
        first_total_controls = _count_total_controls(physical_plaquette_states[0][2])
        consistent_num_controls_in_plaquette_states = all(
            _count_total_controls(plaquette[2]) == first_total_controls
            for plaquette in physical_plaquette_states
        )
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
        n_total_control_links = first_total_controls
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
        states = physical_plaquette_states(lattice, truncation_mode, cutoff)
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
            plaquette: PlaquetteState,
            override_n_c_links_validation: bool = False
    ) -> str:
        """
        Convert plaquette to a bit string encoding.

        Plaquette states take the form of a length-3 tuple. The first element of
        this tuple is a length-4 tuple of integers representing multiplicity indices
        associated to each vertex. The second element is a length-4 tuple of
        i-weights which label irreps on the "active" links of the plaquette.
        The third element is a tuple of 4 per-vertex control tuples, each containing
        the i-weights for the control links at that vertex sorted by FORDER.

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

        Finally, if the argument override_n_c_links_validation is True, then no validation on the total
        number of control links will be performed. This is useful (for example) for a plaquette state
        whose duplicate controls have been removed: on a periodic axis of length 2, one link can be a
        control link at two vertices of a plaquette.
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
        total_c_links = sum(len(vc) for vc in c_links)
        if total_c_links != self._lattice.n_control_links_per_plaquette and (override_n_c_links_validation is False):
            raise ValueError(f"Encountered {total_c_links} control links instead of {self._lattice.n_control_links_per_plaquette}.")

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
        for vertex_controls in c_links:
            for c_link in vertex_controls:
                if not (len(c_link) == 3 and all(isinstance(elem, int) for elem in c_link)):
                    raise ValueError("Link data must take the form of an SU(3) i-Weight. "
                                     "They should be length-3 tuples of ints. "
                                     f"Encountered:\n{c_link}.")
                bit_string_encoding += self._link_bitmap[c_link]

        logging.debug(f"Extracted plaquette data. Proceeding to encode.")

        return bit_string_encoding

    def decode_bit_string_to_plaquette_state(self, bit_string: str) -> PlaquetteState:
        """
        Decode bit string to a plaquette state in terms of iWeights.

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

        The controls are grouped per-vertex, with each vertex's controls sorted
        by FORDER. If unable to decode to physical state data, returns None for
        that degree of freedom.

        Raises NotImplementedError if the lattice does not use periodic boundary
        conditions, since decoding logic for nonperiodic lattices is not yet
        implemented.
        """
        if not self._lattice.periodic_boundary_conds:
            raise NotImplementedError(
                "Decoding plaquette states for nonperiodic lattices is not yet supported."
            )
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

        # Decode control links and group into per-vertex tuples.
        decoded_c_links_flat = [
            self.decode_bit_string_to_link_state(encoded_link) for encoded_link in
            LatticeStateEncoder._split_string_evenly(
                c_links_substring, self._expected_link_bit_string_length)
        ]
        # Number of controls per vertex for periodic lattices.
        n_controls_per_vertex = int(2 * (self._lattice.dim - 1))
        decoded_c_links = tuple(
            tuple(decoded_c_links_flat[i:i + n_controls_per_vertex])
            for i in range(0, len(decoded_c_links_flat), n_controls_per_vertex)
        )

        decoded_plaquette = (
            decoded_vertices,
            decoded_a_links,
            decoded_c_links
        )

        return decoded_plaquette

    @staticmethod
    def _split_string_evenly(string, split_length) -> list[str]:
        """Break up a string into parts of equal length."""
        return [string[i:i+split_length] for i in range(0, len(string), split_length)]
