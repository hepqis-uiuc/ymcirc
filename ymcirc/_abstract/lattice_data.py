"""
Lattice geometry, and generic containers for data on the sites and links of a lattice.

LatticeDef describes a lattice: its dimensionality (d=3/2, a chain of
plaquettes; d=2; or d=3), its length along each axis, its boundary
conditions, and its F-order, the order of the half-links at a site.
LatticeDef takes from pyclebsch's LatticeDef
(https://github.com/hepqis-uiuc/pyclebsch) which sites, links and plaquettes
exist, each plaquette's signature (the half-links at its vertices), and which
of a plaquette's control links repeat another of its control links. pyclebsch
also computes, on that same lattice, the physical plaquette states and matrix
elements ymcirc uses. Only LatticeDef.get_traversal_order orders the sites and links,
which fixes the order of registers in a circuit and of bits in a measured bit
string.

LatticeData[T] adds to LatticeDef a mapping from the sites and links to data
of type T, such as quantum registers (ymcirc.lattice_registers) or measured
link states (ymcirc.parsed_lattice_result), and Plaquette[T] collects the data
on the sites, active links and control links of one plaquette.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from functools import cache
from itertools import product
import logging
from math import ceil, isclose
from typing import TypeVar, Generic, cast
from collections.abc import Iterable, Sequence

from pyclebsch.matrix_elements.lattice_data import (
    LatticeDef as PyclebschLatticeDef,
    LinkDirection,
    SiteControlLinks,
    SiteMultiplicityIndex,
    compute_plaquette_signature,
    plaquettes_by_signature,
)
from pyclebsch.matrix_elements.lattice_data import (
    Plane as PyclebschPlane,
)
from pyclebsch.su_n_operators import IrrepWeight

# Set up module-specific logger
logger = logging.getLogger(__name__)

# Type aliases and constants.
type LatticeVector = tuple[int, ...]
type VertexAddress = LatticeVector
# A link is addressed by a site and a direction, one-indexed: (site, +k) is
# the link from the site along axis k, and (site, -k) the link that arrives at
# the site along axis k.
type LinkAddress = tuple[LatticeVector, int]
type DimensionalitySpecifier = int | float | str  # Allows specification of d = 3/2 via strings or floats.
# A pair of positive directions spanning a plaquette's plane, such as (1, 2).
type Plane = tuple[int, int]
# A signature holds, for each of a plaquette's 4 vertices (counter-clockwise
# from its bottom-left vertex), the directions of ALL the half-links at that
# vertex (both active and control links), in the lattice's F-order. It is the second
# element of pyclebsch's PlaquetteSignature, whose first element is the plane.
# The magnetic Hamiltonian's matrix elements (PlaquetteMatrixElements, below)
# are keyed by plane and signature. ymcirc's Plaquette.signature takes each
# plaquette's signature from pyclebsch, which groups the matrix elements by
# the same signature, so a plaquette's plane and signature look up its
# matrix elements.
type Signature = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
# A plaquette state holds the singlet multiplicity indices at the plaquette's 4
# vertices, the irreps on its 4 active links, and, for each vertex, the irreps
# on its control links in the lattice's F-order. A state does not record the
# directions of its control links. Code that takes states therefore relies on
# this order and cannot check it. A state listed in another F-order is read as
# a different state. Every control link is listed, including one that repeats
# another control of the plaquette on a periodic axis of length 2. The module
# docstring of ymcirc.conventions describes the format, with diagrams. The
# multiplicity indices are None in a state decoded by an encoder that has no
# vertex bits.
type PlaquetteState = (
    tuple[
        tuple[SiteMultiplicityIndex, SiteMultiplicityIndex, SiteMultiplicityIndex, SiteMultiplicityIndex],
        tuple[IrrepWeight, IrrepWeight, IrrepWeight, IrrepWeight],
        tuple[SiteControlLinks, SiteControlLinks, SiteControlLinks, SiteControlLinks],
    ]
    | tuple[
        tuple[None, None, None, None],
        tuple[IrrepWeight, IrrepWeight, IrrepWeight, IrrepWeight],
        tuple[SiteControlLinks, SiteControlLinks, SiteControlLinks, SiteControlLinks],
    ]
)
# Matrix elements between plaquette states, as
# box_plus_box_dagger_matrix_elements in ymcirc._pyclebsch_adapter returns
# them and ymcirc.circuit's LatticeCircuitManager takes them. For each plane
# and plaquette signature, a dict from pairs of plaquette states to the
# matrix element between them.
type PlaquetteMatrixElements = dict[
    tuple[Plane, Signature], dict[tuple[PlaquetteState, PlaquetteState], float]
]
# The indices that locate a control link in a plaquette, as pyclebsch's
# PlaquetteGeometry does: (vertex index, control index). The vertex index
# (0-3) counts the plaquette's vertices counter-clockwise from the
# bottom-left one; each vertex lists its own control links in F-order (as
# in the plaquette's bit string and in PlaquetteGeometry.control_links), and
# the control index is the link's place in that vertex's list. For example,
# (1, 0) is the first control link of the second vertex.
type PlaquetteControlIndices = tuple[int, int]
# The control links of a plaquette that repeat an earlier control link of the
# plaquette, on a periodic axis of length 2. Each key gives the indices of a
# repeat. Its value gives the indices of the link's first occurrence, which a
# plaquette's bit string keeps and the repeat's position leaves out. For
# example, on a periodic d=3/2 lattice of size 2 it is
# {(1, 0): (0, 0), (3, 0): (2, 0)}. The first control of the second vertex is
# the first control of the first vertex. The first control of the fourth
# vertex is the first control of the third vertex.
type DuplicateControlIndices = dict[PlaquetteControlIndices, PlaquetteControlIndices]

# Constants
VERTICAL_DIR_LABEL: int = 2
# d = 3/2 only has two vertices in the "vertical" direction.
VERTICAL_NUM_VERTICES_D_THREE_HALVES: int = 2
# pyclebsch's lattices have 3 axes; a lattice with fewer has length 1
# along the others.
_N_PYCLEBSCH_AXES = 3
# The length pyclebsch's lattices have along an axis that ymcirc's lattice lacks.
_LENGTH_OF_MISSING_AXIS = 1


@cache
def _pyclebsch_lattice(
        num_sites: tuple[int, int, int],
        pbcs: tuple[bool, bool, bool],
        forder: tuple[LinkDirection, ...]
) -> PyclebschLatticeDef:
    """
    Return pyclebsch's lattice of these lengths, boundary conditions and F-order.

    A pyclebsch LatticeDef computes its sites, links and plaquettes once and
    keeps them, and cannot be modified, so one object serves every ymcirc
    LatticeDef of the same lattice. ymcirc creates many of these, one for
    each parsed measurement (ymcirc.parsed_lattice_result), so this avoids
    computing the same geometry again for each one.
    """
    return PyclebschLatticeDef(num_sites, pbcs, list(forder))


@cache
def _duplicate_control_indices_by_plane_and_signature(
        lattice: PyclebschLatticeDef) -> dict[tuple[Plane, Signature], DuplicateControlIndices]:
    """
    LatticeDef._duplicate_control_indices for every plane and signature of a pyclebsch LatticeDef.

    Computed once per lattice. Callers must not modify the result.
    """
    result: dict[tuple[Plane, Signature], DuplicateControlIndices] = {}
    for (plane, site_half_links), plaquette_addresses in plaquettes_by_signature(lattice).items():
        # Plaquettes with the same signature repeat their controls in the same
        # way (plaquettes_by_signature explains why), so the first stands for
        # all.
        geometry = lattice.plaquettes[plaquette_addresses[0]]
        original_indices_of_link = {}
        duplicates: DuplicateControlIndices = {}
        for vertex_idx, (control_links, unique) in enumerate(zip(geometry.control_links, geometry.unique_controls)):
            for control_idx, link in enumerate(control_links):
                if control_idx in unique:
                    original_indices_of_link[link] = (vertex_idx, control_idx)
                else:
                    duplicates[(vertex_idx, control_idx)] = original_indices_of_link[link]
        # The cast narrows pyclebsch's type (which also allows an empty tuple of
        # half-links) to ymcirc's Signature. pyclebsch returns an empty tuple only
        # for a plaquette the lattice lacks. Since site_half_links is a key of
        # plaquettes_by_signature, these half-links belong to one of the
        # lattice's plaquettes.
        result[(plane, cast(Signature, site_half_links))] = duplicates
    return result

T = TypeVar('T')


# TODO should this have distinct typevars for vertex nad link data? They can be the same but are not
# necessarily.
class Plaquette(Generic[T]):
    """
    Class for working with elementary plaquette data associated with a hypercubic lattice.

    This class defines an interface for indexing various kinds of data that are
    associated with the plaquettes on a hypercubic lattice. This consists of the following:
    (1) four vertices, (2) four "active" links, (3) a number of "control" links which depends
    on the dimensionality of the lattice, (4) a pair of lattice directions defining the plane
    in which the plaquette lives.

    An "active" link is a link which forms a CCW oriented
    loop connecting all four vertices, starting with the "bottom-left" vertex:

    v4 ----l3--- v3
    |            |
    |            |
    l4           l2
    |            |
    |            |
    v1 ----l1--- v2

    A "control" link is any other link connected to a plaquette vertex which is not
    an "active" link.

    Subclasses should specify a type for the return data on the lattice.

     Attributes:
        active_links (tuple):
            Retrieve a 4-tuple of the CCW-ordered data on the links in the Plaquette.
        vertices (tuple):
            Retrieve a 4-tuple of the CCW-ordered data on the vertices in the Plaquette.
        control_links (dict):
            Retrieve a nested dictionary with four entries. The key of each entry is one of the four
            vertex labels in the Plaquette, and the entry is a dict whose keys are  all the link labels
            for all control links at that vertex, and the values are the corresponding data on those links.
        bottom_left_vertex (LatticeVector):
            Retrieve the lattice coordinate of the "v1" vertex in the plaquette.
        plane (tuple[int, int]):
            Retrieve a 2-tuple of lattice unit vectors defining the plane of the lattice.
    """

    def __init__(
            self,
            lattice: LatticeData,
            bottom_left_vertex: LatticeVector,
            plane: tuple[int, int],
            **kwargs
    ):
        """Initialize a plaquette from a given LatticeData instance, and data to uniquely specify the plaquette."""
        # Sanity checks.
        e1, e2 = plane
        lattice_direction_labels = list(range(1, ceil(lattice.dim) + 1))
        e1_out_of_range = abs(e1) not in lattice_direction_labels
        e2_out_of_range = abs(e2) not in lattice_direction_labels
        if e1_out_of_range or e2_out_of_range:
            raise ValueError(
                f"To specify a single plaquette, e1 and e2 have to be valid unit vector labels. For the given dimension self.dim = {lattice.dim}, each must be a nonzero int of magnitude at most {ceil(lattice.dim)}.\n"
                f"Got: e1 = {e1}, e2 = {e2}."
            )
        if abs(e1) == abs(e2):
            raise ValueError(f"The inputted edges e1:{e1} and e2:{e2} are not orthogonal and do not span a plaquette.")

        # Collect vertices and active links in proper order,
        vertices_ccw = [bottom_left_vertex]
        for step in [e1, e2, -e1]:
            vertices_ccw.append(lattice.add_unit_vector_to_vertex_vector(vertices_ccw[-1], step))
        link_steps_ccw = [e1, e2, -e1, -e2]
        self._vertices = tuple([lattice.get_vertex(v, **kwargs) for v in vertices_ccw])
        self._ordered_vertex_vectors = vertices_ccw
        self._active_links = tuple([lattice.get_link(link_address, **kwargs) for link_address in zip(vertices_ccw, link_steps_ccw)])
        self._lattice = lattice  # DO NOT MUTATE THIS.

        # Construct the control link dict: at each vertex, its half-links
        # other than those of its two active links.
        active_link_dirs_per_vertex = [{e1, e2}, {-e1, e2}, {-e1, -e2}, {e1, -e2}]
        self._control_links: dict[LatticeVector, dict[LinkAddress, T]] = {}
        for v, active_link_dirs in zip(vertices_ccw, active_link_dirs_per_vertex):
            self._control_links[v] = {}
            for link_dir in lattice._half_link_directions(v):
                if link_dir not in active_link_dirs:
                    self._control_links[v][(v, link_dir)] = lattice.get_link((v, link_dir), **kwargs)

        # Remaining properties are simply set from init args.
        self._plane = plane
        self._bottom_left_vertex = bottom_left_vertex

        # Store any kwargs for later use.
        self._init_kwargs = kwargs

    def __repr__(self):
        class_name = type(self).__name__
        lattice_repr = self._lattice.__repr__()
        return f"{class_name}({lattice_repr}, {self._bottom_left_vertex}, {self._plane})"

    @property
    def active_links(self) -> tuple[T, T, T, T]:
        """Retrieve the data on the active links in counter-clockwise order."""
        return self._active_links

    @property
    def vertices(self) -> tuple[T, T, T, T]:
        """Retrieve the data on the vertices in counter-clockwise order."""
        return self._vertices

    @property
    def control_links(self) -> dict[LatticeVector, dict[LinkAddress, T]]:
        """Retrieve the address dictionary for all the control links."""
        return self._control_links

    @property
    def control_links_per_vertex(self) -> tuple[tuple[T, ...], ...]:
        """Retrieve control link data grouped by vertex, each vertex's in the lattice's F-order.

        Returns a 4-tuple of tuples, one per vertex (CCW from bottom-left).
        Each inner tuple contains the control link data at that vertex,
        sorted by the lattice's F-order.
        """
        forder = self._lattice.forder
        result = []
        for vertex_vector in self._ordered_vertex_vectors:
            ctrl_addrs = list(self._control_links[vertex_vector].keys())
            ctrl_addrs.sort(key=lambda addr: forder.index(addr[1]))
            vertex_data = tuple(self._control_links[vertex_vector][addr] for addr in ctrl_addrs)
            result.append(vertex_data)
        return tuple(result)

    @property
    def control_links_ordered(self) -> tuple[T, ...]:
        """Retrieve control links as a flat tuple, vertex by vertex, each vertex's in the lattice's F-order.

        Control links are ordered by iterating over vertices counter-clockwise
        starting from the bottom-left vertex. Within each vertex, control links
        are sorted by their position in the lattice's F-order (a tuple of the
        six directions).

        Note that this property returns the data associated with control links,
        not the directions in which they extend. For direction information,
        use the control_link_dirs_per_vertex method.
        """
        return tuple(link for vertex in self.control_links_per_vertex for link in vertex)

    @property
    def signature(self) -> Signature:
        """The signature of this plaquette, as pyclebsch computes it.

        For each of the 4 plaquette vertices, a tuple of ALL existing
        half-link directions at that vertex (both active and control), sorted
        by their position in the lattice's F-order. The magnetic Hamiltonian's
        matrix elements (ymcirc._abstract.PlaquetteMatrixElements) are keyed by plane
        and signature, so the plaquette's matrix elements are found under its
        plane and this signature.

        Physically, the signature encodes whether a plaquette is in the
        "interior" of a lattice, on a "corner", on an "edge", etc.

        Raises a KeyError if the lattice has no plaquette in this plane at
        this bottom-left vertex, which pyclebsch requires to be one of
        (1, 2), (1, 3), (2, 3).
        """
        return self._lattice._plaquette_signature(self._bottom_left_vertex, self._plane)

    @property
    def control_link_dirs_per_vertex(self) -> tuple[tuple[int, ...], ...]:
        """Per-vertex control link directions sorted by the lattice's F-order.

        For each of the 4 plaquette vertices, returns a tuple of the
        non-active link directions that exist at that vertex, sorted
        by their position in the lattice's F-order.

        Note that this property returns the directions in which control
        links extend, not the data which are associated with each link.
        For the actual data, use the control_links_ordered method.
        """
        forder = self._lattice.forder
        result = []
        for vertex_vector in self._ordered_vertex_vectors:
            ctrl_addrs = list(self._control_links[vertex_vector].keys())
            ctrl_dirs = sorted([addr[1] for addr in ctrl_addrs], key=lambda d: forder.index(d))
            result.append(tuple(ctrl_dirs))
        return tuple(result)

    @property
    def bottom_left_vertex(self) -> LatticeVector:
        """Retrieve the address for the bottom-left vertex."""
        return self._bottom_left_vertex

    @property
    def plane(self) -> tuple[int, int]:
        """Retrieve the plane of the plaquette."""
        return self._plane


class LatticeDef:
    """
    A class that defines a particular lattice geometry.

    The geometry comes from a pyclebsch LatticeDef of the same lattice, which
    this class holds (wraps) rather than inherits from. Inheriting would make
    pyclebsch's whole interface part of ymcirc's, which conflicts with ymcirc
    in two ways: pyclebsch's coordinates always have three components, while
    ymcirc's have two in d=3/2 and d=2; and pyclebsch's containers have their
    own iteration order, while get_traversal_order must remain the only thing
    that orders ymcirc's registers and bit strings. So this class translates
    coordinates at the boundary, and takes from pyclebsch only facts that
    carry no order.
    """

    _DEFAULT_FORDER: tuple[int, ...] = (1, 2, 3, -1, -2, -3)

    def __init__(
            self,
            dimensions: DimensionalitySpecifier,
            size: int | tuple[int, ...],
            periodic_boundary_conds: bool | tuple[bool, ...] = True,
            forder: Sequence[int] | None = None
    ):
        """
        Initialize a lattice with specified dimensionality, size, and boundary conditions.

        Arguments:
            dimensions:
                An integer, float, or string specifying the dimensionality of the lattice.
            size:
                If an int, the number of unique vertices in every lattice direction
                (for d=3/2, along the chain; it is always two vertices tall). If
                a tuple, the number of unique vertices in each lattice direction,
                for d=2 and d=3 only.
            periodic_boundary_conds:
                If a bool, controls boundary conditions in every lattice direction
                (for d=3/2, along the chain; it is always open vertically). Tuples,
                for boundary conditions in each lattice direction, are not yet
                supported.
            forder:
                Half-link ordering convention, passed to pyclebsch when it
                computes the lattice's plaquette states and matrix elements.
                A sequence (such as a list or a tuple) that is a permutation
                of (1, 2, 3, -1, -2, -3); kept as a tuple. Defaults to
                (1, 2, 3, -1, -2, -3) if None.

        Raises a NotImplementedError for more than three dimensions, which
        pyclebsch does not support, and for tuples of boundary conditions; a
        TypeError for an F-order that is not a sequence (a set or an
        iterator, whose order is not reliable), or is a string; and a
        ValueError for an F-order that is not a permutation of the six
        directions.
        """
        # Set up lattice configuration.
        self._validate_lattice_params(
            dimensions,
            size,
            periodic_boundary_conds
        )
        self._periodic_boundary_conds = periodic_boundary_conds

        # Validate and store forder.
        if forder is None:
            forder = LatticeDef._DEFAULT_FORDER
        if not isinstance(forder, Sequence) or isinstance(forder, str):
            raise TypeError(f"forder must be a sequence, such as a list or a tuple. Got: {forder!r}")
        if sorted(forder) != sorted(LatticeDef._DEFAULT_FORDER):
            raise ValueError(f"forder must be a permutation of {LatticeDef._DEFAULT_FORDER}. Got: {forder}")
        self._forder: tuple[int, ...] = tuple(forder)

        self._configure_lattice(dimensions, size)

    def __repr__(self):
        class_name = type(self).__name__
        return f"{class_name}(dim={self.dim}, size={self.size}, periodic_boundary_conds={self.periodic_boundary_conds}, forder={self.forder})"

    def __str__(self):
        if isinstance(self.periodic_boundary_conds, bool):
            boundary_conds_str = 'periodic' if self.periodic_boundary_conds is True else 'open'
        else:
            if self.all_boundary_conds_periodic is True:
                boundary_conds_str = 'periodic'
            elif False in self.periodic_boundary_conds and any(self.periodic_boundary_conds):
                boundary_conds_str = 'mixed'
            else:
                boundary_conds_str = 'open'
        return f"d={self.dim}, {self.n_plaquettes} plaquettes, shape={self.shape}, {boundary_conds_str} boundary conditions"

    def __eq__(self, o) -> bool:
        """Equality check based on being the same class and having all the same property values."""
        if isinstance(o, LatticeDef):
            return o.__dict__ == self.__dict__
        else:
            return False

    def _validate_lattice_params(self,
                                 dim: DimensionalitySpecifier,
                                 size: int | tuple[int, ...],
                                 periodic_boundary_conds: bool | tuple[bool, ...]):
        if isinstance(periodic_boundary_conds, Iterable):
            raise NotImplementedError("Tuples for boundary conditions not yet supported.")

        is_d_three_halves = dim == "3/2" or isclose(float(dim), 1.5)
        if not (is_d_three_halves or (isinstance(dim, int) and dim > 1)):
            raise ValueError(f"A {dim}-dimensional lattice doesn't make sense.")
        if isinstance(dim, int) and dim > _N_PYCLEBSCH_AXES:
            raise NotImplementedError(
                f"Lattices of more than {_N_PYCLEBSCH_AXES} dimensions are not supported. Encountered: {dim}.")

        if isinstance(size, Iterable):
            if is_d_three_halves:
                raise ValueError(f"The size of a d=3/2 lattice must be specified by an int. Encountered: {size}.")
            if len(tuple(size)) != dim:
                raise ValueError(f"A d={dim} lattice needs one length per dimension. Encountered: {size}.")
        lengths = tuple(size) if isinstance(size, Iterable) else (size,)
        if any(not isinstance(length, int) or length < 2 for length in lengths):
            raise ValueError("Lattice must have at least two vertices in each "
                             f"dimension. A size = {size} doesn't make sense.")

        if not isinstance(periodic_boundary_conds, bool):
            raise TypeError(f"Cannot interpret boundary condition as boolean: {periodic_boundary_conds}.")

    def _configure_lattice(self, dim: DimensionalitySpecifier, size: int | tuple[int, ...]):
        if dim == "3/2" or isclose(float(dim), 1.5):
            # _validate_lattice_params refuses any other size for d=3/2.
            assert isinstance(size, int)
            self._shape: tuple[int, ...] = (size, VERTICAL_NUM_VERTICES_D_THREE_HALVES)
            self._dim: float | int = 1.5
            # The chain is never periodic vertically.
            self._periodic_axes: tuple[bool, ...] = (bool(self._periodic_boundary_conds), False)
        else:
            self._dim = int(dim)
            self._shape = tuple(size) if isinstance(size, Iterable) else (size,) * self._dim
            self._periodic_axes = (bool(self._periodic_boundary_conds),) * self._dim

        # The geometry, from pyclebsch's lattice of the same shape and
        # boundary conditions, with length 1 and open along missing axes. After
        # padding, both tuples have length 3.
        n_missing_axes = _N_PYCLEBSCH_AXES - len(self._shape)
        self._pyclebsch_lattice = _pyclebsch_lattice(
            cast(tuple[int, int, int], self._shape + (_LENGTH_OF_MISSING_AXIS,) * n_missing_axes),
            cast(tuple[bool, bool, bool], self._periodic_axes + (False,) * n_missing_axes),
            # Checked by __init__ to be a permutation of pyclebsch's link directions.
            cast(tuple[LinkDirection, ...], self._forder),
        )
        self._all_vertex_vectors: set[LatticeVector] = {
            self._from_pyclebsch_coordinates(site) for site in self._pyclebsch_lattice.sites}
        self._all_link_addreses: set[LinkAddress] = {
            self._from_pyclebsch_link_address(link) for link in self._pyclebsch_lattice.links}

    def _to_pyclebsch_coordinates(self, vertex_vector: LatticeVector) -> tuple[int, int, int]:
        """The coordinates in pyclebsch's lattice of a vertex: ymcirc's, with 0 along each missing axis."""
        # After padding, the tuple has length 3.
        return cast(tuple[int, int, int], tuple(vertex_vector) + (0,) * (_N_PYCLEBSCH_AXES - len(self._shape)))

    def _from_pyclebsch_coordinates(self, site: tuple[int, ...]) -> tuple[int, ...]:
        """The coordinates of a site of pyclebsch's lattice in this one (the inverse of _to_pyclebsch_coordinates)."""
        return tuple(site[:len(self._shape)])

    def _to_pyclebsch_link_address(self, link_address: LinkAddress) -> tuple[tuple[int, int, int], LinkDirection]:
        """
        The address in pyclebsch's lattice of a link given by its start vertex and positive direction.

        Raises a ValueError if the direction is not a positive direction of
        this lattice.
        """
        vertex_vector, direction = link_address
        if direction not in range(1, len(self._shape) + 1):
            raise ValueError(f"Expected a positive direction of this lattice, encountered {direction}.")
        # Checked above to be one of pyclebsch's link directions.
        return (self._to_pyclebsch_coordinates(vertex_vector), cast(LinkDirection, direction))

    def _from_pyclebsch_link_address(self, link_address: tuple[tuple[int, ...], int]) -> LinkAddress:
        """The address in this lattice of a link of pyclebsch's (the inverse of _to_pyclebsch_link_address)."""
        site, direction = link_address
        return (self._from_pyclebsch_coordinates(site), direction)

    def _wrap_periodic_axes(self, vertex_vector: LatticeVector) -> tuple[int, ...]:
        """vertex_vector with each coordinate along a periodic axis taken modulo the axis's length."""
        return tuple(
            component % length if periodic else component
            for component, length, periodic in zip(vertex_vector, self._shape, self._periodic_axes)
        )

    def _half_link_directions(self, vertex_vector: LatticeVector) -> set[int]:
        """
        The directions of the half-links at a vertex: +k for a link leaving it along axis k, -k for one arriving.

        Raises a KeyError if the lattice has no such vertex.
        """
        site = self._to_pyclebsch_coordinates(self._wrap_periodic_axes(vertex_vector))
        return set(self._pyclebsch_lattice.sites[site])

    def _plaquette_signature(self, bottom_left_vertex: LatticeVector, plane: Plane) -> Signature:
        """
        The signature of the plaquette at bottom_left_vertex in plane, from pyclebsch.

        Raises a KeyError if the lattice has no such plaquette.
        """
        site = self._to_pyclebsch_coordinates(self._wrap_periodic_axes(bottom_left_vertex))
        if (site, plane) not in self._pyclebsch_lattice.plaquettes:
            raise KeyError(f"No plaquette in plane {plane} at vertex {bottom_left_vertex} of {self}.")
        # Checked above to be one of pyclebsch's planes.
        pyclebsch_plane = cast(PyclebschPlane, plane)
        _, site_half_links = compute_plaquette_signature((site, pyclebsch_plane), self._pyclebsch_lattice)
        # The cast narrows pyclebsch's type (which also allows an empty tuple of
        # half-links) to ymcirc's Signature. pyclebsch returns an empty tuple only
        # for a plaquette the lattice lacks. A plaquette the lattice lacks is
        # refused above.
        return cast(Signature, site_half_links)

    def _planes_and_signatures(self) -> frozenset[tuple[Plane, Signature]]:
        """The (plane, signature) pairs of the lattice's plaquettes, from pyclebsch's plaquettes_by_signature."""
        # The cast narrows pyclebsch's type (which also allows an empty tuple of
        # half-links) to ymcirc's Signature. pyclebsch returns an empty tuple only
        # for a plaquette the lattice lacks. Since site_half_links is a key of
        # plaquettes_by_signature, these half-links belong to one of the
        # lattice's plaquettes.
        return frozenset(
            (plane, cast(Signature, site_half_links))
            for plane, site_half_links in plaquettes_by_signature(self._pyclebsch_lattice)
        )

    def _duplicate_control_indices(self, plane: Plane, signature: Signature) -> DuplicateControlIndices:
        """
        Where the plaquettes with this plane and signature repeat a control link.

        On a periodic axis of length 2, one link can be a control link at two
        vertices of a plaquette. For each vertex of a plaquette, pyclebsch's
        PlaquetteGeometry.unique_controls gives the controls that no earlier
        vertex has; every control it leaves out is a duplicate, the same
        link as a control of an earlier vertex. Returns the duplicates as a
        DuplicateControlIndices. Plaquettes with the same
        plane and signature repeat their controls in the same way
        (pyclebsch.matrix_elements.lattice_data.plaquettes_by_signature
        explains why), so one plaquette of them stands for all.

        Returns an empty dict when no control repeats another, and also when
        the lattice has no plaquette with this plane and signature. The
        caller owns the returned dict.
        """
        indices_by_plane_and_signature = _duplicate_control_indices_by_plane_and_signature(
            self._pyclebsch_lattice)
        return dict(indices_by_plane_and_signature.get((plane, signature), {}))

    def _normalize_link_address(self, link_address: LinkAddress) -> LinkAddress:
        """
        Convert link_address to form where the direction label is positive.

        Raises a ValueError if the direction is not a nonzero integer of at
        most the number of axes in magnitude, and a KeyError if the lattice
        has no such link.
        """
        lattice_vector, unit_vector_label = link_address[0], link_address[1]

        # Validate input
        if not isinstance(unit_vector_label, int) or unit_vector_label == 0 \
                or abs(unit_vector_label) > len(self._shape):
            raise ValueError("Unit vectors on the lattice must be specified by nonzero integers "
                             f"of magnitude at most {len(self._shape)}. Encountered: {unit_vector_label}.")

        # Handle negative unit vector case
        if unit_vector_label < 0:
            # Go back one vertex in the direction given by the unit vector.
            lattice_vector = self.add_unit_vector_to_vertex_vector(lattice_vector, unit_vector_label)
            unit_vector_label = abs(unit_vector_label)

        normalized_link_address = (self._wrap_periodic_axes(lattice_vector), unit_vector_label)
        if normalized_link_address not in self._all_link_addreses:
            raise KeyError(f"Link {link_address} doesn't exist on the lattice {self}.")
        return normalized_link_address

    @property
    def n_plaquettes(self) -> int:
        """Retrieve the total number of unique plaquettes in the entire lattice."""
        return len(self._pyclebsch_lattice.plaquettes)

    @property
    def n_vertices(self) -> int:
        """Retrieve the total number of vertices in the entire lattice."""
        return len(self.vertex_addresses)

    @property
    def n_links(self) -> int:
        """Retrieve the total number of unique links in the entire lattice."""
        return len(self.link_addresses)

    @property
    def vertex_addresses(self) -> list[LatticeVector]:
        """Retrieve all of the lattice vectors uniquely labeling lattice vertices."""
        return sorted(list(self._all_vertex_vectors))

    @property
    def link_addresses(self) -> list[LinkAddress]:
        """Retrieve all of the (LatticeVector, positive direction) tuples uniquely labeling lattice links."""
        return sorted(list(self._all_link_addreses))

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the lattice wrt number of vertices in each dimension."""
        return self._shape

    @property
    def size(self) -> int | tuple[int, ...]:
        """
        The size argument that creates this lattice: one length, or the shape.

        One length for d=3/2 (along the chain) and for lattices of the same
        length along every axis, otherwise the shape.
        """
        if self._dim == 1.5 or len(set(self._shape)) == 1:
            return self._shape[0]
        return self._shape

    @property
    def dim(self) -> int | float:
        """
        Return the dimensionality of the lattice.

        Either the integer 2 or 3, or the float 1.5 for the 'd=3/2' case.
        """
        return self._dim

    @property
    def periodic_boundary_conds(self) -> bool | tuple[bool, ...]:
        """
        Return whether the lattice has periodic boundary conditions.

        A bool, as given to the constructor: for d=3/2, whether the chain is
        periodic along its length (it is always open vertically). Tuples of
        boundary conditions, one per direction, are not yet supported.
        """
        return self._periodic_boundary_conds

    @property
    def all_boundary_conds_periodic(self) -> bool:
        """Return whether all directions are periodic."""
        return all(self.periodic_boundary_conds) \
            if isinstance(self.periodic_boundary_conds, Iterable) \
               else self.periodic_boundary_conds

    @property
    def forder(self) -> tuple[int, ...]:
        """Return the F-order (the order of the half-links at a vertex), a tuple of the six directions."""
        return self._forder

    def add_unit_vector_to_vertex_vector(self, vertex_vector: LatticeVector, unit_vec_dir: int):
        """
        Return the lattice vector one site away from vertex vector in the direction unit_vec_dir.

        Negative values for unit_vec_dir yield backward steps. Along a
        periodic axis, the step wraps around; along any other (such as the
        vertical axis of d=3/2), the result may lie outside the lattice.
        """
        if unit_vec_dir == 0:
            raise ValueError(f"Unit vector label must be a nonzero integer.")

        sign = -1 if unit_vec_dir < 0 else 1
        unit_vec_nonzero_component_idx = abs(unit_vec_dir) - 1

        unit_vector = tuple(sign*1 if idx == unit_vec_nonzero_component_idx else 0 for idx in range(len(self.shape)))

        result_vector = tuple(v_comp + u_comp for v_comp, u_comp in zip(vertex_vector, unit_vector))
        return self._wrap_periodic_axes(result_vector)

    def get_traversal_order(self) -> list[tuple[LatticeVector, list[LinkAddress]]]:
        """
        Walk through all the vertex and link addresses in the lattice in a conventional order.

        The convention for this can be thought of as consisting of two nested for loops.
        The first for loop is over the vertices of the lattice. Then,
        for each vertex, another for loop iterates over all the "positive" links
        leaving that vertex. The iteration over vertices is by ordering on the tuples denoting
        lattice coordinates.

        Example for d = 2 with periodic boundary conditions:

        (pbc)        (pbc)        (pbc)
          |            |            |
          |            |            |
          l16          l17          l18
          |            |            |
          |            |            |
        (0,2)--l13---(1,2)--l14---(2,2)--l15---- (pbc)
          |            |            |
          |            |            |
          l10          l11          l12
          |            |            |
          |            |            |
        (0,1)---l7---(1,1)---l8---(2,1)---l9---- (pbc)
          |            |            |
          |            |            |
          l4           l5           l6
          |            |            |
          |            |            |
        (0,0)---l1---(1,0)---l2---(2,0)---l3---- (pbc)


        This will return a nested list of vertex and link addresses
        of the following form:

        [
            ((0,0), [l1, l4]),
            ((0,1), [l7, l10]),
            ((0,2), [l13, l16]),
            ((1,0), [l2, l5]),
            ...
        ]

        Note that the link addresses take the form of (for example):
        l4 == ((0, 0), 2)

        That is: "(vertex vector, positive direction index)".

        In d=3/2, the "top" pbc links in the above diagram are omitted
        because they do not exist on that lattice.
        """
        logger.debug("Constructing lattice traversal data.")
        if self.all_boundary_conds_periodic is False:
            raise NotImplementedError("Iteration through nonperiodic lattices not yet supported.")

        all_vertex_addresses_sorted = sorted(self.vertex_addresses)
        all_addresses_traversal_order = []
        for current_vertex_address in all_vertex_addresses_sorted:
            # The links leaving the vertex in the positive directions, in
            # increasing order of direction, skipping those the lattice
            # lacks (in d=3/2, the vertical links above the chain).
            link_addresses_following_current_vertex_address: list[LinkAddress] = [
                (current_vertex_address, positive_direction)
                for positive_direction in range(1, len(self._shape) + 1)
                if (current_vertex_address, positive_direction) in self._all_link_addreses
            ]
            all_addresses_traversal_order.append((current_vertex_address, link_addresses_following_current_vertex_address))

        return all_addresses_traversal_order


class LatticeData(ABC, LatticeDef, Generic[T]):
    """
    Abstract base class for storing data on a hypercubic lattice.

    This class defines an interface for handling various kinds of data that are
    associated with a hypercubic lattice structure, which includes link and vertex
    degrees of freedom. Subclasses should implement all abstractmethods defined in this
    interface to provide specific functionality for manipulating and analyzing
    lattice data.
    """

    def __init__(
            self,
            dimensions: DimensionalitySpecifier,
            size: int | tuple[int, ...],
            periodic_boundary_conds: bool | tuple[bool, ...] = True,
            forder: Sequence[int] | None = None
    ):
        """
        Initialize a lattice with specified dimensionality, size, and boundary conditions.

        The arguments, and the exceptions raised for invalid ones, are
        described in LatticeDef.__init__.
        """
        super().__init__(dimensions, size, periodic_boundary_conds, forder=forder)

    @abstractmethod
    def get_vertex(self, lattice_vector: LatticeVector) -> T:
        """Retrieve data associated with a specific vertex in the lattice."""
        pass

    @abstractmethod
    def get_link(self, link_address: LinkAddress) -> T:
        """Retrieve data associated with a specific link in the lattice.

        The argument link_address consists of a lattice vector with a
        positive unit_vector_label specifies the link which is in the
        positive direction along the dimension specified by unit_vector_label
        from the vertex given by lattice vector. A negative unit_vector_label
        specifies the opposite link.

        Example (d=3/2 with periodic boundary conditions):

        (0, 1) ----- (1, 1) ----- (pbc)
          |            |
          |            |
        (0, 0) ----- (1, 0) ----- (pbc)

        unit_vector_label = 1 labels the positive horizontal direction.
        unit_vector_label = 2 labels the positive vertical direction.
        We can address the bottom-middle link via either of the following:
            - lattice_vector = (0, 0), unit_vector_label = 1
            - lattice_vector = (1, 0), unit_vector_label = -1
        """
        pass

    def get_plaquettes(self,
                       lattice_vector: LatticeVector,
                       e1: int | None = None,
                       e2: int | None = None,
                       **kwargs
                       ) -> Plaquette[T] | list[Plaquette[T]]:
        """
        Return the list of all "positive" Plaquettes associated with the vertex lattice_vector (a single Plaquette in d=3/2 and d=2).

        The "positivity" convention is that the list of returned plaquettes corresponds to those
        defined by all pairs of orthogonal positive unit vectors at the vertex lattice_vector.
        Conventionally, the "lower" dimension labels the first element of the tuples
        representing planes, and the plaquettes are sorted.

        In d=3/2 and d=2 there is one plane, (1, 2); in d=3 there are three,
        (1, 2), (1, 3) and (2, 3).

        If a particular plaquette is desired, this can be specified by either providing the
        link unit vector directions e1 and e2 to defining a plane. Sign is ignored when
        manually specifying the plane of a specific plaquette.

        Return plaquettes will all have lattice_vector as the "bottom-left" vertex.
        """
        # A definition of convenience.
        only_one_link_given = (e1 is None or e2 is None) and (e1 != e2)

        if isinstance(e1, int) and isinstance(e2, int):  # Case where both link directions are provided.
            return Plaquette[T](lattice=self, bottom_left_vertex=lattice_vector, plane=(e1, e2), **kwargs)
        elif only_one_link_given:  # Nonsense case.
            raise ValueError(
                "Provide no link directions to get all Plaquettes at a vertex, or exactly two link directions "
                f"to get one particular Plaquette. Received e1={e1} and e2={e2}."
            )
        else:  # Case where no link directions provided, need to construct ALL plaquettes.
            if (self.dim == 1.5 or self.dim == 2):  # Only one plaquette for these geometries.
                return Plaquette[T](lattice=self, bottom_left_vertex=lattice_vector, plane=(1, 2), **kwargs)
            elif self.dim > 2:  # Generic case with multiple planes.
                set_of_planes = set((i, j) if i < j else None for i, j in product(range(1, ceil(self.dim + 1)), range(1, ceil(self.dim + 1))))
                not_none_lambda = lambda x: x is not None
                all_planes = sorted(list(filter(not_none_lambda, set_of_planes)))  # Need to strip out a spurious None
                return [Plaquette[T](lattice=self, bottom_left_vertex=lattice_vector, plane=plane, **kwargs) for plane in all_planes]

    def __iter__(self):
        """Use get_traversal_order method for iteration over lattice data."""
        all_lattice_data: list[T] = []
        for current_vertex_address, connected_link_addresses in self.get_traversal_order():
            all_lattice_data.append(self.get_vertex(current_vertex_address))
            for current_link_address in connected_link_addresses:
                all_lattice_data.append(self.get_link(current_link_address))

        yield from all_lattice_data
