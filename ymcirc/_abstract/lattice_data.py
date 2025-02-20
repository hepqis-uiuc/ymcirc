from __future__ import annotations
from abc import ABC, abstractmethod
from itertools import product
from math import ceil, isclose, comb
from typing import List, Tuple, Union, Set, TypeVar, Generic, Iterable

# Type aliases and constants.
LatticeVector = Union[List[int], Tuple[int, ...]]
VertexAddress = LatticeVector
# For indicating particular directions on a lattice. One-indexed.
LinkUnitVectorLabel = int
LinkAddress = Tuple[LatticeVector, LinkUnitVectorLabel]
DimensionalitySpecifier = Union[int, float, str]  # Allows specification of d = 3/2 via strings or floats.

# Constants
VERTICAL_DIR_LABEL: LinkUnitVectorLabel = 2
# d = 3/2 only has two vertices in the "vertical" direction.
VERTICAL_NUM_VERTICES_D_THREE_HALVES: int = 2

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
            Retrieve a length-4 tuple of the CCW-ordered data on the links in the Plaquette.
        vertices (tuple):
            Retrieve a length-4 tuple of the CCW-ordered data on the vertices in the Plaquette.
        control_links (dict):
            Retrieve a nested dictionary with four entries. The key of each entry is one of the four
            vertex labels in the Plaquette, and the entry is a dict whose keys are  all the link labels
            for all control links at that vertex, and the values are the corresponding data on those links.
        bottom_left_vertex (LatticeVector):
            Retrieve the lattice coordinate of the "v1" vertex in the plaquette.
        plane (tuple[LinkUnitVectorLabel, LinkUnitVectorLabel]):
            Retrieve a length-2 tuple of lattice unit vectors defining the plane of the lattice.
    """

    def __init__(
            self,
            lattice: LatticeData,
            bottom_left_vertex: LatticeVector,
            plane: tuple[LinkUnitVectorLabel, LinkUnitVectorLabel]
    ):
        """Initialize a plaquette from a given LatticeData instance, and data to uniquely specify the plaquette."""
        # Sanity checks.
        e1, e2 = plane
        lattice_direction_labels = list(range(1, ceil(lattice.dim) + 1))
        e1_out_of_range = abs(e1) not in lattice_direction_labels
        e2_out_of_range = abs(e2) not in lattice_direction_labels
        if e1_out_of_range or e2_out_of_range:
            raise ValueError(
                f"To specify a single plaquette, e1 and e2 have to be valid unit vector labels. For the given dimension self.dim = {lattice.dim}, must be an int between 1 and {lattice.dim + 1}.\n"
                f"Got: e1 = {e1}, e2 = {e2}."
            )
        if abs(e1) == abs(e2):
            raise ValueError(f"The inputted edges e1:{e1} and e2:{e2} are not orthogonal and do not span a plaquette.")

        # Collect vertices and active links in proper order,
        vertices_ccw = [bottom_left_vertex]
        for step in [e1, e2, -e1]:
            vertices_ccw.append(lattice.add_unit_vector_to_vertex_vector(vertices_ccw[-1], step))
        link_steps_ccw = [e1, e2, -e1, -e2]
        self._vertices = tuple([lattice.get_vertex(v) for v in vertices_ccw])
        self._active_links = tuple([lattice.get_link(link_address) for link_address in zip(vertices_ccw, link_steps_ccw)])

        # Construct the control link dict.
        all_pos_and_neg_link_dirs = set(lattice_direction_labels + [(-1)*dir for dir in lattice_direction_labels])
        active_link_dirs_per_vertex = {
            vertices_ccw[0]: [e1, e2],
            vertices_ccw[1]: [-e1, e2],
            vertices_ccw[2]: [-e1, -e2],
            vertices_ccw[3]: [e1, -e2]
        }
        self._control_links: dict[LatticeVector, dict[LinkAddress, T]] = {}
        for idx, v in enumerate(vertices_ccw):
            self._control_links[v] = {}
            for link_dir in all_pos_and_neg_link_dirs:
                on_upper_vertex_d_three_halves = link_dir == 2 and lattice.dim < 2 and v[1] == 1
                on_lower_vertex_d_three_halves = link_dir == -2 and lattice.dim < 2 and v[1] == 0
                if on_upper_vertex_d_three_halves or on_lower_vertex_d_three_halves:
                    continue
                if link_dir in active_link_dirs_per_vertex[v]:
                    continue
                try:
                    self._control_links[v][(v, link_dir)] = lattice.get_link((v, link_dir))
                except KeyError:  # Depending on boundary conditions, KeyErrors can happen that can be skipped.
                    continue

        # Remaining properties are simply set from init args.
        self._plane = plane
        self._bottom_left_vertex = bottom_left_vertex

    @property
    def active_links(self) -> tuple[T, T, T, T]:
        """Retrieve the data on the active links."""
        return self._active_links

    @property
    def vertices(self) -> tuple[T, T, T, T]:
        """Retrieve the data on the vertices."""
        return self._vertices

    @property
    def control_links(self) -> dict[LatticeVector, dict[LinkAddress, T]]:
        """Retrieve the address dictionary for all the control links."""
        return self._control_links

    @property
    def bottom_left_vertex(self) -> LatticeVector:
        """Retrieve the address for the bottom-left vertex."""
        return self._bottom_left_vertex

    @property
    def plane(self) -> tuple[LinkUnitVectorLabel, LinkUnitVectorLabel]:
        """Retrieve the plane of the plaquette."""
        return self._plane


class LatticeDef:
    """A class that defines a particular lattice geometry."""

    def __init__(
            self,
            dimensions: DimensionalitySpecifier,
            size: int | tuple[int, ...],
            periodic_boundary_conds: bool | tuple[bool, ...] = True
    ):
        """
        Initialize a lattice with specified dimensionality, size, and boundary conditions.

        Arguments:
            dimensions:
                An integer, float, or string specifying the dimensionality of the lattice.
            size:
                If an int, the number of unique vertices in every lattice direction. If
                a tuple, the number of unique vertices in each lattice direction.
            periodic_boundary_conds:
                If a bool, controls boundary counds in every lattice direction. If a tuple,
                controls boundary conditions in each lattice direction.
        """
        # Set up lattice configuration.
        self._validate_lattice_params(
            dimensions,
            size,
            periodic_boundary_conds
        )
        self._periodic_boundary_conds = periodic_boundary_conds
        self._configure_lattice(dimensions, size)

    def _validate_lattice_params(self,
                                 dim: DimensionalitySpecifier,
                                 size: int | tuple[int, ...],
                                 periodic_boundary_conds: bool | tuple[bool, ...]):
        if isinstance(size, Iterable):
            raise NotImplementedError("Tuples for lattice size not yet supported.")
        if isinstance(periodic_boundary_conds, Iterable):
            raise NotImplementedError("Tuples for boundary conditions not yet supported.")

        if not isinstance(size, int) and (dim == "3/2" or isclose(float(dim), 1.5)):
            raise ValueError("The size of a d=3/2 lattice must be specified by an int.")

        if isinstance(size, int) and size < 2:
            raise ValueError("Lattice must have at least two vertices in each "
                             f"dimension. A size = {size} doesn't make sense.")

        if not (dim == "3/2" or isclose(float(dim), 1.5) or (isinstance(dim, int) and dim > 1)):
            raise ValueError(f"A {dim}-dimensional lattice doesn't make sense.")

        if not isinstance(periodic_boundary_conds, bool):
            raise TypeError(f"Cannot interpret boundary condition as boolean: {periodic_boundary_conds}.")

    def _configure_lattice(self, dim: DimensionalitySpecifier, size: int | tuple[int, ...]):
        if isinstance(size, Iterable):
            self._shape: Tuple[int, ...] = size
            self._dim: float | int = int(dim)
        elif dim == "3/2" or isclose(float(dim), 1.5):
            self._shape = (size, VERTICAL_NUM_VERTICES_D_THREE_HALVES)
            self._dim = 1.5
        elif isinstance(dim, int) and dim > 1:
            self._shape = (size,) * int(dim)
            self._dim = int(dim)
        else:
            raise ValueError(f"A {dim}-dimensional lattice doesn't make sense.")
        # Use of sets enforces no duplicates internally.
        self._all_vertex_vectors: Set[LatticeVector] = set(product(
            *[[i for i in range(axis_length)] for axis_length in self._shape]))
        # Use one-indexed labels for positive unit vectors.
        self._lattice_unit_vector_labels = [
            i for i in range(1, ceil(self._dim) + 1)]

        # Construct the link addresses.
        self._all_link_addreses = []
        for vertex_vector in self._all_vertex_vectors:
            for link_unit_vector in self._lattice_unit_vector_labels:
                if self.dim == 1.5 and self._skip_links_above_or_below_d_equals_three_halves(vertex_vector, link_unit_vector):
                    continue  # Skip to next lattice direction.
                else:
                    link_address = (vertex_vector, link_unit_vector)
                    self._all_link_addreses.append(link_address)
        self._all_link_addreses = set(self._all_link_addreses)

    def _normalize_link_address(self, link_address: LinkAddress) -> LinkAddress:
        """Convert link_address to form where the direction label is positive."""
        lattice_vector, unit_vector_label = link_address[0], link_address[1]

        # Validate input
        if unit_vector_label == 0 or not isinstance(unit_vector_label, LinkUnitVectorLabel):
            raise ValueError("Unit vectors on the lattice must be specified by nonzero integers. "
                             f"Encountered: {unit_vector_label}.")

        # Handle negative unit vector case
        unit_vector_label_zero_indexed = abs(unit_vector_label) - 1
        unit_vector = tuple(0 if component != unit_vector_label_zero_indexed else 1 for component in range(ceil(self.dim)))
        if unit_vector_label < 0:
            # Go back one vertex in the direction given by the unit vector.
            lattice_vector = tuple(l_comp - u_comp for l_comp, u_comp in zip(lattice_vector, unit_vector))
            unit_vector_label = abs(unit_vector_label)

        # Handle links on boundaries of lattice.
        if any(component < 0 for component in lattice_vector) or any(component > self.shape[0] for component in lattice_vector):
            # TODO implement boundary logic for periodic boundary conditions
            vertical_dir_idx = VERTICAL_DIR_LABEL - 1
            if self.dim == 1.5 and (lattice_vector[vertical_dir_idx] < 0 or lattice_vector[vertical_dir_idx] > 1):
                raise KeyError(f"Lattice vertex {lattice_vector} doesn't exist for d = 3/2.")
            elif self.all_boundary_conds_periodic:
                lattice_vector = tuple(comp % self.shape[dir_idx] for dir_idx, comp in enumerate(lattice_vector))
            else:
                # TODO implement handling of fixed boundary conditions.
                raise NotImplementedError()

        normalized_link_address = (lattice_vector, unit_vector_label)
        return normalized_link_address

    @property
    def n_plaquettes(self) -> int:
        """Retrieve the total number of unique plaquettes in the entire lattice."""
        if self.all_boundary_conds_periodic is True:
            if self.dim >= 2:
                # For each unique vertex, there are dim choose 2 unique plaquettes
                # that are defined by pairs of positive lattice directions.
                return int(len(self.vertex_addresses) * comb(self.dim, 2))
            else:
                # For d=3/2, only need a count of the number of vertices along the
                # "bottom" rung of the lattice. This is half the total number of
                # vertices.
                return int(len(self.vertex_addresses) / 2)
        else:
            raise NotImplementedError("Number of plaquettes calculation only implemented for periodic lattices.")

    @property
    def n_vertices(self) -> int:
        """Retrieve the total number of vertices in the entire lattice."""
        return len(self.vertex_addresses)

    @property
    def n_links(self) -> int:
        """Retrieve the total number of unique links in the entire lattice."""
        return len(self.link_addresses)

    @property
    def n_control_links_per_plaquette(self) -> int:
        """
        Retrieve the total number of control links per plaquette.

        Note that the same physical link can appear as multiple controls
        for periodic boundary conditions on small lattices. This number is
        NOT necessarily the same as the number of physically unique links
        controlling a plaquette.
        """
        if self.all_boundary_conds_periodic is False:
            raise NotImplementedError("Only periodic boundary conditions have been implemented.")
        else:
            n_controls_per_vertex = int(2 * (self.dim - 1))
            return 4 * n_controls_per_vertex

    @property
    def vertex_addresses(self) -> list[LatticeVector]:
        """Retrieve all of the lattice vectors uniquely labeling lattice vertices."""
        return sorted(list(self._all_vertex_vectors))

    @property
    def link_addresses(self) -> list[Tuple[LatticeVector, LinkUnitVectorLabel]]:
        """Retrieve all of the (LatticeVector, LinkUnitVectorLabel) tuples uniquely labeling lattice links."""
        return sorted(list(self._all_link_addreses))

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the lattice wrt number of vertices in each dimension."""
        return self._shape

    @property
    def dim(self) -> int | float:
        """
        Return the dimensionality of the lattice.

        Either a positive integer greater than or equal to 2,
        or the float 1.5 for the 'd=3/2' case.
        """
        return self._dim

    @property
    def periodic_boundary_conds(self) -> bool | Tuple[bool, ...]:
        """
        Return whether the lattice has periodic boundary conditions.

        If there are mixed boundary conditions, return type is a tuple with
        length given by the dimensionality of the lattice.
        """
        return self._periodic_boundary_conds

    @property
    def all_boundary_conds_periodic(self) -> bool:
        """Return whether all directions are periodic."""
        return all(self.periodic_boundary_conds) \
            if isinstance(self.periodic_boundary_conds, Iterable) \
               else self.periodic_boundary_conds

    def add_unit_vector_to_vertex_vector(self, vertex_vector: LatticeVector, unit_vec_dir: LinkUnitVectorLabel):
        """
        Return the lattice vector one site away from vertex vector in the direction unit_vec_dir.

        Negative values for unit_vec_dir yield backward steps.
        """
        if unit_vec_dir == 0:
            raise ValueError(f"Unit vector label must be a nonzero integer.")

        sign = -1 if unit_vec_dir < 0 else 1
        unit_vec_dir = abs(unit_vec_dir)
        unit_vec_nonzero_component_idx = unit_vec_dir - 1

        unit_vector = tuple(sign*1 if idx == unit_vec_nonzero_component_idx else 0 for idx in range(len(self.shape)))

        result_vector = tuple(v_comp + u_comp for v_comp, u_comp in zip(vertex_vector, unit_vector))

        if self.all_boundary_conds_periodic is True and self.dim != 1.5:
            result_vector = tuple(comp % self.shape[0] for comp in result_vector)
        elif self.all_boundary_conds_periodic is True:
            # Vertical direction is NEVER periodic in d=3/2.
            result_vector = (result_vector[0] % self.shape[0], result_vector[1])
        else:
            raise NotImplementedError("Vector addition not yet implemented on nonperiodic lattices.")

        return result_vector

    @staticmethod
    def _skip_links_above_or_below_d_equals_three_halves(
            vertex_vector: LatticeVector, direction_label: LinkUnitVectorLabel) -> bool:
        """Check whether to skip initalizing unphysical registers for d=3/2."""
        if len(vertex_vector) < 2:
            raise ValueError(f"Vertex vector has length {len(vertex_vector)} < 2.")
        is_lower_vertex = vertex_vector[1] == 0
        if is_lower_vertex and direction_label == -VERTICAL_DIR_LABEL:
            # Don't make a link below the chain.
            return True
        elif not is_lower_vertex and direction_label == VERTICAL_DIR_LABEL:
            # Don't make a link above the chain
            return True
        else:
            return False


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
            periodic_boundary_conds: bool | tuple[bool, ...] = True
    ):
        """
        Initialize a lattice with specified dimensionality, size, and boundary conditions.

        Arguments:
            dimensions:
                An integer, float, or string specifying the dimensionality of the lattice.
            size:
                If an int, the number of unique vertices in every lattice direction. If
                a tuple, the number of unique vertices in each lattice direction.
            periodic_boundary_conds:
                If a bool, controls boundary counds in every lattice direction. If a tuple,
                controls boundary conditions in each lattice direction.
        """
        super().__init__(dimensions, size, periodic_boundary_conds)

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
                       e1: Union[LinkUnitVectorLabel, None] = None,
                       e2: Union[LinkUnitVectorLabel, None] = None
                       ) -> Plaquette[T] | List[Plaquette[T]]:
        """
        Retrieve plaquette(s) assocaited with a specific bottom-left vertex.

        Optionally, provide two lattice unit vectors defining a plane to retrieve
        only one plaquette.
        """
        """
        Return the list of all "positive" Plaquettes associated with the vertex lattice_vector.

        The "positivity" convention is that the list of returned plaquettes corresponds to those
        defined by all pairs of orthogonal positive unit vectors at the vertex lattice_vector.
        Conventionally, the "lower" dimension labels the first element of the tuples
        representing planes, and the plaquettes are sorted.

        Examples:
          - d = 3 has planes labeled by (1, 2), (1, 3), and (2, 3).
          - d = 4 has planes labeled by (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), and (3, 4).

        If a particular plaquette is desired, this can be specified by either providing the
        link unit vector directions e1 and e2 to defining a plane. Sign is ignored when
        manually specifying the plane of a specific plaquette.

        Return plaquettes will all have lattice_vector as the "bottom-left" vertex.
        """
        # A definition of convenience.
        only_one_link_given = (e1 is None or e2 is None) and (e1 != e2)

        if isinstance(e1, int) and isinstance(e2, int):  # Case where both link directions are provided.
            return Plaquette[T](lattice=self, bottom_left_vertex=lattice_vector, plane=(e1, e2))
        elif only_one_link_given:  # Nonsense case.
            raise ValueError(
                "Provide no link directions to get all Plaquettes at a vertex, or exactly two link directions "
                f"to get one particular Plaquette. Received e1={e1} and e2={e2}."
            )
        else:  # Case where no link directions provided, need to construct ALL plaquettes.
            if (self.dim == 1.5 or self.dim == 2):  # Only one plaquette for these geometries.
                return Plaquette[T](lattice=self, bottom_left_vertex=lattice_vector, plane=(1, 2))
            elif self.dim > 2:  # Generic case with multiple planes.
                set_of_planes = set((i, j) if i < j else None for i, j in product(range(1, ceil(self.dim + 1)), range(1, ceil(self.dim + 1))))
                not_none_lambda = lambda x: x is not None
                all_planes = sorted(list(filter(not_none_lambda, set_of_planes)))  # Need to strip out a spurious None
                return [Plaquette[T](lattice=self, bottom_left_vertex=lattice_vector, plane=plane) for plane in all_planes]
