"""Classes for juggling registers in quantum simulations of lattices."""
from __future__ import annotations
from math import isclose, ceil
from dataclasses import dataclass
from qiskit.circuit import QuantumRegister  # type: ignore
from itertools import product
from typing import List, Tuple, Dict, Union, Set

# Type aliases and constants.
LatticeVector = Union[List[int], Tuple[int, ...]]
# For indicating particular directions on a lattice. One-indexed.
LinkUnitVectorLabel = int
DimensionalitySpecifier = Union[int, float, str]  # Allows specification of d = 3/2 via strings or floats.

# Constants
VERTICAL_DIR_LABEL: LinkUnitVectorLabel = 2
# d = 3/2 only has two vertices in the "vertical" direction.
VERTICAL_NUM_VERTICES_D_THREE_HALVES: int = 2


@dataclass
class Plaquette:
    """
    Organize registers into links and vertices.

    Convertion is to start at the bottom-left vertex, and ascend
    with a CCW convention.
    """

    link_registers: list[QuantumRegister]
    vertex_registers: list[QuantumRegister]
    bottom_left_vertex: LatticeVector
    plane: Tuple[LinkUnitVectorLabel, LinkUnitVectorLabel]

    def __post_init__(self):
        """For validating creation data."""
        if len(self.link_registers) != 4 or len(self.vertex_registers) != 4:
            raise ValueError("There should be exactly 4 vertices and 4 links in a Plaquette, "
                             f"but encountered {len(self.link_registers)} links and {len(self.vertex_registers)} vertices.")
        if not (any(isinstance(reg, QuantumRegister) for reg in self.link_registers)) or \
           not (any(isinstance(reg, QuantumRegister) for reg in self.vertex_registers)):
            raise ValueError(
                "Plaquette.vertex_registers and Plaquette.link_registers should be lists of QuantumRegister."
                f" Encountered:\nvertex_registers = {self.vertex_registers}\nlink_registers = {self.link_registers}"
            )

    def get_registers_in_local_hamiltonian_order(self):
        """
        Return the link and vertex registers in a list ordered according to the local Hamiltonian.

        Plaquette local basis states are assumed to take the form:

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
        return self.vertex_registers + self.link_registers


class LatticeRegisters:
    """
    Class for addressing QuantumRegisters in a lattice.

    Each link and vertex is assigned a unique register. The size tuple counts
    the number of vertex registers in each dimension of a (hyper)-rectangular
    lattice.
    """

    def __init__(
            self, dim: DimensionalitySpecifier,
            size: int,
            n_qubits_per_link: int = 1,
            n_qubits_per_vertex: int = 1,
            boundary_conds: str = "periodic"):
        """Initialize all registers needed to simulate the lattice."""
        self._validate_input(
            dim, size, n_qubits_per_link, n_qubits_per_vertex, boundary_conds)
        self._set_boundary_conds(cond_type=boundary_conds)
        self._configure_lattice(dim, size)
        self._initialize_qubit_registers(n_qubits_per_link, n_qubits_per_vertex)

    def _validate_input(self,
                        dim: DimensionalitySpecifier,
                        size: int,
                        n_qubits_per_link: int = 1,
                        n_qubits_per_vertex: int = 1,
                        boundary_conds: str = "periodic"):
        if size < 2:
            raise ValueError("Lattice must have at least two vertices in each "
                             f"dimension. A size = {size} doesn't make sense.")

        if n_qubits_per_vertex < 0:
            raise ValueError("Vertex registers must have nonnegative integer number of qubits. "
                             f"n_qubits_per_vertex = {n_qubits_per_vertex}.")

        if n_qubits_per_link < 1:
            raise ValueError("Link registers must have positive integer number of qubits. "
                             f"n_qubits_per_link = {n_qubits_per_link}.")

    def _set_boundary_conds(self, cond_type: str):
        if cond_type != "periodic":
            # TODO implement this eventually
            self._periodic_boundary_conds = False
            raise NotImplementedError("Lattices with nonperiodic boundary conditions are not currently supported.")
        else:
            self._periodic_boundary_conds = True

    def _configure_lattice(self, dim: DimensionalitySpecifier, size: int):
        if dim == "3/2" or isclose(float(dim), 1.5):
            self._shape: Tuple[int, ...] = (size, VERTICAL_NUM_VERTICES_D_THREE_HALVES)
            self._dim = 1.5
        elif isinstance(dim, int) and dim > 1:
            self._shape = (size,) * dim
            self._dim = dim
        else:
            raise ValueError(f"A {dim}-dimensional lattice doesn't make sense.")
        # Use of sets enforces no duplicates internally.
        self._all_vertex_vectors: Set[LatticeVector] = set(product(
            *[[i for i in range(axis_length)] for axis_length in self._shape]))
        # Use one-indexed labels for positive unit vectors.
        self._lattice_unit_vector_labels = [
            i for i in range(1, ceil(self._dim) + 1)]

    def _initialize_qubit_registers(self, n_qubits_per_link: int, n_qubits_per_vertex: int):
        self._n_qubits_per_link = n_qubits_per_link
        self._n_qubits_per_vertex = n_qubits_per_vertex
        self._vertex_registers: Dict[LatticeVector, QuantumRegister] = {}
        self._link_registers: Dict[
            tuple[LatticeVector, LinkUnitVectorLabel], QuantumRegister] = {}
        for vertex_vector in self._all_vertex_vectors:
            self._vertex_registers[vertex_vector] = QuantumRegister(self._n_qubits_per_vertex, name=f"v:{vertex_vector}")
        for vertex_vector in self._all_vertex_vectors:
            for link_unit_vector in self._lattice_unit_vector_labels:
                if self.dim == 1.5 and self._skip_links_above_or_below_d_equals_three_halves(vertex_vector, link_unit_vector):
                    continue  # Skip to next lattice direction.
                else:
                    self._link_registers[vertex_vector, link_unit_vector] = QuantumRegister(self._n_qubits_per_link, name=f"l:{vertex_vector, link_unit_vector}")

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

    def get_vertex_register(self, lattice_vector: LatticeVector) -> QuantumRegister:
        """Return the QuantumRegister for the vertex specified by lattice_vector."""
        if self.boundary_conds_periodic:
            if self.dim != 1.5:
                lattice_vector = tuple(component % self.shape[0] for component in lattice_vector)
            else:  # Don't do anything to the vertical direction in d=3/2 since that direction is NEVER periodic!
                lattice_vector = (lattice_vector[0] % self.shape[0], ) + lattice_vector[1:]
        else:
            raise NotImplementedError()

        return self._vertex_registers[lattice_vector]

    def get_link_register(self, lattice_vector: LatticeVector, unit_vector_label: LinkUnitVectorLabel) -> QuantumRegister:
        """
        Return the QuantumRegister for the link specified by lattice_vector and unit_vector_label.

        lattice_vector with a positive unit_vector_label specifies the link
        which is in the positive direction along the dimension specified by unit_vector_label
        from the vertex given by lattice vector. A negative unit_vector_label specifies the opposite link.

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
            elif self.boundary_conds_periodic:
                lattice_vector = tuple(comp % self.shape[dir_idx] for dir_idx, comp in enumerate(lattice_vector))
            else:
                # TODO implement handling of fixed boundary conditions.
                raise NotImplementedError()

        return self._link_registers[lattice_vector, unit_vector_label]

    def _get_single_plaquette(
            self,
            lattice_vector: LatticeVector,
            e1: LinkUnitVectorLabel,
            e2: LinkUnitVectorLabel
    ) -> Plaquette:
        """Convert (e1, e2) into unit vectors tuples to add to the lattice vectors."""
        # Sanity checks.
        e1_out_of_range = abs(e1) not in list(range(1, ceil(self.dim) + 1))
        e2_out_of_range = abs(e2) not in list(range(1, ceil(self.dim) + 1))
        if e1_out_of_range or e2_out_of_range:
            raise ValueError(
                f"To specify a single plaquette, e1 and e2 have to be valid unit vector labels. For the given dimension self.dim = {self.dim}, must be an int between 1 and {self.dim + 1}.\n"
                f"Got: e1 = {e1}, e2 = {e2}."
            )
        if abs(e1) == abs(e2):
            raise ValueError(f"The inputted edges e1:{e1} and e2:{e2} are not orthogonal and do not span a plaquette.")

        # Collect vertices and links.
        vertices_ccw = [lattice_vector]
        for step in [e1, e2, -e1]:
            vertices_ccw.append(self.add_unit_vector_to_vertex_vector(vertices_ccw[-1], step))
        link_steps_ccw = [e1, e2, -e1, -e2]
        vertex_regs = [self.get_vertex_register(v) for v in vertices_ccw]
        link_regs = [self.get_link_register(v, l) for v, l in zip(vertices_ccw, link_steps_ccw)]

        return Plaquette(
            link_registers=link_regs,
            vertex_registers=vertex_regs,
            bottom_left_vertex=vertices_ccw[0],
            plane=(e1, e2))

    def get_plaquette_registers(
            self,
            lattice_vector: LatticeVector,
            e1: Union[LinkUnitVectorLabel, None] = None,
            e2: Union[LinkUnitVectorLabel, None] = None
    ) -> Plaquette | List[Plaquette]:
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
            return self._get_single_plaquette(lattice_vector, e1, e2)
        elif only_one_link_given:  # Nonsense case.
            raise ValueError(
                "Provide no link directions to get all Plaquettes at a vertex, or exactly two link directions "
                f"to get one particular Plaquette. Received e1={e1} and e2={e2}."
            )
        else:  # Case where no link directions provided, need to construct ALL plaquettes.
            if (self.dim == 1.5 or self.dim == 2):  # Only one plaquette for these geometries.
                return self._get_single_plaquette(lattice_vector, 1, 2)
            elif self.dim > 2:  # Generic case with multiple planes.
                set_of_planes = set((i, j) if i < j else None for i, j in product(range(1, ceil(self.dim + 1)), range(1, ceil(self.dim + 1))))
                not_none_lambda = lambda x: not x is None
                all_planes = sorted(list(filter(not_none_lambda, set_of_planes)))  # Need to strip out a spurious None
                return [self._get_single_plaquette(lattice_vector, plane[0], plane[1]) for plane in all_planes]            

    def apply_trotter_step(self, evol_type: str = 'both'):
        """
        Apply a single trotter evolution step to the entire lattice.

        evol_type == 'm' for just Magnetic Hamiltonian evolution.
        evol_type == 'e for just Electric Hamiltonian evolution.
        evol_type == anything else for both.
        """
        # TODO surface a step_size argument?
        if evol_type == "m":
            self._trotter_step_magnetic()
        elif evol_type == "e":
            self._trotter_step_electric()
        else:
            self._trotter_step_magnetic()
            self._trotter_step_electric()

    def _trotter_step_electric(self):
        # TODO parallelize the logic.
        raise NotImplementedError()

    def _trotter_step_magnetic(self):
        # TODO parallelize the logic.
        raise NotImplementedError()

    @property
    def n_qubits_per_link(self) -> int:
        """Number of qubits per link register."""
        return self._n_qubits_per_link

    @property
    def n_qubits_per_vertex(self) -> int:
        """Number of qubits per vertex register."""
        return self._n_qubits_per_vertex

    @property
    def vertex_register_keys(self) -> list[LatticeVector]:
        """Return all of the lattice vectors uniquely labeling lattice vertices."""
        return sorted(list(self._vertex_registers.keys()))

    @property
    def link_register_keys(self) -> list[Tuple[LatticeVector, LinkUnitVectorLabel]]:
        """Return all of the (LatticeVector, LinkUnitVectorLabel) uniquely labeling lattice links."""
        return sorted(list(self._link_registers.keys()))

    # TODO Maybe don't implement this in favor of keeping
    # properties which return the actual internal dict keys?
    @property
    def link_unit_vectors(self) -> set[int]:
        """Return all the dimension labels corresponding to positive unit vectors."""
        raise NotImplementedError()

    @property
    def boundary_conds_periodic(self) -> bool:
        """Return whether the lattice has periodic boundary conditions."""
        return self._periodic_boundary_conds

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

        if self.boundary_conds_periodic and self.dim != 1.5:
            result_vector = tuple(comp % self.shape[0] for comp in result_vector)
        elif self.boundary_conds_periodic:
            # Vertical direction is NEVER periodic in d=3/2.
            result_vector = (result_vector[0] % self.shape[0], result_vector[1])

        return result_vector


def test_d_3_2_lattice_initialization():
    """Check creation of lattice with string or int reps of d=3/2."""
    print("Checking that lattice initialization works for d=3/2...")
    sizes = [7, 2]
    expected_shapes = [(7, 2), (2, 2)]
    dims = [3/2, 1.5, "3/2"]
    expected_dim = 1.5
    for current_size, expected_shape in zip(sizes, expected_shapes):
        for current_dim in dims:
            print(f"LatticeRegisters(dim={current_dim}, size={current_size})")
            lattice = LatticeRegisters(dim=current_dim, size=current_size)
            print(f"Results:\n\tlattice.shape = {lattice.shape}\n\tlattice.dim = {lattice.dim}")
            assert lattice.shape == expected_shape
            assert lattice.dim == expected_dim
            assert lattice.n_qubits_per_link == 1  # Default
            assert lattice.n_qubits_per_vertex == 1  # Default
            print("Test passed.")


def test_d_2_lattice_initialization():
    """Check creation of lattice with d=2."""
    print("Checking that lattice initialization works for d=2...")
    lattice = LatticeRegisters(dim=2, size=4, n_qubits_per_link=3, n_qubits_per_vertex=4)
    assert lattice.shape == (4, 4)
    assert lattice.dim == 2
    assert lattice.n_qubits_per_link == 3
    assert lattice.n_qubits_per_vertex == 4
    print("Test passed.")


def test_d_3_lattice_initialization():
    """Check creation of lattice with d=3."""
    print("Checking that lattice initialization works for d=3...")
    lattice = LatticeRegisters(dim=3, size=8, n_qubits_per_link=5, n_qubits_per_vertex=1)
    assert lattice.shape == (8, 8, 8)
    assert lattice.dim == 3
    assert lattice.n_qubits_per_link == 5
    assert lattice.n_qubits_per_vertex == 1
    print("Test passed.")


def test_bad_lattice_dim_fails():
    """Check that a bad dim causes a ValueError."""
    print("Checking that dim = 0.5 causes a ValueError.")
    try:
        LatticeRegisters(dim=0.5, size=7)
    except ValueError as e:
        print(f"It does! Error message: {e}")
    else:
        assert False
    print("Test passed.")


def test_plaquette_validation_works():
    """Check that bad plaquette link or vertex lengths fails."""
    print("4 links and 3 vertices should fail...")
    try:
        link_regs = (QuantumRegister(1),) * 4
        vertex_regs = (QuantumRegister(1),) * 3
        lattice_vector = (0, 0, 1)
        xy_plane = (1, 2)
        p = Plaquette(
            link_registers=link_regs,
            vertex_registers=vertex_regs,
            bottom_left_vertex=lattice_vector,
            plane=xy_plane
        )
        print(p)
    except ValueError as e:
        print(f"It does! Error message: {e}")
    else:
        assert False
    print("Test passed.")


def test_link_and_vertex_register_initialization(dims: DimensionalitySpecifier):
    """
    Check that we get link and vertex registers with the expected number of qubits.

    Also check that periodic boundary conditions and negative unit vector indexing work.
    For dims = 1.5, check that no links are made above or below the lattice (in the 2-direction).
    """
    # Initialize lattice.
    lattice = LatticeRegisters(dim=dims, size=3, n_qubits_per_link=2, n_qubits_per_vertex=3)
    print(f"Checking LatticeRegisters(dim={dims}, size=3, n_qubits_per_link={lattice.n_qubits_per_link}, n_qubits_per_vertex={lattice.n_qubits_per_vertex}) gives registers "
          f"with {lattice.n_qubits_per_link} qubits per link and {lattice.n_qubits_per_vertex} qubits per vertex.")
    print("Link registers internal dict:\n", lattice._link_registers)
    print("Vertex registers internal dict:\n", lattice._vertex_registers)

    # Check vertex and link indexing data.
    expected_vertex_vectors = set(product(*[[i for i in range(axis_length)] for axis_length in lattice.shape]))
    assert not isinstance(dims, str)  # Test ignores possibility of string dimensionality specifier.
    expected_lattice_unit_vector_labels = [i + 1 for i in range(ceil(dims))]
    print("Checking that correct set of vertex vectors was obtained...")
    print(f"\texpected = {expected_vertex_vectors}")
    print(f"\tobtained = {lattice._all_vertex_vectors}")
    assert expected_vertex_vectors == lattice._all_vertex_vectors
    print("Test passed.")
    print("Checking that correct set of lattice unit vector labels was obtained,..")
    print(f"\texpected = {expected_lattice_unit_vector_labels}")
    print(f"\tobtained = {lattice._lattice_unit_vector_labels}")
    assert expected_lattice_unit_vector_labels == lattice._lattice_unit_vector_labels
    print("Test passed.")

    # Check that expected registers are initialized.
    # This means for dim = 1.5 checking there aren't links above
    # or below the lattice. 
    # Generically, we also check that negative indexing works.
    for lattice_vector in lattice._all_vertex_vectors:
        current_vertex_register = lattice.get_vertex_register(lattice_vector)
        print(f"Checking vertex (x={lattice_vector[0]}, y={lattice_vector[1]})...")
        print(f"Expected {lattice.n_qubits_per_vertex} qubits.")
        print(f"Encountered {len(current_vertex_register)} qubits.")
        assert len(current_vertex_register) == lattice.n_qubits_per_vertex
        print("Test passed.")
        for link_unit_vector_label in expected_lattice_unit_vector_labels:
            if dims != 1.5 or link_unit_vector_label != VERTICAL_DIR_LABEL:
                # No special handling for d=3/2 required.
                current_link_register = lattice.get_link_register(lattice_vector, link_unit_vector_label)
                print(f"Checking link (x={lattice_vector[0]}, y={lattice_vector[1]}, e={link_unit_vector_label})...")
                print(f"Expected {lattice.n_qubits_per_link} qubits.")
                print(f"Encountered {len(current_link_register)} qubits.")
                assert len(lattice.get_link_register(lattice_vector, link_unit_vector_label)) == lattice.n_qubits_per_link
                print("Test passed.")

                print(f"Checking that the link with start vertex=({lattice_vector}), link dir={link_unit_vector_label}) "
                      f"== link wit start vertex=({lattice_vector} + link dir), link dir={-link_unit_vector_label})...")
                unit_vector = tuple(0 if component != (link_unit_vector_label - 1) else 1 for component in range(ceil(dims)))
                shifted_lattice_vector = tuple(lattice_vec_comp + unit_vec_comp for lattice_vec_comp, unit_vec_comp in zip(lattice_vector, unit_vector))
                assert current_link_register == lattice.get_link_register(shifted_lattice_vector, -link_unit_vector_label)
                print("Test passed.")
            else:
                # Special handling required for d=3/2 to check
                # that there's no link register "below"
                # or above the lattice.
                is_lower_vertex = lattice_vector[1] == 0
                if is_lower_vertex:
                    try:
                        print("Checking there's no link register below the chain at "
                              f"vertex (x={lattice_vector[0]}, y={lattice_vector[1]})...")
                        lattice.get_link_register(lattice_vector, -link_unit_vector_label)
                    except KeyError as e:
                        print(f"Test passed. Raised KeyError: {e}")
                        pass
                    else:
                        assert False  # Should have raised an KeyError.
                else:
                    try:
                        print("Checking there's no link register above the chain at "
                              f"vertex (x={lattice_vector[0]}, y={lattice_vector[1]})...")
                        lattice.get_link_register(lattice_vector, link_unit_vector_label)
                    except KeyError as e:
                        print(f"Test passed. Raised KeyError: {e}")
                        pass
                    else:
                        assert False  # Should have raised an KeyError.


def test_get_vertex_register_keys(dims: DimensionalitySpecifier, size: int):
    """Check that we get the expected vertex register keys."""
    # Constructed expected results
    if dims == 3:
        expected_vertices = sorted(list((i, j, k) for i, j, k in product(range(size), range(size), range(size))))
    if dims == 2:
        expected_vertices = sorted(list((i, j) for i, j in product(range(size), range(size))))
    if dims == 1.5:
        assert VERTICAL_NUM_VERTICES_D_THREE_HALVES == 2
        expected_vertices = sorted(list(((i, j) for i, j in product(
            range(size), range(VERTICAL_NUM_VERTICES_D_THREE_HALVES)))))

    # Initialize lattice.
    lattice = LatticeRegisters(dim=dims, size=size, n_qubits_per_link=4, n_qubits_per_vertex=1)
    print(f"Checking LatticeRegisters(dim={dims}, size={size}, n_qubits_per_link={lattice.n_qubits_per_link}, n_qubits_per_vertex={lattice.n_qubits_per_vertex}) has vertices:\n"
          f"{expected_vertices}")
    print("Got the following vertices:\n", lattice.vertex_register_keys)

    # The actual tests.
    assert lattice.vertex_register_keys == expected_vertices
    for vertex_register_key in lattice.vertex_register_keys:
        assert lattice.get_vertex_register(vertex_register_key).name == f"v:{vertex_register_key}"
    print("Test passed.")


def test_get_link_register_keys(dims: DimensionalitySpecifier, size: int):
    """Check that we get the expected link register keys."""
    # Construct expected results
    if dims == 3:
        expected_vertices = sorted(list((i, j, k) for i, j, k in product(range(size), range(size), range(size))))
        expected_link_unit_vector_labels = range(1, 4)
    elif dims == 2:
        expected_vertices = sorted(list((i, j) for i, j in product(range(size), range(size))))
        expected_link_unit_vector_labels = range(1, 3)
    elif dims == 1.5:
        assert VERTICAL_NUM_VERTICES_D_THREE_HALVES == 2
        expected_vertices = sorted(list((i, j) for i, j in product(
            range(size), range(VERTICAL_NUM_VERTICES_D_THREE_HALVES))))
        expected_link_unit_vector_labels = range(1, 3)
        expected_link_labels = sorted(list((vertex, unit_vector_dir) for vertex, unit_vector_dir in product(expected_vertices, expected_link_unit_vector_labels)))
    else:
        assert NotImplementedError(f"Test not implemented for dims = {dims}.")
    if dims == 3 or dims == 2:
        expected_link_labels = sorted(list((vertex, unit_vector_dir) for vertex, unit_vector_dir in product(expected_vertices, expected_link_unit_vector_labels)))
    else:
        # d = 3/2 case, there shouldn't be links above or below the chain.
        expected_link_labels = sorted([
            (vertex, unit_vector_dir) for vertex, unit_vector_dir in product(expected_vertices, expected_link_unit_vector_labels)
            if not (unit_vector_dir == VERTICAL_DIR_LABEL and vertex[1] == 1)])

    # Initialize lattice.
    lattice = LatticeRegisters(
        dim=dims, size=size, n_qubits_per_link=4, n_qubits_per_vertex=1)
    print(f"Checking LatticeRegisters(dim={dims}, size={size}, n_qubits_per_link={lattice.n_qubits_per_link}, n_qubits_per_vertex={lattice.n_qubits_per_vertex}) has links:\n"
          f"{expected_link_labels}")
    print("Got the following links:\n", lattice.link_register_keys)

    # The actual test.
    assert lattice.link_register_keys == expected_link_labels
    for link_register_key in lattice.link_register_keys:
        vertex_vector = link_register_key[0]
        link_dir = link_register_key[1]
        assert lattice.get_link_register(lattice_vector=vertex_vector, unit_vector_label=link_dir).name == f"l:{link_register_key}"
    print("Test passed.")


def helper_test_plaquette_equivalence(plaquette1: Plaquette, plaquette2: Plaquette) -> list[bool]:
    """Helper to check that two plaquettes are equal."""
    plaq_equal = plaquette1.link_registers == plaquette2.link_registers
    vertex_equal = plaquette1.vertex_registers == plaquette2.vertex_registers
    bottom_left_equal = plaquette1.bottom_left_vertex == plaquette2.bottom_left_vertex
    plane_equal = plaquette1.plane == plaquette2.plane

    if (plaq_equal and vertex_equal and bottom_left_equal and plane_equal):
        return [True]
    else:
        return [plaq_equal, vertex_equal, bottom_left_equal, plane_equal]


def test_plaquette_value_error_for_non_register_input():
    print("Check that creating a Plaquette with non-register input fails.")
    try:
        Plaquette(link_registers=[1, 2, 3, 4], vertex_registers=[QuantumRegister(0) for _ in range(4)], bottom_left_vertex=(0, 0, 0), plane=(1, 2))
        Plaquette(link_registers=[QuantumRegister(0) for _ in range(4)], vertex_registers=[1, 2, 3, 4], bottom_left_vertex=(0, 0, 0), plane=(1, 2))
    except ValueError:
        pass
    else:
        assert False
    print("Test passed.")


def test_get_plaquette_registers(dims: DimensionalitySpecifier, size: int):
    # Construct lattice register
    lattice = LatticeRegisters(
        dim=dims, size=size, n_qubits_per_link=4, n_qubits_per_vertex=1)

    origin = ceil(dims)*(0,)

    # Grab a single plaquette from the origin. 
    plaquette_origin_xy = lattice.get_plaquette_registers(origin, 1, 2)

    # Construct the expected plaquette
    expected_steps_ccw = [1, 2, -1, -2]
    expected_vertices_ccw = [origin]
    for step in expected_steps_ccw[:-1]:
        expected_vertices_ccw.append(lattice.add_unit_vector_to_vertex_vector(expected_vertices_ccw[-1], unit_vec_dir=step))
    expected_vertex_regs = [lattice.get_vertex_register(v) for v in expected_vertices_ccw]
    expected_link_regs = [lattice.get_link_register(v, l) for v, l in zip(expected_vertices_ccw, expected_steps_ccw)]
    expected_plaquette = Plaquette(expected_link_regs, expected_vertex_regs, origin, (1, 2))

    print(f"Testing grabbing the origin plaquette spanned by e_x and e_y for dim={lattice.dim}")
    pass_lst = helper_test_plaquette_equivalence(expected_plaquette, plaquette_origin_xy)
    print(f"The truth list looks like this {pass_lst}\n")
    assert len(pass_lst) == 1
    print("Test passed!")

    # Grab the positive plaquettes. Only need to test for d=3.
    print(f"Testing grabbing all positive plaquettes for dim={lattice.dim}")
    if lattice.dim == 3:
        expected_pos_plaqs = [lattice.get_plaquette_registers(origin, 1, 2), lattice.get_plaquette_registers(origin, 1, 3), lattice.get_plaquette_registers(origin, 2, 3)]
        pass_lst = list(map(helper_test_plaquette_equivalence, lattice.get_plaquette_registers(origin), expected_pos_plaqs))
        print(f"The truth list looks like this {pass_lst}\n")
        assert len(pass_lst[0]) == 1
        print("Test passed!")

    # Now we check that we don't miss any registers when building plaquettes for the whole lattice.
    print(f"Testing grabbing all the lattice plaquettes for dim={lattice.dim} and size={size}.")
    print("The set of names of all registers across all plaquettes should match the set of names of all vertex and link registers in the lattice.")
    plaquette_lst = []
    for vertex_vector in lattice.vertex_register_keys:
        if vertex_vector[VERTICAL_DIR_LABEL - 1] > 0 and lattice.dim == 1.5:
            continue
        plaquette_lst.append(lattice.get_plaquette_registers(vertex_vector))

    if lattice.dim > 2:
        # Need to flatten plaquette list in this case.
        plaquette_lst = [plaquette for plaquette_group in plaquette_lst for plaquette in plaquette_group]

    lattice_all_vertex_names = set(reg.name for reg in lattice._vertex_registers.values())
    lattice_all_link_names = set(reg.name for reg in lattice._link_registers.values())
    for plaquette in plaquette_lst:
        print("Here is a plaquette: ", plaquette)
        for vertex_reg in plaquette.vertex_registers:
            print("Here is a vertex register: ", vertex_reg)
            assert vertex_reg.name in lattice_all_vertex_names
        for link_reg in plaquette.link_registers:
            assert link_reg.name in lattice_all_link_names
    print("Test passed!")


def test_len_0_vertices_ok_for_d_3_2():
    """
    There is no need for vertex registers in d = 3/2.

    To accomplish this, we check that it is possible to initialize the lattice
    with vertex registers that have zero qubits.
    """
    lattice = LatticeRegisters(1.5, 2, n_qubits_per_link=2, n_qubits_per_vertex=0)
    print(f"Should have created a lattice with {lattice.n_qubits_per_vertex} per vertex register. Checking...")
    for vertex_vector in lattice.vertex_register_keys:
        current_reg = lattice.get_vertex_register(vertex_vector)
        assert current_reg.size == 0
        print(f"Confirmed len({current_reg}) == 0.")

    print("Test passed.")


def test_add_unit_vector_to_vertex_vector():
    """Check some additions, assuming periodic boundary conditions."""
    size = 5
    dims = [1.5, 2, 3]
    for dim in dims:
        print(f"Testing {dim}D lattice vector addition...")
        lattice = LatticeRegisters(dim, 5, boundary_conds="periodic")
        if dim == 1.5:
            vertices = [(i, 0) for i in range(size)] + [(i, 1) for i in range(size)]
        elif dim == 2:
            vertices = [(i, j) for i, j in product(range(size), range(size))]
        else:  # dim == 3
            vertices = [(i, j, k) for i, j, k in product(range(size), range(size), range(size))]
        for vertex_vector in vertices:
            for idx, link_dir in enumerate(range(1, ceil(dim))):
                unit_vector = tuple(1 if idx == link_dir - 1 else 0 for idx in range(ceil(dim)))
                expected_vector = tuple((v + u) % size for v, u in zip(vertex_vector, unit_vector))
                print(f"Checking {vertex_vector} + dir-{link_dir} unit vector == {expected_vector}")
                assert lattice.add_unit_vector_to_vertex_vector(vertex_vector, link_dir) == expected_vector

                negative_unit_vector = tuple(-1 if idx == link_dir - 1 else 0 for idx in range(ceil(dim)))
                expected_vector = tuple((v + u) % size for v, u in zip(vertex_vector, negative_unit_vector))
                print(f"Checking {vertex_vector} - dir-{link_dir} unit vector == {expected_vector}")
                assert lattice.add_unit_vector_to_vertex_vector(vertex_vector, -link_dir) == expected_vector
        print("Test passed.")


def test_get_vertex_register_pbc():
    """Check that fetching vertex registers works with periodic boundary conditions."""
    lattice = LatticeRegisters(2, 4)
    print("Checking that Vertex (6, 8) fetches the register v:(2, 0) on a 2D lattice of size 4...")
    assert lattice.get_vertex_register(lattice_vector=(6, 8)).name == "v:(2, 0)"

    lattice = LatticeRegisters(1.5, 6)
    print("Checking that Vertex (-3, 1) fetches the register v:(3, 1) on a 1.5D lattice of size 6....")
    assert lattice.get_vertex_register(lattice_vector=(-3, 1)).name == "v:(3, 1)"

    print("Checking that Vertex (1, 2) still raises a KeyError on a 1.5D lattice because of no pbc in the vertical direction...")
    try:
        lattice.get_vertex_register(lattice_vector=(1, 2))
    except KeyError:
        pass
    else:
        assert False

    print("Tests of vertex indexing under periodic boundary conditions passed.")


def run_tests():
    """
    Run tests.

    Add tests here when implementing new functionality.
    """
    print()
    test_d_3_2_lattice_initialization()
    print()
    test_bad_lattice_dim_fails()
    print()
    test_d_2_lattice_initialization()
    print()
    test_d_3_lattice_initialization()
    print()
    test_plaquette_validation_works()
    print()
    test_link_and_vertex_register_initialization(dims=3/2)
    print()
    test_link_and_vertex_register_initialization(dims=2)
    print()
    test_link_and_vertex_register_initialization(dims=3)
    print()
    test_get_vertex_register_keys(3, 2)
    test_get_vertex_register_keys(2, 3)
    test_get_vertex_register_keys(1.5, 10)
    print()
    test_get_link_register_keys(3, 2)
    test_get_link_register_keys(2, 4)
    test_get_link_register_keys(1.5, 16)
    test_plaquette_value_error_for_non_register_input()
    test_get_plaquette_registers(1.5, 4)
    test_get_plaquette_registers(2, 4)
    test_get_plaquette_registers(3, 4)
    print()
    test_len_0_vertices_ok_for_d_3_2()
    print()
    test_add_unit_vector_to_vertex_vector()
    print()
    test_get_vertex_register_pbc()

    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
