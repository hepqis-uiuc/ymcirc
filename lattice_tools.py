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

        if n_qubits_per_link < 1 or n_qubits_per_vertex < 1:
            raise ValueError("Link and vertex registers must have positive integer number of qubits. "
                             f"n_qubits_per_link = {n_qubits_per_link}\nn_qubits_per_vertex = {n_qubits_per_vertex}.")

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
            self._vertex_registers[vertex_vector] = QuantumRegister(self._n_qubits_per_vertex)
        for vertex_vector in self._all_vertex_vectors:
            for link_unit_vector in self._lattice_unit_vector_labels:
                if self.dim == 1.5 and self._skip_links_above_or_below_d_equals_three_halves(vertex_vector, link_unit_vector):
                    continue  # Skip to next lattice direction.
                else:
                    self._link_registers[vertex_vector, link_unit_vector] = QuantumRegister(self._n_qubits_per_link)

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
        if any(component < 0 for component in lattice_vector):
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

    def get_plaquette_registers(self, lattice_vector: LatticeVector,
                                e1: Union[LinkUnitVectorLabel, None] = None,
                                e2: Union[LinkUnitVectorLabel, None] = None
                                ) -> Plaquette | List[Plaquette]:
        """
        Return the list of all "positive" Plaquettes associated with the vertex lattice_vector.

        The "positivity" convention is that the list of returned plaquettes corresponds to those
        defined by all pairs of orthogonal positive unit vectors at the vertex lattice_vector.

        If a particular plaquette is desired, this can be specified by either providing the
        link unit vector directions e1 and e2 to defining a plane. Sign is ignored when
        manually specifying the plane of a specific plaquette.

        Return plaquettes will all have lattice_vector as the "bottom-left" vertex.
        """
        raise NotImplementedError()

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

    # The actual test.
    assert lattice.vertex_register_keys == expected_vertices
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
    print("Test passed.")


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

    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
