from __future__ import annotations
from math import ceil
from itertools import product
import pytest
from ymcirc.conventions import IrrepBitmap, VertexMultiplicityBitmap
from ymcirc._abstract.lattice_data import (
    Plaquette, DimensionalitySpecifier,
    VERTICAL_DIR_LABEL, VERTICAL_NUM_VERTICES_D_THREE_HALVES
)
from ymcirc.lattice_registers import LatticeRegisters


def _helper_test_plaquettes_have_same_registers(plaquette1: Plaquette, plaquette2: Plaquette) -> list[bool]:
    """Helper to check that two possibly different plaquettes instances have the same registers."""
    # Uses "is" for checking register equality to make sure these
    # reference the same register object in memory.
    # Always expecting 4 link and vertex registers per plaquette.
    plaq_equal = all(plaquette1.active_links[idx] is plaquette2.active_links[idx] for idx in range(4))
    vertex_equal = all(plaquette1.vertices[idx] is plaquette2.vertices[idx] for idx in range(4))
    bottom_left_equal = plaquette1.bottom_left_vertex == plaquette2.bottom_left_vertex
    plane_equal = plaquette1.plane == plaquette2.plane

    if (plaq_equal and vertex_equal and bottom_left_equal and plane_equal):
        return [True]
    else:
        return [plaq_equal, vertex_equal, bottom_left_equal, plane_equal]


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
            lattice = LatticeRegisters(dimensions=current_dim, size=current_size)
            print(f"Results:\n\tlattice.shape = {lattice.shape}\n\tlattice.dim = {lattice.dim}")
            assert lattice.shape == expected_shape
            assert lattice.dim == expected_dim
            assert lattice.n_qubits_per_link == 1  # Default
            assert lattice.n_qubits_per_vertex == 0  # Default


def test_d_2_lattice_initialization():
    """Check creation of lattice with d=2."""
    print("Checking that lattice initialization works for d=2...")
    lattice = LatticeRegisters(dimensions=2, size=4, n_qubits_per_link=3, n_qubits_per_vertex=4)
    assert lattice.shape == (4, 4)
    assert lattice.dim == 2
    assert lattice.n_qubits_per_link == 3
    assert lattice.n_qubits_per_vertex == 4


def test_d_3_lattice_initialization():
    """Check creation of lattice with d=3."""
    print("Checking that lattice initialization works for d=3...")
    lattice = LatticeRegisters(dimensions=3, size=8, n_qubits_per_link=5, n_qubits_per_vertex=1)
    assert lattice.shape == (8, 8, 8)
    assert lattice.dim == 3
    assert lattice.n_qubits_per_link == 5
    assert lattice.n_qubits_per_vertex == 1


def test_bad_lattice_dim_fails():
    """Check that a bad dim causes a ValueError."""
    print("Checking that dim = 0.5 causes a ValueError.")
    with pytest.raises(ValueError) as e_info:
        LatticeRegisters(dimensions=0.5, size=7)


@pytest.mark.parametrize("dims", [3/2, 2, 3])
def test_get_register_traversal_order(dims):
    """Check that the traversal order for register addresses is as expected."""
    lattice = LatticeRegisters(dimensions=dims, size=3)
    expected_register_traversal_order_dict = {
        3/2: [
            [(0, 0), [((0, 0), 1), ((0, 0), 2)]],
            [(0, 1), [((0, 1), 1)]],
            [(1, 0), [((1, 0), 1), ((1, 0), 2)]],
            [(1, 1), [((1, 1), 1)]],
            [(2, 0), [((2, 0), 1), ((2, 0), 2)]],
            [(2, 1), [((2, 1), 1)]]
        ],
        2: [
            [(0, 0), [((0, 0), 1), ((0, 0), 2)]],
            [(0, 1), [((0, 1), 1), ((0, 1), 2)]],
            [(0, 2), [((0, 2), 1), ((0, 2), 2)]],
            [(1, 0), [((1, 0), 1), ((1, 0), 2)]],
            [(1, 1), [((1, 1), 1), ((1, 1), 2)]],
            [(1, 2), [((1, 2), 1), ((1, 2), 2)]],
            [(2, 0), [((2, 0), 1), ((2, 0), 2)]],
            [(2, 1), [((2, 1), 1), ((2, 1), 2)]],
            [(2, 2), [((2, 2), 1), ((2, 2), 2)]]
        ],
        3: [
            [(0, 0, 0), [((0, 0, 0), 1), ((0, 0, 0), 2), ((0, 0, 0), 3)]],
            [(0, 0, 1), [((0, 0, 1), 1), ((0, 0, 1), 2), ((0, 0, 1), 3)]],
            [(0, 0, 2), [((0, 0, 2), 1), ((0, 0, 2), 2), ((0, 0, 2), 3)]],
            [(0, 1, 0), [((0, 1, 0), 1), ((0, 1, 0), 2), ((0, 1, 0), 3)]],
            [(0, 1, 1), [((0, 1, 1), 1), ((0, 1, 1), 2), ((0, 1, 1), 3)]],
            [(0, 1, 2), [((0, 1, 2), 1), ((0, 1, 2), 2), ((0, 1, 2), 3)]],
            [(0, 2, 0), [((0, 2, 0), 1), ((0, 2, 0), 2), ((0, 2, 0), 3)]],
            [(0, 2, 1), [((0, 2, 1), 1), ((0, 2, 1), 2), ((0, 2, 1), 3)]],
            [(0, 2, 2), [((0, 2, 2), 1), ((0, 2, 2), 2), ((0, 2, 2), 3)]],
            [(1, 0, 0), [((1, 0, 0), 1), ((1, 0, 0), 2), ((1, 0, 0), 3)]],
            [(1, 0, 1), [((1, 0, 1), 1), ((1, 0, 1), 2), ((1, 0, 1), 3)]],
            [(1, 0, 2), [((1, 0, 2), 1), ((1, 0, 2), 2), ((1, 0, 2), 3)]],
            [(1, 1, 0), [((1, 1, 0), 1), ((1, 1, 0), 2), ((1, 1, 0), 3)]],
            [(1, 1, 1), [((1, 1, 1), 1), ((1, 1, 1), 2), ((1, 1, 1), 3)]],
            [(1, 1, 2), [((1, 1, 2), 1), ((1, 1, 2), 2), ((1, 1, 2), 3)]],
            [(1, 2, 0), [((1, 2, 0), 1), ((1, 2, 0), 2), ((1, 2, 0), 3)]],
            [(1, 2, 1), [((1, 2, 1), 1), ((1, 2, 1), 2), ((1, 2, 1), 3)]],
            [(1, 2, 2), [((1, 2, 2), 1), ((1, 2, 2), 2), ((1, 2, 2), 3)]],
            [(2, 0, 0), [((2, 0, 0), 1), ((2, 0, 0), 2), ((2, 0, 0), 3)]],
            [(2, 0, 1), [((2, 0, 1), 1), ((2, 0, 1), 2), ((2, 0, 1), 3)]],
            [(2, 0, 2), [((2, 0, 2), 1), ((2, 0, 2), 2), ((2, 0, 2), 3)]],
            [(2, 1, 0), [((2, 1, 0), 1), ((2, 1, 0), 2), ((2, 1, 0), 3)]],
            [(2, 1, 1), [((2, 1, 1), 1), ((2, 1, 1), 2), ((2, 1, 1), 3)]],
            [(2, 1, 2), [((2, 1, 2), 1), ((2, 1, 2), 2), ((2, 1, 2), 3)]],
            [(2, 2, 0), [((2, 2, 0), 1), ((2, 2, 0), 2), ((2, 2, 0), 3)]],
            [(2, 2, 1), [((2, 2, 1), 1), ((2, 2, 1), 2), ((2, 2, 1), 3)]],
            [(2, 2, 2), [((2, 2, 2), 1), ((2, 2, 2), 2), ((2, 2, 2), 3)]],
        ],
    }

    assert expected_register_traversal_order_dict[dims] == lattice.get_traversal_order()


@pytest.mark.parametrize("dims", [3/2, 2, 3])
def test_link_and_vertex_register_initialization(dims: DimensionalitySpecifier):
    """
    Check that we get link and vertex registers with the expected number of qubits.

    Also check that periodic boundary conditions and negative unit vector indexing work.
    For dims = 1.5, check that no links are made above or below the lattice (in the 2-direction).
    """
    # Initialize lattice.
    lattice = LatticeRegisters(dimensions=dims, size=3, n_qubits_per_link=2, n_qubits_per_vertex=3)
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
    print("Checking that correct set of lattice unit vector labels was obtained,..")
    print(f"\texpected = {expected_lattice_unit_vector_labels}")
    print(f"\tobtained = {lattice._lattice_unit_vector_labels}")

    assert expected_lattice_unit_vector_labels == lattice._lattice_unit_vector_labels

    # Check that expected registers are initialized.
    # This means for dim = 1.5 checking there aren't links above
    # or below the lattice. 
    # Generically, we also check that negative indexing works.
    for lattice_vector in lattice._all_vertex_vectors:
        current_vertex_register = lattice.get_vertex(lattice_vector)
        print(f"Checking vertex (x={lattice_vector[0]}, y={lattice_vector[1]})...")
        print(f"Expected {lattice.n_qubits_per_vertex} qubits.")
        print(f"Encountered {len(current_vertex_register)} qubits.")
        assert len(current_vertex_register) == lattice.n_qubits_per_vertex
        
        for link_unit_vector_label in expected_lattice_unit_vector_labels:
            if dims != 1.5 or link_unit_vector_label != VERTICAL_DIR_LABEL:
                # No special handling for d=3/2 required.
                current_link_register = lattice.get_link((lattice_vector, link_unit_vector_label))
                print(f"Checking link (x={lattice_vector[0]}, y={lattice_vector[1]}, e={link_unit_vector_label})...")
                print(f"Expected {lattice.n_qubits_per_link} qubits.")
                print(f"Encountered {len(current_link_register)} qubits.")
                assert len(lattice.get_link((lattice_vector, link_unit_vector_label))) == lattice.n_qubits_per_link

                print(f"Checking that the link with start vertex=({lattice_vector}), link dir={link_unit_vector_label}) "
                      f"== link wit start vertex=({lattice_vector} + link dir), link dir={-link_unit_vector_label})...")
                unit_vector = tuple(0 if component != (link_unit_vector_label - 1) else 1 for component in range(ceil(dims)))
                shifted_lattice_vector = tuple(lattice_vec_comp + unit_vec_comp for lattice_vec_comp, unit_vec_comp in zip(lattice_vector, unit_vector))
                assert current_link_register == lattice.get_link((shifted_lattice_vector, -link_unit_vector_label))
            else:
                # Special handling required for d=3/2 to check
                # that there's no link register "below"
                # or above the lattice.
                is_lower_vertex = lattice_vector[1] == 0
                if is_lower_vertex:
                    with pytest.raises(KeyError) as e_info:
                        print("Checking there's no link register below the chain at "
                              f"vertex (x={lattice_vector[0]}, y={lattice_vector[1]}) (should cause KeyError)")
                        lattice.get_link((lattice_vector, -link_unit_vector_label))
                else:
                    with pytest.raises(KeyError) as e_info:
                        print("Checking there's no link register above the chain at "
                              f"vertex (x={lattice_vector[0]}, y={lattice_vector[1]}) (should cause KeyError)")
                        lattice.get_link((lattice_vector, link_unit_vector_label))


@pytest.mark.parametrize(("dims", "size"), [
    (3, 2),
    (2, 3),
    (1.5, 10)
])
def test_get_vertex_keys(dims: DimensionalitySpecifier, size: int):
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
    lattice = LatticeRegisters(dimensions=dims, size=size, n_qubits_per_link=4, n_qubits_per_vertex=1)
    print(f"Checking LatticeRegisters(dim={dims}, size={size}, n_qubits_per_link={lattice.n_qubits_per_link}, n_qubits_per_vertex={lattice.n_qubits_per_vertex}) has vertices:\n"
          f"{expected_vertices}")
    print("Got the following vertices:\n", lattice.vertex_addresses)

    # The actual tests.
    assert lattice.vertex_addresses == expected_vertices
    for vertex_register_key in lattice.vertex_addresses:
        assert lattice.get_vertex(vertex_register_key).name == f"v:{vertex_register_key}"


@pytest.mark.parametrize(("dims", "size"), [
    (3, 2),
    (2, 4),
    (1.5, 16)
])
def test_get_link_keys(dims: DimensionalitySpecifier, size: int):
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
        dimensions=dims, size=size, n_qubits_per_link=4, n_qubits_per_vertex=1)
    print(f"Checking LatticeRegisters(dim={dims}, size={size}, n_qubits_per_link={lattice.n_qubits_per_link}, n_qubits_per_vertex={lattice.n_qubits_per_vertex}) has links:\n"
          f"{expected_link_labels}")
    print("Got the following links:\n", lattice.link_addresses)

    # The actual test.
    assert lattice.link_addresses == expected_link_labels
    for link_register_key in lattice.link_addresses:
        vertex_vector = link_register_key[0]
        link_dir = link_register_key[1]
        assert lattice.get_link((vertex_vector, link_dir)).name == f"l:{link_register_key}"


@pytest.mark.parametrize(("dims", "size"), [
    (1.5, 4),
    (2, 4),
    (3, 4)
])
def test_get_plaquettes(dims: DimensionalitySpecifier, size: int):
    print(f"Testing get_plaquettes for d={dims}, size={size}.")
    # Construct lattice register
    lattice = LatticeRegisters(
        dimensions=dims, size=size, periodic_boundary_conds=True, n_qubits_per_link=4, n_qubits_per_vertex=1)

    origin = ceil(dims)*(0,)

    # Grab a single plaquette from the origin.
    plaquette_origin_xy = lattice.get_plaquettes(origin, 1, 2)

    # Construct the expected plaquette
    expected_steps_ccw = [1, 2, -1, -2]
    expected_vertices_ccw = [origin]
    for step in expected_steps_ccw[:-1]:
        expected_vertices_ccw.append(lattice.add_unit_vector_to_vertex_vector(expected_vertices_ccw[-1], unit_vec_dir=step))
    expected_vertex_regs = tuple([lattice.get_vertex(v) for v in expected_vertices_ccw])
    expected_active_link_regs = tuple([lattice.get_link((v, l)) for v, l in zip(expected_vertices_ccw, expected_steps_ccw)])
    expected_control_regs_dict = {
        1.5: {
            (0, 0): {
                ((0, 0), -1)
            },
            (0, 1): {
                ((0, 1), -1)
            },
            (1, 0): {
                ((1, 0), 1)
            },
            (1, 1): {
                ((1, 1), 1)
            }
        },
        2: {
            (0, 0): {
                ((0, 0), -1),
                ((0, 0), -2)
            },
            (0, 1): {
                ((0, 1), -1),
                ((0, 1), 2)
            },
            (1, 0): {
                ((1, 0), 1),
                ((1, 0), -2)
            },
            (1, 1): {
                ((1, 1), 1),
                ((1, 1), 2)
            }
        },
        3: {
            (0, 0, 0): {
                ((0, 0, 0), -1),
                ((0, 0, 0), -2),
                ((0, 0, 0), 3),
                ((0, 0, 0), -3)
            },
            (0, 1, 0): {
                ((0, 1, 0), -1),
                ((0, 1, 0), 2),
                ((0, 1, 0), 3),
                ((0, 1, 0), -3)
            },
            (1, 0, 0): {
                ((1, 0, 0), 1),
                ((1, 0, 0), -2),
                ((1, 0, 0), 3),
                ((1, 0, 0), -3)
            },
            (1, 1, 0): {
                ((1, 1, 0), 1),
                ((1, 1, 0), 2),
                ((1, 1, 0), 3),
                ((1, 1, 0), -3)
            }
        }
    }
    expected_plaquette = Plaquette(lattice, origin, (1, 2))

    # Duplication test.
    print(
        f"Testing grabbing the origin plaquette spanned by e_x=1 and e_y=2 for dim={lattice.dim}"
        " didn't duplicate registers."
    )
    plaquette_nonduplication_test_boolean_mask = _helper_test_plaquettes_have_same_registers(expected_plaquette, plaquette_origin_xy)
    print(f"Plaquette nonduplication test checks: {plaquette_nonduplication_test_boolean_mask}\n")
    assert len(plaquette_nonduplication_test_boolean_mask) == 1

    # Link and vertex data test.
    print("Checking that we got the expected vertex, active, and control link registers.")
    assert plaquette_origin_xy.active_links == expected_active_link_regs, \
        f"Expected: {expected_active_link_regs}.\nRecieved: {plaquette_origin_xy.active_links}"
    assert plaquette_origin_xy.vertices == expected_vertex_regs, \
        f"Expected: {expected_vertex_regs}.\nRecieved: {plaquette_origin_xy.vertices}"
    assert plaquette_origin_xy.control_links.keys() == expected_control_regs_dict[dims].keys(), \
        f"Expected: {set(expected_control_regs_dict[dims].keys())}.\nRecieved: {set(plaquette_origin_xy.control_links.keys())}"
    for vertex in plaquette_origin_xy.control_links.keys():
        assert plaquette_origin_xy.control_links[vertex].keys() == expected_control_regs_dict[dims][vertex], \
            f"Expected: {expected_control_regs_dict[dims][vertex]}.\nRecieved: {set(plaquette_origin_xy.control_links[vertex].keys())}"

    # Grabbing the positive plaquettes test. Only need to test for d=3.
    print(f"Testing grabbing all positive plaquettes for dim={lattice.dim}")
    if lattice.dim == 3:
        expected_pos_plaqs = [lattice.get_plaquettes(origin, 1, 2), lattice.get_plaquettes(origin, 1, 3), lattice.get_plaquettes(origin, 2, 3)]
        plaquette_nonduplication_test_boolean_mask = list(map(_helper_test_plaquettes_have_same_registers, lattice.get_plaquettes(origin), expected_pos_plaqs))
        print(f"Plaquette nonduplication test checks: {plaquette_nonduplication_test_boolean_mask}\n")
        assert len(plaquette_nonduplication_test_boolean_mask[0]) == 1

    # Now we check that we don't miss any registers when building plaquettes for the whole lattice.
    print(f"Testing grabbing all the lattice plaquettes for dim={lattice.dim} and size={size}.")
    print("The set of names of all registers across all plaquettes should match the set of names of all vertex and link registers in the lattice.")
    plaquette_lst = []
    for vertex_vector in lattice.vertex_addresses:
        if vertex_vector[VERTICAL_DIR_LABEL - 1] > 0 and lattice.dim == 1.5:
            continue
        plaquette_lst.append(lattice.get_plaquettes(vertex_vector))

    if lattice.dim > 2:
        # Need to flatten plaquette list in this case.
        plaquette_lst = [plaquette for plaquette_group in plaquette_lst for plaquette in plaquette_group]

    lattice_all_vertex_names = set(reg.name for reg in lattice._vertex_registers.values())
    lattice_all_link_names = set(reg.name for reg in lattice._link_registers.values())
    for plaquette in plaquette_lst:
        print("Here is a plaquette: ", plaquette)
        for vertex_reg in plaquette.vertices:
            print("Here is a vertex register: ", vertex_reg)
            assert vertex_reg.name in lattice_all_vertex_names
        for link_reg in plaquette.active_links:
            assert link_reg.name in lattice_all_link_names


def test_get_registers_in_local_hamiltonian_order():
    print("Confirm that we get registers ordered by vertices, active links, then control links.")
    lattice = LatticeRegisters(dimensions=2, size=3)
    result_register_list = lattice.get_registers_in_local_hamiltonian_order(
        lattice_vector=(0, 0),
        e1=1,
        e2=2)
    plaquette = lattice.get_plaquettes(
        lattice_vector=(0, 0),
        e1=1,
        e2=2)
    expected_num_controls = 8
    expected_num_total_regs = expected_num_controls + (2*4)
    # Initialze the following test data.
    expected_control_regs = []
    actual_num_controls = 0  # Initialize this.
    for v in plaquette.control_links:
        actual_num_controls += len(v)
        for link_address, reg in plaquette.control_links[v].items():
            expected_control_regs.append(reg)

    # The actual tests start here.
    print("Checking that we got the right number of registers...")
    assert expected_num_total_regs == len(result_register_list), f"Expected {expected_num_total_regs}, encountered {len(result_register_list)}."

    print("Scanning through individual result registers...")
    for idx, reg in enumerate(result_register_list[:4]):
        print(f"Vertex: {reg} == {plaquette.vertices[idx]}?")
        assert reg == plaquette.vertices[idx], f"Register mismatch, {reg} != {plaquette.vertices[idx]}"

    for idx, reg in enumerate(result_register_list[4:8]):
        print(f"Active link: {reg} == {plaquette.active_links[idx]}?")
        assert reg == plaquette.active_links[idx], f"Register mismatch, {reg} != {plaquette.active_links[idx]}"

    print("Expecting controls:", expected_control_regs)
    for idx, reg in enumerate(result_register_list[8:]):
        # TODO should check ordering of control registers.
        print(f"Control link {reg} in list of controls?")
        assert reg in expected_control_regs, f"Register mismatch, {reg} not found in {expected_control_regs}."

    print("Total number of control links matches expectation?")
    assert actual_num_controls == expected_num_controls


def test_all_plaquettes_are_indexed_only_one_time():
    """
    Confirm when traversing a lattice that the correct number of plaquettes is indexed.

    Also, shouldn't have repeats.
    """
    dims = [1.5, 2, 3]
    sizes = [3, 2, 3]
    expected_num_plaquettes = [1*3, 1*(2**2), 3*(3**3)]
    for dim, size, expected_num_plaquettes in zip(dims, sizes, expected_num_plaquettes):
        lattice = LatticeRegisters(dim, size)
        print(f"Checking that a size {lattice.shape[0]} lattice in dim {lattice.dim} has {expected_num_plaquettes} plaquettes.")

        # Fetch all the plaquettes in the lattice.
        plaquettes = []
        for vertex in lattice.vertex_addresses:
            if vertex[1] == 1 and lattice.dim == 1.5:
                continue
            if dim < 3:
                plaquettes.append(lattice.get_plaquettes(vertex))
            else:  # Returns a list of multiple plaquettes in this case.
                plaquettes = plaquettes + lattice.get_plaquettes(vertex)

        # Check that total number of fetched plaquettes meets expectation.
        print(len(plaquettes))
        assert len(plaquettes) == expected_num_plaquettes

        # Check that there are no duplicated plaquettes.
        for idx, plaquette_one in enumerate(plaquettes):
            for plaquette_two in plaquettes[idx+1:]:
                print(f"Checking that Plaquettes built at {plaquette_one.bottom_left_vertex} and {plaquette_two.bottom_left_vertex} have different registers.")
                links_equal, vertices_equal, bottom_left_equal, plane_equal =  _helper_test_plaquettes_have_same_registers(plaquette_one, plaquette_two)
                assert links_equal is False and vertices_equal is False


def test_len_0_vertices_ok_for_d_3_2():
    """
    There is no need for vertex registers in d = 3/2.

    To accomplish this, we check that it is possible to initialize the lattice
    with vertex registers that have zero qubits.
    """
    lattice = LatticeRegisters(1.5, 2, n_qubits_per_link=2, n_qubits_per_vertex=0)
    print(f"Should have created a lattice with {lattice.n_qubits_per_vertex} per vertex register. Checking...")
    for vertex_vector in lattice.vertex_addresses:
        current_reg = lattice.get_vertex(vertex_vector)
        assert current_reg.size == 0
        print(f"Confirmed len({current_reg}) == 0.")


def test_add_unit_vector_to_vertex_vector():
    """Check some additions, assuming periodic boundary conditions."""
    size = 5
    dims = [1.5, 2, 3]
    for dim in dims:
        print(f"Testing {dim}D lattice vector addition...")
        lattice = LatticeRegisters(dim, 5, periodic_boundary_conds=True)
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


def test_get_vertex_pbc():
    """Check that fetching vertex registers works with periodic boundary conditions."""
    lattice = LatticeRegisters(2, 4)
    print("Checking that Vertex (6, 8) fetches the register v:(2, 0) on a 2D lattice of size 4...")
    assert lattice.get_vertex(lattice_vector=(6, 8)).name == "v:(2, 0)"

    lattice = LatticeRegisters(1.5, 6)
    print("Checking that Vertex (-3, 1) fetches the register v:(3, 1) on a 1.5D lattice of size 6....")
    assert lattice.get_vertex(lattice_vector=(-3, 1)).name == "v:(3, 1)"

    print("Checking that Vertex (1, 2) still raises a KeyError on a 1.5D lattice because of no pbc in the vertical direction.")
    with pytest.raises(KeyError) as e_info:
        lattice.get_vertex(lattice_vector=(1, 2))


def test_num_qubits_automatic_when_dicts_provided_to_lattice():
    """Shouldn't need to provide num qubits when this can be inferred from the state encoding maps."""
    # Case of link bitmap but no vertex bitmap
    print("Checking that correct number of qubits can automatically be inferred from bitstring encodings when provided...")
    irrep_trunc_dict: IrrepBitmap = {
        (0, 0, 0): "00",
        (1, 0, 0): "10",
        (0, 1, 0): "01"
    }
    lattice = LatticeRegisters(
        dimensions=2,
        size=3,
        link_bitmap=irrep_trunc_dict,
        vertex_bitmap=None
    )
    print("Created a lattice with no vertex bitmap and the following link irrep map:\n", irrep_trunc_dict)
    print("Expecting 2 qubits per link, and default value of 0 qubits per vertex.")
    for vertex, edge_dir in lattice.link_addresses:
        assert len(lattice.get_link((vertex, edge_dir))) == 2
        assert len(lattice.get_vertex(vertex)) == 0  # Default value

    # Case of vertex bitmap but no link bitmap.
    # 3 iweights implies lattice dim=3/2.
    vertex_multiplicity_bitmap: VertexMultiplicityBitmap = {
        0: "0",
        1: "1"
    }
    lattice = LatticeRegisters(
        dimensions="3/2",
        size=2,
        vertex_bitmap=vertex_multiplicity_bitmap,
        n_qubits_per_link=5
    )
    print("Created a lattice with no link irrep bitmap and the following vertex multiplicity bitmap:\n", vertex_multiplicity_bitmap)
    print("Expecting 1 qubits per vertex, and manually set value of 5 qubits per link.")
    for vertex, edge_dir in lattice.link_addresses:
        assert len(lattice.get_link((vertex, edge_dir))) == 5
        assert len(lattice.get_vertex(vertex)) == 1


def test_lattice_init_fails_when_vertex_bitmap_has_wrong_num_links():
    """
    Check for TypeError when vertex bitmap disagrees with lattice dim.
    Specific case: vertex bitmap lists 3 irreps per vertex for a dim-2
    attice (Should have 4 links).
    """
    # This bitmap implies d=3/2. Should fail for d=2 lattice.
    vertex_multiplicity_bitmap: VertexMultiplicityBitmap = {
        (((2, 1, 0,), (2, 1, 0), (2, 1, 0)), 1): "0",
        (((2, 1, 0,), (2, 1, 0), (2, 1, 0)), 2): "1"
    }
    print(f"Checking for TypeError when creating a dim-2 lattice with a bitmap that has 3 links per vertex:\nvertex bitmap = {vertex_multiplicity_bitmap}")
    with pytest.raises(TypeError) as e_info:
        LatticeRegisters(2, 2, vertex_bitmap=vertex_multiplicity_bitmap)


def test_value_error_for_link_bitmap_with_different_lengths():
    print(
        "Checking that creating a lattice with a link truncation bitmap that has"
        " different length bitstrings for different irreps will raise a ValueError."
    )
    bad_irrep_trunc_dict: IrrepBitmap = {
        (0, 0, 0): "001",
        (1, 0, 0): "10",
        (0, 1, 0): "01"
    }
    with pytest.raises(ValueError) as e_info:
        LatticeRegisters(dimensions=2, size=3, link_bitmap=bad_irrep_trunc_dict)


def test_value_error_for_vertex_bitmap_with_different_lengths():
    print(
        "Checking that creating a lattice with a vertex truncation bitmap that has"
        " different length bitstrings for various singlets will raise a ValueError."
    )
    # This bitmap implies dim = 1.5.
    bad_vertex_bitmap: VertexMultiplicityBitmap = {
        0: "0",
        1: "01"
    }
    with pytest.raises(ValueError) as e_info:
        LatticeRegisters(dimensions=1.5, size=3, vertex_bitmap=bad_vertex_bitmap)


def test_type_error_for_bad_vertex_and_link_bitmap_keys():
    print(
        "Check that a TypeError is raise if a dict with the wrong kinds of keys is used "
        "to try creating a LatticeRegisters instance."  
          )
    bad_irrep_trunc_dict: IrrepBitmap = {
        (0, 0): "00",  # Will fail because of non length 3 tuple key.
        (1, 0, 0): "10",
        (0, 1, 0): "01"
    }
    bad_vertex_bitmap: VertexMultiplicityBitmap = {
        (((2, 1, 0,), (2, 1, 0), (2, 1, 0)), 1): "0",
        (((2, 1, 0,), (2, 1, 0), (2, 1, 0))): "1"  # Will fail because missing multiplicity int.
    }

    with pytest.raises(TypeError) as e_info:
        LatticeRegisters(dimensions=2, size=3, link_bitmap=bad_irrep_trunc_dict)

    with pytest.raises(TypeError) as e_info:
        LatticeRegisters(dimensions=2, size=3, vertex_bitmap=bad_vertex_bitmap)


def test_get_bitmaps_from_lattice():
    print("Checking return 'None' when not defined...")
    assert LatticeRegisters(2, 3).link_bitmap is None
    assert LatticeRegisters(2, 3).vertex_bitmap is None

    print("Checking that a copy of the bitmaps are returned when they are defined...")
    irrep_trunc_dict: IrrepBitmap = {
        (0, 0, 0): "00",
        (1, 0, 0): "10",
        (0, 1, 0): "01"
    }
    vertex_bitmap: VertexMultiplicityBitmap = {
        0: "0",
        1: "1"
    }
    lattice = LatticeRegisters(4, 2, link_bitmap=irrep_trunc_dict, vertex_bitmap=vertex_bitmap)
    print(
        f"Should find that:\nlattice.link_bitmap == {irrep_trunc_dict}\n"
        f"lattice.vertex_bitmap == {vertex_bitmap}"
          )

    # The "==" check confirms the values of the dicts are equivalent,
    # while the "is not" check confirms that the two dicts are copies
    # occupying distinct portions of memory. Necessary to avoid accidentally
    # mutating internal data in the LatticeRegisters class.
    assert lattice._link_bitmap == lattice.link_bitmap and \
        lattice._link_bitmap is not lattice.link_bitmap
    assert lattice._vertex_bitmap == lattice.vertex_bitmap and \
        lattice._vertex_bitmap is not lattice.vertex_bitmap


def test_d_equals_2_lattice_from_bitmaps():
    """Check that dim 2 lattice can be created via bitmaps."""
    dims = 2
    vertex_bitmap = {
        0: '000',
        1: '001',
        2: '010',
        3: '011',
        4: '100',
        5: '101'
    }
    link_bitmap = {
        (0, 0, 0): '00',
        (1, 0, 0): '10',
        (1, 1, 0): '01'
    }
    print(f"Trying to create dim-{dims} lattice with bitmaps:\nvertex = {vertex_bitmap}\nlink = {link_bitmap}")
    lattice = LatticeRegisters(dimensions=dims, size=3, link_bitmap=link_bitmap, vertex_bitmap=vertex_bitmap)
    print(f"Created a dim-{lattice.dim} lattice with bitmaps:\nvertex = {lattice.vertex_bitmap}\nlink = {lattice.link_bitmap}.")
    assert lattice.vertex_bitmap == vertex_bitmap
    assert lattice.link_bitmap == link_bitmap


def test_n_qubits_whole_lattice():
    """Check that full lattice has the right number of qubits."""
    # Case format = (dim, n_qubits_per_vertex, n_qubits_per_link, size, expected_qubits_in_lattice)
    lattice_cases = [
        (1.5, 0, 1, 2, 6),
        (1.5, 1, 1, 2, 10),
        (2, 3, 5, 10, (10*10*3 + 2*10*10*5)),
        (3, 1, 2, 20, (20*20*20*1 + 3*20*20*20*2))
    ]
    for dims, n_qubits_per_vertex, n_qubits_per_link, size, expected_qubits_in_lattice in lattice_cases:
        print(
            f"Testing that total qubits in d={dims} lattice with size={size}, {n_qubits_per_link}"
            f" per link, and {n_qubits_per_vertex} qubits per vertex is equal to {expected_qubits_in_lattice}..."
        )
        lattice = LatticeRegisters(dimensions=dims, size=size, n_qubits_per_link=n_qubits_per_link, n_qubits_per_vertex=n_qubits_per_vertex)
        assert lattice.n_total_qubits == expected_qubits_in_lattice, f"Test failed: lattice has {lattice.n_total_qubits} qubits, expected {expected_qubits_in_lattice} qubits."


def test_n_plaquettes_whole_lattice():
    """Check that full lattice has the right number of plaquettes."""
    # Case format = (dim, size, expected_n_plaquettes = size^dim * (dim choose 2))
    lattice_cases = [
        (1.5, 2, 2),
        (1.5, 7, 7),
        (2, 2, 2*2*1),
        (2, 20, 20*20*1),
        (3, 2, 2*2*2*3),
        (3, 15, 15*15*15*3)
    ]
    for dims, size, expected_n_plaquettes in lattice_cases:
        print(
            f"Testing that total qubits in d={dims} lattice with size={size}, has {expected_n_plaquettes} unique plaquettes..."
        )
        lattice = LatticeRegisters(dimensions=dims, size=size)
        assert lattice.n_plaquettes == expected_n_plaquettes, f"Test failed: lattice has {lattice.n_plaquettes} unique plaquettes, expected {expected_n_plaquettes} unique plaquettes."


# TODO there need to be tests for control link ordering in higher dimensions.
def test_control_link_registers_have_correct_ordering():
    print("Checking that the ordering of control link registers matches expectations.\n")

    cases_dict = {
        "d=3/2, large lattice (no register repetitions)": {
            "lattice registers": LatticeRegisters(1.5, 3),
            "bottom left vertex": (1, 0),
            "expected control link names ordered": [
                "l:((0, 0), 1)",
                "l:((2, 0), 1)",
                "l:((2, 1), 1)",
                "l:((0, 1), 1)"
            ]
        },
        "d=3/2, small lattice (with register repetitions)": {
            "lattice registers": LatticeRegisters(1.5, 2),
            "bottom left vertex": (1, 0),
            "expected control link names ordered": [
                "l:((0, 0), 1)",
                "l:((0, 0), 1)",
                "l:((0, 1), 1)",
                "l:((0, 1), 1)"
            ]
        },
        "d=2, large lattice (no register repetitions)": {
            "lattice registers": LatticeRegisters(2, 10),
            "bottom left vertex": (4, 6),
            "expected control link names ordered": [
                "l:((3, 6), 1)",
                "l:((4, 5), 2)",
                "l:((5, 5), 2)",
                "l:((5, 6), 1)",
                "l:((5, 7), 1)",
                "l:((5, 7), 2)",
                "l:((4, 7), 2)",
                "l:((3, 7), 1)",
            ]
        },
        "d=2, small lattice (with register repetitions)": {
            "lattice registers": LatticeRegisters(2, 2),
            "bottom left vertex": (0, 0),
            "expected control link names ordered": [
                "l:((1, 0), 1)",
                "l:((0, 1), 2)",
                "l:((1, 1), 2)",
                "l:((1, 0), 1)",
                "l:((1, 1), 1)",
                "l:((1, 1), 2)",
                "l:((0, 1), 2)",
                "l:((1, 1), 1)",
            ]
        },
    }

    for case_name, case_data in cases_dict.items():
        print(f"Case: {case_name}.")
        result_plaquette = case_data["lattice registers"].get_plaquettes(
            lattice_vector=case_data["bottom left vertex"],
            e1=1,
            e2=2
        )
        result_control_link_registers_ordered = result_plaquette.control_links_ordered
        print(f"Examining plaquette with bottom-left vertex {result_plaquette.bottom_left_vertex} in the {result_plaquette.plane}-plane.")
        assert len(result_control_link_registers_ordered) == len(case_data["expected control link names ordered"]), "Length mismatch in number of control link registers."

        for control_link_register, expected_register_name in zip(result_control_link_registers_ordered, case_data["expected control link names ordered"]):
            print(f"Expected register {expected_register_name}, encountered {control_link_register.name}.")
            assert control_link_register.name == expected_register_name, "Link mismatch occured."
