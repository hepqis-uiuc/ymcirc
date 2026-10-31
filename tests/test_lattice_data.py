"""Tests of LatticeDef's geometry, which it takes from pyclebsch's LatticeDef."""

from __future__ import annotations

import pytest

from ymcirc._abstract.lattice_data import (
    LatticeData,
    LatticeDef,
    LatticeVector,
    LinkAddress,
    Plane,
    Plaquette,
)

_DEFAULT_FORDER = [1, 2, 3, -1, -2, -3]
_NON_DEFAULT_FORDER = [-1, -2, 1, 2, 3, -3]

# (dimensions, size) of every lattice shape the tests below check: the
# periodic lattices ymcirc supported before it took its geometry from
# pyclebsch (d=3/2, d=2 and d=3 at sizes 2 and 3), and rectangular ones,
# with and without an axis of length 2.
_PERIODIC_LATTICES = [
    (1.5, 2),
    (1.5, 3),
    (2, 2),
    (2, 3),
    (2, (4, 3)),
    (2, (3, 2)),
    (3, 2),
    (3, 3),
    (3, (3, 3, 4)),
    (3, (3, 3, 2)),
]


class _LinkAddressLattice(LatticeData[LatticeVector | LinkAddress]):
    """A lattice whose data are its own addresses, so that a Plaquette lists the addresses it assembles."""

    def get_vertex(self, lattice_vector: LatticeVector) -> LatticeVector:
        return tuple(lattice_vector)

    def get_link(self, link_address: LinkAddress) -> LinkAddress:
        return self._normalize_link_address(link_address)


def _padded_to_three_axes(vertex: LatticeVector) -> tuple[int, ...]:
    """Coordinates of vertex with zeros appended up to pyclebsch's three axes."""
    return tuple(vertex) + (0,) * (3 - len(vertex))


@pytest.mark.parametrize(
    ("dimensions", "size", "expected_num_sites", "expected_pbcs"),
    [
        (1.5, 4, (4, 2, 1), (True, False, False)),
        (2, 3, (3, 3, 1), (True, True, False)),
        (2, (4, 3), (4, 3, 1), (True, True, False)),
        (3, (3, 3, 4), (3, 3, 4), (True, True, True)),
    ],
)
def test_pyclebsch_lattice_has_translated_shape_and_boundaries(
    dimensions, size, expected_num_sites, expected_pbcs
):
    """A d=3/2 lattice of size L is pyclebsch's L x 2 x 1 lattice, periodic only along x; d=2 is planar."""
    lattice = LatticeDef(dimensions, size, forder=_NON_DEFAULT_FORDER)
    assert lattice._pyclebsch_lattice.num_sites == expected_num_sites
    assert lattice._pyclebsch_lattice.PBCs == expected_pbcs
    assert list(lattice._pyclebsch_lattice.FORDER) == _NON_DEFAULT_FORDER


@pytest.mark.parametrize(
    ("dimensions", "ymcirc_coordinates", "pyclebsch_coordinates"),
    [
        (1.5, (2, 1), (2, 1, 0)),
        (2, (3, 1), (3, 1, 0)),
        (3, (1, 2, 3), (1, 2, 3)),
    ],
)
def test_coordinates_translate_in_both_directions(
    dimensions, ymcirc_coordinates, pyclebsch_coordinates
):
    lattice = LatticeDef(dimensions, 4)
    assert (
        lattice._to_pyclebsch_coordinates(ymcirc_coordinates) == pyclebsch_coordinates
    )
    assert (
        lattice._from_pyclebsch_coordinates(pyclebsch_coordinates) == ymcirc_coordinates
    )


@pytest.mark.parametrize(
    ("dimensions", "direction"), [(1.5, 3), (2, 3), (2, 0), (2, -1), (3, 4)]
)
def test_link_translation_refuses_other_than_positive_directions(dimensions, direction):
    """A link address given to pyclebsch must have one of the lattice's positive directions."""
    lattice = LatticeDef(dimensions, 3)
    origin = (0,) * len(lattice.shape)
    with pytest.raises(ValueError):
        lattice._to_pyclebsch_link_address((origin, direction))


@pytest.mark.parametrize(("dimensions", "size"), _PERIODIC_LATTICES)
def test_sites_and_links_are_pyclebsch_ones(dimensions, size):
    """The vertices and links of a lattice are pyclebsch's, with ymcirc's coordinates, and round-trip."""
    lattice = LatticeDef(dimensions, size)
    pyclebsch_lattice = lattice._pyclebsch_lattice
    n_axes = len(lattice.shape)

    assert {_padded_to_three_axes(v) for v in lattice.vertex_addresses} == set(
        pyclebsch_lattice.sites
    )
    assert {(_padded_to_three_axes(v), d) for v, d in lattice.link_addresses} == set(
        pyclebsch_lattice.links
    )
    for site in pyclebsch_lattice.sites:
        assert len(lattice._from_pyclebsch_coordinates(site)) == n_axes
        assert (
            lattice._to_pyclebsch_coordinates(lattice._from_pyclebsch_coordinates(site))
            == site
        )
    for link in pyclebsch_lattice.links:
        assert (
            lattice._to_pyclebsch_link_address(
                lattice._from_pyclebsch_link_address(link)
            )
            == link
        )


@pytest.mark.parametrize(
    ("dimensions", "first_link", "second_link"),
    [
        (1.5, ((0, 0), 1), ((1, 0), 1)),
        (2, ((0, 1), 1), ((1, 1), 1)),
        (2, ((1, 0), 2), ((1, 1), 2)),
        (3, ((0, 1, 1), 3), ((0, 1, 0), 3)),
    ],
)
def test_links_on_periodic_axis_of_length_2_are_distinct(
    dimensions, first_link, second_link
):
    """
    On a periodic axis of length 2, two links join the same pair of sites.

    They keep distinct addresses, (start site, positive direction), through
    the translation in both directions.
    """
    lattice = LatticeDef(dimensions, 2)
    pyclebsch_links = lattice._pyclebsch_lattice.links
    first = lattice._to_pyclebsch_link_address(first_link)
    second = lattice._to_pyclebsch_link_address(second_link)

    assert first != second
    assert set(pyclebsch_links[first]) == set(pyclebsch_links[second])
    assert lattice._from_pyclebsch_link_address(first) == first_link
    assert lattice._from_pyclebsch_link_address(second) == second_link
    assert (
        first_link in lattice.link_addresses and second_link in lattice.link_addresses
    )


@pytest.mark.parametrize("forder", [_DEFAULT_FORDER, _NON_DEFAULT_FORDER])
@pytest.mark.parametrize(("dimensions", "size"), _PERIODIC_LATTICES)
def test_plaquette_link_addresses_agree_with_pyclebsch(dimensions, size, forder):
    """
    For every plaquette, the link addresses ymcirc assembles equal pyclebsch's, in order.

    ymcirc's Plaquette assembles the active links, then each vertex's control
    links in F-order, which is the layout of a plaquette's bit string.
    pyclebsch lays out its plaquette states the same way, so the matrix
    elements it computes refer to these links in this order.
    """
    lattice = _LinkAddressLattice(dimensions, size, forder=forder)
    pyclebsch_plaquettes = lattice._pyclebsch_lattice.plaquettes
    assert len(pyclebsch_plaquettes) == lattice.n_plaquettes

    for (site, plane), geometry in pyclebsch_plaquettes.items():
        bottom_left = lattice._from_pyclebsch_coordinates(site)
        plaquette = Plaquette(lattice, bottom_left, plane)
        ymcirc_addresses = [
            (_padded_to_three_axes(vertex), direction)
            for vertex, direction in (
                *plaquette.active_links,
                *plaquette.control_links_ordered,
            )
        ]
        pyclebsch_addresses = [
            *geometry.active_links,
            *(link for vertex_links in geometry.control_links for link in vertex_links),
        ]
        assert ymcirc_addresses == pyclebsch_addresses, f"Plaquette {(site, plane)}."
        assert [_padded_to_three_axes(v) for v in plaquette.vertices] == list(
            geometry.site_coordinates
        )


@pytest.mark.parametrize(
    ("dimensions", "size", "expected"),
    [(1.5, 3, 3), (2, 3, 3), (2, (4, 3), (4, 3)), (3, 2, 2), (3, (3, 3, 4), (3, 3, 4))],
)
def test_size_recreates_lattice(dimensions, size, expected):
    """size is what the constructor takes: one length for d=3/2 and equal axes, else the shape."""
    lattice = LatticeDef(dimensions, size)
    assert lattice.size == expected
    assert LatticeDef(lattice.dim, lattice.size) == lattice


@pytest.mark.parametrize(
    ("dimensions", "size", "expected_n_links", "expected_n_plaquettes"),
    [
        (2, (4, 3), 24, 12),
        (2, (3, 2), 12, 6),
        (3, (3, 3, 4), 108, 108),
        (3, (3, 3, 2), 54, 54),
    ],
)
def test_rectangular_lattice_counts(
    dimensions, size, expected_n_links, expected_n_plaquettes
):
    """
    Links and plaquettes of rectangular periodic lattices.

    A periodic lattice has d links and d(d-1)/2 plaquettes per site, also
    along an axis of length 2, where the two links joining a pair of sites
    are distinct.
    """
    lattice = LatticeDef(dimensions, size)
    assert (lattice.n_links, lattice.n_plaquettes) == (
        expected_n_links,
        expected_n_plaquettes,
    )


_PLANES_D3: list[Plane] = [(1, 2), (1, 3), (2, 3)]


@pytest.mark.parametrize(
    ("dimensions", "size", "periodic", "expected_plaquette_addresses"),
    [
        # The top vertices of d=3/2 have no plaquettes.
        (1.5, 3, True, [((0, 0), (1, 2)), ((1, 0), (1, 2)), ((2, 0), (1, 2))]),
        (1.5, 3, False, [((0, 0), (1, 2)), ((1, 0), (1, 2))]),
        (
            2,
            2,
            True,
            [((0, 0), (1, 2)), ((0, 1), (1, 2)), ((1, 0), (1, 2)), ((1, 1), (1, 2))],
        ),
        (2, (3, 2), False, [((0, 0), (1, 2)), ((1, 0), (1, 2))]),
        # Only the origin has plaquettes in more than one plane, so sorting by
        # plane first would give a different order.
        (
            3,
            2,
            False,
            [
                ((0, 0, 0), (1, 2)),
                ((0, 0, 0), (1, 3)),
                ((0, 0, 0), (2, 3)),
                ((0, 0, 1), (1, 2)),
                ((0, 1, 0), (1, 3)),
                ((1, 0, 0), (2, 3)),
            ],
        ),
        (
            3,
            2,
            True,
            [
                ((x, y, z), plane)
                for x in range(2)
                for y in range(2)
                for z in range(2)
                for plane in _PLANES_D3
            ],
        ),
    ],
)
def test_plaquette_addresses(dimensions, size, periodic, expected_plaquette_addresses):
    """
    plaquette_addresses lists the plaquettes that exist, as (bottom-left vertex, plane), in lexicographic order.

    Bottom-left vertices are compared coordinate by coordinate, and planes
    likewise: (1, 2), then (1, 3), then (2, 3). On an open lattice, the
    vertices on the far boundary along an axis have no plaquette extending
    along that axis.
    """
    lattice = LatticeDef(dimensions, size, periodic_boundary_conds=periodic)
    assert lattice.plaquette_addresses == expected_plaquette_addresses


def test_boundary_conditions_per_axis_d_2():
    """
    A d=2 lattice of shape (3, 2), periodic along axis 1 and open along axis 2.

    Every vertex has a link along axis 1 (3 vertices around a periodic axis),
    and only the vertices with y = 0 have one along axis 2 (2 vertices along
    an open axis): 9 links.
    """
    lattice = LatticeDef(2, (3, 2), periodic_boundary_conds=(True, False))
    expected_link_addresses: list[LinkAddress] = [
        ((0, 0), 1),
        ((0, 0), 2),
        ((0, 1), 1),
        ((1, 0), 1),
        ((1, 0), 2),
        ((1, 1), 1),
        ((2, 0), 1),
        ((2, 0), 2),
        ((2, 1), 1),
    ]
    assert lattice.link_addresses == expected_link_addresses
    assert lattice.periodic_boundary_conds == (True, False)
    assert lattice.all_boundary_conds_periodic is False


def test_boundary_conditions_per_axis_d_3():
    """
    A d=3 lattice of shape (3, 3, 2), periodic along axes 1 and 2 and open along axis 3.

    Each of the 18 vertices has a link along axes 1 and 2, and the 9 with
    z = 0 one along axis 3: 45 links. Every vertex has a plaquette in plane
    (1, 2), and only those with z = 0 in planes (1, 3) and (2, 3): 36
    plaquettes.
    """
    lattice = LatticeDef(3, (3, 3, 2), periodic_boundary_conds=(True, True, False))
    assert (lattice.n_links, lattice.n_plaquettes) == (45, 36)


@pytest.mark.parametrize(
    ("dimensions", "periodic_boundary_conds", "expected_periodic_boundary_conds"),
    [
        (2, (True, True), True),
        (2, (False, False), False),
        (3, (True, True, True), True),
        (3, (False, False, False), False),
    ],
)
def test_boundary_conditions_tuple_with_equal_entries_stored_as_bool(
    dimensions, periodic_boundary_conds, expected_periodic_boundary_conds
):
    """
    A tuple of boundary conditions whose entries are all equal is stored as that bool.

    The lattice then equals the one built with the bool. Comparisons of
    lattices, and the equality and hashing of results parsed on them
    (ParsedLatticeResult), then do not depend on which of the two forms a
    caller passed.
    """
    lattice = LatticeDef(dimensions, 2, periodic_boundary_conds=periodic_boundary_conds)
    assert lattice.periodic_boundary_conds is expected_periodic_boundary_conds
    assert lattice == LatticeDef(
        dimensions, 2, periodic_boundary_conds=expected_periodic_boundary_conds
    )


@pytest.mark.parametrize(
    ("dimensions", "size", "periodic_boundary_conds"),
    [
        (2, (3, 2), (True,)),
        (2, (3, 2), (True, False, False)),
        (3, 2, (True, False)),
    ],
)
def test_boundary_conditions_tuple_of_wrong_length_refused(
    dimensions, size, periodic_boundary_conds
):
    """Invalid: a tuple of boundary conditions whose length is not the number of axes. A ValueError is expected."""
    with pytest.raises(ValueError):
        LatticeDef(dimensions, size, periodic_boundary_conds=periodic_boundary_conds)


@pytest.mark.parametrize("periodic_boundary_conds", [(True, False), (True,)])
def test_boundary_conditions_tuple_refused_for_d_3_2(periodic_boundary_conds):
    """
    Invalid: a tuple of boundary conditions for d=3/2, which takes a bool only, since it is always open across the chain.

    A ValueError is expected, whose message points to the d=2 lattice of
    shape (L, 2) for other boundary conditions.
    """
    with pytest.raises(ValueError, match=r"use the d=2 lattice of shape \(L, 2\)"):
        LatticeDef(1.5, 3, periodic_boundary_conds=periodic_boundary_conds)


def test_boundary_conditions_tuple_of_non_bools_refused():
    """Invalid: a tuple of boundary conditions whose entries are not bools. A TypeError is expected."""
    with pytest.raises(TypeError):
        # A deliberate argument of the wrong type.
        LatticeDef(2, (3, 2), periodic_boundary_conds=(1, 0))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("length", [2, 3])
def test_d_3_2_is_d_2_two_vertices_tall(length, periodic):
    """
    A d=3/2 lattice of size L is the d=2 lattice of shape (L, 2), open along axis 2.

    The two have the same vertices, links and plaquettes, and each plaquette
    the same signature.
    """
    d_3_2 = LatticeDef(1.5, length, periodic_boundary_conds=periodic)
    d_2 = LatticeDef(2, (length, 2), periodic_boundary_conds=(periodic, False))
    assert d_3_2.vertex_addresses == d_2.vertex_addresses
    assert d_3_2.link_addresses == d_2.link_addresses
    assert d_3_2.plaquette_addresses == d_2.plaquette_addresses
    for bottom_left_vertex, plane in d_3_2.plaquette_addresses:
        assert d_3_2._plaquette_signature(
            bottom_left_vertex, plane
        ) == d_2._plaquette_signature(bottom_left_vertex, plane)


def test_more_than_three_dimensions_not_supported():
    """pyclebsch, whose geometry LatticeDef takes, supports at most three dimensions."""
    with pytest.raises(NotImplementedError):
        LatticeDef(4, 2)
