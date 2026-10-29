"""Tests of LatticeDef's geometry, which it takes from pyclebsch's LatticeDef."""

from __future__ import annotations

import pytest

from ymcirc._abstract.lattice_data import (
    LatticeData,
    LatticeDef,
    LatticeVector,
    LinkAddress,
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
def test_plaquette_addresses_agree_with_pyclebsch(dimensions, size, forder):
    """
    For every plaquette, the link addresses ymcirc assembles equal pyclebsch's, in order.

    ymcirc's Plaquette assembles the active links, then each corner's control
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
            *(link for corner_links in geometry.control_links for link in corner_links),
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


def test_more_than_three_dimensions_not_supported():
    """pyclebsch, whose geometry LatticeDef takes, supports at most three dimensions."""
    with pytest.raises(NotImplementedError):
        LatticeDef(4, 2)
