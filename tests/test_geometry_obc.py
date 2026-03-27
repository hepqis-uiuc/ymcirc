"""Tests for OBC and mixed BC geometry layer (Phase 2 of OBC implementation).

Covers:
- Tuple-valued BCs in LatticeDef
- add_unit_vector_to_vertex_vector() on OBC and mixed BC lattices
- _normalize_link_address() on OBC and mixed BC lattices
- get_traversal_order() on OBC and mixed BC lattices
- n_plaquettes for various BC configurations
- n_control_links_per_plaquette for non-periodic lattices
- get_vertex() on OBC lattices
"""
from __future__ import annotations
from math import ceil
from itertools import product
import pytest
from ymcirc._abstract.lattice_data import LatticeDef, Plaquette
from ymcirc.lattice_registers import LatticeRegisters


# ---------------------------------------------------------------------------
# Tuple-valued boundary conditions in LatticeDef
# ---------------------------------------------------------------------------

class TestTupleBoundaryConditions:
    """Tests for accepting tuple-valued periodic_boundary_conds."""

    def test_full_obc_d2(self):
        ld = LatticeDef(dimensions=2, size=3, periodic_boundary_conds=(False, False))
        assert ld.periodic_boundary_conds == (False, False)
        assert ld.all_boundary_conds_periodic is False

    def test_full_pbc_tuple_d2(self):
        ld = LatticeDef(dimensions=2, size=3, periodic_boundary_conds=(True, True))
        assert ld.all_boundary_conds_periodic is True

    def test_mixed_bc_d2(self):
        ld = LatticeDef(dimensions=2, size=3, periodic_boundary_conds=(True, False))
        assert ld.periodic_boundary_conds == (True, False)
        assert ld.all_boundary_conds_periodic is False

    def test_full_obc_d3(self):
        ld = LatticeDef(dimensions=3, size=3, periodic_boundary_conds=(False, False, False))
        assert ld.periodic_boundary_conds == (False, False, False)
        assert ld.all_boundary_conds_periodic is False

    def test_mixed_bc_d3(self):
        ld = LatticeDef(dimensions=3, size=3, periodic_boundary_conds=(True, False, True))
        assert ld.all_boundary_conds_periodic is False

    def test_d_3_2_obc(self):
        """d=3/2: vertical direction is always non-periodic. (True, False) is valid."""
        ld = LatticeDef(dimensions="3/2", size=3, periodic_boundary_conds=(True, False))
        assert ld.all_boundary_conds_periodic is False

    def test_d_3_2_full_obc(self):
        ld = LatticeDef(dimensions="3/2", size=3, periodic_boundary_conds=(False, False))
        assert ld.all_boundary_conds_periodic is False

    def test_d_3_2_vertical_periodic_rejected(self):
        """d=3/2: vertical direction cannot be periodic."""
        with pytest.raises(ValueError, match="Vertical direction cannot be periodic"):
            LatticeDef(dimensions="3/2", size=3, periodic_boundary_conds=(True, True))

    def test_wrong_tuple_length_raises(self):
        with pytest.raises(ValueError, match="tuple length"):
            LatticeDef(dimensions=2, size=3, periodic_boundary_conds=(True, False, True))

    def test_non_bool_tuple_elements_raise(self):
        with pytest.raises(TypeError, match="All elements"):
            LatticeDef(dimensions=2, size=3, periodic_boundary_conds=(True, 0))  # type: ignore[arg-type]

    def test_bool_pbc_still_works(self):
        """Scalar bool PBC should still work as before."""
        ld_true = LatticeDef(dimensions=2, size=3, periodic_boundary_conds=True)
        assert ld_true.all_boundary_conds_periodic is True
        ld_false = LatticeDef(dimensions=2, size=3, periodic_boundary_conds=False)
        assert ld_false.all_boundary_conds_periodic is False


# ---------------------------------------------------------------------------
# add_unit_vector_to_vertex_vector on OBC and mixed BC
# ---------------------------------------------------------------------------

class TestAddUnitVectorOBC:
    """Tests for add_unit_vector_to_vertex_vector() on non-periodic lattices."""

    def test_interior_step_obc_d2(self):
        ld = LatticeDef(dimensions=2, size=4, periodic_boundary_conds=False)
        assert ld.add_unit_vector_to_vertex_vector((1, 1), 1) == (2, 1)
        assert ld.add_unit_vector_to_vertex_vector((1, 1), 2) == (1, 2)
        assert ld.add_unit_vector_to_vertex_vector((1, 1), -1) == (0, 1)
        assert ld.add_unit_vector_to_vertex_vector((1, 1), -2) == (1, 0)

    def test_boundary_step_raises_obc_d2(self):
        ld = LatticeDef(dimensions=2, size=4, periodic_boundary_conds=False)
        # Stepping off the right boundary.
        with pytest.raises(KeyError):
            ld.add_unit_vector_to_vertex_vector((3, 0), 1)
        # Stepping off the top boundary.
        with pytest.raises(KeyError):
            ld.add_unit_vector_to_vertex_vector((0, 3), 2)
        # Stepping off the left boundary.
        with pytest.raises(KeyError):
            ld.add_unit_vector_to_vertex_vector((0, 0), -1)
        # Stepping off the bottom boundary.
        with pytest.raises(KeyError):
            ld.add_unit_vector_to_vertex_vector((0, 0), -2)

    def test_mixed_bc_wraps_only_periodic_dir(self):
        """Mixed BC: x-periodic, y-open."""
        ld = LatticeDef(dimensions=2, size=3, periodic_boundary_conds=(True, False))
        # x wraps.
        assert ld.add_unit_vector_to_vertex_vector((2, 1), 1) == (0, 1)
        # y does NOT wrap.
        with pytest.raises(KeyError):
            ld.add_unit_vector_to_vertex_vector((1, 2), 2)

    def test_obc_d3(self):
        ld = LatticeDef(dimensions=3, size=3, periodic_boundary_conds=False)
        assert ld.add_unit_vector_to_vertex_vector((1, 1, 1), 3) == (1, 1, 2)
        with pytest.raises(KeyError):
            ld.add_unit_vector_to_vertex_vector((1, 1, 2), 3)

    def test_d_3_2_obc_horizontal(self):
        """d=3/2 with open horizontal BC."""
        ld = LatticeDef(dimensions="3/2", size=4, periodic_boundary_conds=(False, False))
        assert ld.add_unit_vector_to_vertex_vector((2, 0), 1) == (3, 0)
        with pytest.raises(KeyError):
            ld.add_unit_vector_to_vertex_vector((3, 0), 1)


# ---------------------------------------------------------------------------
# _normalize_link_address on OBC and mixed BC
# ---------------------------------------------------------------------------

class TestNormalizeLinkAddressOBC:
    """Tests for _normalize_link_address() on non-periodic lattices."""

    def test_positive_dir_interior(self):
        ld = LatticeDef(dimensions=2, size=4, periodic_boundary_conds=False)
        assert ld._normalize_link_address(((1, 1), 1)) == ((1, 1), 1)

    def test_negative_dir_interior(self):
        ld = LatticeDef(dimensions=2, size=4, periodic_boundary_conds=False)
        # Link ((2, 1), -1) should normalize to ((1, 1), 1).
        assert ld._normalize_link_address(((2, 1), -1)) == ((1, 1), 1)

    def test_negative_dir_at_boundary_raises(self):
        """Stepping back from the boundary goes out of bounds."""
        ld = LatticeDef(dimensions=2, size=4, periodic_boundary_conds=False)
        with pytest.raises(KeyError):
            ld._normalize_link_address(((0, 0), -1))
        with pytest.raises(KeyError):
            ld._normalize_link_address(((0, 0), -2))

    def test_mixed_bc_wraps_periodic_dir_only(self):
        """x-periodic, y-open: wrapping in x should work, in y should fail."""
        ld = LatticeDef(dimensions=2, size=3, periodic_boundary_conds=(True, False))
        # Negative x at boundary wraps.
        assert ld._normalize_link_address(((0, 1), -1)) == ((2, 1), 1)
        # Negative y at boundary raises.
        with pytest.raises(KeyError):
            ld._normalize_link_address(((1, 0), -2))


# ---------------------------------------------------------------------------
# n_plaquettes for various BC configurations
# ---------------------------------------------------------------------------

class TestNPlaquettesOBC:
    """Tests for n_plaquettes on non-periodic and mixed BC lattices."""

    @pytest.mark.parametrize("size", [2, 3, 4, 5])
    def test_d2_full_obc(self, size):
        """d=2 OBC: (size-1)^2 plaquettes."""
        ld = LatticeDef(dimensions=2, size=size, periodic_boundary_conds=False)
        assert ld.n_plaquettes == (size - 1) ** 2

    @pytest.mark.parametrize("size", [2, 3, 4])
    def test_d2_full_pbc(self, size):
        """d=2 PBC: size^2 plaquettes."""
        ld = LatticeDef(dimensions=2, size=size, periodic_boundary_conds=True)
        assert ld.n_plaquettes == size ** 2

    def test_d2_mixed_bc(self):
        """d=2 mixed (x-periodic, y-open) size=3: 3*2=6 plaquettes."""
        ld = LatticeDef(dimensions=2, size=3, periodic_boundary_conds=(True, False))
        assert ld.n_plaquettes == 6

    def test_d2_mixed_bc_reversed(self):
        """d=2 mixed (x-open, y-periodic) size=3: 2*3=6 plaquettes."""
        ld = LatticeDef(dimensions=2, size=3, periodic_boundary_conds=(False, True))
        assert ld.n_plaquettes == 6

    @pytest.mark.parametrize("size", [2, 3, 4])
    def test_d3_full_obc(self, size):
        """d=3 OBC: 3*(size-1)^3 is wrong — it's C(3,2) * prod over all dirs.
        For each plane (i,j), count = n_i * n_j * n_other where n_k = size-1 for OBC.
        Three planes, each (size-1)^2 * (size) ... no.
        Actually: for plane (i,j), n_i=size-1, n_j=size-1, n_other = product of shape[k] for k not in {i,j}.
        With d=3 OBC size N: each plane has (N-1)*(N-1)*N... No, the other dirs don't reduce.
        Wait — n_other uses full shape, not reduced. Let me recalculate.
        plane (1,2): n_1=N-1, n_2=N-1, n_other = N (dir 3 full)
        plane (1,3): n_1=N-1, n_3=N-1, n_other = N (dir 2 full)
        plane (2,3): n_2=N-1, n_3=N-1, n_other = N (dir 1 full)
        Total = 3 * (N-1)^2 * N
        """
        ld = LatticeDef(dimensions=3, size=size, periodic_boundary_conds=False)
        expected = 3 * (size - 1) ** 2 * size
        assert ld.n_plaquettes == expected

    def test_d_3_2_full_obc(self):
        """d=3/2 with open horizontal BC, size=4.
        Only one plane (1,2). n_1 = size-1 = 3, n_2 = shape[1]-1 = 2-1 = 1.
        Total = 3*1 = 3.
        """
        ld = LatticeDef(dimensions="3/2", size=4, periodic_boundary_conds=(False, False))
        assert ld.n_plaquettes == 3

    def test_d_3_2_pbc_horizontal(self):
        """d=3/2 with PBC horizontal: n_1=size, n_2=1. Total = size."""
        ld = LatticeDef(dimensions="3/2", size=4, periodic_boundary_conds=(True, False))
        assert ld.n_plaquettes == 4


# ---------------------------------------------------------------------------
# n_control_links_per_plaquette raises for non-periodic
# ---------------------------------------------------------------------------

class TestControlLinksPerPlaquetteOBC:
    def test_raises_for_obc(self):
        ld = LatticeDef(dimensions=2, size=3, periodic_boundary_conds=False)
        with pytest.raises(ValueError, match="not uniform"):
            _ = ld.n_control_links_per_plaquette

    def test_ok_for_pbc(self):
        ld = LatticeDef(dimensions=2, size=3, periodic_boundary_conds=True)
        assert ld.n_control_links_per_plaquette == 8  # 4 vertices * 2 controls each


# ---------------------------------------------------------------------------
# get_traversal_order on OBC and mixed BC
# ---------------------------------------------------------------------------

class TestTraversalOrderOBC:
    def test_d2_obc_link_count(self):
        """On a 3x3 OBC d=2 lattice, boundary vertices should have fewer links."""
        ld = LatticeDef(dimensions=2, size=3, periodic_boundary_conds=False)
        traversal = ld.get_traversal_order()

        # Total vertices: 9.
        assert len(traversal) == 9

        # Count total links in traversal.
        total_links = sum(len(links) for _, links in traversal)
        # d=2 OBC size 3: links in +x = 2*3=6, links in +y = 3*2=6. Total = 12.
        assert total_links == 12

    def test_d2_obc_boundary_vertex_has_fewer_links(self):
        """Corner vertex (2,2) on a 3x3 OBC should have 0 positive links."""
        ld = LatticeDef(dimensions=2, size=3, periodic_boundary_conds=False)
        traversal = dict(ld.get_traversal_order())
        # (2,2) is the top-right corner; both +x and +y go out of bounds.
        assert len(traversal[(2, 2)]) == 0
        # (0,0) is interior enough to have both links.
        assert len(traversal[(0, 0)]) == 2

    def test_d2_pbc_all_vertices_have_2_links(self):
        """On a PBC lattice, every vertex has 2 positive links."""
        ld = LatticeDef(dimensions=2, size=3, periodic_boundary_conds=True)
        traversal = ld.get_traversal_order()
        for _, links in traversal:
            assert len(links) == 2

    def test_d2_mixed_bc(self):
        """x-periodic, y-open, size=3: vertices at y=2 lose +y link, but all have +x."""
        ld = LatticeDef(dimensions=2, size=3, periodic_boundary_conds=(True, False))
        traversal = dict(ld.get_traversal_order())
        # Interior/bottom vertices have both +x and +y.
        assert len(traversal[(0, 0)]) == 2
        assert len(traversal[(1, 1)]) == 2
        # Top-row vertices lose +y.
        assert len(traversal[(0, 2)]) == 1
        assert len(traversal[(2, 2)]) == 1

    def test_d2_obc_plaquette_count_matches_traversal(self):
        """Verify that building plaquettes from traversal gives the right count."""
        size = 4
        ld = LatticeDef(dimensions=2, size=size, periodic_boundary_conds=False)
        lattice = LatticeRegisters(dimensions=2, size=size, periodic_boundary_conds=False)
        plaquettes = []
        for vertex in lattice.vertex_addresses:
            # A plaquette at vertex (i,j) in the (1,2) plane exists if i < size-1 and j < size-1.
            if vertex[0] < size - 1 and vertex[1] < size - 1:
                plaquettes.append(lattice.get_plaquettes(vertex, 1, 2))
        assert len(plaquettes) == ld.n_plaquettes

    def test_d3_obc_boundary_vertex(self):
        """d=3 OBC size 3: vertex (2,2,2) should have 0 positive links."""
        ld = LatticeDef(dimensions=3, size=3, periodic_boundary_conds=False)
        traversal = dict(ld.get_traversal_order())
        assert len(traversal[(2, 2, 2)]) == 0
        assert len(traversal[(0, 0, 0)]) == 3


# ---------------------------------------------------------------------------
# get_vertex on OBC lattices (LatticeRegisters)
# ---------------------------------------------------------------------------

class TestGetVertexOBC:
    def test_interior_vertex(self):
        lattice = LatticeRegisters(dimensions=2, size=4, periodic_boundary_conds=False)
        reg = lattice.get_vertex((1, 2))
        assert reg.name == "v:(1, 2)"

    def test_out_of_bounds_raises(self):
        lattice = LatticeRegisters(dimensions=2, size=4, periodic_boundary_conds=False)
        with pytest.raises(KeyError):
            lattice.get_vertex((4, 0))
        with pytest.raises(KeyError):
            lattice.get_vertex((-1, 0))
        with pytest.raises(KeyError):
            lattice.get_vertex((0, 4))

    def test_pbc_wraps_obc_doesnt(self):
        """Same coordinate: PBC wraps it, OBC rejects it."""
        pbc_lattice = LatticeRegisters(dimensions=2, size=3, periodic_boundary_conds=True)
        obc_lattice = LatticeRegisters(dimensions=2, size=3, periodic_boundary_conds=False)
        # (3,0) should wrap to (0,0) on PBC.
        assert pbc_lattice.get_vertex((3, 0)).name == "v:(0, 0)"
        # (3,0) should fail on OBC.
        with pytest.raises(KeyError):
            obc_lattice.get_vertex((3, 0))

    def test_mixed_bc_wraps_only_periodic_dir(self):
        """x-periodic, y-open."""
        lattice = LatticeRegisters(dimensions=2, size=3, periodic_boundary_conds=(True, False))
        # x wraps.
        assert lattice.get_vertex((3, 0)).name == "v:(0, 0)"
        # y doesn't wrap.
        with pytest.raises(KeyError):
            lattice.get_vertex((0, 3))


# ---------------------------------------------------------------------------
# Link counts on OBC lattices
# ---------------------------------------------------------------------------

class TestLinkCountsOBC:
    def test_d2_obc_link_count(self):
        """d=2 OBC size N: links = N*(N-1) in each direction, 2 directions = 2*N*(N-1)."""
        for size in [2, 3, 4]:
            ld = LatticeDef(dimensions=2, size=size, periodic_boundary_conds=False)
            expected = 2 * size * (size - 1)
            assert ld.n_links == expected, f"size={size}: expected {expected}, got {ld.n_links}"

    def test_d2_pbc_link_count(self):
        """d=2 PBC size N: links = N^2 per direction, 2 directions = 2*N^2."""
        for size in [2, 3, 4]:
            ld = LatticeDef(dimensions=2, size=size, periodic_boundary_conds=True)
            expected = 2 * size ** 2
            assert ld.n_links == expected


# ---------------------------------------------------------------------------
# Plaquette construction on OBC lattices
# ---------------------------------------------------------------------------

class TestPlaquetteOBC:
    def test_corner_plaquette_has_fewer_controls(self):
        """A corner plaquette on a large OBC lattice has fewer control links than interior."""
        size = 5
        lattice = LatticeRegisters(dimensions=2, size=size, periodic_boundary_conds=False)
        corner_plaq = lattice.get_plaquettes((0, 0), 1, 2)
        interior_plaq = lattice.get_plaquettes((2, 2), 1, 2)
        corner_n_controls = len(corner_plaq.control_links_ordered)
        interior_n_controls = len(interior_plaq.control_links_ordered)
        assert corner_n_controls < interior_n_controls

    def test_interior_plaquette_same_as_pbc(self):
        """An interior plaquette on a large OBC lattice should have the same
        number of control links as one on a PBC lattice."""
        size = 5
        obc_lattice = LatticeRegisters(dimensions=2, size=size, periodic_boundary_conds=False)
        pbc_lattice = LatticeRegisters(dimensions=2, size=size, periodic_boundary_conds=True)
        obc_plaq = obc_lattice.get_plaquettes((2, 2), 1, 2)
        pbc_plaq = pbc_lattice.get_plaquettes((2, 2), 1, 2)
        assert len(obc_plaq.control_links_ordered) == len(pbc_plaq.control_links_ordered)

    def test_boundary_plaquette_construction_doesnt_fail(self):
        """Plaquette at the boundary of an OBC lattice should construct fine."""
        lattice = LatticeRegisters(dimensions=2, size=3, periodic_boundary_conds=False)
        # Bottom-left corner
        plaq = lattice.get_plaquettes((0, 0), 1, 2)
        assert plaq.bottom_left_vertex == (0, 0)
        assert plaq.plane == (1, 2)

    def test_plaquette_at_boundary_raises_if_too_far(self):
        """Cannot construct plaquette at (2,2) on a 3x3 OBC because vertex (3,2) doesn't exist."""
        lattice = LatticeRegisters(dimensions=2, size=3, periodic_boundary_conds=False)
        with pytest.raises(KeyError):
            lattice.get_plaquettes((2, 2), 1, 2)

    def test_obc_d2_all_plaquettes_no_repeats(self):
        """All plaquettes on an OBC lattice should be unique."""
        size = 4
        lattice = LatticeRegisters(dimensions=2, size=size, periodic_boundary_conds=False)
        plaquettes = []
        for i in range(size - 1):
            for j in range(size - 1):
                plaquettes.append(lattice.get_plaquettes((i, j), 1, 2))
        # No two plaquettes should share the same bottom_left_vertex.
        bottom_lefts = [p.bottom_left_vertex for p in plaquettes]
        assert len(set(bottom_lefts)) == len(bottom_lefts)

    def test_d2_obc_signature_varies_by_position(self):
        """On a large OBC lattice, signatures differ between corner, edge, and interior."""
        size = 5
        lattice = LatticeRegisters(dimensions=2, size=size, periodic_boundary_conds=False)
        corner_sig = lattice.get_plaquettes((0, 0), 1, 2).signature
        edge_sig = lattice.get_plaquettes((1, 0), 1, 2).signature
        interior_sig = lattice.get_plaquettes((2, 2), 1, 2).signature
        # All three should be different tuples.
        assert corner_sig != interior_sig
        assert corner_sig != edge_sig

    def test_d2_obc_interior_signature_matches_pbc(self):
        """Interior signature on OBC should match PBC signature."""
        size = 5
        obc = LatticeRegisters(dimensions=2, size=size, periodic_boundary_conds=False)
        pbc = LatticeRegisters(dimensions=2, size=size, periodic_boundary_conds=True)
        assert obc.get_plaquettes((2, 2), 1, 2).signature == pbc.get_plaquettes((2, 2), 1, 2).signature


# ---------------------------------------------------------------------------
# d=2 OBC signature enumeration
# ---------------------------------------------------------------------------

class TestSignatureEnumeration:
    """Verify that a 4x4 OBC d=2 lattice has exactly 9 distinct signatures."""

    def test_d2_obc_9_signatures(self):
        size = 4
        lattice = LatticeRegisters(dimensions=2, size=size, periodic_boundary_conds=False)
        sigs = set()
        for i in range(size - 1):
            for j in range(size - 1):
                plaq = lattice.get_plaquettes((i, j), 1, 2)
                sigs.add(plaq.signature)
        assert len(sigs) == 9

    def test_d_3_2_obc_3_signatures(self):
        """d=3/2 OBC lattice: should have 3 distinct signatures (left edge, interior, right edge)."""
        size = 4
        lattice = LatticeRegisters(dimensions="3/2", size=size, periodic_boundary_conds=(False, False))
        sigs = set()
        for i in range(size - 1):
            plaq = lattice.get_plaquettes((i, 0), 1, 2)
            sigs.add(plaq.signature)
        assert len(sigs) == 3
