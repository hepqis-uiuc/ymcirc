"""
Physical plaquette states and magnetic matrix elements, computed by pyclebsch.

This module is the one place where pyclebsch's formats for plaquette states
and matrix elements enter ymcirc (https://github.com/hepqis-uiuc/pyclebsch). It
translates a ymcirc LatticeDef into a pyclebsch LatticeDef, calls pyclebsch's
functions that compute the physical plaquette states and the matrix elements
once per plaquette signature, and converts their results to ymcirc's formats.

pyclebsch keeps the physical plaquette states and the matrix elements it
computes in memory for the lifetime of the process, so a repeated call with
the same arguments computes neither again (unless the caller has capped
pyclebsch's caches and the entries were evicted). Those caches are capped
through pyclebsch itself, which is installed with ymcirc. By default,
there is no cap. A repeated call only converts the cached results to ymcirc's
formats again, which is cheap: the conversion regroups the entries of each
plaquette state and re-nests the matrix elements, in one pass, with no
Clebsch-Gordan coefficients or site factors to compute. The adapter keeps
no second copy of the converted results, so that each call returns results
the caller owns.

Cache caps can be set via pyclebsch.set_physical_plaquette_states_cache_limit(n)
and pyclebsch.set_plaquette_matrix_elements_cache_limit(n), which keep at most
n entries each, an entry being the results for one plaquette signature at one
truncation, evicting the least recently used.
"""

from __future__ import annotations

from math import isclose
from typing import cast

from pyclebsch.cgc import EPS
from pyclebsch.matrix_elements.lattice_data import (
    LatticeDef as PyclebschLatticeDef,
)
from pyclebsch.matrix_elements.lattice_data import (
    LinkDirection,
    SiteControlLinks,
    SiteMultiplicityIndex,
    physical_plaquette_states_by_signature,
)
from pyclebsch.matrix_elements.lattice_data import (
    PlaquetteState as PyclebschPlaquetteState,
)
from pyclebsch.matrix_elements.plaquette_matrix_elements import (
    plaquette_matrix_elements_by_signature,
)
from pyclebsch.su_n_operators import IrrepWeight

from ymcirc._abstract import LatticeDef, Plane, Signature

__all__ = [
    "N_COLORS",
    "PRES",
    "BoxTerms",
    "MatrixElementValue",
    "PlaquetteState",
    "box_terms",
    "physical_plaquette_states",
    "plaq_state_pyclebsch_to_ymcirc_format",
    "plaq_state_ymcirc_to_pyclebsch_format",
]

# ymcirc simulates SU(3).
N_COLORS = 3
# The number of decimals pyclebsch rounds matrix elements to. pyclebsch has no
# default, so this is ymcirc's choice. ymcirc needs the rounding: control
# fusion groups Givens rotations by exact equality of their angles, so matrix
# elements that are equal in exact arithmetic must be bit-identical floats, and
# rounding makes them so. The value has no derivation; it reproduces the matrix
# elements of ymcirc's former data files exactly. A larger value risks equal
# elements no longer being grouped; a smaller one merges distinct elements, an
# angle error of up to 10**-PRES * dt / g**2.
PRES = 10
# The adapter passes pyclebsch's EPS (imported above), the magnitude below
# which pyclebsch counts a Clebsch-Gordan coefficient as rounding noise and
# drops it, as the matching threshold for matrix elements: pyclebsch drops a
# sum of products of Clebsch-Gordan coefficients, a site factor or a matrix
# element smaller than it in magnitude.

# The number of corners of a plaquette.
_N_PLAQUETTE_CORNERS = 4
# On a periodic axis of length 2, two of a plaquette's control links are the
# same link; from length 3 up, they are all distinct. When ymcirc builds
# circuits for a periodic lattice only 2 sites long in some direction, the
# adapter asks pyclebsch for the plaquette states and matrix elements of the
# same lattice made this many sites long in that direction.
# LatticeCircuitManager then throws away the ones that cannot occur on the
# 2-site lattice, where links that are distinct on the longer lattice become
# the same link, as ymcirc's former data files were used.
_SHORTEST_PERIODIC_LENGTH_WITHOUT_REPEATED_CONTROLS = 3

# Matrix element value: a nested dict keyed by plane, then signature.
type MatrixElementValue = dict[Plane, dict[Signature, float]]
# Tuple of 4 vertex multiplicites, tuple of 4 "active links", tuple of 4
# per-vertex control tuples.
type PlaquetteState = (
    tuple[
        tuple[
            SiteMultiplicityIndex,
            SiteMultiplicityIndex,
            SiteMultiplicityIndex,
            SiteMultiplicityIndex,
        ],
        tuple[IrrepWeight, IrrepWeight, IrrepWeight, IrrepWeight],
        tuple[SiteControlLinks, SiteControlLinks, SiteControlLinks, SiteControlLinks],
    ]
    | tuple[
        tuple[None, None, None, None],
        tuple[IrrepWeight, IrrepWeight, IrrepWeight, IrrepWeight],
        tuple[SiteControlLinks, SiteControlLinks, SiteControlLinks, SiteControlLinks],
    ]
)
# Box-term matrix elements keyed by (final, initial) plaquette states.
type BoxTerms = dict[tuple[PlaquetteState, PlaquetteState], MatrixElementValue]


def plaq_state_pyclebsch_to_ymcirc_format(
    plaq_state: PyclebschPlaquetteState,
) -> PlaquetteState:
    """
    Convert a plaquette state from pyclebsch's format to ymcirc's.

    pyclebsch writes a plaquette state as a flat tuple: the four active links,
    the four corners' tuples of control links, and the four corners' singlet
    multiplicity indices. ymcirc groups the same entries, in the same order
    within each group, as (multiplicity indices, active links, control links).

    Raises a ValueError if plaq_state does not have four multiplicity indices.
    """
    a_links = tuple(plaq_state[:_N_PLAQUETTE_CORNERS])
    c_links = tuple(plaq_state[_N_PLAQUETTE_CORNERS : 2 * _N_PLAQUETTE_CORNERS])
    vertices = tuple(plaq_state[2 * _N_PLAQUETTE_CORNERS :])
    if len(vertices) != _N_PLAQUETTE_CORNERS:
        raise ValueError(
            f"Expected {_N_PLAQUETTE_CORNERS} sites. Site count: '{len(vertices)}'."
        )

    return (vertices, a_links, c_links)


def plaq_state_ymcirc_to_pyclebsch_format(
    plaq_state: PlaquetteState,
) -> PyclebschPlaquetteState:
    """Convert a plaquette state from ymcirc's format to pyclebsch's (the inverse of plaq_state_pyclebsch_to_ymcirc_format)."""
    vertices, a_links, c_links = plaq_state
    # ymcirc's type allows multiplicity indices of None, which no state that
    # pyclebsch computes has.
    return cast(PyclebschPlaquetteState, (*a_links, *c_links, *vertices))


def _pyclebsch_lattice(lattice_def: LatticeDef) -> PyclebschLatticeDef:
    """
    Return the pyclebsch lattice on which lattice_def's physical plaquette states and matrix elements are computed.

    The lattice has lattice_def's shape, boundary conditions and F-order, in
    pyclebsch's form of three axes, except that a periodic axis of length 2 is
    lengthened to 3 (see _SHORTEST_PERIODIC_LENGTH_WITHOUT_REPEATED_CONTROLS). A
    d=3/2 lattice of size L is L sites long and periodic in x, and two sites
    tall and open in y.

    Raises a NotImplementedError for lattices that are not periodic, which
    ymcirc does not support yet. (LatticeDef itself refuses lattices with
    different lengths along different directions.)
    """
    if not lattice_def.all_boundary_conds_periodic:
        raise NotImplementedError(
            "Lattices with nonperiodic or mixed boundary conditions not yet supported."
        )

    size = max(
        lattice_def.shape[0], _SHORTEST_PERIODIC_LENGTH_WITHOUT_REPEATED_CONTROLS
    )
    if isclose(lattice_def.dim, 1.5):
        num_sites = (size, lattice_def.shape[1], 1)
        pbcs = (True, False, False)
    elif lattice_def.dim == 2:
        num_sites = (size, size, 1)
        pbcs = (True, True, False)
    elif lattice_def.dim == 3:
        num_sites = (size, size, size)
        pbcs = (True, True, True)
    else:
        raise NotImplementedError(f"Dim {lattice_def.dim} lattice not yet supported.")

    # LatticeDef has checked that its F-order is a permutation of pyclebsch's
    # link directions.
    forder = cast(list[LinkDirection], lattice_def.forder)
    return PyclebschLatticeDef(num_sites, pbcs, forder)


def physical_plaquette_states(
    lattice_def: LatticeDef, truncation_mode: str, cutoff: int
) -> list[PlaquetteState]:
    """
    Return the physical (gauge-invariant) plaquette states of a lattice, in ymcirc's format.

    Arguments:
        lattice_def: the lattice. Its physical plaquette states are
            computed on the lattice _pyclebsch_lattice describes.
        truncation_mode, cutoff: the irrep truncation, as pyclebsch's
            irreps_and_singlets takes it, for example ("T", 1) or ("B", 3).
            pyclebsch validates them.

    The states of every plaquette signature of the lattice are returned
    together, each state once. The caller owns the returned list.
    """
    states_by_signature = physical_plaquette_states_by_signature(
        _pyclebsch_lattice(lattice_def), N_COLORS, truncation_mode, cutoff
    )
    return list(
        {
            plaq_state_pyclebsch_to_ymcirc_format(state)
            for signature_states in states_by_signature.values()
            for state in signature_states
        }
    )


def box_terms(lattice_def: LatticeDef, truncation_mode: str, cutoff: int) -> BoxTerms:
    """
    Return the matrix elements of the plaquette operator (the box term) of a lattice.

    Arguments are as for physical_plaquette_states.

    Returns {(final state, initial state): {plane: {signature: value}}}, with
    states in ymcirc's format, where signature is the F-ordered half-link
    directions at each corner of a plaquette (ymcirc._abstract.Signature).
    Values are rounded to PRES decimals, and products of site factors smaller
    than EPS are dropped. The caller owns the returned dict.
    """
    elements_by_signature = plaquette_matrix_elements_by_signature(
        _pyclebsch_lattice(lattice_def), N_COLORS, truncation_mode, cutoff, EPS, PRES
    )
    result: BoxTerms = {}
    for (plane, site_half_links), elements in elements_by_signature.items():
        for (final, initial), value in elements.items():
            key = (
                plaq_state_pyclebsch_to_ymcirc_format(final),
                plaq_state_pyclebsch_to_ymcirc_format(initial),
            )
            # pyclebsch's site half-links are ymcirc's signature.
            signature = cast(Signature, site_half_links)
            result.setdefault(key, {}).setdefault(plane, {})[signature] = value
    return result
