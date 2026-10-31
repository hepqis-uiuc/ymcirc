"""
Physical plaquette states and magnetic matrix elements, computed by pyclebsch.

This module is the one place where pyclebsch's formats for plaquette states
and matrix elements enter ymcirc (https://github.com/hepqis-uiuc/pyclebsch). It
calls pyclebsch's functions that compute the physical plaquette states and the
matrix elements once per plaquette signature, on the pyclebsch LatticeDef that
a ymcirc LatticeDef wraps, and converts their results to ymcirc's formats.

pyclebsch keeps the physical plaquette states and the matrix elements it
computes in memory for the lifetime of the process, so a repeated call with
the same arguments computes neither again (unless the caller has capped
pyclebsch's caches and the entries were evicted). Those caches are capped
through pyclebsch itself, which is installed with ymcirc. By default,
there is no cap. A repeated call only converts the cached results to ymcirc's
formats again, which is cheap: the conversion regroups the entries of each
plaquette state, in one pass, with no Clebsch-Gordan coefficients or site
factors to compute. The adapter keeps no second copy of the converted results,
so that each call returns results the caller owns.

Cache caps can be set via pyclebsch.set_physical_plaquette_states_cache_limit(n)
and pyclebsch.set_plaquette_matrix_elements_cache_limit(n), which keep at most
n entries each, an entry being the results for one plaquette signature at one
truncation, evicting the least recently used.
"""

from __future__ import annotations

from typing import cast

from pyclebsch.cgc import EPS
from pyclebsch.matrix_elements.lattice_data import (
    PlaquetteState as PyclebschPlaquetteState,
)
from pyclebsch.matrix_elements.lattice_data import (
    physical_plaquette_states_by_signature,
)
from pyclebsch.matrix_elements.plaquette_matrix_elements import (
    plaquette_matrix_elements_by_signature,
)

from ymcirc._abstract import (
    LatticeDef,
    Plane,
    PlaquetteMatrixElements,
    PlaquetteState,
    Signature,
)

__all__ = [
    "N_COLORS",
    "PRES",
    "box_plus_box_dagger_matrix_elements",
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

# The number of vertices of a plaquette.
_N_PLAQUETTE_VERTICES = 4


def plaq_state_pyclebsch_to_ymcirc_format(
    plaq_state: PyclebschPlaquetteState,
) -> PlaquetteState:
    """
    Convert a plaquette state from pyclebsch's format to ymcirc's.

    pyclebsch writes a plaquette state as a flat tuple: the four active links,
    the four vertices' tuples of control links, and the four vertices' singlet
    multiplicity indices. ymcirc groups the same entries, in the same order
    within each group, as (multiplicity indices, active links, control links).

    Raises a ValueError if plaq_state does not have four multiplicity indices.
    """
    a_links = tuple(plaq_state[:_N_PLAQUETTE_VERTICES])
    c_links = tuple(plaq_state[_N_PLAQUETTE_VERTICES : 2 * _N_PLAQUETTE_VERTICES])
    vertices = tuple(plaq_state[2 * _N_PLAQUETTE_VERTICES :])
    if len(vertices) != _N_PLAQUETTE_VERTICES:
        raise ValueError(
            f"Expected {_N_PLAQUETTE_VERTICES} sites. Site count: '{len(vertices)}'."
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


def physical_plaquette_states(
    lattice_def: LatticeDef, truncation_mode: str, cutoff: int
) -> dict[tuple[Plane, Signature], set[PlaquetteState]]:
    """
    Return the physical (gauge-invariant) plaquette states of a lattice, in ymcirc's format.

    Arguments:
        lattice_def: a ymcirc LatticeDef. pyclebsch computes the physical
            plaquette states on the pyclebsch LatticeDef that lattice_def
            wraps (lattice_def._pyclebsch_lattice), which has the same sites,
            links and plaquettes.
        truncation_mode, cutoff: the irrep truncation, as pyclebsch's
            irreps_and_singlets takes it, for example ("T", 1) or ("B", 3).
            pyclebsch validates them.

    Returns {(plane, signature): states}, as pyclebsch groups the states,
    where signature is the F-ordered half-link directions at each vertex of a
    plaquette (ymcirc._abstract.Signature). A state of a plaquette lists every
    control link at every vertex, including the controls that repeat another
    (on a periodic axis of length 2), which carry the same irrep as the
    control they repeat. One state can appear under several keys. The caller
    owns the returned dict.
    """
    states_by_signature = physical_plaquette_states_by_signature(
        lattice_def._pyclebsch_lattice, N_COLORS, truncation_mode, cutoff
    )
    result: dict[tuple[Plane, Signature], set[PlaquetteState]] = {}
    for (plane, site_half_links), states in states_by_signature.items():
        # The cast narrows pyclebsch's type (which also allows an empty tuple of
        # half-links) to ymcirc's Signature. pyclebsch returns an empty tuple only
        # for a plaquette the lattice lacks. Since site_half_links is a key of
        # physical_plaquette_states_by_signature's result, whose keys are those
        # of plaquettes_by_signature, these half-links belong to one of the
        # lattice's plaquettes.
        signature = cast(Signature, site_half_links)
        result[(plane, signature)] = {
            plaq_state_pyclebsch_to_ymcirc_format(state) for state in states
        }
    return result


def box_plus_box_dagger_matrix_elements(
    lattice_def: LatticeDef, truncation_mode: str, cutoff: int
) -> PlaquetteMatrixElements:
    """
    Return the matrix elements of box + box^dagger on a lattice, box being the plaquette operator.

    box + box^dagger is the magnetic Hamiltonian of one plaquette, up to a
    prefactor (ymcirc.circuit's magnetic Trotter step applies it through its
    time-step and coupling parameters).

    Arguments:
        lattice_def: a ymcirc LatticeDef. pyclebsch computes the matrix
            elements on the pyclebsch LatticeDef that lattice_def wraps
            (lattice_def._pyclebsch_lattice), which has the same sites,
            links and plaquettes.
        truncation_mode, cutoff: the irrep truncation, as pyclebsch's
            irreps_and_singlets takes it, for example ("T", 1) or ("B", 3).
            pyclebsch validates them.

    Returns {(plane, signature): {(f, i): <f| box + box^dagger |i>}}, as
    pyclebsch groups the matrix elements, where f and i are plaquette states
    of the plaquettes with that plane and signature, in ymcirc's format
    (ymcirc._abstract.PlaquetteState). f is the final and i the initial
    state of the matrix element <f| box |i> that pyclebsch computes (see
    below for information about <f| box^dagger |i>). signature is the
    F-ordered half-link directions at each vertex of a plaquette
    (ymcirc._abstract.Signature). Each pair of plaquette states with a
    nonzero matrix element appears once, in one order, and no state
    is paired with itself. Values are rounded to PRES decimals, and products
    of site factors smaller than EPS are dropped. The caller owns the
    returned dict.

    pyclebsch computes the matrix elements <f| box |i> of box alone. Each one
    is also the matrix element of box + box^dagger between the same states
    for the reason that in

    <f| box + box^dagger |i> = <f| box |i> + <i| box |f>*,

    the second term is zero whenever the first is not. One way to see
    this is to make use of how the triality (the integer (p - q) mod 3 of the
    SU(3) irrep (p, q)) changes under action by box. box multiplies each of the
    plaquette's four active links by U or by U^dagger, so it changes
    the triality of each one by +/-1 (mod 3), fixed by the link's
    orientation around the plaquette. box^dagger changes each active
    link's triality by the opposite amount. A pair of states with both
    <f| box |i> and <i| box |f> nonzero would need +1 = -1 (mod 3) on every
    active link. For the same reason, box has no diagonal matrix elements.
    """
    elements_by_signature = plaquette_matrix_elements_by_signature(
        lattice_def._pyclebsch_lattice, N_COLORS, truncation_mode, cutoff, EPS, PRES
    )
    result: PlaquetteMatrixElements = {}
    for (plane, site_half_links), elements in elements_by_signature.items():
        # pyclebsch's site half-links are ymcirc's signature. The cast narrows
        # pyclebsch's type (which also allows an empty tuple of half-links) to
        # ymcirc's Signature. pyclebsch returns an empty tuple only for a
        # plaquette the lattice lacks. Since site_half_links is a key of
        # plaquette_matrix_elements_by_signature's result, whose keys are those
        # of plaquettes_by_signature, these half-links belong to one of the
        # lattice's plaquettes.
        signature = cast(Signature, site_half_links)
        result[(plane, signature)] = {
            (
                plaq_state_pyclebsch_to_ymcirc_format(final),
                plaq_state_pyclebsch_to_ymcirc_format(initial),
            ): value
            for (final, initial), value in elements.items()
        }
    return result
