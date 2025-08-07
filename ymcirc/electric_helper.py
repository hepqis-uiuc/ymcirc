"""
Helper functions to construct electric Hamiltonian through Pauli decomposition.

Electric Hamiltonian is diagonal in computational basis. Helper functions first
construct the Pauli decomposition for matrix with 1 at (i,i) and weights
decomposition by the electric Casimir (according to the link_bitmap).
"""
from __future__ import annotations
import logging
from ymcirc.conventions import (IrrepWeight, BitString, IrrepBitmap,
                                LatticeStateEncoder)
from ymcirc.parsed_lattice_result import ParsedLatticeResult
from typing import List

# Set up module-specific logger
logger = logging.getLogger(__name__)


def _bitwise_addition(bit_string_1: str | BitString,
                      bit_string_2: str | BitString) -> int:
    """Add two bit bit strings bitwise mod 2."""
    return sum([(int(bit_string_1[i]) * int(bit_string_2[i]))
                for i in range(len(bit_string_1))]) % 2


def _diagonal_pauli_string(bit_string: str, N: int,
                           casimir: float) -> List[int]:
    """
    Generate a Pauli decomposition.

    Given the binary-ordered bit string computational basis of order 2^N,
    takes in bit_string and generates the Pauli decomposition to put the revelant
    Casimir at (int(bit_string), int(bit_string)).

    For a given Pauli string p (w/t I == 0; Z == 1), we have that the coefficient
    for decomposing bit_string is (casimir/2^N)(-1)^(bit_add(p, bit_string)).
    """
    pauliz_list = []
    for i in range(2**N):
        i_string = str('{0:0' + str(N) + 'b}').format(i)
        pauliz_list.append((1 / (2**N)) * casimir *
                           ((-1)**(_bitwise_addition(i_string, bit_string))))
    return pauliz_list


def gt_pattern_iweight_to_casimir(gt_tuple: IrrepWeight) -> float:
    """
    Generate the Casimir eigenvalue from a GT-pattern i-Weight.

    Physically, this corresponds to the electric energy of a particular single-link state.

    See arXiv:2101.10227, eq 3.
    """
    p = gt_tuple[0] - gt_tuple[1]
    q = gt_tuple[1]
    return (p**2 + q**2 + p * q + 3 * p + 3 * q) / 3.0


def casimirs(link_bitmap: IrrepBitmap) -> List[float]:
    """Generate a list of Casimir eigenvalues from link bitmap i-Weights."""
    return [
        gt_pattern_iweight_to_casimir(irrep)
        for irrep in list(link_bitmap.keys())
    ]


# TODO Use ParsedLatticeResult?
def convert_bitstring_to_evalue(
        global_lattice_bit_string: str,
        lattice_encoder: LatticeStateEncoder,
        warn_on_unphysical_links: bool = True,
        error_on_unphysical_links: bool = False) -> float:
    """
    Convert lattice links to total energy.

    Used for computing the average electric energy. 
    Chunks the bitstring into |v l1 l2 ..> for each vertex when 
    there are multiplicity data on the vertices. Default behavior for unphysical states is to note
    them and assign 0.0 energy. This can by logged with warn_on_unphysical_links,
    or a ValueError can be raised with error_on_unphysical_links.
    """
    # Hacky way to infer size from shape.
    # Don't do this long term because it breaks for non-hypercubic lattice
    # when d >= 2.
    is_lattice_hypercubic = len(set(lattice_encoder.lattice_def.shape)) == 1
    if lattice_encoder.lattice_def.dim >= 2 and is_lattice_hypercubic is False:
        raise NotImplementedError("Electric energy computation for non-hypercubic lattices with dim >= 2 not yet implemented.")
    lattice_size = lattice_encoder.lattice_def.shape[0]
    parsed_global_lattice_bit_string = ParsedLatticeResult(
        dimensions=lattice_encoder.lattice_def.dim,
        size=lattice_size,
        global_lattice_measurement_bit_string=global_lattice_bit_string,
        lattice_encoder=lattice_encoder)

    evalue = 0.0
    for (vertex_address, attached_link_addresses) in parsed_global_lattice_bit_string.get_traversal_order():
        for link_address in attached_link_addresses:
            current_link_iweight = parsed_global_lattice_bit_string.get_link(link_address)
            link_is_unphysical = current_link_iweight == None
            if link_is_unphysical and error_on_unphysical_links:
                raise ValueError(
                    "Computation of electric energy terminated due to unphysical link state at "
                    f"link address {link_address}."
                )
            elif link_is_unphysical and warn_on_unphysical_links:
                warning_msg = f"Unphysical link state encountered at link address {link_address} during computation of electric energy."
                logger.warning(warning_msg)
                evalue += 0
            else:
                if not isinstance(current_link_iweight, tuple):
                    raise TypeError(f"Non-iweight encountered in electric energy computation: {current_link_iweight} of type {type(current_link_iweight)}.")
                evalue += gt_pattern_iweight_to_casimir(current_link_iweight)

    return evalue


def electric_hamiltonian(link_bitmap: IrrepBitmap) -> List[float]:
    """Give the total Pauli decomposition."""
    N = len(list(link_bitmap.values())[0])
    casimir_values = [
        gt_pattern_iweight_to_casimir(irrep)
        for irrep in list(link_bitmap.keys())
    ]
    pauli_strings = [
        _diagonal_pauli_string(bit_string=list(link_bitmap.values())[i],
                               N=N,
                               casimir=casimir_values[i])
        for i in range(len(list(link_bitmap.values())))
    ]

    logger.debug(f"Computing electric Hamiltonian for link bitmap: {link_bitmap}")

    return [sum(x) for x in zip(*pauli_strings)]
