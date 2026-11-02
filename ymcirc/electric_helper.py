"""
Helper functions to construct electric Hamiltonian through Pauli decomposition.

Electric Hamiltonian is diagonal in computational basis. Helper functions first
construct the Pauli decomposition for matrix with 1 at (i,i) and weights
decomposition by the electric Casimir (according to the link_bitmap).
"""
from __future__ import annotations
import logging
from pyclebsch.su_n_operators import calc_casimir
from ymcirc.conventions import BitString, IrrepBitmap

# Set up module-specific logger
logger = logging.getLogger(__name__)


def _bitwise_addition(bit_string_1: str | BitString,
                      bit_string_2: str | BitString) -> int:
    """Add two bit bit strings bitwise mod 2."""
    return sum([(int(bit_string_1[i]) * int(bit_string_2[i]))
                for i in range(len(bit_string_1))]) % 2


def _diagonal_pauli_string(bit_string: str, N: int,
                           casimir: float) -> list[int]:
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


def casimirs(link_bitmap: IrrepBitmap) -> list[float]:
    """Generate a list of Casimir eigenvalues from link bitmap i-Weights."""
    return [
        calc_casimir(irrep)
        for irrep in list(link_bitmap.keys())
    ]


def electric_hamiltonian(link_bitmap: IrrepBitmap) -> list[float]:
    """Give the total Pauli decomposition."""
    N = len(list(link_bitmap.values())[0])
    casimir_values = [
        calc_casimir(irrep)
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
