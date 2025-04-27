"""
Helper functions to construct electric Hamiltonian through Pauli decomposition.

Electric Hamiltonian is diagonal in computational basis. Helper functions first
construct the Pauli decomposition for matrix with 1 at (i,i) and weights
decomposition by the electric Casimir (according to the link_bitmap).
"""
from __future__ import annotations
from ymcirc.conventions import (IrrepWeight, BitString, IrrepBitmap,
                                LatticeStateEncoder)
import numpy as np
from typing import List
import warnings


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


def _gt_pattern_iweight_to_casimir(gt_tuple: IrrepWeight) -> float:
    """
    Generate the Casimir eigenvalue from a GT-pattern i-Weight.

    See arXiv:2101.10227, eq 3.
    """
    p = gt_tuple[0] - gt_tuple[1]
    q = gt_tuple[1]
    return (p**2 + q**2 + p * q + 3 * p + 3 * q) / 3.0


def _handle_electric_energy_unphysical_states(global_state: str,
                                              link_state: str,
                                              note_unphysical_states: bool,
                                              stop_on_unphysical_states: bool) -> float:
    """ 
    Handle unphysical link states in electric energy calculation.
    """
    if (stop_on_unphysical_states):
        # Terminate simulation
        raise ValueError(
            "Computation of electric energy terminated due to unphysical link state "
            + str(link_state) + " in " + str(global_state))
    elif (note_unphysical_states):
        # Just print an error, but continue simulation with electric energy for the state 0.0
        warning_msg = f"Unphysical link state {link_state} in {global_state} found during computation of electric energy."
        warnings.warn(warning_msg)
        return 0.0
    else:
        return 0.0


def casimirs(link_bitmap: IrrepBitmap) -> List[float]:
    """Generate a list of Casimir eigenvalues from link bitmap i-Weights."""
    return [
        _gt_pattern_iweight_to_casimir(irrep)
        for irrep in list(link_bitmap.keys())
    ]


def convert_bitstring_to_evalue(
        global_bitstring: str,
        lattice_encoder: LatticeStateEncoder,
        note_unphysical_states: bool = True,
        stop_on_unphysical_states: bool = False) -> float:
    """
    Convert lattice links to total energy.

    Used for computing the average electric energy. 
    Chunks the bitstring into |v l1 l2 ..> for each vertex when 
    there are bag states. Default behavior for unphysical states is to note
    them and assign 0.0 energy. This can by logged with note_unphysical_states,
    or a ValueError can be raised with stop_on_unphysical_states.
    """
    link_bitmap = lattice_encoder.link_bitmap
    vertex_bitmap = lattice_encoder.vertex_bitmap

    casimirs = [
        _gt_pattern_iweight_to_casimir(irrep)
        for irrep in list(link_bitmap.keys())
    ]
    casimirs_dict = {}

    # encode using iweights (such as (1, 0 ,0))
    for i, iweight in enumerate(list(link_bitmap.keys())):
        casimirs_dict[iweight] = casimirs[i]

    evalue = 0.0
    len_string = lattice_encoder.expected_link_bit_string_length

    has_vertex_bitmap_data = not len(vertex_bitmap) == 0
    if not has_vertex_bitmap_data:
        for i in range(0, len(global_bitstring), len_string):
            link_bitstring_chunk = global_bitstring[i:i + len_string]
            link_state = lattice_encoder.decode_bit_string_to_link_state(
                link_bitstring_chunk)
            # Handle unphysical link state
            if (link_state is None):
                return _handle_electric_energy_unphysical_states(
                    global_bitstring, link_bitstring_chunk,
                    note_unphysical_states, stop_on_unphysical_states)
            evalue += casimirs_dict[link_state]
    else:
        vertex_singlet_length = lattice_encoder.expected_vertex_bit_string_length
        # Find the spatial dimension of the lattice from vertex state
        spatial_dim = int(len(list(vertex_bitmap.keys())[0][0]) / 2.0)
        if (spatial_dim == 1):
            total_length_x = vertex_singlet_length + (spatial_dim +
                                                      1) * len_string
            total_length_y = vertex_singlet_length + spatial_dim * len_string

            chunks = [
                global_bitstring[i:i + total_length_x + total_length_y]
                for i in range(0, len(global_bitstring), total_length_x +
                               total_length_y)
            ]

            for chunk in chunks:
                x_chunk = chunk[:total_length_x]
                y_chunk = chunk[total_length_x:total_length_x + total_length_y]

                for i in range(vertex_singlet_length, total_length_x,
                               len_string):
                    link_bitstring_chunk = x_chunk[i:i + len_string]
                    link_state = lattice_encoder.decode_bit_string_to_link_state(
                        link_bitstring_chunk)
                    # Handle unphysical link state
                    if (link_state is None):
                        return _handle_electric_energy_unphysical_states(
                            global_bitstring, link_bitstring_chunk,
                            note_unphysical_states, stop_on_unphysical_states)
                    evalue += casimirs_dict[link_state]
                for i in range(vertex_singlet_length, total_length_y,
                               len_string):
                    link_bitstring_chunk = y_chunk[i:i + len_string]
                    link_state = lattice_encoder.decode_bit_string_to_link_state(
                        link_bitstring_chunk)
                    # Handle unphysical link state
                    if (link_state is None):
                        return _handle_electric_energy_unphysical_states(
                            global_bitstring, link_bitstring_chunk,
                            note_unphysical_states, stop_on_unphysical_states)
                    evalue += casimirs_dict[link_state]

        else:
            total_length = vertex_singlet_length + spatial_dim * len_string

            chunks = [
                global_bitstring[i:i + total_length]
                for i in range(0, len(global_bitstring), total_length)
            ]

            for chunk in chunks:
                for i in range(vertex_singlet_length, len(chunk), len_string):
                    link_bitstring_chunk = chunk[i:i + len_string]
                    link_state = lattice_encoder.decode_bit_string_to_link_state(
                        link_bitstring_chunk)
                    if (link_state is None):
                        return _handle_electric_energy_unphysical_states(
                            global_bitstring, link_bitstring_chunk,
                            note_unphysical_states, stop_on_unphysical_states)
                    evalue += casimirs_dict[link_state]

    return evalue


def electric_hamiltonian(link_bitmap: IrrepBitmap) -> List[float]:
    """Give the total Pauli decomposition."""
    N = len(list(link_bitmap.values())[0])
    casimir_values = [
        _gt_pattern_iweight_to_casimir(irrep)
        for irrep in list(link_bitmap.keys())
    ]
    pauli_strings = [
        _diagonal_pauli_string(bit_string=list(link_bitmap.values())[i],
                               N=N,
                               casimir=casimir_values[i])
        for i in range(len(list(link_bitmap.values())))
    ]

    return [sum(x) for x in zip(*pauli_strings)]
