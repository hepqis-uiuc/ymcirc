"""
This module is a canonical source for conventions about
bit string encodings of link irrep states and
vertex singlet data.

Currently a work in progress.
"""
from __future__ import annotations
from math import ceil
from typing import List, Tuple, Dict
import numpy as np

# Interal convenience method.
_zero_padded_bitstring = lambda value, total_length: bin(value)[2:].zfill(total_length)

# Useful type aliases.
IrrepWeight = Tuple[int, int, int]
BitString = str
MultiplicityIndex = int
IrrepBitmap = Dict[IrrepWeight, BitString]
SingletsDef = Tuple[Tuple[IrrepWeight, ...], Tuple[MultiplicityIndex, ...]]
VertexBag = Tuple[Tuple[IrrepWeight, ...], MultiplicityIndex]
VertexMultiplicityBitmap = Dict[VertexBag, BitString]

# Configure data for mapping between bitstrings and physical link irreps/vertex singlets.
ONE: IrrepWeight = (0, 0, 0)
THREE: IrrepWeight = (1, 0, 0)
THREE_BAR: IrrepWeight = (1, 1, 0)
EIGHT: IrrepWeight = (2, 1, 0)
IRREP_TRUNCATION_DICT_133BAR: IrrepBitmap = {
    ONE: "00",
    THREE: "10",
    THREE_BAR: "01"
}

IRREP_TRUNCATION_DICT_133BAR8: IrrepBitmap = {
    ONE: "00",
    THREE: "10",
    THREE_BAR: "01",
    EIGHT: "11"
}
D_3HALVES_VERTEX_BAGS: List[SingletsDef] = [
    ((ONE, ONE, ONE), (1,)),
    ((ONE, THREE, THREE_BAR), (1, 2, 3)),  # Should actually only have one singlet, for testing
    ((EIGHT, EIGHT, EIGHT), (1, 2)),
]
N_VERTEX_QUBITS_D3HALVES = ceil(
    np.log2(
        sum([len(multiplicities) if len(multiplicities) > 1 else 0 for irreps, multiplicities in D_3HALVES_VERTEX_BAGS])
    ))

# This needs to have unordered link information to enable consistency checks
# all other states are either unambiguous singlets, or outside the gauge invariant subspace.
VERTEX_SINGLET_DICT_D_3HALVES_133BAR: VertexMultiplicityBitmap = {}
VERTEX_SINGLET_DICT_D_3HALVES_133BAR8: VertexMultiplicityBitmap = {}
encoding_num = 0
for singlets in D_3HALVES_VERTEX_BAGS:
    if len(singlets[1]) < 2:
        continue
    for multiplicity_idx in singlets[1]:
        link_irrep_tuple = singlets[0]
        VERTEX_SINGLET_DICT_D_3HALVES_133BAR8[(link_irrep_tuple, multiplicity_idx)] = _zero_padded_bitstring(encoding_num, N_VERTEX_QUBITS_D3HALVES)
        encoding_num += 1
