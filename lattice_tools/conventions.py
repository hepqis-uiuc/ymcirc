"""
This module is a canonical source for bit string encoding conventions.

########## Irrep link state encodings ##########

There are two encodings of irrep link states:

- IRREP_TRUNCATION_DICT_1_3_3BAR
- IRREP_TRUNCATION_DICT_1_3_3BAR_6_6_BAR_8

These are dictionaries which map length-3 tuples to unique bitstrings.
The tuples represent "i-Weights", which are a way of uniquely labeling
SU(N) irreps which come from working with the Gelfand-Tsetlin pattern calculus.
See Arne et al for more details (https://doi.org/10.1063/1.3521562).
As a convenience, the following constants are defined which take on the correct
i-Weights values:

- ONE = (0, 0, 0)
- THREE = (1, 0, 0)
- THREE_BAR = (1, 1, 0)
- SIX = (2, 0, 0)
- SIX_BAR = (2, 2, 0)
- EIGHT = (2, 1, 0)

There are generically "leftover" states when encoding a particular truncation
into a set of qubits (i.e. the state space of a link in a particular truncation
is usually not of dimension 2^n). These are omitted from the irrep truncation
dicts.

The "bag states" at vertices encode the information needed to establish which
singlet state the vertex is in. This is ambiguous in general because there are
multiple ways to create singlets from link assignments in arbitrary dimension,
and for arbitrary irrep truncations, there will be complete link assignments
which yield various "multiplicites" of singlets.

########## Vertex singlet encodings ##########

The dict VERTEX_SINGLET_BITMAPS can be indexed with strings to obtain bitmaps
for the following cases:

- "d=3/2, T1"
- "d=3/2, T2"
- "d=2, T1"
- "d=2, T2"
- "d=3, T1"
- "d=3, T2"

T1 refers to the ONE, THREE, THREE_BAR truncation, while T2 includes the
additional states SIX, SIX_BAR, and EIGHT. To get the bitmap for a particular
case, use the following syntax:

VERTEX_SINGLET_BITMAPS["d=2, T2"]

Since the key in this dict is a string, spacing and capitalization is
necessary. Once a particular bitmap has been accessed, that will
yield a dict containing keys which are tuples of iweights with a
multiplicity index, and a bitstring. For example:

VERTEX_SINGLET_BITMAPS["d=3/2, T1"] = {
    (((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1): '00',
    (((0, 0, 0), (1, 0, 0), (1, 1, 0)), 1): '01',
    (((1, 0, 0), (1, 0, 0), (1, 0, 0)), 1): '10',
    (((1, 1, 0), (1, 1, 0), (1, 1, 0)), 1): '11'
}

The integer after the tuple of i-Weights tells us that a vertex with these
three irreps on one of the links will have only one singlet. Compare this
with the following case:

VERTEX_SINGLET_BITMAPS["d=2, T1"] = {
    (((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), 1): '000',
    (((0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 1, 0)), 1): '001',
    (((0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)), 1): '010',
    (((0, 0, 0), (1, 1, 0), (1, 1, 0), (1, 1, 0)), 1): '011',
    (((1, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0)), 1): '100',
    (((1, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0)), 2): '101'
}

Observe that there are two multiplicities of the
((1, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0)) set of irreps coming into a
particular vertex; these get assigned to distinct bit strings.

Finally, it is a special quirk of d=3/2, T1 that it is possible to fully deduce
the singlet state with knowledge of only two links. Therefore, the following
bitmap is provided since vertex degrees of freedom are unnecessary in that
case:

VERTEX_SINGLET_DICT_D_3HALVES_133BAR_NO_VERTEX_DATA = {}
"""
from __future__ import annotations
import ast
from pathlib import Path
from typing import Tuple, Dict
import json

# Filesystem stuff.
_PROJECT_ROOT = Path(__file__).parent
_DATA_DIR = _PROJECT_ROOT / "lattice_tools_data/singlet-bitmaps/"
_SINGLET_DATA_FILE_PATHS: Dict[str, Path] = {
    "d=3/2, T1": _DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(3_2)_singlet_bitmaps.json",
    "d=3/2, T2": _DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0)]_dim(3_2)_singlet_bitmaps.json",
    "d=2, T1": _DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(2)_singlet_bitmaps.json",
    "d=2, T2": _DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0)]_dim(2)_singlet_bitmaps.json",
    "d=3, T1": _DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0)]_dim(3)_singlet_bitmaps.json",
    "d=3, T2": _DATA_DIR / "[(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0)]_dim(3)_singlet_bitmaps.json",
}

# Useful type aliases.
IrrepWeight = Tuple[int, int, int]
BitString = str
MultiplicityIndex = int
IrrepBitmap = Dict[IrrepWeight, BitString]
SingletsDef = Tuple[Tuple[IrrepWeight, ...], Tuple[MultiplicityIndex, ...]]
VertexBag = Tuple[Tuple[IrrepWeight, ...], MultiplicityIndex]
VertexMultiplicityBitmap = Dict[VertexBag, BitString]

# Irrep iweights (top row of GT pattern).
ONE: IrrepWeight = (0, 0, 0)
THREE: IrrepWeight = (1, 0, 0)
THREE_BAR: IrrepWeight = (1, 1, 0)
SIX: IrrepWeight = (2, 0, 0)
SIX_BAR: IrrepWeight = (2, 2, 0)
EIGHT: IrrepWeight = (2, 1, 0)

# Irrep encoding bitmaps.
IRREP_TRUNCATION_DICT_1_3_3BAR: IrrepBitmap = {
    ONE: "00",
    THREE: "10",
    THREE_BAR: "01"
}
IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8: IrrepBitmap = {
    ONE: "000",
    THREE: "100",
    THREE_BAR: "001",
    SIX: "110",
    SIX_BAR: "011",
    EIGHT: "111"
}

# Load vertex singlet data from precomputed json files.
# Safer to use ast.literal_eval than eval to convert data keys to tuples.
# The latter can execute arbitrary potentially malicious code while
# the worst case attack vector for litera_eval would be to crash
# the python process.
# See https://docs.python.org/3/library/ast.html#ast.literal_eval
# for more information.
VERTEX_SINGLET_BITMAPS: Dict[str, VertexMultiplicityBitmap] = {}
# Load singlet bag state encoding bitmaps.
for dim_trunc_case, file_path in _SINGLET_DATA_FILE_PATHS.items():
    with file_path.open('r') as json_data:
        d = json.load(json_data)
        VERTEX_SINGLET_BITMAPS[dim_trunc_case] = {ast.literal_eval(key): value for key, value in d.items()}  

# This is a special case where it isn't strictly necessary
# to disambiguate singlets via additional vertex data.
# Essentially, with 3 irreps and 3 links per vertex,
# Knowledge of two link irrep states is sufficient to unambiguously
# determine that of the third link.
VERTEX_SINGLET_DICT_D_3HALVES_133BAR_NO_VERTEX_DATA: VertexMultiplicityBitmap = {}

if __name__ == "__main__":
    print("Testing singlet bitmaps...")
    # Check that there are 6 vertex singlet bitmaps (3 dimensionalities * 2 irrep truncations).
    there_are_six_singlet_bitmaps = len(VERTEX_SINGLET_BITMAPS) == 6
    print(f"\nlen(VERTEX_SINGLET_BITMAPS) == 6? {there_are_six_singlet_bitmaps}.")
    assert there_are_six_singlet_bitmaps

    # Check that all vertex bitmaps have the right amount of distinct singlets.
    # The values for expected_nums come from classical precomputation.
    cases = ["d=3/2, T1", "d=3/2, T2", "d=2, T1", "d=2, T2", "d=3, T1", "d=3, T2"]
    expected_nums = [4, 16, 6, 66, 28, 1646]
    for current_case, expected_num_singlets in zip(cases, expected_nums):
        # Number check.
        test_result_num_singlets = len(VERTEX_SINGLET_BITMAPS[current_case]) == expected_num_singlets
        print(f'\nlen(VERTEX_SINGLET_BITMAPS["{current_case}"]) == {expected_num_singlets}? {test_result_num_singlets}.')
        if test_result_num_singlets is False:
            print(f"Encountered {len(VERTEX_SINGLET_BITMAPS[current_case])} singlets.")

        assert test_result_num_singlets

        # Uniqueness of encoding check.
        test_result_singlets_have_unique_bitstring = expected_num_singlets == len(set(VERTEX_SINGLET_BITMAPS[current_case].values()))
        print(f'# of distinct bitstring encodings for VERTEX_SINGLET_BITMAPS["{current_case}"] == {expected_num_singlets}? {test_result_singlets_have_unique_bitstring}.')
        if test_result_singlets_have_unique_bitstring is False:
            print(f"Encountered {len(set(VERTEX_SINGLET_BITMAPS[current_case].values()))} distinct bitstrings.")

        assert test_result_singlets_have_unique_bitstring

    print("All tests passed.")
