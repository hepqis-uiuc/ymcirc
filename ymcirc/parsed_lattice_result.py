"""Wrapper for parsing bit strings obtained from measuring all the registers in a LatticeRegisters instance."""
from __future__ import annotations
from typing import List, Union
from ymcirc._abstract.lattice_data import (
    LatticeData, Plaquette, DimensionalitySpecifier, LatticeVector,
    LinkUnitVectorLabel, LinkAddress)
from ymcirc.conventions import LatticeStateEncoder, IrrepWeight, MultiplicityIndex
from ymcirc.lattice_registers import LatticeRegisters

# Type alias to deal with the fact that underlying measurement results
# are bit strings, and parsed measurement results are either IrrepWeight
# instances or None depending on whether the bit string decodes succesfully
# or is garbage.
MeasurementData = Union[str, IrrepWeight, MultiplicityIndex, None]


# TODO implement __hash__ method based on the global bit string
# so that this class can be used as a key in a "counts" dictionary.
class ParsedLatticeResult(LatticeData[MeasurementData]):
    """
    Wrapper for string resulting when all qubits in a LatticeRegisters instance are measured.

    Because both this class and the LatticeRegisters class subclasses of LatticeData, they
    automatically utilize the same iteration order when iterating over the data they contain.
    That means ParsedLatticeResult should be used to parse bit strings corresponding to
    measurements of all the registers in a simulation circuit to minimize the possibility
    of subtle errors.

    There is a convenience method from initializing instances of ParsedLatticeResult
    using a LatticeRegisters instance to extract information about lattice geometry.
    Additionally, a LatticeStateEncoder instance is needed because this class contains
    information about how to map bit substrings corresponding to local degrees of
    freedom onto physically meaningful irrep data.
    """

    def __init__(
            self,
            dimensions: DimensionalitySpecifier,
            size: int | tuple[int, ...],
            global_lattice_measurement_bit_string: str,
            lattice_encoder: LatticeStateEncoder,
            periodic_boundary_conds: bool | tuple[bool, ...] = True,
            ):
        """Parse through global_lattice_measurement_bitstring and convert to i-weights."""
        super().__init__(dimensions, size, periodic_boundary_conds)
        raise NotImplementedError("Method not yet implemented.")

    def get_vertex(self, lattice_vector: LatticeVector, get_bit_string: bool = False) -> MeasurementData:
        """
        Return the measurement result for the vertex specified by lattice_vector.

        If get_bit_string is True, then the underlying bit string is returned instead
        of the decoded MultiplicityIndex for the vertex.

        If get_bit_string is False, then returns either a MultiplicityIndex labeling
        the multiplicity of the singlet at that vertex, or None if the bit string
        at the vertex fails to decode.
        """
        raise NotImplementedError("Method not yet implemented.")

    def get_link(self, link_address: LinkAddress, get_bit_string: bool = False) -> MeasurementData:
        """
        Return the measurement result for the link specified by link_address.

        The argument link_address consists of a lattice vector with a
        positive unit_vector_label specifies the link which is in the
        positive direction along the dimension specified by unit_vector_label
        from the vertex given by lattice vector. A negative unit_vector_label
        specifies the opposite link.

        Example (d=3/2 with periodic boundary conditions):

        (0, 1) ----- (1, 1) ----- (pbc)
          |            |
          |            |
        (0, 0) ----- (1, 0) ----- (pbc)

        unit_vector_label = 1 labels the positive horizontal direction.
        unit_vector_label = 2 labels the positive vertical direction.
        We can address the bottom-middle link via either of the following:
            - lattice_vector = (0, 0), unit_vector_label = 1
            - lattice_vector = (1, 0), unit_vector_label = -1

        The conversion to a "normalized" link_address using a positive
        unit_vector_label is automatically handled internally.

        If get_bit_string is True, then the underlying bit string is returned
        instead of the decoded irrep data.

        If get_bit_string is False, then returns either an IrrepWeight labeling
        the irrep on the link, or None if the bit string
        at the link fails to decode.
        """
        raise NotImplementedError("Method not yet implemented.")

    def get_plaquettes(self,
                       lattice_vector: LatticeVector,
                       e1: Union[LinkUnitVectorLabel, None] = None,
                       e2: Union[LinkUnitVectorLabel, None] = None,
                       get_bit_string: bool = False,
                       ) -> Plaquette[MeasurementData] | List[Plaquette[MeasurementData]]:
        """
        Return the list of all "positive" Plaquettes associated with the vertex lattice_vector.

        As with get_vertex and get_link methods, get_bit_string controls whether to
        return underlying bit string data for each degree of freedom, or the decoded
        result.

        The "positivity" convention is that the list of returned plaquettes corresponds to those
        defined by all pairs of orthogonal positive unit vectors at the vertex lattice_vector.
        Conventionally, the "lower" dimension labels the first element of the tuples
        representing planes, and the plaquettes are sorted.

        Examples:
          - d = 3 has planes labeled by (1, 2), (1, 3), and (2, 3).
          - d = 4 has planes labeled by (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), and (3, 4).

        If a particular plaquette is desired, this can be specified by either providing the
        link unit vector directions e1 and e2 to defining a plane. Sign is ignored when
        manually specifying the plane of a specific plaquette.

        Return plaquettes will all have lattice_vector as the "bottom-left" vertex.
        This corresponds to "v1" in the following diagram:

        v4 ----l3--- v3
        |            |
        |            |
        l4           l2
        |            |
        |            |
        v1 ----l1--- v2

        Note that this ordering of data DOES NOT match the ordering of links
        and vertices when iterating over the entire lattice!
        """
        raise NotImplementedError("Method not yet implemented.")

    @classmethod
    def from_lattice_registers(
            cls,
            lattice_registers: LatticeRegisters,
            global_lattice_measurement_bit_string: str,
            lattice_encoder: LatticeStateEncoder
    ) -> ParsedLatticeResult:
        """Initialize from a LatticeRegisters instance and a measurement bitstring corresponding to that instance."""
        raise NotImplementedError("Method not yet implemented.")
