"""Wrapper for parsing bit strings obtained from measuring all the registers in a LatticeRegisters instance."""
from __future__ import annotations
from typing import Dict, List, Union
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
        # Initialize dicts to hold measurement data.
        self._decoded_links: Dict[LinkAddress, IrrepWeight | None] = {}
        self._decoded_vertices: Dict[LatticeVector, MultiplicityIndex | None] = {}
        self._bit_strings_links: Dict[LinkAddress, str] = {}
        self._bit_strings_vertices: Dict[LatticeVector, str] = {}

        # Walk through lattice, decoding each DoF using the encoder.
        start_current_vertex_and_links_substring_idx = 0
        for (lattice_traversal_idx, (current_vertex_address, current_connected_link_addresses)) in enumerate(lattice_encoder.lattice_def.get_traversal_order()):
            # Extract substring for current vertex and connected links.
            num_connected_links = len(current_connected_link_addresses)            
            len_current_vertex_and_links_substring = (
                lattice_encoder.expected_vertex_bit_string_length +
                num_connected_links*lattice_encoder.expected_link_bit_string_length
            )
            end_current_vertex_and_links_substring_idx = start_current_vertex_and_links_substring_idx + len_current_vertex_and_links_substring
            current_vertex_and_links_substring = global_lattice_measurement_bit_string[start_current_vertex_and_links_substring_idx:end_current_vertex_and_links_substring_idx]

            # Store vertex bit string and decoded value.
            vertex_start_idx = 0
            vertex_end_idx = vertex_start_idx + lattice_encoder.expected_vertex_bit_string_length
            self._bit_strings_vertices[current_vertex_address] = current_vertex_and_links_substring[vertex_start_idx:vertex_end_idx]
            self._decoded_vertices[current_vertex_address] = lattice_encoder.decode_bit_string_to_vertex_state(self._bit_strings_vertices[current_vertex_address])

            # Store link bit strings and decoded values.
            current_links_substring = current_vertex_and_links_substring[vertex_end_idx:]
            for current_link_number_idx, current_link_address in enumerate(current_connected_link_addresses):
                current_link_start_idx = current_link_number_idx * lattice_encoder.expected_link_bit_string_length
                current_link_end_idx = current_link_start_idx + lattice_encoder.expected_link_bit_string_length
                self._bit_strings_links[current_link_address] = current_links_substring[current_link_start_idx:current_link_end_idx]
                self._decoded_links[current_link_address] = lattice_encoder.decode_bit_string_to_link_state(self._bit_strings_links[current_link_address])

            start_current_vertex_and_links_substring_idx += len_current_vertex_and_links_substring

    def get_vertex(self, lattice_vector: LatticeVector, get_bit_string: bool = False) -> MeasurementData:
        """
        Return the measurement result for the vertex specified by lattice_vector.

        If get_bit_string is True, then the underlying bit string is returned instead
        of the decoded MultiplicityIndex for the vertex.

        If get_bit_string is False, then returns either a MultiplicityIndex labeling
        the multiplicity of the singlet at that vertex, or None if the bit string
        at the vertex fails to decode.
        """
        if get_bit_string is False:
            return self._decoded_vertices[lattice_vector]
        else:
            return self._bit_strings_vertices[lattice_vector]

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
        normalized_link_address = self._normalize_link_address(link_address)
        if get_bit_string is False:
            return self._decoded_links[normalized_link_address]
        else:
            return self._bit_strings_links[normalized_link_address]

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
        return super().get_plaquettes(lattice_vector, e1, e2, get_bit_string=get_bit_string)


    # TODO this might actually not be needed since traversal order moved to lattice def inside LatticeStateEncoder
    @classmethod
    def from_lattice_registers(
            cls,
            lattice_registers: LatticeRegisters,
            global_lattice_measurement_bit_string: str,
            lattice_encoder: LatticeStateEncoder
    ) -> ParsedLatticeResult:
        """Initialize from a LatticeRegisters instance and a measurement bitstring corresponding to that instance."""
        raise NotImplementedError("Method not yet implemented.")
