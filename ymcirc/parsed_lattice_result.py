"""Wrapper for parsing bit strings obtained from measuring all the registers in a LatticeRegisters instance."""
from __future__ import annotations
import copy
import logging
import warnings
from typing import Dict, List, Union, Optional
from ymcirc._abstract.lattice_data import (
    LatticeData, LatticeDef, Plaquette, DimensionalitySpecifier, LatticeVector,
    LinkUnitVectorLabel, LinkAddress)
from ymcirc.conventions import LatticeStateEncoder, IrrepWeight, MultiplicityIndex

# Set up module-specific logger
logger = logging.getLogger(__name__)


# Type alias to deal with the fact that underlying measurement results
# are bit strings, and parsed measurement results are either IrrepWeight
# instances or None depending on whether the bit string decodes succesfully
# or is garbage.
MeasurementData = Union[str, IrrepWeight, MultiplicityIndex, None]


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

    Note that this class assumes NO ancilla qubits are included in measurement
    strings!
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
        super().__init__(dimensions, size, periodic_boundary_conds, forder=lattice_encoder.lattice_def.forder)

        # Do some validation.
        expected_num_bits = self.n_links * lattice_encoder.expected_link_bit_string_length + self.n_vertices * lattice_encoder.expected_vertex_bit_string_length
        has_non_binary_char = any(char not in ['0', '1'] for char in global_lattice_measurement_bit_string)
        if expected_num_bits != len(global_lattice_measurement_bit_string):
            raise ValueError(f"Expecting length-{expected_num_bits} measurement bit string. Encountered length-{len(global_lattice_measurement_bit_string)} bit string. Please make sure ancilla qubits are stripped out of measurement string, or that you are the right lattice encoder.")
        if self.dim != lattice_encoder.lattice_def.dim:
            raise ValueError(f"Specified a dim-{self.dim} lattice, but using a {LatticeStateEncoder.__name__} with dim-{lattice_encoder.lattice_def.dim}.")
        if self.shape != lattice_encoder.lattice_def.shape:
            raise ValueError(f"Specified a lattice with shape {self.shape}, but using a {LatticeStateEncoder.__name__} for a lattice with shape {lattice_encoder.lattice_def.shape}.")
        if has_non_binary_char is True:
            raise TypeError(f"Measurement bit string {global_lattice_measurement_bit_string} contains one or more non-binary characters.")

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

        # Let's keep these around too. They're handy to have.
        self._global_lattice_measurement_bit_string = global_lattice_measurement_bit_string
        self._lattice_def = lattice_encoder._lattice
        self._encoder = lattice_encoder
        self._lattice_encoder_repr = lattice_encoder.__repr__()

    def __repr__(self):
        class_name = type(self).__name__
        size = self.shape[0]
        return f"{class_name}(dimensions={self.dim}, size={size}, global_lattice_measurement_bit_string={self.global_lattice_measurement_bit_string}, lattice_encoder={self._lattice_encoder_repr}, periodic_boundary_conds={self.periodic_boundary_conds})"

    def __str__(self):
        link_measurements = {link_address: self.get_link(link_address) for link_address in self.link_addresses}
        vertex_measurements = {vertex_address: self.get_vertex(vertex_address) for vertex_address in self.vertex_addresses}
        return f"A parsed measurement of registers for simulation circuit ({self._lattice_def}).\nLink measurements (link address: iweight):\n{link_measurements}\nVertex measurements (vertex address: multiplicity index):\n{vertex_measurements}"

    @property
    def lattice_def(self) -> LatticeDef:
        """Return copy of LatticeDef instance describing the lattice for the global measurement bit string."""
        return copy.deepcopy(self._lattice_def)

    @property
    def global_lattice_measurement_bit_string(self) -> str:
        """
        Return the global lattice measurement bit string.

        For instances created via __init__ (full measurement), returns the
        original bit string. For instances created via factory methods
        (partial measurement), reconstructs the bit string from internal
        data using get_traversal_order(), with "X" placeholders for
        unmeasured degrees of freedom.
        """
        if self._global_lattice_measurement_bit_string is not None:
            return self._global_lattice_measurement_bit_string

        # Reconstruct from traversal order using undecoded (bitstring) data.
        bitstring = ""
        for vertex_addr, link_addrs in self.get_traversal_order():
            bitstring += self.get_vertex(vertex_addr, get_bit_string=True)
            for link_addr in link_addrs:
                bitstring += self.get_link(link_addr, get_bit_string=True)
        return bitstring

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

    def __hash__(self):
        """Hash based on measurement string, and data that uniquely specifies lattice geometry."""
        if not hasattr(self, '_hash_cache'):
            self._hash_cache = hash((self.global_lattice_measurement_bit_string, self._lattice_def.dim, self._lattice_def.shape, self._lattice_def.periodic_boundary_conds))
        return self._hash_cache

    def __eq__(self, other) -> bool:
        """Equality based on the same fields used by __hash__."""
        if not isinstance(other, ParsedLatticeResult):
            return NotImplemented
        return (
            self.global_lattice_measurement_bit_string == other.global_lattice_measurement_bit_string
            and self._lattice_def.dim == other._lattice_def.dim
            and self._lattice_def.shape == other._lattice_def.shape
            and self._lattice_def.periodic_boundary_conds == other._lattice_def.periodic_boundary_conds
        )

    @classmethod
    def _create_partial(cls, encoder: LatticeStateEncoder) -> ParsedLatticeResult:
        """
        Create a partially-initialized instance with placeholder data.

        All links and vertices are initialized with None (decoded) and
        "X"-padded bitstrings (undecoded). Factory methods should overwrite
        entries for measured degrees of freedom.
        """
        lattice_def = encoder.lattice_def
        size = lattice_def.shape[0]
        instance = cls.__new__(cls)
        LatticeDef.__init__(instance, lattice_def.dim, size, lattice_def.periodic_boundary_conds, forder=lattice_def.forder)

        instance._decoded_links = {}
        instance._decoded_vertices = {}
        instance._bit_strings_links = {}
        instance._bit_strings_vertices = {}

        # Pre-fill all addresses with placeholders.
        for vertex_addr in lattice_def.vertex_addresses:
            vertex_addr = tuple(vertex_addr)
            instance._decoded_vertices[vertex_addr] = None
            instance._bit_strings_vertices[vertex_addr] = "X" * encoder.expected_vertex_bit_string_length

        for link_addr in lattice_def.link_addresses:
            instance._decoded_links[link_addr] = None
            instance._bit_strings_links[link_addr] = "X" * encoder.expected_link_bit_string_length

        instance._global_lattice_measurement_bit_string = None
        instance._lattice_def = encoder._lattice
        instance._encoder = encoder
        instance._lattice_encoder_repr = encoder.__repr__()

        return instance

    @staticmethod
    def from_links_and_vertices(
        links_dict: Dict[LinkAddress, IrrepWeight],
        vertices_dict: Union[Dict[LatticeVector, MultiplicityIndex], None] = None,
        *,
        encoder: LatticeStateEncoder,
    ) -> ParsedLatticeResult:
        """
        Create a ParsedLatticeResult from decoded link and vertex data.

        Links and vertices not present in the input dicts will return None
        when queried (decoded) or "X"-padded strings (undecoded bitstring).

        Arguments:
            - links_dict: Maps LinkAddress -> IrrepWeight (decoded link state).
            - vertices_dict: Optional. Maps LatticeVector -> MultiplicityIndex.
            - encoder: LatticeStateEncoder for encoding/decoding and lattice geometry.
        """
        instance = ParsedLatticeResult._create_partial(encoder)

        for link_addr, link_state in links_dict.items():
            normalized = instance._normalize_link_address(link_addr)
            instance._decoded_links[normalized] = link_state
            instance._bit_strings_links[normalized] = encoder.encode_link_state_as_bit_string(link_state)

        if vertices_dict is not None:
            for vertex_addr, mult_idx in vertices_dict.items():
                vertex_addr = tuple(vertex_addr)
                instance._decoded_vertices[vertex_addr] = mult_idx
                instance._bit_strings_vertices[vertex_addr] = encoder.encode_vertex_state_as_bit_string(mult_idx)

        return instance

    @staticmethod
    def from_partial_measurement(
        measurements: List[tuple[LatticeVector, str] | tuple[LinkAddress, str] | tuple[tuple[LinkAddress, LinkUnitVectorLabel, LinkUnitVectorLabel], str]],
        encoder: LatticeStateEncoder,
    ) -> ParsedLatticeResult:
        """
        Create a ParsedLatticeResult from partial measurement data.

        Each element of measurements is a 2-tuple (address, bitstring) where:
        - address is a LatticeVector for vertex measurements (e.g., (0, 0))
        - address is a LinkAddress for link measurements (e.g., ((0, 0), 1))
        - address is (LatticeVector, e1, e2) for plaquette measurements
          (e.g., ((0, 0), 1, 2))

        If an address which doesn't fit into one of these three categories is
        encountered, a ValueError will be raised.

        For plaquette measurements, the bitstring follows the plaquette encoding
        convention: |v1 v2 v3 v4 l1 l2 l3 l4 c1... c2... c3... c4...>.

        Unmeasured degrees of freedom return None (decoded) or "X"-padded
        strings (undecoded bitstring).

        Arguments:
            - measurements: List of (address, bitstring) tuples.
            - encoder: LatticeStateEncoder for decoding and lattice geometry.
        """
        instance = ParsedLatticeResult._create_partial(encoder)
        lattice_def = encoder.lattice_def

        for addr, bitstring in measurements:
            addr_type = ParsedLatticeResult._classify_address(addr) # Raises ValueError for unknown addr.

            if addr_type == "vertex":
                if encoder.expected_vertex_bit_string_length > 0 and len(bitstring) != encoder.expected_vertex_bit_string_length:
                    raise ValueError(
                        f"Vertex bitstring at {addr} has length {len(bitstring)}, "
                        f"expected {encoder.expected_vertex_bit_string_length}."
                    )
                vertex_addr = tuple(addr)
                instance._bit_strings_vertices[vertex_addr] = bitstring
                instance._decoded_vertices[vertex_addr] = encoder.decode_bit_string_to_vertex_state(bitstring)

            elif addr_type == "link":
                if len(bitstring) != encoder.expected_link_bit_string_length:
                    raise ValueError(
                        f"Link bitstring at {addr} has length {len(bitstring)}, "
                        f"expected {encoder.expected_link_bit_string_length}."
                    )
                link_addr = instance._normalize_link_address(addr)
                instance._bit_strings_links[link_addr] = bitstring
                instance._decoded_links[link_addr] = encoder.decode_bit_string_to_link_state(bitstring)

            elif addr_type == "plaquette":
                # TODO: would be nice to find a way to construct using Plaquette class,
                # but might not be possible without significant refactor since that requires
                # a LatticeData instance (one doesn't exist yet when using the from_partial_measurement
                # factory method).
                bottom_left_vertex = tuple(addr[0])
                e1, e2 = addr[1], addr[2]

                # Decode the full plaquette bitstring.
                decoded_plaq = encoder.decode_bit_string_to_plaquette_state(bitstring)
                vertex_mults, a_links, c_links = decoded_plaq

                # Compute plaquette vertex and link addresses.
                v1 = bottom_left_vertex
                v2 = tuple(lattice_def.add_unit_vector_to_vertex_vector(v1, e1))
                v3 = tuple(lattice_def.add_unit_vector_to_vertex_vector(v2, e2))
                v4 = tuple(lattice_def.add_unit_vector_to_vertex_vector(v1, e2))
                vertex_addrs = [v1, v2, v3, v4]

                active_link_addrs = [
                    (v1, e1),     # l1
                    (v2, e2),     # l2
                    (v4, e1),     # l3
                    (v1, e2),     # l4
                ]

                # Populate vertex data.
                link_len = encoder.expected_link_bit_string_length
                vertex_len = encoder.expected_vertex_bit_string_length
                plaq_bits_idx = 0
                for i, v_addr in enumerate(vertex_addrs):
                    if vertex_len > 0:
                        v_bits = bitstring[plaq_bits_idx:plaq_bits_idx + vertex_len]
                        instance._bit_strings_vertices[v_addr] = v_bits
                        instance._decoded_vertices[v_addr] = vertex_mults[i]
                        plaq_bits_idx += vertex_len

                # Populate active link data.
                for i, l_addr in enumerate(active_link_addrs):
                    normalized = instance._normalize_link_address(l_addr)
                    l_bits = bitstring[plaq_bits_idx:plaq_bits_idx + link_len]
                    instance._bit_strings_links[normalized] = l_bits
                    instance._decoded_links[normalized] = a_links[i]
                    plaq_bits_idx += link_len

                # Populate control link data in canonical ordering.
                # Use compute_control_link_dirs_per_vertex to match the
                # bitstring encoding order (same as control_links_ordered).
                control_link_dirs = Plaquette.compute_control_link_dirs_per_vertex(
                    encoder.lattice_def.dim, (e1, e2), encoder.lattice_def.forder
                )
                for vertex_idx, v_addr in enumerate(vertex_addrs):
                    for link_dir in control_link_dirs[vertex_idx]:
                        c_link_addr = (v_addr, link_dir)
                        normalized = instance._normalize_link_address(c_link_addr)
                        c_bits = bitstring[plaq_bits_idx:plaq_bits_idx + link_len]
                        instance._bit_strings_links[normalized] = c_bits
                        instance._decoded_links[normalized] = encoder.decode_bit_string_to_link_state(c_bits)
                        plaq_bits_idx += link_len

        return instance

    @staticmethod
    def _classify_address(addr) -> str:
        """Classify an address as 'vertex', 'link', or 'plaquette'."""
        if isinstance(addr[0], (list, tuple)):
            if len(addr) == 2:
                return "link"
            elif len(addr) == 3:
                return "plaquette"
        elif all(isinstance(x, int) for x in addr):
            return "vertex"
        raise ValueError(f"Cannot classify address: {addr}")

    def get_link_electric_energy(self, link_address: LinkAddress, unphys_mode: Optional[str] = 'warn') -> Union[float, None]:
        """
        Return the electric Casimir energy for the specified link.

        Returns gt_pattern_iweight_to_casimir(irrep) for the link's irrep,
        or None if the decoded to an unphysical state.

        The optional argument unphys_mode customizes the behavior for unphysical links.
        Options are to emit a warning, raise an error, or silently return.

        If the requested link was unmeasured, a KeyError is raised regardless of
        the value of unphys_mode.

        Arguments:
            - link_address: Address of the link, e.g. ((0,0), 1).
            - unphys_mode: 'warn' will cause a warning to be emitted if the
              requested link is unphysical. 'err' will cause a KeyError
              to be raised. If this argument is omitted or takes on any
              other value, None will be silently returned for unphysical
              or unmeasured links.
        """
        from ymcirc.electric_helper import gt_pattern_iweight_to_casimir # TODO: import in method to avoid circuilar import; kinda nasty and would be nice to avoid

        # Deal with unphysical/unmeasured cases first.
        link_state = self.get_link(link_address)
        if link_state is None:
            link_bit_string  = self.get_link(link_address, get_bit_string=True)
            if 'X' in link_bit_string:
                unmeasured_msg = f'Energy requested for unmeasured link.\nAddress: {link_address}\nMeasurement bit string: {link_bit_string}'
                raise KeyError(unmeasured_msg)
            unphys_msg = f'Energy requested for unphysical link.\nAddress: {link_address}\nMeasurement bit string: {link_bit_string}'
            match unphys_mode:
                case 'err':
                    raise KeyError(unphys_msg)
                case 'warn':
                    warnings.warn(unphys_msg)
                case _:
                    pass
            return None
        
        return gt_pattern_iweight_to_casimir(link_state)

    def get_lattice_electric_energy(self, average_result: bool = False, unphys_mode: Optional[str] = 'warn', skip_unmeasured: bool = True) -> float:
        """
        Return the total (or average) electric Casimir energy across all links.

        Iterates over all link addresses using get_traversal_order and sums
        get_link_electric_energy for each link. If average_result is True,
        divides the total by the number of links.

        The optional argument unphys_mode customizes the behavior when
        encountering unphysical links (i.e. when the measured value fails
        to decode to a physical state). Options are to emit a warning, raise an error,
        or silently return. When not raising an error, such links
        are skipped when computing the sum over link energies.

        The optional argument skip_unmeasured overrides the behavior
        of raising an error when attempting to obtain the energy of
        an unmeasured like. If this argument is True, any such links
        are skipped in the sum over lattice link energies.

        Note that If average_result is True, any skipped links will be
        omitted from the count of links in the denominator of the average.

        Arguments:
            - average_result: If True, return energy per link; if False, total.
            - unphys_mode: 'warn' will cause a warning to be emitted if the
              requested link is unphysical. 'err' will cause a ValueError
              to be raised. If this argument is omitted or takes on any
              other value, None will be silently returned for unphysical
              or unmeasured links.
            - skip_unmeasured: If True, unmeasured links will be skipped when
              computing the lattice electric energy. If False, then a KeyError
              will be raised if an unmeasured link is encountered.
        """
        total_energy = 0.0
        none_count = 0
        link_err_count = 0
        link_count = 0

        for vertex_addr, link_addrs in self.get_traversal_order():
            for link_addr in link_addrs:
                try:
                    energy = self.get_link_electric_energy(link_addr, unphys_mode=unphys_mode)
                    if energy is None:
                        none_count += 1
                    else:
                        total_energy += energy
                        link_count += 1
                except KeyError as e:
                    if skip_unmeasured is False: # KeyError can only happen if the link we just tried to measure was unphysical.
                        raise e
                    match unphys_mode:
                        case 'warn':
                            warnings.warn(str(e))
                            link_err_count += 1
                        case 'err':
                            raise e
                        case _:
                            continue

        if none_count + link_err_count > 0 and unphys_mode == 'warn':
            warnings.warn(
                f"Encountered {none_count} unphysical, {link_err_count} unmeasured, "
                f"(and {link_count} physical link(s) while computing lattice electric energy. "
                f"Unphysical/unmeasured links contributed 0 to the sum."
            )

        if average_result:
            return total_energy / link_count

        return total_energy
