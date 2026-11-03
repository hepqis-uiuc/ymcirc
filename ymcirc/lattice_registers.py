"""Classes for juggling registers in quantum simulations of lattices."""
from __future__ import annotations
import copy
import logging
from collections.abc import Sequence
from qiskit.circuit import QuantumRegister  # type: ignore
from ymcirc.conventions import IrrepBitmap, LatticeStateEncoder, VertexMultiplicityBitmap
from ymcirc._abstract.lattice_data import (
    LatticeData, Plaquette, LatticeVector, LatticeDef,
    LinkAddress, DimensionalitySpecifier
)

# Set up module-specific logger
logger = logging.getLogger(__name__)


class LatticeRegisters(LatticeData[QuantumRegister]):
    """
    Class for addressing QuantumRegisters in a lattice.

    Each vertex and each link that exists on the lattice is assigned
    its own register (on an open axis, there are no links past the boundary;
    for irrep truncations with trivial singlet multiplicities no vertex data
    needs to be tracked). The arguments dimensions, size,
    periodic_boundary_conds and forder are those of LatticeDef.

    Defaults to 1 qubit per link and 0 per vertex.

    If link_bitmap or vertex_bitmap is provided, the corresponding argument
    n_qubits_per_link or n_qubits_per_vertex is ignored, and the number of
    qubits is the length of the bit strings in the bitmap. link_bitmap maps
    the i-weights of the irreps on a link to bit strings; vertex_bitmap maps
    singlet multiplicity indices (nonnegative ints) to bit strings, as
    LatticeStateEncoder's link_bitmap and vertex_bitmap do.

    If working with a LatticeStateEncoder instance, the class method
    LatticeRegisters.from_lattice_state_encoder is provided
    to automatically initialize a LatticeRegisters instance for
    the lattice geometry which is consistent with the LatticeStateEncoder.
    This is the recommended pathway for creating a `LatticeRegisters`
    instance.
    """

    def __init__(
            self, dimensions: DimensionalitySpecifier,
            size: int | tuple[int, ...],
            periodic_boundary_conds: bool | tuple[bool, ...] = True,
            n_qubits_per_link: int = 1,
            n_qubits_per_vertex: int = 0,
            link_bitmap: IrrepBitmap | None = None,
            vertex_bitmap: VertexMultiplicityBitmap | None = None,
            forder: Sequence[int] | None = None):
        """Initialize all registers needed to simulate the lattice."""
        super().__init__(dimensions, size, periodic_boundary_conds, forder=forder)
        # Infer qubit requirements if bit mappings provided,
        # and perform validation.
        if link_bitmap is not None:
            all_bitstring_encodings = list(link_bitmap.values())
            n_qubits_per_link = 0 if len(all_bitstring_encodings) == 0 else len(all_bitstring_encodings[0])  # For an empty bit map, there are no states to encode.
        if vertex_bitmap is not None:
            all_bitstring_encodings = list(vertex_bitmap.values())
            n_qubits_per_vertex = 0 if len(all_bitstring_encodings) == 0 else len(all_bitstring_encodings[0])  # For an empty bit map, there are no states to encode.
        self._validate_qubit_params(n_qubits_per_link, n_qubits_per_vertex)

        # Validate state bitmaps (if given).
        if link_bitmap is not None and len(link_bitmap) > 0:
            self._validate_link_bitmap(link_bitmap)
        if vertex_bitmap is not None and len(vertex_bitmap) > 0:
            self._validate_vertex_bitmap(vertex_bitmap)
        self._link_bitmap = link_bitmap
        self._vertex_bitmap = vertex_bitmap

        # Declare the actual QuantumRegister instances for lattice DoFs.
        self._initialize_qubit_registers(n_qubits_per_link, n_qubits_per_vertex)

        logger.info(f"Created {self}. Lattice has {self.n_total_qubits} qubits, {self.n_plaquettes} plaquettes, {self.n_links} links, and {self.n_vertices} vertices.")

    def __repr__(self):
        class_name = type(self).__name__
        return f"{class_name}(dimensions={self.dim}, size={self.size}, periodic_boundary_conds={self._periodic_boundary_conds}, n_qubits_per_link={self.n_qubits_per_link}, n_qubits_per_vertex={self._n_qubits_per_vertex}, link_bitmap={self.link_bitmap}, vertex_bitmap={self.vertex_bitmap})"

    def __str__(self):
        lattice_geometry_str = LatticeDef(dimensions=self.dim, size=self.size, periodic_boundary_conds=self.periodic_boundary_conds).__str__()
        return f"Quantum registers indexable on a lattice with the geometry: {lattice_geometry_str}"        

    def _validate_qubit_params(self, n_qubits_per_link: int = 1, n_qubits_per_vertex: int = 1):
        if n_qubits_per_vertex < 0:
            raise ValueError("Vertex registers must have nonnegative integer number of qubits. "
                             f"n_qubits_per_vertex = {n_qubits_per_vertex}.")

        if n_qubits_per_link < 1:
            raise ValueError("Link registers must have positive integer number of qubits. "
                             f"n_qubits_per_link = {n_qubits_per_link}.")

    def _validate_link_bitmap(self, candidate_dict: IrrepBitmap):
        # Conveniences.
        all_link_bitstrings = list(candidate_dict.values())
        all_link_iweights = list(candidate_dict.keys())
        bit_length = len(all_link_bitstrings[0])

        # Boolean test results.
        bit_lengths_differ = any(len(bit_string) != bit_length for bit_string in candidate_dict.values())
        all_values_are_strings = all(isinstance(bitstring, str) for bitstring in all_link_bitstrings)
        all_keys_are_len_3_tuples = all(isinstance(iweight, tuple) and len(iweight) == 3 for iweight in all_link_iweights)

        # Actual checks.
        if not all_values_are_strings or not all_keys_are_len_3_tuples:
            raise TypeError(f"Expected a dict with keys that are length-three tuples, and values that are strings. Encountered:\n{candidate_dict}")
        if bit_lengths_differ:
            raise ValueError(f"The values of candidate_dict must all have the same bit length. Dict values encountered:\n{list(all_link_bitstrings)}")

    def _validate_vertex_bitmap(self, candidate_dict: VertexMultiplicityBitmap):
        # Conveniences.
        all_vertex_bitstrings = list(candidate_dict.values())
        all_vertex_multiplicities = list(candidate_dict.keys())
        bit_length = len(all_vertex_bitstrings[0])

        # Boolean test results.
        bit_lengths_differ = any(len(bit_string) != bit_length for bit_string in candidate_dict.values())
        all_values_are_strings = all(isinstance(bitstring, str) for bitstring in all_vertex_bitstrings)
        all_keys_are_nonnegative_ints = all(isinstance(key, int) and key >= 0 for key in all_vertex_multiplicities)

        if not all_values_are_strings or not all_keys_are_nonnegative_ints:
            raise TypeError(f"Expected a dict with nonnegative integer keys and bitstring values. Encountered:\n{candidate_dict}")
        if bit_lengths_differ:
            raise ValueError(f"The values of the bitmap must all have the same bit length. Dict values encountered:\n{list(all_vertex_bitstrings)}")

    def _initialize_qubit_registers(self, n_qubits_per_link: int, n_qubits_per_vertex: int):
        self._n_qubits_per_link = n_qubits_per_link
        self._n_qubits_per_vertex = n_qubits_per_vertex
        self._vertex_registers: dict[LatticeVector, QuantumRegister] = {}
        self._link_registers: dict[LinkAddress, QuantumRegister] = {}
        for vertex_vector in self.vertex_addresses:
            self._vertex_registers[vertex_vector] = QuantumRegister(self._n_qubits_per_vertex, name=f"v:{vertex_vector}")
        for link_address in self.link_addresses:
            self._link_registers[link_address] = QuantumRegister(self._n_qubits_per_link, name=f"l:{link_address}")

    def get_vertex(self, lattice_vector: LatticeVector) -> QuantumRegister:
        """Return the QuantumRegister for the vertex specified by lattice_vector."""
        return self._vertex_registers[self._wrap_periodic_axes(lattice_vector)]

    def get_link(self, link_address: LinkAddress) -> QuantumRegister:
        """
        Return the QuantumRegister for the link specified by link_address.

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
        """
        normalized_link_address = self._normalize_link_address(link_address)
        return self._link_registers[normalized_link_address]

    def get_registers_in_local_hamiltonian_order(
            self,
            lattice_vector: LatticeVector,
            e1: int,
            e2: int
    ) -> list[QuantumRegister]:
        """
        Return the link and vertex registers from a plaquette in a list ordered according to the local Hamiltonian.

        Plaquette local basis states are assumed to take the form:

        |v1 v2 v3 v4 l1 l2 l3 l4 c1 .... c2 .... c3 .... c4 ....>

        according to the layout:

        v4 ----l3--- v3
        |            |
        |            |
        l4           l2
        |            |
        |            |
        v1 ----l1--- v2

        where the state label ends with the controls associated with each vertex,
        in F-order at each vertex.

        Every control link is listed, including a control link that repeats
        another control of the plaquette (on a periodic axis of length 2).
        A plaquette's bit string, and the plaquette circuits that
        LatticeCircuitManager builds, leave such repeats out, so on those
        lattices the list is longer than the plaquette's bit string.
        """
        p = Plaquette(lattice=self, bottom_left_vertex=lattice_vector, plane=(e1, e2))
        return [*p.vertices, *p.active_links, *p.control_links_ordered]

    @property
    def n_qubits_per_link(self) -> int:
        """Number of qubits per link register."""
        return self._n_qubits_per_link

    @property
    def n_qubits_per_vertex(self) -> int:
        """Number of qubits per vertex register."""
        return self._n_qubits_per_vertex

    @property
    def n_total_qubits(self) -> int:
        """Number of qubits in entire lattice."""
        return len(self._vertex_registers)*self.n_qubits_per_vertex \
            + len(self._link_registers)*self.n_qubits_per_link

    @property
    def link_bitmap(self) -> IrrepBitmap | None:
        """
        Return a copy of the link truncation dictionary to bitstrings.

        Returns None if not defined.
        """
        return copy.deepcopy(self._link_bitmap)

    @property
    def vertex_bitmap(self) -> VertexMultiplicityBitmap | None:
        """
        Return a copy of the vertex singlet dictionary map to bitstrings.

        Returns None if not defined.
        """

        return copy.deepcopy(self._vertex_bitmap)

    @classmethod
    def from_lattice_state_encoder(cls, lattice_encoder: LatticeStateEncoder) -> LatticeRegisters:
        """Initialize from a LatticeStateEncoder instance."""
        lattice_def: LatticeDef = lattice_encoder.lattice_def
        return LatticeRegisters(
            dimensions=lattice_def.dim,
            size=lattice_def.size,
            periodic_boundary_conds=lattice_def.periodic_boundary_conds,
            link_bitmap=lattice_encoder.link_bitmap,
            vertex_bitmap=lattice_encoder.vertex_bitmap,
            forder=lattice_def.forder
        )
