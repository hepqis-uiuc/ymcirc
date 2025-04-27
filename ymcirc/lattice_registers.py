"""Classes for juggling registers in quantum simulations of lattices."""
from __future__ import annotations
import copy
from math import ceil
from qiskit.circuit import QuantumRegister  # type: ignore
from typing import List, Dict, Union
from ymcirc.conventions import IrrepBitmap, VertexMultiplicityBitmap
from ymcirc._abstract.lattice_data import (
    LatticeData, Plaquette, LatticeVector, VertexAddress,
    LinkUnitVectorLabel, LinkAddress, DimensionalitySpecifier,
    VERTICAL_DIR_LABEL, VERTICAL_NUM_VERTICES_D_THREE_HALVES
)


class LatticeRegisters(LatticeData[QuantumRegister]):
    """
    Class for addressing QuantumRegisters in a lattice.

    Each link and vertex is assigned a unique register. The size tuple counts
    the number of vertex registers in each dimension of a (hyper)-rectangular
    lattice.

    Defaults to 1 qubit per link and 0 per vertex.

    If bitstring maps
    link_truncation_dict and/or vertex_singlet_dict are provided,
    the corresponding argument n_qubits_per_link and n_qubits_per_vertex
    are ignored, and qubit requirements are inferred based on the length
    of the bitstrings in the corresponding dict(s). In this case,
    the number of links per vertex implied by vertex_singlet_dict
    must match the dimensionality of the lattice. In particular,
    the number of iWeights in the keys must be 2*dim for dim >= 2,
    or 3 for dim 1.5. For example, a key for dim 1.5 should look like:

    (((1, 1, 0), (1, 1, 0), (0, 0, 0)), 1)

    This key is a length-two tuple whose first element is a tuple of three
    GT pattern i-weights, and whose second element is an integer indexing
    multiplicity.
    """

    def __init__(
            self, dimensions: DimensionalitySpecifier,
            size: int | tuple[int, ...],
            periodic_boundary_conds: bool | tuple[bool, ...] = True,
            n_qubits_per_link: int = 1,
            n_qubits_per_vertex: int = 0,
            link_truncation_dict: Union[IrrepBitmap, None] = None,
            vertex_singlet_dict: Union[VertexMultiplicityBitmap, None] = None):
        """Initialize all registers needed to simulate the lattice."""
        super().__init__(dimensions, size, periodic_boundary_conds)
        # Infer qubit requirements if bit mappings provided,
        # and perform validation.
        if link_truncation_dict is not None:
            all_bitstring_encodings = list(link_truncation_dict.values())
            n_qubits_per_link = 0 if len(all_bitstring_encodings) == 0 else len(all_bitstring_encodings[0])  # For an empty bit map, there are no states to encode.
        if vertex_singlet_dict is not None:
            all_bitstring_encodings = list(vertex_singlet_dict.values())
            n_qubits_per_vertex = 0 if len(all_bitstring_encodings) == 0 else len(all_bitstring_encodings[0])  # For an empty bit map, there are no states to encode.
        self._validate_qubit_params(n_qubits_per_link, n_qubits_per_vertex)

        # Validate state bitmaps (if given).
        if link_truncation_dict is not None and len(link_truncation_dict) > 0:
            self._validate_link_truncation_dict(link_truncation_dict)
        if vertex_singlet_dict is not None and len(vertex_singlet_dict) > 0:
            self._validate_vertex_singlet_truncation_dict(vertex_singlet_dict)
        self._link_truncation_dict = link_truncation_dict
        self._vertex_singlet_dict = vertex_singlet_dict

        # Declare the actual QuantumRegister instances for lattice DoFs.
        self._initialize_qubit_registers(n_qubits_per_link, n_qubits_per_vertex)
        
    def _validate_qubit_params(self, n_qubits_per_link: int = 1, n_qubits_per_vertex: int = 1):
        if n_qubits_per_vertex < 0:
            raise ValueError("Vertex registers must have nonnegative integer number of qubits. "
                             f"n_qubits_per_vertex = {n_qubits_per_vertex}.")

        if n_qubits_per_link < 1:
            raise ValueError("Link registers must have positive integer number of qubits. "
                             f"n_qubits_per_link = {n_qubits_per_link}.")

    def _validate_link_truncation_dict(self, candidate_dict: IrrepBitmap):
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

    def _validate_vertex_singlet_truncation_dict(self, candidate_dict: VertexMultiplicityBitmap):
        # Conveniences.
        all_vertex_singlet_bitstrings = list(candidate_dict.values())
        all_vertex_singlet_bag_states = list(candidate_dict.keys())
        bit_length = len(all_vertex_singlet_bitstrings[0])
        n_links_per_vertex = ceil(self.dim) * 2 if self.dim != 1.5 else 3
        iweight_len_SU3 = 3

        # Boolean test results.
        bit_lengths_differ = any(len(bit_string) != bit_length for bit_string in candidate_dict.values())
        all_values_are_strings = all(isinstance(bitstring, str) for bitstring in all_vertex_singlet_bitstrings)
        all_keys_are_tuples_of_su3_iweights_and_int = all(
            len(bag) == 2 and isinstance(bag[0], tuple) and len(bag[0]) == n_links_per_vertex and (
                all(isinstance(iweight, tuple) and len(iweight) == iweight_len_SU3 for iweight in bag[0]))
            and isinstance(bag[1], int) for bag in all_vertex_singlet_bag_states)

        if not all_values_are_strings or not all_keys_are_tuples_of_su3_iweights_and_int:
            raise TypeError(f"Expected a dict with keys that are length-two tuples whose first element are themselves length-3 tuples (i-Weights), and whose second elements are integers interpreted as indexing multiplicity. Encountered:\n{candidate_dict}")
        if bit_lengths_differ:
            raise ValueError(f"The values of vertex_singlet_dict must all have the same bit length. Dict values encountered:\n{list(all_vertex_singlet_bitstrings)}")

    def _initialize_qubit_registers(self, n_qubits_per_link: int, n_qubits_per_vertex: int):
        self._n_qubits_per_link = n_qubits_per_link
        self._n_qubits_per_vertex = n_qubits_per_vertex
        self._vertex_registers: Dict[LatticeVector, QuantumRegister] = {}
        self._link_registers: Dict[LinkAddress, QuantumRegister] = {}
        for vertex_vector in self.vertex_addresses:
            self._vertex_registers[vertex_vector] = QuantumRegister(self._n_qubits_per_vertex, name=f"v:{vertex_vector}")
        for link_address in self.link_addresses:
            self._link_registers[link_address] = QuantumRegister(self._n_qubits_per_link, name=f"l:{link_address}")

    def get_vertex(self, lattice_vector: LatticeVector) -> QuantumRegister:
        """Return the QuantumRegister for the vertex specified by lattice_vector."""
        if self.all_boundary_conds_periodic:
            if self.dim != 1.5:
                lattice_vector = tuple(component % self.shape[0] for component in lattice_vector)
            else:  # Don't do anything to the vertical direction in d=3/2 since that direction is NEVER periodic!
                lattice_vector = (lattice_vector[0] % self.shape[0], ) + lattice_vector[1:]
        else:
            raise NotImplementedError()

        return self._vertex_registers[lattice_vector]

    #def get_link(self, lattice_vector: LatticeVector, unit_vector_label: LinkUnitVectorLabel) -> QuantumRegister:
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

    # TODO this needs to be tested
    def get_registers_in_local_hamiltonian_order(
            self,
            lattice_vector: LatticeVector,
            e1: LinkUnitVectorLabel,
            e2: LinkUnitVectorLabel
    ) -> List[QuantumRegister]:
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

        where the state label ends with the controls associated with each vertex.
        """
        p = Plaquette(lattice=self, bottom_left_vertex=lattice_vector, plane=(e1, e2))
        plaquette_register_list = list(p.vertices) + list(p.active_links)
        for vertex in p.control_links.keys():
            for link_address in p.control_links[vertex]:
                plaquette_register_list.append(p.control_links[vertex][link_address])
        return plaquette_register_list

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
    def link_truncation_bitmap(self) -> Union[IrrepBitmap, None]:
        """
        Return a copy of the link truncation dictionary to bitstrings, if defined.
        """
        return copy.deepcopy(self._link_truncation_dict)

    @property
    def vertex_singlet_bitmap(self) -> Union[VertexMultiplicityBitmap, None]:
        """
        Return a copy of the vertex singlet dictionary map to bitstrings, if defined.
        """
        return copy.deepcopy(self._vertex_singlet_dict)
