"""
A collection of utilities for building circuits.
"""
from __future__ import annotations
import copy
import logging
from pathlib import Path
from ymcirc.conventions import (PlaquetteState, LatticeStateEncoder, ONE, THREE, THREE_BAR, MatrixElementValue, HamiltonianData)
from ymcirc.lattice_registers import LatticeRegisters
from ymcirc.givens import (
    givens,
    LPFamily,
    bitstring_value_of_LP_family,
    givens_fused_controls,
    compute_LP_family,
    compute_p_tilde,
    gray_to_index,
)
from ymcirc._abstract.lattice_data import Plaquette, LinkUnitVectorLabel, LinkAddress, LatticeVector, Plane, Signature
from ymcirc.utilities import _check_circuits_logically_equivalent, _flatten_circuit, eta_update, fmt_td
from math import ceil
from qiskit import transpile
from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister
from qiskit.circuit.library.standard_gates import RXGate, CXGate
from qiskit import qasm3, qpy
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import InverseCancellation
from typing import List, Tuple, Set, Union, Dict
import numpy as np

# Set up module-specific logger
logger = logging.getLogger(__name__)

# Circuit construction needs to generically iterate over planes.
# Since it's more efficient to store matrix element data on-disk
# in a formate where the top level key is a pair of plaquette states,
# we re-index when doing circuit construction.
# A dict with key hierarchy: plane -> (bs1, bs2) -> signature -> float.
# Bitstring keys are already trimmed on small periodic lattices.
PlaneKeyedHamiltonianData = Dict[Plane, Dict[Tuple[str, str], Dict[Signature, float]]]


class LatticeCircuitManager:
    """Class for creating quantum simulation circuits from LatticeRegister instances."""

    def __init__(
        self, lattice_encoder: LatticeStateEncoder, mag_hamiltonian: HamiltonianData
    ):
        """
        Create via a LatticeStateEncoder instance and magnetic Hamiltonian matrix elements.

        If the lattice defined by lattice_encoder is small and periodic, then the data in mag_hamiltonian
        will be filtered for consistency. "Small and periodic" means that the lattice has
        periodic boundary conditions, and is small enough that it is possible for a single
        physical lattice link to appear as control links on two different vertices in a given
        plaquette. In this situation, all matrix elements in the magnetic Hamiltonian
        for which the initial or the final state have distinct link state data on one of
        these "shared" control links are discarded, and "duplicate" control links
        are removed from the binary encoding of the plaquette states. This duplicate
        removal is done by removing all but the first instance of a repeated control
        link from the plaquette state.
        """
        # Copies to avoid inadvertently changing the behavior of the
        # LatticeCircuitManager instance.
        self._encoder = copy.deepcopy(lattice_encoder)
        _input_hamiltonian: HamiltonianData = copy.deepcopy(mag_hamiltonian)
        self._cached_mag_evol_circuits: Dict[Tuple[Plane, Signature], QuantumCircuit] = {}
        self._cached_mag_evol_params = {
            "physical_states_for_control_pruning": None,
            "optimize_circuits": None,
            "control_fusion": None,
        }
        # Parameter for the number of ancillas used in circuit. Initialized to 0.
        self._num_ancillas = 0

        # Determine if lattice is small and periodic. If yes, filter out inconsistent Hamiltonian terms
        # and drop repeated references to the same physical control link in mag_hamiltonian bit strings.
        self._lattice_is_small = False
        self._lattice_is_periodic = False
        lattice_size_threshold_for_smallness = 2
        if lattice_encoder.lattice_def.all_boundary_conds_periodic:
            self._lattice_is_periodic = True
        match lattice_encoder.lattice_def.dim:
            case 1.5:
                lattice_size = lattice_encoder.lattice_def.shape[0]
            case 2:
                lattice_size = lattice_encoder.lattice_def.shape[0]
                if lattice_size != lattice_encoder.lattice_def.shape[1]:
                    raise NotImplementedError("Non-square dim 2 lattices not yet supported.")
            case 3:
                lattice_size = lattice_encoder.lattice_def.shape[0]
                if lattice_size != lattice_encoder.lattice_def.shape[1] \
                        or lattice_size != lattice_encoder.lattice_def.shape[2]:
                    raise NotImplementedError("Non-cubic dim 3 lattices not yet supported.")
            case _:
                raise NotImplementedError(f"Dim {lattice_encoder.lattice_def.dim} lattice not yet supported.")
        self._lattice_is_small = True if lattice_size <= lattice_size_threshold_for_smallness else False

        # Cache control link dirs per plane for small periodic lattices (used by consistency/discard methods).
        self._cached_ctrl_dirs_small_and_periodic: Dict[Plane, tuple] = {}
        if self._lattice_is_small and self._lattice_is_periodic:
            _temp_lattice = LatticeRegisters.from_lattice_state_encoder(self._encoder)
            dim = self._encoder.lattice_def.dim
            if dim == 1.5 or dim == 2:
                planes = [(1, 2)]
            elif dim == 3:
                planes = [(1, 2), (1, 3), (2, 3)]
            else:
                raise NotImplementedError(f"Dim {dim} lattice not yet supported.")
            origin = tuple(0 for _ in range(len(self._encoder.lattice_def.shape)))
            for e1, e2 in planes:
                _temp_plaq = _temp_lattice.get_plaquettes(origin, e1, e2)
                self._cached_ctrl_dirs_small_and_periodic[(e1, e2)] = _temp_plaq.control_link_dirs_per_vertex

        # Pivot self._mag_hamiltonian from entry-first to per-plane-first format.
        # On small periodic lattices, also filter inconsistent entries and trim
        # duplicate control links per-plane. A given (bs1, bs2) entry may be
        # consistent for one plane but not another, so filtering is per-plane.
        plane_keyed_hamiltonian_data: PlaneKeyedHamiltonianData = {}
        if self._lattice_is_small is True and self._lattice_is_periodic is True:
            for (final_bs, initial_bs), matrix_elem_value in _input_hamiltonian.items():
                final_ps = lattice_encoder.decode_bit_string_to_plaquette_state(final_bs)
                initial_ps = lattice_encoder.decode_bit_string_to_plaquette_state(initial_bs)

                for plane, sig_dict in matrix_elem_value.items():
                    if (self._plaquette_state_has_inconsistent_controls(final_ps, plane)
                            or self._plaquette_state_has_inconsistent_controls(initial_ps, plane)):
                        continue

                    final_trimmed = self._discard_duplicate_controls_from_plaquette_state(final_ps, plane)
                    initial_trimmed = self._discard_duplicate_controls_from_plaquette_state(initial_ps, plane)
                    trimmed_key = (
                        lattice_encoder.encode_plaquette_state_as_bit_string(
                            final_trimmed, override_n_c_links_validation=True),
                        lattice_encoder.encode_plaquette_state_as_bit_string(
                            initial_trimmed, override_n_c_links_validation=True),
                    )

                    if plane not in plane_keyed_hamiltonian_data:
                        plane_keyed_hamiltonian_data[plane] = {}
                    if trimmed_key in plane_keyed_hamiltonian_data[plane]:
                        plane_keyed_hamiltonian_data[plane][trimmed_key].update(sig_dict)
                    else:
                        plane_keyed_hamiltonian_data[plane][trimmed_key] = dict(sig_dict)
        else:
            # Large or non-periodic: pivot to per-plane-first without filtering/trimming.
            for (bs1, bs2), matrix_elem_value in _input_hamiltonian.items():
                for plane, sig_dict in matrix_elem_value.items():
                    if plane not in plane_keyed_hamiltonian_data:
                        plane_keyed_hamiltonian_data[plane] = {}
                    plane_keyed_hamiltonian_data[plane][(bs1, bs2)] = dict(sig_dict)
        self._mag_hamiltonian = plane_keyed_hamiltonian_data

    def __repr__(self):
        class_name = type(self).__name__
        n_planes = len(self._mag_hamiltonian)
        n_entries = sum(len(v) for v in self._mag_hamiltonian.values())
        return (f"{class_name}({self._encoder.__repr__()}, "
                f"<{n_entries} Hamiltonian entries across {n_planes} plane(s)>)")

    def __str__(self):
        class_name = type(self).__name__
        return f"Circuit manager for lattices of type {self._encoder.lattice_def}.\nAncillas: {self.num_ancillas}\nLink bitmap:{self._encoder.link_bitmap}\nVertex bitmap: {self._encoder.vertex_bitmap}"

    @staticmethod
    def _count_parameters_in_circ(circ: QuantumCircuit, name_prefixes: list[str], separator: str = ',') -> list[int]:
        logger.debug(f"Counting distinct parameters in circuit with the following name prefixes: {name_prefixes}")
        n_params_list = [0,] * len(name_prefixes)
        if len(circ.parameters) > 0:
            for param in circ.parameters:
                param_prefix, param_num = param.name.split(separator)
                for idx, name_prefix in enumerate(name_prefixes):
                    logger.debug(f"{idx}: {param_prefix} == {name_prefix}?")
                    if param_prefix == name_prefix:
                        logger.debug("Yes!")
                        n_params_list[idx] += 1

        logger.debug(f"Final counts: {n_params_list}")
        return n_params_list


    def create_blank_full_lattice_circuit(
        self, lattice: LatticeRegisters
    ) -> QuantumCircuit:
        """
        Return a blank quantum circuit with all link and vertex registers in lattice.

        This uses an ordering which is specified in the LatticeData class which is
        a parent class of LatticeRegisters. See the documentation on LatticeData
        for details.
        """
        all_lattice_registers: List[QuantumRegister] = [reg for reg in lattice]

        return QuantumCircuit(*all_lattice_registers)

    def compute_num_ancillas_needed_from_mag_trotter_step(
            self,
            master_circuit: QuantumCircuit,
            lattice: LatticeRegisters,
            control_fusion: bool = False,
            physical_states_for_control_pruning: Union[None | Set[str]] = None,
            optimize_circuits: bool = False) -> int:
        """
        Computes the number of ancilla qubits needed to perform v-chain synthesis in the master_circuit
        for MCU decomposition. Function constructs the circuit for asingle magnetic trotter step to find the minimum
        required ancillas needed.

        For MCX decomposition into v-chain, (maximum number of controls in a trotter step - 2)
        is the result (check internal givens rotation function for reference).

        Arguments:
          - master_circuit: a QuantumCircuit instance which is built from all
                            the QuantumRegister instances in lattice.
          - lattice: a LatticeRegisters instance which keeps track of all the
                     QuantumRegisters.
          - control_fusion: Optional boolian argument with the default set to False. If it's set
                            to be True, then LP families of givens rotations are first Gray code ordered,
                            then redundant controls are removed.
          - physical_states_for_control_pruning: The set of all physical states encoded as bitstrings.
                                                 If provided, control pruning of multi-control rotation
                                                 gate inside Givens rotation subcircuits will be attempted.
                                                 If the lattice is small and periodic, then duplicate control
                                                 links which are shared between vertices will be stripped
                                                 out first.
                                                 If None, no control pruning is attempted.
          - optimize_circuits: if True, run the qiskit transpiler on each
                               internal givens rotation with the maximum
                               optimization level before composing with
                               master_circuit.

        Returns:
          int
        """
        max_controls = 0

        # Copy the input circuit to avoid mutating it.
        circ_for_n_ancillas_check = copy.deepcopy(master_circuit)

        logger.info("Computing but not caching a magnetic Trotter step to determine num ancillas needed.")
        self.apply_magnetic_trotter_step(circ_for_n_ancillas_check, lattice,
                                         optimize_circuits=optimize_circuits,
                                         physical_states_for_control_pruning=physical_states_for_control_pruning, 
                                         control_fusion=control_fusion, cache_mag_evol_circuit=False)

        for circuit_instruction in circ_for_n_ancillas_check.data:
            if len(circuit_instruction.operation.name) >= 3 and circuit_instruction.operation.name[:3] == "mcx":
                max_controls = max(circuit_instruction.operation.num_ctrl_qubits, max_controls)

        max_ancillas_needed = max_controls - 2

        return max_ancillas_needed

    @property
    def num_ancillas(self) -> int:
        """The size of the ancilla register LatticeCircuitManager expects."""
        return self._num_ancillas

    @num_ancillas.setter
    def num_ancillas(self, n) -> None:
        """Set the size of the ancilla register LatticeCircuitManager expects."""
        # TODO test these validation checks.
        if not isinstance(n, int):
            raise TypeError(f"Number of ancillas {n} is not an integer.")
        if n < 0:
            raise ValueError(f"Number of ancillas {n} is not nonnegative.")

        self._num_ancillas = n

    def add_ancilla_register_to_quantum_circuit(self, master_circuit: QuantumCircuit) -> None:
        """
        Adds ancilla qubits to the master_circuit based on the value of self.num_ancillas.

        Note that this mutates the circuit!
        """
        master_circuit.add_register(AncillaRegister(self.num_ancillas, "anc"))

    def measure_link(
        self,
        circuit: QuantumCircuit,
        lattice: LatticeRegisters,
        link_address: LinkAddress,
    ) -> None:
        """
        Append a measurement of the specified link register to the circuit.

        Adds a ClassicalRegister sized to match the link's QuantumRegister,
        then appends measure gates for each qubit in the link.

        Arguments:
            - circuit: The QuantumCircuit to append measurements to.
            - lattice: LatticeRegisters instance for qubit lookup.
            - link_address: Address of the link to measure, e.g. ((0,0), 1).
        """
        qreg = lattice.get_link(link_address)
        creg = ClassicalRegister(len(qreg), name=f"meas_{qreg.name}")
        circuit.add_register(creg)
        circuit.measure(qreg, creg)

    def measure_vertex(
        self,
        circuit: QuantumCircuit,
        lattice: LatticeRegisters,
        vertex_address: LatticeVector,
    ) -> None:
        """
        Append a measurement of the specified vertex register to the circuit.

        If the vertex register has 0 qubits (e.g., d=3/2 T1 where vertex
        multiplicities are trivial), this is a no-op.

        Arguments:
            - circuit: The QuantumCircuit to append measurements to.
            - lattice: LatticeRegisters instance for qubit lookup.
            - vertex_address: Lattice vector of the vertex to measure, e.g. (0, 0).
        """
        qreg = lattice.get_vertex(vertex_address)
        if len(qreg) == 0:
            return
        creg = ClassicalRegister(len(qreg), name=f"meas_{qreg.name}")
        circuit.add_register(creg)
        circuit.measure(qreg, creg)

    def measure_plaquette(
        self,
        circuit: QuantumCircuit,
        lattice: LatticeRegisters,
        bottom_left_vertex: LatticeVector,
        e1: LinkUnitVectorLabel,
        e2: LinkUnitVectorLabel,
    ) -> None:
        """
        Append measurements for all registers in the specified plaquette.

        Measures all vertex registers, active link registers, and control link
        registers belonging to the plaquette defined by bottom_left_vertex and
        the plane (e1, e2). Deduplicates registers that appear multiple times
        (e.g., shared control links on small periodic lattices).

        Arguments:
            - circuit: The QuantumCircuit to append measurements to.
            - lattice: LatticeRegisters instance for qubit lookup.
            - bottom_left_vertex: Lattice vector of the plaquette's v1 vertex.
            - e1: First lattice direction defining the plaquette plane.
            - e2: Second lattice direction defining the plaquette plane.
        """
        plaquette = lattice.get_plaquettes(bottom_left_vertex, e1, e2)

        # Collect all unique registers to measure.
        seen_names = set()
        regs_to_measure = []

        for reg in plaquette.vertices:
            if len(reg) > 0 and reg.name not in seen_names:
                regs_to_measure.append(reg)
                seen_names.add(reg.name)

        for reg in plaquette.active_links:
            if reg.name not in seen_names:
                regs_to_measure.append(reg)
                seen_names.add(reg.name)

        for reg in plaquette.control_links_ordered:
            if reg.name not in seen_names:
                regs_to_measure.append(reg)
                seen_names.add(reg.name)

        for reg in regs_to_measure:
            creg = ClassicalRegister(len(reg), name=f"meas_{reg.name}")
            circuit.add_register(creg)
            circuit.measure(reg, creg)

    def apply_electric_trotter_step(
        self,
        master_circuit: QuantumCircuit,
        lattice: LatticeRegisters,
        hamiltonian: list[float],
        electric_gray_order: bool = False
    ) -> None:
        """
        Perform an electric Trotter step.

        Appends a circuit with the following new Parameter instances:
            - 'dt_ee__n' (the size of the Trotter time step)
            - 'coupling_g_ee__n' (strong coupling constant value)
        where 'n' is the number of dt_ee and coupling_g_ee parameters that
        already exist in master_circuit. If 'n' is different for the dt
        and coupling_g parameters, an error is raised.


        Implementation uses CX and Zs to implement rotations of Z, I Paulis.
        The single link electric Trotter step is constructed through Z, I
        rotations (e.g. e^(i*coeff*IZZZI))). Such rotatons can be constructed
        through parity circuits (see section 4.2 of arXiv:1001.3855).

        Arguments:
            - master_circuit: A QuantumCircuit instance which is built from all
                              the QuantumRegister instances in lattice.
            - lattice: A LatticeRegisters instance which keeps track of all the
                       QuantumRegisters.
            - hamiltonian: Pauli decompositon of the single link electric
                          Hamiltonian. The list hamiltonian is contains
                          coefficients s.t. for hamiltonian[i] = coeff, coeff
                          is coeff of bitstring(i) with 'Z'=1 and 'I'=0 in the
                          bitstring.
            - electric_gray_order: The Pauli bitstrings corresponding to the Pauli
                                    decomposition of the electric hamiltonian will 
                                    be gray-code ordered if this option is set to be
                                    True. This option is False by default.


        Returns:
            None (master_circuit has the electric Trotter step appended)
        """
        # Construct the dt and coupling parameters for the current electric Trotter step.
        name_prefixes = ['dt_ee', 'coupling_g_ee']
        step_num_separator = '__'
        n_dt_ee_params, n_coupling_g_ee_params = LatticeCircuitManager._count_parameters_in_circ(master_circuit, name_prefixes, separator=step_num_separator)
        if n_dt_ee_params != n_coupling_g_ee_params:
            raise NotImplementedError("Different number of electric dt and electric coupling_g Parameters encountered.")
        dt_ee_current = Parameter(f'dt_ee{step_num_separator}{n_dt_ee_params}')
        coupling_g_ee_current = Parameter(f'coupling_g_ee{step_num_separator}{n_coupling_g_ee_params}')

        N = int(np.log2(len(hamiltonian)))
        angle_mod = ((coupling_g_ee_current * coupling_g_ee_current) / 2) * dt_ee_current
        local_circuit = QuantumCircuit(N)

        # Use the index of the local Pauli-decomposed electric hamiltonian to generate the Pauli bitstrings.
        pauli_bitstring_list = [str("{0:0" + str(N) + "b}").format(i) for i in range(len(hamiltonian))]
        pauli_decomposed_hamiltonian = zip(pauli_bitstring_list,hamiltonian)
        # Gray-Order the Pauli-bitstrings if electric_gray_order == True.
        if electric_gray_order is True:
            pauli_decomposed_hamiltonian = sorted(pauli_decomposed_hamiltonian,key=lambda x: gray_to_index(x[0]))

        # The parity circuit primitive of CXs and Zs.
        for pauli_bitstring, coeff in pauli_decomposed_hamiltonian:
            locs = [
                loc
                for loc, bit in enumerate(pauli_bitstring)
                if bit == "1"
            ]
            for j in locs[1:]:
                local_circuit.cx(j, locs[0])
            if len(locs) != 0:
                local_circuit.rz(2 * angle_mod * coeff, locs[0])
            for j in locs[1:]:
                local_circuit.cx(j, locs[0])

        cancel_cx = PassManager([InverseCancellation([CXGate()])])
        local_circuit = cancel_cx.run(local_circuit)

        # Loop over links for electric Hamiltonian
        for link_address in lattice.link_addresses:
            link_qubits = [
                qubit for qubit in lattice.get_link((link_address[0], link_address[1]))
            ]
            master_circuit.compose(local_circuit, qubits=link_qubits, inplace=True)

    def apply_magnetic_trotter_step(
        self,
        master_circuit: QuantumCircuit,
        lattice: LatticeRegisters,
        optimize_circuits: bool = True,
        physical_states_for_control_pruning: Union[None | Set[str]] = None,
        control_fusion: bool = False,
        cache_mag_evol_circuit: bool = False,
        givens_have_independent_params: bool = False
    ) -> None:
        """
        Add one magnetic Trotter step to the entire lattice circuit.

        Appends a circuit with the following new Parameter instances
        (if givens_have_independent_params is False):
            - 'dt_mag__n' (the size of the Trotter time step)
            - 'coupling_g_mag__n' (strong coupling constant value)
        where 'n' is the number of dt_mag and coupling_g_mag parameters that
        already exist in master_circuit. If 'n' is different for the dt
        and coupling_g parameters, an error is raised.

        If givens_have_independent_params is True, then for each plaquette,
        each Givens rotation will take the single angle parameter theta[m]
        where m is zero-indexed and ranges over all the Givens rotations
        present in the mag_hamiltonian argument used to create the
        LatticeCircuitManager instance.

        Implementation is performed by iterating over every lattice vertex. At each vertex,
        there's an additional iteration over every "positive" plaquette.
        For each such plaquette, the plaquette-local magnetic Trotter step
        is appended to the circuit. This local Trotter step circuit is
        composed of Givens rotations constructed from all Hamiltonian
        matrix elements which match the current plaquette's plane and
        F-order signature.

        Note that this modifies master_circuit directly rather than returning
        a new circuit!

        Arguments:
          - master_circuit: a QuantumCircuit instance which is built from all
                            the QuantumRegister instances in lattice.
          - lattice: a LatticeRegisters instance which keeps track of all the
                     QuantumRegisters.
          - optimize_circuits: if True, run the qiskit transpiler on each
                               internal givens rotation with the maximum
                               optimization level before composing with
                               master_circuit.
          - physical_states_for_control_pruning: The set of all physical states encoded as bitstrings.
                                                 If provided, control pruning of multi-control rotation
                                                 gate inside Givens rotation subcircuits will be attempted.
                                                 If the lattice is small and periodic, then duplicate control
                                                 links which are shared between vertices will be stripped
                                                 out first.
                                                 If None, no control pruning is attempted.
          - control_fusion: Optional boolean argument with the default set to False. If it's set
                            to be True, then LP families of givens rotations are first Gray code ordered,
                            then redundant controls are removed.
          - cache_mag_evol_circuit: Optional boolean argument to cache the magnetic Hamiltonian
                                    evolution circuit(s) once generated, and forevermore use the cache.
          - givens_have_independent_params: Optional boolean argument with the default set to False.
                                            If True, then each individual Givens rotation subcircuit
                                            will be controlled by a unique parameter.
        Returns:
          None (master_circuit has the magnetic Trotter step appended)
        """
        per_plane_pruning_states = self._strip_redundant_controls_if_small_and_periodic_lattice(
            physical_states_for_control_pruning)

        # Construct the dt and coupling parameters for the current magnetic Trotter step,
        name_prefixes = ['dt_mag', 'coupling_g_mag']
        step_num_separator = '__'
        n_dt_mag_params, n_coupling_g_mag_params = LatticeCircuitManager._count_parameters_in_circ(master_circuit, name_prefixes, separator=step_num_separator)
        if n_dt_mag_params != n_coupling_g_mag_params:
            raise NotImplementedError("Different number of magnetic dt and magnetic coupling_g Parameters encountered.")
        dt_mag_current = Parameter(f'dt_mag{step_num_separator}{n_dt_mag_params}')
        coupling_g_mag_current = Parameter(f'coupling_g_mag{step_num_separator}{n_coupling_g_mag_params}')

        # Check if cached circuits need to be invalidated due to changed build params.
        mag_evol_recomputation_needed = (
            (len(self._cached_mag_evol_circuits) == 0)
            or (control_fusion != self._cached_mag_evol_params["control_fusion"])
            or (optimize_circuits != self._cached_mag_evol_params["optimize_circuits"])
            or (
                physical_states_for_control_pruning
                != self._cached_mag_evol_params["physical_states_for_control_pruning"]
            )
        )
        if mag_evol_recomputation_needed:
            self._cached_mag_evol_circuits = {}
            self._cached_mag_evol_params = {
                "physical_states_for_control_pruning": physical_states_for_control_pruning,
                "optimize_circuits": optimize_circuits,
                "control_fusion": control_fusion,
            }

        # Pre-compute per-plane skip indices for small periodic lattices.
        # Maps (plane, vertex_idx) -> set of ctrl_idx values to skip.
        _skip_indices: Dict[Tuple[Plane, int], set[int]] = {}
        if self._lattice_is_small and self._lattice_is_periodic:
            dim = self._encoder.lattice_def.dim
            if dim == 1.5:
                for plane in self._cached_ctrl_dirs_small_and_periodic:
                    n_ctrls = len(self._cached_ctrl_dirs_small_and_periodic[plane][0])
                    _skip_indices[(plane, 1)] = set(range(n_ctrls))
                    _skip_indices[(plane, 3)] = set(range(n_ctrls))
            else:
                for plane, ctrl_dirs in self._cached_ctrl_dirs_small_and_periodic.items():
                    e1, e2 = plane
                    _skip_indices[(plane, 1)] = {ctrl_dirs[1].index(e1)}
                    _skip_indices[(plane, 2)] = {ctrl_dirs[2].index(e2)}
                    _skip_indices[(plane, 3)] = {ctrl_dirs[3].index(e2), ctrl_dirs[3].index(-e1)}

        # Local cache for Hamiltonian data per (plane, signature).
        # On periodic lattices, all plaquettes share the same key, so this
        # avoids re-iterating over self._mag_hamiltonian for every plaquette.
        _per_plane_and_signature_hamiltonian_cache: Dict[Tuple[Plane, Signature], List] = {}

        # Stitch magnetic Hamiltonian evolution circuit onto LatticeRegisters.
        # Vertex iteration loop.
        for vertex_address in lattice.vertex_addresses:
            # Skip creating "top vertex" plaquettes for d=3/2.
            has_no_vertical_periodic_link_three_halves_case = (
                lattice.dim == 1.5 and vertex_address[1] == 1
            )
            if has_no_vertical_periodic_link_three_halves_case:
                continue

            # Get the plaquettes for the current vertex.
            # On non-periodic lattices, boundary vertices may not have valid
            # plaquettes in all (or any) planes — Plaquette construction raises
            # KeyError when a vertex would fall outside the lattice.
            logger.info(f"Fetching all positive plaquettes at vertex {vertex_address}.")
            has_only_one_positive_plaquette = lattice.dim == 1.5 or lattice.dim == 2
            if has_only_one_positive_plaquette:
                try:
                    plaquettes: List[Plaquette] = [
                        lattice.get_plaquettes(vertex_address, 1, 2)
                    ]
                except KeyError:
                    continue
            else:
                # For d >= 3, construct each plane individually so that
                # planes extending beyond the boundary are skipped while
                # valid planes at the same vertex are kept.
                all_planes = sorted(
                    (i, j)
                    for i in range(1, ceil(lattice.dim) + 1)
                    for j in range(i + 1, ceil(lattice.dim) + 1)
                )
                plaquettes: List[Plaquette] = []
                for plane in all_planes:
                    try:
                        plaquettes.append(lattice.get_plaquettes(vertex_address, *plane))
                    except KeyError:
                        continue
                if not plaquettes:
                    continue
            logger.debug(f"Found {len(plaquettes)} plaquette(s).")

            # For each plaquette, apply the the local Trotter step circuit.
            for plaquette in plaquettes:
                # Resolve the Hamiltonian for this plaquette's plane and signature.
                plaquette_plane: Plane = plaquette.plane
                plaquette_signature: Signature = plaquette.signature
                cache_key = (plaquette_plane, plaquette_signature)
                if cache_key in _per_plane_and_signature_hamiltonian_cache:
                    hamiltonian_current_plane_and_signature = _per_plane_and_signature_hamiltonian_cache[cache_key]
                else:
                    hamiltonian_current_plane_and_signature = LatticeCircuitManager._resolve_hamiltonian_for_plaquette(
                        self._mag_hamiltonian, plaquette_plane, plaquette_signature
                    )
                    _per_plane_and_signature_hamiltonian_cache[cache_key] = hamiltonian_current_plane_and_signature

                # Build or fetch the cached template circuit for this (plane, signature).
                # When givens_have_independent_params is True, the template must
                # be reused across all plaquettes so that the same theta[m]
                # Parameter instances are shared (not duplicated). So always
                # consult the in-memory cache in that case.
                # NOTE: On nonperiodic lattices, this logic may fail since
                # in that case, different plaquettes in a lattice may have different
                # matrix elements and therefore different givens rotations.
                # Will need to handle that case down the road. The simplest possibility
                # would be to just forbid this option on such lattices.
                use_cache = cache_mag_evol_circuit or givens_have_independent_params
                if use_cache and (cache_key in self._cached_mag_evol_circuits):
                    logger.info(f"Fetching cached magnetic evolution circuit for cache_key={cache_key}.")
                    plaquette_local_rotation_circuit_template = self._cached_mag_evol_circuits[cache_key]
                else:
                    logger.info(f"Building magnetic evolution circuit for cache_key={cache_key}.")
                    effective_pruning_states = (
                        per_plane_pruning_states.get(plaquette_plane)
                        if per_plane_pruning_states is not None
                        else physical_states_for_control_pruning
                    )
                    plaquette_local_rotation_circuit_template = self._build_mag_evol_circuit(
                        hamiltonian_current_plane_and_signature,
                        control_fusion,
                        effective_pruning_states,
                        coupling_g=Parameter('coupling_g_mag_placeholder'),
                        dt=Parameter('dt_mag_placeholder'),
                        optimize_circuits=optimize_circuits,
                        use_independent_params_for_each_givens_rot=givens_have_independent_params
                    )
                    if use_cache:
                        self._cached_mag_evol_circuits[cache_key] = plaquette_local_rotation_circuit_template

                # Assign step-specific parameters to the template.
                if givens_have_independent_params is False:
                    plaquette_local_rotation_circuit = plaquette_local_rotation_circuit_template.assign_parameters({
                        'coupling_g_mag_placeholder': coupling_g_mag_current,
                        'dt_mag_placeholder': dt_mag_current
                    })
                else:
                    plaquette_local_rotation_circuit = plaquette_local_rotation_circuit_template

                # Collect the local qubits for stitching the plaquette rotation circuit.
                vertex_multiplicity_qubits = []
                a_link_qubits = []
                c_link_qubits = []
                for register in plaquette.vertices:
                    for qubit in register:
                        vertex_multiplicity_qubits.append(qubit)
                for register in plaquette.active_links:
                    for qubit in register:
                        a_link_qubits.append(qubit)
                for vertex_idx, vertex_controls in enumerate(plaquette.control_links_per_vertex):
                    for ctrl_idx, register in enumerate(vertex_controls):
                        # If lattice is small and has PBCs, skip redundant c_link registers.
                        if self._lattice_is_small and self._lattice_is_periodic:
                            skip_set = _skip_indices.get((plaquette_plane, vertex_idx), set())
                            if ctrl_idx in skip_set:
                                continue

                        for qubit in register:
                            c_link_qubits.append(qubit)

                # Now that we have the qubits for the current plaquette,
                # Stitch the local magnetic evolution circuit into master circuit.
                master_circuit.compose(
                    plaquette_local_rotation_circuit,
                    qubits=[
                        *vertex_multiplicity_qubits,
                        *a_link_qubits,
                        *c_link_qubits,
                        *master_circuit.ancillas
                    ],
                    inplace=True
                )

    @staticmethod
    def save_circuit(circ: QuantumCircuit, filename: str | Path, ancilla_reg_name: None | "str" = None) -> None:
        """
        Wrapper to save a lattice circuit to disk.

        The serialization type will be inferred from the extension on
        filename. Currently supported types are QASM and QPY.

        If ancilla_reg_name is provided, then it is assumed that the
        QuantumRegister instance with that name is an ancilla register.
        If ancilla_reg_name is None, then only registers of type
        AncillaRegister will be treated as ancillas.
        """
        if isinstance(filename, str):
            filename = Path(filename)

        match filename.suffix.lower():
            case ".qasm":
                with filename.open('w') as qasm_file:
                    qasm_file.write(qasm3.dumps(circ))
            case ".qpy":
                with open(filename, 'wb') as qpy_file:
                    qpy.dump(circ, qpy_file)
            case _:
                raise ValueError(f"Unsupported file type: {filename.suffix}")

    @staticmethod
    def load_circuit(filename: str | Path, ancilla_reg_name: None | "str" = None) -> QuantumCircuit:
        """
        Wrapper to load a lattice circuit to disk.

        The serialization type will be inferred from the extension on
        filename. Currently supported types are QASM and QPY.

        If ancilla_reg_name is provided, then it is assumed that a
        quantum register with that name is an ancilla register. This is relevant
        for QASM files, which do not have a specific ancilla register type.
        If ancilla_reg_name is None, then only registers of type
        AncillaRegister will be treated as ancillas. This is relevant
        for QPY files.

        If a circuit has been serialized as a QPY file and it has an ancilla register,
        it is recommended to provide an ancilla register
        name anyway due to deserialization bugs that can occur with qiskit.
        """
        if isinstance(filename, str):
            filename = Path(filename)

        match filename.suffix.lower():
            case ".qasm":
                # QASM register names may have unwanted "esc_" prefixes on them.
                # Automatically remove if present.
                loaded_circ = qasm3.load(filename)
                loaded_circ = LatticeCircuitManager._rename_registers_strip_prefix(loaded_circ, prefix="esc_")
                if ancilla_reg_name is not None:
                    loaded_circ = LatticeCircuitManager._convert_register_to_ancilla(loaded_circ, ancilla_reg_name)
                return loaded_circ
            case ".qpy":
                with open(filename, "rb") as handle:
                    loaded_circ = qpy.load(handle)[0]
                return loaded_circ
            case _:
                raise ValueError(f"Unsupported file type: {filename.suffix}")

    def _strip_redundant_controls_if_small_and_periodic_lattice(
        self, physical_states_for_control_pruning: set[str] | None
    ) -> Dict[Plane, set[str] | None] | None:
        """
        For lattices that are small and periodic, strip redundant controls per plane.

        Returns a dict mapping each plane to its stripped physical state set,
        or None if the lattice is not small-and-periodic or if the input is None
        (in which case callers should use the original input set unchanged).
        """
        if (physical_states_for_control_pruning is None
                or not self._lattice_is_periodic
                or not self._lattice_is_small):
            return None

        result: Dict[Plane, set[str] | None] = {}
        for plane in self._cached_ctrl_dirs_small_and_periodic:
            stripped = []
            for plaquette_string in physical_states_for_control_pruning:
                plaquette_state = self._encoder.decode_bit_string_to_plaquette_state(plaquette_string)
                if self._plaquette_state_has_inconsistent_controls(plaquette_state, plane):
                    continue
                trimmed = self._discard_duplicate_controls_from_plaquette_state(plaquette_state, plane)
                trimmed_bs = self._encoder.encode_plaquette_state_as_bit_string(
                    trimmed, override_n_c_links_validation=True)
                stripped.append(trimmed_bs)
            result[plane] = set(stripped) if stripped else None
            logger.info(f"Plane {plane}: {len(stripped)} stripped plaquette states.")

        return result

    @staticmethod
    def _resolve_hamiltonian_for_plaquette(
            hamiltonian: PlaneKeyedHamiltonianData,
            plane: Plane,
            signature: Signature,
    ) -> List[Tuple[str, str, float]]:
        """
        'Resolve' a PlaneKeyedHamiltonianData dict to a flat list for a specific plaquette's plane and signature.

        The hamiltonian is keyed per-plane-first:
        ``plane -> (bs1, bs2) -> signature -> float``.
        This method looks up the given plane, then filters entries by signature.

        Returns:
            A flat list of (bitstring1, bitstring2, float) tuples suitable for
            _build_mag_evol_circuit and _sort_matrix_elements_into_lp_bins.
        """
        plane_data = hamiltonian.get(plane, {})
        resolved: List[Tuple[str, str, float]] = []
        for (bs1, bs2), sig_dict in plane_data.items():
            sig_val = sig_dict.get(signature)
            if sig_val is not None:
                resolved.append((bs1, bs2, float(sig_val)))

        return resolved

    def _build_mag_evol_circuit(
        self,
        hamiltonian_for_specific_plane_and_signature: List[Tuple[str, str, float]],
        control_fusion: bool,
        physical_states_for_control_pruning: Union[None | Set[str]],
        coupling_g: Parameter,
        dt: Parameter,
        optimize_circuits: bool,
        use_independent_params_for_each_givens_rot: bool = False
    ) -> QuantumCircuit:
        """
        Build the magnetic time-evolution circuit for a plaquette.

        Arguments:
            hamiltonian_for_specific_plane_and_signature: A flat list of (bitstring1, bitstring2, float)
                tuples — the resolved matrix elements for a specific
                (plane, signature) combination.
            control_fusion: Whether to fuse controls in Givens rotations.
            physical_states_for_control_pruning: Physical states for pruning.
            coupling_g: Coupling constant parameter.
            dt: Time step parameter.
            optimize_circuits: Whether to transpile with optimization.
            use_independent_params_for_each_givens_rot: If True, coupling_g
                and dt are ignored and a unique theta[m] parameter is assigned
                per Givens rotation.
        """
        n_givens_rotations = len(hamiltonian_for_specific_plane_and_signature)
        logger.info(f"There are {n_givens_rotations} primitive Givens rotation circuits to be constructed for the plaquette.")
        # Sort the bitstrings corresponding to transitions in the magnetic
        # Hamiltonian into LP bins. This step also computes the angle of Givens
        # rotation for each pair of bitstrings. The resulting Givens rotations
        # are characterized by two parameters: dt and the coupling g.
        lp_bin = LatticeCircuitManager._sort_matrix_elements_into_lp_bins(
            hamiltonian_for_specific_plane_and_signature,
            coupling_g,
            dt,
        )
        # If specified by function arguments, replace each Givens rotation
        # angle with an independent theta Parameter.
        if use_independent_params_for_each_givens_rot is True:
            rot_angle_param_vector_iterator = iter(ParameterVector("theta", n_givens_rotations))
            lp_bin = {
                lp_fam: [(bit_string_1, bit_string_2, next(rot_angle_param_vector_iterator))
                         for (bit_string_1, bit_string_2, old_angle) in lp_bin[lp_fam]]
                for lp_fam in lp_bin.keys()
            }
        # Sort according to Gray-order if performing control fusion.
        if control_fusion is True:
            lp_bin = {
                k: lp_bin[k]
                for k in sorted(
                    lp_bin.keys(),
                    key=lambda x: gray_to_index(bitstring_value_of_LP_family(x)),
                )
            }

        # Iterate over all LP bins and apply givens rotation.
        # Also logs progress of circuit construction at INFO level.
        plaquette_circ_n_qubits = len(hamiltonian_for_specific_plane_and_signature[0][0])
        plaquette_local_rotation_circuit = QuantumCircuit(plaquette_circ_n_qubits)
        loop_time_state = None  # For tracking Givens rotation circuit construction progress.
        if (self.num_ancillas > 0):
            plaquette_local_rotation_circuit.add_register(AncillaRegister(self.num_ancillas))
        for idx, (lp_fam, lp_bin_w_angle) in enumerate(lp_bin.items()):
            loop_time_state, eta = eta_update(state=loop_time_state, processed=idx+1, total=len(lp_bin.items()))
            iter_msg = (
                f"Constructing rotation circuit for LP bin {idx + 1}/{len(lp_bin.items())} with {len(lp_bin_w_angle)} Givens rotations."
            )
            eta_msg = "More iterations needed to estimate time remaining." if idx == 0 else f"Estimated time remaining: {fmt_td(eta)}"
            logger.info(iter_msg)
            logger.info(eta_msg)

            # Compute P_tilde once per LP bin for control pruning.
            if physical_states_for_control_pruning is not None:
                p_tilde = compute_p_tilde(lp_fam, physical_states_for_control_pruning)
            else:
                p_tilde = None

            if control_fusion is True:
                fused_circ_for_lp_fam = givens_fused_controls(
                    lp_bin_w_angle, lp_fam, physical_states_for_control_pruning, self.num_ancillas,
                    precomputed_p_tilde=p_tilde,
                )
                plaquette_local_rotation_circuit.compose(
                    fused_circ_for_lp_fam, inplace=True
                )
            else:
                # If control fusion is turned off, givens rotation is applied individually
                # to all bitstrings.
                for bs1, bs2, angle in lp_bin_w_angle:
                    bs1_bs2_circuit = givens(
                        bs1, bs2, angle, physical_states_for_control_pruning, self.num_ancillas,
                        precomputed_p_tilde=p_tilde,
                    )
                    plaquette_local_rotation_circuit.compose(
                        bs1_bs2_circuit, inplace=True
                    )
            if optimize_circuits is True:
                plaquette_local_rotation_circuit = transpile(
                    plaquette_local_rotation_circuit, optimization_level=3
                )

        return plaquette_local_rotation_circuit

    def _plaquette_state_has_inconsistent_controls(self, plaquette: PlaquetteState, plane: Plane) -> bool:
        """
        True if "shared" control links have different states; False otherwise.

        For d=3/2: v1 controls == v2 controls, v3 controls == v4 controls (no plane dependence).

        For d=2 and d=3: uses direction-based lookups via the per-plane cached ctrl dirs,
        so results are correct for any F-order and any plane. On a size-2 periodic lattice,
        the four in-plane sharing pairs are:
        - v1[dir -e1] shares with v2[dir +e1]
        - v1[dir -e2] shares with v4[dir +e2]
        - v2[dir -e2] shares with v3[dir +e2]
        - v3[dir +e1] shares with v4[dir -e1]

        Note that this only makes sense on a small, periodic lattice, so a ValueError
        is raised if the lattice fails those checks.
        """
        if (self._lattice_is_periodic is False) or (self._lattice_is_small is False):
            raise ValueError("Plaquette state consistency check only makes sense on a small, periodic lattice.")

        c_links = plaquette[2]
        dim = self._encoder.lattice_def.dim

        if dim == 1.5:
            # d=3/2: wholesale vertex equality, no plane dependence.
            return c_links[0] != c_links[1] or c_links[2] != c_links[3]

        # d=2 and d=3: direction-based lookup using the per-plane cache.
        ctrl_dirs = self._cached_ctrl_dirs_small_and_periodic[plane]
        e1, e2 = plane
        return (
            (c_links[0][ctrl_dirs[0].index(-e1)] != c_links[1][ctrl_dirs[1].index(e1)]) or
            (c_links[0][ctrl_dirs[0].index(-e2)] != c_links[3][ctrl_dirs[3].index(e2)]) or
            (c_links[1][ctrl_dirs[1].index(-e2)] != c_links[2][ctrl_dirs[2].index(e2)]) or
            (c_links[2][ctrl_dirs[2].index(e1)] != c_links[3][ctrl_dirs[3].index(-e1)])
        )

    def _discard_duplicate_controls_from_plaquette_state(self, plaquette: PlaquetteState, plane: Plane) -> PlaquetteState:
        """
        Return a new instance of the plaquette where duplicate control link data has been discarded.

        For d=3/2: keep v1 and v3 controls, drop v2 and v4 (since v1==v2, v3==v4).

        For d=2 and d=3: uses direction-based trimming via the per-plane cached ctrl dirs.
        Keeps first occurrence of each shared physical link. This method is FORDER-aware.
        - v1: keep all controls
        - v2: drop dir +e1 (duplicates v1's dir -e1)
        - v3: drop dir +e2 (duplicates v2's dir -e2)
        - v4: drop dir +e2 (duplicates v1's dir -e2) and dir -e1 (duplicates v3's dir +e1)

        Since this only makes sense on a small, periodic lattice, a ValueError
        is raised if the lattice is not small and periodic.
        """
        if (self._lattice_is_periodic is False) or (self._lattice_is_small is False):
            raise ValueError("Plaquette state consistency check only makes sense on a small, periodic lattice.")

        vertex_multiplicities, a_links, c_links = plaquette
        dim = self._encoder.lattice_def.dim

        if dim == 1.5:
            physical_c_links = (c_links[0], (), c_links[2], ())
        else:
            # d=2 and d=3: direction-based trimming.
            ctrl_dirs = self._cached_ctrl_dirs_small_and_periodic[plane]
            e1, e2 = plane
            physical_c_links = (
                c_links[0],
                tuple(c for i, c in enumerate(c_links[1])
                      if i != ctrl_dirs[1].index(e1)),
                tuple(c for i, c in enumerate(c_links[2])
                      if i != ctrl_dirs[2].index(e2)),
                tuple(c for i, c in enumerate(c_links[3])
                      if i != ctrl_dirs[3].index(e2) and i != ctrl_dirs[3].index(-e1)),
            )

        plaquette_with_filtered_c_links = (vertex_multiplicities, a_links, physical_c_links)
        return plaquette_with_filtered_c_links

    @staticmethod
    def _rename_registers_strip_prefix(orig_circ: QuantumCircuit, prefix: str) -> QuantumCircuit:
        """
        Return a new QuantumCircuit where any quantum/classical register whose name
        starts with `prefix` is renamed to the same name with that prefix removed.
        Registers without the prefix keep their original names. Register sizes,
        ordering, and all instructions are preserved.
        """
        # Build new quantum registers preserving order and sizes.
        new_qregs = []
        for reg in orig_circ.qregs:
            name = reg.name
            new_name = name[len(prefix):] if name.startswith(prefix) else name
            new_qregs.append(QuantumRegister(len(reg), new_name))

        # Build new classical registers preserving order and sizes.
        new_cregs = []
        for reg in orig_circ.cregs:
            name = reg.name
            new_name = name[len(prefix):] if name.startswith(prefix) else name
            new_cregs.append(ClassicalRegister(len(reg), new_name))

        # Create mapping from old register objects to new register objects.
        old_qregs = list(orig_circ.qregs)
        old_cregs = list(orig_circ.cregs)
        qreg_map = {old_qregs[i]: new_qregs[i] for i in range(len(old_qregs))}
        creg_map = {old_cregs[i]: new_cregs[i] for i in range(len(old_cregs))}

        # Construct new circuit with same name and global metadata intact where applicable.
        new_circ = QuantumCircuit(*new_qregs, *new_cregs, name=orig_circ.name)

        # Copy instructions, mapping bits to the new registers' bits by index.
        for instr, qargs, cargs in orig_circ.data:
            mapped_qargs = []
            for qb in qargs:
                old_reg, qb_index = orig_circ.find_bit(qb).registers[0]
                new_reg = qreg_map[old_reg]
                mapped_qargs.append(new_reg[qb_index])
            mapped_cargs = []
            for cb in cargs:
                old_reg, cb_index = orig_circ.find_bit(cb).registers[0]
                new_reg = creg_map[old_reg]
                mapped_cargs.append(new_reg[cb_index])
            new_circ.append(instr, mapped_qargs, mapped_cargs)

        return new_circ

    @staticmethod
    def _convert_register_to_ancilla(orig_circ: QuantumCircuit, reg_name: str) -> QuantumCircuit:
        """
        Return a new QuantumCircuit that's a copy of `orig_circ` but where the register
        whose name equals `reg_name` (quantum register) is recreated as an AncillaRegister.
        Other registers (quantum and classical) keep their original names, sizes and order.

        Raises:
          ValueError: if no quantum register with the given name exists in `orig_circ`.
        """
        # Collect old registers in order.
        old_qregs = list(orig_circ.qregs)
        old_cregs = list(orig_circ.cregs)

        # Find index of target quantum register.
        target_idx: int | None = None
        for i, r in enumerate(old_qregs):
            if r.name == reg_name:
                target_idx = i
                break
        if target_idx is None:
            raise ValueError(f"No quantum register named {reg_name!r} in circuit")

        # Build new quantum registers: replace the target with AncillaRegister of same size/name.
        new_qregs = []
        for i, r in enumerate(old_qregs):
            if i == target_idx:
                new_qregs.append(AncillaRegister(len(r), name=r.name))
            else:
                new_qregs.append(QuantumRegister(len(r), name=r.name))

        # Build new classical registers (preserve).
        new_cregs = [ClassicalRegister(len(r), name=r.name) for r in old_cregs]

        # Map old register objects to new register objects (positional).
        qreg_map = {old_qregs[i]: new_qregs[i] for i in range(len(old_qregs))}
        creg_map = {old_cregs[i]: new_cregs[i] for i in range(len(old_cregs))}

        # Construct new circuit with same name and global settings.
        new_circ = QuantumCircuit(*new_qregs, *new_cregs, name=orig_circ.name)

        # Copy global circuit metadata if present (optional),
        # preserve global phase, metadata, and header if present.
        if hasattr(orig_circ, "global_phase"):
            new_circ.global_phase = orig_circ.global_phase
        if getattr(orig_circ, "metadata", None) is not None:
            new_circ.metadata = orig_circ.metadata.copy()

        # Copy instructions: qargs/cargs are Bit objects (use register mapping + index).
        for instr, qargs, cargs in orig_circ.data:
            mapped_qargs = []
            for qb in qargs:
                old_reg, qb_index = orig_circ.find_bit(qb).registers[0]
                new_reg = qreg_map[old_reg]
                mapped_qargs.append(new_reg[qb_index])
            mapped_cargs = []
            for cb in cargs:
                old_reg, cb_index = orig_circ.find_bit(cb).registers[0]
                new_reg = creg_map[old_reg]
                mapped_cargs.append(new_reg[cb_index])
            new_circ.append(instr, mapped_qargs, mapped_cargs)

        return new_circ

    @staticmethod
    def _sort_matrix_elements_into_lp_bins(
        bitstrings_w_matrix_element: List[(str, str, float | Parameter)],
        coupling_g: float | Parameter,
        dt: float | Parameter,
    ) -> Dict[LPFamily, List[(str, str, float | Parameter)]]:
        """
        Rearrange magnetic Hamiltonian matrix elements to LP family bins.

        This function does two things:
        1. Sorts tuples of bitstrings into LP bins.
        2. Computes angle of Givens rotation from matrix element

        Input:
            - bitstring_w_matrix_element: this is of the form (bs1, bs2, matrix_element)
                where bs1 is the initial state, bs2 is the final state, and matrix_element
                is the amplitude of transition.
            - coupling_g: value of coupling constant being used.
            - dt: timestep being used.

        Output:
            - dictionary where each key is a LP bin, and the corresponding value is a list of transitions
                that have the same LP value. Each transition is of the form (bitstring1, bitstring2, angle).
        """
        lp_bin = {}
        for (
            bit_string_1,
            bit_string_2,
            matrix_elem,
        ) in bitstrings_w_matrix_element:
            angle = -matrix_elem * (1 /  (coupling_g * coupling_g)) * dt
            lp_fam = compute_LP_family(bit_string_1, bit_string_2)
            if lp_fam not in lp_bin.keys():
                lp_bin[lp_fam] = []
            lp_bin[lp_fam].append((bit_string_1, bit_string_2, angle))
        return lp_bin
