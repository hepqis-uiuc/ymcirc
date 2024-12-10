"""
A collection of utilities for building circuits.
"""
from __future__ import annotations
import copy
from lattice_tools.conventions import LatticeStateEncoder
from lattice_tools.lattice_registers import LatticeRegisters, Plaquette
from lattice_tools.givens import givens
from math import ceil
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister
from typing import List, Tuple
from lattice_tools.electric_helper import electric_hamiltonian
import numpy as np

# A list of tuples: (state bitstring1, state bitstring2, matrix element)
HamiltonianData = List[Tuple[str, str, float]]


class LatticeCircuitManager:
    """Class for creating quantum simulation circuits from LatticeRegister instances."""

    def __init__(self, lattice_encoder: LatticeStateEncoder, mag_hamiltonian: HamiltonianData):
        """Create via a LatticeStateEncoder instance and magnetic Hamiltonian matrix elements."""
        # Copies to avoid inadvertently changing the behavior of the
        # LatticeCircuitManager instance.
        self._encoder = copy.deepcopy(lattice_encoder)
        self._mag_hamiltonian = copy.deepcopy(mag_hamiltonian)

    def create_blank_full_lattice_circuit(self, lattice: LatticeRegisters) -> QuantumCircuit:
        """
        Return a blank quantum circuit with all link and vertex registers in lattice.

        Length-zero registers are skipped (relevant for d=3/2, T1 vertex registers.)

        The convention is to construct the circuit by iterating over all vertices,
        then for each vertex, to iterate over all the "positive" links leaving the vertex.
        The iteration over vertices is by ordering on the tuples denoting
        lattice coordinates.

        Example for d = 2 with periodic boundary conditions:

        (pbc)        (pbc)        (pbc)
          |            |            |
          |            |            |
          l16          l17          l18
          |            |            |
          |            |            |
        (0,2)--l13---(1,2)--l14---(2,2)--l15---- (pbc)
          |            |            |
          |            |            |
          l10          l11          l12
          |            |            |
          |            |            |
        (0,1)---l7---(1,1)---l8---(2,1)---l9---- (pbc)
          |            |            |
          |            |            |
          l4           l5           l6
          |            |            |
          |            |            |
        (0,0)---l1---(1,0)---l2---(2,0)---l3---- (pbc)


        Will be mapped to the ket

        |(0,0) l1 l4 (0,1) l7 l10 (0,2) l13 l16 (1,0) l2 l5 ...>

        where the left-most tensor factor is the top line in the circuit.

        In d=3/2, the "top" pbc links in the above diagram are omitted
        because they do not exist on that lattice.
        """
        all_lattice_registers: List[QuantumRegister] = []
        for vertex_address in lattice.vertex_register_keys:
            # Add the current vertex, and the positive links connected to it.
            # Skip "top" in d = 3/2.
            current_vertex_reg = lattice.get_vertex_register(vertex_address)
            all_lattice_registers.append(current_vertex_reg)
            for positive_direction in range(1, ceil(lattice.dim) + 1):
                has_no_vertical_periodic_link_three_halves_case = \
                    lattice.dim == 1.5 and positive_direction > 1 and vertex_address[1] == 1
                if has_no_vertical_periodic_link_three_halves_case:
                    continue

                current_link_reg = lattice.get_link_register(vertex_address, positive_direction)
                all_lattice_registers.append(current_link_reg)

        return QuantumCircuit(*all_lattice_registers)

    # TODO implement electric trotter step.
    def apply_electric_trotter_step(
            self,
            master_circuit: QuantumCircuit,
            lattice: LatticeRegisters, hamiltonian: list, coupling_g: float = 1.0,
            dt: float = 1.0) -> None:
        """
        Electric trotter step implementation using CX and Zs to implement rotations of Z,I Paulis

        Should modify master_circuit in place rather than returning a new circuit because that's more efficient.
        """
        
        N = int(np.log2(len(hamiltonian)))

        angle_mod = ((coupling_g**2) / 2) * dt

        local_circuit = QuantumCircuit(N) 

        for i in range(len(hamiltonian)):
            locs = [loc for loc, bit in enumerate(str('{0:0' + str(N) + 'b}').format(i)) if bit=='1']
            print(locs)
            for j in locs[:-1]:
                local_circuit.cx(j, locs[-1])
            if (len(locs)!=0): local_circuit.rz(-2*angle_mod*hamiltonian[i], locs[-1])
            for j in locs[:-1]:
                local_circuit.cx(j, locs[-1])

        # Loop over links for electric Hamiltonain

        for link_key in lattice.link_register_keys:
            link_qubits = [qubit for qubit in lattice.get_link_register(link_key[0], link_key[1])]
            master_circuit.compose(
                        local_circuit,
                        qubits= link_qubits,
                        inplace=True
                    )


    # TODO Can we get the circuits in a parameterized way?
    def apply_magnetic_trotter_step(
            self,
            master_circuit: QuantumCircuit,
            lattice: LatticeRegisters,
            coupling_g: float = 1.0,
            dt: float = 1.0,
            optimize_circuits: bool = True) -> None:
        """
        Add one Trotter step.

        Note that this modifies master_circuit directly rather than returning a new circuit!

        Arguments:
          - lattice: a LatticeRegisters instance which keeps track of all the QuantumRegisters.
          - master_circuit: a QuantumCircuit instance which is built from all the
                            QuantumRegister instances in lattice.
          - hamiltonian: a dict whose keys are tuples of bitstrings corrresponding to
                         "plaquette final state" and "plaquette initial state", and whose
                         values correspond to numerical matrix element values.
          - coupling_g: the value of the strong coupling constant.
          - dt: the size of the Trotter time step.
          - optimize_circuits: if true, run the qiskit transpiler on each internal givens rotation
                               with the maximum optimization level before composing with master_circuit.

        Returns:
          A new QuantumCircuit instance which is master_circuit with the Trotter step appended.
        """
        # Big picture: iterate over every lattice vertex, and add the trotter step
        # to every "postive" plaquette at that vertex.
        for vertex_address in lattice.vertex_register_keys:
            # Skip creating "top vertex" plaquettes for d=3/2.
            has_no_vertical_periodic_link_three_halves_case = \
                lattice.dim == 1.5 and vertex_address[1] == 1
            if has_no_vertical_periodic_link_three_halves_case:
                continue

            # Get the plaquettes for the current vertex.
            print(f"Fetching all positive plaquettes at vertex {vertex_address}.")
            has_only_one_positive_plaquette = lattice.dim == 1.5 or lattice.dim == 2
            if has_only_one_positive_plaquette:
                plaquettes: List[Plaquette] = [lattice.get_plaquette_registers(vertex_address, 1, 2)]
            else:
                plaquettes: List[Plaquette] = lattice.get_plaquette_registers(vertex_address)
            print(f"Found {len(plaquettes)} plaquette(s).")

            # For each plaquette, apply the the local Trotter step circuit.
            for plaquette in plaquettes:
                # Collect the local qubits for stitching purposes.
                vertex_qubits = []
                link_qubits = []
                for register in plaquette.vertex_registers:
                    for qubit in register:
                        vertex_qubits.append(qubit)
                for register in plaquette.link_registers:
                    for qubit in register:
                        link_qubits.append(qubit)

                # Append a Givens rotation circuit for each magnetic Hamiltonian
                # matrix element.
                for bitstring_1, bitstring_2, matrix_elem in self._mag_hamiltonian:
                    angle = -matrix_elem * (1 / (2 * (coupling_g**2))) * dt
                    plaquette_local_rotation_circuit = givens(bitstring_1, bitstring_2, angle)
                    if optimize_circuits is True:
                        plaquette_local_rotation_circuit = transpile(
                            plaquette_local_rotation_circuit, optimization_level=3)

                    # Stitch the Givens rotation into master circuit.
                    master_circuit.compose(
                        plaquette_local_rotation_circuit,
                        qubits=[
                            *vertex_qubits,
                            *link_qubits
                        ],
                        inplace=True
                    )

def _test_create_blank_full_lattice_circuit_has_promised_register_order():
    """Check in some cases that we get the ordering promised in the method docstring."""
    # Creating test data.
    # Not physically meaningful, but has the right format.
    irrep_bitmap = {
        (0, 0, 0): "0",
        (1, 0, 0): "1"
    }
    singlet_bitmap_2d = {
        (((0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)), 1): "00",
        (((0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)), 2): "01",
        (((1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)), 1): "10"
    }
    singlet_bitmap_3halves = {
        (((0, 0, 0), (1, 0, 0), (1, 0, 0)), 1): "0",
        (((0, 0, 0), (1, 0, 0), (1, 0, 0)), 2): "1",
    }
    singlet_bitmap_3halves_no_vertices = {
    }
    mag_hamiltonian_2d = [("000110000001", "000110000010", 1.0), ("010110000001", "100110000010", 1.0)]
    mag_hamiltonian_3halves = [("00001111", "11110000", 1.0), ("10100101", "00000001", 1.0)]
    mag_hamiltonian_3halves_no_vertices = [("1111", "000", 1.0), ("1001", "0001", 1.0), ("1101", "0101", 1.0)]
    expected_register_order_2d = [
        'v:(0, 0)', 'l:((0, 0), 1)', 'l:((0, 0), 2)',
        'v:(0, 1)', 'l:((0, 1), 1)', 'l:((0, 1), 2)',
        'v:(1, 0)', 'l:((1, 0), 1)', 'l:((1, 0), 2)',
        'v:(1, 1)', 'l:((1, 1), 1)', 'l:((1, 1), 2)',
    ]
    expected_register_order_3halves = [
        'v:(0, 0)', 'l:((0, 0), 1)', 'l:((0, 0), 2)',
        'v:(0, 1)', 'l:((0, 1), 1)',
        'v:(1, 0)', 'l:((1, 0), 1)', 'l:((1, 0), 2)',
        'v:(1, 1)', 'l:((1, 1), 1)'
    ]
    expected_register_order_3halves_no_vertices = [
        'l:((0, 0), 1)', 'l:((0, 0), 2)',
        'l:((0, 1), 1)',
        'l:((1, 0), 1)', 'l:((1, 0), 2)',
        'l:((1, 1), 1)',
    ]
    test_cases = [
        (expected_register_order_2d, irrep_bitmap, singlet_bitmap_2d, 2, mag_hamiltonian_2d),
        (expected_register_order_3halves, irrep_bitmap, singlet_bitmap_3halves, 1.5, mag_hamiltonian_3halves),
        (expected_register_order_3halves_no_vertices, irrep_bitmap, singlet_bitmap_3halves_no_vertices, 1.5, mag_hamiltonian_3halves_no_vertices)
    ]

    # Iterate over all test cases.
    for expected_register_names_ordered, link_bitmap, vertex_bitmap, dims, hamiltonian in test_cases:
        print(f"Checking register order in a circuit constructed from a {dims}-dimensional lattice.")
        print(f"Link bitmap: {link_bitmap}\nVertex bitmap: {vertex_bitmap}")
        print(f"Expected register ordering: {expected_register_names_ordered}")

        # Create circuit.
        lattice = LatticeRegisters(
            dim=dims,
            size=2,
            link_truncation_dict=link_bitmap,
            vertex_singlet_dict=vertex_bitmap
        )
        circ_mgr = LatticeCircuitManager(
            lattice_encoder=LatticeStateEncoder(link_bitmap, vertex_bitmap),
            mag_hamiltonian=hamiltonian
        )
        master_circuit = circ_mgr.create_blank_full_lattice_circuit(lattice)
        nonzero_regs = [reg for reg in master_circuit.qregs if len(reg) > 0]
        n_nonzero_regs = len(nonzero_regs)

        # Check that the circuit makes sense.
        assert n_nonzero_regs == len(expected_register_names_ordered), f"Expected {len(expected_register_names_ordered)} registers. Encountered {n_nonzero_regs} registers."
        for expected_name, reg in zip(expected_register_names_ordered, nonzero_regs):
            if len(reg) == 0:
                continue
            assert expected_name == reg.name, f"Expected: {expected_name}, encountered: {reg.name}"
            print(f"Verified location of the register for {expected_name}.")

    print("Register order tests passed.")


def _run_tests():
    _test_create_blank_full_lattice_circuit_has_promised_register_order()


if __name__ == "__main__":
    _run_tests()

    print("All tests passed.")
