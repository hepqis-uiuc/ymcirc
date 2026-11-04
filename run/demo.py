from qiskit import transpile
from qiskit_aer import AerSimulator
from ymcirc._abstract import LatticeDef
from ymcirc.conventions import LatticeStateEncoder
from ymcirc.circuit import LatticeCircuitManager
from ymcirc.electric_helper import electric_hamiltonian
from ymcirc.lattice_registers import LatticeRegisters
from ymcirc.measurement_results import MeasurementResults
from ymcirc.parsed_lattice_result import ParsedLatticeResult


COUPLING_G = 1.0
DELTA_T = 0.1


# Script is inside a __name__ == "__main__" block
# because LatticeCircuitManager.for_truncation
# uses pyclebsch to compute magnetic Hamiltonian
# matrix elements, with parallelization via the
# multiprocessing library. This parallelization
# fails if outside a __name__ == "__main__" block.
if __name__ == "__main__":
    # Build a 2x2x2 lattice with open boundary conditions (i.e. a single cube).
    lattice = LatticeDef(
        dimensions=3,
        size=(2, 2, 2),
        periodic_boundary_conds=False)
    encoder = LatticeStateEncoder.for_truncation(
        lattice=lattice,
        truncation_mode="B",
        cutoff=3
    )
    regs = LatticeRegisters.from_lattice_state_encoder(encoder)
    circ_mgr = LatticeCircuitManager.for_truncation(encoder)
    circ = circ_mgr.create_blank_full_lattice_circuit(regs)
    circ_mgr.apply_magnetic_trotter_step(circ, regs)
    circ_mgr.apply_electric_trotter_step(
        circ, regs, electric_hamiltonian(encoder.link_bitmap))

    # Inspect the classes we made, and the resulting circuit.
    print(lattice)
    print(encoder)
    print(regs)
    print(circ_mgr)
    print(f"Circuit has {len(circ.qubits) - len(circ.ancillas)} data qubits, {len(circ.ancillas)} ancilla qubits.")
    print("Rough gate counts:", transpile(circ, basis_gates=["u", "t", "tdg", "cx"], optimization_level=3).count_ops())
    print("Rough gate depth:", transpile(circ, basis_gates=["u", "t", "tdg", "cx"], optimization_level=3).depth())

    # Set circuit parameters.
    parameter_values = {}
    for param in circ.parameters:
        if 'dt' in param.name:
            parameter_values[param] = DELTA_T
        elif 'coupling_g' in param.name:
            parameter_values[param] = COUPLING_G
    circ.assign_parameters(parameter_values, inplace=True)
    print("Circuit parameters assigned.")

    # Execute circuit. Involves transpiling to backend and
    # adding measurements.
    simulator = AerSimulator(method="matrix_product_state", matrix_product_state_max_bond_dimension=16)
    circ.measure_all()
    print("Starting circuit execution.")
    circ = transpile(circ, simulator, optimization_level=3)
    job = simulator.run(circ, shots=10000)
    job_result = job.result().get_counts()
    print("Circuit execution done.")

    # Collect and parse results.
    counts = {}
    for little_endian_state, count in job_result.items():
        big_endian_state = little_endian_state[::-1]
        decoded_state = ParsedLatticeResult(
            dimensions=lattice.dim,
            size=lattice.size,
            global_lattice_measurement_bit_string=big_endian_state,
            lattice_encoder=encoder,
            periodic_boundary_conds=lattice.periodic_boundary_conds
        )
        counts[decoded_state] = count
    mr = MeasurementResults(counts, encoder)

    # Inspect results.
    print(f"Average electric energy: {mr.get_lattice_electric_energy()}")
    print(f"Vacuum persistence probability: {mr.vacuum_persistence_probability()}")
