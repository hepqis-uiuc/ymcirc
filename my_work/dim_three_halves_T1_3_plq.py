#####################################################
# The aim is to write a script to simulate the 
# d=3/2 case for T1 on qiskit_aer for the 
#  three plaquette system and then try to run 
#  the same circuit on bluequbit, even though 
# Patrick hasn't given access to you yet. We are 
# hoping that we will learn more about the inner
# workings of qcd quantum circuits and learn more
# about gauge theory, even though right now, it seems 
# like I am doing this because they didn't have any work
# to give me. On the bright side, if I get good results,
# I might just get my name on the paper they're writing!
# Seems worth it, right? AHHHHHHH this is kinda scary.
# I wore a mask so long that it became my face.
#########################################################

from lattice_tools.conventions import (
    VERTEX_SINGLET_BITMAPS,
    IRREP_TRUNCATION_DICT_1_3_3BAR,
    IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8)
from lattice_tools.circuit import LatticeCircuitManager
from lattice_tools.lattice_registers import LatticeRegisters
from lattice_tools.electric_helper import electric_hamiltonian
from lattice_tools.conventions import HAMILTONIAN_BOX_TERMS, LatticeStateEncoder
from qiskit import transpile
from qiskit_aer.primitives import SamplerV2
from qiskit_aer import AerSimulator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dimensionality_and_truncation_string = "d=3/2, T1"
dimensions = 1.5
linear_size=2
coupling_g = 1.0
trunc_string="T1"
sim_times = np.linspace(0.05, 4.0, num=20)
n_trotter_steps = 2
run_circuit_optimization = False
vertex_bitmap = {}
link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR
n_shots = 1024

lattice_encoder = LatticeStateEncoder(link_bitmap=link_bitmap,vertex_bitmap=vertex_bitmap)
box_term = []
box_dagger_term = []

for (final_plaquette_state, initial_plaquette_state), matrix_element_value in HAMILTONIAN_BOX_TERMS[dimensionality_and_truncation_string].items():
    final_state_bitstring = lattice_encoder.encode_plaquette_state_as_bit_string(final_plaquette_state)
    initial_state_bitstring = lattice_encoder.encode_plaquette_state_as_bit_string(initial_plaquette_state)
    box_term.append((final_state_bitstring, initial_state_bitstring, matrix_element_value))
    box_dagger_term.append((initial_state_bitstring, final_state_bitstring, matrix_element_value))

mag_hamiltonian = box_term + box_dagger_term
df_job_results = pd.DataFrame(columns = ["vacuum_persistence_probability"], index=sim_times)

for sim_time in sim_times:
    dt = sim_time / n_trotter_steps
    lattice = LatticeRegisters(
                dim=dimensions,
                size=linear_size,
                link_truncation_dict=link_bitmap,
                vertex_singlet_dict=vertex_bitmap
            )
    circ_mgr = LatticeCircuitManager(lattice_encoder, mag_hamiltonian)
    master_circuit = circ_mgr.create_blank_full_lattice_circuit(lattice)
    for _ in range(n_trotter_steps):
        circ_mgr.apply_magnetic_trotter_step(
                        master_circuit,
                        lattice,
                        coupling_g=coupling_g,
                        dt=dt,
                        optimize_circuits=run_circuit_optimization
                    )
        circ_mgr.apply_electric_trotter_step(master_circuit, lattice, electric_hamiltonian(link_bitmap), coupling_g=coupling_g,
                        dt=dt)
    master_circuit.measure_all()
    master_circuit = transpile(master_circuit, optimization_level=3)
    sim = AerSimulator()
    sampler = SamplerV2()
    job = sampler.run([master_circuit], shots = n_shots)
    job_result = job.result()
    n_qubits_in_lattice = (lattice.n_qubits_per_vertex * len(lattice.vertex_register_keys)) \
                + (lattice.n_qubits_per_link * len(lattice.link_register_keys))
    current_vacuum_state = "0" * n_qubits_in_lattice
    for state, counts in job_result[0].data.meas.get_counts().items():
            df_job_results.loc[sim_time,state] = counts
            # Make sure vacuum state data exists.
            if current_vacuum_state not in job_result[0].data.meas.get_counts().keys():
                df_job_results.loc[sim_time,current_vacuum_state] = 0
                df_job_results.loc[sim_time,"vacuum_persistence_probability"] = 0
            else:
                df_job_results.loc[sim_time,"vacuum_persistence_probability"] = df_job_results.loc[sim_time,current_vacuum_state] / n_shots

fig, ax = plt.subplots()
df_job_results.plot(y = "vacuum_persistence_probability",ax=ax)
plt.title(f'Vacuum persistence probability ({linear_size} plaquettes)')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()




