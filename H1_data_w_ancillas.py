from ymcirc.lattice_parser import ParsedData
import matplotlib.pyplot as plt
from ymcirc.conventions import (ONE,THREE,THREE_BAR)

parent_directory_path = "/home/drishti/PhD/ymcirc/my_work/2x1_T1_wancillas_15q_smaller_scales/"

def path_to_file(timestep,scale):
    file_name = f"2x1_T1_wancillas_15q.pkl-dt={timestep}-scale={scale}_H1-1E.json"
    return file_name

def link_electric_energy(irrep):
    if irrep == ONE:
        electric_energy = 0
    elif irrep == THREE or THREE_BAR:
        electric_energy = 4/3
    return electric_energy

timesteps = [0.0,0.25,0.5,0.75,1.0]
scales = [x/100 for x in range(0,10)]

electric_energies_diff_scales = []
vacuum_persistence_diff_scales = []
total_physical_hits_diff_scales = []

for scale in scales:
    electric_energy_diff_time = []
    vacuum_persistence_diff_time = []
    total_physical_hits_diff_time = []
    vacuum_count = 0
    for timestep in timesteps:
        file_name = path_to_file(timestep,scale)
        filepath = parent_directory_path + file_name
        parsed_instance = ParsedData(dim="3/2",truncation="T1",lattice_size=2,filepath=filepath)
        parsed_data = ParsedData(dim="3/2",truncation="T1",lattice_size=2,filepath=filepath).parser(ancillas=3)
        electric_energy = 0
        vacuum_persistence = 0
        total_hits = 0
        for bitstring,states_dict in parsed_data.items():
            hits = states_dict["hits"]
            del states_dict["hits"]
            state_electric_energy = 0
            if timestep == 0.0:
                non_vacuum_bins = []
                non_vacuum_hits = []
                if bitstring != "0" * 12:
                    non_vacuum_bins.append(bitstring)
                    non_vacuum_hits.append(hits)
                vacuum_persistence += hits
                total_hits += hits
            else:
                if bitstring == "0" * 12:
                    vacuum_persistence+= hits
                else: 
                    for i in range(len(non_vacuum_bins)):
                        non_vacuum_bitstring = non_vacuum_bins[i]
                        non_vacuum_hit = non_vacuum_hits[i]
                        if bitstring == non_vacuum_bitstring:
                            vacuum_persistence += non_vacuum_hit
                            total_hits += non_vacuum_hit
                            hits = hits - non_vacuum_hit 
                if parsed_instance.vertex_check(states_dict) == True:
                    total_hits += hits
                    for irrep in states_dict.values():
                        state_electric_energy += link_electric_energy(irrep)
                    electric_energy += state_electric_energy*hits
                # elif parsed_instance.correct_unphysical_links(states_dict)[0] == True:
                #     states_dict = parsed_instance.correct_unphysical_links(states_dict)[1]
                #     if parsed_instance.vertex_check(states_dict) == True:
                #         total_hits += hits
                #         for irrep in states_dict.values():
                #             state_electric_energy += link_electric_energy(irrep)
                #         electric_energy += state_electric_energy*hits
        total_physical_hits_diff_time.append(total_hits)
        electric_energy = electric_energy/(2*total_hits)
        vacuum_persistence = vacuum_persistence/total_hits
        electric_energy_diff_time.append(electric_energy)
        vacuum_persistence_diff_time.append(vacuum_persistence)
    electric_energies_diff_scales.append(electric_energy_diff_time)
    vacuum_persistence_diff_scales.append(vacuum_persistence_diff_time)
    total_physical_hits_diff_scales.append(total_physical_hits_diff_time)

for electric_energies_time in electric_energies_diff_scales:
    plt.plot(timesteps,electric_energies_time)
plt.legend(scales)
plt.xlabel("timesteps")
plt.ylabel("Electric Energy")
plt.show()


# for vacuum_persistence_time in vacuum_persistence_diff_scales:
#     plt.plot(timesteps,vacuum_persistence_time)
# plt.legend(scales)
# plt.xlabel("timestep")
# plt.ylabel("Vacuum Persistence")
# plt.show()


# print(total_physical_hits_diff_scales)

for total_hits_time in total_physical_hits_diff_scales:
    plt.plot(timesteps,total_hits_time)
plt.legend(scales)
plt.xlabel("timesteps")
plt.ylabel("Total Physical Hits")
plt.show()

