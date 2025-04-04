from ymcirc.lattice_parser import ParsedData
import matplotlib.pyplot as plt
import matplotlib
from ymcirc.conventions import (ONE,THREE,THREE_BAR)
import numpy as np


parent_directory_path1 = "/home/drishti/PhD/ymcirc/my_work/2x2_wancillas_largest_elements/"

def path_to_file1(timestep,scale):
    file_name = f"2x2_wancillas_largest_elements-dt={timestep}-scale={scale}_H2-1E.json"
    return file_name

def path_to_file2(timestep,scale):
    file_name = f"2x1_T1_wancillas_15q.pkl-dt={timestep}-scale={scale}_H1-1E.json"
    return file_name

def link_electric_energy(irrep):
    if irrep == ONE or "":
        electric_energy = 0
    elif irrep == THREE or THREE_BAR:
        electric_energy = 4/3
    return electric_energy

timesteps = [0.0,0.125,0.25,0.375,0.5]
# timesteps = [0.0]
scales = [0.01,0.1]

electric_energies_unphysical_links_removed_diff_scales = []
electric_energies_ulr_error_diff_scales = []
electric_error_bars_diff_scales = []
corrected_electric_energies_diff_scales = []
vacuum_persistence_diff_scales = []
total_physical_hits_diff_scales = []

for scale in scales:
    corrected_electric_energies_diff_time = []
    electric_energy_unphysical_links_removed_diff_time = []
    electric_error_bars_diff_time = []
    vacuum_persistence_diff_time = []
    total_physical_hits_diff_time = []
    electric_energies_ulr_error_diff_times = []
    vacuum_count = 0
    for timestep in timesteps:
        file_name = path_to_file1(timestep,scale)
        filepath = parent_directory_path1 + file_name
        parsed_instance = ParsedData(dim="2",truncation="T1",lattice_size=2,filepath=filepath)
        parsed_data = parsed_instance.parser(ancillas= 7, site_qubits= 4)
        physical_electric_energy = 0
        electric_energy = 0
        vacuum_persistence = 0
        total_hits = 0
        total_physical_hits = 0
        error_in_electric_energy = 0
        error_in_electric_energy_ulr = 0
        max_hits = 0
        for bitstring,states_dict in parsed_data.items():
            hits = states_dict["hits"]
            site_bitstring = states_dict["site_bitstring"]
            del states_dict["hits"]
            del states_dict["site_bitstring"]
            state_electric_energy = 0
            if bitstring == "0" * 16:
                vacuum_persistence += hits
            for irrep in states_dict.values():
                if irrep == "":
                    state_electric_energy = 0
                    hits = 0
                    break
                else:
                    state_electric_energy += link_electric_energy(irrep)
            total_hits += hits
            electric_energy += state_electric_energy*hits
            error_in_electric_energy_ulr += state_electric_energy**2 * hits
            if parsed_instance.vertex_check(states_dict,site_bitstring) == True:
                total_physical_hits += hits
                physical_electric_energy += state_electric_energy*hits
                error_in_electric_energy += state_electric_energy**2 * hits
            # elif parsed_instance.correct_unphysical_links(states_dict)[0] == True:
            #     states_dict = parsed_instance.correct_unphysical_links(states_dict)[1]
            #     if parsed_instance.vertex_check(states_dict) == True:
            #         total_hits += hits
            #         for irrep in states_dict.values():
            #             state_electric_energy += link_electric_energy(irrep)
            #         electric_energy += state_electric_energy*hits
        total_physical_hits_diff_time.append(total_physical_hits)
        electric_energy = electric_energy/(8*total_hits)
        error_in_electric_energy =  1/(8*total_physical_hits) * np.sqrt(error_in_electric_energy)
        error_in_electric_energy_ulr = 1/(8*total_hits) * np.sqrt(error_in_electric_energy_ulr)
        physical_electric_energy = physical_electric_energy/(8*total_physical_hits)
        vacuum_persistence = vacuum_persistence/total_physical_hits
        electric_energy_unphysical_links_removed_diff_time.append(electric_energy)
        electric_energies_ulr_error_diff_times.append(error_in_electric_energy_ulr)
        vacuum_persistence_diff_time.append(vacuum_persistence)
        corrected_electric_energies_diff_time.append(physical_electric_energy)
        electric_error_bars_diff_time.append(error_in_electric_energy)
    electric_energies_unphysical_links_removed_diff_scales.append(electric_energy_unphysical_links_removed_diff_time)
    vacuum_persistence_diff_scales.append(vacuum_persistence_diff_time)
    total_physical_hits_diff_scales.append(total_physical_hits_diff_time)
    corrected_electric_energies_diff_scales.append(corrected_electric_energies_diff_time)
    electric_error_bars_diff_scales.append(electric_error_bars_diff_time)
    electric_energies_ulr_error_diff_scales.append(electric_energies_ulr_error_diff_times)

comparison_data_electric_energy = []
comparison_data_vacuum_persistence = []
comparison_data_total_hits = []

# for timestep in timesteps:
#     comparison_data_file = parent_directory_path2 + path_to_file2(timestep,0.05)
#     parsed_instance = ParsedData(dim="3/2",truncation="T1",lattice_size=2,filepath=comparison_data_file)
#     parsed_data = ParsedData(dim="3/2",truncation="T1",lattice_size=2,filepath=comparison_data_file).parser(ancillas=3)
#     electric_energy = 0
#     vacuum_persistence = 0
#     total_hits = 0
#     for bitstring,states_dict in parsed_data.items():
#         hits = states_dict["hits"]
#         del states_dict["hits"]
#         state_electric_energy = 0
#         if bitstring == "0" * 12:
#             vacuum_persistence += hits
#         if parsed_instance.vertex_check(states_dict) == True:
#             total_hits += hits
#             for irrep in states_dict.values():
#                 state_electric_energy += link_electric_energy(irrep)
#             electric_energy += state_electric_energy*hits
#     electric_energy = electric_energy/(2*total_hits)
#     vacuum_persistence = vacuum_persistence/total_hits
#     comparison_data_total_hits.append(total_hits)
#     comparison_data_electric_energy.append(electric_energy)
#     comparison_data_vacuum_persistence.append(vacuum_persistence)

# print(total_physical_hits_diff_scales)


# for electric_energies_time in electric_energies_diff_scales:
#     plt.plot(timesteps,electric_energies_time)

#plt.plot(timesteps,electric_energies_unphysical_links_removed_diff_scales[1],linestyle='dashed',color='orange')4
matplotlib.rcParams.update({'font.size': 13})
for i in range(len(corrected_electric_energies_diff_scales)):
    plt.errorbar(timesteps,corrected_electric_energies_diff_scales[i],yerr = electric_error_bars_diff_scales[i],fmt = 'o-',capsize=5, linestyle = 'dashed')
plt.plot([0.0,0.1,0.2,0.3,0.4,0.5],[0,0.012866666666666665, 0.05433333333333333, 0.11086666666666664, 0.19026666666666686, 0.2873000000000002],\
         color = 'black',linewidth = 1.7)
plt.errorbar(timesteps,electric_energies_unphysical_links_removed_diff_scales[1],yerr=electric_energies_ulr_error_diff_scales[1], \
             fmt='o-',linestyle = 'dotted',capsize=5)
# plt.plot(timesteps,comparison_data_electric_energy,linestyle = 'dashed',linewidth=2)
plt.legend(["0%","1%","10%","10% with no" + "\n" + "Gauss corr."],title = "Noise Scales")
plt.xlabel("time")
plt.ylabel(r"$ \langle E^2 \rangle $")
# plt.show()
plt.savefig("./my_work/quantinuum_sim_results.pdf")


# for vacuum_persistence_time in vacuum_persistence_diff_scales:
#     plt.plot(timesteps,vacuum_persistence_time)
# plt.legend(scales)
# plt.xlabel("timestep")
# plt.ylabel("Vacuum Persistence")
# plt.show()

# for total_hits_time in total_physical_hits_diff_scales:
#     plt.plot(timesteps,total_hits_time)
# plt.legend(scales)
# plt.xlabel("timesteps")
# plt.ylabel("Total Physical Hits")
# plt.show()