from ymcirc.lattice_parser import ParsedData
import matplotlib.pyplot as plt
from ymcirc.conventions import (ONE,THREE,THREE_BAR)

parent_directory_path = "/home/drishti/PhD/ymcirc/d=3_2-T1-3x2-largest_elements/"

def path_to_file(timestep,scale):
    file_name = f"d=3_2-T1-3x2-largest_elements-dt={timestep}-scale={scale}_H1-Emulator.json"
    return file_name

def link_electric_energy(irrep):
    if irrep == ONE:
        electric_energy = 0
    elif irrep == THREE or THREE_BAR:
        electric_energy = 4/3
    return electric_energy

timesteps = [0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]
scales = [0.0,0.1,0.25,1.0]

electric_energies_diff_scales = []
vacuum_persistence_diff_scales = []

for scale in scales:
    electric_energy_diff_time = []
    vacuum_persistence_diff_time = []
    for timestep in timesteps:
        file_name = path_to_file(timestep,scale)
        filepath = parent_directory_path + file_name
        parsed_data = ParsedData(dim="3/2",truncation="T1",lattice_size=3,filepath=filepath).parser()
        electric_energy = 0
        total_hits = 0
        for bitstring,states_dict in parsed_data.items():
            hits = states_dict["hits"]
            if bitstring == "0" * 18:
                vacuum_persistence = hits
            del states_dict["hits"]
            total_hits += hits
            for irrep in states_dict.values():
                electric_energy += link_electric_energy(irrep)*hits
        electric_energy = electric_energy/(2*total_hits)
        vacuum_persistence = vacuum_persistence/total_hits
        electric_energy_diff_time.append(electric_energy)
        vacuum_persistence_diff_time.append(vacuum_persistence)
    electric_energies_diff_scales.append(electric_energy_diff_time)
    vacuum_persistence_diff_scales.append(vacuum_persistence_diff_time)

# print(electric_energies_diff_scales)


# for electric_energies_time in electric_energies_diff_scales:
#     plt.plot(timesteps,electric_energies_time)
# plt.legend(scales)
# plt.xlabel("timesteps")
# plt.ylabel("Electric Energy")
# plt.show()

for vacuum_persistence_time in vacuum_persistence_diff_scales:
    plt.plot(timesteps,vacuum_persistence_time)
plt.legend(scales)
plt.xlabel("timestep")
plt.ylabel("Vacuum Persistence")
plt.show()
        
        


        


