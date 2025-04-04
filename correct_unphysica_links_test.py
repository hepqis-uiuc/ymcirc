from ymcirc.lattice_parser import ParsedData

# file = "/home/drishti/PhD/ymcirc/my_work/2x1_T1_wancillas_15q_smaller_scales/2x1_T1_wancillas_15q.pkl-dt=0.0-scale=0.25_H1-1E.json"

# parsed_instance = ParsedData(dim="3/2",truncation="T1",lattice_size=2,filepath=file)
# parsed_data = parsed_instance.parser(ancillas=3)

# for bitstring, state_dict in parsed_data.items():
#     del state_dict["hits"]
#     correction_bool = parsed_instance.correct_unphysical_links(state_dict)[0]
#     if correction_bool == True:
#         print("yes")

file = "/home/drishti/PhD/ymcirc/my_work/2x2_wancillas_largest_elements/2x2_wancillas_largest_elements-dt=0.0-scale=0.01_H2-1E.json"

parsed_instance = ParsedData(dim="2",truncation="T1",lattice_size=2,filepath=file)
test_entry ={}
test_entry[((0,0),1)] = (0,0,0)
test_entry[((0,0),2)] = (1,0,0)
test_entry[((0,1),1)] = (1,0,0)
test_entry[((0,1),2)] = (0,0,0)
test_entry[((1,0),1)] = (1,0,0)
test_entry[((1,0),2)] = (0,0,0)
test_entry[((1,1),1)] = (0,0,0)
test_entry[((1,1),2)] = (1,0,0)
print(parsed_instance.vertex_check(test_entry,"0000"))