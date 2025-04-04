import json
import matplotlib.pyplot as plt

parent_dir = "/home/drishti/PhD/ymcirc/my_work/d=3_2-T1-3x2-largest_elements_smaller_scales/"

def file_path(timestep,scale):
    # print(parent_dir + f"2x1_T1_wancillas_15q.pkl-dt={timestep}-scale={scale}_H1-1E.json")
    return parent_dir + f"d=3_2-T1-3x2-largest_elements-dt={timestep}-scale={scale}_H1-1E.json"

def json_file_data(json_file):
    with open(json_file) as json_data:
        data = json.load(json_data)
    return data


def bitstring_sans_ancilla_func(data):
    bitstring_sans_ancilla = {}

    for bitstring,hits in data.items():
        bitstring = bitstring[:-3]
        if bitstring in bitstring_sans_ancilla.keys():
            bitstring_sans_ancilla[bitstring] += hits
        else:
            bitstring_sans_ancilla[bitstring] = hits

    return bitstring_sans_ancilla


def binary_to_gray(binary_str):
    """Convert a binary string to its Gray code representation."""
    gray = []
    gray.append(binary_str[0])  # The first bit is the same
    for i in range(1, len(binary_str)):
        # XOR the current bit with the previous bit
        gray_bit = str(int(binary_str[i]) ^ int(binary_str[i - 1]))
        gray.append(gray_bit)
    return ''.join(gray)

def gray_order_bitstrings(bitstring_hit_dict):
    """Sort a list of bitstrings in Gray code order without generating the full sequence."""
    # Compute the Gray code for each bitstring and sort based on it
    sorted_bitstrings = dict(sorted(bitstring_hit_dict.items(), key=lambda x: binary_to_gray(x[0])))
    return sorted_bitstrings

histograms_for_diff_scales = []

scales = [x/100 for x in range(1,10)]

# for scale in scales:
#     filepath = file_path(0.0,scale)
#     data = json_file_data(filepath)
#     bitstring_sans_ancilla = bitstring_sans_ancilla_func(data)
#     gray_ordered_bitstring_list = [x for x in gray_order_bitstrings(bitstring_sans_ancilla).keys()]
#     gray_ordered_hits_list = [x for x in gray_order_bitstrings(bitstring_sans_ancilla).values()]
#     histograms_for_diff_scales.append((gray_ordered_bitstring_list,gray_ordered_hits_list))

for scale in scales:
    filepath = file_path(0.0,scale)
    data = json_file_data(filepath)
    gray_order_bitstring_list = [x for x in gray_order_bitstrings(data).keys()]
    gray_order_hits_list = [x for x in gray_order_bitstrings(data).values()]
    histograms_for_diff_scales.append((gray_order_bitstring_list,gray_order_hits_list))

# my_xticks = []
i = 0
for hist_bitstring, hist_hits in histograms_for_diff_scales: 
    plt.hist(hist_bitstring,weights=hist_hits,bins=len(hist_bitstring))
    max_element1 = hist_bitstring[hist_hits.index(max(hist_hits))]
    hist_hits.remove(max(hist_hits))
    max_element2 = hist_bitstring[hist_hits.index(max(hist_hits))]
    hist_hits.remove(max(hist_hits))
    #max_element3 = hist_bitstring[hist_hits.index(max(hist_hits))]
    plt.xticks([max_element1,max_element2])
    plt.title(f"scale= {scales[i]}")
    # plt.savefig(f"/home/drishti/PhD/ymcirc/my_work/gray_code_ordered_hist_2x1_wancillas_scale-{scales[i]}.png")
    plt.show()
    plt.close()
    i += 1
    #my_xticks.append(max_element1)
    
# plt.xticks(my_xticks,rotation = 'vertical')
#plt.xticks([])
#plt.legend(scales)
#plt.show()
# plt.savefig("/home/drishti/PhD/ymcirc/my_work/gray_code_ordered_hist_2x1_wancillas.pdf")



    



