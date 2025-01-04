# 1) Read Quantinuum Data

import json
from lattice_tools.lattice_registers import LatticeRegisters
from lattice_tools.conventions import (
    VERTEX_SINGLET_BITMAPS,
    IRREP_TRUNCATION_DICT_1_3_3BAR,
    IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8)


#create a class that stores all the parsed json data
class ParsedData:
    def __init__(self,
                dim:str,
                truncation:str,
                lattice_size:int,
                filepath:str):
        self.dim = dim
        self.truncation = truncation
        self.size = lattice_size
        self.rawdata = filepath
        #Read the json file
        json_file = filepath
        with open(json_file) as json_data:
            data = json.load(json_data)
        #import link truncation and vertex dictionaries
        if dim == "3/2" and truncation == "T1":
            link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR
            vertex_bitmap = {}
        elif dim == "3/2" and truncation == "T2":
            link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8
            vertex_bitmap = VERTEX_SINGLET_BITMAPS["d=3/2, T2"]
        elif dim == "2" and truncation == "T1":
            link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR
            vertex_bitmap = VERTEX_SINGLET_BITMAPS["d=2, T1"]
        elif dim == "2" and truncation == "T2":
            link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8
            vertex_bitmap = VERTEX_SINGLET_BITMAPS["d=2, T2"]
        self._data = data
        self._vertex_bitmap = vertex_bitmap
        self._link_bitmap = link_bitmap
    
    def parser(self):
        lattice_size= self.size
        dim = self.dim
        vertex_bitmap = self._vertex_bitmap
        link_bitmap = self._link_bitmap
        data = self._data
        parsed_dict = {}
        if vertex_bitmap != {}:
            nqubits_vertex = len(list(vertex_bitmap.values())[0])
        else:
            nqubits_vertex = 0
        nqubits_link = len(list(link_bitmap.values())[0])
        for dat_bitstring, hits in data.items():
            og_dat_bitstring = dat_bitstring
            parsed_dict_entry = {}
            for i in range(lattice_size):
                for j in range(lattice_size):
                    # For dim = 3/2, we must be careful in the y direction.
                    go_in_y_dir = True
                    if dim == "3/2":
                        if j==1:
                            go_in_y_dir = False
                        elif j>1:
                            break
                    # Obtaining the vertex and link qubit values
                    if nqubits_vertex != 0:
                        vertex_bits = dat_bitstring[:nqubits_vertex]
                        dat_bitstring = dat_bitstring[nqubits_vertex:]
                    link_bits_xdir = dat_bitstring[:nqubits_link]
                    if go_in_y_dir == True:
                        link_bits_ydir = dat_bitstring[nqubits_link:2*nqubits_link]
                        dat_bitstring = dat_bitstring[2*nqubits_link:]
                    else:
                        dat_bitstring = dat_bitstring[nqubits_link:]
                    # Matching the vertex and link qubit values with the bitmap dictionaries
                    if nqubits_vertex != 0 :
                        physical_vertex_found = False
                        for vertex_key, ref_bitstring_vertex in vertex_bitmap.items():
                            if ref_bitstring_vertex == vertex_bits:
                                physical_vertex_found = True
                                parsed_dict_entry[(i,j)] = vertex_key
                                break
                        if physical_vertex_found == False:
                            parsed_dict_entry[(i,j)] = ""
                    physical_link_xdir_found = False
                    for link_key, ref_bitstring_link in link_bitmap.items():
                        if ref_bitstring_link == link_bits_xdir:
                            physical_link_xdir_found = True
                            parsed_dict_entry[((i,j),1)] = link_key
                            break
                    if physical_link_xdir_found == False:
                        parsed_dict_entry[((i,j),1)] = ""
                    if go_in_y_dir == True:
                        physical_link_ydir_found = False
                        for link_key, ref_bitstring_link in link_bitmap.items():
                            if ref_bitstring_link == link_bits_ydir:
                                physical_link_ydir_found = True
                                parsed_dict_entry[((i,j),2)] = link_key
                                break
                        if physical_link_ydir_found == False:
                            parsed_dict_entry[((i,j),2)] = ""
            parsed_dict_entry["hits"] = hits
            parsed_dict[og_dat_bitstring] = parsed_dict_entry
        return parsed_dict

quant_run = ParsedData("3/2","T1",2,"n_trotter=1-t=0.0_H1-1.json")
print(quant_run.parser()['110111001101'])



                            






    




    
    


    


        
        



