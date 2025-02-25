# 1) Read Quantinuum Data

import json
from lattice_tools.lattice_registers import LatticeRegisters
from lattice_tools.conventions import (
    VERTEX_SINGLET_BITMAPS,
    IRREP_TRUNCATION_DICT_1_3_3BAR,
    IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8,
    ONE,THREE,THREE_BAR,SIX,SIX_BAR,EIGHT)
from labellines import labelLines
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
import base64


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

    def vertex_check(self,parsed_entry):
        dim = self.dim
        size = self.size
        vertex_bitmap = self._vertex_bitmap
        vertex_check_values = {}
        if vertex_bitmap == {}:
            return {}
        else:
            if dim == "3/2":
                for i in range(size):
                    for j in range(2):
                        vertexbag = {parsed_entry[(i,j)][k] for k in range(3)}
                        if j==0:
                            incoming_links = {
                                parsed_entry[((i,j),1)],
                                parsed_entry[((i,j),2)],
                                parsed_entry[(((i-1)%size,j),1)]
                            }
                        elif j==1 :
                            incoming_links = {
                                parsed_entry[((i,j),1)],
                                parsed_entry[(((i-1)%size,j),1)],
                                parsed_entry[((i,j-1),2)]
                            }
                        if vertexbag == incoming_links:
                            vertex_check_values[(i,j)] = "Correct"
                        else:
                            vertex_check_values[(i,j)] = "Corrupted"
            if dim == "2":
                for i in range(size):
                    for j in range(size):
                        vertexbag = {parsed_entry[(i,j)][k] for k in range(4)}
                        incoming_links = {
                            parsed_entry[((i,j),1)],
                            parsed_entry[((i,j),2)],
                            parsed_entry[(((i-1)%size,j),1)],
                            parsed_entry[((i,(j-1)%size),2)]
                        }
                    if vertexbag == incoming_links:
                        vertex_check_values[(i,j)] = "Correct"
                    else:
                        vertex_check_values[(i,j)] = "Corrupted"
            return vertex_check_values

    def lattice_images(self):
        dim = self.dim
        size = self.size
        parseddata = self.parser()
        image_dict = {}
        #dictionary of colors
        color_dict = {}
        color_dict[ONE] = "black"
        color_dict[THREE] = "blue"
        color_dict[THREE_BAR] = "indigo"
        color_dict[SIX] = "cyan"
        color_dict[SIX_BAR] = "teal"
        color_dict[EIGHT] = "seagreen"
        color_dict[''] = "red" #Unphysical links are given red color
        #dictionary of link states
        link_state = {}
        link_state[ONE] = "1"
        link_state[THREE] = "3"
        link_state[THREE_BAR] = "$\\bar{3}$"
        link_state[SIX] = "6"
        link_state[SIX_BAR] = "$\\bar{6}$"
        link_state[EIGHT] = "8"
        link_state[''] = "U"
        #Construction of the lattice images
        if dim == "3/2":
            yiter = 2
        elif dim == "2":
            yiter = size
        for key, parsed_item in parseddata.items():
            vertex_check = self.vertex_check(parsed_item)
            check_vertices = False
            if vertex_check != {}:
                check_vertices = True
            plt.axis('off')
            ax = plt.gca()
            ax.set_aspect('equal',adjustable='box')
            xval=[]
            for i in range(0,size):
                for j in range(0,yiter):
                    if check_vertices == True:
                        if vertex_check[(i,j)] == "Correct":
                            plt.scatter(i,j,color='black',marker='o')
                        elif vertex_check[(i,j)] == "Corrupted":
                            plt.scatter(i,j,color = 'red',marker='o')
                    elif check_vertices == False:
                        plt.scatter(i,j,color='black',marker='o')
                    if dim == "3/2":
                        if j==0:
                            link1 = parsed_item[((i,j),1)]
                            link2 = parsed_item[((i,j),2)]
                            plt.plot([i,i+1],[j,j],color = color_dict[link1],label = link_state[link1])
                            xval.append(i+0.5)
                            plt.plot([i,i],[j,j+1],color = color_dict[link2],label = link_state[link2])
                            xval.append(i)
                        if j==1 :
                            link1 = parsed_item[((i,j),1)]
                            plt.plot([i,i+1],[j,j],color = color_dict[link1], label = link_state[link1])
                            xval.append(i+0.5)
                    elif dim == "2":
                        link1 = parsed_item[((i,j),1)]
                        link2 = parsed_item[((i,j),2)]
                        plt.plot([i,i+1],[j,j],color = color_dict[link1],label = link_state[link1])
                        xval.append(i+0.5)
                        plt.plot([i,i],[j,j+1],color = color_dict[link2],label = link_state[link2])
                        xval.append(i)
            lines=ax.get_lines()
            labelLines(lines,ha="center",xvals=xval,align=False)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            image_dict[key] = image_base64
        return image_dict
    
    def lattice_visualizer(self,bitstring):
        image_dict = self.lattice_images()
        image_data = base64.b64decode(image_dict[bitstring])
        image = mpimg.imread(io.BytesIO(image_data), format='PNG')
        plt.imshow(image)
        plt.axis('off')
        plt.show()

quant_run = ParsedData("3/2","T1",2,"n_trotter=1-t=0.1_H1-1E.json")
#print(quant_run.parser()['110111001101'])
image = quant_run.lattice_visualizer('000100011010') 




                            






    




    
    


    


        
        



