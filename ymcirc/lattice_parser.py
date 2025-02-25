"""
Data structures for interacting with measurement outcomes on lattices.

Currently a work in progress.
"""
import json
import copy
import csv
import pickle
from ymcirc.conventions import (
    VERTEX_SINGLET_BITMAPS,
    IRREP_TRUNCATION_DICT_1_3_3BAR,
    IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8,
    ONE, THREE, THREE_BAR, SIX, SIX_BAR, EIGHT)
from labellines import labelLines
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import io
import base64


class ParsedData:
    """
    This class can be used to parse raw bitstrings from simulations, store the parsed data, and generate lattice images corresponding to the bitstrings.
    """
    def __init__(self,
                dim:str,
                truncation:str,
                lattice_size:int,
                filepath:str):
        """
        To create an instance of ParsedData, input the following:
        1)the dimension of the lattice ("3/2","2", etc.)
        2) the irrep truncation ("T1","T2", etc.)
        3) lattice size (i.e., number of vertices)
        4) the filepath where the raw json data is stored.

        These inputs can be accessed through the attributes dim, trunctaion, size, and rawdata respectively.
        """
        self.dim = dim
        self.truncation = truncation
        self.size = lattice_size
        self.rawdata = filepath
        # Read the json file
        json_file = filepath
        with open(json_file) as json_data:
            data = json.load(json_data)
        # import link truncation and vertex dictionaries
        # TO DO: Replace these cumbersome definitions with
        # LatticeRegisters class.
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
        # Dealing with a case when the input is neither d=2 or d=3/2.
        elif dim != "3/2" and dim != "2":
            raise ValueError("The class ParsedData can only deal with dim=3/2 or dim=2 for now")
        self._data = data
        self._vertex_bitmap = vertex_bitmap
        self._link_bitmap = link_bitmap
        self.pickled_dict = ""
        self.image = ""
    
    def parser(self,reverse=False,print_to_file=False,ancillas: int = None):
        """ 
        This method parses the raw bitstrings. An optional argument "reverse" can be passed in order to deal with endianess issues.

        The output of this method is a nested dictionary. The outer nest has a bitstring as a key and its corresponding entry is another dictionary
        which contains the parsed form of the bitstring. The keys of this inner nest dictionary are either links or vertices and their 
        corresponding entry is the state of the link/vertex. For example, a d=3/2 2 plaquette lattice in the T1 irrep truncation
        with the bitstring key "001000100101" will have as its entry the dictionary 

        {((0,0),1):(0, 0, 0), ((0,0),2):(1, 0, 0), ((0,1),1):(0, 0, 0), ((1,0),1):(1, 0, 0), ((1,0),2):(1, 1, 0), ((1,1),1):(1, 1, 0), "hits": 14}

        The "hits" key refers to the number of times the bitstring was obtained in the shots data.

        If the optional argument "print_to_file" is set to be True, then the parsed data will be printed to a file, the location of which
        is printed in the terminal.
 
         """
        lattice_size= self.size
        dim = self.dim
        vertex_bitmap = self._vertex_bitmap
        link_bitmap = self._link_bitmap
        data = self._data
        filepath = self.rawdata
        parsed_dict = {}
        if vertex_bitmap != {}:
            nqubits_vertex = len(list(vertex_bitmap.values())[0])
        else:
            nqubits_vertex = 0
        nqubits_link = len(list(link_bitmap.values())[0])
        for dat_bitstring, hits in data.items():
            if reverse==True:
                dat_bitstring = dat_bitstring[::-1]
            if ancillas != None:
                dat_bitstring = dat_bitstring[:-ancillas]
            og_dat_bitstring = dat_bitstring
            if og_dat_bitstring in parsed_dict.keys():
                parsed_dict[og_dat_bitstring]["hits"] += hits
            else:
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
        parsed_filepath = filepath + "-parsed"
        #print(parsed_dict) to file if print_to_file is True
        if print_to_file == True:
            with open(parsed_filepath,"w") as fp:
                json.dump(parsed_dict,fp)
            print("Parsed bitstrings stored at location  "+parsed_filepath)
        return parsed_dict

    def vertex_check(self,parsed_entry):
        """ This method checks if the input bitstring has consistent vertex information. """
        dim = self.dim
        size = self.size
        physical_vertex_gauge_states = self._vertex_gauge_csv_to_list()
        if dim == "3/2":
            physical_vertex_count = 0
            for i in range(size):
                for j in range(2):
                    vertex_links = self._gather_vertex_links((i,j))
                    vertex_link_state  = []
                    for link, orientation in vertex_links:
                        vertex_link_state.append(parsed_entry[link])
                    vertex_gauge_state = self._get_vertex_gauge_state(vertex_links,vertex_link_state)
                    for physical_vertex_state,singlet in physical_vertex_gauge_states:
                        if physical_vertex_state == vertex_gauge_state:
                            physical_vertex_count += 1
            if physical_vertex_count ==2*size :
                return True
            else:
                return False
    
    def correct_unphysical_links(self,parsed_entry):
        unphysical_links = []
        physical_vertex_states = self._vertex_gauge_csv_to_list()
        for link, state in parsed_entry.items():
            if state == "":
                unphysical_links.append(link)
        corrected_unphysical_states = []
        for unphysical_link in unphysical_links:
            vertex1,vertex2 = self._vertices_connected_to_link(unphysical_link)
            vertex_state1 = self._gather_vertex_links(vertex1)
            vertex_state2= self._gather_vertex_links(vertex2)
            for vertex_link, orientation in vertex_state1:
                if vertex_link != unphysical_link:
                    if parsed_entry[vertex_link] == "":
                        return (False,None)
                if vertex_link == unphysical_link:
                    unphysical_state_idx1 = vertex_state1.index([unphysical_link,orientation])
                    unphysical_state_orientation1 = orientation
            for vertex_link, orientation in vertex_state2:
                if vertex_link != unphysical_link:
                    if parsed_entry[vertex_link] == "":
                        return (False,None)
                if vertex_link == unphysical_link:
                    unphysical_state_idx2 = vertex_state2.index([unphysical_link,orientation])
                    unphysical_state_orientation2 = orientation
            vertex_link_state1 = []
            vertex_link_state2 = []
            for link,orientation in vertex_state1:
                vertex_link_state1.append(parsed_entry[link])
            for link,orientation in vertex_state2:
                vertex_link_state2.append(parsed_entry[link])
            vertex1_gauge_state = self._get_vertex_gauge_state(vertex_state1,vertex_link_state1)
            vertex2_gauge_state = self._get_vertex_gauge_state(vertex_state2,vertex_link_state2)
            for physical_state, singlet in physical_vertex_states:
                physical_link_at_unphysical_site =physical_state[unphysical_state_idx1]
                comparison_physical_state = copy.deepcopy(physical_state)
                comparison_physical_state[unphysical_state_idx1] = ""
                if comparison_physical_state == vertex1_gauge_state:
                    unphysical_link_possibility1 = physical_link_at_unphysical_site
            for physical_state, singlet in physical_vertex_states:
                physical_link_at_unphysical_site =physical_state[unphysical_state_idx2]
                comparison_physical_state = copy.deepcopy(physical_state)
                comparison_physical_state[unphysical_state_idx2] = ""
                # print(physical_state)
                # print(comparison_physical_state)
                if comparison_physical_state == vertex2_gauge_state:
                    unphysical_link_possibility2 = physical_link_at_unphysical_site
            if self._determine_link_orientation(unphysical_state_orientation1,unphysical_link_possibility1) \
                == self._determine_link_orientation(unphysical_state_orientation2,unphysical_link_possibility2):
                if unphysical_state_orientation1 == 1:
                    corrected_unphysical_states.append(unphysical_link_possibility1)
                elif unphysical_state_orientation2 == 1:
                    corrected_unphysical_states.append(unphysical_link_possibility2)
        if len(unphysical_links) == len(corrected_unphysical_states):
            for i in range(len(unphysical_links)):
                unphysical_link = unphysical_links[i]
                corrected_state = corrected_unphysical_states[i]
                parsed_entry[unphysical_link] = corrected_state
            return (True,parsed_entry)
        elif len(unphysical_links) != len(corrected_unphysical_states):
            return (False,None)
            

    def _vertices_connected_to_link(self,link):
        # get the vertices connected to the links
        size = self.size
        vertex1 = link[0]
        if link[1] == 1:
            if vertex1[0] != size-1:
                vertex2 = (vertex1[0]+1,vertex1[1]) 
            elif vertex1[0] == size-1:
                vertex2 = (0,vertex1[1])
        elif link[1] == 2:
            vertex2 = (vertex1[0],vertex1[1]+1)
        return (vertex1,vertex2)

                        
    def _convert_mathematica_state_to_py_state(self,math_state:str):
        if math_state == "1":
            return ONE
        elif math_state ==  "3":
            return THREE
        elif math_state == "-3":
            return THREE_BAR
              
    def _vertex_gauge_csv_to_list(self):
        # dim = self.dim
        # truncation = self.truncation
        csv_file = "/home/drishti/PhD/ymcirc/ymcirc/ymcirc_data/vertex-gauss-law/vertex_gauss_law_d=1.5_irrep=T1.csv"
        with open(csv_file,'r') as file:
            vertex_gauge_data = list(csv.reader(file))
        new_vertex_gauge_data = []
        for link_states, singlet in vertex_gauge_data:
            link_states = link_states.strip("{}")
            link_states = link_states.split(", ")
            singlet = singlet.strip("{}")
            py_link_states = []
            for math_links in link_states:
                py_link = self._convert_mathematica_state_to_py_state(math_links)
                py_link_states.append(py_link)
            new_vertex_gauge_data.append([py_link_states,singlet])
        return new_vertex_gauge_data

    def _gather_vertex_links(self,vertex):
        size = self.size
        i = vertex[0]
        j = vertex[1]
        if j==0 :
            y_link = [((i,0),2),1]
            right_link = [((i,0),1),1]
            if i != 0:
                left_link = [((i-1,0),1),-1]
            elif i == 0:
                left_link = [((size-1,0),1),-1]
        elif j ==1:
            y_link = [((i,0),2),-1]
            right_link = [((i,1),1),1]
            if i !=0:
                left_link = [((i-1,1),1),-1]
            elif i == 0:
                left_link = [((size-1,1),1),-1]
        return [left_link,y_link,right_link]
    
    def _determine_link_orientation(self,orientation,state):
        if orientation == 1:
            return state
        elif orientation == -1:
            if state == ONE :
                return state
            elif state == THREE:
                return THREE_BAR
            elif state ==THREE_BAR:
                return THREE
            elif state == "":
                return ""
            
    def _get_vertex_gauge_state(self,vertex_state,state):
        vertex_gauge_state = []
        for i in range(len(vertex_state)):
            link_state = state[i]
            link = vertex_state[i][0]
            orientation = vertex_state[i][1]
            vertex_gauge_state.append(self._determine_link_orientation(orientation,link_state))
        return vertex_gauge_state


    def lattice_images(self,parsed_data,generate_pdf=False):
        """ 
        This method takes as input the parsed_data obtained from the parser and generates images
        for all the bitstrings. These images can be saved in a PDF if the option "generate_pdf"
        is set to be true. The location of the pdf is printed in the terminal. If the "generate_pdf"
        option is not passed, the lattice images are saved as a pickled file.
        """
        filepath = self.rawdata
        dim = self.dim
        size = self.size
        image_dict = {}
        if generate_pdf == True:
            pdf_filepath = filepath + "-image_pdf"
            pdf = PdfPages(pdf_filepath)
        # dictionary of colors
        color_dict = {}
        color_dict[ONE] = "black"
        color_dict[THREE] = "blue"
        color_dict[THREE_BAR] = "indigo"
        color_dict[SIX] = "cyan"
        color_dict[SIX_BAR] = "teal"
        color_dict[EIGHT] = "seagreen"
        color_dict[''] = "red" # Unphysical links are given red color
        # dictionary of link states
        link_state = {}
        link_state[ONE] = "1"
        link_state[THREE] = "3"
        link_state[THREE_BAR] = "$\\bar{3}$"
        link_state[SIX] = "6"
        link_state[SIX_BAR] = "$\\bar{6}$"
        link_state[EIGHT] = "8"
        link_state[''] = "U"
        # Construction of the lattice images
        if dim == "3/2":
            yiter = 2
        elif dim == "2":
            yiter = size
        for key, parsed_item in parsed_data.items():
            vertex_check = self.vertex_check(parsed_item)
            check_vertices = False
            if vertex_check != {}:
                check_vertices = True
            plt.axis('off')
            ax = plt.gca()
            ax.set_aspect('equal',adjustable='box')
            ax.set_title(key+" , hits = " + str(parsed_item["hits"]))
            xval=[]
            for i in range(0,size):
                for j in range(0,yiter):
                    if check_vertices == True:
                        if vertex_check[f"(i,j)"] == "Correct":
                            plt.scatter(i,j,color='black',marker='o')
                        elif vertex_check[f"(i,j)"] == "Corrupted":
                            plt.scatter(i,j,color = 'red',marker='o')
                    elif check_vertices == False:
                        plt.scatter(i,j,color='black',marker='o')
                    if dim == "3/2":
                        if j==0:
                            link1 = parsed_item[f"((i,j),1)"]
                            link2 = parsed_item[f"((i,j),2)"]
                            plt.plot([i,i+1],[j,j],color = color_dict[link1],label = link_state[link1])
                            xval.append(i+0.5)
                            plt.plot([i,i],[j,j+1],color = color_dict[link2],label = link_state[link2])
                            xval.append(i)
                        if j==1 :
                            link1 = parsed_item[f"((i,j),1)"]
                            plt.plot([i,i+1],[j,j],color = color_dict[link1], label = link_state[link1])
                            xval.append(i+0.5)
                    elif dim == "2":
                        link1 = parsed_item[f"((i,j),1)"]
                        link2 = parsed_item[f"((i,j),2)"]
                        plt.plot([i,i+1],[j,j],color = color_dict[link1],label = link_state[link1])
                        xval.append(i+0.5)
                        plt.plot([i,i],[j,j+1],color = color_dict[link2],label = link_state[link2])
                        xval.append(i)
            lines=ax.get_lines()
            labelLines(lines,ha="center",xvals=xval,align=False)
            if generate_pdf == True:
                pdf.savefig()
                plt.close()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            image_dict[key] = image_base64

        image_dict_filepath = filepath + "-images"
        with open(image_dict_filepath,"wb") as fp:
            pickle.dump(image_dict,fp)
        self.image = image_dict_filepath
        print("The image dictionary is stored at the location  "+ image_dict_filepath)
        if generate_pdf==True:
            pdf.close()
            print("The image pdf is stored at the location  " + pdf_filepath)
    
    def lattice_visualizer(self,bitstring,image_filepath=None):
        """ 
        This method can be used to generate lattice diagrams of a particular bitstring. It takes as input
        the bitstring that needs to be translated into a lattice diagram. 

        The "lattice_images" method must be run before using this method. If the "lattice_images" method was
        run in a previous session, the optional "image_filepath" argument can be passed which specifies the 
        filepath of the pickled file generated by the "lattice_images" method.
        """
        if self.image == "" and image_filepath == None:
            raise IOError("You must first run the lattice_images method before trying to run the visualizer")
        else:
            if image_filepath == None:
                with open(self.image,"rb") as fp:
                    image_dict = pickle.load(fp)
            else:
                with open(image_filepath,"rb") as fp:
                    image_dict = pickle.load(fp)
        image_data = base64.b64decode(image_dict[bitstring])
        image = mpimg.imread(io.BytesIO(image_data), format='PNG')
        plt.imshow(image)
        plt.axis('off')
        plt.show()


