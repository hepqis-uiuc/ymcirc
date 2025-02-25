#################################################
# This is a temporary code to make visualizations of
# lattice states in dim = 3/2, truncation = T1
# I am working on a code that will work for more general cases.
# This code has other functions as well, but this is the most
# useful one for now.

# To make visualizations of a particular state of a
# lattice in dim=3/2, follow the following steps:
# 1. Create a QuantumLattice instance:
#             qc = QuantumLattice(size)
# where size is the number of plaquettes in the lattice.
#
# 2. Call the function visualize_lattice_state:
#             qc.visualize_lattice_state(plaq_state,savefig--> Optional)
# plaq_state is a list of strings. Each string element
# denotes the state of a link. The list is ordered in the following way:

#     ------1---- -------4------ ..........  
#     |          |             |           |
#     |          |             |           | 
#     0          3             6          PBC
#     |          |             |           |
#     |          |             |           |
#     -----2----- ------5------  .........

# Use "o" to denote the 1 state, "t" to denote the 3 state,
# and "tb" to denote the three-bar state. For example, in a
# two plaquette lattice, the code to generate a visual could look like
# qc = QuantumLattice(2)
# plaq_state = ["t","o","o","tb","tb","t"]
# qc.visualize_lattice_state(plaq_state,savefig=[True,"mylattice"])
# savefig is an optional argument, its default value is False. If it's set
# to be true, then it'll save a figure called "mylattice.png" in your current
# folder. The terminal will show some range errors once this script is run,
# but it's noting consequential, as far as I can tell.
###############################################################################

from qiskit import *
from qiskit_aer import AerSimulator
import pylatexenc
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from labellines import labelLines
import numpy as np


class QuantumLattice:

    def __init__(self,size):
        self.size = size

    #create quantum register for the lattice
    def quantum_register(self):
        linksize = 3*self.size
        qc = QuantumCircuit(2*linksize,2)
        return qc
    
    #This function visualizes a particular state of the lattice
    def visualize_lattice_state(self,plaq_state,savefig=[False,None]):
        def state_color(state):
            if state=="o":
                col = 'black'
            elif state=="t":
                col = 'cyan'
            elif state=="tb":
                col = 'magenta'
            return col
        def state_string(state):
            if state=="o":
                statestr = "1"
            elif state=="t":
                statestr = "3"
            elif state=="tb":
                statestr = "$\\bar{3}$"
            return statestr
        vertex_size = self.size+1
        x = np.arange(0,vertex_size)
        y = np.arange(0,2)
        X,Y = np.meshgrid(x,y)
        plt.ylim(-0.1,1.2)
        plt.axis('off')
        ax = plt.gca()
        ax.set_aspect('equal',adjustable='box')
        plt.scatter(X,Y,color = 'black',marker='o')
        xval=[]
        for i in range(0,self.size):
             plt.plot([i,i+1],[0,0],color=state_color(plaq_state[3*i+2]),label=state_string(plaq_state[3*i+2]))
             xval.append(i+0.5)
             plt.plot([i,i],[0,1],color=state_color(plaq_state[3*i]),label=state_string(plaq_state[3*i]))
             xval.append(i)
             plt.plot([i,i+1],[1,1],color=state_color(plaq_state[3*i+1]),label=state_string(plaq_state[3*i+1]))
             xval.append(i+0.5)
        plt.plot([vertex_size-1,vertex_size-1],[0,1],color=state_color(plaq_state[0]),label=state_string(plaq_state[0]))
        xval.append(vertex_size-0.5)    
        lines=ax.get_lines()
        labelLines(lines,ha="center",xvals=xval,align=False)
        #plt.show()
        if savefig[0] == True:
            plt.savefig(savefig[1])
        else:
            plt.show()

    
    #a visual of the lattice being used.
    def visualize_lattice(self,**kwargs):
        xsite = kwargs.get('x',None)
        ysite = kwargs.get('y',None)
        dir = kwargs.get('d',None)
        bits = kwargs.get('bit_values',None)
        vertex_size = self.size+1
        x = np.arange(0,vertex_size)
        y = np.arange(0,2)
        X,Y = np.meshgrid(x,y)
        plt.ylim(-0.1,1.2)
        plt.axis('off')
        ax = plt.gca()
        ax.set_aspect('equal',adjustable='box')
        plt.scatter(X,Y,color = 'black',marker='o')
        xval=[]
        for i in range(0,self.size):
            if i == xsite:
                #ax.text(self.size/2,-1,'qubits #'+str(bits[0])+' and #'+str(bits[1])+' are used for this link',ha='center')
                if ysite == 0:
                    if dir==1:
                        plt.plot([i,i+1],[0,0],color='blue',label=str(3*i+2),linewidth=2)
                        xval.append(i+0.5)
                        plt.plot([i,i],[0,1],color='black',label=str(3*i))
                        xval.append(i)
                        plt.plot([i,i+1],[1,1],color='black',label=str(3*i+1))
                        xval.append(i+0.5)
                    elif dir==2:
                        plt.plot([i,i+1],[0,0],color='black',label=str(3*i+2))
                        xval.append(i+0.5)
                        plt.plot([i,i],[0,1],color='blue',label=str(3*i),linewidth=2)
                        xval.append(i)
                        plt.plot([i,i+1],[1,1],color='black',label=str(3*i+1))
                        xval.append(i+0.5)
                elif ysite==1:
                    if dir==2:
                        raise ValueError("improper direction value")
                    elif dir==1:
                        plt.plot([i,i+1],[0,0],color='black',label=str(3*i+2))
                        xval.append(i+0.5)
                        plt.plot([i,i],[0,1],color='black',label=str(3*i))
                        xval.append(i)
                        plt.plot([i,i+1],[1,1],color='blue',label=str(3*i+1),linewidth=2)
                        xval.append(i+0.5)
            else:
                plt.plot([i,i+1],[0,0],color='black',label=str(3*i+2))
                xval.append(i+0.5)
                plt.plot([i,i],[0,1],color='black',label=str(3*i))
                xval.append(i)
                plt.plot([i,i+1],[1,1],color='black',label=str(3*i+1))
                xval.append(i+0.5)
        if xsite==self.size:
            plt.plot([vertex_size-1,vertex_size-1],[0,1],color='blue',linewidth=2,label=str(0))
            #ax.text(self.size/2,-1,'qubits #'+str(bits[0])+' and #'+str(bits[1])+' are used for this link',ha='center')
        else:
            plt.plot([vertex_size-1,vertex_size-1],[0,1],color='black',label=str(0))
        xval.append(vertex_size-0.5)    
        lines=ax.get_lines()
        labelLines(lines,ha="center",xvals=xval,align=False)
        
        plt.show()
        #plt.savefig("./3_plaquette_visual_normal.png",bbox_inches='tight')


    #locate the bitstrings related to pos (x,y,d). If y=0, then d=1,2. If y=1, then d=1
    def link_site(self,x,y,d,**kwargs):
        visual = kwargs.get('visual',False)
        if x>self.size or y>1 or x<0 or y<0:
            raise ValueError("lattice site specifications not compatible with lattice size")
        if d!=1 and d!=2:
                raise ValueError("d can only have values 1 (x-dir) or 2 (y dir)")
        if x!=self.size:
            if y == 0:
                if d==1:
                    bit_values = [2*(3*x+2),2*(3*x+2)+1]
                elif d==2:
                    bit_values = [2*3*x,2*3*x+1]

            if y==1:
                if d==2:
                    raise ValueError("y=1,d=2 not allowed for dim 3/2")
                
                if d==1:
                    bit_values = [2*(3*x+1),2*(3*x+1)+1]
        elif x==self.size:
            if y==1:
                raise ValueError("lattice site specifications not compatible with lattice size")
            if d==1:
                raise ValueError("lattice site specifications not compatible with lattice size")
            if d==2:
                bit_values=[0,1]

        if visual == True:        
            self.visualize_lattice(x=x,y=y,d=d,bit_values=bit_values)

        return bit_values

    #given a quantum circuit, you can define an evolution for the lattice.
    def lattice_circuit(self,q_gate_circuit:QuantumCircuit):
        lattice_qc = self.quantum_register()
        lattice_qc.initialize(0)
        if q_gate_circuit.num_qubits > lattice_qc.num_qubits:
            raise ValueError("quantum circuit can't have more qubits than the quantum lattice qubits")
        
        else:
            lattice_qc = lattice_qc.compose(q_gate_circuit)

        return lattice_qc
    
    #Measure a link
    def measure_lattice_circuit(self,lattice_qc:QuantumCircuit,x,y,d,shots,**kwargs):
        hist = kwargs.get('hist',False)
        link_to_measure = self.link_site(x,y,d)
        lattice_qc = self.lattice_circuit(lattice_qc)
        lattice_qc.measure(link_to_measure[0],0)
        lattice_qc.measure(link_to_measure[1],1)
        simulator = AerSimulator()
        lattice_qc = transpile(lattice_qc,simulator)
        result = simulator.run(lattice_qc,shots=shots).result()
        counts = result.get_counts(lattice_qc)
        if hist==True:
            plot_histogram(counts)
            plt.show()
        #return counts
    
    def E2_ev(self,lattice_qc:QuantumCircuit,x,y,d,shots):
        counts = self.measure_lattice_circuit(lattice_qc,x,y,d,shots)
        p11 = counts['11']/shots
        p10 = counts['10']/shots
        p01 = counts['01']/shots
        p00 = counts['00']/shots
        E2_ev = (p10+p01)*4/3 
        return E2_ev
        

### Basic Test
#my_qc = QuantumCircuit(12)

#my_qc.h(0)
#my_qc.cx(0,1)

#two_plaquette_lattice.measure_lattice_circuit(my_qc,0,0,2,1000,hist=True)

#two_plaquette_lattice.link_site(2,0,2,visual=True)

# my_qc.h(8)
# my_qc = two_plaquette_lattice.lattice_circuit(my_qc)
# my_qc.draw(output="mpl")
# plt.show()

# for i in range(1,12,2):
#     my_qc.h(i)
#     if i != 11:
#         my_qc.cx(i,i+1)

# two_plaquette_lattice.measure_lattice_circuit(my_qc,1,0)

# print(two_plaquette_lattice.E2_ev(my_qc,1,1,1,1000))
    

        