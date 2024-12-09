# Cianan Conefrey-Shinozaki
# Oct 2024


from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import YGate, XGate
from qiskit.circuit.library.standard_gates import RYGate, RXGate
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from random import random
import numpy as np
from scipy.linalg import expm


def givens(bit_string_1: str, bit_string_2: str, angle: float, reverse: bool = False) -> QuantumCircuit:
    """
    Expects two bitstrings in physicist / big-endian notation, e.g. "1001" or "0100", and a float as an angle. Optional reverse arg for little-endian.
    """
    if bit_string_1 == bit_string_2: # returns identity
        return QuantumCircuit(len(bit_string_1))

    
    bit_string_1_reversed = bit_string_1[::-1] # rewrite the string as little-endian Qiskit notation
    bit_string_2_reversed = bit_string_2[::-1]

    if len(bit_string_1) != len(bit_string_2):
        raise ValueError('bitstrings must be the same length')

    num_qubits = len(bit_string_1)
    
    circ = QuantumCircuit(num_qubits)

    if num_qubits == 1:
        circ.rx(angle,0)
        return circ
        
    
    for ii in range(num_qubits-1,-1,-1): # This finds where to put RY
        if bit_string_1_reversed[ii] != bit_string_2_reversed[ii]:
            target = ii
            break
    

    # This tells the later controls what value to control on
    ctrl_val = 0 if bit_string_1_reversed[target] == "1" else 1
    
    
    ctrls = list(range(0,num_qubits))
    ctrls.remove(target) # This provides the list of RX controls
    
    
    ctrl_state = bit_string_1_reversed[:target] + bit_string_1_reversed[(target+1):] # This provides what type of control
    ctrl_state = ctrl_state[::-1] # This reversal cancels some other reversal
    
    multiRX = RXGate(angle).control(num_ctrl_qubits=len(ctrls), ctrl_state=ctrl_state) # This builds the multicontrol X gate function

    # obsolete:
    # multiX = XGate().control(num_ctrl_qubits = 1, ctrl_state=ctrl_val)
    
    Xcirc = QuantumCircuit(num_qubits) # These are the "preamble" and "tail" control-X of the Givens

    for ii in range(target-1,-1,-1): # These instruct where to put the X gates
        if bit_string_1_reversed[ii] != bit_string_2_reversed[ii]:
            Xcirc.cx(control_qubit=target, target_qubit=ii, ctrl_state = ctrl_val)
    
    circ.append(
        multiRX,
        ctrls+[target]
    ) # This adds multiY to the circuit
    
    #obsolete:
    #circ = Xcirc.compose(circ.compose(Xcirc))
    circ.compose(Xcirc, inplace=True) # inplace arg looks weird but is much faster!
    Xcirc.compose(circ, inplace=True)
    if reverse == True:
        Xcirc = Xcirc.reverse_bits()
    return Xcirc
    
    #return circ
    #Xcirc.draw()


def givens2(strings: list, angle: float, reverse: bool = False) -> QuantumCircuit:
    """
    Expects a nested list of pairs of bitstrings in physicist / big-endian notation, e.g. strings = [['10','01'],['00','11']], and a float as an angle. Optional reverse arg for little-endian.
    """
    if len(strings) != 2:
        raise ValueError('Need two pairs')
    
    for ii in range(2):
        for jj in range(2):
            if len(strings[ii][jj]) != len(strings[ii - 1][jj]) or len(strings[ii][jj]) != len(strings[ii][jj-1]):
                raise ValueError('All bitstrings must be same length')
    
    
    num_qubits = len(strings[0][0])
    
    
    strings_reversed = strings
    for ii in range(2):
        for jj in range(2):
            strings_reversed[ii][jj] = strings[ii][jj][::-1]
    # print(strings_reversed)
    
    pin = None
    for ii in range(num_qubits-1,-1,-1): # This finds where to put RY
        if strings_reversed[0][0][ii] != strings_reversed[0][1][ii] and strings_reversed[1][0][ii] != strings_reversed[1][1][ii]:
            pin = ii
            break
    if pin == None:
        raise ValueError('Your bitstrings lack the special sauce')
    
    
    # print(f"pin = {pin}")
    
    lock = None
    type_2 = True
    for ii in range(num_qubits-1,-1,-1):
        property_1 = strings_reversed[0][0][ii] == strings_reversed[0][1][ii]
        property_2 = strings_reversed[1][0][ii] == strings_reversed[1][1][ii]
        property_3 = strings_reversed[0][0][ii] != strings_reversed[1][0][ii]
        if property_1 and property_2 and property_3:
            lock = ii
            type_2 = False
            break
    if type_2 == True:
        candidates = list(range(num_qubits-1,-1,-1))
        candidates.remove(pin)
        for ii in candidates:
            if strings_reversed[0][0][ii] != strings_reversed[0][1][ii] and strings_reversed[1][0][ii] != strings_reversed[1][1][ii]:
                lock = ii
                break
            
    if lock == None:
        raise ValueError('Your bitstrings lack the special sauce')
    
    bad1 = strings[0][0][pin] + strings[0][0][lock] == strings[1][0][pin] + strings[1][0][lock]
    bad2 = strings[0][0][pin] + strings[0][0][lock] == strings[1][1][pin] + strings[1][1][lock]
    if type_2 and (bad1 or bad2):
        raise ValueError('Your bitstrings lack the special sauce')
    
    
    
    
    #print(f"lock = {lock}")
    
    R_ctrls = list(range(0,num_qubits))
    R_ctrls.remove(pin)
    R_ctrls.remove(lock)
    
    R_ctrl_state = strings[0][0]
    if pin < lock:
        R_ctrl_state = R_ctrl_state[:pin] + R_ctrl_state[(pin+1):lock] + R_ctrl_state[(lock+1):]
    else:
        R_ctrl_state = R_ctrl_state[:lock] + R_ctrl_state[(lock+1):pin] + R_ctrl_state[(pin+1):]
    
    R_ctrl_state = R_ctrl_state[::-1]
    
    #print(f"R_ctrls = {R_ctrls}")
    #print(f"R_ctrl_state = {R_ctrl_state}")
    
    R_circ = QuantumCircuit(num_qubits)
    if R_ctrls == []:
        R_circ.rx(angle,pin)
    else:
        R_circ.append(
            RXGate(angle).control(num_ctrl_qubits = num_qubits - 2, ctrl_state = R_ctrl_state),
            R_ctrls + [pin]
        )
    
    #print(f"R_circ = \n{R_circ.draw(reverse_bits = True)}")
    
    X_2_circ = QuantumCircuit(num_qubits)
    
    for ii in R_ctrls + [lock]:
        if strings[0][0][ii] != strings[0][1][ii]:
            X_2_circ.cx(pin,ii, ctrl_state = strings[0][1][pin])
    
    #print(f"X_2_circ = \n{X_2_circ.draw(reverse_bits=True)}")
    
    if strings[0][0][pin] == strings[1][0][pin]:
        string_3 = strings[1][0]
        string_4 = strings[1][1]
    else:
        string_3 = strings[1][1]
        string_4 = strings[1][0]
    
    X_3_circ = QuantumCircuit(num_qubits)
    
    for ii in R_ctrls:
        if strings[0][0][ii] != string_3[ii]:
            X_3_circ.mcx([lock] + [pin], ii, ctrl_state = string_3[pin] + string_3[lock])
    
    #print(f"X_3_circ = \n{X_3_circ.draw(reverse_bits=True)}")
    
    X_4_circ = QuantumCircuit(num_qubits)
    
    for ii in R_ctrls:
        if strings[0][1][ii] != string_4[ii]:
            X_4_circ.mcx([lock] + [pin], ii, ctrl_state = string_4[pin] + string_4[lock])
    
    #print(f"X_4_circ = \n{X_4_circ.draw(reverse_bits=True)}")
    X_3_circ.compose(X_4_circ, inplace=True)
    R_circ.compose(X_2_circ, inplace=True)
    X_2_circ.compose(R_circ, inplace=True)
    X_2_circ.compose(X_3_circ, inplace=True)
    X_3_circ.compose(X_2_circ, inplace=True)
    
    #print(X_3_circ.draw(reverse_bits=True))
    return X_3_circ



def make_bitstring(length):
    return ''.join(f'{int(random()>0.5)}' for _ in range(length))

def test_givens():
    test_circ = QuantumCircuit(2)
    test_circ.cx(control_qubit = 1, target_qubit = 0, ctrl_state = "1")
    test_circ.append(
        RXGate(1).control(ctrl_state = "1"),
        [0,1]
    )
    test_circ.cx(control_qubit = 1, target_qubit = 0, ctrl_state = "1")
    
    op_test = Operator(test_circ)
    op_given = givens('01','10',1)
    assert op_test.equiv(op_given), "Failed first test"
    print("givens first test satisfied")

    

    N = int(random()*5+2)
    str1 = make_bitstring(N)
    str2 = make_bitstring(N)
    print(f"First random string = {str1}")
    print(f"Second random string = {str2}")
    

    angle = random()
    print(f"Random angle = {angle}")

    U2 = np.array(Operator(givens(str1,str2,angle)))
    H = np.zeros((2**N,2**N))
    #str1 = str1[::-1]
    #str2 = str2[::-2]
    H[int(str1,2),int(str2,2)] = 1
    H[int(str2,2),int(str1,2)] = 1
    U1 = expm(-1j/2*angle*H)

    assert np.max(U2-U1) < 10**(-10), "Failed random test"
    print("givens random test satisfied")

def test_givens2():
    N = 6
    random_strings = ['101100','001011','111011','110001']
    strings = [[random_strings[0],random_strings[1]],[random_strings[2],random_strings[3]]]
    
    print(f"Strings are {strings}")

    angle = random()
    print(f"Random angle = {angle}")

    H = np.zeros((2**N,2**N))
    H[int(strings[0][0],2),int(strings[0][1],2)] = 1
    H[int(strings[1][0],2),int(strings[1][1],2)] = 1

    H = H+H.transpose()

    U1 = expm(-1j/2*angle*H)

    U2 = np.array(Operator(givens2(strings,angle)))

    #print(abs(np.max(U1-U2)))
    assert abs(np.max(U2-U1)) < 10**(-7), "Failed givens2 test"
    print("givens2 test satisfied")
    print("All tests satisfied")

if __name__ == "__main__":
    test_givens()
    test_givens2()


