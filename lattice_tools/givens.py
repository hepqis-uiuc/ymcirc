# Cianan Conefrey-Shinozaki
# Oct 2024


from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import YGate, XGate
from qiskit.circuit.library.standard_gates import RYGate, RXGate
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator


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

if __name__ == "__main__":
    test_givens()
