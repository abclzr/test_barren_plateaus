import pdb
import numpy as np
from tqdm import tqdm
import tensorcircuit as tc
from tensorcircuit.channels import depolarizingchannel, isotropicdepolarizingchannel



class Density_Matrix_Simulator:
    def __init__(self, num_qubits, noise_level=0):
        self.dmc = tc.DMCircuit(num_qubits)
        if noise_level == 0:
            self.has_noise = False
        else:
            self.has_noise = True
            self.cs = isotropicdepolarizingchannel(0.0004, 1)
            self.cd = isotropicdepolarizingchannel(0.003, 2)
        self.instructions = []
    
    def refresh_dmc(self):
        if self.dmc is not None:
            self.dmc = tc.DMCircuit(self.dmc._nqubits, dminputs=self.dmc.state())

    def add_gate(self, gate_name: str, qubits: list, params=None):
        self.instructions.append((gate_name, qubits, params))
    
    def contract_all_gates(self):
        post_selection_rate_list = []
        for inst in self.instructions:
            gate_name, qubits, params = inst
            if gate_name == "h":
                self.dmc.h(qubits[0])
            elif gate_name == "x":
                self.dmc.x(qubits[0])
            elif gate_name == "reset":
                self.dmc.reset(qubits[0])
            elif gate_name == "cnot":
                self.dmc.cnot(qubits[0], qubits[1])
            elif gate_name == "rz":
                self.dmc.RZ(params[0], qubits[0])
            elif gate_name == "rzz":
                self.dmc.RZZ(params[0], qubits[0], qubits[1])
            elif gate_name == "rxx":
                self.dmc.RXX(params[0], qubits[0], qubits[1])
            elif gate_name == 'post_select':
                p = self.post_select(qubits[0], params[0])
                post_selection_rate_list.append(p)
            else:
                raise ValueError(f"Unsupported gate: {gate_name}")
        
    def post_select(self, qubit: int, flag=0):
        if flag == 0:
            gate = np.array(
                [
                    [1, 0],
                    [0, 0],
                ],
            )
        else:
            gate = np.array(
                [
                    [0, 0],
                    [0, 1],
                ],
            )
        self.dmc.unitary(qubit, unitary=gate, name="non_unitary")
        p = self.dmc.state().trace().real
        print("Trace after post-selecting qubit ", qubit, ": ", p)
        self.dmc.unitary(qubit, unitary=np.array([[1/np.sqrt(p), 0], [0, 1/np.sqrt(p)]]), name="renormalize")
        print("Trace after renormalizing qubit ", qubit, ": ", self.dmc.state().trace())
        return p

