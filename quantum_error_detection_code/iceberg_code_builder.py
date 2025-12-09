import pdb
import numpy as np
from tqdm import tqdm
import tensorcircuit as tc
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Pauli, Statevector, SparsePauliOp
from qiskit.circuit import Parameter, Gate
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.quantum_info import AerStatevector
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)

from classical_register_allocator import ClassicalRegisterAllocator
# Define a custom CX-equivalent gate
class IdealCX(Gate):
    def __init__(self):
        super().__init__('ideal_cx', 2, [])

    def _define(self):
        qc = QuantumCircuit(2, name='ideal_cx')
        qc.cx(0, 1)
        self.definition = qc
ideal_cx = IdealCX()


class Iceberg_Code_Builder:
    def __init__(self, base_circuit: QuantumCircuit, t, b, qubit_list, a1, a2, clbit_allocator: ClassicalRegisterAllocator, dmc=None):
        self.base_circuit = base_circuit
        self.t = t
        self.b = b
        self.qubit_list = qubit_list.copy()
        assert len(qubit_list) >= 2 and len(qubit_list) % 2 == 0, "qubit_list should contain even number of qubits."
        self.a1 = a1
        self.a2 = a2
        self.clbit_allocator = clbit_allocator
        self.dmc = dmc
        self.syndrome_bits = []

    def initialize(self):
        syndrome_bit_index = self.clbit_allocator.get_clbit_index()
        self.base_circuit.h(self.t)
        self.base_circuit.cx(self.t, self.qubit_list[0])
        for i in range(0, len(self.qubit_list)-1):
            self.base_circuit.cx(self.qubit_list[i], self.qubit_list[i+1])
        self.base_circuit.cx(self.qubit_list[-1], self.b)
        self.base_circuit.reset(self.a1)
        self.base_circuit.cx(self.t, self.a1)
        self.base_circuit.cx(self.b, self.a1)
        self.base_circuit.measure(self.a1, syndrome_bit_index)
        self.syndrome_bits.append(syndrome_bit_index)
        if self.dmc is not None:
            print("Trace before initialization: ", self.dmc.state().trace())
            self.dmc.h(self.t)
            self.dmc.cnot(self.t, self.qubit_list[0])
            for i in range(0, len(self.qubit_list)-1):
                self.dmc.cnot(self.qubit_list[i], self.qubit_list[i+1])
            self.dmc.cnot(self.qubit_list[-1], self.b)
            self.dmc.reset(self.a1)
            self.post_select(self.a1)
            self.dmc.cnot(self.t, self.a1)
            self.dmc.cnot(self.b, self.a1)
        
    def logical_RZZ(self, theta: float, qubit1: int, qubit2: int):
        self.base_circuit.rzz(theta, self.qubit_list[qubit1], self.qubit_list[qubit2])
        if self.dmc is not None:
            self.dmc.RZZ(self.qubit_list[qubit1], self.qubit_list[qubit2], theta=theta)
    
    def logical_RX(self, theta: float, qubit: int):
        self.base_circuit.rxx(theta, self.t, self.qubit_list[qubit])
        if self.dmc is not None:
            self.dmc.RXX(self.t, self.qubit_list[qubit], theta=theta)
    
    def logical_RZ(self, theta: float, qubit: int):
        self.base_circuit.rzz(theta, self.b, self.qubit_list[qubit])
        if self.dmc is not None:
            self.dmc.RZZ(self.b, self.qubit_list[qubit], theta=theta)

    def logical_RZs(self, theta: float, qubit_indices: list[int]):
        physical_qubits = [self.qubit_list[i] for i in qubit_indices]
        if len(qubit_indices) % 2 == 1:
            physical_qubits.append(self.b)
        if len(physical_qubits) > (len(self.qubit_list) + 2) / 2:
            physical_qubits = [q for q in self.qubit_list + [self.t, self.b] if q not in physical_qubits]
        for i in range(len(physical_qubits)-2):
            self.base_circuit.cx(physical_qubits[i], physical_qubits[i+1])
        self.base_circuit.rzz(theta, physical_qubits[-2], physical_qubits[-1])
        for i in range(len(physical_qubits)-3, -1, -1):
            self.base_circuit.cx(physical_qubits[i], physical_qubits[i+1])
        
        if self.dmc is not None:
            for i in range(len(physical_qubits)-2):
                self.dmc.cnot(physical_qubits[i], physical_qubits[i+1])
            self.dmc.RZZ(physical_qubits[-2], physical_qubits[-1], theta=theta)
            for i in range(len(physical_qubits)-3, -1, -1):
                self.dmc.cnot(physical_qubits[i], physical_qubits[i+1])
    
    def logical_X(self, qubit: int):
        self.base_circuit.x(self.t)
        self.base_circuit.x(self.qubit_list[qubit])
        if self.dmc is not None:
            self.dmc.x(self.t)
            self.dmc.x(self.qubit_list[qubit])
    
    def post_select(self, qubit: int, flag=0):
        if flag == 0:
            gate = np.array(
                [
                    [1, 0],
                    [0, 0],
                ]
            )
        else:
            gate = np.array(
                [
                    [0, 0],
                    [0, 1],
                ]
            )
        self.dmc.unitary(qubit, unitary=gate, name="non_unitary")
        p = self.dmc.state().trace().real
        print("Trace after post-selecting qubit ", qubit, ": ", p)
        self.dmc.unitary(qubit, unitary=np.array([[1/np.sqrt(p), 0], [0, 1/np.sqrt(p)]]), name="renormalize")
        # print("Trace after renormalizing qubit ", qubit, ": ", self.dmc.state().trace())

    
    def syndrome_measurement(self):
        syndrome_bit_index1 = self.clbit_allocator.get_clbit_index()
        syndrome_bit_index2 = self.clbit_allocator.get_clbit_index()
        self.base_circuit.reset(self.a1)
        self.base_circuit.reset(self.a2)
        
        self.base_circuit.h(self.a2)
        self.base_circuit.append(ideal_cx, [self.a2, self.t])
        # self.base_circuit.cx(self.a2, self.t)
        self.base_circuit.append(ideal_cx, [self.t, self.a1])
        # self.base_circuit.cx(self.t, self.a1)
        self.base_circuit.append(ideal_cx, [self.qubit_list[0], self.a1])
        # self.base_circuit.cx(self.qubit_list[0], self.a1)
        self.base_circuit.append(ideal_cx, [self.a2, self.qubit_list[0]])
        # self.base_circuit.cx(self.a2, self.qubit_list[0])
        
        for i in range(1, len(self.qubit_list)):
            self.base_circuit.append(ideal_cx, [self.a2, self.qubit_list[i]])
            # self.base_circuit.cx(self.a2, self.qubit_list[i])
            self.base_circuit.append(ideal_cx, [self.qubit_list[i], self.a1])
            # self.base_circuit.cx(self.qubit_list[i], self.a1)
        
        self.base_circuit.append(ideal_cx, [self.b, self.a1])
        # self.base_circuit.cx(self.b, self.a1)
        self.base_circuit.append(ideal_cx, [self.a2, self.b])
        # self.base_circuit.cx(self.a2, self.b)
        
        self.base_circuit.h(self.a2)
        self.base_circuit.measure(self.a1, syndrome_bit_index1)
        self.base_circuit.measure(self.a2, syndrome_bit_index2)
        self.syndrome_bits.append(syndrome_bit_index1)
        self.syndrome_bits.append(syndrome_bit_index2)
    
    def measurement(self) -> list[int]:
        syndrome_bit_index1 = self.clbit_allocator.get_clbit_index()
        syndrome_bit_index2 = self.clbit_allocator.get_clbit_index()
        self.base_circuit.reset(self.a1)
        self.base_circuit.reset(self.a2)
        
        self.base_circuit.h(self.a1)
        
        self.base_circuit.cx(self.a1, self.t)
        self.base_circuit.cx(self.a1, self.a2)
        for i in range(len(self.qubit_list)):
            self.base_circuit.cx(self.a1, self.qubit_list[i])
        self.base_circuit.cx(self.a1, self.a2)
        self.base_circuit.cx(self.a1, self.b)
        
        self.base_circuit.h(self.a1)
        
        self.base_circuit.measure(self.a1, syndrome_bit_index1)
        self.base_circuit.measure(self.a2, syndrome_bit_index2)
        self.syndrome_bits.append(syndrome_bit_index1)
        self.syndrome_bits.append(syndrome_bit_index2)
        
        self.cbits = [self.clbit_allocator.get_clbit_index() for _ in range(len(self.qubit_list)+2)]
        self.base_circuit.measure(self.t, self.cbits[0])
        for i in range(len(self.qubit_list)):
            self.base_circuit.measure(self.qubit_list[i], self.cbits[i+1])
        self.base_circuit.measure(self.b, self.cbits[len(self.qubit_list)+1])
        if self.dmc is not None:
            self.dmc.reset(self.a1)
            self.dmc.reset(self.a2)
            self.dmc.h(self.a1)
            
            self.dmc.cnot(self.a1, self.t)
            self.dmc.cnot(self.a1, self.a2)
            for i in range(len(self.qubit_list)):
                self.dmc.cnot(self.a1, self.qubit_list[i])
            self.dmc.cnot(self.a1, self.a2)
            self.dmc.cnot(self.a1, self.b)
            
            self.dmc.h(self.a1)
            
            self.post_select(self.a1)
            self.post_select(self.a2)
            # checking the parity of the bitstring
            project_0 = tc.gates.Gate([[1, 0], [0, 0]])
            project_1 = tc.gates.Gate([[0, 0], [0, 1]])
            self.dmc.apply_general_kraus([project_0, project_1], [[self.t], [self.t]])
            self.dmc.apply_general_kraus([project_0, project_1], [[self.b], [self.b]])
            for i in range(len(self.qubit_list)):
                self.dmc.apply_general_kraus([project_0, project_1], [[self.qubit_list[i]], [self.qubit_list[i]]])
            self.dmc.cnot(self.b, self.t)
            for i in range(len(self.qubit_list)):
                self.dmc.cnot(self.qubit_list[i], self.t)
            self.post_select(self.t)
            # decoding
            for i in range(len(self.qubit_list)):
                self.dmc.cnot(self.b, self.qubit_list[i])
            # self.dmc.measure(self.t, self.cbits[0])
            # for i in range(len(self.qubit_list)):
            #     self.dmc.measure(self.qubit_list[i], self.cbits[i+1])
            # self.dmc.measure(self.b, self.cbits[len(self.qubit_list)+1])
        return self.cbits
    
    def decode(self, bitstring: str) -> str:
        for syndrome_bit_index in self.syndrome_bits:
            if bitstring[-syndrome_bit_index-1] == '1':
                return 'invalid'
        measured_bitstring = ''.join([bitstring[-cbit-1] for cbit in self.cbits])
        if measured_bitstring.count('1') % 2 == 1:
            return 'invalid'
        if measured_bitstring[len(self.cbits)-1] == '0':
            return measured_bitstring[1:-1]
        else:
            return ''.join(['1' if bit == '0' else '0' for bit in measured_bitstring[1:-1]])
    
    def copy_for_a_new_circuit(self, new_base_circuit: QuantumCircuit, new_clbit_allocator: ClassicalRegisterAllocator, new_dmc: tc.DMCircuit):
        new_builder = Iceberg_Code_Builder(new_base_circuit, self.t, self.b, self.qubit_list, self.a1, self.a2, new_clbit_allocator, new_dmc)
        new_builder.syndrome_bits = self.syndrome_bits.copy()
        return new_builder