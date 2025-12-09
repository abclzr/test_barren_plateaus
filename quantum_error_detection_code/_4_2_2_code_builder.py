import pdb
import numpy as np
from tqdm import tqdm
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

class IdealCX(Gate):
    def __init__(self):
        super().__init__('ideal_cx', 2, [])

    def _define(self):
        qc = QuantumCircuit(2, name='ideal_cx')
        qc.cx(0, 1)
        self.definition = qc
ideal_cx = IdealCX()

class _4_2_2_Code_Builder:
    def __init__(self, base_circuit: QuantumCircuit, t, b, q1, q2, a1, a2, clbit_allocator: ClassicalRegisterAllocator):
        self.base_circuit = base_circuit
        self.t = t
        self.b = b
        self.q1 = q1
        self.q2 = q2
        self.a1 = a1
        self.a2 = a2
        self.clbit_allocator = clbit_allocator
        self.syndrome_bits = []

    def initialize(self):
        syndrome_bit_index = self.clbit_allocator.get_clbit_index()
        self.base_circuit.h(self.t)
        self.base_circuit.cx(self.t, self.q1)
        self.base_circuit.cx(self.q1, self.q2)
        self.base_circuit.cx(self.q2, self.b)
        self.base_circuit.reset(self.a1)
        self.base_circuit.cx(self.t, self.a1)
        self.base_circuit.cx(self.b, self.a1)
        self.base_circuit.measure(self.a1, syndrome_bit_index)
        self.syndrome_bits.append(syndrome_bit_index)
    
    def transversal_Hadamard(self):
        self.base_circuit.h(self.t)
        self.base_circuit.h(self.b)
        self.base_circuit.h(self.q1)
        self.base_circuit.h(self.q2)
        # self.base_circuit.swap(self.q1, self.q2)
    
    def transversal_CNOT(self, other_code_builder):
        self.base_circuit.cx(self.t, other_code_builder.t)
        self.base_circuit.cx(self.b, other_code_builder.b)
        self.base_circuit.cx(self.q1, other_code_builder.q1)
        self.base_circuit.cx(self.q2, other_code_builder.q2)
    
    def logical_RZZ(self, theta: float):
        self.base_circuit.rzz(theta, self.q1, self.q2)
    
    def logical_RX(self, theta: float, qubit: int):
        self.base_circuit.rxx(theta, self.t, self.q1 if qubit == 1 else self.q2)
    
    def logical_RZ(self, theta: float, qubit: int):
        self.base_circuit.rzz(theta, self.b, self.q1 if qubit == 1 else self.q2)

    def logical_X(self, qubit: int):
        self.base_circuit.x(self.t)
        self.base_circuit.x(self.q1 if qubit == 1 else self.q2)

    def syndrome_measurement(self):
        syndrome_bit_index1 = self.clbit_allocator.get_clbit_index()
        syndrome_bit_index2 = self.clbit_allocator.get_clbit_index()
        self.base_circuit.reset(self.a1)
        self.base_circuit.reset(self.a2)
        
        self.base_circuit.h(self.a2)
        self.base_circuit.append(ideal_cx, [self.a2, self.t])
        self.base_circuit.append(ideal_cx, [self.t, self.a1])
        self.base_circuit.append(ideal_cx, [self.q1, self.a1])
        self.base_circuit.append(ideal_cx, [self.a2, self.q1])
        
        self.base_circuit.append(ideal_cx, [self.a2, self.q2])
        self.base_circuit.append(ideal_cx, [self.q2, self.a1])
        self.base_circuit.append(ideal_cx, [self.b, self.a1])
        self.base_circuit.append(ideal_cx, [self.a2, self.b])
        
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
        self.base_circuit.cx(self.a1, self.q1)
        self.base_circuit.cx(self.a1, self.q2)
        self.base_circuit.cx(self.a1, self.a2)
        self.base_circuit.cx(self.a1, self.b)
        
        self.base_circuit.h(self.a1)
        
        self.base_circuit.measure(self.a1, syndrome_bit_index1)
        self.base_circuit.measure(self.a2, syndrome_bit_index2)
        self.syndrome_bits.append(syndrome_bit_index1)
        self.syndrome_bits.append(syndrome_bit_index2)
        
        self.cbits = [self.clbit_allocator.get_clbit_index() for _ in range(4)]
        self.base_circuit.measure(self.t, self.cbits[0])
        self.base_circuit.measure(self.q1, self.cbits[1])
        self.base_circuit.measure(self.q2, self.cbits[2])
        self.base_circuit.measure(self.b, self.cbits[3])
        return self.cbits
    
    def decode(self, bitstring: str) -> str:
        for syndrome_bit_index in self.syndrome_bits:
            if bitstring[-syndrome_bit_index-1] == '1':
                return 'invalid'
        measured_bitstring = ''.join([bitstring[-self.cbits[0]-1], bitstring[-self.cbits[1]-1], bitstring[-self.cbits[2]-1], bitstring[-self.cbits[3]-1]])
        if measured_bitstring.count('1') % 2 == 1:
            return 'invalid'
        if measured_bitstring[3] == '0':
            return measured_bitstring[1:3]
        else:
            return ''.join(['1' if bit == '0' else '0' for bit in measured_bitstring[1:3]])
    
    def copy_for_a_new_circuit(self, new_base_circuit: QuantumCircuit, new_clbit_allocator: ClassicalRegisterAllocator):
        new_builder = _4_2_2_Code_Builder(new_base_circuit, self.t, self.b, self.q1, self.q2, self.a1, self.a2, new_clbit_allocator)
        new_builder.syndrome_bits = self.syndrome_bits.copy()
        return new_builder