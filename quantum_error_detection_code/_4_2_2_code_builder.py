import pdb
import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Pauli, Statevector, SparsePauliOp
from qiskit.circuit import Parameter
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.quantum_info import AerStatevector
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)


class ClassicalRegisterAllocator:
    def __init__(self, clibts: int):
        self.clibts = clibts
        self.cnt = 0
    
    def get_clbit_index(self):
        self.cnt += 1
        return self.cnt - 1


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
    
    def transversal_Hadamard(self):
        self.base_circuit.h(self.t)
        self.base_circuit.h(self.b)
        self.base_circuit.h(self.q1)
        self.base_circuit.h(self.q2)
    
    def transversal_CNOT(self, other_code_builder):
        self.base_circuit.cx(self.t, other_code_builder.t)
        self.base_circuit.cx(self.b, other_code_builder.b)
        self.base_circuit.cx(self.q1, other_code_builder.q1)
        self.base_circuit.cx(self.q2, other_code_builder.q2)
    
    def logical_RZZ(self, theta: float):
        self.base_circuit.rzz(theta, self.q1, self.q2)
    
    def logical_RX(self, theta: float, qubit: int):
        self.base_circuit.rxx(theta, self.t, self.q1 if qubit == 1 else self.q2)

    def syndrome_measurement(self):
        syndrome_bit_index1 = self.clbit_allocator.get_clbit_index()
        syndrome_bit_index2 = self.clbit_allocator.get_clbit_index()
        self.base_circuit.reset(self.a1)
        self.base_circuit.reset(self.a2)
        
        self.base_circuit.h(self.a2)
        self.base_circuit.cx(self.a2, self.t)
        self.base_circuit.cx(self.t, self.a1)
        self.base_circuit.cx(self.q1, self.a1)
        self.base_circuit.cx(self.a2, self.q1)
        
        self.base_circuit.cx(self.a2, self.q2)
        self.base_circuit.cx(self.q2, self.a1)
        self.base_circuit.cx(self.b, self.a1)
        self.base_circuit.cx(self.a2, self.b)
        
        self.base_circuit.h(self.a2)
        self.base_circuit.measure(self.a1, syndrome_bit_index1)
        self.base_circuit.measure(self.a2, syndrome_bit_index2)
    
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
        
        cbits = [self.clbit_allocator.get_clbit_index() for _ in range(4)]
        self.base_circuit.measure(self.t, cbits[0])
        self.base_circuit.measure(self.q1, cbits[1])
        self.base_circuit.measure(self.q2, cbits[2])
        self.base_circuit.measure(self.b, cbits[3])
        return cbits
    
    def decode(self, bitstring: str) -> str:
        assert len(bitstring) == 4
        if bitstring.count('1') % 2 == 1:
            return 'invalid'
        if bitstring[-1] == '0':
            return bitstring[2:4]
        else:
            return ''.join(['1' if bit == '0' else '0' for bit in bitstring[2:4]])