import pdb
import ast
from tqdm import tqdm
import random
import numpy as np
import cotengra as ctg
from scipy.sparse import dok_matrix
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.converters import circuit_to_dag, dag_to_circuit

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_aer.quantum_info import AerStatevector
from pgmQC.model.tensorRV_network_builder import TensorRVNetworkBuilder

def sample_gene(num_qubits, singe_qubit_gate_name, double_qubit_gate_name, num_layers):
    gene = ''
    for _ in range(num_layers * 2 * num_qubits + num_qubits):
        gene += str(random.randint(0, 1))
    return gene

def build_ansatz(num_qubits, single_qubit_gate_name, double_qubit_gate_name, num_layers, gene=None):
    # Create a quantum circuit
    qc = QuantumCircuit(num_qubits)
    cnt = 0
    for _ in range(num_layers):
        # Add the single qubit gates
        for i in range(num_qubits):
            if gene is not None and gene[_ * 2 * num_qubits + i] == '0':
                continue
            parameter_name = Parameter(f"theta_{_ * 2 * num_qubits + i}")
            if single_qubit_gate_name == 'rx':
                qc.rx(parameter_name, i)
            elif single_qubit_gate_name == 'ry':
                qc.ry(parameter_name, i)
            elif single_qubit_gate_name == 'rz':
                qc.rz(parameter_name, i)
        # Add the entangling gates
        for i in range(num_qubits-1):
            if gene is not None and gene[_ * 2 * num_qubits + num_qubits + i] == '0':
                continue
            parameter_name = Parameter(f"theta_{_ * 2 * num_qubits + num_qubits + i}")
            if double_qubit_gate_name == 'cx':
                qc.cx(i, i+1)
            elif double_qubit_gate_name == 'cz':
                qc.cz(i, i+1)
            elif double_qubit_gate_name == 'rzz':
                qc.cx(i, i + 1)
                qc.rz(parameter_name, i+1)
                qc.cx(i, i + 1)
    # Repeat the single qubit gates
    for i in range(num_qubits):
        if gene is not None and gene[num_layers * 2 * num_qubits + i] == '0':
            continue
        parameter_name = Parameter(f"theta_{num_layers * 2 * num_qubits + i}")
        if single_qubit_gate_name == 'rx':
            qc.rx(parameter_name, i)
        elif single_qubit_gate_name == 'ry':
            qc.ry(parameter_name, i)
        elif single_qubit_gate_name == 'rz':
            qc.rz(parameter_name, i)
    return qc