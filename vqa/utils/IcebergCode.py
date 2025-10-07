import pdb
import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import CircuitInstruction
from qiskit.quantum_info import Pauli, Statevector, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit import transpile

def convert_icebergcode_logical_operator(ins: CircuitInstruction, ice_circ: QuantumCircuit, t: int, b: int):
    if ins.operation.name == 'rz':
        angle = ins.operation.params[0]
        qubit = ins.qubits[0]._index
        ice_circ.rzz(angle, qubit, b)
    elif ins.operation.name == 'rx':
        angle = ins.operation.params[0]
        qubit = ins.qubits[0]._index
        ice_circ.rxx(angle, qubit, t)
    elif ins.operation.name == 'rzz':
        angle = ins.operation.params[0]
        qubit1 = ins.qubits[0]._index
        qubit2 = ins.qubits[1]._index
        ice_circ.rzz(angle, qubit1, qubit2)
    elif ins.operation.name == 'rxx':
        angle = ins.operation.params[0]
        qubit1 = ins.qubits[0]._index
        qubit2 = ins.qubits[1]._index
        ice_circ.rxx(angle, qubit1, qubit2)

def convert_logical_1q_rotation(opname: str, qubit: int, angle: float, ice_circ: QuantumCircuit, t: int, b: int):
    if opname == 'rz':
        ice_circ.rzz(angle, qubit, b)
    elif opname == 'rx':
        ice_circ.rxx(angle, qubit, t)
    else:
        assert False, f"unsupported opname: {opname}"

def convert_measurement_basis(ice_circ: QuantumCircuit, pauli: str, t: int, b: int):
    for i, p in enumerate(pauli[::-1]):
        if p == 'I' or p == 'Z':
            continue
        elif p == 'X':
            convert_logical_1q_rotation('rz', i, np.pi/2, ice_circ, t, b)
            convert_logical_1q_rotation('rx', i, np.pi/2, ice_circ, t, b)
        elif p == 'Y':
            convert_logical_1q_rotation('rx', i, np.pi/2, ice_circ, t, b)

def add_icebergcode_measurement(ice_circ: QuantumCircuit, k: int, t: int, b: int):
    a1 = b + 1
    a2 = b + 2
    ice_circ.h(a1)
    ice_circ.cx(a1, t)
    ice_circ.cx(a1, a2)
    for i in range(k):
        ice_circ.cx(a1, i)
    ice_circ.cx(a1, a2)
    ice_circ.cx(a1, b)
    ice_circ.h(a1)
    for i in range(k + 4):
        ice_circ.measure(i, i)

def add_icebergcode_decoding2(ice_circ: QuantumCircuit, k: int, t: int, b: int):
    for i in range(k):
        ice_circ.cx(i, t)
    ice_circ.cx(b, t)
    for i in range(k):
        ice_circ.cx(b, i)
    ice_circ.h(b)

def convert_to_icebergcode_circuit(circuit: QuantumCircuit):
    transpiled_circuit = transpile(circuit, basis_gates=['rz', 'rx', 'rxx', 'rzz'], optimization_level=3)
    ice_circ = QuantumCircuit(transpiled_circuit.num_qubits + 4)
    k = transpiled_circuit.num_qubits
    t = k
    b = k + 1
    ice_circ.h(0)
    for i in range(transpiled_circuit.num_qubits + 1):
        ice_circ.cx(i, i + 1)
    for ins in transpiled_circuit:
        convert_icebergcode_logical_operator(ins, ice_circ, t, b)
    return ice_circ