import numpy as np
import pdb

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector

k = 6
n = k + 2
t = k
b = k + 1

logical_circuit = QuantumCircuit(k)
physical_circuit = QuantumCircuit(n)
physical_circuit.h(0)
for i in range(n - 1):
    physical_circuit.cx(i, i + 1)

def check_syndrome(statevec : Statevector):
    Sx = Pauli('X' * n)
    Sz = Pauli('Z' * n)
    print("Sx expectation:", (statevec).expectation_value(Sx))
    print("Sz expectation:", (statevec).expectation_value(Sz))

def add_logical_rz(theta: float, qubit: int, l_c: QuantumCircuit, p_c: QuantumCircuit):
    l_c.rz(theta, qubit)
    p_c.rzz(theta, qubit, b)

def add_logical_rzz(theta: float, qubit1: int, qubit2: int, l_c: QuantumCircuit, p_c: QuantumCircuit):
    l_c.rzz(theta, qubit1, qubit2)
    p_c.rzz(theta, qubit1, qubit2)

def add_logical_rx(theta: float, qubit: int, l_c: QuantumCircuit, p_c: QuantumCircuit):
    l_c.rx(theta, qubit)
    p_c.rxx(theta, qubit, t)

def add_logical_ry(theta: float, qubit: int, l_c: QuantumCircuit, p_c: QuantumCircuit):
    add_logical_rx(np.pi/2, qubit, l_c, p_c)
    add_logical_rz(theta, qubit, l_c, p_c)
    add_logical_rx(-np.pi/2, qubit, l_c, p_c)

def add_logical_rxx(theta: float, qubit1: int, qubit2: int, l_c: QuantumCircuit, p_c: QuantumCircuit):
    l_c.rxx(theta, qubit1, qubit2)
    p_c.rxx(theta, qubit1, qubit2)

def decode(statevec: Statevector) -> np.ndarray:
    ins_list = QuantumCircuit(n)
    for i in range(k):
        ins_list.cx(i, t)
    ins_list.cx(b, t)
    for i in range(k):
        ins_list.cx(b, i)
    ins_list.h(b)
    ret = statevec.evolve(ins_list)
    a = ret.data.reshape(4, -1)
    assert np.abs(np.real_if_close(a[1:])).sum() < 1e-10
    return a[0]

def do_sequence(seq, l_c: QuantumCircuit, p_c: QuantumCircuit):
    for gate in seq:
        if gate[0] == 'rz':
            add_logical_rz(gate[1], gate[2], l_c, p_c)
        elif gate[0] == 'rx':
            add_logical_rx(gate[1], gate[2], l_c, p_c)
        elif gate[0] == 'ry':
            add_logical_ry(gate[1], gate[2], l_c, p_c)
        elif gate[0] == 'rzz':
            add_logical_rzz(gate[1], gate[2], gate[3], l_c, p_c)
        elif gate[0] == 'rxx':
            add_logical_rxx(gate[1], gate[2], gate[3], l_c, p_c)
        else:
            raise ValueError("Unknown gate")

sequence_length = 10
sequence = []
for i in range(sequence_length):
    op = np.random.choice(['rz', 'rx', 'ry', 'rzz', 'rxx'])
    if op in ['rz', 'rx', 'ry']:
        qubit = np.random.choice([_ for _ in range(k)])
        angle = np.random.uniform(0, 2 * np.pi)
        sequence.append((op, angle, qubit))
    else:
        qubit1, qubit2 = np.random.choice([_ for _ in range(k)], size=2, replace=False)
        angle = np.random.uniform(0, 2 * np.pi)
        sequence.append((op, angle, qubit1, qubit2))
print('Sequence:', sequence)
do_sequence(sequence, logical_circuit, physical_circuit)
add_logical_rx(np.pi/2, 0, logical_circuit, physical_circuit)
add_logical_rx(np.pi/2, 1, logical_circuit, physical_circuit)
physical_state = Statevector(physical_circuit)
logical_state = Statevector(logical_circuit).data
check_syndrome(physical_state)
decode_state = decode(physical_state)
print(decode_state)
print(logical_state)
print(f'All close? : {np.allclose(decode_state, logical_state)}')