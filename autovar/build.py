import pdb
import numpy as np
from clapton.clifford import *
from qiskit.circuit.library import HGate, RXGate, RZGate, RYGate, CZGate, CXGate, SwapGate
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def build_circuit_from_clifford(paramed_clifford_circuit: ParametrizedCliffordCircuit) -> tuple[QuantumCircuit, list[Parameter], list[float]]:
    n_qubits = paramed_clifford_circuit.num_physical_qubits
    qiskit_circuit = QuantumCircuit(n_qubits)
    param_cnt = 0
    param_list = []
    initial_point = []
    for gate in paramed_clifford_circuit.gates:
        if isinstance(gate, ParametrizedRXClifford):
            if gate.is_fixed():
                qiskit_circuit.rx(gate.k * np.pi/2, gate.qbs[0])
            else:
                param_list.append(Parameter('theta_' + str(param_cnt)))
                initial_point.append(gate.k * np.pi/2)
                param_cnt += 1
                qiskit_circuit.rx(param_list[-1], gate.qbs[0])
        elif isinstance(gate, ParametrizedRYClifford):
            if gate.is_fixed():
                qiskit_circuit.ry(gate.k * np.pi/2, gate.qbs[0])
            else:
                param_list.append(Parameter('theta_' + str(param_cnt)))
                initial_point.append(gate.k * np.pi/2)
                param_cnt += 1
                qiskit_circuit.ry(param_list[-1], gate.qbs[0])
        elif isinstance(gate, ParametrizedRZClifford):
            if gate.is_fixed():
                qiskit_circuit.rz(gate.k * np.pi/2, gate.qbs[0])
            else:
                param_list.append(Parameter('theta_' + str(param_cnt)))
                initial_point.append(gate.k * np.pi/2)
                param_cnt += 1
                qiskit_circuit.rz(param_list[-1], gate.qbs[0])
        elif isinstance(gate, Parametrized2QClifford):
            if gate.k == 0:
                pass
            elif gate.k == 1:
                qiskit_circuit.cx(gate.qbs[0], gate.qbs[1])
            elif gate.k == 2:
                qiskit_circuit.cz(gate.qbs[1], gate.qbs[0])
            elif gate.k == 3:
                qiskit_circuit.swap(gate.qbs[0], gate.qbs[1])
        else:
            raise ValueError(f"Unknown gate type {type(gate)}")
    return qiskit_circuit, param_list, initial_point

def get_circular_BSansatz(N, reps=1, fix_2q=False, initial_state=None):
    pcirc = ParametrizedCliffordCircuit()
    if initial_state is not None:
        for i, bit in enumerate(initial_state):
            if bit:
                pcirc.RX(i).fix(2)

    for _ in range(reps):
        for i in range(N):
            pcirc.RY(i)
        for i in range(N):
            pcirc.RZ(i)
        for i in range(N):
            control = (i-1) % N
            target = i
            if fix_2q:
                pcirc.Q2(control, target).fix(1)
            else:
                pcirc.Q2(control, target)
    for i in range(N):
        pcirc.RY(i)
    for i in range(N):
        pcirc.RZ(i)
    return pcirc

def apply_BS_gate(circuit: QuantumCircuit, control: int, target: int, param: Parameter):
    """
    Apply a BS gate to the circuit.
    """
    circuit.rz(-3 * np.pi/4, target)
    circuit.cx(target, control)
    circuit.rz(np.pi/2, control)
    circuit.rz(param/2, control)
    circuit.ry(-np.pi/2, target)
    circuit.ry(param/2, target)
    circuit.cx(control, target)
    circuit.ry(np.pi/2, target)
    circuit.ry(-param/2, target)
    circuit.cx(target, control)
    circuit.rz(np.pi/2, control)
    circuit.rz(np.pi/4, target)
    

def build_BS_ansatz(N, reps=1, fix_2q=False, initial_state=None) -> tuple[QuantumCircuit, list[Parameter], list[float]]:
    qiskit_circuit = QuantumCircuit(N)
    if initial_state is not None:
        for i, bit in enumerate(initial_state):
            if bit:
                qiskit_circuit.x(i)
    param_cnt = 0
    param_list = []
    initial_point = []
    assert N % 2 == 0, "N must be even"
    reverse_flag = False
    for _ in range(reps):
        reverse_flag = not reverse_flag
        for i in range(0, N-1, 2):
            param_list.append(Parameter('theta_' + str(param_cnt)))
            param_cnt += 1
            initial_point.append(0)
            if reverse_flag:
                apply_BS_gate(qiskit_circuit, i, i+1, param_list[-1])
            else:
                apply_BS_gate(qiskit_circuit, i+1, i, param_list[-1])
        if (N-1) % 2 == 0:
            param_list.append(Parameter('theta_' + str(param_cnt)))
            param_cnt += 1
            initial_point.append(0)
            if reverse_flag:
                apply_BS_gate(qiskit_circuit, N-1, 0, param_list[-1])
            else:
                apply_BS_gate(qiskit_circuit, 0, N-1, param_list[-1])
        for i in range(1, N-1, 2):
            param_list.append(Parameter('theta_' + str(param_cnt)))
            param_cnt += 1
            initial_point.append(0)
            if reverse_flag:
                apply_BS_gate(qiskit_circuit, i, i+1, param_list[-1])
            else:
                apply_BS_gate(qiskit_circuit, i+1, i, param_list[-1])
        if (N-1) % 2 == 1:
            param_list.append(Parameter('theta_' + str(param_cnt)))
            param_cnt += 1
            initial_point.append(0)
            if reverse_flag:
                apply_BS_gate(qiskit_circuit, N-1, 0, param_list[-1])
            else:
                apply_BS_gate(qiskit_circuit, 0, N-1, param_list[-1])
    return qiskit_circuit, param_list, initial_point