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