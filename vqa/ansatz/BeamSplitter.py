import pdb
import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit
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


def apply_BS_gate(circuit: QuantumCircuit, control: int, target: int, theta: Parameter):
    """
    Apply a BS gate to the circuit.
    """
    circuit.rz(-3/4*np.pi, target)
    circuit.cx(target, control)
    circuit.rz((theta+np.pi)/2, control)
    circuit.ry((theta-np.pi)/2, target)
    circuit.cx(control, target)
    circuit.ry((np.pi-theta)/2, target)
    circuit.cx(target, control)
    circuit.rz(np.pi/2, control)
    circuit.rz(np.pi/4, target)

def build_HWPA_BS(N, reps=1, initial_state=None) -> tuple[QuantumCircuit, list[Parameter], list[float]]:
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

class Trainable_HWPA_BeamSplitter:
    """
    A class to create a trainable HWPA circuit for VQE.
    """
    def __init__(self, num_spatial_orbitals, num_particles, mapper, reps):
        self.num_spatial_orbitals = num_spatial_orbitals
        self.num_particles = num_particles
        self.mapper = mapper
        
        hf=HartreeFock(
            num_spatial_orbitals,
            num_particles,
            mapper,
        )
        self.reps = reps
        self.initial_state = hf._bitstr
        qiskit_circuit, param_list, initial_point = build_HWPA_BS(N=len(hf._bitstr), reps=reps, initial_state=hf._bitstr)

        self.num_qubits = qiskit_circuit.num_qubits
        self.num_clbits = qiskit_circuit.num_qubits
        self.num_parameters = len(param_list)
        self.original_parameters = param_list
        transpiled_ansatz = transpile(qiskit_circuit, optimization_level=3, basis_gates=['rz', 'ry', 'x', 'u3', 'cz', 'cx'])
        try:
            transpiled_ansatz.draw('mpl', filename=f'HWPA_{self.num_qubits}_{reps}.png')
        except Exception as e:
            print(f"Error drawing circuit: {e}")
        self.trainable_ansatz = transpiled_ansatz
            
    def parameters(self):
        return self.original_parameters
    
    def reorganize(self, transpiled_ansatz: QuantumCircuit):
        """
        Reorganize the parameter shared by multiple rz gates to different parameters.
        """
        new_ansatz = QuantumCircuit(transpiled_ansatz.num_qubits)
        new_parameters = []
        new_parameters_cnt = 0
        new_to_old_parameters = {}
        for ins in transpiled_ansatz:
            if ins.operation.name == 'rz':
                new_parameters.append(Parameter(f'θ{new_parameters_cnt}'))
                new_parameters_cnt += 1
                old_parameter = list(ins.operation.params[0].parameters)[0]
                new_to_old_parameters[new_parameters[-1]] = old_parameter
                new_ansatz.rz(ins.operation.params[0].subs({old_parameter: new_parameters[-1]}), ins.qubits[0])
            else:
                new_ansatz.append(ins, ins.qubits)
        return new_ansatz, new_parameters, new_to_old_parameters

    
    def set_objective_function(self, paulis: list[str], coeffs: list[float]):
        """
        Set the objective function for the ansatz.
        """
        self.paulis = paulis
        self.coeffs = coeffs
        
    def _evaluate_by_trainable_ansatz(self, param_dict: dict[Parameter, float])-> float:
        parameterized_ansatz = self.trainable_ansatz.assign_parameters(param_dict)
        statevec = Statevector.from_instruction(parameterized_ansatz)
        total_energy = 0
        for pauli, coeff in zip(self.paulis, self.coeffs):
            p = Pauli(pauli)
            expectation = statevec.expectation_value(p).real
            total_energy += expectation * coeff
        return total_energy

    def _evaluate_by_trainable_ansatz_with_noise(self, param_dict: dict[Parameter, float], noise_model: NoiseModel)-> float:

        parameterized_ansatz = self.trainable_ansatz.assign_parameters(param_dict)

        # Print noise model info
        print(noise_model)
        backend = AerSimulator(noise_model=noise_model, shots=10000)
        ret = 0
        for pauli, coeff in zip(self.paulis, self.coeffs):
            circuit_with_measure_basis = QuantumCircuit(self.num_qubits+1, self.num_clbits+1)
            circuit_with_measure_basis = circuit_with_measure_basis.compose(parameterized_ansatz)
            def cx_sequence(circuit, sequence):
                for control in sequence:
                    circuit.cx(control, control+1)
            cx_sequence(circuit_with_measure_basis, list(range(self.num_qubits)))
            circuit_with_measure_basis.reset(self.num_qubits)
            circuit_with_measure_basis.measure(self.num_qubits, self.num_clbits)
            cx_sequence(circuit_with_measure_basis, list(range(self.num_qubits-1, -1, -1)))
            for i, p in enumerate(pauli[::-1]):
                if p == 'I' or p == 'Z':
                    circuit_with_measure_basis.measure(i, i)
                    continue
                elif p == 'X':
                    circuit_with_measure_basis.h(i)
                    circuit_with_measure_basis.measure(i, i)
                elif p == 'Y':
                    circuit_with_measure_basis.sdg(i)
                    circuit_with_measure_basis.h(i)
                    circuit_with_measure_basis.measure(i, i)
            result = backend.run(circuit_with_measure_basis).result()
            def analyze_counts(counts, pauli):
                expval = 0
                sum_values = 0
                for bitstring, count in counts.items():
                    bit_val = 1
                    if bitstring[0] == '1':
                        continue
                    for i, p in enumerate(pauli[::-1]):
                        if p == 'I':
                            continue
                        elif p == 'X' or p == 'Y' or p == 'Z':
                            if bitstring[-i-1] == '1':
                                bit_val *= -1
                    expval += bit_val * count
                    sum_values += count
                expval /= sum_values
                return expval
            counts = result.get_counts()
            expval = analyze_counts(counts, pauli)
            ret += expval * coeff
            print(f"Pauli: {pauli}, Coeff: {coeff}, Expval: {expval}")
        return ret

    
    def evaluate_objective_function(self, param_dict: dict[Parameter, float])-> float:
        """
        Evaluate the objective function for the ansatz.
        """
        return self._evaluate_by_trainable_ansatz(param_dict)

    def evaluate_objective_function_with_noise(self, param_dict: dict[Parameter, float], noise_model: NoiseModel)-> float:
        """
        Evaluate the objective function for the ansatz.
        """
        return self._evaluate_by_trainable_ansatz_with_noise(param_dict, noise_model)
