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

def apply_HW_gate(circuit: QuantumCircuit, control: int, target: int, theta: Parameter, phi: Parameter):
    """
    Apply a HW gate to the circuit.
    """
    circuit.cx(target, control)
    
    circuit.ry(-theta-np.pi/2, target)
    circuit.rz(-phi-np.pi, target)
    
    circuit.cx(control, target)
    
    circuit.rz(phi+np.pi, target)
    circuit.ry(theta+np.pi/2, target)
    
    circuit.cx(target, control)

    
def apply_RBS_gate(circuit: QuantumCircuit, control: int, target: int, theta: Parameter):
    """
    Apply a RBS gate to the circuit.
    """
    circuit.h(control)
    circuit.h(target)
    circuit.cz(control, target)
    
    circuit.ry(theta/2, control)
    circuit.ry(-theta/2, target)

    circuit.cz(target, control)
    circuit.h(control)
    circuit.h(target)

def build_HWPA(N, reps=1, initial_state=None) -> tuple[QuantumCircuit, list[Parameter], list[float]]:
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
            param_list.append(Parameter('phi_' + str(param_cnt)))
            param_cnt += 1
            initial_point.append(0)
            initial_point.append(0)
            if reverse_flag:
                apply_HW_gate(qiskit_circuit, i, i+1, param_list[-2], param_list[-1])
            else:
                apply_HW_gate(qiskit_circuit, i+1, i, param_list[-2], param_list[-1])
        if (N-1) % 2 == 0:
            param_list.append(Parameter('theta_' + str(param_cnt)))
            param_list.append(Parameter('phi_' + str(param_cnt)))
            param_cnt += 1
            initial_point.append(0)
            initial_point.append(0)
            if reverse_flag:
                apply_HW_gate(qiskit_circuit, N-1, 0, param_list[-2], param_list[-1])
            else:
                apply_HW_gate(qiskit_circuit, 0, N-1, param_list[-2], param_list[-1])
        for i in range(1, N-1, 2):
            param_list.append(Parameter('theta_' + str(param_cnt)))
            param_list.append(Parameter('phi_' + str(param_cnt)))
            param_cnt += 1
            initial_point.append(0)
            initial_point.append(0)
            if reverse_flag:
                apply_HW_gate(qiskit_circuit, i, i+1, param_list[-2], param_list[-1])
            else:
                apply_HW_gate(qiskit_circuit, i+1, i, param_list[-2], param_list[-1])
        if (N-1) % 2 == 1:
            param_list.append(Parameter('theta_' + str(param_cnt)))
            param_list.append(Parameter('phi_' + str(param_cnt)))
            param_cnt += 1
            initial_point.append(0)
            initial_point.append(0)
            if reverse_flag:
                apply_HW_gate(qiskit_circuit, N-1, 0, param_list[-2], param_list[-1])
            else:
                apply_HW_gate(qiskit_circuit, 0, N-1, param_list[-2], param_list[-1])
    return qiskit_circuit, param_list, initial_point

def build_HWPA_RBS(N, reps=1, initial_state=None) -> tuple[QuantumCircuit, list[Parameter], list[float]]:
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
                apply_RBS_gate(qiskit_circuit, i, i+1, param_list[-1])
            else:
                apply_RBS_gate(qiskit_circuit, i+1, i, param_list[-1])
        if (N-1) % 2 == 0:
            param_list.append(Parameter('theta_' + str(param_cnt)))
            param_cnt += 1
            initial_point.append(0)
            if reverse_flag:
                apply_RBS_gate(qiskit_circuit, N-1, 0, param_list[-1])
            else:
                apply_RBS_gate(qiskit_circuit, 0, N-1, param_list[-1])
        for i in range(1, N-1, 2):
            param_list.append(Parameter('theta_' + str(param_cnt)))
            param_cnt += 1
            initial_point.append(0)
            if reverse_flag:
                apply_RBS_gate(qiskit_circuit, i, i+1, param_list[-1])
            else:
                apply_RBS_gate(qiskit_circuit, i+1, i, param_list[-1])
        if (N-1) % 2 == 1:
            param_list.append(Parameter('theta_' + str(param_cnt)))
            param_cnt += 1
            initial_point.append(0)
            if reverse_flag:
                apply_RBS_gate(qiskit_circuit, N-1, 0, param_list[-1])
            else:
                apply_RBS_gate(qiskit_circuit, 0, N-1, param_list[-1])
    return qiskit_circuit, param_list, initial_point

class TrainableHWPA:
    """
    A class to create a trainable HWPA circuit for VQE.
    """
    def __init__(self, num_spatial_orbitals, num_particles, mapper):
        self.num_spatial_orbitals = num_spatial_orbitals
        self.num_particles = num_particles
        self.mapper = mapper
        
        hf=HartreeFock(
            num_spatial_orbitals,
            num_particles,
            mapper,
        )
        reps = 8
        qiskit_circuit, param_list, initial_point = build_HWPA(N=len(hf._bitstr), reps=reps, initial_state=hf._bitstr)

        self.num_qubits = qiskit_circuit.num_qubits
        self.num_parameters = len(param_list)
        self.original_parameters = param_list
        transpiled_ansatz = transpile(qiskit_circuit, optimization_level=3, basis_gates=['rz', 'ry', 'x', 'u3', 'cz', 'cx'])
        transpiled_ansatz.draw('mpl', filename=f'HWPA_{self.num_qubits}_{reps}.png')
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
        print(statevec)
        total_energy = 0
        for pauli, coeff in zip(self.paulis, self.coeffs):
            p = Pauli(pauli)
            expectation = statevec.expectation_value(p).real
            total_energy += expectation * coeff
        return total_energy

    def _evaluate_by_trainable_ansatz_with_noise(self, param_dict: dict[Parameter, float])-> float:
        parameterized_ansatz = self.trainable_ansatz.assign_parameters(param_dict)
        bit_flip = pauli_error([('X', 0.001), ('I', 1 - 0.001)])
        noise_model = NoiseModel()

        # Add depolarizing error to all single qubit u1, u2, u3 gates
        error = bit_flip.tensor(bit_flip)
        noise_model.add_all_qubit_quantum_error(error, ['cx'])

        # Print noise model info
        print(noise_model)

        backend = AerSimulator(noise_model=noise_model)
        with Session(backend=backend) as session:
            estimator = Estimator(mode=session)
            # estimator.options.default_shots = 10000
            pub = (self.trainable_ansatz, [SparsePauliOp(self.paulis, np.array(self.coeffs))], param_dict)
            result = estimator.run(pubs=[pub]).result()
            energy = result[0].data.evs[0]
        return energy
    
    def evaluate_objective_function(self, param_dict: dict[Parameter, float])-> float:
        """
        Evaluate the objective function for the ansatz.
        """
        return self._evaluate_by_trainable_ansatz(param_dict)
    
    def evaluate_objective_function_with_noise(self, param_dict: dict[Parameter, float])-> float:
        """
        Evaluate the objective function for the ansatz.
        """
        return self._evaluate_by_trainable_ansatz_with_noise(param_dict)
