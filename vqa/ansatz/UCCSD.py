import pdb
import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from qiskit.circuit import Parameter
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit import transpile


class TrainableUCCSD:
    """
    A class to create a trainable UCCSD ansatz circuit for VQE.
    """
    def __init__(self, num_spatial_orbitals, num_particles, mapper):
        self.num_spatial_orbitals = num_spatial_orbitals
        self.num_particles = num_particles
        self.mapper = mapper

        # Define the ansatz circuit
        ansatz = UCCSD(
            num_spatial_orbitals,
            num_particles,
            mapper,
            initial_state=HartreeFock(
                num_spatial_orbitals,
                num_particles,
                mapper,
            ),
        )
        self.num_qubits = ansatz.num_qubits
        self.num_parameters = ansatz.num_parameters
        self.original_parameters = list(ansatz.parameters)
        transpiled_ansatz = transpile(ansatz, optimization_level=3, basis_gates=['rz', 'u3', 'cx'])
        self.trainable_ansatz, self.trainable_parameters, self.trainable_to_original_params_map = \
            self.reorganize(transpiled_ansatz)
    
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

    def evaluate_objective_function(self, param_dict: dict[Parameter, float])-> float:
        """
        Evaluate the objective function for the ansatz.
        """
        trainable_param_values = {}
        for new_param, old_param in self.trainable_to_original_params_map.items():
            value = param_dict[old_param]
            trainable_param_values[new_param] = value
        return self._evaluate_by_trainable_ansatz(trainable_param_values)
    
    def calculate_gradient(self, param_dict: dict[Parameter, float])->dict[Parameter, float]:
        trainable_param_values = {}
        for new_param, old_param in self.trainable_to_original_params_map.items():
            value = param_dict[old_param]
            trainable_param_values[new_param] = value
        
        grad_old_params = {old_param: 0. for old_param in param_dict.keys()}
        for new_param, old_param in tqdm(self.trainable_to_original_params_map.items()):
            trainable_param_values[new_param] += np.pi/2
            f1 = self._evaluate_by_trainable_ansatz(trainable_param_values)
            trainable_param_values[new_param] -= np.pi
            f2 = self._evaluate_by_trainable_ansatz(trainable_param_values)
            trainable_param_values[new_param] += np.pi/2
            grad = (f1 - f2)/2
            grad_old_params[old_param] += grad
        return grad_old_params