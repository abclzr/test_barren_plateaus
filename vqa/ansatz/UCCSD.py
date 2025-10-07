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

from vqa.utils.IcebergCode import add_icebergcode_measurement, convert_to_icebergcode_circuit, convert_measurement_basis


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
        self.num_clbits = ansatz.num_qubits
        self.num_parameters = ansatz.num_parameters
        self.original_parameters = list(ansatz.parameters)
        transpiled_ansatz = transpile(ansatz, optimization_level=3, basis_gates=['rz', 'x', 'u3', 'cx'])
        transpiled_ansatz.draw('mpl', filename=f'UCCSD_{self.num_qubits}.png')
        self.trainable_ansatz = transpiled_ansatz
        self.trainable_parameters = self.original_parameters
        # self.trainable_ansatz, self.trainable_parameters, self.trainable_to_original_params_map = \
        #     self.reorganize(transpiled_ansatz)
    
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
        print("Statevector:", statevec)
        total_energy = 0
        for pauli, coeff in zip(self.paulis, self.coeffs):
            p = Pauli(pauli)
            expectation = statevec.expectation_value(p).real
            total_energy += expectation * coeff
        return total_energy

    def _evaluate_by_trainable_ansatz_with_noise(self, param_dict: dict[Parameter, float], noise_rate: float)-> float:

        parameterized_ansatz = self.trainable_ansatz.assign_parameters(param_dict)
        bit_flip = pauli_error([('X', noise_rate), ('I', 1 - noise_rate)])
        noise_model = NoiseModel()

        # Add depolarizing error to all single qubit u1, u2, u3 gates
        error = bit_flip.tensor(bit_flip)
        noise_model.add_all_qubit_quantum_error(error, ['cx'])

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

    def _evaluate_by_trainable_ansatz_with_code(self, param_dict: dict[Parameter, float], noise_rate: float)-> float:

        parameterized_ansatz = self.trainable_ansatz.assign_parameters(param_dict)
        bit_flip = pauli_error([('X', noise_rate), ('I', 1 - noise_rate)])
        noise_model = NoiseModel()

        # Add depolarizing error to all single qubit u1, u2, u3 gates
        error = bit_flip.tensor(bit_flip)
        noise_model.add_all_qubit_quantum_error(error, ['cx'])

        # Print noise model info
        print(noise_model)
        backend = AerSimulator(noise_model=noise_model, shots=10000)
        ret = 0
        coded_circuit = convert_to_icebergcode_circuit(parameterized_ansatz)
        k = parameterized_ansatz.num_qubits
        t = coded_circuit.num_qubits - 4
        b = coded_circuit.num_qubits - 3
        a1 = coded_circuit.num_qubits - 2
        a2 = coded_circuit.num_qubits - 1
        for pauli, coeff in zip(self.paulis, self.coeffs):
            circuit_with_measure_basis = QuantumCircuit(coded_circuit.num_qubits, coded_circuit.num_qubits)
            circuit_with_measure_basis = circuit_with_measure_basis.compose(coded_circuit)
            convert_measurement_basis(circuit_with_measure_basis, pauli, t, b)
            add_icebergcode_measurement(circuit_with_measure_basis, k, t, b)
            result = backend.run(circuit_with_measure_basis).result()
            def analyze_counts(counts, pauli):
                expval = 0
                sum_values = 0
                for bitstring, count in counts.items():
                    bit_val = 1
                    # print("Bitstring:", bitstring)
                    if bitstring[:2] != '00' or bitstring[-b-1:].count('1') % 2 != 0:
                        continue
                    # assert bitstring[:2] == '00'
                    # assert bitstring[-b-1:].count('1') % 2 == 0
                    if bitstring[-b-1] == '0':
                        bs = bitstring[-t:]
                    else:
                        bs = ''.join('1' if x == '0' else '0' for x in bitstring[-t:])
                    for i, p in enumerate(pauli[::-1]):
                        if p == 'I':
                            continue
                        elif p == 'X' or p == 'Y' or p == 'Z':
                            if bs[-i-1] == '1':
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
        # trainable_param_values = {}
        # for new_param, old_param in self.trainable_to_original_params_map.items():
        #     value = param_dict[old_param]
        #     trainable_param_values[new_param] = value
        return self._evaluate_by_trainable_ansatz(param_dict)

    def evaluate_objective_function_with_noise(self, param_dict: dict[Parameter, float], noise_rate: float)-> float:
        """
        Evaluate the objective function for the ansatz.
        """
        # trainable_param_values = {}
        # for new_param, old_param in self.trainable_to_original_params_map.items():
        #     value = param_dict[old_param]
        #     trainable_param_values[new_param] = value
        return self._evaluate_by_trainable_ansatz_with_code(param_dict, noise_rate)


    # def calculate_gradient(self, param_dict: dict[Parameter, float])->dict[Parameter, float]:
    #     trainable_param_values = {}
    #     for new_param, old_param in self.trainable_to_original_params_map.items():
    #         value = param_dict[old_param]
    #         trainable_param_values[new_param] = value
        
    #     grad_old_params = {old_param: 0. for old_param in param_dict.keys()}
    #     for new_param, old_param in tqdm(self.trainable_to_original_params_map.items()):
    #         trainable_param_values[new_param] += np.pi/2
    #         f1 = self._evaluate_by_trainable_ansatz(trainable_param_values)
    #         trainable_param_values[new_param] -= np.pi
    #         f2 = self._evaluate_by_trainable_ansatz(trainable_param_values)
    #         trainable_param_values[new_param] += np.pi/2
    #         grad = (f1 - f2)/2
    #         grad_old_params[old_param] += grad
    #     return grad_old_params