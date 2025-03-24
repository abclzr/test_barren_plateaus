import argparse
from qiskit import QuantumCircuit, transpile
import time, sys, os
import numpy as np
import pickle
import pdb
import random
from tqdm import tqdm
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper, ParityMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
import ast
from qiskit.quantum_info import Pauli, Statevector, SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer.quantum_info import AerStatevector

def save_results_to_file(results, filename):
    with open(filename, 'wb') as file:
        pickle.dump(results, file)

def load_results_from_file(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        return {}

def get_variance_of_gradients(problem, mapper, num_trials):
    ansatz = UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
        ),
    )
    
    second_q_op = problem.hamiltonian.second_q_op()
    qubit_op_before_reduction = mapper.map(second_q_op)
    # Only pick the top 20 Paulis, otherwise the simulation will take too long
    top_20_paulis = sorted(qubit_op_before_reduction.to_list(), key=lambda x: abs(x[1]), reverse=True)[:20]
    pauli_list = [Pauli(label) for label, _ in top_20_paulis]
    weight_list = [weight for _, weight in top_20_paulis]
    qubit_op = SparsePauliOp(pauli_list, weight_list)
    
    gradients = []
    for _ in tqdm(range(num_trials)):
        params = [random.uniform(-np.pi, np.pi) for _ in range(ansatz.num_parameters)]
        param_index = random.randint(0, ansatz.num_parameters - 1)
        shift = 3.14159 / 2

        params_shifted_forward = params.copy()
        params_shifted_forward[param_index] += shift

        params_shifted_backward = params.copy()
        params_shifted_backward[param_index] -= shift

        circuit_forward = ansatz.assign_parameters(params_shifted_forward)
        circuit_backward = ansatz.assign_parameters(params_shifted_backward)

        # expectation_forward = Statevector.from_instruction(circuit_forward).expectation_value(qubit_op)
        # expectation_backward = Statevector.from_instruction(circuit_backward).expectation_value(qubit_op)
        expectation_forward = AerStatevector(circuit_forward, device='GPU').expectation_value(qubit_op)
        expectation_backward = AerStatevector(circuit_backward, device='GPU').expectation_value(qubit_op)
        
        gradient = (expectation_forward - expectation_backward) / 2
        gradients.append(gradient)

    mean_gradient = 0
    variance_gradient = np.mean([g**2 for g in gradients])
    
    print(f"Mean of gradients: {mean_gradient}")
    print(f"Variance of gradients: {variance_gradient}")
    return mean_gradient, variance_gradient

if __name__ == '__main__':
    num_trials = 1000
    driver_dict = {
        'LiH': PySCFDriver(atom="Li .0 .0 .0; H .0 .0 1.6", basis='sto3g'),
        'BeH2': PySCFDriver(atom="Be .0 .0 .0; H .0 .0 -1.3; H .0 .0 1.3", basis='sto3g'),
        'CH4': PySCFDriver(atom="C .0 .0 .0; H .0 .0 1.0; H .0 .0 -1.0; H .0 1.0 .0; H .0 -1.0 .0", basis='sto3g'),
        'MgH2': PySCFDriver(atom="Mg .0 .0 .0; H .0 .0 -1.3; H .0 .0 1.3", basis='sto3g'),
        'LiCl': PySCFDriver(atom="Li .0 .0 .0; Cl .0 .0 -1.5", basis='sto3g'),
        'CO2': PySCFDriver(atom="C .0 .0 .0; O .0 .0 1.0; O .0 .0 -1.0", basis='sto3g')
    }
    mapper_dict = {
        'jordan_wigner': JordanWignerMapper(),
        'bravyi_kitaev': BravyiKitaevMapper(),
        'parity': ParityMapper()
    }
    
    results = load_results_from_file('results2.pkl')
    for mole in ['LiH', 'BeH2', 'CH4', 'MgH2', 'LiCl', 'CO2']:
        driver = driver_dict[mole]
        problem = driver.run()
        for mapper_name in ['jordan_wigner', 'bravyi_kitaev', 'parity']:
            mapper = mapper_dict[mapper_name]
            task_name = f'{mole} with {mapper_name} mapper'
            if task_name not in results:
                print(f"Working on {task_name}...")
                mean, variance = get_variance_of_gradients(problem, mapper, num_trials)
                results[task_name] = (mean, variance)
                save_results_to_file(results, 'results2.pkl')
            else:
                print(f"Already computed {task_name}")

    
        