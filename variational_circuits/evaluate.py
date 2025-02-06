import pdb
import ast
import os
from tqdm import tqdm
import random
import pickle
import numpy as np
import cotengra as ctg
import matplotlib.pyplot as plt
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
from variational_circuits.build_problem import build_problem
from variational_circuits.build_circuit import build_ansatz
import time

def evaluate_ansatz(ansatz, hamiltonian):
    mole_ps = []
    for pauli in hamiltonian.paulis:
        mole_ps.append(str(pauli))
    mole_weights = []
    for coeff in hamiltonian.coeffs:
        mole_weights.append(coeff.real)
    mole_table = dict(zip(mole_ps, mole_weights))
    num_qubits = len(hamiltonian.paulis[0])
    
    dag = circuit_to_dag(ansatz)
    dag.draw(filename='dag.png')
    builder = TensorRVNetworkBuilder(dag)
    true_false_network, uncontracted_nodes, tensorRV_list = builder.build()
    tensorBoolean = builder.find_models()
    
    ansatz_ps = tensorBoolean.paulistrings()
    expressivity = len(ansatz_ps) / (4 ** hamiltonian.num_qubits)
    print(f'Expressivity: {expressivity}')
    expressivity_on_mole_paulistrings = len(set(ansatz_ps) & set(mole_ps)) / len(mole_ps)
    print(f'Expressivity on molecule: {expressivity_on_mole_paulistrings}')

    trainability_as_an_universal_ansatz = 0
    var_on_mole_paulistrings = 0
    mean_on_mole_paulistrings = 0
    skip_cov_propagation = False
    if not skip_cov_propagation:
        print('Contracting tensors...')
        start_time = time.time()
        tensorRV_results = builder.contract_tensors(obs=mole_ps)
        end_time = time.time()
        print(f'Tensors contracted. Time taken: {end_time - start_time} seconds')
        for tensorRV, ps in zip(tensorRV_results, mole_ps):
            # tensorRV are all scalar in this case when you put 'obs=mole_ps' in 'contract_tensors'
            weight = mole_table[ps]
            var_on_mole_paulistrings += tensorRV.variance_of_trace_as_scalar(num_qubits) * weight * weight
            trainability_as_an_universal_ansatz += tensorRV.variance_of_trace_as_scalar(num_qubits)
            # calculate mean on molecule paulistrings
            mean_on_mole_paulistrings += tensorRV.mean_of_trace_as_scalar(num_qubits) * mole_table[ps]
    # mu minus 3 sigma is the estimated lowest energy
    estimated_lowest_energy = mean_on_mole_paulistrings - 3 * np.sqrt(var_on_mole_paulistrings)
    print(f'Trainability as an universal ansatz: {trainability_as_an_universal_ansatz}')
    print(f'Variance on molecule paulistrings: {var_on_mole_paulistrings}')
    print(f'Estimated lowest energy: {estimated_lowest_energy}')
    return {
        'expressivity': expressivity,
        'expressivity_on_mole_paulistrings': expressivity_on_mole_paulistrings,
        'trainability_as_an_universal_ansatz': trainability_as_an_universal_ansatz,
        'var_on_mole_paulistrings': var_on_mole_paulistrings,
        'estimated_lowest_energy': estimated_lowest_energy
    }

if __name__ == '__main__':
    molecule_name = 'H2'
    hamiltonian = build_problem(molecule_name)
    num_qubits = len(hamiltonian.paulis[0])
    population = 100

    data_list = []
    gene_list = []
    optimized_params_list = []
    final_cost_list = []
    
    # Initialize lists to store metrics
    expressivity_list = []
    expressivity_on_mole_paulistrings_list = []
    trainability_as_an_universal_ansatz_list = []
    variance_on_mole_paulistrings_list = []
    estimated_lowest_energy_list = []
    
    for _ in tqdm(range(population)):
        dir = f'experiment_data/{molecule_name}_ryrzz/ansatz_' + str(_)
        
        # load data from a pickle file
        with open(os.path.join(dir, 'results.pkl'), 'rb') as f:
            data = pickle.load(f)
            gene = data['gene']
            optimized_params = data['optimized_params']
            final_cost = data['final_cost']
            cost_history_dict = data['cost_history_dict']
            data_list.append(data)
            gene_list.append(gene)
            optimized_params_list.append(optimized_params)
            final_cost_list.append(final_cost)
        
        metric_file_path = os.path.join(dir, 'ansatz_metric.pkl')
        # if os.path.exists(metric_file_path):
        if False:
            with open(metric_file_path, 'rb') as metric_file:
                ansatz_metric = pickle.load(metric_file)
        else:
            ansatz = build_ansatz(num_qubits, 'ry', 'rzz', 3, gene)
            ansatz_metric = evaluate_ansatz(ansatz, hamiltonian)
            with open(metric_file_path, 'wb') as metric_file:
                pickle.dump(ansatz_metric, metric_file)
        
        expressivity_list.append(ansatz_metric['expressivity'])
        expressivity_on_mole_paulistrings_list.append(ansatz_metric['expressivity_on_mole_paulistrings'])
        trainability_as_an_universal_ansatz_list.append(ansatz_metric['trainability_as_an_universal_ansatz'])
        variance_on_mole_paulistrings_list.append(ansatz_metric['var_on_mole_paulistrings'])
        estimated_lowest_energy_list.append(ansatz_metric['estimated_lowest_energy'])
    
    # sort the data by final cost, return the sorted indices
    sorted_indices = list(reversed(np.argsort(final_cost_list)))

    # Prepare the figure and subplots
    fig, axs = plt.subplots(5, 1, figsize=(10, 20))

    # Define the metrics to plot
    metrics = ['expressivity', 'expressivity_on_mole_paulistrings', 'trainability_as_an_universal_ansatz', 'variance_on_mole_paulistrings', 'estimated_lowest_energy']
    right_y_labels = ['Expressivity', 'Expressivity on Molecule', 'Trainability as an Universal Ansatz', 'Variance on Molecule', 'Estimated Lowest Energy']

    sorted_final_cost_list = [final_cost_list[i] for i in sorted_indices]
    expressivity_list = [expressivity_list[i] for i in sorted_indices]
    expressivity_on_mole_paulistrings_list = [expressivity_on_mole_paulistrings_list[i] for i in sorted_indices]
    trainability_as_an_universal_ansatz_list = [trainability_as_an_universal_ansatz_list[i] for i in sorted_indices]
    variance_on_mole_paulistrings_list = [variance_on_mole_paulistrings_list[i] for i in sorted_indices]
    estimated_lowest_energy_list = [estimated_lowest_energy_list[i] for i in sorted_indices]

    
    # Plot each metric
    for i, metric in enumerate(metrics):
        axs[i].plot(sorted_final_cost_list, label='Ground State Energy', color='b')
        axs[i].set_ylabel('Energy', color='b')
        axs[i].tick_params(axis='y', labelcolor='b')

        # Create a twin y-axis to plot the metric
        ax2 = axs[i].twinx()
        if metric == 'expressivity':
            ax2.plot(expressivity_list, label=right_y_labels[i], color='r')
        elif metric == 'expressivity_on_mole_paulistrings':
            ax2.plot(expressivity_on_mole_paulistrings_list, label=right_y_labels[i], color='r')
        elif metric == 'trainability_as_an_universal_ansatz':
            ax2.plot(trainability_as_an_universal_ansatz_list, label=right_y_labels[i], color='r')
        elif metric == 'variance_on_mole_paulistrings':
            ax2.plot(variance_on_mole_paulistrings_list, label=right_y_labels[i], color='r')
        elif metric == 'estimated_lowest_energy':
            ax2.plot(estimated_lowest_energy_list, label=right_y_labels[i], color='r')
        
        ax2.set_ylabel(right_y_labels[i], color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Set the title for each subplot
        axs[i].set_title(f'Ground State Energy vs {right_y_labels[i]}')

    # Set the x-axis label for the last subplot
    axs[-1].set_xlabel('Sorted Index')

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Save the figure
    plt.savefig(f'{molecule_name}_metrics_vs_ground_state_energy.png')
    plt.close()