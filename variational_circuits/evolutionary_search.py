import pdb
import ast
from tqdm import tqdm
import random
import numpy as np
import cotengra as ctg
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
from variational_circuits.build_circuit import build_ansatz, sample_gene
import time

def simplify_hamiltonian(hamiltonian):
    labels = hamiltonian.paulis
    coeffs = hamiltonian.coeffs
    
    top_indices = np.argsort(np.abs(coeffs))[-20:]
    top_labels = labels[top_indices]
    top_coeffs = coeffs[top_indices]
    simplified_hamiltonian = SparsePauliOp(top_labels, top_coeffs)
    return simplified_hamiltonian

def evaluate_gene(gene, hamiltonian) -> float:
    print(f"Evaluating gene: {gene}")
    ansatz = build_ansatz(hamiltonian.num_qubits, 'rx', 'cx', 3, gene)
    dag = circuit_to_dag(ansatz)
    builder = TensorRVNetworkBuilder(dag)
    builder.build()
    return builder.estimated_ground_state_energy(hamiltonian)

class GeneticAlgorithm:
    def __init__(self):
        pass
    
    def mutate(self, gene):
        gene = list(gene)
        for i in range(len(gene)):
            if random.random() < mutation_rate:
                gene[i] = random.choice(['0', '1'])
        return ''.join(gene)

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def select_parents(self, population, fitnesses):
        fitnesses = np.array(fitnesses)
        total_fitness = np.sum(np.exp(-fitnesses))
        selection_probs = np.exp(-fitnesses) / total_fitness
        parents = random.choices(population, weights=selection_probs, k=2)
        return parents

    

if __name__ == '__main__':
    # Best gene for H2 is 1000010110010001001100011101
    # Best gene for LiH is 001010111101001001010011111000001001011101001010101011011011000110101100100011011101, Best Fitness = -9.020646678699565
    molecule_name = 'LiH'
    hamiltonian = build_problem(molecule_name)
    num_qubits = len(hamiltonian.paulis[0])
    simplified_hamiltonian = simplify_hamiltonian(hamiltonian)
    
    population = 20
    initial_genes = [sample_gene(num_qubits, 'rx', 'cx', 3) for _ in range(population)]
    
    generations = 50
    mutation_rate = 0.1

    genetic_algorithm = GeneticAlgorithm()
    for generation in range(generations):
        fitnesses = [evaluate_gene(gene, simplified_hamiltonian) for gene in initial_genes]
        new_population = []
        for _ in range(population // 2):
            parent1, parent2 = genetic_algorithm.select_parents(initial_genes, fitnesses)
            child1 = genetic_algorithm.mutate(genetic_algorithm.crossover(parent1, parent2))
            child2 = genetic_algorithm.mutate(genetic_algorithm.crossover(parent2, parent1))
            new_population.extend([child1, child2])
        initial_genes = new_population
        best_gene = initial_genes[np.argmin(fitnesses)]
        best_fitness = min(fitnesses)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    best_gene = initial_genes[np.argmin(fitnesses)]
    print(f"Best gene found: {best_gene}")