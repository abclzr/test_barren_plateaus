from tqdm import tqdm
import os
import pickle
import pdb
import ast
from tqdm import tqdm
import random
import numpy as np
import cotengra as ctg
# SciPy minimizer routine
from scipy.optimize import minimize

# Plotting functions
import matplotlib.pyplot as plt

from variational_circuits.build_circuit import build_ansatz, sample_gene
from variational_circuits.build_problem import build_problem, plot_training_curve, train_ansatz
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run variational quantum eigensolver.')
    parser.add_argument('--molecule', type=str, required=True, help='Name of the molecule')
    args = parser.parse_args()

    molecule_name = args.molecule
    hamiltonian = build_problem(molecule_name)
    population = 100
    
    num_qubits = hamiltonian.num_qubits
    for _ in tqdm(range(population)):
        gene = sample_gene(num_qubits, 'ry', 'rzz', 3)
        ansatz = build_ansatz(num_qubits, 'ry', 'rzz', 3, gene)
        
        dir = f'experiment_data/{molecule_name}_ryrzz/ansatz_' + str(_)
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        ansatz.draw(output='mpl', filename=os.path.join(dir, 'ansatz.png'))
        
        res, cost_history_dict = train_ansatz(hamiltonian, ansatz)
        
        # Save data to a pickle file
        with open(os.path.join(dir, 'results.pkl'), 'wb') as f:
            pickle.dump({
                'cost_history_dict': cost_history_dict,
                'optimized_params': res.x,
                'final_cost': res.fun,
                'gene': gene
            }, f)

        # Save data to a plain text file
        with open(os.path.join(dir, 'results.txt'), 'w') as f:
            f.write(f"Gene: {gene}\n")
            f.write(f"Final Cost: {res.fun}\n")
            f.write(f"Optimized Parameters: {res.x}\n")
            f.write(f"Cost History: {cost_history_dict['cost_history']}\n")
        
        plot_training_curve(dir)