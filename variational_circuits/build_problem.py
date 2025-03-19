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

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.converters import circuit_to_dag, dag_to_circuit

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper, ParityMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_aer import AerSimulator
from qiskit_aer.quantum_info import AerStatevector
from pgmQC.model.tensorRV_network_builder import TensorRVNetworkBuilder
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator

from variational_circuits.build_circuit import build_ansatz, sample_gene

def cost_func(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance
        cost_history_dict: Dictionary for storing intermediate results

    Returns:
        float: Energy estimate
    """
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")

    return energy
def build_problem(problem_name):
    if problem_name == 'H2':
        problem = PySCFDriver(atom="H .0 .0 -0.6614; H .0 .0 0.6614", basis='sto3g').run()
    if problem_name == 'LiH':
        problem = PySCFDriver(atom="Li .0 .0 .0; H .0 .0 1.3", basis='sto3g').run()

    # service = QiskitRuntimeService(channel="ibm_quantum")
    # backend = service.least_busy(operational=True, simulator=False)
    # mapper = JordanWignerMapper()
    mapper = ParityMapper()
    # Define the Hamiltonian
    second_q_op = problem.hamiltonian.second_q_op()
    qubit_op = mapper.map(second_q_op)
    hamiltonian = qubit_op
    return hamiltonian

def train_ansatz(hamiltonian, ansatz):
    backend = AerSimulator()
    # ansatz = UCCSD(
    #     problem.num_spatial_orbitals,
    #     problem.num_particles,
    #     mapper,
    #     initial_state=HartreeFock(
    #         problem.num_spatial_orbitals,
    #         problem.num_particles,
    #         mapper,
    #     ),
    # )
    # ansatz = TwoLocal(4, 'ry', 'rzz', 'linear', reps=5, insert_barriers=True)
    # ansatz = transpile(ansatz, basis_gates=['u1', 'u2', 'u3', 'cx'])\
    num_params = ansatz.num_parameters
    x0 = 2 * np.pi * np.zeros(num_params)#np.random.random(num_params)
    
    global cost_history_dict
    cost_history_dict = {
        "prev_vector": None,
        "iters": 0,
        "cost_history": [],
    }
    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        # estimator.options.default_shots = 10000

        res = minimize(
            cost_func,
            x0,
            args=(ansatz, hamiltonian, estimator),
            method="cobyla",
        )
    return res, cost_history_dict


from tqdm import tqdm
import os
import pickle

def plot_training_curve(dir):
    # Plot the training curve
    plt.figure()
    plt.plot(cost_history_dict['cost_history'])
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Training Curve')
    plt.savefig(os.path.join(dir, 'training_curve.png'))
    plt.close()

if __name__ == '__main__':
    hamiltonian = build_problem('H2')
    population = 100
    
    num_qubits = hamiltonian.num_qubits
    for _ in tqdm(range(population)):
        gene = sample_gene(num_qubits, 'ry', 'rzz', 3)
        ansatz = build_ansatz(num_qubits, 'ry', 'rzz', 3, gene)
        
        dir = 'experiment_data/ryrzz/ansatz_' + str(_)
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        ansatz.draw(output='mpl', filename=os.path.join(dir, 'ansatz.png'))
        
        global cost_history_dict
        cost_history_dict = {
            "prev_vector": None,
            "iters": 0,
            "cost_history": [],
        }
        res = train_ansatz(hamiltonian, ansatz)
        
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
        