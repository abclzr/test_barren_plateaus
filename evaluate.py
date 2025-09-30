import pdb
import pickle
import os
import numpy as np
import time

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper, ParityMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit import QuantumCircuit, transpile

from qiskit_aer import AerSimulator
from qiskit_aer.quantum_info import AerStatevector

from scipy.optimize import minimize
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from vqa.ansatz.UCCSD import TrainableUCCSD
from vqa.ansatz.HWPA import TrainableHWPA
from vqa.ansatz.BeamSplitter import Trainable_HWPA_BeamSplitter


mole_name = 'H2'
mapper_name = 'jordan_wigner'
ansatz_name = 'HWPA'
reps = 3
os.makedirs("training_data", exist_ok=True)
if ansatz_name == 'HWPA':
    filename = f"training_data/{mole_name}_{mapper_name}_{ansatz_name}_{reps}.pickle"
elif ansatz_name == 'UCCSD':
    filename = f"training_data/{mole_name}_{mapper_name}_{ansatz_name}.pickle"
# Load result_cobyla
with open(filename, "rb") as f:
    result_cobyla = pickle.load(f)

print(f"Loaded optimization result from {filename}")
driver_dict = {
    'H2': PySCFDriver(atom="H .0 .0 .0; H .0 .0 0.735", basis='sto3g'),
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

driver = driver_dict[mole_name]
problem = driver.run()
mapper = mapper_dict[mapper_name]
second_q_op = problem.hamiltonian.second_q_op()
qubit_op_before_reduction = mapper.map(second_q_op)

paulis = []
coeffs = []
for pauli, coeff in qubit_op_before_reduction.label_iter():
    paulis.append(pauli)
    coeffs.append(coeff.real)

n_qubits = len(paulis[0])
# train_ansatz(hamiltonian=qubit_op_before_reduction, ansatz=ansatz)
if ansatz_name == 'UCCSD':
    ansatz = TrainableUCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper)
elif ansatz_name == 'HWPA':
    ansatz = Trainable_HWPA_BeamSplitter(problem.num_spatial_orbitals, problem.num_particles, mapper, reps)
ansatz.set_objective_function(paulis, coeffs)
print(ansatz.trainable_ansatz.count_ops())

def objective_function(param_values, noise_rate=0):
    # return ansatz.evaluate_objective_function(param_values)
    return ansatz.evaluate_objective_function_with_noise(param_values, noise_rate)

# Convert parameters to a numpy array for optimization
initial_params = np.array([0. for param in ansatz.parameters()])

# Define a wrapper for the optimizer
def scipy_objective(params):
    param_dict = dict(zip(ansatz.parameters(), params))
    ret = objective_function(param_dict)
    print(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}, Current parameters: {params}, Objective function value: {ret}")
    return ret

print(f'Hartree Fock Energy: {scipy_objective(initial_params)}')

print(f'Nuclear repulsion energy: {problem.nuclear_repulsion_energy}')

# Update map_param_values with optimized parameters from COBYLA
optimized_params_cobyla = dict(zip(ansatz.parameters(), result_cobyla.x))
print("COBYLA Optimized Parameters:", optimized_params_cobyla)
results = []
for noise_rate in [0, 0.00001, 0.0001, 0.001, 0.01]:
    value = objective_function(optimized_params_cobyla, noise_rate)
    print("COBYLA Optimized Objective Function Value:", value, f"with noise rate {noise_rate}")
    results.append((noise_rate, value))
print("All results:", results)
