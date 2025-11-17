import pdb
import pickle
import os
import numpy as np
import time

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper, ParityMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit import QuantumCircuit, transpile
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)

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
ansatz_name = 'UCCSD'
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
    'LiH': PySCFDriver(atom="Li .0 .0 .0; H .0 .0 1.5699", basis='sto3g'),
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

def objective_function(param_values, noise_model=None):
    # return ansatz.evaluate_objective_function(param_values)
    return ansatz.evaluate_objective_function_with_noise(param_values, noise_model)

# Convert parameters to a numpy array for optimization
initial_params = np.array([0. for param in ansatz.parameters()])

# Define a wrapper for the optimizer
def scipy_objective(params):
    param_dict = dict(zip(ansatz.parameters(), params))
    ret = objective_function(param_dict)
    print(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}, Current parameters: {params}, Objective function value: {ret}")
    return ret

# print(f'Hartree Fock Energy: {scipy_objective(initial_params)}')

print(f'Nuclear repulsion energy: {problem.nuclear_repulsion_energy}')

# Update map_param_values with optimized parameters from COBYLA
optimized_params_cobyla = dict(zip(ansatz.parameters(), result_cobyla.x))
print("COBYLA Optimized Parameters:", optimized_params_cobyla)
results = []
for noise_level in [1e-3, 1e-2, 1e-1, 1]:
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.0004 * noise_level, 1), ['u1', 'u2', 'u3', 's', 'x', 'sdg', 'rx', 'ry', 'rz'])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.003 * noise_level, 2), ['rzz', 'rxx', 'ryy'])
    noise_model.add_all_qubit_readout_error(ReadoutError([[1 - 0.003 * noise_level, 0.003 * noise_level], [0.003 * noise_level, 1 - 0.003 * noise_level]]))
    # noise_model.add_all_qubit_quantum_error(depolarizing_error(0.0004 * noise_level, 1), ['u1', 'u2', 'u3', 'h', 's', 'x', 'sdg', 'rx', 'ry', 'rz'])
    # noise_model.add_all_qubit_quantum_error(depolarizing_error(0.003 * noise_level, 2), ['cx', 'cz', 'rzz', 'rxx', 'ryy'])
    # noise_model.add_all_qubit_readout_error(ReadoutError([[1 - 0.003 * noise_level, 0.003 * noise_level], [0.003 * noise_level, 1 - 0.003 * noise_level]]))
    value = objective_function(optimized_params_cobyla, noise_model)
    print("COBYLA Optimized Objective Function Value:", value)
    results.append((noise_level, value))
print(results)
with open("output.txt", "w") as f:
    f.write(str(results))