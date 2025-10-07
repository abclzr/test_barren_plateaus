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
mapper_name = 'bravyi_kitaev'
ansatz_name = 'UCCSD'
method = 'BFGS'
reps = 83

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

def objective_function(param_values):
    return ansatz.evaluate_objective_function(param_values)

def gradient_function(param_values):
    return ansatz.calculate_gradient(param_values)

# Initialize parameters
map_param_values = {param: 0.0 for param in ansatz.parameters()}

# Convert parameters to a numpy array for optimization
initial_params = np.array(list(map_param_values.values()))

# Define a wrapper for the optimizer
def scipy_objective(params):
    param_dict = dict(zip(map_param_values.keys(), params))
    ret = objective_function(param_dict)
    print(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}, Current parameters: {params}, Objective function value: {ret}")
    if not hasattr(scipy_objective, "min_ret") or ret < scipy_objective.min_ret:
        scipy_objective.min_ret = ret
        scipy_objective.min_params = params.copy()
    return ret

def scipy_gradient(params):
    param_dict = dict(zip(map_param_values.keys(), params))
    grad_dict = gradient_function(param_dict)
    return np.array([grad_dict[param] for param in map_param_values.keys()])

print(f'Nuclear repulsion energy: {problem.nuclear_repulsion_energy}')
# Perform COBYLA optimization
start_cobyla = time.time()

result_cobyla = minimize(
    fun=scipy_objective,
    x0=initial_params,
    method=method,
    options={'disp': True},
)
end_cobyla = time.time()
print(f"COBYLA optimization completed in {end_cobyla - start_cobyla:.2f} seconds")

# Update map_param_values with optimized parameters from COBYLA
optimized_params_cobyla = dict(zip(map_param_values.keys(), result_cobyla.x))
print("COBYLA Optimized Parameters:", optimized_params_cobyla)
print("COBYLA Optimized Objective Function Value:", result_cobyla.fun)
print("Hartree Fock Energy:", scipy_objective(np.zeros(len(ansatz.parameters()))))
os.makedirs("training_data", exist_ok=True)
if ansatz_name == "HWPA":
    output_path = f"training_data/{mole_name}_{mapper_name}_{ansatz_name}_{reps}.pickle"
elif ansatz_name == "UCCSD":
    output_path = f"training_data/{mole_name}_{mapper_name}_{ansatz_name}.pickle"
with open(output_path, "wb") as f:
    pickle.dump(result_cobyla, f)
print(f"COBYLA result saved to {output_path}")
print("Minimum objective function value:", scipy_objective.min_ret)
print("Number of observable Pauli strings: ", len(paulis))
print(paulis)
print(coeffs)
# Perform gradient descent optimization
# result = minimize(
#     fun=scipy_objective,
#     x0=initial_params,
#     jac=scipy_gradient,
#     method='BFGS',
#     callback=lambda x: print(f"Current parameters: {x}, Objective function value: {scipy_objective(x)}"),
# )
# # Update map_param_values with optimized parameters
# optimized_params = dict(zip(map_param_values.keys(), result.x))
# print("Optimized Parameters:", optimized_params)
# print("Optimized Objective Function Value:", result.fun)