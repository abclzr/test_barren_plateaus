import pdb
import numpy as np

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper, ParityMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

from autovar.initialization import initialize
from autovar.build import build_circuit_from_clifford, build_BS_ansatz
from autovar.evaluation import get_expectation_value, calc_ground_state_energy
from clapton.clapton import claptonize
from clapton.ansatzes import circular_ansatz
from clapton.clifford import ParametrizedCliffordCircuit

from scipy.optimize import minimize



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

driver = driver_dict['LiH']
problem = driver.run()
mapper = mapper_dict['jordan_wigner']
second_q_op = problem.hamiltonian.second_q_op()
qubit_op_before_reduction = mapper.map(second_q_op)

paulis = []
coeffs = []
for pauli, coeff in qubit_op_before_reduction.label_iter():
    paulis.append(pauli)
    coeffs.append(coeff.real)

n_qubits = len(paulis[0])

from vqa.ansatz.UCCSD import TrainableUCCSD

ansatz = TrainableUCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper)
ansatz.set_objective_function(paulis, coeffs)
def objective_function(param_values):
    return ansatz.evaluate_objective_function(param_values)

def gradient_function(param_values):
    return ansatz.calculate_gradient(param_values)

# Initialize parameters
map_param_values = {param: 0. for param in ansatz.parameters()}

# Convert parameters to a numpy array for optimization
initial_params = np.array(list(map_param_values.values()))

# Define a wrapper for the optimizer
def scipy_objective(params):
    param_dict = dict(zip(map_param_values.keys(), params))
    return objective_function(param_dict)

def scipy_gradient(params):
    param_dict = dict(zip(map_param_values.keys(), params))
    grad_dict = gradient_function(param_dict)
    return np.array([grad_dict[param] for param in map_param_values.keys()])

print(f'Hartree Fock Energy: {scipy_objective(initial_params)}')

# Perform gradient descent optimization
result = minimize(
    fun=scipy_objective,
    x0=initial_params,
    jac=scipy_gradient,
    method='BFGS',
    callback=lambda x: print(f"Current parameters: {x}, Objective function value: {scipy_objective(x)}"),
)

# Update map_param_values with optimized parameters
optimized_params = dict(zip(map_param_values.keys(), result.x))
print("Optimized Parameters:", optimized_params)
print("Optimized Objective Function Value:", result.fun)