import pdb
import pickle
import os
import numpy as np
import time
import sys

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
from _4_2_2_code_builder import _4_2_2_Code_Builder, ClassicalRegisterAllocator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vqa.ansatz.UCCSD import TrainableUCCSD

noise_level = 1
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.0004 * noise_level, 1), ['u1', 'u2', 'u3', 's', 'x', 'sdg', 'rx', 'ry', 'rz'])
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.003 * noise_level, 2), ['rzz', 'rxx', 'ryy', 'cx'])
noise_model.add_all_qubit_readout_error(ReadoutError([[1 - 0.003 * noise_level, 0.003 * noise_level], [0.003 * noise_level, 1 - 0.003 * noise_level]]))

backend = AerSimulator(noise_model=noise_model, shots=10000)
mole_name = 'H2'
mapper_name = 'jordan_wigner'
ansatz_name = 'UCCSD'
reps = 3
os.makedirs("training_data", exist_ok=True)
if ansatz_name == 'HWPA':
    filename = f"../training_data/{mole_name}_{mapper_name}_{ansatz_name}_{reps}.pickle"
elif ansatz_name == 'UCCSD':
    filename = f"../training_data/{mole_name}_{mapper_name}_{ansatz_name}.pickle"
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

param2 = result_cobyla.x[2]

circ = QuantumCircuit(12, 20)
clbit_allocator = ClassicalRegisterAllocator(20)
block1_builder = _4_2_2_Code_Builder(circ, 0, 1, 2, 3, 8, 9, clbit_allocator)
block2_builder = _4_2_2_Code_Builder(circ, 4, 5, 6, 7, 10, 11, clbit_allocator)
block1_builder.initialize()
block2_builder.initialize()
block1_builder.logical_X(1)
block2_builder.logical_X(1)

double_excitation_ansatz_operator = ansatz.operators[2]
ps_list = []
for pauli_coeff_pair in double_excitation_ansatz_operator:
    pauli = pauli_coeff_pair.paulis[0]
    coeff = pauli_coeff_pair.coeffs[0]
    ps = []
    for i in range(4):
        if pauli.z[i]:
            if pauli.x[i]:
                ps.append('Y')
            else:
                ps.append('Z')
        else:
            ps.append('X')
    ps_list.append(''.join(ps))
previous_same = []
posterior_same = []
for i, ps in enumerate(ps_list):
    if i == 0:
        previous_same.append([False for _ in range(4)])
    else:
        previous_same.append([ps[j] == ps_list[i-1][j] for j in range(4)])
    if i == len(ps_list) - 1:
        posterior_same.append([False for _ in range(4)])
    else:
        posterior_same.append([ps[j] == ps_list[i+1][j] for j in range(4)])
    
block1_builder.transversal_Hadamard()
block2_builder.transversal_Hadamard()
for i, pauli_coeff_pair in enumerate(double_excitation_ansatz_operator):
    pauli = pauli_coeff_pair.paulis[0]
    coeff = pauli_coeff_pair.coeffs[0]
    # if pauli.z[0] and pauli.x[0]:
    #     block1_builder.logical_RZ(-np.pi / 2, 1)
    # if pauli.z[1] and pauli.x[1]:
    #     block1_builder.logical_RZ(-np.pi / 2, 2)
    # if pauli.z[2] and pauli.x[2]:
    #     block2_builder.logical_RZ(-np.pi / 2, 1)
    # if pauli.z[3] and pauli.x[3]:
    #     block2_builder.logical_RZ(-np.pi / 2, 2)
    if pauli.z[0] and pauli.x[0] and not previous_same[i][0]:
        block1_builder.logical_RX(-np.pi / 2, 2)
    if pauli.z[1] and pauli.x[1] and not previous_same[i][1]:
        block1_builder.logical_RX(-np.pi / 2, 1)
    if pauli.z[2] and pauli.x[2] and not previous_same[i][2]:
        block2_builder.logical_RX(-np.pi / 2, 2)
    if pauli.z[3] and pauli.x[3] and not previous_same[i][3]:
        block2_builder.logical_RX(-np.pi / 2, 1)
    block1_builder.transversal_CNOT(block2_builder)
    block2_builder.logical_RZZ(coeff.real * 2 * param2)
    block1_builder.transversal_CNOT(block2_builder)
    if pauli.z[0] and pauli.x[0] and not posterior_same[i][0]:
        block1_builder.logical_RX(np.pi / 2, 2)
    if pauli.z[1] and pauli.x[1] and not posterior_same[i][1]:
        block1_builder.logical_RX(np.pi / 2, 1)
    if pauli.z[2] and pauli.x[2] and not posterior_same[i][2]:
        block2_builder.logical_RX(np.pi / 2, 2)
    if pauli.z[3] and pauli.x[3] and not posterior_same[i][3]:
        block2_builder.logical_RX(np.pi / 2, 1)
    # if pauli.z[0] and pauli.x[0]:
    #     block1_builder.logical_RZ(np.pi / 2, 1)
    # if pauli.z[1] and pauli.x[1]:
    #     block1_builder.logical_RZ(np.pi / 2, 2)
    # if pauli.z[2] and pauli.x[2]:
    #     block2_builder.logical_RZ(np.pi / 2, 1)
    # if pauli.z[3] and pauli.x[3]:
    #     block2_builder.logical_RZ(np.pi / 2, 2)

block1_builder.transversal_Hadamard()
block2_builder.transversal_Hadamard()

circ = transpile(circ, optimization_level=3, basis_gates=['rzz', 'h', 'rz', 'x', 'cx', 'swap', 'rxx'])
print(circ)
final_expval = 0
for pauli, coeff in zip(paulis, coeffs):
    circuit_with_measure_basis = QuantumCircuit(circ.num_qubits, circ.num_clbits)
    circuit_with_measure_basis = circuit_with_measure_basis.compose(circ)
    clbit_allocator_with_measure_basis = clbit_allocator.copy()
    block1_builder_with_measure_basis = block1_builder.copy_for_a_new_circuit(circuit_with_measure_basis, clbit_allocator_with_measure_basis)
    block2_builder_with_measure_basis = block2_builder.copy_for_a_new_circuit(circuit_with_measure_basis, clbit_allocator_with_measure_basis)
    map_codeblock_and_qubit = {
        0:(block1_builder_with_measure_basis, 1),
        1:(block1_builder_with_measure_basis, 2),
        2:(block2_builder_with_measure_basis, 1),
        3:(block2_builder_with_measure_basis, 2)
    }
    for i, p in enumerate(pauli[::-1]):
        if p == 'I' or p == 'Z':
            continue
        elif p == 'X':
            builder, qubit = map_codeblock_and_qubit[i]
            builder.logical_RZ(np.pi/2, qubit)
            builder.logical_RX(np.pi/2, qubit)
        elif p == 'Y':
            builder, qubit = map_codeblock_and_qubit[i]
            builder.logical_RX(np.pi/2, qubit)
    cbits1 = block1_builder_with_measure_basis.measurement()
    cbits2 = block2_builder_with_measure_basis.measurement()

    def analyze_counts(counts, pauli):
        expval = 0
        sum_total = 0
        sum_discard = 0
        sum_keep = 0
        for bitstring, count in counts.items():
            bit_val = 1
            # print("Bitstring:", bitstring)
            decode1 = block1_builder_with_measure_basis.decode(bitstring)
            if decode1 == 'invalid':
                print(bitstring)
                sum_discard += count
                continue
            decode2 = block2_builder_with_measure_basis.decode(bitstring)
            if decode2 == 'invalid':
                print(bitstring)
                sum_discard += count
                continue
            decode_bitstring = decode1 + decode2
            for i, p in enumerate(pauli[::-1]):
                if p == 'I':
                    continue
                elif p == 'X' or p == 'Y' or p == 'Z':
                    if decode_bitstring[i] == '1':
                        bit_val *= -1
            expval += bit_val * count
            sum_keep += count
        expval /= sum_keep
        return expval, sum_keep, sum_discard
    circuit_with_measure_basis = transpile(circuit_with_measure_basis, optimization_level=3, basis_gates=['rzz', 'h', 'rz', 'x', 'cx', 'swap', 'rxx'])

    result = backend.run(circuit_with_measure_basis).result()
    expval, sum_keep, sum_discard = analyze_counts(result.get_counts(), pauli)
    final_expval += expval * coeff
    print(f"Pauli: {pauli}, Coeff: {coeff}, Expval: {expval} Shots kept: {sum_keep}, Shots discarded: {sum_discard}")

print(f"Final expval: {final_expval}")
exit()
ansatz = TrainableUCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper)
ansatz.set_objective_function(paulis, coeffs)

def objective_function(param_values, noise_model=None):
    # return ansatz.evaluate_objective_function(param_values)
    return ansatz.evaluate_objective_function_with_noise(param_values, noise_model)


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
print("UCCSD: ", objective_function(optimized_params_cobyla, NoiseModel()))
exit()
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