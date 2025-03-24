import pdb
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from qiskit.circuit import Parameter
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA, SLSQP, COBYLA
from qiskit_aer.primitives import Estimator

def get_expectation_value(circuit: QuantumCircuit, param_list: list[Parameter], initial_point: list[float], paulis: list[str], coeffs: list[float]) -> float:
    circuit = circuit.assign_parameters(dict(zip(param_list, initial_point)))
    statevec = Statevector.from_instruction(circuit)
    total_energy = 0
    for pauli, coeff in zip(paulis, coeffs):
        p = Pauli(pauli[::-1])
        expectation = statevec.expectation_value(p).real
        total_energy += expectation * coeff
    return total_energy

def calc_ground_state_energy(problem, mapper):
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
    optimizer = COBYLA()#SLSQP(maxiter=10)
    noiseless_estimator = Estimator(approximation=True)
    vqe = VQE(noiseless_estimator, ansatz, optimizer)
    vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op_before_reduction)
    vqe_result = problem.interpret(vqe_calc).total_energies
    return vqe_result