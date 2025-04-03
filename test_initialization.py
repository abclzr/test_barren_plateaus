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

def get_circular_ansatz(N, reps=1, fix_2q=False, initial_state=None):
    pcirc = ParametrizedCliffordCircuit()
    if initial_state is not None:
        for i, bit in enumerate(initial_state):
            if bit:
                pcirc.RX(i).fix(2)

    for _ in range(reps):
        for i in range(N):
            pcirc.RY(i)
        for i in range(N):
            pcirc.RZ(i)
        for i in range(N):
            control = (i-1) % N
            target = i
            if fix_2q:
                pcirc.Q2(control, target).fix(1)
            else:
                pcirc.Q2(control, target)
    for i in range(N):
        pcirc.RY(i)
    for i in range(N):
        pcirc.RZ(i)
    return pcirc


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

hf=HartreeFock(
    problem.num_spatial_orbitals,
    problem.num_particles,
    mapper,
)
vqe_pcirc = get_circular_ansatz(N=len(paulis[0]), reps=0, fix_2q=True, initial_state=hf._bitstr)
n_params = 0
for gate in vqe_pcirc.gates:
    print(gate.label, gate.is_fixed())
    if not gate.is_fixed():
        gate.assign(0)
        n_params += 1

# ks_best, _, energy_best = claptonize(
#     [pauli[::-1] for pauli in paulis],
#     coeffs,
#     vqe_pcirc,
#     n_proc=4,           # total number of processes in parallel
#     n_starts=4,         # number of random genetic algorithm starts in parallel
#     n_rounds=1,         # number of budget rounds, if None it will terminate itself
#     callback=print,     # callback for internal parameter (#iteration, energies, ks) processing
#     budget=20           # budget per genetic algorithm instance
# )

ks_best = np.array([0 for _ in range(n_params)])
print(vqe_pcirc.stim_circuit().diagram())
print(ks_best)

# qiskit_circuit, param_list, initial_point = build_circuit_from_clifford(vqe_pcirc)
qiskit_circuit, param_list, initial_point = build_BS_ansatz(N=len(paulis[0]), reps=4, fix_2q=True, initial_state=hf._bitstr)
energy = get_expectation_value(qiskit_circuit, param_list, initial_point, paulis, coeffs)
print(energy)
energy_list = []
def objective_function(params):
    energy = get_expectation_value(qiskit_circuit, param_list, params, paulis, coeffs)
    # print(f'Iteration {len(energy_list)}: {energy}')
    energy_list.append(energy)
    return energy

print(initial_point)
result = minimize(objective_function, initial_point, method='COBYLA', options={'disp': True})
ground_state_energy = result.fun
# print(energy_list)
print("lowest energy:", ground_state_energy + problem.nuclear_repulsion_energy)
print("Hartree-Fock energy:", problem.reference_energy)

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

print("Ground state energy:", calc_ground_state_energy(problem, mapper))