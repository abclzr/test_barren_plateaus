import sys
sys.path.append("../")
from clapton.clapton import claptonize
from clapton.ansatzes import circular_ansatz
import numpy as np

# define Hamiltonian, e.g. 3q Heisenberg model with random coefficients
paulis = ["XXI", "IXX", "YYI", "IYY", "ZZI", "IZZ"]
coeffs = np.random.random(len(paulis))

# define parametrized Clifford circuit that is being optimized over
# here we use the circular_ansatz template
# we fix 2q gates as they will not be optimized over
vqe_pcirc = circular_ansatz(N=len(paulis[0]), reps=1, fix_2q=True)

# the circuit consists of parametrized gates
for gate in vqe_pcirc.gates:
    print(gate.label, gate.is_fixed())
    
# non-fixed gates will be optimized over
# RY and RZ gates can assume 4 values k = 0,1,2,3 which descripe multiples of pi/2

# the initial parameters are all 0
vqe_pcirc.read()

# we can look at the corresponding stim circuit
vqe_pcirc.stim_circuit().diagram()

# we can assign a different set of parameters
vqe_pcirc.assign([0,1,2,3,0,1,2,3,0,1,2,3])

vqe_pcirc.stim_circuit().diagram()

# we can perform CAFQA by using the main optimization function "claptonize"
ks_best, _, energy_best = claptonize(
    paulis,
    coeffs,
    vqe_pcirc,
    n_proc=4,           # total number of processes in parallel
    n_starts=4,         # number of random genetic algorithm starts in parallel
    n_rounds=1,         # number of budget rounds, if None it will terminate itself
    callback=print,     # callback for internal parameter (#iteration, energies, ks) processing
    budget=20           # budget per genetic algorithm instance
)

print(f'Parameters: {ks_best}')
print(f'Energy: {energy_best}')

# the corresponding circuit is
vqe_pcirc.assign(ks_best)
print(vqe_pcirc.stim_circuit().diagram())