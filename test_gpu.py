import qiskit
import pdb
from qiskit_aer import AerSimulator
from qiskit_aer.quantum_info import AerStatevector
circ = qiskit.QuantumCircuit(3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)

# backend = AerSimulator(method='statevector', device='GPU')
# result = backend.run(circ).result()
statevec = AerStatevector.from_instruction(circ)
statevec2 = AerStatevector(circ, device='GPU')
pdb.set_trace()
print(statevec)