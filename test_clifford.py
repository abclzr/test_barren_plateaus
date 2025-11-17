from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Pauli, SparsePauliOp
 
# Bell state generation circuit
qc = QuantumCircuit(4)
t = 2
b = 3
qc.h(b)
qc.cx(b, 0)
qc.cx(b, 1)
qc.cx(b, t)
qc.cx(0, t)
qc.cx(1, t)

cliff = Clifford(qc) 
# Print the Clifford
print(cliff)
 
# Print the Clifford destabilizer rows
print(cliff.to_labels(mode="D"))
 
# Print the Clifford stabilizer rows
print(cliff.to_labels(mode="S"))

print(cliff.adjoint())

i = cliff.adjoint().compose(cliff)
x = Pauli('IIIX')
xx = Pauli('IIXX')
z = Pauli('IIIZ')
y = Pauli('IIIY')
yy = Pauli('IIYY')
zz = Pauli('IIZZ')
xz = Pauli('IIXZ')
zx = Pauli('IIZX')
print(i)
print(x.evolve(cliff, frame='s'))
print(z.evolve(cliff, frame='s'))
print(xx.evolve(cliff, frame='s'))
print(zz.evolve(cliff, frame='s'))
print(y.evolve(cliff, frame='s'))
print(yy.evolve(cliff, frame='s'))
print(xz.evolve(cliff, frame='s'))
print(zx.evolve(cliff, frame='s'))