import numpy as np
from pgmQC.utils.setting import CNOT, I, X, Y, Z

states_2q = [np.kron(a, b) for a in [I, X, Y, Z] for b in [I, X, Y, Z]]

tensor = []
for input_state in states_2q:
    rho = CNOT @ input_state @ CNOT
    tensor_row = []
    for output_state in states_2q:
        tensor_row.append(np.trace(rho @ output_state).real / 4)
    tensor.append(tensor_row)

print(tensor)