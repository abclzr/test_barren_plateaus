import pdb
import ast
from tqdm import tqdm
import random
import numpy as np
import cotengra as ctg

import pickle
import matplotlib.pyplot as plt

from scipy.sparse import dok_matrix
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.converters import circuit_to_dag, dag_to_circuit

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp, DensityMatrix, Statevector
from qiskit.circuit.library import TwoLocal
from qiskit_aer.quantum_info import AerStatevector
from pgmQC.model.tensorRV_network_builder import TensorRVNetworkBuilder, TensorRV

def get_tensorRV(op, var_list):
    paulis = [ Pauli('I'), Pauli('X'), Pauli('Y'), Pauli('Z') ]
    pauli_index_lookup = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    strings = paulis
    n = op.num_qubits
    while n > 1:
        new_strings = []
        for a in strings:
            for b in paulis:
                new_strings.append(a.expand(b))
        strings = new_strings
        n = n-1
    # for every input variables state, get the output variables state
    values = []
    for string in strings:
        sparse = SparsePauliOp.from_operator(DensityMatrix(string).evolve(op))
        out_state_values = np.zeros(4 ** op.num_qubits)
        for label, coeff in sparse.label_iter(): # type: ignore
            index = 0
            # neccecary to reverse the label, because qiskit's SparsePauliOp('IX') will be 'X' on qubit 0 and 'I' on qubit 1
            for pauli in reversed(label):
                index = index * 4 + pauli_index_lookup[pauli]
            out_state_values[index] = coeff
        
        values.append(out_state_values)

    # create the factor
    values = np.concatenate(values, axis=0).real
    tensorRV = TensorRV(values, np.zeros([4] * (2 * op.num_qubits)), var_list)
    return tensorRV

def contract_tensors(tensorRV_list, uncontracted_vars):
    """
    Calculate the path of contracting a list of tensors, and then contract them.
    
    Args:
        tensor_list: List of tensors to contract.
        uncontracted_vars: List of uncontracted variables.
    
    Returns:
        A tensor representing the result of the contraction.
    """
    size_dict = {}
    inputs = []
    for tensorRV in tensorRV_list:
        inputs.append(tuple(tensorRV.variables))
        for var in tensorRV.variables:
            size_dict[var] = 4
    output = tuple(uncontracted_vars)
    opt = ctg.HyperOptimizer()
    tree = opt.search(inputs, output, size_dict)
    # print(tree)
    # print(tree.contraction_width(), tree.contraction_cost())
    path = tree.get_path()
    assert len(path) == len(tensorRV_list) - 1, "The number of contractions should be equal to the number of tensors minus 1."
        
    for i, j in path:
        new_tensorRV = tensorRV_list[i] @ tensorRV_list[j]
        if i > j:
            del tensorRV_list[i]
            del tensorRV_list[j]
        else:
            del tensorRV_list[j]
            del tensorRV_list[i]
        tensorRV_list.append(new_tensorRV)
    
    assert len(tensorRV_list) == 1, "The number of tensors should be 1 after contraction."
    return tensorRV_list[0]

def generate_circuit(n_qubit, n_layers, gate_list, theta_list, flag):
    circuit = QuantumCircuit(n_qubits)
    cnt = 0
    for i in range(n_qubits):
        circuit.ry(np.pi/4, i)
    for _ in range(n_layers):
        for i in range(n_qubits):
            gate = gate_list[cnt]
            theta = theta_list[cnt]
            if cnt == 0:
                theta = theta + flag
            cnt += 1
            if gate == 'rx':
                circuit.rx(theta, i)
            elif gate == 'ry':
                circuit.ry(theta, i)
            elif gate == 'rz':
                circuit.rz(theta, i)
        for i in range(0, n_qubits-1, 2):
            circuit.cz(i, i+1)
        for i in range(1, n_qubits-1, 2):
            circuit.cz(i, i+1)
    return circuit

def emperical_sample(n_qubit, n_layers):
    num_samples = 10000
    grad_list = []
    for _ in tqdm(range(num_samples)):
        gate_list = [random.choice(['rx', 'ry', 'rz']) for _ in range(n_qubit*n_layers)]
        theta_list = [random.uniform(-np.pi, np.pi) for _ in range(n_qubit*n_layers)]
        cnt = 0
        circuit1 = generate_circuit(n_qubit, n_layers, gate_list, theta_list, np.pi/2)
        circuit2 = generate_circuit(n_qubit, n_layers, gate_list, theta_list, -np.pi/2)
        expval1 = Statevector.from_instruction(circuit1).expectation_value(Pauli('ZZ'+'I'*(n_qubits-2)), qargs=[i for i in range(n_qubits)])
        expval2 = Statevector.from_instruction(circuit2).expectation_value(Pauli('ZZ'+'I'*(n_qubits-2)), qargs=[i for i in range(n_qubits)])
        grad = 0.5 * (expval1 - expval2)
        grad_list.append(grad)
    
    mean = np.mean(grad_list)
    var = np.mean(np.array(grad_list) ** 2)
    print(f'Mean: {mean}, Var: {var}')
    return var

def main(n_qubits, n_layers=1):
    circuit = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        circuit.ry(np.pi/4, i)
    for _ in range(n_layers):
        for i in range(n_qubits):
            circuit.rz(np.pi/4, i)
        for i in range(0, n_qubits-1, 2):
            circuit.cz(i, i+1)
        for i in range(1, n_qubits-1, 2):
            circuit.cz(i, i+1)
    var_id_each_qubit = [0 for _ in range(n_qubits)]
    def get_varname(qubit_id):
        return f'q{qubit_id}_{var_id_each_qubit[qubit_id]}'
    
    first_time_flag = True
    tensorRV_list = []
    for i in range(n_qubits):
        tensorRV_list.append(TensorRV(np.array([1, 0, 0, 1]), np.zeros([4, 4]), [get_varname(i)]))
    for inst in circuit:
        var_list = []
        for qubit in inst.qubits:
            var_list.append(get_varname(qubit._index))
        for qubit in inst.qubits:
            var_id_each_qubit[qubit._index] += 1
            var_list.append(get_varname(qubit._index))
        if inst.operation.name == 'ry':
            tensorRV_list.append(get_tensorRV(inst.operation, var_list))
        elif inst.operation.name == 'rz':
            if first_time_flag:
                first_time_flag = False
                mean, cov = TensorRVNetworkBuilder.get_differentiated_RX()
                tensorRV_rx = TensorRV(mean, cov, var_list)
                mean, cov = TensorRVNetworkBuilder.get_differentiated_RY()
                tensorRV_ry = TensorRV(mean, cov, var_list)
                mean, cov = TensorRVNetworkBuilder.get_differentiated_RZ()
                tensorRV_rz = TensorRV(mean, cov, var_list)
                tensorRV_list.append(TensorRVNetworkBuilder.merge_channels(tensorRV_rx + tensorRV_ry + tensorRV_rz))
            else:
                mean, cov = TensorRVNetworkBuilder.get_RX()
                tensorRV_rx = TensorRV(mean, cov, var_list)
                mean, cov = TensorRVNetworkBuilder.get_RY()
                tensorRV_ry = TensorRV(mean, cov, var_list)
                mean, cov = TensorRVNetworkBuilder.get_RZ()
                tensorRV_rz = TensorRV(mean, cov, var_list)
                tensorRV_list.append(TensorRVNetworkBuilder.merge_channels(tensorRV_rx + tensorRV_ry + tensorRV_rz))
                # mean, cov = TensorRVNetworkBuilder.get_RD()
                # tensorRV_rd = TensorRV(mean, cov, var_list)
                # tensorRV_list.append(tensorRV_rd)
        elif inst.operation.name == 'cz':
            tensorRV_list.append(get_tensorRV(inst.operation, var_list))
    tensorRV_list.append(TensorRV(np.array([0, 0, 0, 1]), np.zeros([4, 4]), [get_varname(0)]))
    tensorRV_list.append(TensorRV(np.array([0, 0, 0, 1]), np.zeros([4, 4]), [get_varname(1)]))
    for i in range(2, n_qubits):
        tensorRV_list.append(TensorRV(np.array([1, 0, 0, 0]), np.zeros([4, 4]), [get_varname(i)]))
    
    ret = contract_tensors(tensorRV_list, [])
    if len(ret.cov) == 0:
        return 0
    return ret.cov[0][1]

def plot_variance(var_list, qubits_list):
    # Save var_list and qubits_list to a pickle file
    # with open('variance_data.pkl', 'wb') as f:
    #     pickle.dump({'var_list': var_list, 'qubits_list': qubits_list}, f)

    # Read var_list and qubits_list from the pickle file
    with open('variance_data.pkl', 'rb') as f:
        data = pickle.load(f)
        var_list = data['var_list']
        qubits_list = data['qubits_list']
    plt.figure(figsize=(10, 6))
    plt.plot(qubits_list, var_list, marker='o')
    plt.yscale('log')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Variance')
    plt.title('Variance vs Number of Qubits')
    plt.grid(True)
    plt.savefig('variance_vs_qubits.pdf')

def test_gate_list_numerical(n_qubits, n_layers, gate_list):
    num_samples = 10000
    grad_list = []
    efficient = True
    for _ in tqdm(range(num_samples)):
        if efficient:
            theta_list = [random.choice([0, np.pi/2, np.pi, -np.pi/2]) for _ in range(n_qubits*n_layers)]
        else:
            theta_list = [random.uniform(-np.pi, np.pi) for _ in range(n_qubits*n_layers)]
        cnt = 0
        
        circuit1 = generate_circuit(n_qubits, n_layers, gate_list, theta_list, np.pi/2)
        circuit2 = generate_circuit(n_qubits, n_layers, gate_list, theta_list, -np.pi/2)
        expval1 = Statevector.from_instruction(circuit1).expectation_value(Pauli('ZZ'+'I'*(n_qubits-2)), qargs=[n_qubits-i-1 for i in range(n_qubits)])
        expval2 = Statevector.from_instruction(circuit2).expectation_value(Pauli('ZZ'+'I'*(n_qubits-2)), qargs=[n_qubits-i-1 for i in range(n_qubits)])
        grad = 0.5 * (expval1 - expval2)
        
        # circuit1 = generate_circuit(n_qubits, n_layers, gate_list, theta_list, 0)
        # grad = Statevector.from_instruction(circuit1).expectation_value(Pauli('ZZ'+'I'*(n_qubits-2)), qargs=[n_qubits-i-1 for i in range(n_qubits)])
        grad_list.append(grad)
    
    mean = np.mean(grad_list)
    var = np.mean(np.array(grad_list) ** 2)
    
    return var

def test_gate_list_analytical(n_qubits, n_layers, gate_list):
    circuit = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        circuit.ry(np.pi/4, i)
    for _ in range(n_layers):
        for i in range(n_qubits):
            if gate_list[_ * n_qubits + i] == 'rx':
                circuit.rx(Parameter(f'theta_{i}_{_}'), i)
            elif gate_list[_ * n_qubits + i] == 'ry':
                circuit.ry(Parameter(f'theta_{i}_{_}'), i)
            elif gate_list[_ * n_qubits + i] == 'rz':
                circuit.rz(Parameter(f'theta_{i}_{_}'), i)
        for i in range(0, n_qubits-1, 2):
            circuit.cz(i, i+1)
        for i in range(1, n_qubits-1, 2):
            circuit.cz(i, i+1)
    var_id_each_qubit = [0 for _ in range(n_qubits)]
    def get_varname(qubit_id):
        return f'q{qubit_id}_{var_id_each_qubit[qubit_id]}'
    
    first_time_flag = True
    tensorRV_list = []
    for i in range(n_qubits):
        tensorRV_list.append(TensorRV(np.array([1, 0, 0, 1]), np.zeros([4, 4]), [get_varname(i)]))
    for inst in circuit:
        var_list = []
        for qubit in inst.qubits:
            var_list.append(get_varname(qubit._index))
        for qubit in inst.qubits:
            var_id_each_qubit[qubit._index] += 1
            var_list.append(get_varname(qubit._index))
        has_parameter = False
        for param in inst.operation.params:
            if isinstance(param, Parameter):
                has_parameter = True
                break
        if not has_parameter:
            tensorRV_list.append(get_tensorRV(inst.operation, var_list))
        else:
            if first_time_flag:
                first_time_flag = False
                if inst.operation.name == 'rx':
                    mean, cov = TensorRVNetworkBuilder.get_differentiated_RX()
                elif inst.operation.name == 'ry':
                    mean, cov = TensorRVNetworkBuilder.get_differentiated_RY()
                elif inst.operation.name == 'rz':
                    mean, cov = TensorRVNetworkBuilder.get_differentiated_RZ()
                else:
                    raise ValueError('Invalid gate name')
                tensorRV = TensorRV(mean, cov, var_list)
                tensorRV_list.append(tensorRV)
            else:
                if inst.operation.name == 'rx':
                    mean, cov = TensorRVNetworkBuilder.get_RX()
                elif inst.operation.name == 'ry':
                    mean, cov = TensorRVNetworkBuilder.get_RY()
                elif inst.operation.name == 'rz':
                    mean, cov = TensorRVNetworkBuilder.get_RZ()
                else:
                    raise ValueError('Invalid gate name')
                tensorRV = TensorRV(mean, cov, var_list)
                tensorRV_list.append(tensorRV)
    tensorRV_list.append(TensorRV(np.array([0, 0, 0, 1]), np.zeros([4, 4]), [get_varname(0)]))
    tensorRV_list.append(TensorRV(np.array([0, 0, 0, 1]), np.zeros([4, 4]), [get_varname(1)]))
    for i in range(2, n_qubits):
        tensorRV_list.append(TensorRV(np.array([1, 0, 0, 0]), np.zeros([4, 4]), [get_varname(i)]))
    
    ret = contract_tensors(tensorRV_list, [])
    if len(ret.cov) == 0:
        return 0
    return ret.cov[0][1]

if __name__ == "__main__":
    var_list = []
    qubits_list = [3, 4]#, 5, 6, 7, 8, 9, 10]
    # for n_qubits in qubits_list:
    #     var = emperical_sample(n_qubits, n_qubits)
    #     var_list.append(var)
    #     print(main(n_qubits=n_qubits, n_layers=10*n_qubits))
    for n_qubits in qubits_list:
        for _ in range(10):
            n_layers = n_qubits
            gate_list = [random.choice(['rx', 'ry', 'rz']) for _ in range(n_qubits*n_layers)]
            var_numerical = test_gate_list_numerical(n_qubits, n_layers, gate_list)
            var_analytical = test_gate_list_analytical(n_qubits, n_layers, gate_list)
            print(f'#qubits: {n_qubits}, Numerical: {var_numerical}, Analytical: {var_analytical}')
    # plot_variance(var_list, qubits_list)