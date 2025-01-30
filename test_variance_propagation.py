import pdb
import ast
from tqdm import tqdm
import random
import numpy as np
import cotengra as ctg
from scipy.sparse import dok_matrix
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.converters import circuit_to_dag, dag_to_circuit

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_aer.quantum_info import AerStatevector
from pgmQC.model.tensorRV_network_builder import TensorRVNetworkBuilder

"""
    Rz(θ)•I•Rz^dag(θ) = I
    Rz(θ)•X•Rz^dag(θ) = cos(θ)X - sin(θ)Y
    Rz(θ)•Y•Rz^dag(θ) = sin(θ)X + cos(θ)Y
    Rz(θ)•Z•Rz^dag(θ) = Z
"""
def get_RZ():
    mean = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    cov = np.zeros((4, 4, 4, 4))
    cov[1, 1, 1, 1] = .5
    cov[1, 2, 1, 2] = .5
    cov[2, 1, 2, 1] = .5
    cov[2, 2, 2, 2] = .5
    cov[1, 1, 2, 2] = .5
    cov[2, 2, 1, 1] = .5
    cov[1, 2, 2, 1] = -.5
    cov[2, 1, 1, 2] = -.5
    return mean, cov

"""
    Rx(θ)•I•Rx^dag(θ) = I
    Rx(θ)•X•Rx^dag(θ) = X
    Rx(θ)•Y•Rx^dag(θ) = cos(θ)Y - sin(θ)Z
    Rx(θ)•Z•Rx^dag(θ) = sin(θ)Y + cos(θ)Z
"""
def get_RX():
    mean = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    cov = np.zeros((4, 4, 4, 4))
    cov[2, 2, 2, 2] = .5
    cov[2, 3, 2, 3] = .5
    cov[3, 2, 3, 2] = .5
    cov[3, 3, 3, 3] = .5
    cov[2, 2, 3, 3] = .5
    cov[3, 3, 2, 2] = .5
    cov[2, 3, 3, 2] = -.5
    cov[3, 2, 2, 3] = -.5
    return mean, cov

"""
    Ry(θ)•I•Ry^dag(θ) = I
    Ry(θ)•X•Ry^dag(θ) = cos(θ)X + sin(θ)Z
    Ry(θ)•Y•Ry^dag(θ) = Y
    Ry(θ)•Z•Ry^dag(θ) = -sin(θ)X + cos(θ)Z
"""
def get_RY():
    mean = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    cov = np.zeros((4, 4, 4, 4))
    cov[1, 1, 1, 1] = .5
    cov[1, 3, 1, 3] = .5
    cov[3, 1, 3, 1] = .5
    cov[3, 3, 3, 3] = .5
    cov[1, 1, 3, 3] = .5
    cov[3, 3, 1, 1] = .5
    cov[1, 3, 3, 1] = -.5
    cov[3, 1, 1, 3] = -.5
    return mean, cov

def get_Hadamard():
    mean = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0]])
    cov = np.zeros((4, 4, 4, 4))
    return mean, cov

def get_initial_state():
    mean = np.array([[.5, 0, 0, 0.5]])
    cov = np.zeros((1, 4, 1, 4))
    return mean, cov

def get_CNOT():
    mean = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    cov = np.zeros((16, 16, 16, 16))
    return mean, cov

from pgmQC.utils.setting import I, X, Y, Z
def get_1q_gate(gate: np.ndarray):
    tensor = []
    for input_state in [I, X, Y, Z]:
        rho = gate * input_state * gate.conj().T
        row_tensor = []
        for output_state in [I, X, Y, Z]:
            row_tensor.append(np.trace(rho @ output_state).real / 2)
        tensor.append(row_tensor)
    return np.array(tensor), np.zeros((4, 4, 4, 4))

class RV_Matrix:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
    
    """
        C = AB
        C_ij = Σ_k A_ik * B_kj
        E(C_ij) = Σ_k E(A_ik) * E(B_kj)
        Cov(C_ij, C_qs) = Σ_k Σ_r { Cov(A_ik, A_qr) * E(B_kj) * E(B_rs) + E(A_ik) * E(A_qr) * Cov(B_kj, B_rs) + Cov(A_ik, A_qr) * Cov(B_kj, B_rs) }
    """
    def __matmul__(self, other):
        E_A = self.mean
        Cov_A = self.cov
        E_B = other.mean
        Cov_B = other.cov
        n, m = E_A.shape
        m2, l = E_B.shape
        assert m == m2
        E_C = np.zeros((n, l))
        Cov_C = np.zeros((n, l, n, l))
        
        for i in range(n):
            for j in range(l):
                E_C[i, j] = np.sum(E_A[i, :] * E_B[:, j])
        
        for i in range(n):
            for j in range(l):
                for q in range(n):
                    for s in range(l):
                        for k in range(m):
                            for r in range(m):
                                Cov_C[i, j, q, s] += Cov_A[i, k, q, r] * E_B[k, j] * E_B[r, s] + E_A[i, k] * E_A[q, r] * Cov_B[k, j, r, s] + Cov_A[i, k, q, r] * Cov_B[k, j, r, s]
        return RV_Matrix(E_C, Cov_C)



class RV_Matrix_Sparse:
    def __init__(self, mean, cov):
        """
        mean: A sparse dictionary-of-keys (DOK) matrix representing the mean matrix.
        cov: A sparse dictionary-of-keys (DOK) matrix representing the covariance.
        """
        self.mean = dok_matrix(mean)  # Using sparse DOK format for efficient updates and access
        n, m = mean.shape
        self.cov = dok_matrix(cov.reshape(n * m, n * m))  # Using sparse DOK format for efficient updates and access
    
    """
        C = AB
        C_ij = Σ_k A_ik * B_kj
        E(C_ij) = Σ_k E(A_ik) * E(B_kj)
        Cov(C_ij, C_qs) = Σ_k Σ_r { Cov(A_ik, A_qr) * E(B_kj) * E(B_rs) + E(A_ik) * E(A_qr) * Cov(B_kj, B_rs) + Cov(A_ik, A_qr) * Cov(B_kj, B_rs) }
    """
    def __matmul__(self, other):
        """
        Sparse implementation of matrix multiplication, propagating mean and covariance.
        """
        E_A = self.mean
        Cov_A = self.cov
        E_B = other.mean
        Cov_B = other.cov

        n, m = E_A.shape
        m2, l = E_B.shape
        assert m == m2, "Matrix dimensions do not match for multiplication"

        # Compute the sparse mean of the result
        E_C = dok_matrix((n, l))
        for (i, k), val_A in E_A.items():
            for (k2, j), val_B in E_B.items():
                if k == k2:
                    E_C[i, j] += val_A * val_B

        # Compute the sparse covariance of the result
        Cov_C = dok_matrix((n * l, n * l))
        
        # Cov(C_ij, C_qs) += Σ_k Σ_r { Cov(A_ik, A_qr) * E(B_kj) * E(B_rs) }
        for (i_k, q_r), cov_A_val in Cov_A.items():
            i, k = divmod(i_k, m)
            q, r = divmod(q_r, m)
            for (k2, j), val_B in E_B.items():
                for (r2, s), val_B2 in E_B.items():
                    if k == k2 and r == r2:
                        idx_C1 = (i * l + j, q * l + s)
                        Cov_C[idx_C1] += cov_A_val * val_B * val_B2
        # Cov(C_ij, C_qs) += Σ_k Σ_r { E(A_ik) * E(A_qr) * Cov(B_kj, B_rs) }
        for (k_j, r_s), cov_B_val in Cov_B.items():
            k, j = divmod(k_j, l)
            r, s = divmod(r_s, l)
            for (i, k2), val_A in E_A.items():
                for (q, r2), val_A2 in E_A.items():
                    if k == k2 and r == r2:
                        idx_C2 = (i * l + j, q * l + s)
                        Cov_C[idx_C2] += val_A * val_A2 * cov_B_val
        # Cov(C_ij, C_qs) += Σ_k Σ_r { Cov(A_ik, A_qr) * Cov(B_kj, B_rs) }
        for (i_k, q_r), cov_A_val in Cov_A.items():
            i, k = divmod(i_k, m)
            q, r = divmod(q_r, m)
            for (k_j, r_s), cov_B_val in Cov_B.items():
                k2, j = divmod(k_j, l)
                r2, s = divmod(r_s, l)

                if k == k2 and r == r2:
                    idx_C3 = (i * l + j, q * l + s)
                    Cov_C[idx_C3] += cov_A_val * cov_B_val

        return RV_Matrix_Sparse(E_C, Cov_C)

    def to_dense_mean(self):
        """
        Converts the sparse mean matrix to a dense numpy array.
        """
        return self.mean.toarray()

    def to_dense_cov(self):
        """
        Converts the sparse covariance matrix to a dense numpy array.
        """
        return self.cov.toarray()

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
    print(tree)
    print(tree.contraction_width(), tree.contraction_cost())
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

def main():
    RZ = RV_Matrix(*get_RZ())
    RX = RV_Matrix(*get_RX())
    RY = RV_Matrix(*get_RY())
    H = RV_Matrix(*get_Hadamard())
    initial_state = RV_Matrix(*get_initial_state())
    
    RZ_sparse = RV_Matrix_Sparse(*get_RZ())
    RX_sparse = RV_Matrix_Sparse(*get_RX())
    RY_sparse = RV_Matrix_Sparse(*get_RY())
    H_sparse = RV_Matrix_Sparse(*get_Hadamard())
    initial_state_sparse = RV_Matrix_Sparse(*get_initial_state())
    

    # initial_state = initial_state @ H @ RZ
    # initial_state_sparse = initial_state_sparse @ H_sparse @ RZ_sparse
    # initial_state = initial_state @ RX @ RZ
    # initial_state_sparse = initial_state_sparse @ RX_sparse @ RZ_sparse
    initial_state = initial_state @ RX @ RZ @ RY @ RX @ RZ @ RY
    initial_state_sparse = initial_state_sparse @ RX_sparse @ RZ_sparse @ RY_sparse @ RX_sparse @ RZ_sparse @ RY_sparse

    # for _ in ['I', 'X', 'Y', 'Z']:
    circuit = QuantumCircuit(1)
    angle = Parameter("angle_0")
    circuit.rx(angle, 0)
    angle = Parameter("angle_1")
    circuit.rz(angle, 0)
    angle = Parameter("angle_2")
    circuit.ry(angle, 0)
    angle = Parameter("angle_3")
    circuit.rx(angle, 0)
    angle = Parameter("angle_4")
    circuit.rz(angle, 0)
    angle = Parameter("angle_5")
    circuit.ry(angle, 0)
    dag = circuit_to_dag(circuit)
    dag.draw(filename='dag.png')
    builder = TensorRVNetworkBuilder(dag)
    markov_net, uncontracted_nodes, tensorRV_list = builder.build()
    
    result = contract_tensors(tensorRV_list, uncontracted_nodes)
    for key, value in result.cov:
        print(key, value * 4)
    
    
    print(initial_state.mean * 2)
    print(initial_state.cov * 4)
    
    print(initial_state_sparse.to_dense_mean() * 2)
    print(initial_state_sparse.to_dense_cov() * 4)
    exit()
    problem = PySCFDriver(atom="H .0 .0 .0; H .0 .0 1.", basis='sto3g').run()
    mapper = JordanWignerMapper()
    # ansatz = UCCSD(
    #     problem.num_spatial_orbitals,
    #     problem.num_particles,
    #     mapper,
    #     initial_state=HartreeFock(
    #         problem.num_spatial_orbitals,
    #         problem.num_particles,
    #         mapper,
    #     ),
    # )
    ansatz = TwoLocal(4, 'ry', 'rzz', 'linear', reps=5, insert_barriers=True)

    second_q_op = problem.hamiltonian.second_q_op()
    qubit_op = mapper.map(second_q_op)
    print(qubit_op)
    circ = transpile(ansatz, basis_gates=['u3', 'cx', 'rz'])
    print(circ)
    
    obs = ['']
    for _ in range(ansatz.num_qubits):
        new_obs = []
        for s in obs:
            for p in ['I', 'X', 'Y', 'Z']:
                new_obs.append(s + p)
        obs = new_obs
    
    results = []
    num_trials = 1000
    for _ in tqdm(range(num_trials)):
        params = [random.uniform(-np.pi, np.pi) for _ in range(ansatz.num_parameters)]
        circuit = ansatz.assign_parameters(params)
        
        result = []
        for ob in obs:
            expectation = AerStatevector(circuit, device='GPU').expectation_value(Pauli(ob))
            result.append(expectation)
        results.append(result)
    
    results = np.array(results)
    mean = np.mean(results, axis=0)
    vars = np.mean((results - mean) ** 2, axis=0)

    mean_dic = {}
    vars_dic = {}
    for i, ob in enumerate(obs):
        mean_dic[ob] = mean[i]
        vars_dic[ob] = vars[i]
    
    paulis = ast.literal_eval(qubit_op.paulis.__str__())
    average_of_vars = 0
    for pauli in paulis:
        print(f"{pauli}, mean: {mean_dic[pauli]}, vars: {vars_dic[pauli]}")
        average_of_vars += vars_dic[pauli]
    average_of_vars /= len(paulis)
    
    print("===============Not in Paulis===============")
    average_of_vars_not_in_paulis = 0
    for ob in obs:
        if ob not in paulis:
            print(f"{ob}, mean: {mean_dic[ob]}, vars: {vars_dic[ob]}")
            average_of_vars_not_in_paulis += vars_dic[ob]
    average_of_vars_not_in_paulis /= len(obs) - len(paulis)
    
    print("Average of vars in Paulis:", average_of_vars)
    print("Average of vars not in Paulis:", average_of_vars_not_in_paulis)
if __name__ == "__main__":
    main()