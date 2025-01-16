import pdb
import numpy as np

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


def main():
    RZ = RV_Matrix(*get_RZ())
    RX = RV_Matrix(*get_RX())
    RY = RV_Matrix(*get_RY())
    H = RV_Matrix(*get_Hadamard())
    initial_state = RV_Matrix(*get_initial_state())

    initial_state = initial_state @ RX @ RZ

    print(initial_state.mean)
    print(initial_state.cov)
    
if __name__ == "__main__":
    main()