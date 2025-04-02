import sympy
import numpy as np

# Define the rotation matrices and the CNOT gate
def rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]])

def ry(theta):
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [1j * np.sin(theta / 2), np.cos(theta / 2)]])

def cx():
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])

def cx2():
    return np.array([[1, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0]])

# Define the parameters
param = sympy.symbols('theta')

# Define the unitary for the full circuit
U_full = np.eye(4)

# Apply the gates in sequence
U_full = np.matmul(U_full, np.kron(rz(-3 * np.pi / 4), np.eye(2)))  # Rz(-3pi/4)
U_full = np.matmul(U_full, cx())  # CX(target, control)
U_full = np.matmul(U_full, np.kron(np.eye(2), rz(np.pi / 2)))  # Rz(pi/2)
U_full = np.matmul(U_full, np.kron(np.eye(2), rz(param / 2)))  # Rz(param/2)
U_full = np.matmul(U_full, np.kron(ry(-np.pi / 2), np.eye(2)))  # Ry(-pi/2)
U_full = np.matmul(U_full, np.kron(ry(param / 2), np.eye(2)))  # Ry(param/2)
U_full = np.matmul(U_full, cx2())  # CX(control, target)
U_full = np.matmul(U_full, np.kron(ry(np.pi / 2), np.eye(2)))  # Ry(pi/2)
U_full = np.matmul(U_full, np.kron(ry(-param / 2), np.eye(2)))  # Ry(-param/2)
U_full = np.matmul(U_full, cx())  # CX(target, control)
U_full = np.matmul(U_full, np.kron(rz(np.pi / 4), rz(np.pi / 2)))  # Rz(pi/2)

U_full  # The resulting unitary matrix
