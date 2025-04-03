import sympy
import numpy as np
from pgmQC.utils.setting import X, Y, Z, I
# Define the rotation matrices and the CNOT gate
def rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]])

def ry(theta):
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]])

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


def decomposed_Ubs(param):
# Define the unitary for the full circuit
    U_full = np.eye(4)

    # Apply the gates in sequence
    U_full = np.matmul(U_full, np.kron(rz(-3 * np.pi / 4), np.eye(2)))  # Rz(-3pi/4)
    U_full = np.matmul(U_full, cx())  # CX(target, control)
    U_full = np.matmul(U_full, np.kron(np.eye(2), rz((param + np.pi) / 2)))  # Rz(pi/2)
    U_full = np.matmul(U_full, np.kron(ry((param-np.pi) / 2), np.eye(2)))  # Ry(-pi/2)
    U_full = np.matmul(U_full, cx2())  # CX(control, target)
    U_full = np.matmul(U_full, np.kron(ry((np.pi-param) / 2), np.eye(2)))  # Ry(pi/2)
    U_full = np.matmul(U_full, cx())  # CX(target, control)
    U_full = np.matmul(U_full, np.kron(rz(np.pi / 4), rz(np.pi / 2)))  # Rz(pi/2)

    return U_full  # The resulting unitary matrix

def Ubs(theta):
    return np.array([[1, 0, 0, 1],
                     [0, (np.cos(theta)+1j*np.sin(theta)+1)/2, (1+1j)*(np.cos(theta)+1j*np.sin(theta)-1)/(2*np.sqrt(2)), 0],
                     [0, (1-1j)*(np.cos(theta)+1j*np.sin(theta)-1)/(2*np.sqrt(2)), (np.cos(theta)+1j*np.sin(theta)+1)/2, 0],
                     [0, 0, 0, 1]])

theta = 0
print(Ubs(0))
print(decomposed_Ubs(0))

operator = np.array([[0, 0, 0, 0],
                     [0, .5, (1+1j)/(2*np.sqrt(2)), 0],
                     [0, (1-1j)/(2*np.sqrt(2)), .5, 0],
                     [0, 0, 0, 0]])
str_to_matrix = {'I': I, 'X' : X, 'Y' : Y, 'Z' : Z}
for outer in ['I', 'X', 'Y', 'Z']:
    for inner in ['I', 'X', 'Y', 'Z']:
        basis = np.kron(str_to_matrix[outer], str_to_matrix[inner])
        coeff = np.trace(np.matmul(basis,operator)) / 4
        print(f'{outer+inner} : {coeff}')
        
# Hbs = 1/4 (II - ZZ) + 1/(4*sqrt(2)) (XX+XY-YX+YY)