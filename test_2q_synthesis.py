import pennylane as qml
import numpy as np

def Ubs(theta):
    return np.array([[1, 0, 0, 1],
                     [0, (np.cos(theta)+1j*np.sin(theta)+1)/2, (1+1j)*(np.cos(theta)+1j*np.sin(theta)-1)/(2*np.sqrt(2)), 0],
                     [0, (1-1j)*(np.cos(theta)+1j*np.sin(theta)-1)/(2*np.sqrt(2)), (np.cos(theta)+1j*np.sin(theta)+1)/2, 0],
                     [0, 0, 0, 1]])
    
def RX(theta):
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]])

def RY(theta):
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]])

def RZ(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]])

def CNOT():
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])
def CNOT2():
    return np.array([[1, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0]])

theta = np.pi
U = Ubs(theta)
print(U)
decomp = qml.ops.two_qubit_decomposition(np.array(U), wires=[0, 1])
print(decomp)