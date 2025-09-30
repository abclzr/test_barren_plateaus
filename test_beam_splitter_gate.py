import numpy as np

def RZ(theta):
    """Returns the unitary matrix for a rotation around the Z-axis.

    Args:
        theta (float): The angle of rotation in radians.
    Returns:
        np.ndarray: The 2x2 unitary matrix representing the RZ gate.
    """
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]])

def RY(theta):
    """Returns the unitary matrix for a rotation around the Y-axis.

    Args:
        theta (float): The angle of rotation in radians.
    Returns:
        np.ndarray: The 2x2 unitary matrix representing the RY gate.
    """
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]])

def CNOT_10():
    """Returns the unitary matrix for a CNOT gate.

    Returns:
        np.ndarray: The 4x4 unitary matrix representing the CNOT gate.
    """
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])

def CNOT_01():
    """Returns the unitary matrix for a CNOT gate with reversed control and target.

    Returns:
        np.ndarray: The 4x4 unitary matrix representing the CNOT gate.
    """
    return np.array([[1, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0]])

def beam_splitter_gate(theta):
    """Returns the unitary matrix for a beam splitter gate.

    Args:
        theta (float): The angle parameter for the beam splitter.
        phi (float): The phase parameter for the beam splitter.
    Returns:
        np.ndarray: The 2x2 unitary matrix representing the beam splitter gate.
    """
    Identity = np.array([[1, 0], [0, 1]])
    op1 = np.kron(RZ(-3/4*np.pi), Identity)
    op2 = CNOT_10()
    op3 = np.kron(RY((theta-np.pi)/2), RZ((theta+np.pi)/2))
    op4 = CNOT_01()
    op5 = np.kron(RY((np.pi-theta)/2), Identity)
    op6 = CNOT_10()
    op7 = np.kron(RZ(np.pi/4), RZ(np.pi/2))
    return op7 @ op6 @ op5 @ op4 @ op3 @ op2 @ op1

if __name__ == "__main__":
    theta = np.random.uniform(-np.pi, np.pi)
    U = beam_splitter_gate(theta)
    print("Theta:", theta)
    print("Beam Splitter Gate U:\n", U)