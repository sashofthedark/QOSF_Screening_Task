import numpy as np
from math import sqrt 

class OneQubitGates():
    def __init__(self):
        self.x = np.array([
            [0,1],
            [1,0]])
        #X Pauli gate
        self.z = np.array([
            [1,0],
            [0,-1]])
        #Z pauli gate
        self.h = (1/sqrt(2)) * np.array([
            [1,1],
            [1,-1]])
        #Hadamard gate
        self.unity = np.array([
            [1,0],
            [0,1]])

class TwoQubitGates():
    def __init__(self):
        self.cnot = np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,0,1],
            [0,0,1,0]])

class ThreeQubitGates():
    def __init__(self):
        self.toffoli = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,1],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,1,0,0,0,0]])
        #it's actually the toffoli gate where the "upper" qubit is the target and two "lower"
        #ones are the "controls"
    