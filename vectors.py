import numpy as np
from math import sqrt

class OneQubitVector():
    #returns a column vector
    def __init__(self,a1,a2):
        self.vector = np.array([
            [a1],
            [a2]])

class TwoQubitVector():
    #returns a column vector
    def __init__(self,a1,a2,a3,a4):
        self.vector = np.array([
            [a1],
            [a2],
            [a3],
            [a4]])

class ThreeQubitVector():
    #returns a column vector
    def __init__(self,a1,a2,a3,a4,a5,a6,a7,a8):
        self.vector = np.array([
            [a1],
            [a2],
            [a3],
            [a4],
            [a5],
            [a6],
            [a7],
            [a8]])

class VectorConv():
    def TensorProd(vector1,vector2):
        #convert two one-qubit vectors to one two-qubit vector easily
        return np.kron(vector1,vector2)
