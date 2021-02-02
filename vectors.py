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

    def TensorProdTwo(construct1,construct2):
        #construct is a numpy array, either a vector or a matrix
        return np.kron(construct1,construct2)

    def TensorProdThree(construct1,construct2,construct3):
        #construct is a numpy array, either a vector or a matrix
        #it's Const1 X Const2 X Const3 (first 2 and 3, then 1 with the resulting product)
        return np.kron(np.kron(construct2,construct3),construct1)
    
