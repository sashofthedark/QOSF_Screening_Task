import numpy as np
from math import sqrt

class VectorOfQubits():
    def __init__(self,Values: list):
        self.rowvector = np.array([Values])
        self.colvector = np.transpose(self.rowvector)

class EntangledVector():
    def __init__(self,NormalizationConstant,FirstVector: list,SecondVector: list):
        '''
        this function creates an entangled vector and returns it as an instance of the class VectorOfQubits
        '''
        VectorWithoutNorm = []

        for (item1,item2) in zip(FirstVector,SecondVector):
            VectorWithoutNorm.append(item1 + item2)
            #creating an element-wise sum of two lists representing the two wavefunctions 
            # that are part of the entangled state
        VectorWithNorm = [NormalizationConstant*element for element in VectorWithoutNorm]

        self.rowvector = np.array([VectorWithNorm])
        self.colvector = np.transpose(self.rowvector)

