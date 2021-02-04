import numpy as np
from math import sqrt

class VectorOfQubits():
    def __init__(self,Values = list):
        self.rowvector = np.array([Values])
        self.colvector = np.transpose(self.rowvector)