import numpy as np
from math import sqrt

class OneQubitVector():
    def __init__(self,a1,a2):
        self.vector = np.array([[a1,a2]])

class TwoQubitVector():
    def __init__(self,a1,a2,a3,a4):
        self.vector = np.array([[a1,a2,a3,a4]])

class ThreeQubitVector():
    def __init__(self,a1,a2,a3,a4,a5,a6,a7,a8):
        self.vector = np.array([[a1,a2,a3,a4,a5,a6,a7,a8]])