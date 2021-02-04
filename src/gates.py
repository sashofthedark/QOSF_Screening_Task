import numpy as np
from math import sqrt 
from vectors import VectorOfQubits
from vectorconv import VectorConv

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

    def BitFlipGate(self):
            FirstGate = VectorConv.TensorProdTwo(TwoQubitGates.cnot,OneQubitGates.unity)
            #tensor product between a cnot gate and unity matrix

            ZeroZeroMatrix = VectorConv.TensorProdTwo(
            VectorOfQubits([1,0]).rowvector,
            VectorOfQubits(1,0).colvector)

            OneOneMatrix = VectorConv.TensorProdTwo(
            VectorOfQubits([0,1]).rowvector,
            VectorOfQubits([0,1]).colvector)

            SecondGate_1 = VectorConv.TensorProdThree(ZeroZeroMatrix,OneQubitGates.unity,OneQubitGates.unity)
            SecondGate_2 = VectorConv.TensorProdThree(OneOneMatrix,OneQubitGates.unity,OneQubitGates.x)

            SecondGate = SecondGate_1 + SecondGate_2
            ThirdGate = self.toffoli
            BitFlipMatrix = (ThirdGate.dot(SecondGate)).dot(FirstGate)
            return BitFlipMatrix

