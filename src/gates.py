import numpy as np
from math import sqrt 
from vectors import VectorOfQubits
from vectorconv import VectorConv

def CallClassInit(cls):
    cls.__ClassInit__()
    return cls

class OneQubitGates():
    x = np.array([
        [0,1],
        [1,0]])
    #X Pauli gate

    z = np.array([
        [1,0],
        [0,-1]])
    #Z pauli gate

    h = (1/sqrt(2)) * np.array([
        [1,1],
        [1,-1]])
    #Hadamard gate

    unity = np.array([
        [1,0],
        [0,1]])

class TwoQubitGates():
    cnot = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,0,1],
        [0,0,1,0]])

@CallClassInit
class ThreeQubitGates():
    toffoli = np.array([
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

    @classmethod
    def __ClassInit__(cls):
        cls.bitflip = cls.__BitFlipGate__()

    @classmethod
    def __BitFlipGate__(cls):
            FirstGate = VectorConv.TensorProdTwo(TwoQubitGates.cnot,OneQubitGates.unity)
            #tensor product between a cnot gate and unity matrix

            ZeroZeroMatrix = VectorConv.TensorProdTwo(
            VectorOfQubits([1,0]).colvector,
            VectorOfQubits([1,0]).rowvector)

            OneOneMatrix = VectorConv.TensorProdTwo(
            VectorOfQubits([0,1]).colvector,
            VectorOfQubits([0,1]).rowvector)

            SecondGate_1 = VectorConv.TensorProdThree(ZeroZeroMatrix,OneQubitGates.unity,OneQubitGates.unity)
            SecondGate_2 = VectorConv.TensorProdThree(OneOneMatrix,OneQubitGates.unity,OneQubitGates.x)

            SecondGate = SecondGate_1 + SecondGate_2
            ThirdGate = cls.toffoli
            BitFlipMatrix = (ThirdGate.dot(SecondGate)).dot(FirstGate)
            return BitFlipMatrix