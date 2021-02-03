import numpy as np
from math import sqrt 
from vectors import OneQubitVector,TwoQubitVector,ThreeQubitVector,VectorConv
from gates import OneQubitGates,TwoQubitGates,ThreeQubitGates
from numpy.random import choice

def PerformBitFlipCorrection(InputVector = ThreeQubitVector):
    Vector = ThreeQubitVector.colvector
    return ThreeQubitGates.BitFlipGate.dot(Vector)

def ApplyNoise(prob_x,prob_z,InputVector = ThreeQubitVector):
    if (prob_x + prob_z) > 1:
        raise ValueError("Sum of probabilities greater than one")
    elif prob_x >= 0.25 or prob_z >= 0.25:
        raise ValueError("The error correction code is not effective for these probabilities")
    else:
        prob_unity = 1 - prob_x - prob_z
        prob_dist = [prob_unity,prob_x,prob_z]
        gates = [OneQubitGates.unity,OneQubitGates.x,OneQubitGates.z]
        SelectedGate = choice(gates,1,prob_dist)
        VectorAfterNoise = SelectedGate.dot(InputVector.colvector)
        return VectorAfterNoise