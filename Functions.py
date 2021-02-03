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

def RetrieveFirstQubit(InputVector = ThreeQubitVector):
    #this function assumes the second and third qubits are either |00> or |11>
    #(this is the case in this particular example)
    RowVector = InputVector.rowvector
    if RowVector[0][0] == 0 and RowVector[0][4] == 0:
        return OneQubitVector(RowVector[0][3],RowVector[0][7])
    else:
        return OneQubitVector(RowVector[0][0],RowVector[0][4])
        #returning an instance of the class OneQubitVector with the correct components
        #corresponding to the first qubit (this discards the ancilla qubits)