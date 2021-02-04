import numpy as np
from math import sqrt 
from vectors import VectorOfQubits
from gates import OneQubitGates,TwoQubitGates,ThreeQubitGates
from numpy.random import choice

def PerformBitFlipCorrection(InputVector = VectorOfQubits):
    Vector = VectorOfQubits.colvector
    return ThreeQubitGates.BitFlipGate.dot(Vector)

def ApplyNoise(prob_x,prob_z,InputVector = VectorOfQubits):
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

def RetrieveFirstQubit(InputVector = VectorOfQubits):
    #this function assumes the second and third qubits are either |00> or |11>
    #(this is the case in this particular example)
    RowVector = InputVector.rowvector
    if RowVector[0][0] == 0 and RowVector[0][4] == 0:
        return VectorOfQubits([RowVector[0][3],RowVector[0][7]])
    else:
        return VectorOfQubits([RowVector[0][0],RowVector[0][4]])
        #returning an instance of the class OneQubitVector with the correct components
        #corresponding to the first qubit (this discards the ancilla qubits)

def CircuitAndCorrection(
    prob_x, 
    prob_p):
    UpperQubit = VectorOfQubits(1,0).colvector
    #this is a zero state (column vector)
    LowerQubit = OneQubitGates.h * VectorOfQubits(1,0).colvector
    #this yields a plus state (column vector)
    #pass the first and second qubits separately to ApplyNoise function 
    # which will apply noise to them, pass the probabilities to this function
    #Apply Hadamard gate to upper qubit

    #create an entangled state of three qubits for the upper qubit (with |00> as ancilla)
    #create an entangled state of three qubits for the lower qubit (with |00> as ancilla)

    #Apply BitFlip error correction circuit to upper three qubits
    #Apply BitFlip error correction circuit to lower three qubits

    #Apply Hadamard gate to upper qubit (create a three-qubit gate with two unity one-qubit matrices)

    #Create a two-qubit state from upper and lower qubit (discard the ancilla qubits)
    #apply CNOT gate to the two-qubit state

    #compare with desired result 
    #return True if comparison successful, and False otherwise.
    pass