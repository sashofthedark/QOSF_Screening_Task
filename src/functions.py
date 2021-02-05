import numpy as np
from math import sqrt 
from vectors import VectorOfQubits
from gates import OneQubitGates,TwoQubitGates,ThreeQubitGates
from vectorconv import VectorConv
from numpy.random import choice

def PerformBitFlipCorrection(InputVector: VectorOfQubits):
    Vector = InputVector.colvector
    return ThreeQubitGates.BitFlipGate().dot(Vector)

def ApplyNoise(prob_x,prob_z,InputVector: VectorOfQubits.colvector):
    '''
    Applies either unity, X or Z gates randomly, using the provided probabilities and input column vector
    '''
    if (prob_x + prob_z) > 1:
        raise ValueError("Sum of probabilities greater than one")
    elif prob_x >= 0.25 or prob_z >= 0.25:
        raise ValueError("The error correction code is not effective for these probabilities")
    else:
        prob_unity = 1 - prob_x - prob_z
        prob_dist = [prob_unity,prob_x,prob_z]
        gates = [OneQubitGates.unity,OneQubitGates.x,OneQubitGates.z]
        SelectedGate = choice(gates,1,prob_dist)
        VectorAfterNoise = SelectedGate.dot(InputVector)
        #InputVector is a column vector
        return VectorAfterNoise

def RetrieveFirstQubit(InputVector: VectorOfQubits):
    '''
    this function assumes the second and third qubits are either |00> or |11>
    (this is the case in this particular example)
    '''
    RowVector = InputVector.rowvector
    if RowVector[0][0] == 0 and RowVector[0][4] == 0:
        return VectorOfQubits([RowVector[0][3],RowVector[0][7]])
    else:
        return VectorOfQubits([RowVector[0][0],RowVector[0][4]])
        #returning an instance of the class OneQubitVector with the correct components
        #corresponding to the first qubit (this discards the ancilla qubits)

def CircuitAndCorrection(prob_x, prob_z):
    '''
    This function receives a |0+> initial vector, applies noise to each qubit based on the provided probabilities 
    and then performs error correction to get the correct final entangled wavefunction
    '''
    ZeroState = VectorOfQubits([1,0]).colvector
    UpperQubit = ZeroState
    #this is a zero state (column vector)
    LowerQubit = OneQubitGates.h.dot(UpperQubit)
    #this is a plus state (column vector)

    #pass the first and second qubits separately to ApplyNoise function
    # which will apply noise to them, pass the probabilities to this function
    UpperQubitAfterNoise = ApplyNoise(prob_x,prob_z, UpperQubit) 
    LowerQubitAfterNoise = ApplyNoise(prob_x,prob_z, LowerQubit)
    #Apply Hadamard gate to upper qubit (column vector)
    UpperQubitNoiseAndHadamard = OneQubitGates.h.dot(UpperQubitAfterNoise)
    #create an entangled state of three qubits for the upper qubit (with |00> as ancilla)

    UpperThreeQubits = VectorConv.TensorProdThree(
        UpperQubitNoiseAndHadamard,ZeroState,ZeroState)
    #create an entangled state of three qubits for the lower qubit (with |00> as ancilla)
    LowerThreeQubits = VectorConv.TensorProdThree(
        LowerQubitAfterNoise,ZeroState,ZeroState)
    #Apply BitFlip error correction circuit to upper three qubits
    #Apply BitFlip error correction circuit to lower three qubits
    UpperAfterCorrection = PerformBitFlipCorrection(UpperThreeQubits)
    LowerAfterCorrection = PerformBitFlipCorrection(LowerThreeQubits)
    #Apply Hadamard gate to upper qubit (create a three-qubit gate with two unity one-qubit matrices)
    HadThree = VectorConv.TensorProdThree(OneQubitGates.h,OneQubitGates.unity,OneQubitGates.unity)
    UpperAfterCorrectionAndHadamard = HadThree.dot(UpperAfterCorrection)
    #Create a two-qubit state from upper and lower qubit (discard the ancilla qubits)
    UpperQubitFinal = RetrieveFirstQubit(UpperAfterCorrectionAndHadamard)
    LowerQubitFinal = RetrieveFirstQubit(LowerAfterCorrection)
    #make them a two-qubit state (perform tensor product)
    FinalBeforeCnot = VectorConv.TensorProdTwo(UpperQubitFinal,LowerQubitFinal)
    #apply CNOT gate to the two-qubit state
    FinalAfterCnot = TwoQubitGates.cnot.dot(FinalBeforeCnot)
    #return an instance of the VectorOfQubits class as output
    FinalRowVector = np.transpose(FinalAfterCnot)
    return VectorOfQubits(FinalRowVector)