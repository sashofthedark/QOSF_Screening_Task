import numpy as np
from math import sqrt 
from vectors import VectorOfQubits
from gates import OneQubitGates,TwoQubitGates,ThreeQubitGates
from vectorconv import VectorConv
from numpy.random import choice

def PerformBitFlipCorrection(InputVector: VectorOfQubits):
    '''
    This fuction applies the bit flip correction gate to an input quantum state. 
    The state should be a three qubit state
    '''
    if len(InputVector.rowvector[0,:])!=8 or len(InputVector.colvector[:,0])!=8:
        raise ValueError('The input state is not a three qubit state')
    Vector = InputVector.colvector
    return ThreeQubitGates.BitFlipGate().dot(Vector)

def ApplyNoise(prob_x,prob_z,InputVector: np.array):
    '''
    Applies either unity, X or Z gates randomly, using the provided probabilities and 
    one qubit input column vector
    '''
    prob_unity = 1 - prob_x - prob_z
    prob_dist = [prob_unity,prob_x,prob_z]
    gates = [OneQubitGates.unity,OneQubitGates.x,OneQubitGates.z]
    SelectedGateIndex = choice(a=3,p=prob_dist)
    SelectedGate = gates[SelectedGateIndex]
    VectorAfterNoise = SelectedGate.dot(InputVector)
    #InputVector is a column vector
    return VectorAfterNoise

def RetrieveFirstQubit(InputVector: VectorOfQubits):
    '''
    this function assumes the second and third qubits are either |0> or |1>
    It accepts as input an instance of the VectorOfQubits class, and returns 
    an instance of VectorOfQubits corresponding to the upper (first) qubit
    '''
    InputStateRow = InputVector.rowvector
    NonzeroIndices = np.nonzero(InputStateRow)
    if len(NonzeroIndices[1]) == 2:
        #there are two nonzero elements in the array
        UpperQubitIndex = NonzeroIndices[1][0]
        LowerQubitIndex = NonzeroIndices[1][1] 

    else:
        NonzeroQubitIndex = NonzeroIndices[1][0]
        #there is only one nonzero element in the array

        if (NonzeroQubitIndex + 4) > 7:
            UpperQubitIndex = NonzeroQubitIndex - 4
            LowerQubitIndex = NonzeroQubitIndex 

        else:
            UpperQubitIndex = NonzeroQubitIndex
            LowerQubitIndex = NonzeroQubitIndex + 4

    return VectorOfQubits([InputStateRow[0,UpperQubitIndex] , InputStateRow[0,LowerQubitIndex]])     

    

def CircuitAndCorrection(prob_x, prob_z):
    '''
    This function receives a |0+> initial vector, applies noise to each qubit
    based on the provided probabilities and then performs error 
    correction to get the correct final entangled wavefunction.
    '''
    if (prob_x + prob_z) > 1:
        raise ValueError("Sum of probabilities greater than one")
    elif prob_x >= 0.25 or prob_z >= 0.25:
        raise ValueError("The error correction code is not effective for these probabilities")
    else:
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