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
    if len(InputVector.rowvector[0,:])!=8:
        raise ValueError('The input state is not a three qubit state')
    Vector = InputVector.colvector
    return ThreeQubitGates.bitflip.dot(Vector)

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
    NonzeroIndicesElement = NonzeroIndices[1][:]
    #all of the indices of InputStateRow which lead to a nonzero element in it
    FilteredIndexList = [
        item for item in NonzeroIndicesElement if  InputStateRow[0,item] > 1e-6]
    #leave only the indices which lead to elements greater than the threshold
    FilteredIndexList.sort()
    #sort indices in ascending order

    if len(FilteredIndexList) == 2:
        #in the case we have found two nonzero elements
        UpperQubitIndex = FilteredIndexList[0]
        #index leading to the first qubit element
        LowerQubitIndex = FilteredIndexList[1]
        #index leading to the second qubit element

    else:
        NonzeroQubitIndex = FilteredIndexList[0]
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

    else:

        ZeroState = VectorOfQubits([1,0]).colvector
        LowerQubit = ZeroState
        #this is a zero state (column vector)
        UpperQubit = OneQubitGates.h.dot(LowerQubit)
        #this is a plus state (column vector)

        #pass the first and second qubits separately to ApplyNoise function
        # which will apply noise to them, pass the probabilities to this function
        UpperQubitAfterNoise = ApplyNoise(prob_x,prob_z, UpperQubit) 
        LowerQubitAfterNoise = ApplyNoise(prob_x,prob_z, LowerQubit)
        #Apply Hadamard gate to upper qubit (column vector)

        UpperQubitNoiseAndHadamard = OneQubitGates.h.dot(UpperQubitAfterNoise)
        #create an entangled state of three qubits for the upper qubit (with |00> as ancilla)

        UpperThreeQubitsCol = VectorConv.TensorProdThree(
            UpperQubitNoiseAndHadamard,ZeroState,ZeroState)
        UpperThreeQubitsRow = np.transpose(UpperThreeQubitsCol)
        #create an entangled state of three qubits for the lower qubit (with |00> as ancilla)

        LowerThreeQubits = VectorConv.TensorProdThree(
            LowerQubitAfterNoise,ZeroState,ZeroState)
        LowerThreeQubitsRow = np.transpose(LowerThreeQubits)
        #same for lower qubit

        UpperThreeQubitsClass = VectorOfQubits(list(UpperThreeQubitsRow[0,:]))
        LowerThreeQubitsClass = VectorOfQubits(list(LowerThreeQubitsRow[0,:]))
        #these are of class VectorOfQubits - which is the input for PerformBitFlipCorrection

        UpperAfterCorrection = PerformBitFlipCorrection(UpperThreeQubitsClass)
        LowerAfterCorrection = PerformBitFlipCorrection(LowerThreeQubitsClass)
        #Apply BitFlip error correction circuit to upper and lower three qubits

        HadThree = VectorConv.TensorProdThree(OneQubitGates.h,OneQubitGates.unity,OneQubitGates.unity)
        UpperAfterCorrectionAndHadamard = HadThree.dot(UpperAfterCorrection)
        #Apply Hadamard gate to upper qubit and create a three-qubit gate with two unity one-qubit matrices

        UpperCorrHadClass = VectorOfQubits(list(
            np.transpose(UpperAfterCorrectionAndHadamard)[0,:]))
        LowerCorrClass = VectorOfQubits(list(
            np.transpose(LowerAfterCorrection)[0,:]))

        UpperQubitFinal = RetrieveFirstQubit(UpperCorrHadClass)
        LowerQubitFinal = RetrieveFirstQubit(LowerCorrClass)
        #Create a two-qubit state from upper and lower qubit (discard the ancilla qubits)

        #make them a two-qubit state (perform tensor product)
        FinalBeforeCnot = VectorConv.TensorProdTwo(
            UpperQubitFinal.colvector,LowerQubitFinal.colvector)
        #apply CNOT gate to the two-qubit state
        FinalAfterCnot = TwoQubitGates.cnot.dot(FinalBeforeCnot)
        #return a numpy array which has one column
        return FinalAfterCnot
        
        #we expect to get an entangled state (1/sqrt(2))*(|00> + |11>)