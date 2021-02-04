from vectors import VectorOfQubits
from gates import OneQubitGates,TwoQubitGates,ThreeQubitGates
from Functions import VectorConv
from math import sqrt

def Main():
    #define a list of probabilities for X error gate
    prob_x = [0.02,0.3,0.15,0.1]
    #define a list of probabilities for the Z error gate
    prob_z = [0.1,0.04,0.2,0.1]
    #define upper and lower qubits (as |0> and |+> states)
    UpperQubit = VectorOfQubits(1,0)
    LowerQubit = VectorOfQubits(1/sqrt(2),1/sqrt(2))
    #run the function which applies the noise and error correction and returns a list of Boolean results 
    #(True if error correction succeeded and False otherwise)
    #function parameters are initial states (upper and lower qubits), prob_X, prob_Z, and desired final state.

if __name__ == '__main__':
    Main()