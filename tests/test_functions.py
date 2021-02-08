import os
import sys
import unittest
from math import sqrt

import numpy as np
from parameterized import parameterized

sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))

import functions
from numpy.core import function_base
from vectorconv import VectorConv
from vectors import VectorOfQubits,EntangledVector


class TestFunctions(unittest.TestCase):
    ZeroState = VectorOfQubits([1,0]).rowvector
    OneState = VectorOfQubits([0,1]).rowvector
    PlusState = VectorOfQubits([1/sqrt(2),1/sqrt(2)]).rowvector

    def test_PerformBitFlipCorrection_type(self):
        wrong_input = VectorOfQubits([1,0,0])
        with self.assertRaises(ValueError):
            functions.PerformBitFlipCorrection(wrong_input)

    @parameterized.expand([
        ['100',OneState, ZeroState, ZeroState, [0,0,0,1,0,0,0,0]],
        ['011',ZeroState, OneState, OneState, [0,0,0,0,0,0,0,1]],
        ['111',OneState, OneState, OneState, [0,0,0,0,1,0,0,0]]
    ])
    def test_PerformBitFlipCorrection(self, name, FirstQubit, SecondQubit, ThirdQubit, OutState):
        InputState = VectorConv.TensorProdThree(FirstQubit, SecondQubit, ThirdQubit)
        OutputVector = np.transpose(functions.PerformBitFlipCorrection(VectorOfQubits(InputState[0,:])))
        np.testing.assert_array_equal(OutputVector,VectorOfQubits(OutState).rowvector)

    @parameterized.expand([
        ['X', np.transpose(OneState), 1, 0, np.transpose(ZeroState)],
        ['Z', np.transpose(OneState), 0, 1, -1*np.transpose(OneState)],
        ['Unity', np.transpose(OneState), 0, 0, np.transpose(OneState)]
    ])
    def test_ApplyNoise(self, name, input_state, p_x, p_z, OutState):
        InputState = input_state
        ExpOutState = functions.ApplyNoise(p_x, p_z, InputState)
        np.testing.assert_array_equal(ExpOutState, OutState)

    @parameterized.expand([ 
        ['zero',ZeroState,OneState,OneState],
        ['one', OneState,OneState,OneState],
        ['plus',PlusState,ZeroState,ZeroState]
    ])
    def test_RetrieveFirstQubit(self,name,FirstQubit,SecondQubit,ThirdQubit):
        InputState = VectorConv.TensorProdThree(
            np.transpose(FirstQubit),np.transpose(SecondQubit),np.transpose(ThirdQubit))
        print(len(InputState))
        #generate the three qubit input state
        TransposedState = list(np.transpose(InputState)[0,:])
        InputStateV = VectorOfQubits(TransposedState)
        OutputState = functions.RetrieveFirstQubit(InputStateV)
        np.testing.assert_array_almost_equal(OutputState.rowvector,FirstQubit)

    def test_CircuitAndCorrectionRaises(self):
        p_x_wrong1 = 1.1
        p_x_right1 = 0.2
        p_z_wrong1 = 0.99
        p_z_right1 = 0.04
        #the sum of p_x and p_z in these examples is greater than 1

        with self.assertRaises(ValueError):
            functions.CircuitAndCorrection(p_x_wrong1,p_z_right1)
        with self.assertRaises(ValueError):
            functions.CircuitAndCorrection(p_x_right1,p_z_wrong1)

    @parameterized.expand([
        ['first',0.01,0.1],
        ['second',0.2,0.03],
        ['third',0.1,0.14],
        ['fourth',0.6,0.01],
        ['fifth',0.01,0.7]
    ])
    def test_CircuitAndCorrection(self,name,p_x,p_z):
        ExpOutputState = EntangledVector((1/sqrt(2)),[1,0,0,0],[0,0,0,1]).colvector
        np.testing.assert_array_almost_equal(
            functions.CircuitAndCorrection(p_x,p_z),ExpOutputState)

if __name__ == 'main':
    unittest.main
