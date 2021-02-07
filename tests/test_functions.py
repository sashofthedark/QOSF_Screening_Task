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
from vectors import VectorOfQubits


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

    def test_CircuitAndCorrection(self):
        pass

if __name__ == 'main':
    unittest.main
