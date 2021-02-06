import numpy as np
from math import sqrt
import unittest
from parameterized import parameterized

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))

from numpy.core import function_base
import functions
from vectors import VectorOfQubits
from vectorconv import VectorConv

class TestFunctions(unittest.TestCase):
    ZeroState = VectorOfQubits([1,0]).rowvector
    OneState = VectorOfQubits([0,1]).rowvector

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

    def test_ApplyNoise(self):
        pass

    def test_RetrieveFirstQubit(self):
        pass

    def test_CircuitAndCorrection(self):
        pass

if __name__ == 'main':
    unittest.main