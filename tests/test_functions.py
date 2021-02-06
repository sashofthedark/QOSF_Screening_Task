import numpy as np
from math import sqrt
import unittest

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))

from numpy.core import function_base
import functions
from vectors import VectorOfQubits

class TestFunctions(unittest.TestCase):
    def test_PerformBitFlipCorrection_type(self):
        wrong_input = VectorOfQubits([1,0,0])
        with self.assertRaises(ValueError):
            functions.PerformBitFlipCorrection(wrong_input)

    def test_ApplyNoise(self):
        pass

    def test_RetrieveFirstQubit(self):
        pass

    def test_CircuitAndCorrection(self):
        pass

if __name__ == 'main':
    unittest.main