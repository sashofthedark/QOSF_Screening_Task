import numpy as np
from math import sqrt
import unittest
from src.vectorconv import VectorConv
from parameterized import parameterized
from vectors import VectorOfQubits
from gates import OneQubitGates,TwoQubitGates

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))

class TestVectorConv(unittest.TestCase):

    PlusState = VectorOfQubits([1/sqrt(2),1/sqrt(2)])
    MinusState = VectorOfQubits([1/sqrt(2),-1/sqrt(2)])
    ZeroState = VectorOfQubits([[1,0]])
    OneState = VectorOfQubits([[0,1]])
    ZeroOneOne = VectorOfQubits([[0,0,0,1,0,0,0,0]])
    ZTensorX = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,-1,0]])
    ZTensorXTensorX = np.array([
        [0,0,0,1,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,-1],
        [0,0,0,0,0,0,-1,0],
        [0,0,0,0,0,-1,0,0],
        [0,0,0,0,-1,0,0,0]
    ])

    @parameterized.expand([
        ['matrices', OneQubitGates.z, OneQubitGates.x,ZTensorX],
        ['two_col',PlusState.colvector,MinusState.colvector,VectorOfQubits([0.5,-0.5,0.5,-0.5]).colvector]
    ])
    def test_VectorConv_ProdTwo(self,name,construct1,construct2,Output):
        KronResult = VectorConv.TensorProdTwo(construct1,construct2)
        np.testing.assert_array_almost_equal(KronResult,Output)

    @parameterized.expand([
        ['col_vec',ZeroState.colvector,OneState.colvector,OneState.colvector,ZeroOneOne.colvector],
        ['matrices',OneQubitGates.z, OneQubitGates.x, OneQubitGates.x, ZTensorXTensorX]
    ])
    def test_VectorConv_ProdThree(self,name,construct1,construct2,construct3,Output):
        KronResult = VectorConv.TensorProdThree(construct1,construct2,construct3)
        print(KronResult)
        np.testing.assert_array_almost_equal(KronResult,Output,verbose=True)

if __name__ == 'main':
    unittest.main