class VectorConv():

    def TensorProdTwo(construct1,construct2):
        #construct is either a column vector, a row vector or a matrix
        return np.kron(construct1,construct2)

    def TensorProdThree(construct1,construct2,construct3):
        #construct is either a column vector, a row vector or a matrix
        #it's Const1 X Const2 X Const3 (first 2 and 3, then 1 with the resulting product)
        return np.kron(np.kron(construct2,construct3),construct1)