class LinearOperator(object):
    def __init__(self, shape, rm=None, lm=None):
        '''
        Defines a wrapper operator class for some object that performs the
        matrix-vector product A times x and/or x times A.

        Parameters
        ----------
        shape : tuple
          Shape of the underlying operator
        rm : callable
          Function that takes torch Tensor x and returns A x
        lm : callable
          Function that takes torch Tensor x and returns x^T A
        '''

        self.shape = shape
        self.right_multiply = rm
        self.left_multiply = lm

    def __matmul__(self, x):
        return self.right_multiply(x)

    def __rmatmul__(self, x):
        return self.left_multiply(x)
