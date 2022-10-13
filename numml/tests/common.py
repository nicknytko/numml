import torch
import torch.linalg as tla
import numml.sparse as sp
import numpy as np


def relerr(A, Ahat):
    A_fl = A.to_dense().flatten()
    Ahat_fl = Ahat.to_dense().flatten()
    return tla.norm(A_fl - Ahat_fl) / tla.norm(A_fl)


def random_sparse(rows, cols, sparsity):
    '''
    Creates a random matrix of the given size, whose entries are unit normally distributed.

    Parameters
    ----------
    rows : int
      Number of rows
    cols : int
      Number of columns
    sparsity : float
      Sparsity ratio between 0 and 1.  Number of nonzeros is given by ceil(sparsity * rows * cols)

    Returns
    -------
    rand_dense : torch.Tensor
      Dense matrix with the given parameters
    rand: SparseCSRTensor
      The same matrix as in rand_dense but in sparse representation
    '''

    nnz = int(np.ceil(sparsity * (rows * cols)))
    A_d = torch.zeros((rows, cols))
    idx_r, idx_c = torch.randint(rows, (nnz,)), torch.randint(cols, (nnz,))
    A_d[idx_r, idx_c] = torch.randn(nnz)

    return A_d, sp.SparseCSRTensor(A_d)
