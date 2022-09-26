import torch
import numml_torch_cpp
import numpy as np
import numml.sparse as sp

def sp_fd(f, A, h=1e-3):
    '''
    Centered sparse finite differences, used to check gradients.

    Parameters
    ----------
    f : function
      Function that takes a SparseCSRTensor as its parameter and returns a scalar
    A : SparseCSRTensor
      Tensor that will be perturbed to approximate gradient information
    h : float
      Perturbation value

    Returns
    -------
    grad_A : SparseCSRTensor
      Approximate gradient of f with respect to A
    '''

    M = A.copy()

    for i in range(M.nnz):
        A_data_fwd = torch.clone(A.data)
        A_data_fwd[i] += h

        A_data_bwd = torch.clone(A.data)
        A_data_bwd[i] -= h

        f_fwd = f(sp.SparseCSRTensor((A_data_fwd, A.indices, A.indptr), A.shape))
        f_bwd = f(sp.SparseCSRTensor((A_data_bwd, A.indices, A.indptr), A.shape))

        M.data[i] = (f_fwd - f_bwd) / (2*h)

    return M
