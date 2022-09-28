import torch
import torch.autograd
import numml_torch_cpp
import numpy as np
import numml.sparse as sp

def to_scalar(f):
    '''
    Small helper for the gradient functions below.  If a function gives some
    tensor output instead of a scalar, return the sum of the entries as the gradient
    of this operation wrt each entry will be 1.
    '''

    if (isinstance(f, torch.Tensor) or
        isinstance(f, np.ndarray)):
        return f.sum()
    elif isinstance(f, int) or isinstance(f, float):
        return f
    else:
        raise RuntimeError(f'Unknown type for output: {type(f)}')

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

        f_fwd = to_scalar(f(sp.SparseCSRTensor((A_data_fwd, A.indices, A.indptr), A.shape)))
        f_bwd = to_scalar(f(sp.SparseCSRTensor((A_data_bwd, A.indices, A.indptr), A.shape)))

        M.data[i] = (f_fwd - f_bwd) / (2*h)

    return M

def sp_grad(f, A):
    '''
    Wrapper around torch's autograd function to return gradient
    information for a sparse tensor.

    Parameters
    ----------
    f : function
      Function that takes a SparseCSRTensor as its parameter and returns a scalar
    A : SparseCSRTensor
      Tensor for which the gradient of f will be taken with respect to

    Returns
    -------
    grad_A : SparseCSRTensor
      The gradient of f with respect to the nonzero entries of A.
      Computed using Torch's autogradient module.
    '''

    if not A.requires_grad:
        A = A.copy()
        A.requires_grad = True

    (grad_A_data,) = torch.autograd.grad(to_scalar(f(A)), A.data)
    return sp.SparseCSRTensor((grad_A_data, A.indices, A.indptr), A.shape)
