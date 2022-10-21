import torch
import torch.autograd
import numml_torch_cpp
import numpy as np
import numml.sparse as sp

def _to_scalar(f):
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

def _f_wargs(f, X, fargs):
    if fargs is None:
        return f(X)
    else:
        return f(X, *fargs)

def sp_fd(f, A, fargs=None, h=1e-3):
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
      Approximate gradient of f with respect to A evaluated at A
    '''

    M = A.copy()

    for i in range(M.nnz):
        A_data_fwd = torch.clone(A.data)
        A_data_fwd[i] += h

        A_data_bwd = torch.clone(A.data)
        A_data_bwd[i] -= h

        with torch.no_grad():
            f_fwd = _to_scalar(_f_wargs(sp.SparseCSRTensor((A_data_fwd, A.indices, A.indptr), A.shape), fargs))
            f_bwd = _to_scalar(_f_wargs(sp.SparseCSRTensor((A_data_bwd, A.indices, A.indptr), A.shape), fargs))

        M.data[i] = (f_fwd - f_bwd) / (2*h)

    return M

def reg_fd(f, x, fargs=None, h=1e-3):
    '''
    Centered finite differences on non-sparse inputs, used to check gradients.

    Parameters
    ----------
    f : function
      Function that takes a torch Tensor as its parameter and returns a scalar
    x : torch.Tensor
      Tensor that will be permuted to approximate gradient information
    h : float
      Perturbation value

    Returns
    -------
    grad_x : torch.Tensor
      Approximate gradient of f with respect to x evaluated at x
    '''

    dx = torch.empty_like(x).flatten()

    for i in range(len(dx)):
        x_fwd = x.clone().flatten()
        x_fwd[i] += h
        x_fwd = x_fwd.reshape(x.shape)

        x_bwd = x.clone().flatten()
        x_bwd[i] -= h
        x_bwd = x_bwd.reshape(x.shape)

        with torch.no_grad():
            f_fwd = _to_scalar(_f_wargs(f, x_fwd, fargs))
            f_bwd = _to_scalar(_f_wargs(f, x_bwd, fargs))

        dx[i] = (f_fwd - f_bwd) / (2*h)

    return dx.reshape(x.shape)


def fd(f, X, fargs=None, h=1e-3):
    if isinstance(X, sp.SparseCSRTensor):
        return sp_fd(f, X, fargs, h)
    elif isinstance(X, torch.Tensor):
        return reg_fd(f, X, fargs, h)
    else:
        raise RuntimeError(f'Encountered unknown type when trying to take finite difference: {X.type}.')


def sp_grad(f, A, fargs=None):
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

    A = A.copy().detach()
    A.requires_grad = True

    (grad_A_data,) = torch.autograd.grad(_to_scalar(_f_wargs(f, A, fargs)), A.data)
    return sp.SparseCSRTensor((grad_A_data, A.indices, A.indptr), A.shape)

def reg_grad(f, X, fargs=None):
    X = X.clone().detach()
    X.requires_grad = True

    (grad_X,) = torch.autograd.grad(_to_scalar(_f_wargs(f, X, fargs)), X)
    return grad_X

def grad(f, X, fargs=None):
    if isinstance(X, sp.SparseCSRTensor):
        return sp_grad(f, X, fargs)
    elif isinstance(X, torch.Tensor):
        return reg_grad(f, X, fargs)
    else:
        raise RuntimeError(f'Encountered unknown type when trying to take gradient: {X.type}.')

def unsqueeze_like(x, y):
    '''
    Unsqueezes tensor x so that it has the same singleton dimensions as y

    Parameters
    ----------
    x : torch.Tensor
      Tensor to unsqueeze
    y : torch.Tensor
      Tensor to copy dimensionality
    '''

    one_dim = torch.where(torch.tensor(y.size()) == 1)[0]
    for dim in one_dim:
        x = torch.unsqueeze(x, dim)
    return x
