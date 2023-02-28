import torch
import numml_torch_cpp
import numpy as np
import numml.utils as utils
import numml.sparse as sp


class sptrsv(torch.autograd.Function):
    '''
    Sparse triangular solve with single right-hand-side.
    '''

    @staticmethod
    def forward(ctx, shape, A_data, A_indices, A_indptr, lower, unit, b):
        if shape[0] != shape[1]:
            raise RuntimeError(f'Expected square matrix for triangular solve, got {shape[0]} x {shape[1]}.')
        if A_data.type() != b.type():
            raise RuntimeError(f'Matrix and vector should be same data type, got {A_data.type()} and {b.type()}, respectively.')

        x = numml_torch_cpp.sptrsv_forward(shape[0], shape[1], A_data, A_indices, A_indptr, lower, unit, b)
        ctx.save_for_backward(A_data, A_indices, A_indptr, b, x)
        ctx.lower = lower
        ctx.unit = unit
        ctx.shape = shape

        return x

    @staticmethod
    def backward(ctx, grad_x):
        A_data, A_indices, A_indptr, b, x = ctx.saved_tensors
        lower = ctx.lower
        unit = ctx.unit
        shape = ctx.shape

        grad_A_data, grad_b = numml_torch_cpp.sptrsv_backward(grad_x, x, shape[0], shape[1],
                                                              A_data, A_indices, A_indptr, lower, unit, b)

        return (None, # shape
                grad_A_data, # A_data
                None, # A_indices
                None, # A_indptr
                None, # lower
                None, # unit
                grad_b) # b


def splu(A):
    '''
    Compute the sparse LU factorization of an invertible matrix (without pivoting).

    Note that this does not propagate gradient information.  If you need a solve with
    gradients, use spsolve.
    '''

    M_data, M_indices, M_indptr = numml_torch_cpp.splu(A.shape[0], A.shape[1], A.data, A.indices, A.indptr)[:3]
    return sp.SparseCSRTensor((M_data, M_indices, M_indptr), A.shape)


    return sp.SparseCSRTensor((M_data, M_indices, M_indptr), A.shape)


def splu_solve(A_LU, b):
    '''
    Given a sparse LU factorization, solve Ax = b.
    Parameters
    ----------
    A_LU : numml.sparse.SparseCSRTensor
      The output of splu()
    b : torch.Tensor
      Vector right-hand-side
    Returns
    -------
    x : torch.Tensor
      Solution to the matrix equation.
    '''

    y = spsolve_triangular(A_LU, b, True, True)
    z = spsolve_triangular(A_LU, y, False, False)
    return z


class spsolve_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_shape, A_data, A_indices, A_indptr, b):
        M_data, M_indices, M_indptr, Mt_data, Mt_indices, Mt_indptr = \
            numml_torch_cpp.splu(A_shape[0], A_shape[1], A_data, A_indices, A_indptr)

        y = numml_torch_cpp.sptrsv_forward(A_shape[0], A_shape[1], M_data, M_indices, M_indptr, True, True, b)
        x = numml_torch_cpp.sptrsv_forward(A_shape[0], A_shape[1], M_data, M_indices, M_indptr, False, False, y)

        ctx.shape = A_shape
        ctx.save_for_backward(A_data, A_indices, A_indptr, Mt_data, Mt_indices, Mt_indptr, x)

        return x

    @staticmethod
    def backward(ctx, grad_x):
        A_data, A_indices, A_indptr, Mt_data, Mt_indices, Mt_indptr, x = ctx.saved_tensors
        shape = ctx.shape

        grad_A_data, grad_b = numml_torch_cpp.spsolve_backward(grad_x, x, shape[0], shape[1],
                                                               Mt_data, Mt_indices, Mt_indptr,
                                                               A_data, A_indices, A_indptr)

        return (None, # A_shape
                grad_A_data, # A_data
                None, # A_indices
                None, # A_indptr
                grad_b) # b


def spsolve(A, b):
    return spsolve_fn.apply(A.shape, A.data, A.indices, A.indptr, b)


def spsolve_triangular(A, b, lower=True, unit_diagonal=False):
    '''
    Solve the equation $Ax = b$ for x, assuming A is a triangular matrix.

    Parameters
    ----------
    A : (M, M) sparse CSR matrix
      Square triangular matrix to solve.
    b : torch.tensor of shape (M,)
      Right-hand-side vector
    lower : bool, optional
      Is A lower triangular? If not, then will assume upper-triangular.
    unit_diagonal : bool, optional
      If true, diagonal entries will not be accessed and will instead be assumed
      to be 1.

    Returns
    -------
    x : torch.tensor of shape (M,)
      Solution to the matrix equation.
    '''
    return sptrsv.apply(A.shape, A.data, A.indices, A.indptr, lower, unit_diagonal, b)
