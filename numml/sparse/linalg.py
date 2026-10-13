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


class SparseLU():
    '''
    Sparse LU factorization of a matrix returned by `splu()`

    This holds on to the original matrix alongside the factors, so that solves with
    the factorization can propagate gradient information back to the entries of A.
    '''

    def __init__(self, A, Pc, Pr, M, Mt):
        self.A = A
        self.Pc = Pc
        self.Pr = Pr
        self.M = M
        self.Mt = Mt

    @property
    def shape(self):
        return self.A.shape

    def solve(self, b):
        return splu_solve(self, b)


def splu(A):
    '''
    Compute the sparse LU factorization of an invertible matrix (with pivoting).

    The factorization object is intended to be an opaque handle passed to `splu_solve()`.
    Gradients will be propagated to the entries of A through any subsequent solves.

    Parameters
    ----------
    A : numml.sparse.SparseCSRTensor
      Square matrix to factorize

    Returns
    -------
    A_LU : SparseLU
      Factorization of A
    '''

    Pc, Pr, M_data, M_indices, M_indptr, Mt_data, Mt_indices, Mt_indptr = \
        numml_torch_cpp.splu(A.shape[0], A.shape[1], A.data, A.indices, A.indptr)

    return SparseLU(A, Pc, Pr,
                    sp.SparseCSRTensor((M_data, M_indices, M_indptr), A.shape),
                    sp.SparseCSRTensor((Mt_data, Mt_indices, Mt_indptr), A.shape))


class splu_solve_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_shape, A_data, A_indices, A_indptr, Pc, Pr,
                M_data, M_indices, M_indptr, Mt_data, Mt_indices, Mt_indptr, b):
        if A_data.type() != b.type():
            raise RuntimeError(f'Matrix and vector should be same data type, got {A_data.type()} and {b.type()}, respectively.')

        y = numml_torch_cpp.sptrsv_forward(A_shape[0], A_shape[1], M_data, M_indices, M_indptr, True, True,
                                           numml_torch_cpp.permute_inverse(b, Pr))
        x = numml_torch_cpp.permute(numml_torch_cpp.sptrsv_forward(A_shape[0], A_shape[1], M_data, M_indices, M_indptr, False, False, y), Pc)

        ctx.shape = A_shape
        ctx.save_for_backward(A_data, A_indices, A_indptr,
                              Mt_data, Mt_indices, Mt_indptr, x, Pr, Pc)

        return x

    @staticmethod
    def backward(ctx, grad_x):
        A_data, A_indices, A_indptr,\
            Mt_data, Mt_indices, Mt_indptr, x, Pr, Pc = ctx.saved_tensors
        shape = ctx.shape

        grad_A_data, grad_b = numml_torch_cpp.spsolve_backward(grad_x, x, shape[0], shape[1],
                                                               Mt_data, Mt_indices, Mt_indptr,
                                                               A_data, A_indices, A_indptr,
                                                               Pr, Pc)

        return (None, # A_shape
                grad_A_data, # A_data
                None, # A_indices
                None, # A_indptr
                None, # Pc
                None, # Pr
                None, # M_data
                None, # M_indices
                None, # M_indptr
                None, # Mt_data
                None, # Mt_indices
                None, # Mt_indptr
                grad_b) # b


def splu_solve(A_LU, b):
    '''
    Given a sparse LU factorization, solve Ax = b.

    Parameters
    ----------
    A_LU : SparseLU
      The output of splu()
    b : torch.Tensor
      Vector right-hand-side

    Returns
    -------
    x : torch.Tensor
      Solution to the matrix equation.
    '''

    A, M, Mt = A_LU.A, A_LU.M, A_LU.Mt
    return splu_solve_fn.apply(A.shape, A.data, A.indices, A.indptr, A_LU.Pc, A_LU.Pr,
                               M.data, M.indices, M.indptr, Mt.data, Mt.indices, Mt.indptr, b)


def spsolve(A, b):
    '''
    Solve the equation $Ax = b$ for x using a sparse LU factorization.

    Parameters
    ----------
    A : (M, M) sparse CSR matrix
      Square matrix to solve.
    b : torch.tensor of shape (M,)
      Right-hand-side vector

    Returns
    -------
    x : torch.tensor of shape (M,)
      Solution to the matrix equation.
    '''

    return splu_solve(splu(A), b)


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
