import torch
import torch.autograd
import numml_torch_cpp
import numpy as np


class coo_to_csr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, row_ind, col_ind, shape):
        data = torch.clone(values).detach()
        dtype = data.dtype
        N = len(row_ind)

        # Sort COO representation by rows, then columns.  We save the inverse of the
        # sort so that we can propagate gradients.
        argsort = torch.Tensor(np.lexsort((col_ind, row_ind))).long()
        argsort_inv = torch.empty(N, dtype=torch.long)
        argsort_inv[argsort] = torch.arange(N)

        # create rowpointer indices as cumulative sum of nonzeros in each row
        rowsort = row_ind[argsort]
        rowptr = torch.zeros(shape[0] + 1, dtype=dtype)

        nnz_per_row = torch.zeros(shape[0])
        for i in range(N):
            nnz_per_row[row_ind[i]] += 1

        nnz = len(row_ind)
        cumsum = 0
        for i in range(shape[0]):
            rowptr[i] = cumsum
            cumsum += nnz_per_row[i]
        rowptr[-1] = nnz

        ctx.save_for_backward(argsort_inv, row_ind, col_ind)
        return data, col_ind[argsort], rowptr.long()

    @staticmethod
    def backward(ctx, data, col_ind, rowptr):
        argsort_inv, row_ind, col_ind = ctx.saved_tensors
        return data[argsort_inv], None, None, None


class spgemv(torch.autograd.Function):
    '''
    Sparse general matrix times vector product
    '''

    @staticmethod
    def forward(ctx, A_shape, alpha, A_data, A_col_ind, A_rowptr, x, beta, y):
        Ax = torch.zeros(A_shape[0], dtype=x.dtype)
        for row_i in range(len(A_rowptr) - 1):
            for col_j in range(A_rowptr[row_i], A_rowptr[row_i + 1]):
                Ax[row_i] += A_data[col_j] * x[A_col_ind[col_j]]
        z = alpha * Ax + beta * y

        ctx.save_for_backward(Ax, x, y, A_data, A_col_ind, A_rowptr)
        ctx.shape = A_shape
        ctx.alpha = alpha
        ctx.beta = beta

        return z

    @staticmethod
    def backward(ctx, df_dz):
        Ax, x, y, A_data, A_col_ind, A_rowptr = ctx.saved_tensors
        A_shape = ctx.shape
        alpha = ctx.alpha
        beta = ctx.beta

        # A_data
        d_A_data = torch.clone(A_data)
        for row in range(len(A_rowptr) - 1):
            for i in range(A_rowptr[row], A_rowptr[row+1]):
                col = A_col_ind[i]
                d_A_data[i] = alpha * df_dz[row] * x[col]

        # df_dx
        df_dx = torch.zeros_like(x)
        for row in range(len(A_rowptr) - 1):
            for i in range(A_rowptr[row], A_rowptr[row+1]):
                col = A_col_ind[i]
                df_dx[col] += df_dz[row] * A_data[i] * alpha

        return (None, # A_shape
                torch.sum(df_dz * Ax), # alpha
                d_A_data, # A_data
                None, # A_col_ind
                None, # A_rowptr
                df_dx, # x
                torch.sum(df_dz * y), # beta
                df_dz * beta) # y


class ScaleVecPrimitive(torch.autograd.Function):
    '''
    Atomic torch function for scaling a specific entry of a vector, while keeping
    all other entries the same.
    '''

    @staticmethod
    def forward(ctx, x, i, alpha):
        '''
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
          Vector to perform the operation on
        i : integer
          Entry index to scale
        alpha : float
          Amount to scale the entry by

        Returns
        -------
        z : torch.Tensor
          Scaled vector with same shape as x
        '''

        z = torch.clone(x)
        z[i] = z[i] * alpha
        ctx.alpha = alpha
        ctx.i = i
        ctx.xi = x[i].detach()
        return z

    @staticmethod
    def backward(ctx, z_grad):
        '''
        Backward pass

        Parameters
        ----------
        z_grad : torch.Tensor
          Vector gradient of scalar loss function wrt z

        Returns
        -------
        x_grad : torch.Tensor
          Vector gradient of scalar loss function wrt x
        i_grad : None
          Gradient of scalar loss wrt i (does not exist)
        alpha_grad : float
          Derivative of scalar loss wrt alpha
        '''

        alpha = ctx.alpha
        i = ctx.i

        x_grad = z_grad.clone()
        x_grad[i] *= alpha.item()

        return x_grad, None, ctx.xi * z_grad[i]

p_scale = ScaleVecPrimitive.apply

class TriSolveSub(torch.autograd.Function):
    '''
    Atomic torch function for computing the vector operation

    x_i = x_i - \\alpha x_j

    Used for the element-wise substitution in triangular solves.
    '''

    @staticmethod
    def forward(ctx, x, i, j, Mij):
        z = torch.clone(x)
        z[i] = z[i] - Mij * x[j]

        ctx.Mij = Mij.detach()
        ctx.i = i
        ctx.j = j
        ctx.xj = x[j].detach()
        return z

    @staticmethod
    def backward(ctx, x_grad):
        i, j = ctx.i, ctx.j
        mij = ctx.Mij

        grad_x = torch.clone(x_grad)
        grad_x[j] += -mij * x_grad[i]

        return grad_x, None, None, -ctx.xj * x_grad[i]
p_trisolvesub = TriSolveSub.apply


def sstrsv(upper, unit, A_shape, A_data, A_col_ind, A_rowptr, b):
    '''
    Sparse Solve Triangular System with Single Vector
    Solves an upper or lower triangular system with a single right-hand-side vector.  The
    triangular system can optionally have unit diagonal.
    '''

    rows, cols = A_shape
    assert(rows == cols)
    x = torch.clone(b)

    if upper: # Upper triangular system
        for row in range(rows-1, -1, -1): # Backwards from last row
            diag_entry = None

            for i in range(A_rowptr[row+1]-1, A_rowptr[row]-1, -1): # Go backwards from the end of the row
                col = A_col_ind[i]
                if col == row and not unit:
                    diag_entry = A_data[i]
                elif col >= row:
                    x = p_trisolvesub(x, row, col, A_data[i]) # x[row] -= A_data[i] * x[col]
                else: # col <= row
                    break # Stop because we are to the left of the diagonal

            # Rescale entry in x by diagonal entry of A
            if not unit:
                if diag_entry is None:
                    raise RuntimeError('Triangular system is singular: no nonzero entry on diagonal.  Did you mean to pass unit=True?')
                if diag_entry == 0.:
                    raise RuntimeError('Triangular system is singular: explicit zero entry given on diagonal.')

                x = p_scale(x, row, 1. / diag_entry) # x[row] /= diag_entry
    else: # Lower triangular system
        for row in range(rows): # Forward from first row
            diag_entry = None

            for i in range(A_rowptr[row], A_rowptr[row + 1]): # Go forward from the start of the row
                col = A_col_ind[i]
                if col == row and not unit:
                    diag_entry = A_data[i]
                elif col <= row:
                    x = p_trisolvesub(x, row, col, A_data[i])
                else: # col >= row:
                    break # Stop because we are to the right of the diagonal

            # Rescale entry in x by diagonal entry of A
            if not unit:
                if diag_entry is None:
                    raise RuntimeError('Triangular system is singular: no nonzero entry on diagonal.  Did you mean to pass unit=True?')
                if diag_entry == 0.:
                    raise RuntimeError('Triangular system is singular: explicit zero entry given on diagonal.')

                x = p_scale(x, row, 1. / diag_entry)

    return x


class splincomb(torch.autograd.Function):
    '''
    Computes the linear combination of two sparse matrices like
    C = \\alpha A + \\beta B.
    '''

    @staticmethod
    def forward(ctx, shape,
                alpha, A_data, A_col_ind, A_rowptr,
                beta,  B_data, B_col_ind, B_rowptr):

        C_data = []
        C_col_ind = []
        C_rowptr = []

        rows, cols = shape

        for row in range(rows):
            C_rowptr.append(len(C_data))

            i_A = (A_rowptr[row]).item()
            i_B = (B_rowptr[row]).item()

            end_A = (A_rowptr[row + 1]).item()
            end_B = (B_rowptr[row + 1]).item()

            # Merge row of A and B
            while i_A < end_A and i_B < end_B:
                col_A = A_col_ind[i_A]
                col_B = B_col_ind[i_B]

                if col_A < col_B:
                    C_data.append(alpha * A_data[i_A])
                    C_col_ind.append(col_A)
                    i_A += 1
                elif col_A > col_B:
                    C_data.append(beta * B_data[i_B])
                    C_col_ind.append(col_B)
                    i_B += 1
                else: # col_A == col_B
                    C_data.append(alpha * A_data[i_A] + beta * B_data[i_B])
                    C_col_ind.append(col_A)
                    i_A += 1
                    i_B += 1

            # Exhausted shared indices, now add rest of row of A/B
            while i_A < end_A:
                C_data.append(alpha * A_data[i_A])
                C_col_ind.append(A_col_ind[i_A])
                i_A += 1
            while i_B < end_B:
                C_data.append(beta * B_data[i_B])
                C_col_ind.append(B_col_ind[i_B])
                i_B += 1

        C_rowptr.append(len(C_data))

        C_data = torch.Tensor(C_data)
        C_col_ind = torch.Tensor(C_col_ind).long()
        C_rowptr = torch.Tensor(C_rowptr).long()

        ctx.save_for_backward(torch.tensor(alpha), A_data, A_col_ind, A_rowptr, torch.tensor(beta), B_data, B_col_ind, B_rowptr, C_data, C_col_ind, C_rowptr)
        ctx.shape = shape

        return (C_data, C_col_ind, C_rowptr)


    @staticmethod
    def backward(ctx, grad_C_data, _grad_C_col_ind, _grad_C_rowptr):
        alpha, A_data, A_col_ind, A_rowptr, beta, B_data, B_col_ind, B_rowptr, C_data, C_col_ind, C_rowptr = ctx.saved_tensors
        shape = ctx.shape

        # d_da = alpha * grad_c (*) mask(A)
        grad_A = torch.zeros_like(A_data)
        for row in range(shape[0]):
            A_i = (A_rowptr[row]).item()
            C_i = (C_rowptr[row]).item()

            A_end = (A_rowptr[row+1]).item()
            C_end = (C_rowptr[row+1]).item()

            while A_i < A_end and C_i < C_end:
                A_col = A_col_ind[A_i]
                C_col = C_col_ind[C_i]

                if A_col < C_col:
                    A_i += 1
                elif A_col > C_col:
                    C_i += 1
                else:
                    grad_A[A_i] = grad_C_data[C_i] * alpha
                    A_i += 1
                    C_i += 1

        # d_db = beta * grad_c (*) mask(B)
        grad_B = torch.zeros_like(B_data)
        for row in range(shape[0]):
            B_i = (B_rowptr[row]).item()
            C_i = (C_rowptr[row]).item()

            B_end = (B_rowptr[row+1]).item()
            C_end = (C_rowptr[row+1]).item()

            while B_i < B_end and C_i < C_end:
                B_col = B_col_ind[B_i]
                C_col = C_col_ind[C_i]

                if B_col < C_col:
                    B_i += 1
                elif B_col > C_col:
                    C_i += 1
                else:
                    grad_B[B_i] = grad_C_data[C_i] * beta
                    B_i += 1
                    C_i += 1

        return (None, # shape
                torch.sum(A_data), # alpha
                grad_A, # A_data
                None, # A_col_ind
                None, # A_rowptr
                torch.sum(B_data), # beta
                grad_B, # B_data
                None, # B_col_ind
                None) # B_rowptr


def eye(N, k=0):
    assert(abs(k) < N)

    N_k = N - abs(k)
    rows = None
    cols = None
    vals = torch.ones(N_k)

    if k == 0:
        rows = torch.arange(N)
        cols = torch.arange(N)
    elif k > 0:
        rows = torch.arange(N_k)
        cols = torch.arange(k, N)
    else:
        rows = torch.arange(abs(k), N)
        cols = torch.arange(N_k)

    return SparseCSRTensor((vals, (rows, cols)), shape=(N, N))

class SparseCSRTensor(object):
    def __init__(self, arg1, shape=None):
        if isinstance(arg1, torch.Tensor):
            if arg1.layout == torch.sparse_coo:
                # Input is torch sparse COO

                arg1 = arg1.coalesce()
                vals = arg1.values()
                rows, cols = arg1.indices()
                self.data, self.indices, self.indptr = coo_to_csr.apply(vals, rows, cols, arg1.shape)
            else:
                # Input is torch dense

                rows, cols = torch.nonzero(arg1).T
                nz = arg1[rows, cols]
                self.data, self.indices, self.indptr = coo_to_csr.apply(nz, rows, cols, arg1.shape)

            if shape is not None:
                self.shape = shape
            else:
                self.shape = arg1.shape
        elif isinstance(arg1, tuple):
            if len(arg1) == 3:
                # Input is CSR: (data, indices, indptr)
                assert(shape is not None)

                self.data = torch.clone(arg1[0])
                self.indices = torch.clone(arg1[1])
                self.indptr = torch.clone(arg1[2])
                self.shape = shape
            elif len(arg1) == 2:
                # Input is COO: (data, (row_ind, col_ind))
                assert(shape is not None)

                data, (rows, cols) = arg1
                self.data, self.indices, self.indptr = coo_to_csr.apply(data, rows, cols, shape)
                self.shape = shape
        else:
            raise RuntimeError(f'Unknown type given as argument of SparseCSRTensor: {type(arg1)}')

    def spmv(self, x):
        return spgemv.apply(self.shape, torch.tensor(1.), self.data, self.indices, self.indptr, x, torch.tensor(0.), torch.zeros(self.shape[0]))

    def solve_triangular(self, upper, unit, b):
        return sstrsv(upper, unit, self.shape, self.data, self.indices, self.indptr, b)

    def __matmul__(self, x):
        dims = len(torch.squeeze(x).shape)
        if dims == 1:
            return self.spmv(x)
        elif dims == 2:
            raise RuntimeError('not implemented: spmm/spspmm')
        else:
            raise RuntimeError(f'invalid tensor found for sparse multiply: mode {dims} tensor found.')

    def to_dense(self):
        X = torch.zeros(self.shape)
        for row_i in range(len(self.indptr) - 1):
            for data_j in range(self.indptr[row_i], self.indptr[row_i + 1]):
                X[row_i, self.indices[data_j]] = self.data[data_j]
        return X

    def sum(self):
        return self.data.sum()

    def abs(self):
        return SparseCSRTensor((torch.abs(self.data), self.indices, self.indptr), self.shape)

    def __add__(self, othr):
        assert(self.shape == othr.shape)

        C = splincomb.apply(self.shape,
                            torch.tensor(1.), self.data, self.indices, self.indptr,
                            torch.tensor(1.), othr.data, othr.indices, othr.indptr)
        return SparseCSRTensor(C, self.shape)

    def __sub__(self, othr):
        assert(self.shape == othr.shape)

        C = splincomb.apply(self.shape,
                            torch.tensor(1.) , self.data, self.indices, self.indptr,
                            torch.tensor(-1.), othr.data, othr.indices, othr.indptr)
        return SparseCSRTensor(C, self.shape)

    def __mul__(self, other):
        if (isinstance(other, float) or
            isinstance(other, int) or
            (isinstance(other, torch.Tensor) and len(other.shape) == 0)):
            return SparseCSRTensor((self.data * other, self.indices, self.indptr), shape=self.shape)
        elif isinstance(other, torch.Tensor) and len(torch.squeeze(other).shape) == 2:
            raise RuntimeError('Element-wise hadamard product not implemented')
        else:
            raise RuntimeError(f'Unknown type for multiplication: {type(other)}.')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return SparseCSRTensor((-self.data, self.indices, self.indptr), shape=self.shape)

    def __repr__(self):
        grad_str = ''
        if self.data.grad_fn is not None:
            grad_str = f', grad_fn=<{self.data.grad_fn.__class__.__name__}>'

        return f"<{self.shape[0]}x{self.shape[1]} sparse matrix tensor of type '{self.data.dtype}'\n\twith {len(self.data)} stored elements in Compressed Sparse Row format{grad_str}>"

    @property
    def requires_grad(self):
        return self.data.requires_grad

    @requires_grad.setter
    def requires_grad(self, b):
        self.data.requires_grad = b

    @property
    def grad(self):
        return SparseCSRTensor((self.data.grad, self.indices, self.indptr), shape=self.shape)

    @property
    def grad_fn(self):
        return self.data.grad_fn

    @property
    def nnz(self):
        '''
        Number of stored values, including explicit zeros
        '''
        return len(self.data)
