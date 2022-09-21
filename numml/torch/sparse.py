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

        argsort = torch.Tensor(np.lexsort((col_ind, row_ind))).long()
        argsort_inv = torch.empty(N, dtype=torch.long)
        argsort_inv[argsort] = torch.arange(N)

        # create row indices
        rowsort = row_ind[argsort]
        rowptr = torch.zeros(shape[0] + 1, dtype=dtype)
        cur_row = -1
        for i in range(N):
            if cur_row != row_ind[i]:
                cur_row = row_ind[i]
                rowptr[cur_row] = i
        rowptr[-1] = len(data)

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

    x_i = x_i - \alpha x_j

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
        return self.vals.sum()
