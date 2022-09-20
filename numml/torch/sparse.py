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


# class csr_spmv(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, data, col_ind, rowptr, shape, x):
#         y = torch.zeros(shape[0], dtype=x.dtype)
#         for row_i in range(len(rowptr) - 1):
#             for col_j in range(rowptr[row_i], rowptr[row_i + 1]):
#                 y[row_i] += data[col_j] * x[col_ind[col_j]]
#         return y

#     @staticmethod
#     def backward(ctx, data, col_ind, rowptr):
#         return None, None, None, None, None


class spgemv(torch.autograd.Function):
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


class SparseCSRTensor(object):
    def __init__(self, arg1, shape=None):
        if isinstance(arg1, torch.Tensor):
            if arg1.layout == torch.sparse_coo:
                arg1 = arg1.coalesce()
                vals = arg1.values()
                rows, cols = arg1.indices()
                self.data, self.indices, self.indptr = coo_to_csr.apply(vals, rows, cols, arg1.shape)
                if shape is not None:
                    self.shape = shape
                else:
                    self.shape = arg1.shape
            else:
                raise RuntimeError('not implemented: csr from dense')

    def spmv(self, x):
        return spgemv.apply(self.shape, torch.tensor(1.), self.data, self.indices, self.indptr, x, torch.tensor(0.), torch.zeros(self.shape[0]))

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
