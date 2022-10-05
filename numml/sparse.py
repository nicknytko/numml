import torch
import numml_torch_cpp
import numpy as np


class coo_to_csr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, row_ind, col_ind, shape, sorted=False):
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
        Ax, z = numml_torch_cpp.spgemv_forward(A_shape[0], A_shape[1], alpha,
                                               A_data, A_col_ind, A_rowptr, x, beta, y)

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

        grad_A, grad_x = numml_torch_cpp.spgemv_backward(df_dz, A_shape[0], A_shape[1], alpha,
                                                         A_data, A_col_ind, A_rowptr, x, beta, y)

        return (None, # A_shape
                torch.sum(df_dz * Ax), # alpha
                grad_A, # A_data
                None, # A_col_ind
                None, # A_rowptr
                grad_x, # x
                torch.sum(df_dz * y), # beta
                df_dz * beta) # y


class spgemm(torch.autograd.Function):
    '''
    General sparse matrix times sparse matrix product
    '''

    @staticmethod
    def forward(ctx,
                A_shape, A_data, A_indices, A_indptr,
                B_shape, B_data, B_indices, B_indptr):

        assert(A_shape[1] == B_shape[0])
        C_shape = (A_shape[0], B_shape[1])

        C_data, C_indices, C_indptr = numml_torch_cpp.spgemm_forward(A_shape[0], A_shape[1], A_data, A_indices, A_indptr,
                                                                     B_shape[0], B_shape[1], B_data, B_indices, B_indptr)

        ctx.save_for_backward(A_data, A_indices, A_indptr, B_data, B_indices, B_indptr, C_data, C_indices, C_indptr)
        ctx.A_shape = A_shape
        ctx.B_shape = B_shape
        ctx.C_shape = C_shape

        return (C_shape, C_data, C_indices, C_indptr)


    @staticmethod
    def backward(ctx, _grad_C_shape, grad_C_data, _grad_C_indices, _grad_C_indptr):
        A_data, A_indices, A_indptr, B_data, B_indices, B_indptr, C_data, C_indices, C_indptr = ctx.saved_tensors
        A_shape = ctx.A_shape
        B_shape = ctx.B_shape
        C_shape = ctx.C_shape

        grad_A, grad_B = numml_torch_cpp.spgemm_backward(grad_C_data, C_indices, C_indptr,
                                                         A_shape[0], A_shape[1], A_data, A_indices, A_indptr,
                                                         B_shape[0], B_shape[1], B_data, B_indices, B_indptr)

        return (None, # A_shape
                grad_A,
                None, # A_indices
                None, # A_indptr
                None, # B_shape
                grad_B,
                None, # B_indices
                None) # B_indptr


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
                elif col > row:
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
                elif col < row:
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

        ctx.save_for_backward(alpha, A_data, A_col_ind, A_rowptr, beta, B_data, B_col_ind, B_rowptr, C_data, C_col_ind, C_rowptr)
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

def diag(x, k=0):
    N_k = len(x)
    N = N_k + abs(k)
    rows = None
    cols = None

    if k == 0:
        rows = torch.arange(N)
        cols = torch.arange(N)
    elif k > 0:
        rows = torch.arange(N_k)
        cols = torch.arange(k, N)
    else:
        rows = torch.arange(abs(k), N)
        cols = torch.arange(N_k)

    return SparseCSRTensor((x, (rows, cols)), shape=(N, N))

def spouter(x, y):
    '''
    Returns the outer product of two vectors as a sparse matrix.
    Note that this is only beneficial if x and y are themselves reasonably sparse, otherwise
    it is likely better to use the regular dense outer product.

    Parameters
    ----------
    x : torch.Tensor
      First term in outer product
    y : torch.Tensor
      Second term in outer product

    Returns
    -------
    xyT : SparseCSRTensor
      The product x * y^T as a sparse tensor.
    '''

    # A_ij = x_i y_j
    x_nz = torch.nonzero(x).flatten()
    y_nz = torch.nonzero(y).flatten()

    row, col = torch.meshgrid(x_nz, y_nz, indexing='ij')
    row = row.flatten()
    col = col.flatten()

    val = x[row] * y[col]
    return SparseCSRTensor((val, (row, col)), shape=(len(x), len(y)))

class tril(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shape,
                A_data, A_col_ind, A_rowptr,
                k):
        L_data = []
        L_indices = []
        L_indptr = []

        L_to_A = []

        for row in range(shape[0]):
            L_indptr.append(len(L_data))

            for i in range(A_rowptr[row], A_rowptr[row + 1]):
                col = A_col_ind[i].item()
                if col > row + k:
                    break

                L_data.append(A_data[i])
                L_indices.append(A_col_ind[i])
                L_to_A.append(i)
        L_indptr.append(len(L_data))

        L_data = torch.Tensor(L_data)
        L_indices = torch.Tensor(L_indices).long()
        L_indptr = torch.Tensor(L_indptr).long()

        L_to_A = torch.Tensor(L_to_A).long()
        ctx.save_for_backward(A_col_ind, A_rowptr, L_to_A)
        ctx.shape = shape

        return L_data, L_indices, L_indptr

    @staticmethod
    def backward(ctx, grad_L_data, _grad_L_indices, _grad_L_indptr):
        A_col_ind, A_rowptr, L_to_A = ctx.saved_tensors
        shape = ctx.shape

        grad_A_data = torch.zeros_like(A_col_ind)
        for i in range(len(grad_L_data)):
            grad_A_data[L_to_A[i]] = grad_L_data[i]

        return (None, # shape
                grad_A_data, # A_data
                None, # A_col_ind
                None, # A_rowptr
                None) # k

class triu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shape,
                A_data, A_col_ind, A_rowptr,
                k):
        U_data = []
        U_indices = []
        U_indptr = []

        U_to_A = []

        for row in range(shape[0]):
            U_indptr.append(len(U_data))

            for i in range(A_rowptr[row], A_rowptr[row + 1]):
                col = A_col_ind[i].item()
                if col < row + k:
                    continue

                U_data.append(A_data[i])
                U_indices.append(A_col_ind[i])
                U_to_A.append(i)
        U_indptr.append(len(U_data))

        U_data = torch.Tensor(U_data)
        U_indices = torch.Tensor(U_indices).long()
        U_indptr = torch.Tensor(U_indptr).long()

        U_to_A = torch.Tensor(U_to_A).long()
        ctx.save_for_backward(A_col_ind, A_rowptr, U_to_A)
        ctx.shape = shape

        return U_data, U_indices, U_indptr

    @staticmethod
    def backward(ctx, grad_U_data, _grad_U_indices, _grad_U_indptr):
        A_col_ind, A_rowptr, U_to_A = ctx.saved_tensors
        shape = ctx.shape

        grad_A_data = torch.zeros_like(A_col_ind)
        for i in range(len(grad_U_data)):
            grad_A_data[U_to_A[i]] = grad_U_data[i]

        return (None, # shape
                grad_A_data, # A_data
                None, # A_col_ind
                None, # A_rowptr
                None) # k

class spcolscale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shape,
                A_data, A_col_ind, A_rowptr,
                row_start, column):
        B_data = torch.clone(A_data.detach())

        diag_entry = None
        for i in range(A_rowptr[row_start], A_rowptr[row_start + 1]):
            if A_col_ind[i] == row_start:
                diag_entry = A_data[i]
                break

        if diag_entry is None:
            raise RuntimeError('Matrix has no explicit diagonal entry!')
        if diag_entry == 0.:
            raise RuntimeError('Division by zero: matrix has explicit zero on diagonal.')

        B_data[torch.logical_and(A_col_ind == column, torch.arange(len(A_data)) >= A_rowptr[row_start+1])] /= diag_entry

        ctx.save_for_backward(A_data, A_col_ind, A_rowptr)
        ctx.row_start = row_start
        ctx.column = column
        ctx.shape = shape

        return B_data, A_col_ind, A_rowptr

    @staticmethod
    def backward(ctx, grad_B_data, _grad_B_col_ind, _grad_B_rowptr):
        A_data, A_col_ind, A_rowptr = ctx.saved_tensors
        row_start = ctx.row_start
        column = ctx.column
        shape = ctx.shape

        grad_A = torch.clone(grad_B_data)
        kk_entry = None
        for row in range(row_start, shape[0]):
            for i in range(A_rowptr[row], A_rowptr[row + 1]):
                col = A_col_ind[i]

                if col == column:
                    if row == row_start:
                        kk_entry = i
                    else:
                        grad_A[i] /= A_data[kk_entry]
                        grad_A[kk_entry] -= grad_B_data[i] * (A_data[i] / (A_data[kk_entry]) ** 2.)
                    break

        return (None,   # shape
                grad_A, # A_data
                None,   # A_col_ind
                None,   # A_rowptr
                None,   # row_start
                None)   # column

def splu(A):
    '''
    Sparse LU factorization *without* pivoting

    Parameters
    ----------
    A : SparseCSRTensor
      Sparse tensor on which to perform the LU factorization

    Returns
    -------
    M : SparseCSRTensor
      Sparse tensor M in which entries below the main diagonal correspond
      to 'L', while entries including and above the main diagonal correspond
      to 'U'
    '''

    # Helper to do the column scaling
    def apply_spcolscale(A, k):
        B_data, B_indices, B_indptr = spcolscale.apply(A.shape, A.data, A.indices, A.indptr, k, k)
        return SparseCSRTensor((B_data, B_indices, B_indptr), shape=A.shape)

    M = A.copy()
    for k in range(A.shape[0] - 1):
        M = apply_spcolscale(M, k)
        M_row = torch.zeros(A.shape[1])
        M_col = torch.zeros(A.shape[0])

        # Get row of M
        for i in range(M.indptr[k], M.indptr[k+1]):
            col = M.indices[i]
            if col > k:
                M_row[col] = M.data[i]

        # Get col of M
        for row in range(k+1, M.shape[0]):
            for i in range(M.indptr[row], M.indptr[row + 1]):
                col = M.indices[i]
                if col == k:
                    M_col[row] = M.data[i]
                    break

        M = M - spouter(M_col, M_row)
    return M


def splu_solve(LU, b):
    y = LU.solve_triangular(upper=False, unit=True, b=b)
    return LU.solve_triangular(upper=True, unit=False, b=y)

def spsolve(A, b):
    '''
    Solves a sparse system of linear equations for a vector right-hand-side.

    Parameters
    ----------
    A : SparseCSRTensor
      Sparse, square tensor encoding some system of equations.
    b : torch.Tensor
      Dense right-hand-side vector.

    Returns
    -------
    x : torch.Tensor
      The solution to A^{-1} b
    '''

    LU = splu(A)
    return splu_solve(LU, b)

class sptranspose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_shape, A_data, A_indices, A_indptr):
        At_data, At_indices, At_indptr, At_to_A_idx = \
            numml_torch_cpp.csr_transpose_forward(A_shape[0], A_shape[1], A_data, A_indices, A_indptr)

        ctx.save_for_backward(At_to_A_idx)
        return A_shape[::-1], At_data, At_indices, At_indptr

    @staticmethod
    def backward(ctx, _grad_A_shape, grad_At_data, _grad_At_indices, _grad_At_indptr):
        (At_to_A_idx,) = ctx.saved_tensors
        grad_A = numml_torch_cpp.csr_transpose_backward(grad_At_data, At_to_A_idx)

        return (None, # A_shape
                grad_A, # A_data
                None, # A_indices
                None) # A_indptr


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
        return spgemv.apply(self.shape,
                            torch.tensor(1.).to(x.device), self.data, self.indices, self.indptr, x,
                            torch.tensor(0.).to(x.device), torch.zeros(self.shape[0]).to(x.device))

    def solve_triangular(self, upper, unit, b):
        return sstrsv(upper, unit, self.shape, self.data, self.indices, self.indptr, b)

    def spgemm(self, othr):
        C_shape, C_data, C_indices, C_indptr = spgemm.apply(self.shape, self.data, self.indices, self.indptr,
                                                            othr.shape, othr.data, othr.indices, othr.indptr)
        return SparseCSRTensor((C_data, C_indices, C_indptr), C_shape)

    def __matmul__(self, x):
        dims = None
        if isinstance(x, torch.Tensor):
            dims = len(torch.squeeze(x).shape)
        elif isinstance(x, SparseCSRTensor):
            dims = 2
        else:
            raise RuntimeError(f'Unknown type for matmul: {type(x)}.')

        if dims == 1:
            return self.spmv(x)
        elif dims == 2:
            if isinstance(x, SparseCSRTensor):
                return self.spgemm(x)
            elif isinstance(x, torch.Tensor):
                raise RuntimeError('not implemented: sparse times dense matrix product')
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

    def __rtruediv__(self, other):
        if (isinstance(other, float) or
            isinstance(other, int) or
            (isinstance(other, torch.Tensor) and len(other.shape) == 0)):
            return SparseCSRTensor((other / self.data, self.indices, self.indptr), shape=self.shape)
        else:
            raise RuntimeError(f'Unknown type for right-division: {type(other)}.')

    def __neg__(self):
        return SparseCSRTensor((-self.data, self.indices, self.indptr), shape=self.shape)

    def __pow__(self, p):
        return SparseCSRTensor((self.data ** p, self.indices, self.indptr), shape=self.shape)

    def __repr__(self):
        grad_str = ''
        dev_str = ''
        if self.data.grad_fn is not None:
            grad_str = f', grad_fn=<{self.data.grad_fn.__class__.__name__}>'
        elif self.requires_grad:
            grad_str = ', requires_grad=True'
        if self.device.type != 'cpu':
            dev_str = f', device=\'{str(self.device)}\''

        return f"<{self.shape[0]}x{self.shape[1]} sparse matrix tensor of type '{self.data.dtype}'\n\twith {len(self.data)} stored elements in Compressed Sparse Row format{dev_str}{grad_str}>"

    def clone(self):
        return SparseCSRTensor((self.data, self.indices, self.indptr), shape=self.shape)

    def copy(self):
        return self.clone()

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

    @property
    def diagonal_idx(self):
        '''
        Returns the indices of the diagonal entries of the matrix such that
        idx[i] gives the diagonal index of the i'th row
        '''

        idx = torch.ones(min(self.shape[0], self.shape[1])).long() * -1

        for row in range(min(self.shape[0], self.shape[1])):
            for i in range(self.indptr[row], self.indptr[row+1]):
                col = self.indices[i].item()
                if row == col:
                    idx[row] = i

        return idx

    @property
    def row_indices(self):
        '''
        Returns the row number of each nonzero entry
        '''

        idx = torch.zeros(self.nnz).long()

        for row in range(self.shape[0]):
            for i in range(self.indptr[row], self.indptr[row+1]):
                idx[i] = row

        return idx

    def diagonal(self, k=0):
        max_diag = min(self.shape[0], self.shape[1])
        D = torch.zeros(max_diag - abs(k))

        for row in range(max_diag):
            for i in range(self.indptr[row], self.indptr[row+1]):
                col = self.indices[i].item()
                if row == col:
                    D[row] = self.data[i]

        return D

    def tril(self, k=0):
        L_data, L_indices, L_indptr = tril.apply(self.shape, self.data, self.indices, self.indptr, k)
        return SparseCSRTensor((L_data, L_indices, L_indptr), self.shape)

    def triu(self, k=0):
        U_data, U_indices, U_indptr = triu.apply(self.shape, self.data, self.indices, self.indptr, k)
        return SparseCSRTensor((U_data, U_indices, U_indptr), self.shape)

    def detach(self):
        return SparseCSRTensor((self.data.detach(), self.indices, self.indptr), self.shape)

    def transpose(self):
        At_shape, At_data, At_indices, At_indptr = sptranspose.apply(self.shape, self.data, self.indices, self.indptr)
        return SparseCSRTensor((At_data, At_indices, At_indptr), At_shape)

    @property
    def T(self):
        return self.transpose()

    @property
    def device(self):
        assert(self.data.device == self.indptr.device and self.indptr.device == self.indices.device)
        return self.data.device

    def to(self, device):
        return SparseCSRTensor((self.data.to(device), self.indices.to(device), self.indptr.to(device)), self.shape)

    def cpu(self):
        return self.to('cpu')
