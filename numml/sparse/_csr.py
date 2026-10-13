import torch
import numml_torch_cpp
import numpy as np
import numml.utils as utils
import importlib
import numml.sparse.linalg as spla
import numml.profiler

# Try to import scipy and cupy for conversion routines
class NotFoundModule:
    def __init__(self, modname):
        self.modname = modname
    def __getattr__(self, key):
        raise RuntimeError(f'Cannot get field {key}, {self.modname} is not installed or not available for import on this system.')


class LazyImportModule:
    def __init__(self, modname):
        self.modname = modname
        self.module = None

    def __getattr__(self, key):
        if self.module is None:
            self.module = importlib.import_module(self.modname)
        return getattr(self.module, key)


def try_import(modname):
    try:
        return importlib.import_module(modname)
    except ImportError:
        return NotFoundModule(modname)


def try_import_lazy(modname):
    return LazyImportModule(modname)


sci_sp = try_import('scipy.sparse')
cupy = try_import_lazy('cupy')
cupy_sp = try_import_lazy('cupyx.scipy.sparse')

# Profiling, for debug
class DummyProfiler():
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, type, value, tb):
        if type is not None:
            raise value
        return True
Profiler = DummyProfiler

def _debug_set_profiler_enabled(val):
    global Profiler
    if val:
        Profiler = numml.profiler.Profiler
    else:
        Profiler = DummyProfiler


def numpy_to_torch_dtype(np_type):
    if np_type == np.int32:
        return torch.int32
    elif np_type == np.int64:
        return torch.int64
    elif np_type == np.float32:
        return torch.float32
    elif np_type == np.float64:
        return torch.float64
    else:
        raise RuntimeError(f'Unknown numpy type for conversion: {np_type}.')


def coo_to_csr(values, row_ind, col_ind, shape, sort=True):
    '''
    Conversion from COO to CSR format
    '''

    if sort:
        _, col_sort = torch.sort(col_ind)

        values = values[col_sort]
        row_ind = row_ind[col_sort]
        col_ind = col_ind[col_sort]

        _, row_sort = torch.sort(row_ind, stable=True)
        values = values[row_sort]
        row_ind = row_ind[row_sort]
        col_ind = col_ind[row_sort]

    nnz = torch.bincount(row_ind, minlength=shape[0])
    cumsum = torch.cumsum(nnz, 0)
    cumsum = torch.cat((torch.tensor([0], device=values.device), cumsum))

    return values, col_ind, cumsum


def _empty_csr(rows, A_data, A_indices, A_indptr):
    '''
    CSR arrays for a matrix with the given number of rows and no stored entries,
    with data types and device matching those of A.

    The native kernels are not guaranteed to handle inputs without any entries (a CUDA
    kernel cannot be launched with zero blocks, for example), so these trivial cases are
    handled on the Python side before dispatching.
    '''

    return A_data.new_zeros(0), A_indices.new_zeros(0), A_indptr.new_zeros(rows + 1)


class spgemv(torch.autograd.Function):
    '''
    Sparse general matrix times vector product
    '''

    @staticmethod
    def forward(ctx, A_shape, A_data, A_col_ind, A_rowptr, x):
        if A_data.type() != x.type():
            raise RuntimeError(f'Matrix and vector should be same data type, got {A_data.type()} and {x.type()}, respectively.')

        if A_data.numel() == 0:
            Ax = x.new_zeros(A_shape[0])
        else:
            with Profiler('spgemv forward'):
                Ax = numml_torch_cpp.spgemv_forward(A_shape[0], A_shape[1], A_data, A_col_ind, A_rowptr, x)

        ctx.save_for_backward(Ax, x, A_data, A_col_ind, A_rowptr)
        ctx.shape = A_shape

        return Ax

    @staticmethod
    def backward(ctx, df_dz):
        Ax, x, A_data, A_col_ind, A_rowptr = ctx.saved_tensors
        A_shape = ctx.shape

        if A_data.numel() == 0:
            grad_A, grad_x = torch.zeros_like(A_data), torch.zeros_like(x)
        else:
            with Profiler('spgemv backward'):
                grad_A, grad_x = numml_torch_cpp.spgemv_backward(df_dz, A_shape[0], A_shape[1],
                                                                 A_data, A_col_ind, A_rowptr, x)

        return (None, # A_shape
                grad_A, # A_data
                None, # A_col_ind
                None, # A_rowptr
                grad_x) # x


class spgemm(torch.autograd.Function):
    '''
    General sparse matrix times sparse matrix product
    '''

    @staticmethod
    def forward(ctx,
                A_shape, A_data, A_indices, A_indptr,
                B_shape, B_data, B_indices, B_indptr):

        if A_data.type() != B_data.type():
            raise RuntimeError(f'Matrices should be same data type, got {A_data.type()} and {B_data.type()}, respectively.')
        if (A_shape[1] != B_shape[0]):
            raise RuntimeError(f'Incompatible matrix shapes for multiplication.  Got {A_shape} and {B_shape}.')

        C_shape = (A_shape[0], B_shape[1])

        if A_data.numel() == 0 or B_data.numel() == 0:
            C_data, C_indices, C_indptr = _empty_csr(A_shape[0], A_data, A_indices, A_indptr)
        else:
            with Profiler('spgemm forward'):
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

        if A_data.numel() == 0 or B_data.numel() == 0:
            grad_A, grad_B = torch.zeros_like(A_data), torch.zeros_like(B_data)
        else:
            with Profiler('spgemm backward'):
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


class splincomb(torch.autograd.Function):
    '''
    Computes the linear combination of two sparse matrices like
    C = \\alpha A + \\beta B.
    '''

    @staticmethod
    def forward(ctx, shape,
                alpha, A_data, A_col_ind, A_rowptr,
                beta,  B_data, B_col_ind, B_rowptr):

        if A_data.type() != B_data.type():
            raise RuntimeError(f'Matrices should be same data type, got {A_data.type()} and {B_data.type()}, respectively.')

        if A_data.numel() == 0 and B_data.numel() == 0:
            C_data, C_col_ind, C_rowptr = _empty_csr(shape[0], A_data, A_col_ind, A_rowptr)
        else:
            with Profiler('splincomb forward'):
                C_data, C_col_ind, C_rowptr = numml_torch_cpp.splincomb_forward(shape[0], shape[1],
                                                                                alpha, A_data, A_col_ind, A_rowptr,
                                                                                beta,  B_data, B_col_ind, B_rowptr)

        ctx.save_for_backward(alpha, A_data, A_col_ind, A_rowptr, beta, B_data, B_col_ind, B_rowptr, C_data, C_col_ind, C_rowptr)
        ctx.shape = shape

        return (C_data, C_col_ind, C_rowptr)


    @staticmethod
    def backward(ctx, grad_C_data, _grad_C_col_ind, _grad_C_rowptr):
        alpha, A_data, A_col_ind, A_rowptr, beta, B_data, B_col_ind, B_rowptr, C_data, C_col_ind, C_rowptr = ctx.saved_tensors
        shape = ctx.shape

        if A_data.numel() == 0 and B_data.numel() == 0:
            grad_A, grad_B = torch.zeros_like(A_data), torch.zeros_like(B_data)
        else:
            with Profiler('splincomb backward'):
                grad_A, grad_B = numml_torch_cpp.splincomb_backward(shape[0], shape[1],
                                                                    alpha, A_data, A_col_ind, A_rowptr,
                                                                    beta,  B_data, B_col_ind, B_rowptr,
                                                                    grad_C_data, C_col_ind, C_rowptr)

        return (None, # shape
                torch.sum(A_data), # alpha
                grad_A, # A_data
                None, # A_col_ind
                None, # A_rowptr
                torch.sum(B_data), # beta
                grad_B, # B_data
                None, # B_col_ind
                None) # B_rowptr


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

        L_data = torch.Tensor(L_data).to(A_data.device)
        L_indices = torch.Tensor(L_indices).long().to(A_data.device)
        L_indptr = torch.Tensor(L_indptr).long().to(A_data.device)

        L_to_A = torch.Tensor(L_to_A).long().to(A_data.device)
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

        U_data = torch.Tensor(U_data).to(A_data.device)
        U_indices = torch.Tensor(U_indices).long().to(A_data.device)
        U_indptr = torch.Tensor(U_indptr).long().to(A_data.device)

        U_to_A = torch.Tensor(U_to_A).long().to(A_data.device)
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


class sptranspose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_shape, A_data, A_indices, A_indptr):

        if A_data.numel() == 0:
            At_data, At_indices, At_indptr = _empty_csr(A_shape[1], A_data, A_indices, A_indptr)
            At_to_A_idx = A_indices.new_zeros(0)
        else:
            with Profiler('sptranspose forward'):
                At_data, At_indices, At_indptr, At_to_A_idx = \
                    numml_torch_cpp.csr_transpose_forward(A_shape[0], A_shape[1], A_data, A_indices, A_indptr)

        ctx.save_for_backward(At_to_A_idx)
        return A_shape[::-1], At_data, At_indices, At_indptr

    @staticmethod
    def backward(ctx, _grad_A_shape, grad_At_data, _grad_At_indices, _grad_At_indptr):
        (At_to_A_idx,) = ctx.saved_tensors

        if At_to_A_idx.numel() == 0:
            grad_A = torch.zeros_like(grad_At_data)
        else:
            with Profiler('sptranspose backward'):
                grad_A = numml_torch_cpp.csr_transpose_backward(grad_At_data, At_to_A_idx)

        return (None, # A_shape
                grad_A, # A_data
                None, # A_indices
                None) # A_indptr


class spdmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_shape, A_data, A_indices, A_indptr, B):
        ctx.save_for_backward(A_data, A_indices, A_indptr, B)
        ctx.A_shape = A_shape

        if A_data.type() != B.type():
            raise RuntimeError(f'Matrices should be same data type, got {A_data.type()} and {B.type()}, respectively.')
        if (A_shape[1] != B.shape[0]):
            raise RuntimeError(f'Incompatible matrix shapes for multiplication.  Got {A_shape} and {B.shape}.')

        if A_data.numel() == 0 or B.numel() == 0:
            C = B.new_zeros((A_shape[0], B.shape[1]))
        else:
            with Profiler('spdmm forward'):
                C = numml_torch_cpp.spdmm_forward(A_shape[0], A_shape[1],
                                                  A_data, A_indices, A_indptr, B)
        return C

    @staticmethod
    def backward(ctx, grad_C):
        A_data, A_indices, A_indptr, B = ctx.saved_tensors
        A_shape = ctx.A_shape

        if A_data.numel() == 0 or B.numel() == 0:
            grad_A, grad_B = torch.zeros_like(A_data), torch.zeros_like(B)
        else:
            with Profiler('spdmm backward'):
                grad_A, grad_B = numml_torch_cpp.spdmm_backward(A_shape[0], A_shape[1],
                                                                A_data, A_indices, A_indptr,
                                                                B, grad_C)

        return (None, # A_shape
                grad_A, # A_data
                None, # A_indices
                None, # A_indptr
                grad_B) # B


class sptodense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_shape, A_data, A_indices, A_indptr):
        ctx.save_for_backward(A_data, A_indices, A_indptr)
        ctx.A_shape = A_shape

        if A_data.numel() == 0:
            A_d = A_data.new_zeros((A_shape[0], A_shape[1]))
        else:
            with Profiler('to dense forward'):
                A_d = numml_torch_cpp.csr_to_dense_forward(A_shape[0], A_shape[1],
                                                           A_data, A_indices, A_indptr)
        return A_d

    @staticmethod
    def backward(ctx, grad_A_dense):
        A_data, A_indices, A_indptr = ctx.saved_tensors
        A_shape = ctx.A_shape

        if A_data.numel() == 0:
            grad_A_data = torch.zeros_like(A_data)
        else:
            with Profiler('to dense backward'):
                grad_A_data = numml_torch_cpp.csr_to_dense_backward(grad_A_dense, A_shape[0], A_shape[1],
                                                                    A_data, A_indices, A_indptr)

        return (None,        # A_shape
                grad_A_data, # A_data
                None,        # A_indices
                None)        # A_indptr


class sprowsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_shape, A_data, A_indices, A_indptr):
        ctx.save_for_backward(A_data, A_indices, A_indptr)
        ctx.A_shape = A_shape

        if A_data.numel() == 0:
            x = A_data.new_zeros(A_shape[0])
        else:
            with Profiler('row sum forward'):
                x = numml_torch_cpp.csr_row_sum_forward(A_shape[0], A_shape[1],
                                                        A_data, A_indices, A_indptr)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        A_data, A_indices, A_indptr = ctx.saved_tensors
        A_shape = ctx.A_shape

        if A_data.numel() == 0:
            grad_A_data = torch.zeros_like(A_data)
        else:
            with Profiler('row sum backward'):
                grad_A_data = numml_torch_cpp.csr_row_sum_backward(grad_x, A_shape[0], A_shape[1],
                                                                   A_data, A_indices, A_indptr)

        return (None,        # A_shape
                grad_A_data, # A_data
                None,        # A_indices
                None)        # A_indptr

class spdiag(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_shape, A_data, A_indices, A_indptr):
        ctx.save_for_backward(A_data, A_indices, A_indptr)
        ctx.A_shape = A_shape

        if A_data.numel() == 0:
            x = A_data.new_zeros(min(A_shape[0], A_shape[1]))
        else:
            with Profiler('extract diagonal forward'):
                x = numml_torch_cpp.csr_extract_diagonal_forward(A_shape[0], A_shape[1],
                                                                 A_data, A_indices, A_indptr)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        A_data, A_indices, A_indptr = ctx.saved_tensors
        A_shape = ctx.A_shape

        if A_data.numel() == 0:
            grad_A_data = torch.zeros_like(A_data)
        else:
            with Profiler('extract diagonal backward'):
                grad_A_data = numml_torch_cpp.csr_extract_diagonal_backward(grad_x, A_shape[0], A_shape[1],
                                                                            A_data, A_indices, A_indptr)

        return (None,        # A_shape
                grad_A_data, # A_data
                None,        # A_indices
                None)        # A_indptr


def eye(N, k=0, dtype=torch.float32, device='cpu'):
    if abs(k) >= N:
        raise RuntimeError(f'Specified bandwidth for eye() is larger than matrix size.  Got k={abs(k)}, N={N}.')

    N_k = N - abs(k)
    rows = None
    cols = None
    vals = torch.ones(N_k, dtype=dtype, device=device)

    if k == 0:
        rows = torch.arange(N, device=device)
        cols = torch.arange(N, device=device)
    elif k > 0:
        rows = torch.arange(N_k, device=device)
        cols = torch.arange(k, N, device=device)
    else:
        rows = torch.arange(abs(k), N, device=device)
        cols = torch.arange(N_k, device=device)

    return SparseCSRTensor((vals, (rows, cols)), shape=(N, N), dtype=dtype, device=device)


def diag(x, k=0):
    N_k = len(x)
    N = N_k + abs(k)
    rows = None
    cols = None

    device = x.device
    dtype = x.dtype

    if k == 0:
        rows = torch.arange(N, device=device)
        cols = torch.arange(N, device=device)
    elif k > 0:
        rows = torch.arange(N_k, device=device)
        cols = torch.arange(k, N, device=device)
    else:
        rows = torch.arange(abs(k), N, device=device)
        cols = torch.arange(N_k, device=device)

    return SparseCSRTensor((x, (rows, cols)), shape=(N, N), dtype=x.dtype, device=device)


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

    x_nz = torch.nonzero(x).flatten()
    y_nz = torch.nonzero(y).flatten()

    row, col = torch.meshgrid(x_nz, y_nz, indexing='ij')
    row = row.flatten()
    col = col.flatten()

    val = x[row] * y[col]
    return SparseCSRTensor((val, (row, col)), shape=(len(x), len(y)))


def unpack_csr(A):
    return A.data, A.indices, A.indptr, A.shape


def repack_csr(A_data, A_indices, A_indptr, A_shape):
    return SparseCSRTensor((A_data, A_indices, A_indptr), A_shape, clone=False)


def _normalize_index(idx, n, device):
    '''
    Converts a single-axis index into a flat tensor of nonnegative entries.

    Parameters
    ----------
    idx : int, slice, range, list, np.ndarray, or torch.Tensor
      Index along one axis.  Tensors/arrays may be integer-valued or a boolean mask.
    n : int
      Length of the axis being indexed
    device : torch.device
      Device to place the output on

    Returns
    -------
    index : torch.Tensor or None
      Long tensor of indices in [0, n), or None if the full axis is selected.
    is_scalar : bool
      If the index was a single integer
    '''

    if idx is None or idx is Ellipsis:
        raise IndexError('None and Ellipsis are not supported for sparse tensor indexing.')

    if isinstance(idx, slice):
        if idx == slice(None):
            return None, False
        return torch.arange(*idx.indices(n), device=device), False

    if isinstance(idx, range):
        idx = torch.arange(idx.start, idx.stop, idx.step)

    is_scalar = False
    if isinstance(idx, (bool, np.bool_)):
        raise IndexError('Boolean scalars are not supported for sparse tensor indexing.')
    elif isinstance(idx, (int, np.integer)):
        is_scalar = True
        idx = torch.tensor([int(idx)])
    else:
        idx = torch.as_tensor(idx)
        if idx.ndim == 0 and idx.dtype != torch.bool:
            is_scalar = True
            idx = idx.reshape(1)

    idx = idx.to(device)
    if idx.ndim != 1:
        raise IndexError(f'Sparse tensor indices must be one-dimensional along each axis, got {idx.ndim} dimensions.')

    if idx.dtype == torch.bool:
        if len(idx) != n:
            raise IndexError(f'Boolean mask of length {len(idx)} does not match axis of length {n}.')
        return torch.nonzero(idx).flatten(), False

    if len(idx) == 0:
        return torch.empty(0, dtype=torch.long, device=device), False
    if idx.is_floating_point() or idx.is_complex():
        raise IndexError(f'Sparse tensor indices must be integers or booleans, got {idx.dtype}.')

    idx = idx.long()
    idx = torch.where(idx < 0, idx + n, idx)
    if torch.any((idx < 0) | (idx >= n)):
        raise IndexError(f'Index out of bounds for axis of length {n}.')

    return idx, is_scalar


def _csr_select_rows(A_data, A_indices, A_indptr, rows):
    '''
    Extracts the given rows (in order, with repetition) from a CSR matrix.
    '''

    counts = A_indptr[rows + 1] - A_indptr[rows]
    indptr = torch.cat((torch.zeros(1, dtype=torch.long, device=rows.device), torch.cumsum(counts, 0)))

    # Index into A of each output entry
    src = (torch.repeat_interleave(A_indptr[rows] - indptr[:-1], counts) +
           torch.arange(indptr[-1].item(), device=rows.device))

    return A_data[src], A_indices[src], indptr


def _csr_select_cols(A_data, A_indices, A_indptr, A_cols, cols):
    '''
    Extracts the given columns (in order, with repetition) from a CSR matrix.
    '''

    device = cols.device
    rows = len(A_indptr) - 1
    nnz = len(A_indices)

    # Entries of cols_sorted in [offset[j], offset[j] + count[j]) are requests for
    # column j of A, which get placed in columns cols_perm[...] of the output.
    _, cols_perm = torch.sort(cols, stable=True)
    count = torch.bincount(cols, minlength=A_cols)
    offset = torch.cumsum(count, 0) - count

    # Each entry of A is repeated as many times as its column is requested
    row_ind = torch.repeat_interleave(torch.arange(rows, device=device), A_indptr[1:] - A_indptr[:-1])
    rep = count[A_indices]
    src = torch.repeat_interleave(torch.arange(nnz, device=device), rep)
    k = torch.arange(len(src), device=device) - torch.repeat_interleave(torch.cumsum(rep, 0) - rep, rep)

    out_cols = cols_perm[offset[A_indices[src]] + k]
    out_rows = row_ind[src]

    if len(cols) > 1 and not torch.all(cols[1:] > cols[:-1]):
        # Columns were permuted or repeated, so need to re-sort each row
        order = torch.argsort(out_rows * len(cols) + out_cols)
        src = src[order]
        out_cols = out_cols[order]
        out_rows = out_rows[order]

    nnz_per_row = torch.bincount(out_rows, minlength=rows)
    indptr = torch.cat((torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(nnz_per_row, 0)))

    return A_data[src], out_cols, indptr


class SparseCSRTensor(object):
    def __init__(self, arg1, shape=None, clone=True, dtype=None, device=None):
        '''
        Compressed Sparse Row matrix (tensor) with gradient support on
        nonzero entries.

        This constructor mimics that in SciPy's CSR matrix class, and
        can be constructed in several ways:

            csr_matrix(D)
                with a dense tensor

            csr_matrix(S)
                with a sparse torch COO tensor or any SciPy sparse matrix

            csr_matrix((data, (row_ind, col_ind)), shape=(M, N))
                where data, row_ind, col_ind define the entries for a COO
                representation

            csr_matrix((data, indices, indptr), shape=(M, N))
                constructing in the standard CSR representation
        '''

        if isinstance(arg1, torch.Tensor):
            if arg1.layout == torch.sparse_coo:
                # Input is torch sparse COO

                arg1 = arg1.coalesce()
                vals = arg1.values()
                rows, cols = arg1.indices()
                self.data, self.indices, self.indptr = coo_to_csr(vals, rows, cols, arg1.shape, sort=True)
            else:
                # Input is torch dense

                rows, cols = torch.nonzero(arg1).T
                nz = arg1[rows, cols]
                self.data, self.indices, self.indptr = coo_to_csr(nz, rows, cols, arg1.shape, sort=True)

            if shape is not None:
                self.shape = shape
            else:
                self.shape = arg1.shape
        elif (
                isinstance(arg1, sci_sp.spmatrix) or
                isinstance(arg1, sci_sp.sparray)
             ):
            arg_csr = arg1.tocsr()
            if dtype is None:
                dtype = numpy_to_torch_dtype(arg_csr.data.dtype)
            if device is None:
                device = torch.device('cpu')

            self.data = torch.Tensor(arg_csr.data.copy()).type(dtype).to(device)
            self.indices = torch.Tensor(arg_csr.indices.copy()).long().to(device)
            self.indptr = torch.Tensor(arg_csr.indptr.copy()).long().to(device)
            self.shape = arg_csr.shape
        elif isinstance(arg1, tuple):
            if len(arg1) == 3:
                # Input is CSR: (data, indices, indptr)
                assert(shape is not None)

                if isinstance(arg1[0], np.ndarray):
                    if dtype is None:
                        dtype = numpy_to_torch_dtype(arg_csr.data.dtype)
                    if device is None:
                        device = torch.device('cpu')

                    self.data = torch.Tensor(arg1[0].copy()).type(dtype).to(device)
                elif isinstance(arg1[1], torch.Tensor):
                    if dtype is None:
                        dtype = arg1[0].dtype
                    if device is None:
                        device = arg1[0].device

                    if clone:
                        self.data = torch.clone(arg1[0]).type(dtype).to(device)
                    else:
                        self.data = arg1[0]

                if isinstance(arg1[1], np.ndarray):
                    self.indices = torch.Tensor(arg1[1].copy()).long().to(device)
                elif isinstance(arg1[1], torch.Tensor):
                    if clone:
                        self.indices = torch.clone(arg1[1]).long().to(device)
                    else:
                        self.indices = arg1[1]

                if isinstance(arg1[2], np.ndarray):
                    self.indptr = torch.Tensor(arg1[2].copy()).long().to(device)
                elif isinstance(arg1[2], torch.Tensor):
                    if clone:
                        self.indptr = torch.clone(arg1[2]).long().to(device)
                    else:
                        self.indptr = arg1[2]

                self.shape = shape
            elif len(arg1) == 2:
                # Input is COO: (data, (row_ind, col_ind))
                assert(shape is not None)

                data, (rows, cols) = arg1
                self.data, self.indices, self.indptr = coo_to_csr(data, rows, cols, shape)
                self.shape = shape
        else:
            raise RuntimeError(f'Unknown type given as argument of SparseCSRTensor: {type(arg1)}')

    def __getitem__(self, key):
        '''
        Extract a submatrix by row and/or column, like A[rows, cols].

        Each of rows and cols may be an integer, a slice, a list/array/tensor of
        integer indices, or a boolean mask.  Index arrays may be unsorted and may contain
        duplicates.  Gradients are propagated to the nonzero entries.

        Note that unlike NumPy/SciPy, indexing with two arrays selects the submatrix
        formed by the given rows and columns, rather than the individual entries
        (rows[0], cols[0]), (rows[1], cols[1]), ...

        Returns
        -------
        sub : SparseCSRTensor or torch.Tensor
          The (len(rows), len(cols)) submatrix.  If both rows and cols are integers, then
          the single entry is returned as a scalar-valued Torch tensor.
        '''

        if not isinstance(key, tuple):
            key = (key,)
        if len(key) == 1:
            key = (key[0], slice(None))
        elif len(key) != 2:
            raise IndexError(f'Sparse tensor takes at most 2 indices, got {len(key)}.')

        rows, row_scalar = _normalize_index(key[0], self.shape[0], self.device)
        cols, col_scalar = _normalize_index(key[1], self.shape[1], self.device)

        if rows is None and cols is None:
            return self.clone()

        data, indices, indptr = self.data, self.indices, self.indptr
        shape = (self.shape[0], self.shape[1])

        if rows is not None:
            data, indices, indptr = _csr_select_rows(data, indices, indptr, rows)
            shape = (len(rows), shape[1])
        if cols is not None:
            data, indices, indptr = _csr_select_cols(data, indices, indptr, shape[1], cols)
            shape = (shape[0], len(cols))

        if row_scalar and col_scalar:
            return data.sum()

        return SparseCSRTensor((data, indices, indptr), shape, clone=False)

    def spmv(self, x):
        if self.shape[1] != torch.numel(x):
            raise RuntimeError(f'Number of columns of tensor must equal number of rows of vector.  Got shapes: {self.shape} and {x.shape}')

        y = spgemv.apply(self.shape, self.data, self.indices, self.indptr, x.squeeze())
        y = utils.unsqueeze_like(y, x)
        return y


    def solve_triangular(self, upper, unit, b):
        '''
        Solve this (triangular) system for some right-hand-side vector.

        Parameters
        ----------
        upper : bool
          Is this matrix upper triangular?
        unit : bool
          Assume unit diagonal?  If true, will ignore diagonal entries.
        b : torch.Tensor
          Right-hand-side vector

        Returns
        -------
        x : torch.Tensor
          Solution to the matrix equation Ax = b, where A is triangular.
        '''

        return spla.sptrsv.apply(self.shape, self.data, self.indices, self.indptr, not upper, unit, b)

    def spspmm(self, othr):
        C_shape, C_data, C_indices, C_indptr = spgemm.apply(self.shape, self.data, self.indices, self.indptr,
                                                            othr.shape, othr.data, othr.indices, othr.indptr)
        return SparseCSRTensor((C_data, C_indices, C_indptr), C_shape)

    def spdmm(self, othr):
        return spdmm.apply(self.shape, self.data, self.indices, self.indptr, othr)

    def __rmatmul__(self, x):
        # Do this in the slowest way possible computationally
        return (self.transpose() @ x.T).T

    def __matmul__(self, x):
        dims = None
        if isinstance(x, torch.Tensor):
            dims = x.ndim
        elif isinstance(x, SparseCSRTensor):
            dims = 2
        else:
            raise RuntimeError(f'Unknown type for matmul: {type(x)}.')

        if dims == 0:
            # Multiplication by a scalar, special case depending on what our shape is
            if self.shape[0] == 1 and self.shape[1] == 1:
                # 1x1 matrix times scalar -> scalar
                return self.data * x.item()
            elif self.shape[0] == 1 or self.shape[1] == 1:
                # 1xn or nx1 times scalar -> dense vector
                return self.to_dense() * x.item()
            else:
                # mxn times scalar -> sparse matrix
                return self.__mul__(x.item())
        elif dims == 1:
            return self.spmv(x)
        elif dims == 2:
            if isinstance(x, SparseCSRTensor):
                return self.spspmm(x)
            elif isinstance(x, torch.Tensor):
                return self.spdmm(x)
        else:
            if isinstance(x, torch.Tensor) and x.shape[-2] == self.shape[1]:
                # If we're multiplying by a 3d or more complex shape, assume this is a batched SPDMM
                # x should have shape (a_1, a_2, ..., a_k, n, b), where a_1 ... a_k and b are some arbitrary dimensions
                ndim = x.ndim

                # First, transpose to shape (n, a_1, a_2, ..., a_k, b) then flatten to (n, a_1 * a_2 * ... a_k * b)
                x_tr = torch.transpose(x, 0, ndim-2)
                x_fl = torch.flatten(x_tr, 1)

                # Now, batch multiply
                y_fl = self.spdmm(x_fl)

                # Reshape and transpose back to correct shape
                new_shape = (self.shape[0],) + tuple(x_tr.shape[1:])
                y_tr = y_fl.reshape(new_shape)
                return torch.transpose(y_tr, 0, ndim-2)
            elif torch.any(torch.tensor(x.shape) == 1):
                # If the shapes don't line up for batch multiplication, try to squeeze and try again.
                return self.__matmul__(torch.squeeze(x))
            else:
                # Otherwise, error out.
                raise RuntimeError(f'invalid tensor found for sparse multiply: mode {dims} tensor found.  Trying to multiply {self.shape} sparse matrix by {x.shape} tensor.')

    def to_dense(self):
        '''
        Converts the CSR representation to a dense Torch tensor.
        Will propagate gradients to/from nonzero entries if they exist.

        Returns
        -------
        dense : torch.Tensor
          Dense output
        '''

        return sptodense.apply(self.shape, self.data, self.indices, self.indptr)

    def to_coo(self):
        '''
        Converts the CSR reprsentation to Torch's built-in sparse COO tensor format.
        Will propagate gradients on the data if they exist.

        Returns
        -------
        coo : torch.Tensor
          Sparse COO output
        '''

        # take advantage of the fact that sorted COO and CSR have entries in the same order
        row_i = torch.empty(self.nnz, dtype=torch.long)
        for i in range(self.shape[0]):
            row_i[self.indptr[i]:self.indptr[i+1]] = i
        row_i = row_i.to(self.device)
        return torch.sparse_coo_tensor(torch.row_stack((row_i, self.indices)), self.data, self.shape, device=self.device)

    def tocoo(self):
        '''
        Converts the Torch CSR representation to SciPy sparse COO.
        Gradient information on the data will be lost, if it exists.

        Returns
        -------
        csr : scipy.sparse.csr_matrix
          SciPy sparse CSR output
        '''
        # take advantage of the fact that sorted COO and CSR have entries in the same order
        row_i = np.empty(self.nnz, dtype=np.int32)
        for i in range(self.shape[0]):
            row_i[self.indptr[i]:self.indptr[i+1]] = i
        row_i = row_i
        col_i = self.indices.numpy().astype(np.int32)
        return sci_sp.coo_matrix((self.data.detach().numpy(), (row_i, col_i)), shape=self.shape)


    def to_scipy_csr(self):
        '''
        Converts the Torch CSR representation to SciPy sparse CSR.
        Gradient information on the data will be lost, if it exists.

        Returns
        -------
        csr : scipy.sparse.csr_matrix
          SciPy sparse CSR output
        '''

        return sci_sp.csr_matrix((self.data.cpu().detach().numpy(),
                                  self.indices.cpu().numpy(), self.indptr.cpu().numpy()), self.shape)

    def tocsr(self):
        '''
        Alias for to_scipy_csr()
        '''
        return self.to_scipy_csr()

    def to_cupy_csr(self, device=None):
        '''
        Converts the Torch CSR representatino to a CuPy sparse GPU CSR.
        Gradient information on the data will be lost, if it exists.
        If this matrix is not on the GPU, then it will be sent to the device
        that is passed as an argument, otherwise it will assume cuda:0.

        Parameters
        ----------
        device : torch.Device, optional
          CUDA device this should be moved to if it isn't already on the GPU

        Returns
        -------
        A_cupy : cupyx.scipy.sparse.csr_matrix
          Sparse CuPy CSR matrix.
        '''

        if self.device.type == 'cuda':
            data_cupy = cupy.asarray(self.data.detach())
            indices_cupy = cupy.asarray(self.indices.detach())
            indptr_cupy = cupy.asarray(self.indptr.detach())
            return cupy_sp.csr_matrix((data_cupy, indices_cupy, indptr_cupy), self.shape)
        else:
            gpu = torch.device('cuda:0')
            data_cupy = cupy.asarray(self.data.detach().to(gpu))
            indices_cupy = cupy.asarray(self.indices.detach().to(gpu))
            indptr_cupy = cupy.asarray(self.indptr.detach().to(gpu))
            return cupy_sp.csr_matrix((data_cupy, indices_cupy, indptr_cupy), self.shape)

    def sum(self):
        '''
        Computes the sum over all entries.

        Returns
        -------
        sum : float
          Summation over nonzero data entries
        '''

        return self.data.sum()

    def row_sum(self):
        '''
        Computes the sum of entries per row.

        Returns
        row_sum : torch.Tensor
          Tensor such that row_sum[i] is the sum of entries in row i
        '''

        return sprowsum.apply(self.shape, self.data, self.indices, self.indptr)

    def abs(self):
        '''
        Takes the absolute value of each entry, and returns the output as a new tensor.

        Returns
        -------
        abs : SparseCSRTensor
          Tensor whose entries have been absolute value-d
        '''

        return SparseCSRTensor((torch.abs(self.data), self.indices, self.indptr), self.shape)

    def __abs__(self):
        return self.abs()

    def _isscalar(x):
        return (isinstance(x, float) or
                isinstance(x, int) or
                (isinstance(x, torch.Tensor) and len(x.shape) == 0))

    def __add__(self, othr):
        if SparseCSRTensor._isscalar(othr):
            return SparseCSRTensor((self.data + othr, self.indices, self.indptr), shape=self.shape)
        elif isinstance(othr, SparseCSRTensor):
            if self.shape != othr.shape:
                raise RuntimeError(f'Incompatible shapes for entrywise addition, got {self.shape} and {othr.shape}, respectively.')

            C = splincomb.apply(self.shape,
                                torch.tensor(1.), self.data, self.indices, self.indptr,
                                torch.tensor(1.), othr.data, othr.indices, othr.indptr)
            return SparseCSRTensor(C, self.shape)
        else:
            raise RuntimeError(f'Unknown type for sparse tensor addition: {type(othr)}.')

    def __sub__(self, othr):
        if SparseCSRTensor._isscalar(othr):
            return SparseCSRTensor((self.data + othr, self.indices, self.indptr), shape=self.shape)
        elif isinstance(othr, SparseCSRTensor):
            if self.shape != othr.shape:
                raise RuntimeError(f'Incompatible shapes for entrywise subtraction, got {self.shape} and {othr.shape}, respectively.')

            C = splincomb.apply(self.shape,
                                torch.tensor(1.) , self.data, self.indices, self.indptr,
                                torch.tensor(-1.), othr.data, othr.indices, othr.indptr)
            return SparseCSRTensor(C, self.shape)
        else:
            raise RuntimeError(f'Unknown type for sparse tensor addition: {type(othr)}.')

    def __mul__(self, other):
        if SparseCSRTensor._isscalar(other):
            return SparseCSRTensor((self.data * other, self.indices, self.indptr), shape=self.shape)
        elif isinstance(other, torch.Tensor) and len(torch.squeeze(other).shape) == 2:
            raise RuntimeError('Element-wise hadamard product not implemented')
        else:
            raise RuntimeError(f'Unknown type for multiplication: {type(other)}.')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if SparseCSRTensor._isscalar(other):
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
        '''
        Create a copy of this tensor.  Mutations made to the output tensor will not affect
        the original tensor.

        Returns
        -------
        cloned : SparseCSRTensor
          Copied tensor.
        '''

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
        Number of stored values, including explicit zeros if they exist.
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

    def diagonal(self):
        '''
        Returns the diagonal entries of the matrix expressed as a flattened min{m, n} vector,
        given that the matrix has dimensions m x n.

        Returns
        -------
        diag : torch.Tensor
          Diagonal entries
        '''

        return spdiag.apply(self.shape, self.data, self.indices, self.indptr)

    def trace(self):
        '''
        Returns the trace of the matrix, i.e., sum of the diagonal entries.

        Returns
        -------
        trace : torch.Tensor
          Matrix trace as a scalar-valued Torch tensor.
        '''

        return self.diagonal().sum()

    def conjugate(self):
        '''
        Element-wise complex conjugation
        Returns
        -------
        A_conjugate : SparseCSRTensor
          Matrix whose entries have complex conjugation applied
        '''

        return SparseCSRTensor((self.data.conj(), self.indices, self.indptr), self.shape)

    def conj(self):
        '''
        Element-wise complex conjugation
        Returns
        -------
        A_conjugate : SparseCSRTensor
          Matrix whose entries have complex conjugation applied
        '''

        return self.conjugate()

    def tril(self, k=0):
        L_data, L_indices, L_indptr = tril.apply(self.shape, self.data, self.indices, self.indptr, k)
        return SparseCSRTensor((L_data, L_indices, L_indptr), self.shape)

    def triu(self, k=0):
        U_data, U_indices, U_indptr = triu.apply(self.shape, self.data, self.indices, self.indptr, k)
        return SparseCSRTensor((U_data, U_indices, U_indptr), self.shape)

    def detach(self):
        return SparseCSRTensor((self.data.detach(), self.indices, self.indptr), self.shape)

    def transpose(self):
        '''
        Take the transpose of this tensor.

        Note: the transposition is done eagerly.  There is no intermediate CSC format that is returned.

        Returns
        -------
        At : SparseCSRTensor
          Transposed version of the tensor.
        '''

        At_shape, At_data, At_indices, At_indptr = sptranspose.apply(self.shape, self.data, self.indices, self.indptr)
        return SparseCSRTensor((At_data, At_indices, At_indptr), At_shape)

    def hermitian(self):
        '''
        Take the Hermitian transpose of this tensor.
        Returns
        Ah : SparseCSRTensor
          Transposed, conjugated version of the tensor.
        '''

        At_shape, At_data, At_indices, At_indptr = sptranspose.apply(self.shape, self.data.conj(), self.indices, self.indptr)
        return SparseCSRTensor((At_data, At_indices, At_indptr), At_shape)

    @property
    def T(self):
        return self.transpose()

    @property
    def H(self):
        return self.hermitian()

    @property
    def device(self):
        assert(self.data.device == self.indptr.device and self.indptr.device == self.indices.device)
        return self.data.device

    def to(self, device):
        return SparseCSRTensor((self.data.to(device), self.indices.to(device), self.indptr.to(device)), self.shape)

    def cpu(self):
        return self.to('cpu')

    def float(self):
        return SparseCSRTensor((self.data.float(), self.indices, self.indptr), self.shape)

    def double(self):
        return SparseCSRTensor((self.data.double(), self.indices, self.indptr), self.shape)
