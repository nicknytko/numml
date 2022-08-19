import torch
import torch.autograd
import numml_torch_cpp
import numpy as np

class torchSparseCOOToNummlCSR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coo_tensor):
        row_ind, col_ind = coo_tensor.indices()
        data = coo_tensor.values()
        dtype = data.dtype
        N = len(row_ind)

        argsort = torch.Tensor(np.lexsort((col_ind, row_ind))).long()
        argsort_inv = torch.empty(N, dtype=torch.long)
        argsort_inv[argsort] = torch.arange(N)

        # create row indices
        rowsort = row_ind[argsort]
        rowptr = torch.zeros(coo_tensor.shape[0], dtype=dtype)
        cur_row = -1
        for i in range(N):
            if cur_row != row_ind[i]:
                cur_row = row_ind[i]
                rowptr[cur_row] = i

        ctx.save_for_backward(argsort_inv, row_ind, col_ind)
        return SparseCSRTensor(data, col_ind[argsort], rowptr, coo_tensor.shape)

    @staticmethod
    def backward(ctx, csr_tensor):
        argsort_inv, row_ind, col_ind = ctx.saved_tensors
        return torch.sparse_coo_tensor(torch.Tensor((row_ind, col_ind)),
                                       csr_tensor.data[argsort_inv],
                                       size=csr_tensor.shape)

class nummlCSRSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, csr):
        ctx.save_for_backward(csr)
        return torch.sum(csr.data)

    @staticmethod
    def backward(ctx, grad):
        csr, = ctx.saved_tensors
        return SparseCSRTensor(grad * torch.ones_like(csr.data), csr.indices, csr.indptr, csr.shape)

class SparseCSRTensor(object):
    def __init__(self, data, indices, indptr, shape):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape

    def sum(self):
        return nummlCSRSum.apply(self)
