import torch
import numml.sparse as sp
import numml_torch_cpp

gpu = torch.device('cuda:0')

def dense_lu(A):
    # taken from the numpde book
    assert(A.shape[0] == A.shape[1])

    M = A.clone()
    n = A.shape[0]

    for k in range(0, n - 1):
        for i in range(k + 1, n):
            M[i, k] = M[i, k] / M[k, k]
            for j in range(k + 1, n):
                M[i, j] -= M[i, k] * M[k, j]
    return M

A_fill = torch.Tensor([
    [5., 1., 1., 1.],
    [1., 5., 0., 0.],
    [1., 0., 5., 0.],
    [1., 0., 0., 5.]
])
A_fill = sp.SparseCSRTensor(A_fill).to(gpu)

print(' -- Filled (arrow) matrix -- ')
print(A_fill.to_dense())
print(' Dense LU (no pivoting)')
print(dense_lu(A_fill.to_dense()))
print(' Sparse LU')
M_data, M_indices, M_indptr, Mt_data, Mt_indices, Mt_indptr = numml_torch_cpp.splu(A_fill.shape[0], A_fill.shape[1], A_fill.data, A_fill.indices, A_fill.indptr)
print(sp.SparseCSRTensor((M_data, M_indices, M_indptr), A_fill.shape).to_dense())

print(' -- No fill (reversed arrow) matrix -- ')
A_opt = torch.Tensor([
    [5., 0., 0., 1.],
    [0., 5., 0., 1.],
    [0., 0., 5., 1.],
    [1., 1., 1., 5.]
])
A_opt = sp.SparseCSRTensor(A_opt).to(gpu)
print(A_opt.to_dense())
print(' Dense LU (no pivoting)')
print(dense_lu(A_opt.to_dense()))
numml_torch_cpp.splu(A_opt.shape[0], A_opt.shape[1], A_opt.data, A_opt.indices, A_opt.indptr)
M_data, M_indices, M_indptr, Mt_data, Mt_indices, Mt_indptr = numml_torch_cpp.splu(A_opt.shape[0], A_opt.shape[1], A_opt.data, A_opt.indices, A_opt.indptr)
print(' Sparse LU')
print(sp.SparseCSRTensor((M_data, M_indices, M_indptr), A_opt.shape).to_dense())

N = 8
A_fd = (sp.eye(N) * 2 - sp.eye(N, k=-1) - sp.eye(N, k=1)).to(gpu)
print(' -- finite difference in 1D -- ')
print(A_fd.to_dense())
print(' Dense LU (no pivoting)')
print(dense_lu(A_fd.to_dense()))
print(' Sparse LU')
M_data, M_indices, M_indptr, Mt_data, Mt_indices, Mt_indptr = numml_torch_cpp.splu(A_fd.shape[0], A_fd.shape[1], A_fd.data, A_fd.indices, A_fd.indptr)
print(sp.SparseCSRTensor((M_data, M_indices, M_indptr), A_fd.shape).to_dense())
