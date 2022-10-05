import torch
import torch.linalg as tla
import numml.sparse as sp
import pytest

gpu = torch.device('cuda:0')

def test_trivial():
    N = 16
    A = sp.eye(N)*2 - sp.eye(N, k=-1) - sp.eye(N,k=1)
    A_c = A.to(gpu)

    At = A.T
    At_c = A_c.T

    assert(torch.allclose(At.data, At_c.data.cpu()))
    assert(torch.allclose(At.indptr, At_c.indptr.cpu()))
    assert(torch.allclose(At.indices, At_c.indices.cpu()))

    assert(torch.allclose(At.data, A.data))
    assert(torch.allclose(At.indptr, A.indptr))
    assert(torch.allclose(At.indices, A.indices))

def test_skinny_random():
    it = 10
    for i in range(it):
        N_r, N_c = 512, 64
        A_d = torch.zeros((N_r, N_c))

        nnz = (N_r * N_c) // 2
        idx_r, idx_c = torch.randint(N_r, (nnz,)), torch.randint(N_c, (nnz,))
        A_d[idx_r, idx_c] = torch.randn(nnz)
        At_d = A_d.T

        A = sp.SparseCSRTensor(A_d)
        A_c = A.to(gpu)

        At = A.T
        At_c = A_c.T

        assert(torch.allclose(At.to_dense(), At_c.cpu().to_dense()))
        assert(torch.allclose(At.to_dense(), At_d))

def test_fat_random():
    it = 10
    for i in range(it):
        N_r, N_c = 64, 512
        A_d = torch.zeros((N_r, N_c))

        nnz = (N_r * N_c) // 2
        idx_r, idx_c = torch.randint(N_r, (nnz,)), torch.randint(N_c, (nnz,))
        A_d[idx_r, idx_c] = torch.randn(nnz)
        At_d = A_d.T

        A = sp.SparseCSRTensor(A_d)
        A_c = A.to(gpu)

        At = A.T
        At_c = A_c.T

        assert(torch.allclose(At.to_dense(), At_c.cpu().to_dense()))
        assert(torch.allclose(At.to_dense(), At_d))

def test_skinny_random_grad():
    it = 10
    for i in range(it):
        N_r, N_c = 512, 64
        A_d = torch.zeros((N_r, N_c))

        nnz = (N_r * N_c) // 2
        idx_r, idx_c = torch.randint(N_r, (nnz,)), torch.randint(N_c, (nnz,))
        A_d[idx_r, idx_c] = torch.randn(nnz)
        At_d = A_d.T

        A = sp.SparseCSRTensor(A_d)
        A_c = A.to(gpu)
        A.requires_grad = True
        A_c.requires_grad = True

        rand_mult = torch.randn(A.nnz)
        rand_mult_c = rand_mult.to(gpu)

        At = A.T
        At_c = A_c.T

        At.data = At.data * rand_mult
        At.sum().backward()
        At_c.data = At_c.data * rand_mult_c
        At_c.sum().backward()

        assert(torch.allclose(A.grad.to_dense(), A_c.grad.to_dense().cpu()))

def test_fat_random_grad():
    it = 10
    for i in range(it):
        N_r, N_c = 64, 512
        A_d = torch.zeros((N_r, N_c))

        nnz = (N_r * N_c) // 2
        idx_r, idx_c = torch.randint(N_r, (nnz,)), torch.randint(N_c, (nnz,))
        A_d[idx_r, idx_c] = torch.randn(nnz)
        At_d = A_d.T

        A = sp.SparseCSRTensor(A_d)
        A_c = A.to(gpu)
        A.requires_grad = True
        A_c.requires_grad = True

        rand_mult = torch.randn(A.nnz)
        rand_mult_c = rand_mult.to(gpu)

        At = A.T
        At_c = A_c.T

        At.data = At.data * rand_mult
        At.sum().backward()
        At_c.data = At_c.data * rand_mult_c
        At_c.sum().backward()

        assert(torch.allclose(A.grad.to_dense(), A_c.grad.to_dense().cpu()))
