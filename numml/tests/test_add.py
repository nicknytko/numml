import torch
import torch.linalg as tla
import numml.sparse as sp
import pytest
import common

gpu = torch.device('cuda:0')

def test_add_random():
    N = 6
    it = 10
    for i in range(it):
        A_d, A = common.random_sparse(N, N, 0.25)
        B_d, B = common.random_sparse(N, N, 0.25)

        A_c = A.to(gpu)
        B_c = B.to(gpu)

        assert(torch.allclose(A_d+B_d, (A+B).to_dense()))
        assert(torch.allclose((A_c+B_c).data.cpu(), (A+B).data))

def test_sub_random():
    N = 32
    it = 10
    for i in range(it):
        A_d, A = common.random_sparse(N, N, 0.25)
        B_d, B = common.random_sparse(N, N, 0.25)

        A_c = A.to(gpu)
        B_c = B.to(gpu)

        assert(torch.allclose(A_d-B_d, (A-B).to_dense()))
        assert(torch.allclose((A_c-B_c).data.cpu(), (A-B).data))
