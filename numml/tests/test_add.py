import torch
import torch.linalg as tla
import numml.sparse as sp
import numml.utils as utils
import pytest
import common

gpu = torch.device('cuda:0')

def test_add_random():
    N = 64
    it = 10
    for i in range(it):
        A_d, A = common.random_sparse(N, N, 0.25)
        B_d, B = common.random_sparse(N, N, 0.25)

        A_c = A.to(gpu)
        B_c = B.to(gpu)

        assert(torch.allclose(A_d+B_d, (A+B).to_dense()))
        assert(torch.allclose((A_c+B_c).data.cpu(), (A+B).data))

def test_sub_random():
    N = 64
    it = 10
    for i in range(it):
        A_d, A = common.random_sparse(N, N, 0.25)
        B_d, B = common.random_sparse(N, N, 0.25)

        A_c = A.to(gpu)
        B_c = B.to(gpu)

        assert(torch.allclose(A_d-B_d, (A-B).to_dense()))
        assert(torch.allclose((A_c-B_c).data.cpu(), (A-B).data))

def test_add_random_grad():
    N = 32
    it = 10
    for i in range(it):
        A_d, A = common.random_sparse(N, N, 0.25)
        B_d, B = common.random_sparse(N, N, 0.25)

        A_c = A.to(gpu)
        B_c = B.to(gpu)

        A_ng = A.clone()
        B_ng = B.clone()

        def fd_A_helper(A_in):
            return (A_in + B_ng).sum()

        def fd_B_helper(B_in):
            return (A_ng + B_in).sum()

        A.requires_grad = True
        B.requires_grad = True

        A_c.requires_grad = True
        B_c.requires_grad = True

        (A+B).sum().backward()
        (A_c+B_c).sum().backward()

        assert(torch.allclose(A.grad.data, A_c.grad.data.cpu()))
        assert(torch.allclose(B.grad.data, B_c.grad.data.cpu()))
        assert(common.relerr(utils.sp_fd(fd_A_helper, A_ng, h=1e-4), A.grad) < 0.1)
        assert(common.relerr(utils.sp_fd(fd_B_helper, B_ng, h=1e-4), B.grad) < 0.1)
