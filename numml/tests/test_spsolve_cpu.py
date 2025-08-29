import torch
import torch.linalg as tla
import numml.sparse as sp
import numml.utils as utils
import pytest
import common


def test_invert_random():
    N = 8
    _, A = common.random_sparse(N, N, 0.25)
    A = A + sp.eye(N) * N

    it = 10
    for i in range(it):
        u = torch.randn(N)

        assert(torch.allclose(u, sp.linalg.spsolve(A, A @ u)))


def test_invert_fd():
    N = 8
    N2 = N * N
    A = sp.eye(N2) * 4 - sp.eye(N2, k=1) - sp.eye(N2, k=-1) - sp.eye(N2, k=-N) - sp.eye(N2,k=N)

    it = 10
    for i in range(it):
        u = torch.randn(N2)

        assert(tla.norm(u - sp.linalg.spsolve(A, A @ u)) < 1e-5)
