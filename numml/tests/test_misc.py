import torch
import torch.linalg as tla
import numml.sparse as sp
import pytest
import common


def test_to_dense():
    N = 16
    A = sp.eye(N)*3 + sp.eye(N, k=1)*2 + sp.eye(N,k=-1)*2 + sp.eye(N, k=2) + sp.eye(N,k=-2)
    A_d = A.to_dense()
    assert(torch.all(torch.diag(A_d, 0) == 3.))
    assert(torch.all(torch.diag(A_d, 1) == 2.))
    assert(torch.all(torch.diag(A_d, -1) == 2.))
    assert(torch.all(torch.diag(A_d, 2) == 1.))
    assert(torch.all(torch.diag(A_d, -2) == 1.))
    assert(torch.all(torch.diag(A_d, 3) == 0.))
    assert(torch.all(torch.diag(A_d, -3) == 0.))
