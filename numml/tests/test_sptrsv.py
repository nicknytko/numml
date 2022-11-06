import torch
import torch.linalg as tla
import numml.sparse as sp
import numml_torch_cpp
import pytest
import random
import common
import faulthandler
faulthandler.enable()


def test_identity():
    N = 32
    I = sp.eye(N)
    it = 10

    for i in range(it):
        x = torch.randn(N)
        for lower in [False, True]:
            for unit in [False, True]:
                y = numml_torch_cpp.sptrsv_forward(I.shape[0], I.shape[1], I.data, I.indices, I.indptr,
                                                   lower, unit, x)
                assert(torch.allclose(y, x))

def test_lower():
    N = 32
    A = sp.eye(N) * 2 - sp.eye(N, k=-1)
    it = 10

    for i in range(it):
        x_e = torch.randn(N)
        b = A @ x_e
        x_g = numml_torch_cpp.sptrsv_forward(A.shape[0], A.shape[1], A.data, A.indices, A.indptr, True, False, b)
        assert(tla.norm(b-A@x_g) < 1e-4)

    Au = sp.eye(N) - sp.eye(N, k=-1) * 2
    for i in range(it):
        x_e = torch.randn(N)
        b = Au @ x_e
        x_g = numml_torch_cpp.sptrsv_forward(Au.shape[0], Au.shape[1], Au.data, Au.indices, Au.indptr, True, True, b)
        assert(tla.norm(b-Au@x_g) < 1e-4)


def test_upper():
    N = 32
    A = sp.eye(N) * 2 - sp.eye(N, k=1)
    it = 10

    for i in range(it):
        x_e = torch.randn(N)
        b = A @ x_e
        x_g = numml_torch_cpp.sptrsv_forward(A.shape[0], A.shape[1], A.data, A.indices, A.indptr, False, False, b)
        assert(tla.norm(b-A@x_g) < 1e-4)

    Au = sp.eye(N) - sp.eye(N, k=1) * 2
    for i in range(it):
        x_e = torch.randn(N)
        b = Au @ x_e
        x_g = numml_torch_cpp.sptrsv_forward(Au.shape[0], Au.shape[1], Au.data, Au.indices, Au.indptr, False, True, b)
        assert(tla.norm(b-Au@x_g) < 1e-4)
