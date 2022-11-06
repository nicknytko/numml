import torch
import torch.linalg as tla
import numml.sparse as sp
import numml_torch_cpp
import pytest
import random
import common

gpu = torch.device('cuda:0')

def test_identity():
    N = 32
    I = sp.eye(N)
    I_c = I.to(gpu)
    it = 10

    for i in range(it):
        x = torch.randn(N)
        x_c = x.to(gpu)
        for upper in [False, True]:
            for unit in [False, True]:
                y = I.solve_triangular(upper, unit, x)
                y_c = I_c.solve_triangular(upper, unit, x_c)
                assert(torch.allclose(y, x))
                assert(torch.allclose(y_c.cpu(), x))

def test_lower():
    N = 32
    A = sp.eye(N) * 2 - sp.eye(N, k=-1)
    A_c = A.to(gpu)
    it = 10

    for i in range(it):
        x_e = torch.randn(N)
        b = A @ x_e
        b_c = b.to(gpu)
        x_g = A.solve_triangular(False, False, b)
        x_g_c = A_c.solve_triangular(False, False, b_c)

        assert(torch.allclose(x_g, x_g_c.cpu()))
        assert(tla.norm(b-A@x_g) / tla.norm(b) < 1e-6)

    Au = sp.eye(N) - sp.eye(N, k=-1) * 2
    Au_c = Au.to(gpu)
    for i in range(it):
        x_e = torch.randn(N)
        b = Au @ x_e
        b_c = b.to(gpu)

        x_g = Au.solve_triangular(False, True, b)
        x_g_c = Au_c.solve_triangular(False, True, b_c)

        assert(torch.allclose(x_g, x_g_c.cpu()))
        assert(tla.norm(b-Au@x_g) /tla.norm(b) < 1e-6)


def test_upper():
    N = 32
    A = sp.eye(N) * 2 - sp.eye(N, k=1)
    A_c = A.to(gpu)
    it = 10

    for i in range(it):
        x_e = torch.randn(N)
        b = A @ x_e
        b_c = b.to(gpu)
        x_g = A.solve_triangular(True, False, b)
        x_g_c = A_c.solve_triangular(True, False, b_c)

        assert(torch.allclose(x_g, x_g_c.cpu()))
        assert(tla.norm(b-A@x_g) / tla.norm(b) < 1e-6)

    Au = sp.eye(N) - sp.eye(N, k=1) * 2
    Au_c = Au.to(gpu)
    for i in range(it):
        x_e = torch.randn(N)
        b = Au @ x_e
        b_c = b.to(gpu)

        x_g = Au.solve_triangular(True, True, b)
        x_g_c = Au_c.solve_triangular(True, True, b_c)

        assert(torch.allclose(x_g, x_g_c.cpu()))
        assert(tla.norm(b-Au@x_g) /tla.norm(b) < 1e-6)
