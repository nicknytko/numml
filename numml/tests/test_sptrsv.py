import torch
import torch.linalg as tla
import numml.sparse as sp
import numml.utils as utils
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
        assert(tla.norm(b-A@x_g) / tla.norm(b) < 1e-5)

    Au = sp.eye(N) - sp.eye(N, k=-1) * 2
    Au_c = Au.to(gpu)
    for i in range(it):
        x_e = torch.randn(N)
        b = Au @ x_e
        b_c = b.to(gpu)

        x_g = Au.solve_triangular(False, True, b)
        x_g_c = Au_c.solve_triangular(False, True, b_c)

        assert(torch.allclose(x_g, x_g_c.cpu()))
        assert(tla.norm(b-Au@x_g) /tla.norm(b) < 1e-5)


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
        assert(tla.norm(b-A@x_g) / tla.norm(b) < 1e-5)

    Au = sp.eye(N) - sp.eye(N, k=1) * 2
    Au_c = Au.to(gpu)
    for i in range(it):
        x_e = torch.randn(N)
        b = Au @ x_e
        b_c = b.to(gpu)

        x_g = Au.solve_triangular(True, True, b)
        x_g_c = Au_c.solve_triangular(True, True, b_c)

        assert(torch.allclose(x_g, x_g_c.cpu()))
        assert(tla.norm(b-Au@x_g) /tla.norm(b) < 1e-5)


def test_rand_lower():
    it = 10

    for i in range(it):
        N = torch.randint(32, 2048, (1,)).item()
        A_d, A = common.random_sparse_tri(N, N, 0.05, upper=False, include_diag=False)
        A = A + sp.eye(N)*2. # set diagonal to be 2.
        A = A.double()

        A_c = A.to(gpu)

        b = torch.randn(N, dtype=torch.double)
        b_c = b.to(gpu)

        x = A.solve_triangular(False, False, b)
        x_c = A_c.solve_triangular(False, False, b_c)

        print('N', N)
        print('ans diff', tla.norm(x - x_c.cpu()))
        print('cpu', x)
        print('gpu', x_c.cpu())

        assert(torch.allclose(x, x_c.cpu()))
        assert(tla.norm(b-A@x) / tla.norm(b) < 1e-5)


def test_grad_lower():
    N = 32
    A = sp.eye(N) * 2 - sp.eye(N, k=-1)
    A_c = A.to(gpu)
    it = 10

    for i in range(it):
        x_e = torch.randn(N)
        b = A @ x_e
        b_c = b.to(gpu)

        Ag = A.copy()
        Ag_c = A_c.copy()
        Ag.requires_grad = True
        Ag_c.requires_grad = True
        b.requires_grad = True
        b_c.requires_grad = True

        x_g = Ag.solve_triangular(False, False, b)
        x_g_c = Ag_c.solve_triangular(False, False, b_c)

        x_g.sum().backward()
        x_g_c.sum().backward()

        def grad_A_helper(A_in):
            return (A_in.solve_triangular(False, False, b.detach())).sum()
        def grad_b_helper(b_in):
            return (A.solve_triangular(False, False, b_in)).sum()

        assert(torch.allclose(Ag.data.grad, Ag_c.data.grad.cpu()))
        assert(torch.allclose(b.grad, b_c.grad.cpu()))
        assert(common.relerr(utils.fd(grad_A_helper, A).data, Ag.data.grad) < 1e-3)
        assert(common.relerr(utils.fd(grad_b_helper, b.detach()), b.grad) < 1e-3)

    Au = sp.eye(N) - sp.eye(N, k=-1) * 2
    Au_c = Au.to(gpu)
    for i in range(it):
        x_e = torch.randn(N)
        b = Au @ x_e
        b_c = b.to(gpu)

        Aug = Au.copy()
        Aug_c = Au_c.copy()
        Aug.requires_grad = True
        Aug_c.requires_grad = True
        b.requires_grad = True
        b_c.requires_grad = True

        x_g = Aug.solve_triangular(False, True, b)
        x_g_c = Aug_c.solve_triangular(False, True, b_c)

        x_g.sum().backward()
        x_g_c.sum().backward()

        def grad_A_helper(A_in):
            return (A_in.solve_triangular(False, False, b.detach())).sum() # We have to have unit off, or we won't get grads in half of the entries
        def grad_b_helper(b_in):
            return (Au.solve_triangular(False, True, b_in)).sum()

        assert(torch.allclose(Aug.data.grad, Aug_c.data.grad.cpu()))
        assert(torch.allclose(b.grad, b_c.grad.cpu()))
        assert(common.relerr(utils.fd(grad_A_helper, Au).data, Aug.data.grad) < 1e-3)
        assert(common.relerr(utils.fd(grad_b_helper, b.detach()), b.grad) < 1e-3)


def test_grad_upper():
    N = 32
    A = sp.eye(N) * 2 - sp.eye(N, k=1)
    A_c = A.to(gpu)
    it = 10

    for i in range(it):
        x_e = torch.randn(N)
        b = A @ x_e
        b_c = b.to(gpu)

        Ag = A.copy()
        Ag_c = A_c.copy()
        Ag.requires_grad = True
        Ag_c.requires_grad = True
        b.requires_grad = True
        b_c.requires_grad = True

        x_g = Ag.solve_triangular(True, False, b)
        x_g_c = Ag_c.solve_triangular(True, False, b_c)

        x_g.sum().backward()
        x_g_c.sum().backward()

        def grad_A_helper(A_in):
            return (A_in.solve_triangular(True, False, b.detach())).sum()
        def grad_b_helper(b_in):
            return (A.solve_triangular(True, False, b_in)).sum()

        assert(torch.allclose(Ag.data.grad, Ag_c.data.grad.cpu()))
        assert(torch.allclose(b.grad, b_c.grad.cpu()))
        assert(common.relerr(utils.fd(grad_A_helper, A).data, Ag.data.grad) < 1e-3)
        assert(common.relerr(utils.fd(grad_b_helper, b.detach()), b.grad) < 1e-3)

    Au = sp.eye(N) - sp.eye(N, k=1) * 2
    Au_c = Au.to(gpu)
    for i in range(it):
        x_e = torch.randn(N)
        b = Au @ x_e
        b_c = b.to(gpu)

        Aug = Au.copy()
        Aug_c = Au_c.copy()
        Aug.requires_grad = True
        Aug_c.requires_grad = True
        b.requires_grad = True
        b_c.requires_grad = True

        x_g = Aug.solve_triangular(True, True, b)
        x_g_c = Aug_c.solve_triangular(True, True, b_c)

        x_g.sum().backward()
        x_g_c.sum().backward()

        def grad_A_helper(A_in):
            return (A_in.solve_triangular(True, False, b.detach())).sum() # We have to have unit off, or we won't get grads in half of the entries
        def grad_b_helper(b_in):
            return (Au.solve_triangular(True, True, b_in)).sum()

        assert(torch.allclose(Aug.data.grad, Aug_c.data.grad.cpu()))
        assert(torch.allclose(b.grad, b_c.grad.cpu()))
        assert(common.relerr(utils.fd(grad_A_helper, Au).data, Aug.data.grad) < 1e-3)
        assert(common.relerr(utils.fd(grad_b_helper, b.detach()), b.grad) < 1e-3)
