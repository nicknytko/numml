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


def test_permute_grad():
    N = 16

    it = 10
    for i in range(it):
        x = torch.randn(N, dtype=torch.double)
        w = torch.randn(N, dtype=torch.double)
        P = torch.randperm(N)

        assert(torch.allclose(utils.permute(x, P), x[P]))
        assert(torch.allclose(utils.permute_inverse(utils.permute(x, P), P), x))

        for perm in [utils.permute, utils.permute_inverse]:
            def grad_x_helper(x_in):
                return perm(x_in, P) @ w

            assert(common.relerr(utils.fd(grad_x_helper, x), utils.grad(grad_x_helper, x)) < 1e-6)


def test_splu_solve_grad():
    N = 8
    N2 = N * N
    A = (sp.eye(N2) * 4 - sp.eye(N2, k=1) - sp.eye(N2, k=-1) - sp.eye(N2, k=-N) - sp.eye(N2,k=N)).double()
    A_LU = sp.linalg.splu(A)

    it = 10
    for i in range(it):
        b = torch.randn(N2, dtype=torch.double)
        w = torch.randn(N2, dtype=torch.double)

        def splu_helper(b_in):
            return sp.linalg.splu_solve(A_LU, b_in) @ w
        def spsolve_helper(b_in):
            return sp.linalg.spsolve(A, b_in) @ w

        assert(torch.allclose(sp.linalg.splu_solve(A_LU, b), sp.linalg.spsolve(A, b)))

        grad_b = utils.grad(splu_helper, b)
        assert(torch.allclose(grad_b, utils.grad(spsolve_helper, b)))
        assert(common.relerr(utils.fd(splu_helper, b), grad_b) < 1e-6)


def test_splu_solve_grad_A():
    N = 6
    N2 = N * N
    A = (sp.eye(N2) * 4 - sp.eye(N2, k=1) - sp.eye(N2, k=-1) - sp.eye(N2, k=-N) - sp.eye(N2,k=N)).double()

    it = 5
    for i in range(it):
        b = torch.randn(N2, dtype=torch.double)
        w = torch.randn(N2, dtype=torch.double)

        # Reuse one factorization for several solves
        def splu_helper(A_in):
            A_LU = sp.linalg.splu(A_in)
            return sp.linalg.splu_solve(A_LU, sp.linalg.splu_solve(A_LU, b)) @ w
        def spsolve_helper(A_in):
            return sp.linalg.spsolve(A_in, sp.linalg.spsolve(A_in, b)) @ w

        grad_A = utils.grad(splu_helper, A)
        assert(torch.allclose(grad_A.data, utils.grad(spsolve_helper, A).data))
        assert(common.relerr(utils.fd(splu_helper, A, h=1e-5), grad_A) < 1e-6)
