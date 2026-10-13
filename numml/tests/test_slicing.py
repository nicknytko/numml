import torch
import torch.linalg as tla
import numpy as np
import numml.sparse as sp
import pytest
import common

gpu = torch.device('cuda:0')
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')

R, C = 12, 9

# Row and column indices for a 12 x 9 matrix
indices = [
    (slice(None), slice(None)),
    (slice(2, 7), slice(1, 5)),
    (slice(None, None, 2), slice(1, None, 3)),
    (slice(None, None, -1), slice(7, 2, -2)),
    (slice(-4, None), slice(None, -2)),
    (slice(3, 3), slice(None)),
    (slice(None), slice(4, 4)),
    (3, slice(None)),
    (slice(None), 4),
    (-1, slice(2, 6)),
    ([0, 3, 5, 8], [1, 2, 6]),
    ([8, 3, 0, 5], [6, 1, 2]),
    ([1, 1, 4, 1], [3, 0, 3, 3, 8]),
    ([-1, 0, -12], [-9, -1]),
    ([], [1, 2]),
    ([1, 2], []),
    (np.array([5, 2, 9]), torch.tensor([0, 8, 4])),
    (range(1, 10, 3), range(8, 0, -2)),
    (torch.arange(R) % 3 == 0, np.arange(C) % 2 == 1),
    (torch.zeros(R, dtype=torch.bool), slice(None)),
    ([2, 7], slice(None)),
    (slice(None), [5, 0, 5]),
    (torch.tensor(4), [0, 1]),
]


def dense_index(A_d, r, c):
    def as_index(i):
        if isinstance(i, slice):
            return torch.arange(*i.indices(len(ar)))
        if isinstance(i, int) or (isinstance(i, torch.Tensor) and i.ndim == 0):
            return torch.tensor([int(i)])
        i = torch.as_tensor(np.asarray(i))
        if i.dtype == torch.bool:
            return torch.nonzero(i).flatten()
        return i.long()

    ar = torch.arange(A_d.shape[0])
    ri = ar[as_index(r)]
    ar = torch.arange(A_d.shape[1])
    ci = ar[as_index(c)]
    return A_d[ri][:, ci]


def check_sorted(A):
    for i in range(A.shape[0]):
        row = A.indices[A.indptr[i]:A.indptr[i+1]]
        assert(torch.all(row[1:] > row[:-1]))
    assert(A.indptr[0] == 0 and A.indptr[-1] == A.nnz)
    assert(len(A.indptr) == A.shape[0] + 1)


@pytest.mark.parametrize('r,c', indices)
def test_submatrix(r, c):
    A_d, A = common.random_sparse(R, C, 0.4)
    B = A[r, c]
    B_d = dense_index(A_d, r, c)

    assert(tuple(B.shape) == tuple(B_d.shape))
    assert(torch.equal(B.to_dense(), B_d))
    check_sorted(B)


@needs_cuda
@pytest.mark.parametrize('r,c', indices)
def test_submatrix_cuda(r, c):
    A_d, A = common.random_sparse(R, C, 0.4)
    B = A.to(gpu)[r, c]
    B_d = dense_index(A_d, r, c)

    assert(B.device.type == 'cuda')
    assert(tuple(B.shape) == tuple(B_d.shape))
    assert(torch.equal(B.to_dense().cpu(), B_d))
    check_sorted(B.cpu())


def test_rows_only():
    A_d, A = common.random_sparse(R, C, 0.4)

    assert(torch.equal(A[[4, 1, 1]].to_dense(), A_d[[4, 1, 1]]))
    assert(torch.equal(A[2:8:2].to_dense(), A_d[2:8:2]))
    assert(torch.equal(A[5].to_dense(), A_d[5:6]))


def test_scalar():
    A_d, A = common.random_sparse(R, C, 0.4)

    for i in range(-R, R):
        for j in range(-C, C):
            a = A[i, j]
            assert(isinstance(a, torch.Tensor) and a.ndim == 0)
            assert(a == A_d[i, j])


def test_no_alias():
    A_d, A = common.random_sparse(R, C, 0.4)
    B = A[:, :]
    B.data[:] = 0.
    assert(torch.equal(A.to_dense(), A_d))


def test_errors():
    _, A = common.random_sparse(R, C, 0.4)

    with pytest.raises(IndexError):
        A[R, 0]
    with pytest.raises(IndexError):
        A[0, -C - 1]
    with pytest.raises(IndexError):
        A[[0, R], :]
    with pytest.raises(IndexError):
        A[torch.ones(R + 1, dtype=torch.bool), :]
    with pytest.raises(IndexError):
        A[0, 0, 0]
    with pytest.raises(IndexError):
        A[None, 0]
    with pytest.raises(IndexError):
        A[[[0, 1], [2, 3]], :]
    with pytest.raises(IndexError):
        A[[0.5, 1.], :]


@pytest.mark.parametrize('r,c', indices)
def test_grad(r, c):
    A_d, A = common.random_sparse(R, C, 0.4)
    A_d = A_d.double()
    A = A.double()
    A_d.requires_grad = True
    A.requires_grad = True

    B_d = dense_index(A_d, r, c)
    W = torch.randn(B_d.shape, dtype=torch.double)

    (B_d * W).sum().backward()
    (A[r, c].to_dense() * W).sum().backward()

    assert(torch.allclose(A.grad.to_dense(), A_d.grad * (A_d != 0)))


def test_scalar_grad():
    A_d, A = common.random_sparse(R, C, 0.4)
    A.requires_grad = True

    i, j = [int(x) for x in torch.nonzero(A_d)[3]]
    A[i, j].backward()

    grad = torch.zeros_like(A_d)
    grad[i, j] = 1.
    assert(torch.equal(A.grad.to_dense(), grad))


def test_local_solve():
    N = 32
    A_d, _ = common.random_sparse(N, N, 0.2)
    A_d = A_d.double() + torch.eye(N, dtype=torch.double) * N
    A = sp.SparseCSRTensor(A_d)
    r = torch.randn(N, dtype=torch.double)

    for inds in [torch.arange(4, 12), torch.randperm(N)[:10], slice(0, N, 3)]:
        A_sub_d = dense_index(A_d, inds, inds)
        x = sp.linalg.spsolve(A[inds, inds], r[inds])
        assert(torch.allclose(x, tla.solve(A_sub_d, r[inds])))

        du = torch.zeros(N, dtype=torch.double)
        du[inds] += x
        assert(torch.allclose(A_sub_d @ du[inds], r[inds]))


def test_local_solve_grad():
    N = 32
    A_d, _ = common.random_sparse(N, N, 0.2)
    A_d = A_d.double() + torch.eye(N, dtype=torch.double) * N
    A = sp.SparseCSRTensor(A_d)
    r = torch.randn(N, dtype=torch.double)
    inds = torch.randperm(N)[:10]

    A_d.requires_grad = True
    A.requires_grad = True

    tla.solve(A_d[inds][:, inds], r[inds]).sum().backward()
    sp.linalg.spsolve(A[inds, inds], r[inds]).sum().backward()

    mask = (A_d != 0)
    assert(torch.allclose(A.grad.to_dense(), A_d.grad * mask))
