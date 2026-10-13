import torch
import numml.sparse as sp
import pytest
import common

N = 5

devices = [
    torch.device('cpu'),
    pytest.param(torch.device('cuda:0'),
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')),
]

# Shapes with no stored entries: no rows, no columns, neither, and all-zero
shapes = [(0, N), (N, 0), (0, 0), (N, N)]


def empty(shape, device):
    '''
    Returns dense and sparse versions of a matrix of the given shape with no stored entries.
    '''

    A_d = torch.zeros(shape, device=device)
    A = sp.SparseCSRTensor(torch.zeros(shape)).to(device)
    assert(A.nnz == 0)
    return A_d, A


def full(rows, cols, device):
    '''
    Returns dense and sparse versions of a random matrix, which has stored entries
    whenever the given shape allows for it.
    '''

    if rows == 0 or cols == 0:
        return empty((rows, cols), device)

    A_d, A = common.random_sparse(rows, cols, 0.5)
    return A_d.to(device), A.to(device)


def check(A, A_d):
    '''
    Checks a sparse matrix against its expected dense representation, and that the
    CSR structure itself is well-formed.
    '''

    assert(A.device.type == A_d.device.type)
    assert(tuple(A.shape) == tuple(A_d.shape))
    assert(len(A.indptr) == A.shape[0] + 1)
    assert(A.indptr[-1] == A.nnz and len(A.indices) == A.nnz)
    assert(torch.equal(A.to_dense(), A_d))


def check_dense(x, x_d):
    assert(x.device.type == x_d.device.type)
    assert(x.dtype == x_d.dtype)
    assert(tuple(x.shape) == tuple(x_d.shape))
    assert(torch.equal(x, x_d))


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('shape', shapes)
def test_to_dense(shape, device):
    A_d, A = empty(shape, device)
    check_dense(A.to_dense(), A_d)


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('shape', shapes)
def test_spmv(shape, device):
    A_d, A = empty(shape, device)
    x = torch.randn(shape[1], device=device)
    check_dense(A @ x, A_d @ x)


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('shape', shapes)
def test_transpose(shape, device):
    A_d, A = empty(shape, device)
    check(A.T, A_d.T)


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('shape', shapes)
def test_add(shape, device):
    A_d, A = empty(shape, device)
    check(A + A, A_d)
    check(A - A, A_d)


@pytest.mark.parametrize('device', devices)
def test_add_nonempty(device):
    Z_d, Z = empty((N, N), device)
    A_d, A = full(N, N, device)

    check(A + Z, A_d)
    check(Z + A, A_d)
    check(Z - A, -A_d)


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('shape', shapes)
def test_spgemm(shape, device):
    A_d, A = empty(shape, device)
    L_d, L = full(N, shape[0], device)
    R_d, R = full(shape[1], N, device)

    check(A @ R, A_d @ R_d)
    check(L @ A, L_d @ A_d)
    check(A @ A.T, A_d @ A_d.T)
    check(A.T @ A, A_d.T @ A_d)


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('shape', shapes)
def test_spdmm(shape, device):
    A_d, A = empty(shape, device)
    B = torch.randn(shape[1], 3, device=device)
    check_dense(A @ B, A_d @ B)


@pytest.mark.parametrize('device', devices)
def test_spdmm_no_columns(device):
    A_d, A = full(N, N, device)
    B = torch.randn(N, 0, device=device)
    check_dense(A @ B, A_d @ B)


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('shape', shapes)
def test_rmatmul(shape, device):
    A_d, A = empty(shape, device)
    B = torch.randn(3, shape[0], device=device)
    check_dense(B @ A, B @ A_d)


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('shape', shapes)
def test_reductions(shape, device):
    A_d, A = empty(shape, device)

    check_dense(A.row_sum(), A_d.sum(dim=1))
    check_dense(A.diagonal(), torch.diagonal(A_d))
    check_dense(A.sum(), A_d.sum())
    check_dense(A.trace(), torch.diagonal(A_d).sum())


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('shape', shapes)
def test_tri(shape, device):
    A_d, A = empty(shape, device)
    check(A.tril(), A_d)
    check(A.triu(), A_d)


@pytest.mark.parametrize('device', devices)
def test_solve_triangular(device):
    _, A = empty((0, 0), device)
    b = torch.zeros(0, device=device)

    for upper in (False, True):
        for unit in (False, True):
            check_dense(A.solve_triangular(upper, unit, b), b)


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('shape', shapes)
def test_grad_dense_operand(shape, device):
    # Nothing depends on the dense operand, so its gradient should be identically zero

    _, A = empty(shape, device)
    A.requires_grad = True

    x = torch.randn(shape[1], device=device, requires_grad=True)
    (A @ x).sum().backward()
    check_dense(x.grad, torch.zeros_like(x))
    check_dense(A.data.grad, torch.zeros(0, device=device))

    B = torch.randn(shape[1], 3, device=device, requires_grad=True)
    (A @ B).sum().backward()
    check_dense(B.grad, torch.zeros_like(B))


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('shape', shapes)
def test_grad_sparse_operand(shape, device):
    _, A = empty(shape, device)
    _, L = full(N, shape[0], device)
    _, R = full(shape[1], N, device)
    A.requires_grad = True
    L.requires_grad = True
    R.requires_grad = True

    loss = ((L @ A).sum() + (A @ R).sum() + (A + A).sum() + A.T.sum() +
            A.to_dense().sum() + A.row_sum().sum() + A.diagonal().sum())
    loss.backward()

    check_dense(A.data.grad, torch.zeros(0, device=device))
    check_dense(L.data.grad, torch.zeros_like(L.data))
    check_dense(R.data.grad, torch.zeros_like(R.data))


@pytest.mark.parametrize('device', devices)
def test_grad_spdmm_no_columns(device):
    _, A = full(N, N, device)
    A.requires_grad = True
    B = torch.randn(N, 0, device=device, requires_grad=True)

    (A @ B).sum().backward()
    check_dense(A.data.grad, torch.zeros_like(A.data))
    check_dense(B.grad, torch.zeros_like(B))


@pytest.mark.parametrize('device', devices)
def test_no_pending_error(device):
    # A failed kernel launch on the GPU may only be reported by a later, unrelated call

    for shape in shapes:
        _, A = empty(shape, device)
        A.to_dense(); A.row_sum(); A.diagonal(); A.T

    _, A = full(N, N, device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    assert(torch.equal(A.to(device).data, A.data))
