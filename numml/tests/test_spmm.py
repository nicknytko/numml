import torch
import torch.linalg as tla
import numml.sparse as sp
import numml.utils as utils
import pytest
import common


gpu = torch.device('cuda:0')

# The large matrix should be big enough to require multiple CUDA warps
A_N = 16; AL_N = 2048
A = sp.eye(A_N)*2 - sp.eye(A_N, k=-1) - sp.eye(A_N,k=1); AL = sp.eye(AL_N)*2 - sp.eye(AL_N, k=-1) - sp.eye(AL_N,k=1)
B = (-A).copy(); BL = (-AL).copy()

A_d = A.to_dense(); AL_d = AL.to_dense()
B_d = B.to_dense(); BL_d = BL.to_dense()

A_c = A.to(gpu); AL_c = AL.to(gpu)
B_c = B.to(gpu); BL_c = BL.to(gpu)


def test_matmat():
    C = A@B
    C_c = A_c@B_c

    assert(torch.allclose(C.data, C_c.data.cpu()))
    assert(torch.allclose(A_d@B_d, C.to_dense()))


def test_matmat_large():
    C = AL@BL
    C_c = AL_c@BL_c

    assert(torch.allclose(C.data, C_c.data.cpu()))
    assert(torch.allclose(AL_d@BL_d, C.to_dense()))


def test_skinny():
    # Create a tall, skinny interpolation matrix P

    P_d = torch.zeros((A_N, 4))
    for i in range(4):
        P_d[i*4:(i+1)*4, i] = 1.
    P = sp.SparseCSRTensor(P_d)
    P_c = P.to(gpu)

    AH = P.T @ A @ P
    AH_c = P_c.T @ A_c @ P_c

    assert(torch.allclose(AH.data, AH_c.data.cpu()))
    assert(torch.allclose(AH.to_dense(), P_d.T@A_d@P_d))


def grad_C_helper(C):
    grad_C = sp.SparseCSRTensor((torch.ones_like(C.data), C.indices, C.indptr), C.shape)
    return grad_C.to_dense()


def dense_mask(A):
    mask = torch.zeros(A.shape, dtype=torch.bool)
    for row in range(A.shape[0]):
        for col_i in range(A.indptr[row], A.indptr[row + 1]):
            col = A.indices[col_i]
            mask[row, col] = True
    return mask


def test_grad_A():
    Ag = AL.copy()
    Ag.requires_grad = True
    Ag_c = AL_c.copy()
    Ag_c.requires_grad = True

    Bg = BL.copy()
    Bg_c = BL_c.copy()

    C = (Ag @ Bg)
    C_c = (Ag_c @ Bg_c)

    C.sum().backward()
    C_c.sum().backward()

    assert(torch.allclose(Ag.data.grad, Ag_c.data.grad.cpu()))

    # Analytic gradient is (grad C * B^T) (*) mask(A)
    grad = (grad_C_helper(C) @ BL.to_dense().T) * dense_mask(AL)
    assert(torch.allclose(grad, Ag.grad.to_dense()))


def test_grad_B():
    Ag = AL.copy()
    Ag_c = AL_c.copy()

    Bg = BL.copy()
    Bg.requires_grad = True
    Bg_c = BL_c.copy()
    Bg_c.requires_grad = True

    C = (Ag @ Bg)
    C_c = (Ag_c @ Bg_c)

    C.sum().backward()
    C_c.sum().backward()

    assert(torch.allclose(Bg.data.grad, Bg_c.data.grad.cpu()))

    # Analytic gradient is (A^T * grad_C) (*) mask(B)
    grad = (AL.to_dense().T @ grad_C_helper(C)) * dense_mask(BL)
    assert(torch.allclose(grad, Bg.grad.to_dense()))


def test_skinny_grad_A():
    Ag = A.copy()
    Ag.requires_grad = True
    Ag_c = A_c.copy()
    Ag_c.requires_grad = True

    P_d = torch.zeros((A_N, 4))
    for i in range(4):
        P_d[i*4:(i+1)*4, i] = 1.
    P = sp.SparseCSRTensor(P_d)
    P_c = P.to(gpu)

    AH = P.T @ Ag @ P
    AH_c = P_c.T @ Ag_c @ P_c

    AH.sum().backward()
    AH_c.sum().backward()

    def grad_A_fd_helper(A):
           return (P.T @ A @ P).sum()

    assert(torch.allclose(Ag.grad.data, Ag_c.grad.data.cpu()))
    assert(common.relerr(utils.sp_fd(grad_A_fd_helper, A), Ag.grad) < 1e-3)


def test_skinny_grad_P():
    P_d = torch.zeros((A_N, 4))
    for i in range(4):
        P_d[i*4:(i+1)*4, i] = 1.
    P = sp.SparseCSRTensor(P_d)
    P_c = P.to(gpu)
    P.requires_grad = True
    P_c.requires_grad = True

    AH = P.T @ A @ P
    AH_c = P_c.T @ A_c @ P_c

    AH.sum().backward()
    AH_c.sum().backward()

    def grad_A_fd_helper(P):
           return (P.T @ A @ P).sum()

    assert(torch.allclose(P.grad.data, P_c.grad.data.cpu()))
    assert(common.relerr(utils.sp_fd(grad_A_fd_helper, P.detach()), P.grad) < 1e-3)
