import torch
import torch.linalg as tla
import numml.sparse as sp
import numml.utils as utils
import pytest

gpu = torch.device('cuda:0')

# The large matrix should be big enough to require multiple CUDA warps
A_N = 16; AL_N = 2048
A = sp.eye(A_N)*2 - sp.eye(A_N, k=-1) - sp.eye(A_N,k=1); AL = sp.eye(AL_N)*2 - sp.eye(AL_N, k=-1) - sp.eye(AL_N,k=1)

A_d = A.to_dense(); AL_d = AL.to_dense()
A_c = A.to(gpu); AL_c = AL.to(gpu)

def test_identity():
    it = 20
    for i in range(it):
        Nr = torch.randint(10, 1000)
        Nc = torch.randint(5, 15)
        X = torch.randn(Nr, Nc)
        X_c = X.to(gpu)

        I = sp.eye(Nr)
        I_c = I.to(gpu)

        assert(torch.allclose(X, I @ X))
        assert(torch.allclose(X_c, I_c @ X))

def test_random_small():
    it = 10
    for i in range(it):
        Nc = torch.randint(3, 10)
        X = torch.randn(A_N, Nc)
        X_c = X.to(gpu)

        AX_d = A_d @ X

        assert(torch.allclose(AX_d, A@X))
        assert(torch.allclose(AX_d, (A_c@X_c).cpu()))

def test_random_large():
    it = 10
    for i in range(it):
        Nc = torch.randint(10, 20)
        X = torch.randn(AL_N, Nc)
        X_c = X.to(gpu)

        AX_d = AL_d @ X

        assert(torch.allclose(AX_d, AL@X))
        assert(torch.allclose(AX_d, (AL_c@X_c).cpu()))

def test_backward_grad_A():
    # grad_A := (grad_C * B^T) (*) mask(A)
    #  => gA_{ij} = \sum_k gC_{ik} b_{jk}

    it = 10
    for i in range(it):
        Nc = torch.randint(10, 30)
        X = torch.randn(AL_N, Nc)
        X_c = X.to(gpu)

        grad_C = torch.randn(AL_N, Nc)
        grad_C_c = grad_C.to(gpu)

        Ag = AL.copy()
        Ag.requires_grad = True

        Ag_c = AL_c.copy()
        Ag_c.requires_grad = True

        ((Ag@x) * grad_C).sum().backward()
        ((Ag_c@x_c) * grad_C_c).sum().backward()

        assert(torch.allclose(Ag.data.grad, Ag_c.data.grad.cpu()))

        # make sure gradient entries are what we expect
        for i in range(AL_N):
            for i_i in range(Ag.indptr[i], Ag.indptr[i + 1]):
                j = Ag.indices[i_i]
                gA_ij = grad_C[i] @ X[j]
                assert(abs(Ag.data.grad[i_i] - gA_ij) < 1e-5)
