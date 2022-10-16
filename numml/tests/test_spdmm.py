import torch
import torch.linalg as tla
import numml.sparse as sp
import numml.utils as utils
import pytest
import random
import common

gpu = torch.device('cuda:0')

# The large matrix should be big enough to require multiple CUDA warps
A_N = 16; AL_N = 2048
A = sp.eye(A_N)*2 - sp.eye(A_N, k=-1) - sp.eye(A_N,k=1); AL = sp.eye(AL_N)*2 - sp.eye(AL_N, k=-1) - sp.eye(AL_N,k=1)

A_d = A.to_dense(); AL_d = AL.to_dense()
A_c = A.to(gpu); AL_c = AL.to(gpu)

def test_identity():
    it = 20
    for i in range(it):
        Nr = random.randint(10, 1000)
        Nc = random.randint(5, 15)
        X = torch.randn(Nr, Nc)
        X_c = X.to(gpu)

        I = sp.eye(Nr)
        I_c = I.to(gpu)

        assert(torch.allclose(X, I @ X))
        assert(torch.allclose(X_c, I_c @ X_c))

def test_random_small():
    it = 10
    for i in range(it):
        Nc = random.randint(3, 10)
        X = torch.randn(A_N, Nc)
        X_c = X.to(gpu)
        print('X_c shape', X_c.shape)
        print(X_c)

        AX_d = A_d @ X

        assert(torch.allclose(AX_d, A@X))
        assert(torch.allclose(AX_d, (A_c@X_c).cpu()))

def test_random_large():
    it = 5
    for i in range(it):
        Nc = random.randint(3, 6)
        X = torch.randn(AL_N, Nc)
        X_c = X.to(gpu)

        AX_d = AL_d @ X

        assert(torch.allclose(AX_d, AL@X))
        assert(torch.allclose(AX_d, (AL_c@X_c).cpu()))

def test_backward_grad_A():
    # grad_A := (grad_C * B^T) (*) mask(A)
    #  => gA_{ij} = \sum_k gC_{ik} b_{jk}

    it = 5
    for i in range(it):
        Nc = random.randint(3, 6)
        X = torch.randn(AL_N, Nc)
        X_c = X.to(gpu)

        grad_C = torch.randn(AL_N, Nc)
        grad_C_c = grad_C.to(gpu)

        Ag = AL.copy()
        Ag.requires_grad = True

        Ag_c = AL_c.copy()
        Ag_c.requires_grad = True

        ((Ag@X) * grad_C).sum().backward()
        ((Ag_c@X_c) * grad_C_c).sum().backward()

        print(Ag.data.grad.to_dense())
        print(Ag_c.data.grad.to_dense())
        print('abserr', tla.norm(Ag.data.grad.to(gpu) - Ag_c.data.grad))
        assert(common.relerr(Ag.data.grad, Ag_c.data.grad.cpu()) < 1e-6)

        # make sure gradient entries are what we expect
        for i in range(AL_N):
            for i_i in range(Ag.indptr[i], Ag.indptr[i + 1]):
                j = Ag.indices[i_i]
                gA_ij = grad_C[i] @ X[j]
                assert(abs(Ag.data.grad[i_i] - gA_ij) < 1e-5)

def test_backward_grad_B():
    # grad_B := (A^T * grad_C)

    it = 5
    for i in range(it):
        Nc = random.randint(3, 6)
        X = torch.randn(A_N, Nc)
        X_c = X.to(gpu)

        X.requires_grad = True
        X_c.requires_grad = True

        grad_C = torch.randn(A_N, Nc)
        grad_C_c = grad_C.to(gpu)

        ((A@X) * grad_C).sum().backward()
        ((A_c@X_c) * grad_C_c).sum().backward()

        assert(torch.allclose(X.grad, X_c.grad.cpu()))

        grad_B_d = A_d.T @ grad_C
        assert(torch.allclose(grad_B_d, X.grad))
