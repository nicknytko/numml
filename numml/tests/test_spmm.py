import torch
import torch.linalg as tla
import numml.sparse as sp
import pytest

# gpu = torch.device('cuda:0')

# # The large matrix should be big enough to require multiple CUDA warps
# A_N = 16; AL_N = 2048
# A = sp.eye(A_N)*2 - sp.eye(A_N, k=-1) - sp.eye(A_N,k=1); AL = sp.eye(AL_N)*2 - sp.eye(AL_N, k=-1) - sp.eye(AL_N,k=1)
# A_d = A.to_dense(); AL_d = AL.to_dense()
# A_c = A.to(gpu); AL_c = AL.to(gpu)

# def test_zeros():
#     assert torch.all((A @ torch.zeros(A_N)) == 0)
#     assert torch.all((A_c @ torch.zeros(A_N, device=gpu)) == 0)

# def relerr(x, xhat):
#     return tla.norm(x-xhat)/tla.norm(x)

# def test_elementary():
#     for i in range(A_N):
#         e = torch.zeros(A_N)
#         e[i] = 1.
#         e_c = e.to(gpu)

#         # Ae should give the respective column back
#         Ae = A @ e
#         Ae_c = A_c @ e_c

#         assert(torch.allclose(Ae, Ae_c.cpu()))
#         assert(Ae[i] == 2)
#         assert(Ae_c[i] == 2)
#         if i > 0:
#             assert(Ae[i-1] == -1)
#             assert(Ae_c[i-1] == -1)
#         if i < A_N - 1:
#             assert(Ae[i+1] == -1)
#             assert(Ae_c[i+1] == -1)

# def test_random():
#     it = 50
#     for i in range(it):
#         b = torch.rand(A_N)
#         b_c = b.to(gpu)

#         x_d = A_d @ b

#         assert(relerr(x_d, A@b) < 1e-7)
#         assert(relerr(x_d, (A_c@b_c).cpu()) < 1e-7)

# def test_random_large():
#     it = 50
#     for i in range(it):
#         b = torch.rand(AL_N)
#         b_c = b.to(gpu)

#         x_d = AL_d @ b

#         assert(relerr(x_d, AL@b) < 1e-7)
#         assert(relerr(x_d, (AL_c@b_c).cpu()) < 1e-7)
