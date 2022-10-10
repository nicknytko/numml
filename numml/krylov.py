import torch
import torch.linalg as tla
import numml.sparse as sp


def conjugate_residual(A, b, x=None, rtol=1e-6):
    '''
    Solves the matrix equation

    Ax =

    for indefinite, symmetric A using the method of
    Conjugate Residuals (Saad 2003)

    Parameters
    ----------
    A : numml.sparse.SparseCSRTensor or numml.sparse.LinearOperator
      System matrix
    b : torch.Tensor
      Right-hand-side vector
    x : torch.Tensor
      Initial guess to the solution.  If not given, will default to zero.
    rtol : float
      Relative tolerance for stopping condition.  Will terminate the algorithm when
      ||b - Ax|| / ||b|| <= rtol

    Returns
    -------
    x_sol : torch.Tensor
      Approximate solution to the matrix equation.
    '''

    assert(A.shape[0] == A.shape[1])

    r = None
    if x is None:
        x = torch.zeros(A.shape[1])
        r = b.copy()
    else:
        r = b - A @ x

    p = r.copy()
    Ar = A @ p
    Ap = Ar.copy()

    nrm_b = tla.norm(b)

    while tla.norm(r) / nrm_b > rtol:
        alpha = (r @ Ar) / (Ap @ Ap)
        x = x + alpha * p

        r_new = r - alpha * Ap
        Ar_new = A @ r
        beta = (r_new @ Ar_new) / (r @ Ar)
        p = r_new + beta * p

        Ap = Ar_new + beta * Ap
        Ar = Ar_new
        r = r_new

    return x
