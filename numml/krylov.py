import torch
import torch.linalg as tla
import numml.sparse as sp


def conjugate_residual(A, b, x=None, M=None, rtol=1e-6, iterations=None):
    '''
    Solves the matrix equation

    Ax = b

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
    M : numml.sparse.SparseCSRTensor or numml.sparse.LinearOperator
      Preconditioner, if it exists.  If not given this will behave like the identity.
      Note we have the condition that M must be symmetric, positive definite (SPD)
    rtol : float
      Relative tolerance for stopping condition.  Will terminate the algorithm when
      ||b - Ax|| / ||b|| <= rtol

    Returns
    -------
    x_sol : torch.Tensor
      Approximate solution to the matrix equation.
    res_hist : list of torch.Tensor
      Norm of the residual at each iteration, including before the first iteration.
    '''

    assert(A.shape[0] == A.shape[1])

    r = None
    if x is None:
        x = torch.zeros(A.shape[1], device=b.device)
        r = b.clone()
    else:
        r = b - A @ x

    if M is None:
        # No preconditioner means we use the identity
        M = sp.LinearOperator(A.shape, lambda x: x.clone(), lambda x: x.clone())

    r = M @ r
    p = r.clone()
    Ar = A @ p
    Ap = Ar.clone()

    nrm_b = tla.norm(b)
    it = 0
    res_hist = [tla.norm(r)]

    while tla.norm(r) / nrm_b > rtol:
        MAp = M@Ap
        alpha = (r @ Ar) / (Ap @ MAp)
        x = x + alpha * p

        r_new = r - alpha * MAp
        Ar_new = A @ r_new
        beta = (r_new @ Ar_new) / (r @ Ar)
        p = r_new + beta * p

        Ap = Ar_new + beta * Ap
        Ar = Ar_new
        r = r_new

        res_hist.append(tla.norm(r))

        it += 1
        if iterations is not None and it >= iterations:
            break

    return x, res_hist


def conjugate_gradient(A, b, x=None, M=None, rtol=1e-6, iterations=None):
    '''
    Solves the matrix equation

    Ax = b

    for symmetric positive definite (SPD) A, using the method of
    conjugate gradients (Saad 2003)

    Parameters
    ----------
    A : numml.sparse.SparseCSRTensor or numml.sparse.LinearOperator
      System matrix
    b : torch.Tensor
      Right-hand-side vector
    x : torch.Tensor
      Initial guess to the solution.  If not given, will default to zero.
    M : numml.sparse.SparseCSRTensor or numml.sparse.LinearOperator
      Preconditioner, if it exists.  If not given this will behave like the identity.
      This should also be SPD.
    rtol : float
      Relative tolerance for stopping condition.  Will terminate the algorithm when
      ||b - Ax|| / ||b|| <= rtol

    Returns
    -------
    x_sol : torch.Tensor
      Approximate solution to the matrix equation.
    res_hist : list of torch.Tensor
      Norm of the residual at each iteration, including before the first iteration.
    '''

    assert(A.shape[0] == A.shape[1])

    r = None
    if x is None:
        x = torch.zeros(A.shape[1], device=b.device)
        r = b.clone()
    else:
        r = b - A @ x

    if M is None:
        # No preconditioner means we use the identity
        M = sp.LinearOperator(A.shape, rm=lambda x: x.clone(), lm=lambda x: x.clone())

    z = M @ r
    p = z.clone()
    norm_b = tla.norm(b)
    it = 0
    norm_r = tla.norm(r)
    res_hist = [norm_r]

    while norm_r / norm_b > rtol:
        Ap = A@p
        rz = r@z
        alpha = rz/(Ap@p)
        x = x + alpha * p
        r = r - alpha * Ap
        z = M@r
        beta = (r@z) / rz
        p = z + beta * p

        norm_r = tla.norm(r)
        if torch.any(torch.isnan(norm_r)):
            # if we have NaN r norm then we likely aren't converging, return early
            return x, torch.stack(res_hist)

        res_hist.append(norm_r)
        it += 1
        if iterations is not None and it >= iterations:
            break

    return x, torch.stack(res_hist)
