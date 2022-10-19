import torch
import torch.autograd
import torch.autograd.functional
import torch.linalg as tla
import numml.sparse as sp


class FixedPointIteration(torch.autograd.Function):
    '''
    An adjoint solver for the fixed point iteration

    \[x^{(k+1)} = f(x^{(k)}, \theta_1, \theta_2, \ldots, \theta_n)\],

    where $\theta_1, \theta_2, \ldots, \theta_n$ are differentiable parameters
    and x^({k)} is a function of all $theta_i$ for $k>0$.  This computes the
    vector-jacobian product like

    \[ w^T dx^{*}/d_{theta_i} = w^T (I - df/dx)^{-1} df/d_{theta_i}, \]

    where $w^T (I - df/dx)^{-1}$ is itself solved by a fixed-point iteration.

    Follows the derivation from https://implicit-layers-tutorial.org/implicit_functions/
    '''

    @staticmethod
    def _clean_grads(tup):
        def clean(t):
            if isinstance(t, torch.Tensor):
                tc = t.detach()
                tc.requires_grad = t.requires_grad
                return tc
            elif isinstance(t, sp.SparseCSRTensor):
                Tc = t.detach()
                Tc.requires_grad = t.requires_grad
                return Tc
            else:
                return t
        return tuple(map(clean, tup))

    @staticmethod
    def _vjp_single(f, inputs, d_idx, v):
        grad = None
        with torch.enable_grad():
            try:
                grad = torch.autograd.grad(f(*inputs), inputs[d_idx], grad_outputs=v)[0]
            except Exception as e:
                pass # Don't require grad, return None
        return grad

    @staticmethod
    def _vjp(f, inputs, v):
        grads = [None] * len(inputs)
        cleaned_inputs = FixedPointIteration._clean_grads(inputs)
        for i in range(len(inputs)):
            if isinstance(inputs[i], torch.Tensor):
                grads[i] = FixedPointIteration._vjp_single(f, cleaned_inputs, i, v)
        return grads

    @staticmethod
    def forward(ctx, max_diff_tol, max_iter, f, x, *fargs):
        assert(max_diff_tol is not None or max_iter is not None)

        x_orig = x.clone().detach()
        it = 0

        with torch.no_grad():
            while True:
                xold = x
                x = f(x, *fargs)
                diff = tla.norm(xold - x)
                it += 1

                if (max_iter is not None and
                    it >= max_iter):
                    break
                if (max_diff_tol is not None and
                    diff >= max_diff):
                    break

        ctx.save_for_backward(x)
        ctx.fargs = fargs
        ctx.f = f
        return x

    @staticmethod
    def backward(ctx, grad_x_star):
        (x_star, ) = ctx.saved_tensors
        fargs = ctx.fargs
        f = ctx.f

        # Fixed point iteration to find w^T = z^T + w^T df/dx
        # => w^T = v^T(I - df/dx)^{-1}
        # => (I - df/dx)^T w = v
        w = grad_x_star.detach().clone()
        it = 0

        while True:
            wTdfdx = FixedPointIteration._vjp_single(f, (x_star, *fargs), 0, w)

            w_n = grad_x_star + wTdfdx
            diff_w = tla.norm(w_n-w)
            w = w_n
            if diff_w < 1e-4:
                break

            it += 1
            if it > 300:
                # no convergence, return zero grad
                w = torch.zeros_like(grad_x_star)
                break

        # Now, return w^T df/dtheta = z(I - df/dx)^{-1} df/dtheta
        grads = FixedPointIteration._vjp(f, (x_star, *fargs), w)

        return (None,   # f
                None,   # max_diff_tol
                None,   # max_iter
                *grads) # x, *fargs

def fp_wrapper(f, x, *fargs, max_diff_tol=None, max_iter=None):
    return FixedPointIteration.apply(max_diff_tol, max_iter, f, x, *fargs)
