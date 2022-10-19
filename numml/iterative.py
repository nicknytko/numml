import torch
import torch.autograd
import torch.autograd.functional
import torch.linalg as tla


class FixedPointIteration(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, farg, x, max_diff_tol=1e-4):
        x_orig = x.clone().detach()
        diff = torch.inf
        while diff > max_diff_tol:
            xold = x
            x = f(farg, x)
            diff = tla.norm(xold-x)
        ctx.save_for_backward(farg, x_orig, x)
        ctx.f = f
        return x

    @staticmethod
    def backward(ctx, grad_x_star):
        farg, x_orig, x_star = ctx.saved_tensors
        f = ctx.f

        print('fp backward')
        print(farg)
        #dfdx = torch.autograd.functional.jacobian(f, (farg, x_orig), create_graph=True)
        #print(dfdx)
        dfdx = torch.empty_like(farg)
        x_g = x_star.clone()
        x_g.requires_grad = True
        print(x_g)
        for i in range(dfdx.shape[0]):
            e = torch.zeros(dfdx.shape[1])
            e[i] = 1.
            with torch.enable_grad():
                out_i = f(farg, x_g)
                dfdx[i] = torch.autograd.grad(out_i, x_g, e, retain_graph=True, create_graph=True)[0]
        print(dfdx)
        return (None, None, None, None)
