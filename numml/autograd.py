import torch
import torch.autograd
import numml.sparse as sp
import traceback


def clean_grads(tup):
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


def vjp_single(f, inputs, d_idx, v, cleaned=False):
    '''
    Computes the vector-Jacobian product of a single input.  Will gracefully
    handle cases with non-tensor inputs and use of the sparse csr wrapper.

    Parameters
    ----------
    f : function
      Function for which the VJP should be computed.
    inputs : iterable
      Iterable of inputs to pass the function
    d_idx : integer
      Integer in the range [0, len(inputs)) that indicates which argument
      to take gradients with respect to.
    v : torch.Tensor
      Vector v in the VJP
    cleaned : boolean
      Should be set to true if inputs[d_idx] requires gradient and the gradient
      has already been zeroed.

    Returns
    -------
    vJ : torch.Tensor
      The vector v right-multiplied by the Jacobian.
    '''

    grad = None
    if not cleaned:
        cleaned_inputs = clean_grads(inputs)
        cleaned_inputs[d_idx].requires_grad = True
    else:
        cleaned_inputs = inputs

    with torch.enable_grad():
        if not cleaned_inputs[d_idx].requires_grad:
            return None

        if isinstance(cleaned_inputs[d_idx], sp.SparseCSRTensor):
            inpt = cleaned_inputs[d_idx]
            grad_data = torch.autograd.grad(f(*cleaned_inputs), inpt.data, grad_outputs=v)[0]
            grad = sp.SparseCSRTensor((grad_data, inpt.indices, inpt.indptr), inpt.shape)
        else:
            grad = torch.autograd.grad(f(*cleaned_inputs), cleaned_inputs[d_idx], grad_outputs=v)[0]

    return grad


def vjp(f, inputs, v):
    '''
    Computes vector-Jacobian products of a function, for each input that is differentiable.

    Parameters
    ----------
    f : function
      Function for which the VJP should be computed.
    inputs : iterable
      Iterable of inputs to pass the function
    v : torch.Tensor
      Vector v in the VJP.

    Returns
    -------
    vJs : iterable of torch.Tensor
      The vector v right-multiplied by each Jacobian.
      A value of None is used for non-differentiable inputs.
    '''

    grads = [None] * len(inputs)
    cleaned_inputs = clean_grads(inputs)
    for i in range(len(inputs)):
        if isinstance(inputs[i], torch.Tensor):
            grads[i] = vjp_single(f, cleaned_inputs, i, v, cleaned=True)
    return tuple(grads)
