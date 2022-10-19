import torch
import torch.nn
import numml.sparse as sp


class GCNConv(torch.nn.Module):
    '''
    A spectral graph convolution operator based on Kipf & Welling's
    GCN layer.  This operates directly on matrices instead of using
    a graph abstraction.
    '''

    def __init__(self, in_channels, out_channels, normalize=True):
        '''
        Initializes the layer.

        Parameters
        ----------
        in_channels : integer
          Number of input node features
        out_channels : integer
          Number of output node features
        normalize : bool
          When performing the layer update, do we normalize like
          A \gets \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}
        '''

        super().__init__()

        self.weights = torch.nn.Parameter(torch.randn(in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.randn(out_channels))
        self.normalize = normalize
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, A, X):
        '''
        Performs a forward pass.

        Parameters
        ----------
        A : numml.sparse.SparseCSRTensor or numml.sparse.LinearOperator
          Input "graph"
        X : torch.Tensor
          Node features to convolve

        Returns
        -------
        Y : torch.Tensor
          Output convolved node features
        '''

        if len(X.shape) == 1:
            X = X.unsqueeze(1)

        if self.normalize:
            D = (A.row_sum() + 1.) ** -0.5
            XTheta = X @ self.weights
            Xprime = D[:, None] * (A @ (D[:, None] * XTheta))
        else:
            Xprime = A @ (X @ self.weights)

        return Xprime + self.bias


class TAGConv(torch.nn.Module):
    '''
    A spectral graph convolution operator from "Topology Adaptive Graph
    Convolutional Networks", where the output is a polynomial of the adj
    matrix.  This operates directly on matrices instead of using
    a graph abstraction.
    '''

    def __init__(self, in_channels, out_channels, k=2, normalize=True):
        '''
        Initializes the layer.

        Parameters
        ----------
        in_channels : integer
          Number of input node features
        out_channels : integer
          Number of output node features
        k : integer
          Degree of the polynomial, i.e., "number of hops".
        normalize : bool
          When performing the layer update, do we normalize like
          A \gets \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}
        '''

        super().__init__()

        self.k = k
        self.weights = torch.nn.Parameter(torch.randn(k, in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.randn(out_channels))
        self.normalize = normalize
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, A, X):
        '''
        Performs a forward pass.

        Parameters
        ----------
        A : numml.sparse.SparseCSRTensor or numml.sparse.LinearOperator
          Input "graph"
        X : torch.Tensor
          Node features to convolve

        Returns
        -------
        Y : torch.Tensor
          Output convolved node features
        '''

        if len(X.shape) == 1:
            X = X.unsqueeze(1)

        Xc = X
        H = torch.zeros(X.shape[0], self.out_channels, device=X.device)
        if self.normalize:
            D = (A.row_sum() + 1.) ** -0.5

        for k in range(self.k):
            # perform A @ X
            if self.normalize:
                Xc = D[:, None] * (A @ (D[:, None] * Xc))
            else:
                Xc = A @ Xc

            # sum over (A@X@Theta_k)
            H = H + (Xc @ self.weights[k])

        return H + self.bias


class SeqWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, A, x):
        return self.module(x)


class Sequential(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.modlist = torch.nn.ModuleList(list(args))

    def forward(self, A, X):
        for layer in self.modlist:
            X = layer(A, X)
        return torch.squeeze(X)

    import numpy as np


def model_to_vector(model):
    vec_size = 0
    state_dict = model.state_dict()

    # Scan state dictionary to determine output vector length
    for weight in model.state_dict().values():
        vec_size += torch.prod(torch.tensor(weight.detach().shape))
    weights_vector = torch.zeros(vec_size)

    # Finally, populate vector
    cur_spot = 0
    for name, weights in state_dict.items():
        vector = weights.detach().flatten()
        n = len(vector)

        weights_vector[cur_spot:cur_spot+n] = vector
        cur_spot += n

    return weights_vector

# https://discuss.pytorch.org/t/combining-two-models/77716/6
def get_last_module(model, indices):
    mod = model
    for idx in indices:
        mod = getattr(mod, idx)
    return mod


def vector_to_model(model, weights_vector):
    pointer = 0
    model_params = list(model.named_parameters())

    for name, p in model_params:
        indices = name.split(".")
        mod = get_last_module(model, indices[:-1])
        p_name = indices[-1]
        if isinstance(p, torch.nn.Parameter):
            delattr(mod, p_name)

        num_param = p.numel()
        setattr(mod, p_name, weights_vector[pointer:(pointer + num_param)].view_as(p))
        pointer += num_param
