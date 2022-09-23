class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return 2. * x

    @staticmethod
    def backward(ctx, grad_x):
        return 2. * grad_x
