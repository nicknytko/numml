## numml: Differentiable numerics for PyTorch

A library for PyTorch providing sparse, differentiable CSR support.

### Prerequisites

- PyTorch 2.0+
- For CUDA acceleration, an Nvidia GPU that supports at least `sm_60` (Pascal) architecture.

### Installation

Clone normally and install with pip,
```
pip3 install . --no-build-isolation
```

If CUDA is not detected on your system, this will silently default to compiling only CPU
implementations: you can run pip with verbose (`-v`) for a sanity check on this.

It is suggested to use `--no-build-isolation` as the build process will default to pulling the
regular `torch` PyPI package.  This may be incorrect if you have, for example, installed torch
with support for a specific version of CUDA.

### Tests

Run tests using pytest like
```
pytest numml/tests
```

Note that the test cases will assume you are running on a machine with CUDA installed and you have compiled with CUDA support.

### Citing

[Optimized Sparse Matrix Operations for Reverse Mode Automatic Differentiation](https://doi.org/10.1137/24M1711832) [(arxiv preprint)](https://arxiv.org/abs/2212.05159)
```
@misc{nytko2025optimized,
    author = {Nytko, Nicolas and Taghibakhshi, Ali and Zaman, Tareq Uz and MacLachlan, Scott and Olson, Luke N. and West, Matthew},
    title = {Optimized Sparse Matrix Operations for Reverse Mode Automatic Differentiation},
    journal = {SIAM Journal on Scientific Computing},
    volume = {47},
    number = {5},
    pages = {C1115-C1143},
    year = {2025},
    doi = {10.1137/24M1711832},
    URL = {https://doi.org/10.1137/24M1711832}
}
```
