## numml: Differentiable numerics for PyTorch

A library for PyTorch providing sparse, differentiable CSR support.

### Installation

There are submodules needed for CUDA library dependencies, clone this repo with
```
git clone --recurse-submodules https://github.com/nicknytko/numml
```

then, install with
```
pip3 install .
```

If CUDA is not detected on your system, this will silently default to compiling only CPU
implementations: you can run pip with verbose (`-v`) for a sanity check on this.

### Tests

Run tests using pytest like
```
pytest numml/tests
```

Note that the test cases will assume you are running on a machine with CUDA installed and you have compiled with CUDA support.

### Citing

[Optimized Sparse Matrix Operations for Reverse Mode Automatic Differentiation](https://arxiv.org/abs/2212.05159)
```
@misc{NytkoSparse2022,
  doi = {10.48550/ARXIV.2212.05159},
  url = {https://arxiv.org/abs/2212.05159},
  author = {Nytko, Nicolas and Taghibakhshi, Ali and Zaman, Tareq Uz and MacLachlan, Scott and Olson, Luke N. and West, Matt},
  keywords = {Machine Learning (cs.LG), Mathematical Software (cs.MS), Numerical Analysis (math.NA), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Mathematics, FOS: Mathematics},
  title = {Optimized Sparse Matrix Operations for Reverse Mode Automatic Differentiation},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
