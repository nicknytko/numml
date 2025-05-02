## numml: Differentiable numerics for PyTorch

A library for PyTorch providing sparse, differentiable CSR support.

### Prerequisites

- PyTorch 2.0+
- For CUDA acceleration, an Nvidia GPU that supports at least `sm_60` (Pascal) architecture.

### Installation

Clone normally and install with pip,
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
@misc{nytko2023optimized,
      title={Optimized Sparse Matrix Operations for Reverse Mode Automatic Differentiation}, 
      author={Nicolas Nytko and Ali Taghibakhshi and Tareq Uz Zaman and Scott MacLachlan and Luke N. Olson and Matt West},
      year={2023},
      eprint={2212.05159},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
