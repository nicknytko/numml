numml: sparse differentiable matrix operations for pytorch

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
