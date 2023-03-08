from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os
import platform
import subprocess

def find_cpp_files(directory, allow_cuda):
    files = os.listdir(directory)
    out = []
    for f in files:
        if f.endswith('.cpp') or (allow_cuda and f.endswith('.cu')):
            out.append(os.path.join(directory, f))
        elif os.path.isdir(f):
            out = out + find_cpp_files(os.path.join(directory, f))
    return out

# Native code for sparse CSR implementations
# Detect if we have cuda installed and compile accordingly
native_ext = None

cxx_args = [
    '-O2',
    '-std=c++17',
    '-w'
]
libraries = []

# Platform specific build options
has_superlu = False

if platform.system() == 'Darwin':
    macos_include_path = os.path.join('/', 'opt', 'homebrew', 'include')
    cxx_args.append(f'-I{macos_include_path}')
    has_superlu = os.path.exists(os.path.join(macos_include_path, 'supermatrix.h'))

elif platform.system() == 'Linux':
    cpp_verbose_output = subprocess.run(['cpp', '-v'], input='', capture_output=True).stderr.decode('utf-8').split('\n')
    start_search_delim = '#include <...> search starts here:'
    end_search_delim = 'End of search list.'

    start_idx = cpp_verbose_output.index(start_search_delim)
    end_idx = cpp_verbose_output.index(end_search_delim)
    if end_idx <= start_idx + 1:
        raise RuntimeError('cpp: no library search directories found?')

    search_paths = cpp_verbose_output[(start_idx + 1):end_idx]
    for path in search_paths:
        path = path.strip()

        if os.path.exists(os.path.join(path, 'supermatrix.h')):
            has_superlu = True
            print(f'Found SuperLU library headers at {path}')
            break
        elif os.path.exists(os.path.join(path, 'superlu')):
            has_superlu = True
            lib_path = os.path.join(path, 'superlu')
            print(f'Found SuperLU library headers at {lib_path}')
            cxx_args.append(f'-I{lib_path}')
            break
else:
    pass # ¯\_(ツ)_/¯

# Library specific options
if not has_superlu:
    print('No SuperLU found on this system... using built-in LU factorization.')
    cxx_args.append('-DSUPERLU_ENABLED=0')
else:
    cxx_args.append('-DSUPERLU_ENABLED=1')
    libraries.append('superlu')

# CUDA compilation options
if 'CUDA_HOME' in os.environ or 'CUDA_PATH' in os.environ:
    print('Detected CUDA, compiling with CUDA acceleration...')
    cxx_args.append('-DCUDA_ENABLED=1')

    native_ext = CUDAExtension(name='numml_torch_cpp',
                               sources=find_cpp_files('cpp', allow_cuda=True),
                               include_dirs=[
                                   os.path.join(os.getcwd(), 'ext/cuCollections/include')
                               ],
                               extra_compile_args={
                                   'nvcc': ['-std=c++17'],
                                   'cxx': cxx_args
                               },
                               libraries=libraries)

else:
    print('No CUDA detected, compiling CPU implementation only...')
    cxx_args.append('-DCUDA_ENABLED=0')

    native_ext = CppExtension(name='numml_torch_cpp',
                              sources=find_cpp_files('cpp', allow_cuda=False),
                              extra_compile_args=cxx_args,
                              libraries=libraries)

print('Compiling with CXX options', cxx_args)

setup(name='numml',
      version='0.0.1',
      ext_modules=[native_ext],
      cmdclass={
          'build_ext': BuildExtension
      },
      author='Nicolas Nytko',
      author_email='nnytko2@illinois.edu',
      packages=['numml', 'numml.sparse']
)
