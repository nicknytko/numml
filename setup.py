from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os

def find_cpp_files(directory, allow_cuda):
    files = os.listdir(directory)
    out = []
    for f in files:
        if f.endswith('.cpp') or (allow_cuda and f.endswith('.cu')):
            out.append(os.path.join(directory, f))
        elif os.path.isdir(f):
            out = out + find_cpp_files(os.path.join(directory, f))
    return out

# Detect if we have cuda installed and compile accordingly
native_ext = None

if 'CUDA_HOME' in os.environ or 'CUDA_PATH' in os.environ:
    print('Detected CUDA, compiling with CUDA acceleration...')
    native_ext = CUDAExtension(name='numml_torch_cpp',
                               sources=find_cpp_files('cpp', allow_cuda=True),
                               include_dirs=[
                                   os.path.join(os.getcwd(), 'ext/cuCollections/include')
                               ],
                               extra_compile_args={
                                   'nvcc': ['-std=c++17'],
                                   'cxx': ['-DCUDA_ENABLED=1', '-O2', '-std=c++17', '-w'],
                               })

else:
    print('No CUDA detected, compiling CPU implementation only...')
    native_ext = CppExtension(name='numml_torch_cpp',
                           sources=find_cpp_files('cpp', allow_cuda=False),
                           extra_compile_args=[
                               '-std=c++17',
                               '-DCUDA_ENABLED=0',
                               '-O2'
                           ])

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
