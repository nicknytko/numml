from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

def find_cpp_files(directory):
    files = os.listdir(directory)
    out = []
    for f in files:
        if f.endswith('.cpp'):
            out.append(os.path.join(directory, f))
        elif os.is_dir(f):
            out = out + find_cpp_files(os.path.join(directory, f))
    return out

setup(name='numml',
      version='0.0.1',
      ext_modules=[
          cpp_extension.CppExtension('numml_torch_cpp', find_cpp_files('cpp')),
      ],
      cmdclass={
          'build_ext': cpp_extension.BuildExtension
      },
      author='Nicolas Nytko',
      author_email='nnytko2@illinois.edu',
      packages=['numml', 'numml.torch']
)
