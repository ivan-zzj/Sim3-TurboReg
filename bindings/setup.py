"""Thin Python extension over the C++/LibTorch implementation (CPU and CUDA tensors)."""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

ROOT = Path(__file__).resolve().parents[1]
sources = [str(ROOT / 'turboreg/src' / name) for name in [
    'turboreg.cpp', 'rigid_transform.cpp', 'similarity_transform.cpp',
    'core_turboreg_gpu.cpp', 'utils_pcr.cpp', 'model_selection.cpp',
    'scale_turboreg.cpp', 'sim3.cpp',
]] + [str(ROOT / 'bindings/pybinding.cpp')]

# CUDA kernels are optional: CPU-only LibTorch builds retain the existing path.
use_cuda = os.environ.get('TURBOREG_BUILD_CUDA', '0') == '1'
if use_cuda and CUDA_HOME is None:
    raise RuntimeError('TURBOREG_BUILD_CUDA=1 requires a CUDA toolkit matching PyTorch; set CUDA_HOME.')
extension = CUDAExtension if use_cuda else CppExtension
if use_cuda:
    sources.append(str(ROOT / 'turboreg/src/sim3_cuda.cu'))
compile_args = {'cxx': ['-O2', '-std=c++17']}
if use_cuda:
    compile_args['nvcc'] = ['-O2', '-std=c++17', '--fmad=false']

setup(
    name='turboreg_gpu',
    version='1.1.0',
    author='Shaocheng Yan',
    description='C++/LibTorch Sim(3) registration with thin Python bindings',
    ext_modules=[extension(
        name='turboreg_gpu',
        sources=sources,
        include_dirs=[str(ROOT / 'turboreg/include'),
                      os.environ.get('EIGEN3_INCLUDE_DIR', '/usr/include/eigen3')],
        define_macros=[('TURBOREG_WITH_CUDA_KERNELS', '1')] if use_cuda else [],
        extra_compile_args=compile_args,
    )],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch'],
)
