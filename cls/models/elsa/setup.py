"""
This file is modified from DDFNet: https://github.com/thefoxofsky/ddfnet
"""
import os
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def make_cuda_ext(name, sources, sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    return extension(
        name=f'{name}',
        sources=sources,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


if __name__ == '__main__':
    setup(
        name='elsa',
        version=1.0,
        description='Enhanced Local Self-Attention',
        ext_modules=[
            make_cuda_ext(
                name='elsa_ext',
                sources=['src/elsa_ext.cpp'],
                sources_cuda=[
                    'src/cuda/elsa_cuda.cpp',
                    'src/cuda/elsa_cuda_kernel.cu'
                ]),
            make_cuda_ext(
                name='elsa_faster_ext',
                sources=['src/elsa_faster_ext.cpp'],
                sources_cuda=[
                    'src/cuda/elsa_faster_cuda.cpp',
                    'src/cuda/elsa_faster_cuda_kernel.cu'
                ])
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
