#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                "cxx": ['-g', '-std=c++14'],
                "nvcc": [
                    "-g", "-G", "-std=c++14",
                    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                    "--expt-relaxed-constexpr",
                    "--use_fast_math",
                    "-Xcompiler", "-std=c++14"
                ]
            })
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
