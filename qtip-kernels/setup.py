from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="qtip_kernels_cuda",
    ext_modules=[
        CUDAExtension(
            name="qtip_kernels",
            sources=[
                "src/wrapper.cpp",
                "src/inference.cu",
                "src/qtip_torch.cu"
            ],
            extra_compile_args={
                "cxx": ["-O3", "--fast-math", "-lineinfo", "-std=c++17"],
                "nvcc": [
                    "-O3", "--use_fast_math", "-lineinfo", "-keep",
                    "-std=c++17",
                    # A100-specific: SASS for sm_80 + PTX fallback for newer archs.
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_80,code=compute_80",
                    # ptxas: register usage + spill/lmem diagnostics.
                    "-Xptxas=-v,-warn-spills,-warn-lmem-usage",
                    "--expt-relaxed-constexpr",
                ]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)