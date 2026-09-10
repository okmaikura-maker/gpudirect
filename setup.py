# distutils bdist_msi 用の薄いラッパ(pyproject と同じ内容)。
from setuptools import setup
setup(
    name="gpudirect",
    version="0.2.0",
    description="Direct GPU/iGPU from Python via nvcuda.dll / OpenCL.dll (no CUDA Toolkit).",
    packages=["gpudirect"],
    package_data={"gpudirect": ["kernels/*.ptx"]},
    py_modules=["fastnumpy"],
    python_requires=">=3.8",
)
