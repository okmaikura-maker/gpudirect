# gpudirect

Call the **GPU — and the iGPU — directly** from Python, with **zero pip dependencies**
and **no CUDA Toolkit**. It drives the driver's own DLLs (`nvcuda.dll` for CUDA,
`OpenCL.dll` for everything incl. integrated GPUs) straight through `ctypes`, runs
**hand-written PTX / OpenCL kernels**, and even trains a small Transformer on the GPU.

Verified on Windows 10, Python 3.10, NVIDIA GTX 1650 + Intel HD Graphics 530.

## Install (local)

```
pip install gpudirect-0.1.0-py3-none-any.whl
```

or run the MSI (`gpudirect-0.1.0.msi`) — it installs the package into your Python.

`numpy` is optional (needed only for `fastnumpy` and the AI/training demos):

```
pip install numpy
```

## One line: saturate every GPU + iGPU

```
python -m gpudirect.turbo          # max out all GPUs/iGPUs for 10s
```
```python
import gpudirect.turbo as turbo
turbo.saturate()                   # same, from Python
```

## Run your own kernel (general purpose)

```python
import numpy as np, gpudirect.easy as ge
g = ge.GPU()
a = g.to_gpu(np.arange(8, dtype=np.float32))
out = g.empty(8, np.float32)
k = g.kernel(PTX_SOURCE, "my_kernel")     # JIT hand-written PTX
k(grid=(1,1,1), block=(8,1,1), args=[a, out, 8])
print(out.get())
```

## numpy-style on the GPU

```python
import fastnumpy as fnp
c = (fnp.array(A) @ fnp.array(B) + fnp.array(A) * 2.0).relu()
print(c.numpy())
```

## What's inside

| module | what it does |
|---|---|
| `gpudirect` | `nvcuda.dll` (CUDA Driver API) via ctypes: device, memory, PTX/cubin, launch |
| `gpudirect.easy` | general-purpose `GPU` / `GpuArray` / `Kernel` facade |
| `gpudirect.opencl` | `OpenCL.dll` via ctypes: enumerate & saturate any device incl. iGPU |
| `gpudirect.turbo` | one-line saturation of all GPUs/iGPUs |
| `gpudirect.transformer` / `train_gpt` | GPT inference **and training**, all in hand-written PTX |
| `fastnumpy` | numpy-style array ops (`+ - * / @`, relu) on the GPU |

## Notes

- CUDA path is NVIDIA-only; iGPU / other vendors go through OpenCL.
- `turbo` is a saturation tool (benchmark / stress / thermal). It runs the GPU
  flat out — mind heat and power, especially on laptops. `Ctrl+C` stops it.
- No CUDA Toolkit, CuPy, PyTorch, or pyopencl required — only vendor drivers.

MIT License.
