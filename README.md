# gpudirect

**EN** — Call the **GPU — and the iGPU — directly** from Python, with **zero pip
dependencies** and **no CUDA Toolkit**. It drives the driver's own DLLs
(`nvcuda.dll` for CUDA, `OpenCL.dll` for everything incl. integrated GPUs) straight
through `ctypes`, runs **hand-written PTX / OpenCL kernels**, and even trains a
small Transformer on the GPU.

**日本語** — **GPU も iGPU も Python から直接**叩くライブラリ。**pip追加ゼロ・CUDA
Toolkit不要**。ドライバ同梱の DLL(`nvcuda.dll`=CUDA、`OpenCL.dll`=iGPU含む全ベンダ)を
`ctypes` で直に呼び、**手書き PTX / OpenCL カーネル**を実行。小さな Transformer の
GPU 学習までできる。

> Verified on / 動作確認: Windows 10, Python 3.10, NVIDIA GTX 1650 + Intel HD Graphics 530.

---

## Install / インストール

```
pip install gpudirect-0.1.0-py3-none-any.whl
```

**EN** — or run the MSI (`gpudirect-0.1.0.msi`); it installs the package into your
Python. `numpy` is optional (only for `fastnumpy` and the AI/training demos).

**日本語** — もしくは MSI(`gpudirect-0.1.0.msi`)を実行するとローカルの Python に
インストールされる。`numpy` は任意(`fastnumpy` と AI/学習デモでのみ必要)。

```
pip install numpy
```

## One line: saturate every GPU + iGPU / 一行で全 GPU・iGPU を最大飽和

```
python -m gpudirect.turbo          # max out all GPUs/iGPUs / 全 GPU・iGPU をフル稼働
```
```python
import gpudirect.turbo as turbo
turbo.saturate()
```

## Run your own kernel / 自作カーネルを走らせる

**EN** — general-purpose: give PTX, get a callable, pass numpy arrays.
**日本語** — 汎用: PTX を渡す → 関数化 → numpy 配列を渡すだけ。

```python
import numpy as np, gpudirect.easy as ge
g = ge.GPU()
a = g.to_gpu(np.arange(8, dtype=np.float32))
out = g.empty(8, np.float32)
k = g.kernel(PTX_SOURCE, "my_kernel")     # JIT hand-written PTX / 手書きPTXをJIT
k(grid=(1,1,1), block=(8,1,1), args=[a, out, 8])
print(out.get())
```

## numpy-style on the GPU / GPU で numpy 風

```python
import fastnumpy as fnp
c = (fnp.array(A) @ fnp.array(B) + fnp.array(A) * 2.0).relu()
print(c.numpy())
```

## What's inside / 中身

| module | EN | 日本語 |
|---|---|---|
| `gpudirect` | `nvcuda.dll` (CUDA Driver API) via ctypes: device, memory, PTX/cubin, launch | CUDAドライバAPI直叩き:デバイス・メモリ・PTX/cubin・起動 |
| `gpudirect.easy` | general-purpose `GPU` / `GpuArray` / `Kernel` facade | 汎用 `GPU`/`GpuArray`/`Kernel` |
| `gpudirect.opencl` | `OpenCL.dll` via ctypes: enumerate & saturate any device incl. iGPU | OpenCL直叩き:iGPU含む全デバイス列挙・飽和 |
| `gpudirect.turbo` | one-line saturation of all GPUs/iGPUs | 全GPU/iGPUを一行で飽和 |
| `gpudirect.transformer` / `train_gpt` | GPT inference **and training**, all hand-written PTX | GPTの推論**と学習**を全部手書きPTXで |
| `fastnumpy` | numpy-style array ops (`+ - * / @`, relu) on the GPU | numpy風配列演算をGPUで |

## Notes / 注意

**EN**
- CUDA path is NVIDIA-only; iGPU / other vendors go through OpenCL.
- `turbo` is a saturation tool (benchmark / stress / thermal). It runs the GPU flat
  out — mind heat and power, especially on laptops. `Ctrl+C` stops it.
- No CUDA Toolkit, CuPy, PyTorch, or pyopencl required — only vendor drivers.

**日本語**
- CUDA 経路は NVIDIA 専用。iGPU や他ベンダは OpenCL 経由。
- `turbo` は飽和ツール(ベンチ/負荷/発熱試験用)。全力で回すので発熱・電力に注意
  (特にノートPC)。`Ctrl+C` で停止。
- CUDA Toolkit / CuPy / PyTorch / pyopencl は一切不要。ドライバのみ。

MIT License.
