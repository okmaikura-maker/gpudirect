"""
gpudirect.easy — どのベンダの GPU でも、どのライブラリの配列でも扱える汎用 API。

対応デバイス:
  - NVIDIA        : CUDA バックエンド(nvcuda.dll)、カーネルは PTX
  - AMD / Intel / iGPU : OpenCL バックエンド(OpenCL.dll)、カーネルは OpenCL C

どのライブラリとも噛み合う(相互運用):
  - 入力: numpy 配列 / list / bytes / バッファプロトコル / __array__ を持つ物
          (torch CPU tensor 等) / __dlpack__ を持つ物 を受け取れる
  - 出力: GpuArray/CLArray は __array__ を実装。np.asarray(x) や、array-protocol を
          読む他ライブラリからそのまま numpy として取り出せる。

    import numpy as np, gpudirect.easy as ge
    ge.devices()                       # 全ベンダの GPU/iGPU 一覧
    g = ge.GPU(0)                      # 既定=NVIDIA(CUDA)
    g = ge.GPU(0, backend="opencl")   # iGPU 等(OpenCL)
"""
import ctypes
import numpy as np

from . import Device, Context, DeviceMemory, _driver as _d
from . import opencl as _ocl


# ---- どのライブラリの配列でも numpy に変換して受け取る ----
def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(x)
    if isinstance(x, (GpuArray, CLArray)):
        return x.get()
    if hasattr(x, "__dlpack__"):
        try:
            return np.ascontiguousarray(np.from_dlpack(x))
        except Exception:
            pass
    if hasattr(x, "detach"):            # torch tensor
        try:
            return np.ascontiguousarray(x.detach().cpu().numpy())
        except Exception:
            pass
    if hasattr(x, "__array__"):
        return np.ascontiguousarray(np.asarray(x))
    return np.ascontiguousarray(np.asarray(x))


def devices():
    """全ベンダの GPU/iGPU を列挙。backend と device 番号つき。"""
    out = []
    try:                                    # NVIDIA (CUDA)
        for i in range(__import__("gpudirect").device_count()):
            d = Device(i)
            out.append(dict(backend="cuda", index=i, name=d.name, kind="GPU",
                            vendor="NVIDIA"))
    except Exception:
        pass
    cl = [d for d in _ocl.list_devices() if d["kind"] == "GPU"]   # 全ベンダ (OpenCL)
    for i, d in enumerate(cl):
        vendor = ("Intel" if "Intel" in d["platform"] else
                  "AMD" if ("AMD" in d["platform"] or "Advanced Micro" in d["platform"]) else
                  "NVIDIA" if "NVIDIA" in d["platform"] else d["platform"])
        out.append(dict(backend="opencl", index=i, name=d["name"], kind="GPU",
                        vendor=vendor, handle=d["handle"]))
    return out


# =====================================================================
# CUDA バックエンド(NVIDIA / PTX)
# =====================================================================
class GpuArray:
    def __init__(self, ctx, shape, dtype):
        self.ctx = ctx
        self.shape = tuple(shape) if hasattr(shape, "__iter__") else (int(shape),)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape))
        self.nbytes = self.size * self.dtype.itemsize
        self.mem = DeviceMemory(ctx, max(self.nbytes, 1))

    @classmethod
    def from_any(cls, ctx, arr):
        arr = _to_numpy(arr)
        o = cls(ctx, arr.shape, arr.dtype)
        buf = (ctypes.c_char*arr.nbytes).from_buffer_copy(arr.tobytes())
        _d.check(_d.cuMemcpyHtoD(o.mem.ptr, buf, arr.nbytes), "H2D"); return o

    def get(self):
        buf = (ctypes.c_char*self.nbytes)()
        _d.check(_d.cuMemcpyDtoH(buf, self.mem.ptr, self.nbytes), "D2H")
        return np.frombuffer(bytes(buf), dtype=self.dtype).reshape(self.shape).copy()

    def __array__(self, dtype=None):
        a = self.get(); return a.astype(dtype) if dtype else a
    def __repr__(self): return f"GpuArray(shape={self.shape}, dtype={self.dtype}, cuda)"


class Kernel:
    def __init__(self, func): self._f = func
    def __call__(self, grid=(1, 1, 1), block=(1, 1, 1), args=(), shared_mem=0, sync=True):
        packed = []
        for a in args:
            if isinstance(a, GpuArray): packed.append(a.mem)
            elif isinstance(a, np.integer): packed.append(int(a))
            elif isinstance(a, np.floating): packed.append(float(a))
            else: packed.append(a)
        self._f.launch(grid=grid, block=block, args=packed, shared_mem=shared_mem, sync=sync)


class _Module:
    def __init__(self, mod): self._m = mod
    def kernel(self, name): return Kernel(self._m.function(name))


class CudaGPU:
    backend = "cuda"
    def __init__(self, device=0):
        self.device = Device(device); self.ctx = self.device.create_context()
    @property
    def name(self): return self.device.name
    def info(self): return self.device.info()
    def to_gpu(self, arr): return GpuArray.from_any(self.ctx, arr)
    def empty(self, shape, dtype=np.float32): return GpuArray(self.ctx, shape, dtype)
    def zeros(self, shape, dtype=np.float32):
        a = GpuArray(self.ctx, shape, dtype); a.mem.memset(0); return a
    def module(self, ptx, verbose=False): return _Module(self.ctx.load_ptx(ptx, verbose=verbose))
    def module_cubin(self, cubin): return _Module(self.ctx.load_cubin(cubin))
    def kernel(self, ptx, name, verbose=False): return self.module(ptx, verbose).kernel(name)
    def synchronize(self): _d.check(_d.cuCtxSynchronize(), "sync")
    def close(self): self.ctx.destroy()
    def __enter__(self): return self
    def __exit__(self, *e): self.close()


# =====================================================================
# OpenCL バックエンド(AMD / Intel / iGPU / NVIDIA も可 / OpenCL C)
# =====================================================================
class CLArray:
    def __init__(self, comp, shape, dtype):
        self.comp = comp
        self.shape = tuple(shape) if hasattr(shape, "__iter__") else (int(shape),)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape))
        self.nbytes = self.size*self.dtype.itemsize
        self.buf = comp.buffer(max(self.nbytes, 1))

    @classmethod
    def from_any(cls, comp, arr):
        arr = _to_numpy(arr)
        o = cls(comp, arr.shape, arr.dtype); o.buf.write(arr.tobytes()); return o

    def get(self):
        return np.frombuffer(self.buf.read(self.nbytes), dtype=self.dtype).reshape(self.shape).copy()
    def __array__(self, dtype=None):
        a = self.get(); return a.astype(dtype) if dtype else a
    def __repr__(self): return f"CLArray(shape={self.shape}, dtype={self.dtype}, opencl)"


class CLKernelWrap:
    def __init__(self, k): self._k = k
    def __call__(self, global_size, args=()):
        packed = []
        for a in args:
            if isinstance(a, CLArray): packed.append(a.buf)
            elif isinstance(a, (float, np.floating)): packed.append(float(a))
            else: packed.append(int(a))
        self._k(global_size, packed)


class CLGPU:
    backend = "opencl"
    def __init__(self, device=0):
        gpus = [d for d in _ocl.list_devices() if d["kind"] == "GPU"]
        self._info = gpus[device]; self.comp = _ocl.CLCompute(self._info["handle"])
    @property
    def name(self): return self._info["name"]
    def info(self): return dict(name=self._info["name"], vendor=self._info["platform"],
                                cu=self._info["cu"], mhz=self._info["mhz"], backend="opencl")
    def to_gpu(self, arr): return CLArray.from_any(self.comp, arr)
    def empty(self, shape, dtype=np.float32): return CLArray(self.comp, shape, dtype)
    def zeros(self, shape, dtype=np.float32):
        a = CLArray(self.comp, shape, dtype); a.buf.write(b"\x00"*a.nbytes); return a
    def kernel(self, source, name):
        self.comp.build(source); return CLKernelWrap(self.comp.kernel(name))
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *e): self.close()


# =====================================================================
# 統一の入口
# =====================================================================
def GPU(device=0, backend=None):
    """GPU を掴む。backend 省略時は NVIDIA(CUDA)、"opencl" で AMD/Intel/iGPU。"""
    if backend is None:
        backend = "cuda"
    if backend == "cuda":
        return CudaGPU(device)
    if backend == "opencl":
        return CLGPU(device)
    raise ValueError(f"unknown backend: {backend}")
