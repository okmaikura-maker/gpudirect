"""
gpudirect.easy — 汎用 GPU 直叩き API。

どんな用途でも使える最小・汎用のインターフェース。numpy 配列を GPU に載せ、
自作 PTX カーネルを「関数のように」呼び、結果を numpy で受け取る。
CUDA Toolkit / CuPy / PyTorch 不要(NVIDIA ドライバの nvcuda.dll のみ)。

例:
    import numpy as np, gpudirect.easy as ge

    g = ge.GPU()                       # デバイス 0 を掴む
    a = g.to_gpu(np.arange(8, dtype=np.float32))
    b = g.to_gpu(np.ones(8, np.float32))
    c = g.empty(8, np.float32)

    saxpy = g.kernel(PTX_SOURCE, "saxpy")   # PTX を JIT して関数化
    n = 8
    saxpy(grid=(1,1,1), block=(n,1,1), args=[a, b, c, 2.0, n])
    print(c.get())                     # numpy で回収

任意 dtype (f4/f8/i4/i8/u1/u4/i2/…) 対応。カーネル引数は GpuArray / int /
float / numpy スカラを混在可。cubin(機械語)も load_cubin で直接ロード可。
"""
import ctypes
import numpy as np

from . import Device, Context, DeviceMemory, _driver as _d


class GpuArray:
    """GPU 上の配列。shape と numpy dtype を保持し、get() で numpy に戻す。"""

    def __init__(self, ctx, shape, dtype):
        self.ctx = ctx
        self.shape = tuple(shape) if hasattr(shape, "__iter__") else (int(shape),)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape))
        self.nbytes = self.size * self.dtype.itemsize
        self.mem = DeviceMemory(ctx, max(self.nbytes, 1))

    @classmethod
    def from_numpy(cls, ctx, arr):
        arr = np.ascontiguousarray(arr)
        obj = cls(ctx, arr.shape, arr.dtype)
        buf = (ctypes.c_char * arr.nbytes).from_buffer_copy(arr.tobytes())
        _d.check(_d.cuMemcpyHtoD(obj.mem.ptr, buf, arr.nbytes), "H2D")
        return obj

    def get(self):
        """GPU → numpy(元の shape / dtype で)。"""
        buf = (ctypes.c_char * self.nbytes)()
        _d.check(_d.cuMemcpyDtoH(buf, self.mem.ptr, self.nbytes), "D2H")
        return np.frombuffer(bytes(buf), dtype=self.dtype).reshape(self.shape).copy()

    def copy_from(self, arr):
        """既存の GPU バッファに numpy を書き込む(形状一致が前提)。"""
        arr = np.ascontiguousarray(arr.astype(self.dtype))
        buf = (ctypes.c_char * arr.nbytes).from_buffer_copy(arr.tobytes())
        _d.check(_d.cuMemcpyHtoD(self.mem.ptr, buf, arr.nbytes), "H2D")
        return self

    def fill_zero(self):
        self.mem.memset(0); return self

    def __repr__(self):
        return f"GpuArray(shape={self.shape}, dtype={self.dtype})"


class Kernel:
    """PTX/cubin 内の 1 カーネル。関数のように呼べる。"""

    def __init__(self, func):
        self._f = func

    def __call__(self, grid=(1, 1, 1), block=(1, 1, 1), args=(), shared_mem=0, sync=True):
        packed = []
        for a in args:
            if isinstance(a, GpuArray):
                packed.append(a.mem)                 # デバイスポインタとして渡る
            elif isinstance(a, (np.integer,)):
                packed.append(int(a))
            elif isinstance(a, (np.floating,)):
                packed.append(float(a))
            else:
                packed.append(a)                     # int / float / ctypes
        self._f.launch(grid=grid, block=block, args=packed,
                       shared_mem=shared_mem, sync=sync)


class Module:
    def __init__(self, mod):
        self._m = mod
    def kernel(self, name):
        return Kernel(self._m.function(name))


class GPU:
    """GPU 1 台への汎用ハンドル。メモリ確保・カーネル JIT・実行の入口。"""

    def __init__(self, device=0):
        self.device = Device(device)
        self.ctx = self.device.create_context()

    # --- 情報 ---
    @property
    def name(self):
        return self.device.name
    def info(self):
        return self.device.info()

    # --- メモリ ---
    def to_gpu(self, arr):
        """numpy 配列(または array-like)を GPU に転送。"""
        return GpuArray.from_numpy(self.ctx, np.asarray(arr))
    def empty(self, shape, dtype=np.float32):
        return GpuArray(self.ctx, shape, dtype)
    def zeros(self, shape, dtype=np.float32):
        return GpuArray(self.ctx, shape, dtype).fill_zero()

    # --- カーネル ---
    def module(self, ptx, verbose=False):
        """PTX(手書き/生成、str か bytes、コメント可)を JIT してモジュール化。"""
        return Module(self.ctx.load_ptx(ptx, verbose=verbose))
    def module_cubin(self, cubin):
        """cubin(コンパイル済み SASS=機械語)を JIT なしで直接ロード。"""
        return Module(self.ctx.load_cubin(cubin))
    def kernel(self, ptx, name, verbose=False):
        """PTX を JIT して、その中の関数 name を呼び出し可能にして返す。"""
        return self.module(ptx, verbose=verbose).kernel(name)

    def synchronize(self):
        _d.check(_d.cuCtxSynchronize(), "sync")

    def close(self):
        self.ctx.destroy()
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.close()
