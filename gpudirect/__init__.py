"""
gpudirect — nvcuda.dll を Windows API (LoadLibraryW/GetProcAddress) で直接握り、
CUDA Driver API を叩いて GPU を直接使う自作ライブラリ。

依存: なし (pip 追加ゼロ / CUDA Toolkit 不要 / CuPy・PyTorch 不要)。
必要なのは NVIDIA ドライバ同梱の nvcuda.dll のみ。

2 つのレベルでカーネルを投入できる:
  - アセンブリレベル:  手書き PTX を load_ptx() でドライバ内蔵 JIT に渡す
  - 機械語レベル:      SASS 入り cubin を load_cubin() で JIT なしで直接ロード

使い方の最短例:
    import gpudirect as gd
    ctx = gd.Device(0).create_context()
    mod = ctx.load_ptx(open("kernel.ptx","rb").read())
    fn  = mod.function("vecadd")
    a = ctx.to_device([1,2,3], "f4")
    ...
    fn.launch(grid=(1,1,1), block=(3,1,1), args=[a, ...])
"""
import ctypes
import array as _array

from . import _driver as _d
from ._driver import CudaError

__all__ = ["init", "device_count", "driver_version",
           "Device", "Context", "Module", "Function", "DeviceMemory", "ManagedArray",
           "CudaError"]

_initialized = False


def init():
    """cuInit を一度だけ呼ぶ。各 API は内部でこれを呼ぶので通常は不要。"""
    global _initialized
    if not _initialized:
        _d.check(_d.cuInit(0), "cuInit")
        _initialized = True


def driver_version():
    """インストール済みドライバの CUDA バージョン (例: 13020 = 13.2)。"""
    init()
    v = ctypes.c_int()
    _d.check(_d.cuDriverGetVersion(ctypes.byref(v)), "cuDriverGetVersion")
    return v.value


def device_count():
    init()
    n = ctypes.c_int()
    _d.check(_d.cuDeviceGetCount(ctypes.byref(n)), "cuDeviceGetCount")
    return n.value


# array の typecode → バイト数
_TYPECODE_SIZE = {"b": 1, "B": 1, "h": 2, "H": 2, "i": 4, "I": 4,
                  "l": 4, "L": 4, "q": 8, "Q": 8, "f": 4, "d": 8}
# よく使う numpy 風エイリアス
_DTYPE_ALIAS = {"f4": "f", "f8": "d", "i4": "i", "i8": "q",
                "u4": "I", "u1": "B", "i1": "b"}


def _norm_dtype(dt):
    return _DTYPE_ALIAS.get(dt, dt)


class Device:
    """物理 GPU 1 台。"""

    def __init__(self, ordinal=0):
        init()
        self.ordinal = ordinal
        self.handle = ctypes.c_int()
        _d.check(_d.cuDeviceGet(ctypes.byref(self.handle), ordinal),
                 "cuDeviceGet")

    @property
    def name(self):
        buf = ctypes.create_string_buffer(256)
        _d.check(_d.cuDeviceGetName(buf, 256, self.handle), "cuDeviceGetName")
        return buf.value.decode()

    def _attr(self, code):
        v = ctypes.c_int()
        _d.check(_d.cuDeviceGetAttribute(ctypes.byref(v), code, self.handle),
                 "cuDeviceGetAttribute")
        return v.value

    @property
    def total_memory(self):
        b = ctypes.c_size_t()
        _d.check(_d.cuDeviceTotalMem(ctypes.byref(b), self.handle),
                 "cuDeviceTotalMem")
        return b.value

    @property
    def compute_capability(self):
        return (self._attr(_d.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
                self._attr(_d.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR))

    def info(self):
        cc = self.compute_capability
        return {
            "name": self.name,
            "compute_capability": f"{cc[0]}.{cc[1]}",
            "sm_arch": f"sm_{cc[0]}{cc[1]}",
            "total_memory_MiB": self.total_memory // (1024 * 1024),
            "sm_count": self._attr(_d.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
            "warp_size": self._attr(_d.CU_DEVICE_ATTRIBUTE_WARP_SIZE),
            "max_threads_per_block":
                self._attr(_d.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
            "clock_MHz": self._attr(_d.CU_DEVICE_ATTRIBUTE_CLOCK_RATE) // 1000,
        }

    def create_context(self):
        return Context(self)


class Context:
    """GPU コンテキスト。メモリ確保・モジュールロード・実行の起点。"""

    def __init__(self, device):
        self.device = device
        self.handle = _d.CUcontext()
        _d.check(_d.cuCtxCreate(ctypes.byref(self.handle), 0, device.handle),
                 "cuCtxCreate")

    def make_current(self):
        _d.check(_d.cuCtxSetCurrent(self.handle), "cuCtxSetCurrent")

    # --- メモリ --------------------------------------------------------------
    def malloc(self, nbytes):
        return DeviceMemory(self, nbytes)

    def zeros_shared(self, shape, dtype="f4"):
        """
        ゼロコピー配列(Unified Memory)を確保して返す。
        CPU と GPU が同じ物理アドレスを共有するので、H2D/D2H の転送が要らない。
        呼び出し側は .np で numpy 配列として直接読み書きでき、その変更は
        コピーなしでそのまま GPU カーネルから見える(逆も同様)。
        """
        return ManagedArray(self, shape, dtype)

    def to_device(self, data, dtype="f4"):
        """
        Python の list / array / bytes を GPU に転送し DeviceMemory を返す。
        dtype: "f4","f8","i4","i8","u4","u1" など (array の typecode も可)。
        """
        tc = _norm_dtype(dtype)
        if isinstance(data, (bytes, bytearray)):
            buf = (ctypes.c_char * len(data)).from_buffer_copy(bytes(data))
            nbytes = len(data)
        else:
            arr = _array.array(tc, data)
            nbytes = arr.buffer_info()[1] * arr.itemsize
            buf = (ctypes.c_char * nbytes).from_buffer_copy(arr.tobytes())
        mem = DeviceMemory(self, nbytes)
        mem.copy_from_host(buf, nbytes)
        mem.dtype = tc
        return mem

    # --- モジュール ----------------------------------------------------------
    def load_ptx(self, ptx, verbose=False):
        """
        手書き / 生成済み PTX(アセンブリ)をロード。ドライバ内蔵 JIT が
        この GPU 用の SASS(機械語)に変換する。verbose=True で JIT ログを取得。
        ptx は str でも bytes でも可 (自動で NUL 終端を付与)。
        """
        if isinstance(ptx, bytes):
            ptx = ptx.decode("utf-8-sig")   # BOM を除去
        # ドライバの JIT は行頭 // コメントで INVALID_PTX になることがあるため、
        # // 以降を各行から取り除く (PTX に文字列リテラルは無いので安全)。
        clean_lines = []
        for line in ptx.splitlines():
            idx = line.find("//")
            if idx != -1:
                line = line[:idx]
            clean_lines.append(line)
        ptx = "\n".join(clean_lines).encode("ascii") + b"\x00"
        return Module(self, ptx, is_cubin=False, verbose=verbose)

    def load_cubin(self, cubin):
        """
        cubin(コンパイル済み SASS = GPU の実機械語コンテナ)を JIT なしで
        直接ロードする。これがユーザーモードから到達できる最も低い実行層。
        cubin は bytes、または .cubin ファイルのパス(str)。
        """
        if isinstance(cubin, str):
            with open(cubin, "rb") as f:
                cubin = f.read()
        return Module(self, cubin, is_cubin=True)

    # --- 後始末 --------------------------------------------------------------
    def destroy(self):
        if self.handle:
            _d.cuCtxDestroy(self.handle)
            self.handle = _d.CUcontext()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()


class DeviceMemory:
    """GPU 上の線形メモリ 1 ブロック。"""

    def __init__(self, ctx, nbytes):
        self.ctx = ctx
        self.nbytes = nbytes
        self.dtype = None
        self.ptr = _d.CUdeviceptr()
        _d.check(_d.cuMemAlloc(ctypes.byref(self.ptr), nbytes), "cuMemAlloc")

    def copy_from_host(self, buf, nbytes=None):
        nbytes = nbytes if nbytes is not None else self.nbytes
        _d.check(_d.cuMemcpyHtoD(self.ptr, buf, nbytes), "cuMemcpyHtoD")

    def copy_to_host(self, dtype=None, count=None):
        """GPU → ホスト。array.array で返す。"""
        tc = _norm_dtype(dtype or self.dtype or "B")
        isz = _TYPECODE_SIZE[tc]
        n = count if count is not None else self.nbytes // isz
        buf = (ctypes.c_char * (n * isz))()
        _d.check(_d.cuMemcpyDtoH(buf, self.ptr, n * isz), "cuMemcpyDtoH")
        out = _array.array(tc)
        out.frombytes(bytes(buf))
        return out

    def memset(self, value=0):
        _d.check(_d.cuMemsetD8(self.ptr, value & 0xFF, self.nbytes),
                 "cuMemsetD8")

    def copy_from_dev(self, src, nbytes=None):
        """GPU→GPU コピー (デバイス内)。"""
        nbytes = nbytes if nbytes is not None else self.nbytes
        _d.check(_d.cuMemcpyDtoD(self.ptr, src.ptr, nbytes), "cuMemcpyDtoD")

    def free(self):
        if self.ptr and self.ptr.value:
            _d.cuMemFree(self.ptr)
            self.ptr = _d.CUdeviceptr()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class ManagedArray:
    """
    ゼロコピー配列(CUDA Unified Memory)。

    普通の DeviceMemory は「CPU側バッファ → cuMemcpyHtoD → GPU側バッファ」
    「GPU側バッファ → cuMemcpyDtoH → CPU側バッファ」と、都度コピーが発生する。
    ManagedArray は cuMemAllocManaged で確保した 1 本のメモリを CPU/GPU 両方が
    直接読み書きする。だからコピーが要らない(ゼロコピー) —— .np で得られる
    numpy 配列への書き込みは、そのままカーネル起動時に GPU から見える。

    制約: cuCtxSynchronize() を挟まずに CPU と GPU が同時アクセスすると
    未定義動作になるため、GPU 側の処理後は必ず ctx.synchronize 相当を呼ぶこと
    (Function.launch の既定 sync=True がこれを行う)。
    """

    def __init__(self, ctx, shape, dtype="f4"):
        self.ctx = ctx
        self.shape = tuple(shape) if hasattr(shape, "__iter__") else (int(shape),)
        tc = _norm_dtype(dtype)
        self._typecode = tc
        itemsize = _TYPECODE_SIZE[tc]
        self.size = 1
        for s in self.shape:
            self.size *= s
        self.nbytes = self.size * itemsize
        self.ptr = ctypes.c_uint64()
        _d.check(_d.cuMemAllocManaged(ctypes.byref(self.ptr), max(self.nbytes, 1),
                                      _d.CU_MEM_ATTACH_GLOBAL), "cuMemAllocManaged")
        # ctypes 経由でこのアドレスに直接かぶさる numpy 配列を作る(コピー無し)
        import numpy as _np
        _np_dtype = {"f": _np.float32, "d": _np.float64, "i": _np.int32,
                    "q": _np.int64, "I": _np.uint32, "B": _np.uint8,
                    "b": _np.int8}[tc]
        ctype = {"f": ctypes.c_float, "d": ctypes.c_double, "i": ctypes.c_int32,
                "q": ctypes.c_int64, "I": ctypes.c_uint32, "B": ctypes.c_uint8,
                "b": ctypes.c_int8}[tc]
        buf = (ctype * self.size).from_address(self.ptr.value)
        self.np = _np.ctypeslib.as_array(buf).reshape(self.shape)

    def free(self):
        if self.ptr and self.ptr.value:
            _d.cuMemFree(_d.CUdeviceptr(self.ptr.value))
            self.ptr = ctypes.c_uint64()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    def __repr__(self):
        return f"ManagedArray(shape={self.shape}, dtype={self._typecode}, zero-copy)"


class Module:
    """ロード済みの PTX / cubin モジュール。"""

    def __init__(self, ctx, image, is_cubin, verbose=False):
        self.ctx = ctx
        self.handle = _d.CUmodule()
        self.jit_log = ""
        img = (ctypes.c_char * len(image)).from_buffer_copy(image)

        if is_cubin or not verbose:
            _d.check(_d.cuModuleLoadData(ctypes.byref(self.handle), img),
                     "cuModuleLoadData")
        else:
            # JIT ログ付きロード (PTX のみ意味を持つ)
            LOGSZ = 8192
            log = ctypes.create_string_buffer(LOGSZ)
            elog = ctypes.create_string_buffer(LOGSZ)
            keys = (ctypes.c_int * 5)(
                _d.CU_JIT_INFO_LOG_BUFFER,
                _d.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                _d.CU_JIT_ERROR_LOG_BUFFER,
                _d.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                _d.CU_JIT_LOG_VERBOSE,
            )
            # optionValues は void*[]。サイズやフラグはその値をポインタ幅の
            # スロットに詰めて渡す (アドレスではなく即値として解釈される)。
            vals = (ctypes.c_void_p * 5)(
                ctypes.cast(log, ctypes.c_void_p),
                ctypes.c_void_p(LOGSZ),
                ctypes.cast(elog, ctypes.c_void_p),
                ctypes.c_void_p(LOGSZ),
                ctypes.c_void_p(1),
            )
            try:
                _d.check(_d.cuModuleLoadDataEx(
                    ctypes.byref(self.handle), img, 5,
                    ctypes.cast(keys, ctypes.c_void_p),
                    ctypes.cast(vals, ctypes.c_void_p)),
                    "cuModuleLoadDataEx")
            except CudaError:
                self.jit_log = (log.value + b"\n" + elog.value).decode(errors="replace")
                raise
            self.jit_log = log.value.decode(errors="replace")

    def function(self, name):
        return Function(self, name)

    def global_var(self, name):
        """__device__ グローバル変数の (device_ptr, size) を返す。"""
        ptr = _d.CUdeviceptr()
        size = ctypes.c_size_t()
        _d.check(_d.cuModuleGetGlobal(ctypes.byref(ptr), ctypes.byref(size),
                                      self.handle, name.encode("ascii")),
                 "cuModuleGetGlobal")
        return ptr, size.value

    def unload(self):
        if self.handle:
            _d.cuModuleUnload(self.handle)
            self.handle = _d.CUmodule()


class Function:
    """モジュール内のカーネル関数 1 つ。"""

    def __init__(self, module, name):
        self.module = module
        self.name = name
        self.handle = _d.CUfunction()
        _d.check(_d.cuModuleGetFunction(ctypes.byref(self.handle),
                                        module.handle, name.encode("ascii")),
                 "cuModuleGetFunction")

    def _pack(self, args):
        """
        カーネル引数を cuLaunchKernel 用の void** に変換する。
        受け付ける型:
          DeviceMemory          -> そのデバイスポインタ
          int                   -> 32bit 符号付き整数
          float                 -> 32bit float
          ctypes インスタンス    -> そのまま
        """
        holders = []          # GC 防止のため実体を保持
        ptr_arr = (ctypes.c_void_p * len(args))()
        for i, a in enumerate(args):
            if isinstance(a, DeviceMemory):
                cval = _d.CUdeviceptr(a.ptr.value)
            elif isinstance(a, bool):
                cval = ctypes.c_int(1 if a else 0)
            elif isinstance(a, int):
                cval = ctypes.c_int(a)
            elif isinstance(a, float):
                cval = ctypes.c_float(a)
            elif isinstance(a, ctypes._SimpleCData):
                cval = a
            else:
                raise TypeError(f"未対応のカーネル引数型: {type(a)}")
            holders.append(cval)
            ptr_arr[i] = ctypes.cast(ctypes.byref(cval), ctypes.c_void_p)
        return ptr_arr, holders

    def launch(self, grid=(1, 1, 1), block=(1, 1, 1), args=(),
               shared_mem=0, sync=True):
        """
        カーネルを起動する。grid / block は (x,y,z) または int。
        sync=True なら cuCtxSynchronize まで待つ。
        """
        if isinstance(grid, int):
            grid = (grid, 1, 1)
        if isinstance(block, int):
            block = (block, 1, 1)
        ptr_arr, _holders = self._pack(args)
        _d.check(_d.cuLaunchKernel(
            self.handle,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_mem, None,
            ctypes.cast(ptr_arr, ctypes.c_void_p) if args else None,
            None), "cuLaunchKernel")
        if sync:
            _d.check(_d.cuCtxSynchronize(), "cuCtxSynchronize")


# ============================================================================
# 全機能をトップレベルに統合(遅延ロード)。 import gpudirect as gd だけで
#   gd.GPU / gd.devices / gd.saturate / gd.array / gd.GpuGPTTrainer ... が使える。
# numpy 未導入でも基本 import は軽いまま(初めて触れた時に必要な物だけ読む)。
# ============================================================================
_SUBMODULES = ("easy", "opencl", "turbo", "fastnumpy", "interop")
_LAZY = {
    # 汎用API(全ベンダ)
    "GPU": ("easy", "GPU"),
    "devices": ("easy", "devices"),
    "GpuArray": ("easy", "GpuArray"),
    "CLArray": ("easy", "CLArray"),
    # 一行飽和
    "saturate": ("turbo", "saturate"),
    # numpy 風(fastnumpy)
    "array": ("fastnumpy", "array"),
    "zeros_like_np": ("fastnumpy", "zeros"),
    "ones": ("fastnumpy", "ones"),
    "matmul": ("fastnumpy", "matmul"),
    "relu": ("fastnumpy", "relu"),
    "farray": ("fastnumpy", "farray"),
    # ライブラリ連携
    "to_numpy": ("interop", "to_numpy"),
    "to_library": ("interop", "to_library"),
}
__all__ = list(__all__) + list(_SUBMODULES) + list(_LAZY.keys())


def __getattr__(name):
    import importlib
    if name in _SUBMODULES:
        return importlib.import_module("." + name, __name__)
    if name in _LAZY:
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module("." + mod, __name__), attr)
    raise AttributeError(f"module 'gpudirect' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_SUBMODULES) + list(_LAZY.keys()))
