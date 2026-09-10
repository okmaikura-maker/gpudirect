"""
Windows API 直呼びで nvcuda.dll (CUDA Driver API) を叩く低レベルバインディング。

方針:
  - モジュールのロードは ctypes の自動ローダ (WinDLL) に任せず、
    kernel32.dll の LoadLibraryW を「生の Windows API」として直接呼ぶ。
  - 各 CUDA 関数のアドレスは GetProcAddress で自分で解決する。
  - 得た関数アドレスは CFUNCTYPE のプロトタイプでそのまま呼び出す
    (CUDA Driver API は x64 では cdecl 相当の単一呼出規約)。

注意: ネイティブ関数ポインタを「呼び出す」ための機構は CPython 標準では
ctypes しか存在しない。ここでの ctypes はあくまで呼出の実体であり、
DLL のロードとシンボル解決は Windows API (kernel32) 側で明示的に行っている。
これより下 (MMIO / GPU ページテーブル) はカーネルモード専用。

pip 追加パッケージ・CUDA Toolkit・CuPy・PyTorch は一切不要。
"""
import ctypes
from ctypes import wintypes

# --- kernel32: 生の Windows API でローダを握る --------------------------------
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)

_LoadLibraryW = _k32.LoadLibraryW
_LoadLibraryW.restype = wintypes.HMODULE
_LoadLibraryW.argtypes = [wintypes.LPCWSTR]

_GetProcAddress = _k32.GetProcAddress
# 第2引数は LPCSTR(bytes)。返り値は x64 で 64bit なので c_void_p で受ける。
_GetProcAddress.restype = ctypes.c_void_p
_GetProcAddress.argtypes = [wintypes.HMODULE, wintypes.LPCSTR]

_hmod = _LoadLibraryW("nvcuda.dll")
if not _hmod:
    err = ctypes.get_last_error()
    raise ImportError(
        f"LoadLibraryW('nvcuda.dll') 失敗 (GetLastError={err})。"
        " NVIDIA ドライバが入っているか確認してください。"
    )


# --- 型エイリアス ------------------------------------------------------------
CUdeviceptr = ctypes.c_uint64          # 64bit 環境ではポインタ幅の整数
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUstream = ctypes.c_void_p
CUresult = ctypes.c_int
P = ctypes.c_void_p


# --- GetProcAddress でシンボルを解決し、呼び出し可能な関数ポインタを作る -------
_resolved = {}


def _resolve(name, restype, argtypes):
    """
    GetProcAddress で name のアドレスを取り、CFUNCTYPE 経由で
    直接呼び出せる callable を返す。
    """
    addr = _GetProcAddress(_hmod, name.encode("ascii"))
    if not addr:
        err = ctypes.get_last_error()
        raise ImportError(f"GetProcAddress('{name}') 失敗 (GetLastError={err})")
    proto = ctypes.CFUNCTYPE(restype, *argtypes)
    fn = proto(addr)
    _resolved[name] = (addr, fn)
    return fn


def symbol_address(name):
    """解決済みシンボルの生アドレス(int)を返す。デバッグ/確認用。"""
    if name in _resolved:
        return _resolved[name][0]
    return _GetProcAddress(_hmod, name.encode("ascii"))


# --- JIT オプション定数 (cuModuleLoadDataEx 用) -------------------------------
CU_JIT_INFO_LOG_BUFFER = 3
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4
CU_JIT_ERROR_LOG_BUFFER = 5
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
CU_JIT_LOG_VERBOSE = 12
CU_JIT_TARGET = 9

# --- デバイス属性定数 (必要な分だけ) -----------------------------------------
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13


# --- 関数の解決 (すべて GetProcAddress 経由) ----------------------------------
cuGetErrorName = _resolve("cuGetErrorName", CUresult, [CUresult, P])
cuGetErrorString = _resolve("cuGetErrorString", CUresult, [CUresult, P])


class CudaError(RuntimeError):
    """CUDA Driver API がゼロ以外を返したときに投げる例外。"""

    def __init__(self, code, where=""):
        self.code = code
        name = ctypes.c_char_p()
        estr = ctypes.c_char_p()
        try:
            cuGetErrorName(code, ctypes.byref(name))
            cuGetErrorString(code, ctypes.byref(estr))
        except Exception:
            pass
        n = name.value.decode() if name.value else "?"
        s = estr.value.decode() if estr.value else "?"
        prefix = f"{where}: " if where else ""
        super().__init__(f"{prefix}{n} ({code}) - {s}")


def check(code, where=""):
    if code != 0:
        raise CudaError(code, where)
    return code


cuInit = _resolve("cuInit", CUresult, [ctypes.c_uint])
cuDriverGetVersion = _resolve("cuDriverGetVersion", CUresult, [P])

cuDeviceGetCount = _resolve("cuDeviceGetCount", CUresult, [P])
cuDeviceGet = _resolve("cuDeviceGet", CUresult, [P, ctypes.c_int])
cuDeviceGetName = _resolve("cuDeviceGetName", CUresult,
                           [ctypes.c_char_p, ctypes.c_int, CUdevice])
cuDeviceGetAttribute = _resolve("cuDeviceGetAttribute", CUresult,
                                [P, ctypes.c_int, CUdevice])
cuDeviceTotalMem = _resolve("cuDeviceTotalMem_v2", CUresult, [P, CUdevice])

cuCtxCreate = _resolve("cuCtxCreate_v2", CUresult, [P, ctypes.c_uint, CUdevice])
cuCtxDestroy = _resolve("cuCtxDestroy_v2", CUresult, [CUcontext])
cuCtxSynchronize = _resolve("cuCtxSynchronize", CUresult, [])
cuCtxSetCurrent = _resolve("cuCtxSetCurrent", CUresult, [CUcontext])

cuMemAlloc = _resolve("cuMemAlloc_v2", CUresult, [P, ctypes.c_size_t])
cuMemFree = _resolve("cuMemFree_v2", CUresult, [CUdeviceptr])
cuMemcpyHtoD = _resolve("cuMemcpyHtoD_v2", CUresult,
                        [CUdeviceptr, P, ctypes.c_size_t])
cuMemcpyDtoH = _resolve("cuMemcpyDtoH_v2", CUresult,
                        [P, CUdeviceptr, ctypes.c_size_t])
cuMemsetD8 = _resolve("cuMemsetD8_v2", CUresult,
                      [CUdeviceptr, ctypes.c_ubyte, ctypes.c_size_t])
cuMemcpyDtoD = _resolve("cuMemcpyDtoD_v2", CUresult,
                        [CUdeviceptr, CUdeviceptr, ctypes.c_size_t])

cuModuleLoadData = _resolve("cuModuleLoadData", CUresult, [P, P])
cuModuleLoadDataEx = _resolve("cuModuleLoadDataEx", CUresult,
                              [P, P, ctypes.c_uint, P, P])
cuModuleUnload = _resolve("cuModuleUnload", CUresult, [CUmodule])
cuModuleGetFunction = _resolve("cuModuleGetFunction", CUresult,
                               [P, CUmodule, ctypes.c_char_p])
cuModuleGetGlobal = _resolve("cuModuleGetGlobal_v2", CUresult,
                             [P, P, CUmodule, ctypes.c_char_p])

cuLaunchKernel = _resolve("cuLaunchKernel", CUresult, [
    CUfunction,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,   # grid
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,   # block
    ctypes.c_uint,                                 # shared mem bytes
    CUstream,                                       # stream
    P,                                              # kernelParams (void**)
    P,                                              # extra (void**)
])
