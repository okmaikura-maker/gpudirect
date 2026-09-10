"""
gpudirect.interop — 主要ライブラリ全部と相互連携する橋渡し。

入力(どれでも受け取る):
    numpy / PyTorch / CuPy / TensorFlow / JAX / pandas / list /
    バッファプロトコル / __array__ / __dlpack__ / __cuda_array_interface__

出力(どれにでも渡せる):
    .numpy() / .torch() / .cupy() / .pandas() / .tensorflow() / .jax() /
    .to("<lib>") / .tolist() / __dlpack__ / __array__
    （CUDA 配列は __cuda_array_interface__ を公開 → CuPy/Numba とゼロコピー連携）

各ライブラリは "必要になったときだけ" import する(未導入でも gpudirect は動く)。
"""
import numpy as np


def _c(a):
    return np.ascontiguousarray(a)


def to_numpy(x):
    """あらゆるライブラリの配列/テンソルを numpy に変換して受け取る。"""
    if isinstance(x, np.ndarray):
        return _c(x)
    tn = type(x).__name__
    mod = type(x).__module__ or ""
    # gpudirect 自身の配列
    if tn in ("GpuArray", "CLArray") and hasattr(x, "get"):
        return x.get()
    # PyTorch
    if mod.startswith("torch"):
        try:
            return _c(x.detach().to("cpu").numpy())
        except Exception:
            pass
    # CuPy
    if mod.startswith("cupy"):
        import cupy as cp
        return _c(cp.asnumpy(x))
    # TensorFlow
    if mod.startswith("tensorflow"):
        return _c(np.asarray(x))
    # JAX
    if mod.startswith("jax"):
        return _c(np.asarray(x))
    # pandas
    if mod.startswith("pandas"):
        return _c(x.to_numpy())
    # DLPack(規格として多くのライブラリが対応)
    if hasattr(x, "__dlpack__"):
        try:
            return _c(np.from_dlpack(x))
        except Exception:
            pass
    # __cuda_array_interface__(GPU 常駐。numpy 単体では読めないので cupy 経由)
    if hasattr(x, "__cuda_array_interface__"):
        try:
            import cupy as cp
            return _c(cp.asnumpy(cp.asarray(x)))
        except Exception:
            pass
    # 一般の array-like / buffer / list
    if hasattr(x, "__array__"):
        return _c(np.asarray(x))
    return _c(np.asarray(x))


def to_library(a, lib):
    """numpy 配列 a を指定ライブラリの形に変換して返す。"""
    a = _c(a)
    lib = lib.lower()
    if lib in ("numpy", "np"):
        return a
    if lib in ("torch", "pytorch"):
        import torch
        return torch.from_numpy(a)
    if lib in ("cupy", "cp"):
        import cupy as cp
        return cp.asarray(a)
    if lib in ("pandas", "pd"):
        import pandas as pd
        return pd.Series(a) if a.ndim == 1 else pd.DataFrame(a)
    if lib in ("tensorflow", "tf"):
        import tensorflow as tf
        return tf.convert_to_tensor(a)
    if lib in ("jax", "jnp"):
        import jax.numpy as jnp
        return jnp.asarray(a)
    if lib in ("list", "py"):
        return a.tolist()
    if lib == "dlpack":
        return a.__dlpack__()
    raise ValueError(f"未知のライブラリ: {lib}")


# easy の GpuArray / CLArray に相互連携メソッドを注入する
def attach(cls, cuda=False):
    def numpy(self): return self.get()
    def to(self, lib): return to_library(self.get(), lib)
    def torch(self): return to_library(self.get(), "torch")
    def cupy(self): return to_library(self.get(), "cupy")
    def pandas(self): return to_library(self.get(), "pandas")
    def tensorflow(self): return to_library(self.get(), "tensorflow")
    def jax(self): return to_library(self.get(), "jax")
    def tolist(self): return self.get().tolist()
    def __dlpack__(self, *a, **k): return self.get().__dlpack__(*a, **k)
    cls.numpy = numpy; cls.to = to; cls.torch = torch; cls.cupy = cupy
    cls.pandas = pandas; cls.tensorflow = tensorflow; cls.jax = jax
    cls.tolist = tolist; cls.__dlpack__ = __dlpack__
    if cuda:
        def _cai(self):
            return {"shape": tuple(self.shape), "typestr": self.dtype.str,
                    "data": (int(self.mem.ptr.value), False),
                    "strides": None, "version": 3}
        cls.__cuda_array_interface__ = property(_cai)
    return cls
