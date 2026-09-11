"""
gpudirect.pack — データを渡された瞬間に、GPU が読みやすい形へ自動で束ねる。

やること:
  1. パディング   : 行列を 64/16 の倍数に揃え、カーネル内の端数分岐をなくす
  2. ひとまとめ転送: 複数の配列を 1 本のバッファに連結し、H2D 転送を 1 回にまとめる
                    (転送ごとに ctypes 呼び出し・ドライバ往復が発生するため、
                     まとめるほど呼び出し回数が減って速い)
  3. 自動で切り戻す: 計算後はパディング分を取り除いて元の形状で返す

使い方(自動・意識不要):
    from gpudirect import pack
    c = fnp.array(A) @ fnp.array(B)   # 内部で自動的にパディング+ひとまとめ転送される

明示的に使う場合:
    from gpudirect import pack
    padded, orig_shape = pack.pad_to_tile(A, tile_m=64, tile_k=16)
    bundle = pack.Bundle(ctx)
    da = bundle.add(A); db = bundle.add(B)
    bundle.upload()          # ここで初めて 1 回だけ GPU に転送
"""
import numpy as np
import ctypes

from . import DeviceMemory, _driver as _d


def pad_to_tile(A, tile_m=64, tile_k=16, axis_m=0, axis_k=1):
    """
    2次元配列 A を、指定タイルの倍数になるようゼロパディングする。
    戻り値: (パディング済み配列, 元の shape)。パディング不要なら A をそのまま返す。
    """
    A = np.ascontiguousarray(A, dtype=np.float32)
    m, k = A.shape
    mp = ((m + tile_m - 1)//tile_m) * tile_m
    kp = ((k + tile_k - 1)//tile_k) * tile_k
    if mp == m and kp == k:
        return A, (m, k)
    out = np.zeros((mp, kp), np.float32)
    out[:m, :k] = A
    return out, (m, k)


def crop(A, shape):
    """pad_to_tile で膨らませた配列を元の shape に切り戻す。"""
    m, k = shape
    return A[:m, :k] if A.shape != (m, k) else A


class Bundle:
    """
    複数の numpy 配列を 1 本の GPU バッファに束ね、H2D 転送を 1 回にまとめる。
    「ポポイと一塊」= 複数回の転送呼び出しを 1 回に圧縮する仕組み。
    """
    def __init__(self, ctx):
        self.ctx = ctx
        self._chunks = []       # [(numpy配列, offset_bytes, nbytes)]
        self._total = 0
        self._mem = None

    def add(self, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        off = self._total
        nbytes = arr.nbytes
        self._chunks.append((arr, off, nbytes))
        self._total += nbytes
        return _BundleView(self, off, arr.shape, arr.dtype)

    def upload(self):
        """束ねた全配列を、1 回の H2D 転送で GPU に送る。"""
        if self._total == 0:
            return
        blob = bytearray(self._total)
        for arr, off, nbytes in self._chunks:
            blob[off:off+nbytes] = arr.tobytes()
        self._mem = DeviceMemory(self.ctx, self._total)
        buf = (ctypes.c_char * self._total).from_buffer(blob)
        _d.check(_d.cuMemcpyHtoD(self._mem.ptr, buf, self._total), "bundle H2D")
        return self


class _BundleView:
    """Bundle 内の 1 個の配列を指す、オフセット付きのデバイスポインタ。"""
    def __init__(self, bundle, offset, shape, dtype):
        self.bundle = bundle; self.offset = offset
        self.shape = shape; self.dtype = dtype

    @property
    def ptr(self):
        """Bundle.upload() 後に有効な、この配列先頭のデバイスポインタ。"""
        base = self.bundle._mem.ptr
        return _d.CUdeviceptr(base.value + self.offset)
