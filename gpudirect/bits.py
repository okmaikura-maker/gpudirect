"""
gpudirect.bits — 真の「GPUが一番やりやすい0/1」表現: ビットパッキング。

補足(正直な注記): 値を桁数で表す「一進数」(例: 5 を 11111 のように長さで
表す)は、GPU には向かない。値が大きいほどビット数が線形に増え、メモリも
演算も爆発的に悪化するため、固定長 32bit の演算器を持つ GPU とは相性が悪い。

本物の GPU 向け 0/1 活用は「ビットパッキング」: 32 個の真偽値を 1 個の
u32 に詰め、AND/OR/XOR や XNOR+popcount(一致ビット数を 1 命令で数える)で
一気に処理する。メモリは 1/32、演算も 32 要素分を 1 命令でまとめて行える。
バイナリニューラルネット(XNOR-Net 等)で実際に使われている手法。
"""
import os
import numpy as np

from . import Device, Context, DeviceMemory, _driver as _d

_KDIR = os.path.join(os.path.dirname(__file__), "kernels")
_ctx_cache = {}


def _ctx():
    if 0 not in _ctx_cache:
        _ctx_cache[0] = Device(0).create_context()
    return _ctx_cache[0]


def _kernels(ctx):
    if not hasattr(ctx, "_bitops"):
        m = ctx.load_ptx(open(os.path.join(_KDIR, "bitops.ptx"), "rb").read())
        ctx._bitops = {n: m.function(n) for n in
                      ("bit_and", "bit_or", "bit_xor", "bit_not", "xnor_popcount")}
    return ctx._bitops


def pack(bool_arr):
    """bool の numpy 配列を 32bit ワードに詰めた BitArray にして GPU へ送る。"""
    a = np.ascontiguousarray(bool_arr, dtype=bool)
    n_bits = a.size
    words = np.packbits(a.reshape(-1), bitorder="little")
    # packbits は 8bit(uint8)単位なので 4 個まとめて 1 word(u32)にする
    pad = (-len(words)) % 4
    if pad:
        words = np.concatenate([words, np.zeros(pad, np.uint8)])
    u32 = words.view(np.uint32)
    ctx = _ctx()
    mem = ctx.to_device(u32.tolist(), "u4")
    return BitArray(ctx, mem, n_bits, a.shape)


class BitArray:
    """32bit ワードにパックされた真偽値配列(GPU上)。"""

    def __init__(self, ctx, mem, n_bits, shape):
        self.ctx = ctx; self.mem = mem
        self.n_bits = n_bits; self.shape = shape
        self.n_words = (n_bits + 31) // 32

    def _binary(self, other, name):
        k = _kernels(self.ctx)[name]
        out = self.ctx.malloc(self.n_words * 4)
        t = 256
        k.launch(grid=((self.n_words+t-1)//t, 1, 1), block=(t, 1, 1),
                args=[self.mem, other.mem, out, self.n_words])
        return BitArray(self.ctx, out, self.n_bits, self.shape)

    def __and__(self, o): return self._binary(o, "bit_and")
    def __or__(self, o): return self._binary(o, "bit_or")
    def __xor__(self, o): return self._binary(o, "bit_xor")

    def __invert__(self):
        k = _kernels(self.ctx)["bit_not"]
        out = self.ctx.malloc(self.n_words * 4)
        t = 256
        k.launch(grid=((self.n_words+t-1)//t, 1, 1), block=(t, 1, 1),
                args=[self.mem, out, self.n_words])
        return BitArray(self.ctx, out, self.n_bits, self.shape)

    def match_count(self, other):
        """自分と other の、値が一致するビットの総数(XNOR+popcount)。"""
        k = _kernels(self.ctx)["xnor_popcount"]
        out = self.ctx.malloc(self.n_words * 4)
        t = 256
        k.launch(grid=((self.n_words+t-1)//t, 1, 1), block=(t, 1, 1),
                args=[self.mem, other.mem, out, self.n_words])
        counts = np.array(out.copy_to_host("u4", self.n_words))
        # 最後のワードは端数ビット(未使用部分)を含むので、水増し分を引く
        pad_bits = self.n_words*32 - self.n_bits
        return int(counts.sum()) - pad_bits  # 未使用ビットは両側 0 で一致してしまうため補正

    def numpy(self):
        u32 = np.array(self.mem.copy_to_host("u4", self.n_words))
        bits = np.unpackbits(u32.view(np.uint8), bitorder="little")
        return bits[:self.n_bits].reshape(self.shape).astype(bool)


# ============================================================================
# 2-bit (4状態) 記号配列: 1ワードに16記号。true/false だけでは足りない、
# 4種類の状態(例: 未知/A/B/C, DNA の ACGT 等)を int8 の 4倍密に詰める。
# 正直な注記: これは「速さ」ではなく「密度」の技術。GPU のクロック速度
# (オンオフの物理的な速さ)はソフトウェアからは一切変えられない。
# ============================================================================
def pack_quat(symbols):
    """0〜3 の整数配列を 2bit/記号、16記号/word に詰めて GPU へ送る。"""
    a = np.ascontiguousarray(symbols, dtype=np.uint8)
    if a.max(initial=0) > 3:
        raise ValueError("記号は 0〜3 の範囲にしてください(2bit = 4状態)")
    n_sym = a.size
    flat = a.reshape(-1)
    pad = (-len(flat)) % 16
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.uint8)])
    words = np.zeros(len(flat)//16, np.uint32)
    for i in range(16):
        words |= flat[i::16].astype(np.uint32) << (i*2)
    ctx = _ctx()
    mem = ctx.to_device(words.tolist(), "u4")
    return QuatArray(ctx, mem, n_sym, a.shape)


class QuatArray:
    """2bit(4状態)記号を 16個/word に詰めた配列(GPU上)。"""

    def __init__(self, ctx, mem, n_sym, shape):
        self.ctx = ctx; self.mem = mem
        self.n_sym = n_sym; self.shape = shape
        self.n_words = (n_sym + 15) // 16

    def _kernel(self):
        if not hasattr(self.ctx, "_quatops"):
            m = self.ctx.load_ptx(open(os.path.join(_KDIR, "quatops.ptx"), "rb").read())
            self.ctx._quatops = m.function("quat_match_count")
        return self.ctx._quatops

    def match_count(self, other):
        """自分と other で、値が一致する記号(2bitペア)の総数。"""
        k = self._kernel()
        out = self.ctx.malloc(self.n_words * 4)
        t = 256
        k.launch(grid=((self.n_words+t-1)//t, 1, 1), block=(t, 1, 1),
                args=[self.mem, other.mem, out, self.n_words])
        counts = np.array(out.copy_to_host("u4", self.n_words))
        pad_sym = self.n_words*16 - self.n_sym
        return int(counts.sum()) - pad_sym  # 末尾パディング(両側00)の水増し分を補正

    def numpy(self):
        words = np.array(self.mem.copy_to_host("u4", self.n_words))
        out = np.zeros(self.n_words*16, np.uint8)
        for i in range(16):
            out[i::16] = (words >> (i*2)) & 0b11
        return out[:self.n_sym].reshape(self.shape)
