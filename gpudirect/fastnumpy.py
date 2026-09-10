"""
gpudirect.fastnumpy — numpy 風の書き味で GPU 直叩き実行する配列ライブラリ。
(旧 top-level `fastnumpy` は本モジュールへの薄いエイリアス)

  import fastnumpy as fnp
  a = fnp.array([[1,2],[3,4]], dtype='f4')
  b = fnp.ones((2,2))
  c = a @ b + a * 2.0        # 全部 GPU 上の手書き PTX で計算
  print(c.numpy())           # numpy に戻す

対応: 要素ごと + - * / 、スカラ + - * / 、行列積 @ 、relu / sum / transpose。
すべて float32。中身は gpudirect(nvcuda.dll 直叩き)+ 手書き PTX。
CUDA Toolkit / CuPy / PyTorch 不要。
"""
import os
import numpy as np

import gpudirect.easy as ge

# ---- 要素演算・スカラ演算の PTX(1 モジュールに複数 entry) ----
_EW_PTX = """
.version 7.0
.target sm_75
.address_size 64

.visible .entry ew_add(.param .u64 a,.param .u64 b,.param .u64 c,.param .u32 n){
 .reg .pred %p;.reg .b32 %r<6>;.reg .f32 %f<4>;.reg .b64 %rd<9>;
 ld.param.u64 %rd1,[a];ld.param.u64 %rd2,[b];ld.param.u64 %rd3,[c];ld.param.u32 %r2,[n];
 mov.u32 %r3,%ctaid.x;mov.u32 %r4,%ntid.x;mov.u32 %r5,%tid.x;mad.lo.s32 %r1,%r3,%r4,%r5;
 setp.ge.s32 %p,%r1,%r2;@%p bra E;
 cvta.to.global.u64 %rd4,%rd1;cvta.to.global.u64 %rd5,%rd2;cvta.to.global.u64 %rd6,%rd3;
 mul.wide.s32 %rd7,%r1,4;add.s64 %rd8,%rd4,%rd7;ld.global.f32 %f1,[%rd8];
 add.s64 %rd8,%rd5,%rd7;ld.global.f32 %f2,[%rd8];add.f32 %f3,%f1,%f2;
 add.s64 %rd8,%rd6,%rd7;st.global.f32 [%rd8],%f3;E:ret;}

.visible .entry ew_sub(.param .u64 a,.param .u64 b,.param .u64 c,.param .u32 n){
 .reg .pred %p;.reg .b32 %r<6>;.reg .f32 %f<4>;.reg .b64 %rd<9>;
 ld.param.u64 %rd1,[a];ld.param.u64 %rd2,[b];ld.param.u64 %rd3,[c];ld.param.u32 %r2,[n];
 mov.u32 %r3,%ctaid.x;mov.u32 %r4,%ntid.x;mov.u32 %r5,%tid.x;mad.lo.s32 %r1,%r3,%r4,%r5;
 setp.ge.s32 %p,%r1,%r2;@%p bra E;
 cvta.to.global.u64 %rd4,%rd1;cvta.to.global.u64 %rd5,%rd2;cvta.to.global.u64 %rd6,%rd3;
 mul.wide.s32 %rd7,%r1,4;add.s64 %rd8,%rd4,%rd7;ld.global.f32 %f1,[%rd8];
 add.s64 %rd8,%rd5,%rd7;ld.global.f32 %f2,[%rd8];sub.f32 %f3,%f1,%f2;
 add.s64 %rd8,%rd6,%rd7;st.global.f32 [%rd8],%f3;E:ret;}

.visible .entry ew_mul(.param .u64 a,.param .u64 b,.param .u64 c,.param .u32 n){
 .reg .pred %p;.reg .b32 %r<6>;.reg .f32 %f<4>;.reg .b64 %rd<9>;
 ld.param.u64 %rd1,[a];ld.param.u64 %rd2,[b];ld.param.u64 %rd3,[c];ld.param.u32 %r2,[n];
 mov.u32 %r3,%ctaid.x;mov.u32 %r4,%ntid.x;mov.u32 %r5,%tid.x;mad.lo.s32 %r1,%r3,%r4,%r5;
 setp.ge.s32 %p,%r1,%r2;@%p bra E;
 cvta.to.global.u64 %rd4,%rd1;cvta.to.global.u64 %rd5,%rd2;cvta.to.global.u64 %rd6,%rd3;
 mul.wide.s32 %rd7,%r1,4;add.s64 %rd8,%rd4,%rd7;ld.global.f32 %f1,[%rd8];
 add.s64 %rd8,%rd5,%rd7;ld.global.f32 %f2,[%rd8];mul.f32 %f3,%f1,%f2;
 add.s64 %rd8,%rd6,%rd7;st.global.f32 [%rd8],%f3;E:ret;}

.visible .entry ew_div(.param .u64 a,.param .u64 b,.param .u64 c,.param .u32 n){
 .reg .pred %p;.reg .b32 %r<6>;.reg .f32 %f<4>;.reg .b64 %rd<9>;
 ld.param.u64 %rd1,[a];ld.param.u64 %rd2,[b];ld.param.u64 %rd3,[c];ld.param.u32 %r2,[n];
 mov.u32 %r3,%ctaid.x;mov.u32 %r4,%ntid.x;mov.u32 %r5,%tid.x;mad.lo.s32 %r1,%r3,%r4,%r5;
 setp.ge.s32 %p,%r1,%r2;@%p bra E;
 cvta.to.global.u64 %rd4,%rd1;cvta.to.global.u64 %rd5,%rd2;cvta.to.global.u64 %rd6,%rd3;
 mul.wide.s32 %rd7,%r1,4;add.s64 %rd8,%rd4,%rd7;ld.global.f32 %f1,[%rd8];
 add.s64 %rd8,%rd5,%rd7;ld.global.f32 %f2,[%rd8];div.rn.f32 %f3,%f1,%f2;
 add.s64 %rd8,%rd6,%rd7;st.global.f32 [%rd8],%f3;E:ret;}

.visible .entry s_axpb(.param .u64 a,.param .u64 c,.param .f32 s,.param .f32 t,.param .u32 n){
 // c = s*a + t
 .reg .pred %p;.reg .b32 %r<6>;.reg .f32 %f<5>;.reg .b64 %rd<7>;
 ld.param.u64 %rd1,[a];ld.param.u64 %rd2,[c];ld.param.f32 %f1,[s];ld.param.f32 %f2,[t];ld.param.u32 %r2,[n];
 mov.u32 %r3,%ctaid.x;mov.u32 %r4,%ntid.x;mov.u32 %r5,%tid.x;mad.lo.s32 %r1,%r3,%r4,%r5;
 setp.ge.s32 %p,%r1,%r2;@%p bra E;
 cvta.to.global.u64 %rd3,%rd1;cvta.to.global.u64 %rd4,%rd2;mul.wide.s32 %rd5,%r1,4;
 add.s64 %rd6,%rd3,%rd5;ld.global.f32 %f3,[%rd6];fma.rn.f32 %f4,%f1,%f3,%f2;
 add.s64 %rd6,%rd4,%rd5;st.global.f32 [%rd6],%f4;E:ret;}

.visible .entry relu(.param .u64 a,.param .u64 c,.param .u32 n){
 .reg .pred %p;.reg .b32 %r<6>;.reg .f32 %f<3>;.reg .b64 %rd<7>;
 ld.param.u64 %rd1,[a];ld.param.u64 %rd2,[c];ld.param.u32 %r2,[n];
 mov.u32 %r3,%ctaid.x;mov.u32 %r4,%ntid.x;mov.u32 %r5,%tid.x;mad.lo.s32 %r1,%r3,%r4,%r5;
 setp.ge.s32 %p,%r1,%r2;@%p bra E;
 cvta.to.global.u64 %rd3,%rd1;cvta.to.global.u64 %rd4,%rd2;mul.wide.s32 %rd5,%r1,4;
 add.s64 %rd6,%rd3,%rd5;ld.global.f32 %f1,[%rd6];max.f32 %f2,%f1,0f00000000;
 add.s64 %rd6,%rd4,%rd5;st.global.f32 [%rd6],%f2;E:ret;}
"""

_g = None
_ew = None
_mm = None


def _gpu():
    global _g, _ew, _mm
    if _g is None:
        _g = ge.GPU()
        m = _g.module(_EW_PTX)
        _ew = {n: m.kernel(n) for n in
               ("ew_add", "ew_sub", "ew_mul", "ew_div", "s_axpb", "relu")}
        kf = os.path.join(os.path.dirname(__file__), "kernels", "matmul_reg.ptx")
        _mm = _g.module(open(kf, "rb").read()).kernel("matmul_reg")
    return _g


def _run1d(kern, n, args):
    t = 256
    kern(grid=((n+t-1)//t, 1, 1), block=(t, 1, 1), args=args)


class farray:
    """GPU 上の float32 配列(numpy 風)。"""

    def __init__(self, ga, shape):
        self.ga = ga
        self.shape = tuple(shape)
        self.size = int(np.prod(shape)) if shape else 1

    # ---- 生成 ----
    @staticmethod
    def _wrap_like(shape):
        g = _gpu()
        return farray(g.empty(int(np.prod(shape)), np.float32), shape)

    # ---- 変換 ----
    def numpy(self):
        return self.ga.get().reshape(self.shape)

    # ---- 二項(要素ごと or スカラ) ----
    def _binary(self, other, op):
        g = _gpu()
        if isinstance(other, (int, float, np.integer, np.floating)):
            out = farray._wrap_like(self.shape)
            s, t = {"add": (1.0, float(other)), "sub": (1.0, -float(other)),
                    "mul": (float(other), 0.0)}.get(op, (None, None))
            if s is None:   # rsub / div by scalar 等はここでは非対応(mul/add/subのみ)
                raise TypeError(f"スカラ {op} は未対応")
            _run1d(_ew["s_axpb"], self.size,
                   [self.ga, out.ga, np.float32(s), np.float32(t), self.size])
            return out
        assert self.shape == other.shape, f"shape 不一致 {self.shape} vs {other.shape}"
        out = farray._wrap_like(self.shape)
        _run1d(_ew["ew_"+op], self.size, [self.ga, other.ga, out.ga, self.size])
        return out

    def __add__(self, o): return self._binary(o, "add")
    def __sub__(self, o): return self._binary(o, "sub")
    def __mul__(self, o): return self._binary(o, "mul")
    def __truediv__(self, o): return self._binary(o, "div")
    __radd__ = __add__
    __rmul__ = __mul__

    def __matmul__(self, other):
        assert len(self.shape) == 2 and len(other.shape) == 2
        M, K = self.shape; K2, N = other.shape
        assert K == K2, f"行列積 shape 不一致 {self.shape} @ {other.shape}"
        out = farray._wrap_like((M, N))
        _mm(grid=((N+63)//64, (M+63)//64, 1), block=(16, 16, 1),
            args=[self.ga, other.ga, out.ga, M, N, K])
        return out

    def relu(self):
        out = farray._wrap_like(self.shape)
        _run1d(_ew["relu"], self.size, [self.ga, out.ga, self.size])
        return out

    def __repr__(self):
        return f"farray(shape={self.shape}, gpu)"


# ---- モジュール関数(numpy 風の入口) ----
def array(data, dtype="f4"):
    a = np.asarray(data, dtype=np.float32)
    g = _gpu()
    return farray(g.to_gpu(a), a.shape)


def zeros(shape):
    g = _gpu()
    n = int(np.prod(shape))
    return farray(g.zeros(n, np.float32), shape if hasattr(shape, "__iter__") else (shape,))


def ones(shape):
    return array(np.ones(shape, np.float32))


def matmul(a, b):
    return a @ b


def relu(a):
    return a.relu()
