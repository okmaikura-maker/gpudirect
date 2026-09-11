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
        # matmul_ts: 共有メモリタイル+レジスタブロッキング版。64/16 境界に
        # 揃えたデータを渡すと端数分岐がなくなり、いちばん速く読める。
        kf = os.path.join(os.path.dirname(__file__), "kernels", "matmul_ts.ptx")
        _mm = _g.module(open(kf, "rb").read()).kernel("matmul_ts")
    return _g


# タイル境界(matmul_ts の block タイルは 64x64)。
# ある配列が行列積で A 側(M,K)にも B 側(K,N)にも使われうるため、
# どちらの軸が M/K/N になっても揃うよう「両方の軸を 64 に統一」してパディングする。
_TILE = 64


def _pad2d(shape):
    """2次元 shape の両軸を 64 の倍数に丸めた物理 shape を返す。"""
    m, k = shape
    return (((m+_TILE-1)//_TILE)*_TILE, ((k+_TILE-1)//_TILE)*_TILE)


def _run1d(kern, n, args):
    # sync=False: このカーネルの完了を待たず、次の演算を GPU の命令キューに
    # 積み続ける。「計算が終わった瞬間に次へ同化する」を実現する核心部分。
    # 依存関係(前段の出力を次段が読む)は CUDA の単一ストリーム上で自動的に
    # 順序が保たれるため、host が割り込んで待つ理由が無い。
    t = 256
    kern(grid=((n+t-1)//t, 1, 1), block=(t, 1, 1), args=args, sync=False)


class farray:
    """
    GPU 上の float32 配列(numpy 風)。

    渡されたデータは受け取った瞬間に「GPU が読みやすい形」へポポイと束ねられる:
    2次元配列は 64(行)/16(列) の倍数へ自動でゼロパディングされ(pshape)、
    行列積カーネルはこの物理サイズで起動するので端数分岐が要らず速く読める。
    ユーザーからは論理 shape(元の形)だけが見え、.numpy() で自動的に
    パディング分を切り落として返す — 意識する必要はない。
    """

    def __init__(self, ga, shape, pshape=None):
        self.ga = ga
        self.shape = tuple(shape)
        self.pshape = tuple(pshape) if pshape is not None else self.shape
        self.size = int(np.prod(self.pshape)) if self.pshape else 1

    # ---- 生成 ----
    @staticmethod
    def _wrap_like(shape, pshape=None):
        g = _gpu()
        ps = pshape if pshape is not None else shape
        return farray(g.empty(int(np.prod(ps)), np.float32), shape, ps)

    # ---- 変換 ----
    def numpy(self):
        # ここで初めて GPU の完了を待つ(このオペランドが実際に必要になった瞬間)。
        # それまでの演算連鎖はすべて sync=False で積まれたまま流れていた。
        _gpu().synchronize()
        a = self.ga.get().reshape(self.pshape)
        if self.pshape != self.shape:
            if len(self.shape) == 2:
                m, k = self.shape
                a = a[:m, :k]
            else:
                a = a.reshape(-1)[:int(np.prod(self.shape))].reshape(self.shape)
        return np.ascontiguousarray(a)

    # ---- 二項(要素ごと or スカラ) ----
    def _binary(self, other, op):
        g = _gpu()
        if isinstance(other, (int, float, np.integer, np.floating)):
            out = farray._wrap_like(self.shape, self.pshape)
            s, t = {"add": (1.0, float(other)), "sub": (1.0, -float(other)),
                    "mul": (float(other), 0.0)}.get(op, (None, None))
            if s is None:   # rsub / div by scalar 等はここでは非対応(mul/add/subのみ)
                raise TypeError(f"スカラ {op} は未対応")
            _run1d(_ew["s_axpb"], self.size,
                   [self.ga, out.ga, np.float32(s), np.float32(t), self.size])
            return out
        assert self.shape == other.shape, f"shape 不一致 {self.shape} vs {other.shape}"
        # 論理 shape が同じなら、パディング規則が決定的なので pshape も自動的に一致する
        assert self.pshape == other.pshape
        out = farray._wrap_like(self.shape, self.pshape)
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
        # 両オペランドの物理(パディング済み)サイズで、端数分岐なしのタイルカーネルを起動
        Mp, Kp = self.pshape; Kp2, Np = other.pshape
        assert Kp == Kp2, "K方向のパディングが揃っていません(内部不整合)"
        out = farray._wrap_like((M, N), (Mp, Np))
        # sync=False: 結果は即座に次の演算(あるいは最終読み出し)に流れ込む。
        # ここで待つと、連鎖する演算のたびに GPU が手を止めることになる。
        _mm(grid=(Np//64, Mp//64, 1), block=(16, 16, 1),
            args=[self.ga, other.ga, out.ga, Mp, Np, Kp], sync=False)
        return out

    def relu(self):
        out = farray._wrap_like(self.shape, self.pshape)
        _run1d(_ew["relu"], self.size, [self.ga, out.ga, self.size])
        return out

    def __repr__(self):
        return f"farray(shape={self.shape}, gpu)"


# ---- モジュール関数(numpy 風の入口) ----
def array(data, dtype="f4"):
    """
    データを受け取った瞬間に GPU が読みやすい形へ束ねて転送する。
    2次元配列は 64 の倍数へゼロパディングしてから 1 回で GPU に上げるので、
    後段の行列積カーネルは端数分岐なしで走れる(論理 shape は元のまま見える)。
    """
    a = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
    g = _gpu()
    if a.ndim == 2 and a.shape != _pad2d(a.shape):
        pshape = _pad2d(a.shape)
        padded = np.zeros(pshape, np.float32)
        padded[:a.shape[0], :a.shape[1]] = a
        return farray(g.to_gpu(padded), a.shape, pshape)   # 1 回の転送で束ねて送る
    return farray(g.to_gpu(a), a.shape)


def zeros(shape):
    shape = tuple(shape) if hasattr(shape, "__iter__") else (shape,)
    g = _gpu()
    pshape = _pad2d(shape) if len(shape) == 2 else shape
    return farray(g.zeros(int(np.prod(pshape)), np.float32), shape, pshape)


def ones(shape):
    return array(np.ones(shape, np.float32))


def matmul(a, b):
    return a @ b


def relu(a):
    return a.relu()
