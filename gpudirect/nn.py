"""
gpudirect.nn — 手書き PTX カーネル (matmul + bias_relu) だけで動く
多層パーセプトロン(MLP)の GPU 推論エンジン。

CuPy も PyTorch も使わない。全結合層の順伝播を
  Z = X @ W ; Z = ReLU/linear(Z + b)
という 2 カーネルの連鎖で GPU 上に流す。重みは一度 GPU に載せたら常駐。

学習は CPU 側(numpy 等)で行い、得た重みを load_layers() で流し込む想定。
"""
import os
import ctypes

from . import _driver as _d
from . import Device, Context, DeviceMemory

_KDIR = os.path.join(os.path.dirname(__file__), "kernels")


class GpuMLP:
    """全結合 + ReLU の MLP を GPU 上で順伝播させる。"""

    def __init__(self, ctx):
        self.ctx = ctx
        self._mm = ctx.load_ptx(
            open(os.path.join(_KDIR, "matmul.ptx"), "rb").read()).function("matmul")
        self._br = ctx.load_ptx(
            open(os.path.join(_KDIR, "bias_relu.ptx"), "rb").read()).function("bias_relu")
        self.layers = []          # [(W_dev, b_dev, in, out), ...]

    def load_layers(self, weights):
        """
        weights: [(W, b), ...] のリスト。
          W: 形 (in, out) を row-major で平坦化した float のシーケンス、
             もしくは W[i][j] の 2 次元リスト。
          b: 長さ out の float シーケンス。
        """
        self.layers = []
        for W, b in weights:
            W2 = self._as_2d(W)
            nin = len(W2)
            nout = len(W2[0])
            flat = [W2[i][j] for i in range(nin) for j in range(nout)]
            Wd = self.ctx.to_device(flat, "f4")
            bd = self.ctx.to_device(list(b), "f4")
            self.layers.append((Wd, bd, nin, nout))

    @staticmethod
    def _as_2d(W):
        # numpy 配列でも Python list でも受ける
        if hasattr(W, "tolist"):
            return W.tolist()
        if W and not isinstance(W[0], (list, tuple)):
            raise ValueError("W は (in,out) の 2 次元で渡してください")
        return [list(r) for r in W]

    def _matmul(self, X_dev, batch, nin, W_dev, nout):
        """C[batch,nout] = X[batch,nin] @ W[nin,nout] を GPU で計算。"""
        C = self.ctx.malloc(batch * nout * 4)
        C.dtype = "f4"
        tx, ty = 16, 16
        grid = ((nout + tx - 1) // tx, (batch + ty - 1) // ty, 1)
        self._mm.launch(grid=grid, block=(tx, ty, 1),
                        args=[X_dev, W_dev, C, batch, nout, nin], sync=False)
        return C

    def forward(self, X, batch):
        """
        X: 入力バッチ (batch*in_features の平坦 float シーケンス、または DeviceMemory)。
        戻り値: 出力ロジット(batch*out の list) と、各行の argmax クラス list。
        最終層は linear、それ以外は ReLU。
        """
        if isinstance(X, DeviceMemory):
            cur = X
        else:
            cur = self.ctx.to_device(list(X), "f4")

        last = len(self.layers) - 1
        for li, (Wd, bd, nin, nout) in enumerate(self.layers):
            cur = self._matmul(cur, batch, nin, Wd, nout)
            do_relu = 0 if li == last else 1
            total = batch * nout
            self._br.launch(grid=(total + 255) // 256, block=256,
                            args=[cur, bd, batch, nout, do_relu], sync=False)
        _d.check(_d.cuCtxSynchronize(), "forward sync")

        out_dim = self.layers[-1][3]
        logits = list(cur.copy_to_host("f4", batch * out_dim))
        preds = []
        for r in range(batch):
            row = logits[r * out_dim:(r + 1) * out_dim]
            preds.append(row.index(max(row)))
        return logits, preds


class FusedMLP:
    """
    順伝播の全層 + バイアス + ReLU + argmax を 1 カーネル・1 起動で実行する
    融合版 MLP。層間で GPU メモリへ書き戻さないため起動も往復もゼロ。
    (1 スレッド = 1 サンプル。最大層幅 <= 64。)
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self._fn = ctx.load_ptx(
            open(os.path.join(_KDIR, "mlp_fused.ptx"), "rb").read()
        ).function("mlp_fused")
        self._packed = False

    def load_layers(self, weights):
        """weights: [(W(in,out), b(out)), ...] を 1 本の連結バッファに詰めて常駐。"""
        W_flat, B_flat = [], []
        dims, woff, boff = [], [], []
        first = GpuMLP._as_2d(weights[0][0])
        dims.append(len(first))                     # in0
        wo = bo = 0
        for W, b in weights:
            W2 = GpuMLP._as_2d(W)
            nin, nout = len(W2), len(W2[0])
            woff.append(wo)
            boff.append(bo)
            for i in range(nin):
                for j in range(nout):
                    W_flat.append(W2[i][j])
            B_flat.extend(list(b))
            wo += nin * nout
            bo += nout
            dims.append(nout)                        # out_l

        self.L = len(weights)
        self.out_dim = dims[-1]
        self.dW = self.ctx.to_device(W_flat, "f4")
        self.dB = self.ctx.to_device(B_flat, "f4")
        self.dDims = self.ctx.to_device(dims, "i4")
        self.dWoff = self.ctx.to_device(woff, "i4")
        self.dBoff = self.ctx.to_device(boff, "i4")
        self.in_dim = dims[0]
        self._packed = True

    def forward(self, X, batch):
        """1 起動で全サンプルを推論。戻り値: (logits list, preds list)。"""
        assert self._packed, "load_layers を先に呼んでください"
        dX = X if isinstance(X, DeviceMemory) else self.ctx.to_device(list(X), "f4")
        dOut = self.ctx.malloc(batch * self.out_dim * 4); dOut.dtype = "f4"
        dPred = self.ctx.malloc(batch * 4); dPred.dtype = "i4"

        threads = 128
        blocks = (batch + threads - 1) // threads
        self._fn.launch(
            grid=blocks, block=threads,
            args=[dX, self.dW, self.dB, self.dDims, self.dWoff, self.dBoff,
                  dOut, dPred, batch, self.L])   # ← ここ 1 回だけ

        logits = list(dOut.copy_to_host("f4", batch * self.out_dim))
        preds = list(dPred.copy_to_host("i4", batch))
        return logits, preds
