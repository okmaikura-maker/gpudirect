"""
gpudirect.train — 順伝播・逆伝播・Adam をすべて GPU の手書き PTX で回す
MLP 学習エンジン。効率のため:
  - 行列積は register-blocking 版 (matmul_reg / matmul_nt / matmul_tn)
  - Adam は 1 カーネルに融合 (adam.ptx)
CPU に出るのは損失計算の softmax-CE(バッチ×クラスの小行列)だけ。

層 l の順伝播: Z = A @ W + b ; A_next = relu(Z) (最終層は linear)
逆伝播:
  dW = A^T @ dZ     (matmul_tn)
  db = colsum(dZ)   (bias_grad)
  dA = dZ @ W^T     (matmul_nt)
  dZ_prev = dA * (A>0)   (relu_bwd)
"""
import os
import numpy as np

from . import Device, Context, DeviceMemory, _driver as _d

_KDIR = os.path.join(os.path.dirname(__file__), "kernels")


class GpuMLPTrainer:
    def __init__(self, ctx, dims, seed=0):
        """dims: [in, h1, ..., out]。全結合 + ReLU(最終層 linear)。"""
        self.ctx = ctx
        self.dims = dims
        self.nl = len(dims) - 1
        self._k = {}
        for n in ("matmul_reg", "matmul_nt", "matmul_tn",
                  "bias_grad", "relu_bwd", "bias_relu", "adam"):
            self._k[n] = ctx.load_ptx(
                open(os.path.join(_KDIR, f"{n}.ptx"), "rb").read()).function(n)
        r = np.random.default_rng(seed)
        self.W, self.b = [], []
        self.mW, self.vW, self.mb, self.vb = [], [], [], []
        for i in range(self.nl):
            nin, nout = dims[i], dims[i+1]
            w = (r.standard_normal((nin, nout)) * np.sqrt(2.0/nin)).astype(np.float32)
            self.W.append(self._dev(w.ravel())); self.b.append(self._dev(np.zeros(nout, np.float32)))
            self.mW.append(self._zeros(nin*nout)); self.vW.append(self._zeros(nin*nout))
            self.mb.append(self._zeros(nout)); self.vb.append(self._zeros(nout))
        self.t = 0

    # --- device helpers ---
    def _dev(self, a):
        a = np.asarray(a, np.float32); m = self.ctx.to_device(a.ravel().tolist(), "f4"); m.dtype = "f4"; return m
    def _buf(self, n): m = self.ctx.malloc(n*4); m.dtype = "f4"; return m
    def _zeros(self, n): m = self._buf(n); m.memset(0); return m

    def _mm(self, A, B, C, M, N, K):     # C = A@B (register-block)
        self._k["matmul_reg"].launch(grid=((N+63)//64, (M+63)//64, 1), block=(16, 16, 1),
                                     args=[A, B, C, M, N, K], sync=False)
    def _mm_nt(self, A, B, C, M, N, K):  # C = A@B^T
        self._k["matmul_nt"].launch(grid=((N+15)//16, (M+15)//16, 1), block=(16, 16, 1),
                                    args=[A, B, C, M, N, K], sync=False)
    def _mm_tn(self, A, B, C, I, J, R):  # C = A^T@B
        self._k["matmul_tn"].launch(grid=((J+15)//16, (I+15)//16, 1), block=(16, 16, 1),
                                    args=[A, B, C, I, J, R], sync=False)

    def forward(self, Xdev, batch):
        """活性を device に保持しつつ順伝播。logits を host に返す。"""
        self.acts = [Xdev]           # A_0 = 入力
        cur = Xdev
        for i in range(self.nl):
            nin, nout = self.dims[i], self.dims[i+1]
            Z = self._buf(batch*nout)
            self._mm(cur, self.W[i], Z, batch, nout, nin)
            relu = 0 if i == self.nl-1 else 1
            self._k["bias_relu"].launch(grid=(batch*nout+255)//256, block=256,
                                        args=[Z, self.b[i], batch, nout, relu], sync=False)
            self.acts.append(Z); cur = Z
        _d.check(_d.cuCtxSynchronize(), "fwd")
        out = self.dims[-1]
        return np.array(cur.copy_to_host("f4", batch*out)).reshape(batch, out)

    def backward_and_step(self, dlogits, batch, lr):
        """dlogits(host [batch,out]) から GPU で逆伝播 + Adam。"""
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        bc1, bc2 = 1-b1**self.t, 1-b2**self.t
        dZ = self._dev(dlogits.ravel())
        for i in reversed(range(self.nl)):
            nin, nout = self.dims[i], self.dims[i+1]
            A = self.acts[i]
            dW = self._buf(nin*nout); db = self._buf(nout)
            self._mm_tn(A, dZ, dW, nin, nout, batch)          # dW = A^T @ dZ
            self._k["bias_grad"].launch(grid=(nout+127)//128, block=128,
                                        args=[dZ, db, batch, nout], sync=False)
            if i > 0:
                dA = self._buf(batch*nin)
                self._mm_nt(dZ, self.W[i], dA, batch, nin, nout)   # dA = dZ @ W^T
                self._k["relu_bwd"].launch(grid=(batch*nin+255)//256, block=256,
                                           args=[dA, self.acts[i], batch*nin], sync=False)
                dZ_next = dA
            # Adam: W, b
            self._adam(self.W[i], dW, self.mW[i], self.vW[i], nin*nout, lr, b1, b2, bc1, bc2, eps)
            self._adam(self.b[i], db, self.mb[i], self.vb[i], nout, lr, b1, b2, bc1, bc2, eps)
            if i > 0:
                dZ = dZ_next
        _d.check(_d.cuCtxSynchronize(), "bwd")

    def _adam(self, P, G, M, V, n, lr, b1, b2, bc1, bc2, eps):
        self._k["adam"].launch(grid=(n+255)//256, block=256,
                               args=[P, G, M, V, n, float(lr), float(b1), float(b2),
                                     float(bc1), float(bc2), float(eps)], sync=False)

    # weights を host に取り出す(検証・保存用)
    def get_weights(self):
        out = []
        for i in range(self.nl):
            nin, nout = self.dims[i], self.dims[i+1]
            W = np.array(self.W[i].copy_to_host("f4", nin*nout)).reshape(nin, nout)
            b = np.array(self.b[i].copy_to_host("f4", nout))
            out.append((W, b))
        return out
