"""
gpudirect.train_gpt — Transformer(GPT)の学習を GPU の手書き PTX で回す。
順伝播・逆伝播・Adam をすべて自作カーネルで実行(埋め込みの更新のみ host)。
NumpyGPT と同一構成(pre-LN / single-head / weight-tied, ReLU FFN)。

1 ステップ = 1 系列(長さ T)。重い計算(行列積・注意・LN・FFN の前後)は
すべて GPU 常駐。CPU に出るのは損失の softmax-CE と埋め込みの scatter だけ。
"""
import os
import math
import numpy as np

from . import Device, Context, DeviceMemory, _driver as _d

_KDIR = os.path.join(os.path.dirname(__file__), "kernels")


class GpuGPTTrainer:
    def __init__(self, ctx, cfg, weights, lr=2e-3, seed=0):
        self.ctx = ctx
        self.cfg = cfg
        self.d = cfg["d_model"]; self.L = cfg["n_layer"]
        self.V = cfg["vocab"]; self.block = cfg["block"]
        self.lr = lr
        self.scale = 1.0/math.sqrt(self.d)
        self._k = {}
        for n in ("matmul_reg", "matmul_nt", "matmul_tn", "bias_relu",
                  "bias_grad", "relu_bwd", "adam", "ln_fwd", "ln_bwd",
                  "softmax_causal", "softmax_bwd", "add_inplace"):
            self._k[n] = ctx.load_ptx(open(os.path.join(_KDIR, f"{n}.ptx"), "rb").read()).function(n)
        # device 常駐の重み(埋め込み以外)と Adam 状態
        self.dev_keys = [k for k in weights if k not in ("tok_emb", "pos_emb")]
        self.W = {k: self._dev(weights[k]) for k in self.dev_keys}
        self.M = {k: self._zeros(weights[k].size) for k in self.dev_keys}
        self.Vd = {k: self._zeros(weights[k].size) for k in self.dev_keys}
        self.shape = {k: np.asarray(weights[k]).shape for k in self.dev_keys}
        # 埋め込みは host(小さいので numpy + host Adam)
        self.tok = weights["tok_emb"].astype(np.float64).copy()
        self.pos = weights["pos_emb"].astype(np.float64).copy()
        self.mtok = np.zeros_like(self.tok); self.vtok = np.zeros_like(self.tok)
        self.mpos = np.zeros_like(self.pos); self.vpos = np.zeros_like(self.pos)
        self.t = 0

    def _dev(self, a):
        a = np.asarray(a, np.float32); m = self.ctx.to_device(a.ravel().tolist(), "f4"); m.dtype = "f4"; return m
    def _buf(self, n): m = self.ctx.malloc(n*4); m.dtype = "f4"; return m
    def _zeros(self, n): m = self._buf(n); m.memset(0); return m
    def _mm(self, A, B, C, M, N, K):
        self._k["matmul_reg"].launch(grid=((N+63)//64, (M+63)//64, 1), block=(16, 16, 1), args=[A, B, C, M, N, K], sync=False)
    def _mmnt(self, A, B, C, M, N, K):
        self._k["matmul_nt"].launch(grid=((N+15)//16, (M+15)//16, 1), block=(16, 16, 1), args=[A, B, C, M, N, K], sync=False)
    def _mmtn(self, A, B, C, I, J, R):
        self._k["matmul_tn"].launch(grid=((J+15)//16, (I+15)//16, 1), block=(16, 16, 1), args=[A, B, C, I, J, R], sync=False)
    def _bias(self, X, b, M, N, relu):
        self._k["bias_relu"].launch(grid=(M*N+255)//256, block=256, args=[X, b, M, N, relu], sync=False)
    def _bgrad(self, dY, db, R, N):
        self._k["bias_grad"].launch(grid=(N+127)//128, block=128, args=[dY, db, R, N], sync=False)
    def _lnf(self, X, g, b, Y, mean, istd, M, N):
        self._k["ln_fwd"].launch(grid=(M+127)//128, block=128, args=[X, g, b, Y, mean, istd, M, N], sync=False)
    def _lnb(self, dY, X, g, mean, istd, dX, dg, db, M, N):
        dg.memset(0); db.memset(0)
        self._k["ln_bwd"].launch(grid=(M+127)//128, block=128, args=[dY, X, g, mean, istd, dX, dg, db, M, N], sync=False)
    def _add(self, A, B, n):
        self._k["add_inplace"].launch(grid=(n+255)//256, block=256, args=[A, B, n], sync=False)

    def step(self, tokens, tgt):
        d, V, L, T = self.d, self.V, self.L, len(tokens)
        self.t += 1
        w = self.W
        x0 = (self.tok[tokens] + self.pos[:T]).astype(np.float32)   # host 埋め込み
        x = self._dev(x0)                       # 残差ストリーム
        cache = []
        for l in range(L):
            p = f"b{l}."
            xin = self._buf(T*d); xin.copy_from_dev(x, T*d*4)         # LN1 入力を保存
            a = self._buf(T*d); m1 = self._buf(T); i1 = self._buf(T)
            self._lnf(xin, w[p+"ln1_g"], w[p+"ln1_b"], a, m1, i1, T, d)
            Q = self._buf(T*d); Kk = self._buf(T*d); Vv = self._buf(T*d)
            self._mm(a, w[p+"Wq"], Q, T, d, d); self._bias(Q, w[p+"bq"], T, d, 0)
            self._mm(a, w[p+"Wk"], Kk, T, d, d); self._bias(Kk, w[p+"bk"], T, d, 0)
            self._mm(a, w[p+"Wv"], Vv, T, d, d); self._bias(Vv, w[p+"bv"], T, d, 0)
            P = self._buf(T*T); self._mmnt(Q, Kk, P, T, T, d)
            self._k["softmax_causal"].launch(grid=(T+127)//128, block=128, args=[P, T, float(self.scale)], sync=False)
            O = self._buf(T*d); self._mm(P, Vv, O, T, d, T)
            proj = self._buf(T*d); self._mm(O, w[p+"Wo"], proj, T, d, d); self._bias(proj, w[p+"bo"], T, d, 0)
            self._add(x, proj, T*d)                                  # x1 = x + attn
            x1 = self._buf(T*d); x1.copy_from_dev(x, T*d*4)          # LN2 入力を保存
            a2 = self._buf(T*d); m2 = self._buf(T); i2 = self._buf(T)
            self._lnf(x1, w[p+"ln2_g"], w[p+"ln2_b"], a2, m2, i2, T, d)
            h = self._buf(T*4*d); self._mm(a2, w[p+"W1"], h, T, 4*d, d); self._bias(h, w[p+"b1"], T, 4*d, 1)
            f = self._buf(T*d); self._mm(h, w[p+"W2"], f, T, d, 4*d); self._bias(f, w[p+"b2"], T, d, 0)
            self._add(x, f, T*d)                                     # x2 = x1 + f
            cache.append(dict(xin=xin, a=a, m1=m1, i1=i1, Q=Q, K=Kk, V=Vv, P=P, O=O,
                              x1=x1, a2=a2, m2=m2, i2=i2, h=h))
        # 最終 LN + 重み共有ロジット
        xfin = self._buf(T*d); xfin.copy_from_dev(x, T*d*4)
        xf = self._buf(T*d); mf = self._buf(T); iff = self._buf(T)
        self._lnf(xfin, w["lnf_g"], w["lnf_b"], xf, mf, iff, T, d)
        dtok_dev = self._dev(self.tok)                              # tok_emb を device に(logits 用)
        logits = self._buf(T*V); self._mmnt(xf, dtok_dev, logits, T, V, d)
        _d.check(_d.cuCtxSynchronize(), "fwd")
        lg = np.array(logits.copy_to_host("f4", T*V)).reshape(T, V)

        # ---- 損失 & dlogits (host, 全 T 位置平均) ----
        tgt = np.asarray(tgt)
        z = lg - lg.max(1, keepdims=True); e = np.exp(z); sm = e/e.sum(1, keepdims=True)
        idx = np.arange(T)
        loss = -np.log(sm[idx, tgt] + 1e-12).mean()
        dl = sm.copy(); dl[idx, tgt] -= 1; dl /= T
        dl = dl.astype(np.float32)

        # ---- backward ----
        dlog = self._dev(dl)
        # logits = xf @ tok^T : dxf = dlog @ tok ; dtok_tie = dlog^T @ xf
        dxf = self._buf(T*d); self._mm(dlog, dtok_dev, dxf, T, d, V)
        dtok_tie = self._buf(V*d); self._mmtn(dlog, xf, dtok_tie, V, d, T)
        # LNf backward
        dxs = self._buf(T*d); dgf = self._buf(d); dbf = self._buf(d)
        self._lnb(dxf, xfin, w["lnf_g"], mf, iff, dxs, dgf, dbf, T, d)
        self._adam_dev("lnf_g", dgf); self._adam_dev("lnf_b", dbf)
        dx = dxs

        grads = {}                                  # device grad buffers for adam
        for l in reversed(range(L)):
            p = f"b{l}."; c = cache[l]
            # FFN
            df = dx
            dW2 = self._buf(4*d*d); self._mmtn(c["h"], df, dW2, 4*d, d, T)
            db2 = self._buf(d); self._bgrad(df, db2, T, d)
            dh = self._buf(T*4*d); self._mmnt(df, w[p+"W2"], dh, T, 4*d, d)
            self._k["relu_bwd"].launch(grid=(T*4*d+255)//256, block=256, args=[dh, c["h"], T*4*d], sync=False)
            dW1 = self._buf(d*4*d); self._mmtn(c["a2"], dh, dW1, d, 4*d, T)
            db1 = self._buf(4*d); self._bgrad(dh, db1, T, 4*d)
            da2 = self._buf(T*d); self._mmnt(dh, w[p+"W1"], da2, T, d, 4*d)
            # LN2 backward + residual
            dx1 = self._buf(T*d); dg2 = self._buf(d); dbb2 = self._buf(d)
            self._lnb(da2, c["x1"], w[p+"ln2_g"], c["m2"], c["i2"], dx1, dg2, dbb2, T, d)
            self._add(dx1, dx, T*d)                 # dx1 = ln2_path + dx2
            # attention
            dattn = dx1
            dWo = self._buf(d*d); self._mmtn(c["O"], dattn, dWo, d, d, T)
            dbo = self._buf(d); self._bgrad(dattn, dbo, T, d)
            dO = self._buf(T*d); self._mmnt(dattn, w[p+"Wo"], dO, T, d, d)
            dP = self._buf(T*T); self._mmnt(dO, c["V"], dP, T, T, d)
            dV = self._buf(T*d); self._mmtn(c["P"], dO, dV, T, d, T)
            dS = self._buf(T*T); self._k["softmax_bwd"].launch(grid=(T+127)//128, block=128, args=[c["P"], dP, dS, T, float(self.scale)], sync=False)
            dQ = self._buf(T*d); self._mm(dS, c["K"], dQ, T, d, T)
            dK = self._buf(T*d); self._mmtn(dS, c["Q"], dK, T, d, T)
            dWq = self._buf(d*d); self._mmtn(c["a"], dQ, dWq, d, d, T); dbq = self._buf(d); self._bgrad(dQ, dbq, T, d)
            dWk = self._buf(d*d); self._mmtn(c["a"], dK, dWk, d, d, T); dbk = self._buf(d); self._bgrad(dK, dbk, T, d)
            dWv = self._buf(d*d); self._mmtn(c["a"], dV, dWv, d, d, T); dbv = self._buf(d); self._bgrad(dV, dbv, T, d)
            da = self._buf(T*d); self._mmnt(dQ, w[p+"Wq"], da, T, d, d)
            dak = self._buf(T*d); self._mmnt(dK, w[p+"Wk"], dak, T, d, d); self._add(da, dak, T*d)
            dav = self._buf(T*d); self._mmnt(dV, w[p+"Wv"], dav, T, d, d); self._add(da, dav, T*d)
            # LN1 backward + residual
            dxblk = self._buf(T*d); dg1 = self._buf(d); dbb1 = self._buf(d)
            self._lnb(da, c["xin"], w[p+"ln1_g"], c["m1"], c["i1"], dxblk, dg1, dbb1, T, d)
            self._add(dxblk, dx1, T*d)              # dx = ln1_path + dx1
            # Adam for this layer's device weights
            self._adam_dev(p+"Wq", dWq); self._adam_dev(p+"bq", dbq)
            self._adam_dev(p+"Wk", dWk); self._adam_dev(p+"bk", dbk)
            self._adam_dev(p+"Wv", dWv); self._adam_dev(p+"bv", dbv)
            self._adam_dev(p+"Wo", dWo); self._adam_dev(p+"bo", dbo)
            self._adam_dev(p+"ln2_g", dg2); self._adam_dev(p+"ln2_b", dbb2)
            self._adam_dev(p+"W1", dW1); self._adam_dev(p+"b1", db1)
            self._adam_dev(p+"W2", dW2); self._adam_dev(p+"b2", db2)
            self._adam_dev(p+"ln1_g", dg1); self._adam_dev(p+"ln1_b", dbb1)
            dx = dxblk
        _d.check(_d.cuCtxSynchronize(), "bwd")

        # ---- 埋め込みの勾配 (host Adam) ----
        dtok = np.array(dtok_tie.copy_to_host("f4", V*d)).reshape(V, d).astype(np.float64)
        dx0 = np.array(dx.copy_to_host("f4", T*d)).reshape(T, d).astype(np.float64)
        np.add.at(dtok, np.array(tokens), dx0)      # 埋め込みルックアップの勾配
        dpos = np.zeros_like(self.pos); dpos[:T] += dx0
        self._host_adam(self.tok, dtok, self.mtok, self.vtok)
        self._host_adam(self.pos, dpos, self.mpos, self.vpos)
        return loss

    def _adam_dev(self, key, G):
        b1, b2, eps = 0.9, 0.999, 1e-8
        bc1, bc2 = 1-b1**self.t, 1-b2**self.t
        n = int(np.prod(self.shape[key]))
        self._k["adam"].launch(grid=(n+255)//256, block=256,
            args=[self.W[key], G, self.M[key], self.Vd[key], n,
                  float(self.lr), float(b1), float(b2), float(bc1), float(bc2), float(eps)], sync=False)

    def _host_adam(self, P, G, M, V):
        b1, b2, eps = 0.9, 0.999, 1e-8
        bc1, bc2 = 1-b1**self.t, 1-b2**self.t
        M[:] = b1*M + (1-b1)*G; V[:] = b2*V + (1-b2)*G**2
        P -= self.lr*(M/bc1)/(np.sqrt(V/bc2)+eps)

    def export(self):
        w = {}
        for k in self.dev_keys:
            sh = self.shape[k]; n = int(np.prod(sh))
            w[k] = np.array(self.W[k].copy_to_host("f4", n)).reshape(sh)
        w["tok_emb"] = self.tok.astype(np.float32); w["pos_emb"] = self.pos.astype(np.float32)
        return w
