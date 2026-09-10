"""
gpudirect.transformer — 手書き PTX カーネルだけで動く Transformer(GPT)の
GPU 推論エンジン。CuPy / PyTorch 不使用。

構成 (pre-LN, single-head, weight-tied):
  x = Emb[token] + Pos
  for each block:
      a = LN1(x); (Q,K,V) = a@Wq,a@Wk,a@Wv (+bias)
      S = softmax_causal(Q@K^T / sqrt(d)); O = S@V
      x = x + (O@Wo + bo)
      h = LN2(x); x = x + (relu(h@W1+b1)@W2 + b2)
  x = LNf(x); logits = x @ Emb^T

行列積・注意・LayerNorm・softmax・残差、すべて自作 PTX で GPU 上に流す。
学習は CPU 側 (numpy) で行い、重み(numpy 配列)を渡して推論する。
"""
import os
import math

import numpy as np

from . import Device, Context, _driver as _d

_KDIR = os.path.join(os.path.dirname(__file__), "kernels")
_TX = _TY = 16


class GpuTransformer:
    def __init__(self, ctx, weights, cfg):
        """
        weights: numpy 配列の dict。キーは下記参照。
        cfg: {"d_model","n_layer","n_head"(=1),"vocab","block"} 。
        """
        self.ctx = ctx
        self.cfg = cfg
        self.d = cfg["d_model"]
        self.L = cfg["n_layer"]
        self.V = cfg["vocab"]
        self._k = {}
        for name in ("matmul", "matmul_reg", "matmul_ts", "matmul_nt",
                     "layernorm", "softmax_causal", "add_inplace", "bias_relu",
                     "attention_fused", "attention_warp"):
            self._k[name] = ctx.load_ptx(
                open(os.path.join(_KDIR, f"{name}.ptx"), "rb").read()
            ).function(name)
        # 重みを GPU に常駐
        self.w = {k: self._dev(v) for k, v in weights.items()}
        self.wnp = weights

    # --- helpers -------------------------------------------------------------
    def _dev(self, a):
        a = np.asarray(a, np.float32)
        m = self.ctx.to_device(a.ravel().tolist(), "f4")
        m.dtype = "f4"
        return m

    def _buf(self, nfloat):
        m = self.ctx.malloc(nfloat * 4)
        m.dtype = "f4"
        return m

    def _mm(self, A, B, C, M, N, K):
        # レジスタブロッキング版 (タイル化なし / 1 スレッド 4x4 出力)
        self._k["matmul_reg"].launch(
            grid=((N + 63)//64, (M + 63)//64, 1),
            block=(16, 16, 1), args=[A, B, C, M, N, K], sync=False)

    def _mm_nt(self, A, B, C, M, N, K):
        self._k["matmul_nt"].launch(
            grid=((N + _TX - 1)//_TX, (M + _TY - 1)//_TY, 1),
            block=(_TX, _TY, 1), args=[A, B, C, M, N, K], sync=False)

    def _bias(self, X, b, M, N, relu):
        self._k["bias_relu"].launch(
            grid=(M*N + 255)//256, block=256, args=[X, b, M, N, relu], sync=False)

    def _ln(self, X, g, b, M, N):
        self._k["layernorm"].launch(
            grid=(M + 127)//128, block=128, args=[X, g, b, M, N], sync=False)

    def _softmax(self, S, T, scale):
        self._k["softmax_causal"].launch(
            grid=(T + 127)//128, block=128, args=[S, T, float(scale)], sync=False)

    def _add(self, A, B, n):
        self._k["add_inplace"].launch(
            grid=(n + 255)//256, block=256, args=[A, B, n], sync=False)

    # --- forward -------------------------------------------------------------
    def forward_logits(self, tokens):
        """tokens: int list (長さ T)。最終位置のロジット(numpy [V]) を返す。"""
        d, V, T = self.d, self.V, len(tokens)
        Emb = self.wnp["tok_emb"]           # [V,d]
        Pos = self.wnp["pos_emb"]           # [block,d]
        x0 = (Emb[tokens] + Pos[:T]).astype(np.float32)   # 埋め込みは host で
        x = self._dev(x0)

        tmp = self._buf(T*d)                 # LN 出力
        Q = self._buf(T*d); Kk = self._buf(T*d); Vv = self._buf(T*d)
        S = self._buf(T*T)
        O = self._buf(T*d)
        proj = self._buf(T*d)
        hbuf = self._buf(T*self.d*4)         # FFN hidden [T,4d]
        h2 = self._buf(T*d)
        scale = 1.0 / math.sqrt(d)

        for l in range(self.L):
            w = lambda s: self.w[f"b{l}.{s}"]
            # --- attention ---
            tmp.copy_from_dev(x, T*d*4)
            self._ln(tmp, w("ln1_g"), w("ln1_b"), T, d)
            self._mm(tmp, w("Wq"), Q, T, d, d); self._bias(Q, w("bq"), T, d, 0)
            self._mm(tmp, w("Wk"), Kk, T, d, d); self._bias(Kk, w("bk"), T, d, 0)
            self._mm(tmp, w("Wv"), Vv, T, d, d); self._bias(Vv, w("bv"), T, d, 0)
            # ワープ融合注意: 1 ワープ=1 クエリ行。Q@K^T+softmax+@V を 1 起動、
            # 中間 S 不要、d を 32 レーンで分担 (shfl リダクション)。
            wpb = 4
            self._k["attention_warp"].launch(
                grid=(T + wpb - 1)//wpb, block=wpb*32,
                args=[Q, Kk, Vv, O, T, d, float(scale)], sync=False)
            self._mm(O, w("Wo"), proj, T, d, d); self._bias(proj, w("bo"), T, d, 0)
            self._add(x, proj, T*d)              # 残差
            # --- FFN ---
            tmp.copy_from_dev(x, T*d*4)
            self._ln(tmp, w("ln2_g"), w("ln2_b"), T, d)
            self._mm(tmp, w("W1"), hbuf, T, 4*d, d); self._bias(hbuf, w("b1"), T, 4*d, 1)
            self._mm(hbuf, w("W2"), h2, T, d, 4*d); self._bias(h2, w("b2"), T, d, 0)
            self._add(x, h2, T*d)                # 残差

        self._ln(x, self.w["lnf_g"], self.w["lnf_b"], T, d)
        logits = self._buf(T*V)
        self._mm_nt(x, self.w["tok_emb"], logits, T, V, d)   # weight tying
        _d.check(_d.cuCtxSynchronize(), "tf forward")
        allv = np.array(logits.copy_to_host("f4", T*V)).reshape(T, V)
        return allv[-1]

    def generate(self, prompt_ids, n_new, temperature=0.8, top_k=None, seed=0):
        rng = np.random.default_rng(seed)
        ids = list(prompt_ids)
        block = self.cfg["block"]
        for _ in range(n_new):
            ctx_ids = ids[-block:]
            logits = self.forward_logits(ctx_ids).astype(np.float64)
            logits /= max(temperature, 1e-6)
            if top_k:
                thr = np.sort(logits)[-top_k]
                logits[logits < thr] = -1e9
            logits -= logits.max()
            p = np.exp(logits); p /= p.sum()
            ids.append(int(rng.choice(len(p), p=p)))
        return ids
