"""
超重い本物の AI: 文字レベル GPT(Transformer)を手書き PTX カーネルで動かす。

- CPU(numpy)で小さな GPT を学習(自己注意 + FFN + LayerNorm、逆伝播も numpy)。
- 学習した重みを GPU に載せ、順伝播を全部自作 PTX で実行して文章生成。
- GPU の出力が CPU と一致することを確認。
- 最後に大きな設定で GPU 飽和ベンチ(FLOP/s・使用率)を測る。

    python llm_demo.py
CuPy / PyTorch 不使用。
"""
import math
import time
import subprocess

import numpy as np

import gpudirect as gd
from gpudirect.transformer import GpuTransformer

# ----------------------------------------------------------------------------
# 学習コーパス(構造のある短文。tiny GPT が続きを学べる程度に反復)
# ----------------------------------------------------------------------------
CORPUS = (
    "the gpu runs the kernel and the kernel runs the gpu. "
    "we write the ptx by hand and the driver builds the code. "
    "the tensor flows through the layers and the answer comes out. "
    "attention looks back at every token and picks what matters. "
) * 40


# ----------------------------------------------------------------------------
# numpy Transformer (GPU 版と同一構成: pre-LN / single-head / weight-tied)
# ----------------------------------------------------------------------------
class NumpyGPT:
    def __init__(self, V, d, n_layer, block, seed=0):
        r = np.random.default_rng(seed)
        s = 0.02
        self.cfg = dict(vocab=V, d_model=d, n_layer=n_layer, n_head=1, block=block)
        self.d, self.L, self.V, self.block = d, n_layer, V, block
        p = {}
        p["tok_emb"] = r.normal(0, s, (V, d))
        p["pos_emb"] = r.normal(0, s, (block, d))
        for l in range(n_layer):
            p[f"b{l}.ln1_g"] = np.ones(d); p[f"b{l}.ln1_b"] = np.zeros(d)
            for w in "qkv":
                p[f"b{l}.W{w}"] = r.normal(0, s, (d, d)); p[f"b{l}.b{w}"] = np.zeros(d)
            p[f"b{l}.Wo"] = r.normal(0, s, (d, d)); p[f"b{l}.bo"] = np.zeros(d)
            p[f"b{l}.ln2_g"] = np.ones(d); p[f"b{l}.ln2_b"] = np.zeros(d)
            p[f"b{l}.W1"] = r.normal(0, s, (d, 4*d)); p[f"b{l}.b1"] = np.zeros(4*d)
            p[f"b{l}.W2"] = r.normal(0, s, (4*d, d)); p[f"b{l}.b2"] = np.zeros(d)
        p["lnf_g"] = np.ones(d); p["lnf_b"] = np.zeros(d)
        self.p = {k: v.astype(np.float64) for k, v in p.items()}
        self.scale = 1.0 / math.sqrt(d)

    # --- 基本演算 ---
    @staticmethod
    def ln_f(x, g, b):
        mu = x.mean(-1, keepdims=True); xc = x - mu
        var = (xc**2).mean(-1, keepdims=True); istd = 1/np.sqrt(var+1e-5)
        xn = xc*istd
        return xn*g + b, (xn, istd, g)

    @staticmethod
    def ln_b(dout, cache):
        xn, istd, g = cache
        N = xn.shape[-1]
        dg = (dout*xn).sum((0, 1)); db = dout.sum((0, 1))
        dxn = dout*g
        dx = (istd/N)*(N*dxn - dxn.sum(-1, keepdims=True)
                       - xn*(dxn*xn).sum(-1, keepdims=True))
        return dx, dg, db

    def forward(self, idx):
        # idx: [B,T]
        B, T = idx.shape
        d = self.d; p = self.p
        cache = {}
        x = p["tok_emb"][idx] + p["pos_emb"][:T]        # [B,T,d]
        cache["idx"] = idx
        mask = np.triu(np.ones((T, T)), 1).astype(bool)  # j>i を隠す
        for l in range(self.L):
            pf = f"b{l}."
            a, c_ln1 = self.ln_f(x, p[pf+"ln1_g"], p[pf+"ln1_b"])
            Q = a@p[pf+"Wq"]+p[pf+"bq"]; Kk = a@p[pf+"Wk"]+p[pf+"bk"]; Vv = a@p[pf+"Wv"]+p[pf+"bv"]
            S = np.einsum("btd,bsd->bts", Q, Kk)*self.scale
            S = np.where(mask, -1e9, S)
            S -= S.max(-1, keepdims=True)
            P = np.exp(S); P /= P.sum(-1, keepdims=True)
            O = np.einsum("bts,bsd->btd", P, Vv)
            attn = O@p[pf+"Wo"]+p[pf+"bo"]
            x = x + attn
            a2, c_ln2 = self.ln_f(x, p[pf+"ln2_g"], p[pf+"ln2_b"])
            hpre = a2@p[pf+"W1"]+p[pf+"b1"]; h = np.maximum(hpre, 0)
            f = h@p[pf+"W2"]+p[pf+"b2"]
            x = x + f
            cache[l] = (a, c_ln1, Q, Kk, Vv, P, O, a2, c_ln2, hpre, h)
        xf, c_lnf = self.ln_f(x, p["lnf_g"], p["lnf_b"])
        logits = xf@p["tok_emb"].T
        cache["xf"] = xf; cache["c_lnf"] = c_lnf; cache["x_final"] = x
        return logits, cache

    def loss_and_grad(self, idx, tgt):
        B, T = idx.shape; d = self.d; p = self.p
        logits, cache = self.forward(idx)
        z = logits - logits.max(-1, keepdims=True)
        e = np.exp(z); sm = e/e.sum(-1, keepdims=True)
        ll = np.log(sm[np.arange(B)[:, None], np.arange(T)[None], tgt] + 1e-12)
        loss = -ll.mean()
        g = {k: np.zeros_like(v) for k, v in p.items()}
        dlogits = sm.copy()
        dlogits[np.arange(B)[:, None], np.arange(T)[None], tgt] -= 1
        dlogits /= (B*T)
        # weight tying: logits = xf @ tok_emb.T
        g["tok_emb"] += np.einsum("btv,btd->vd", dlogits, cache["xf"])
        dxf = np.einsum("btv,vd->btd", dlogits, p["tok_emb"])
        dx, dg, db = self.ln_b(dxf, cache["c_lnf"])
        g["lnf_g"] += dg; g["lnf_b"] += db
        mask = np.triu(np.ones((T, T)), 1).astype(bool)
        for l in reversed(range(self.L)):
            pf = f"b{l}."
            a, c_ln1, Q, Kk, Vv, P, O, a2, c_ln2, hpre, h = cache[l]
            # FFN
            df = dx
            g[pf+"b2"] += df.sum((0, 1)); g[pf+"W2"] += np.einsum("bth,btd->hd", h, df)
            dh = np.einsum("btd,hd->bth", df, p[pf+"W2"]); dh *= (hpre > 0)
            g[pf+"b1"] += dh.sum((0, 1)); g[pf+"W1"] += np.einsum("btd,bth->dh", a2, dh)
            da2 = np.einsum("bth,dh->btd", dh, p[pf+"W1"])
            dxln2, dg, db = self.ln_b(da2, c_ln2)
            g[pf+"ln2_g"] += dg; g[pf+"ln2_b"] += db
            dx = dx + dxln2                     # 残差
            # attention
            dattn = dx
            g[pf+"bo"] += dattn.sum((0, 1)); g[pf+"Wo"] += np.einsum("btd,bte->de", O, dattn)
            dO = np.einsum("btd,ed->bte", dattn, p[pf+"Wo"])
            dP = np.einsum("btd,bsd->bts", dO, Vv)
            dVv = np.einsum("bts,btd->bsd", P, dO)
            dS = P*(dP - (dP*P).sum(-1, keepdims=True))
            dS = np.where(mask, 0.0, dS)*self.scale
            dQ = np.einsum("bts,bsd->btd", dS, Kk)
            dKk = np.einsum("bts,btd->bsd", dS, Q)
            da = np.zeros_like(a)
            for W, bb, dz in (("Wq", "bq", dQ), ("Wk", "bk", dKk), ("Wv", "bv", dVv)):
                g[pf+bb] += dz.sum((0, 1)); g[pf+W] += np.einsum("btd,bte->de", a, dz)
                da += np.einsum("bte,de->btd", dz, p[pf+W])
            dxln1, dg, db = self.ln_b(da, c_ln1)
            g[pf+"ln1_g"] += dg; g[pf+"ln1_b"] += db
            dx = dx + dxln1                     # 残差
        # embeddings
        idxc = cache["idx"]
        np.add.at(g["tok_emb"], idxc, dx)
        g["pos_emb"][:T] += dx.sum(0)
        return loss, g


def train(model, data, steps, B, lr=3e-3):
    p = model.p
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    block = model.block
    rng = np.random.default_rng(1)
    t0 = time.perf_counter()
    for t in range(1, steps+1):
        ix = rng.integers(0, len(data)-block-1, size=B)
        idx = np.stack([data[i:i+block] for i in ix])
        tgt = np.stack([data[i+1:i+block+1] for i in ix])
        loss, g = model.loss_and_grad(idx, tgt)
        for k in p:
            m[k] = b1*m[k] + (1-b1)*g[k]
            v[k] = b2*v[k] + (1-b2)*g[k]**2
            mh = m[k]/(1-b1**t); vh = v[k]/(1-b2**t)
            p[k] -= lr*mh/(np.sqrt(vh)+eps)
        if t % 100 == 0 or t == 1:
            print(f"  step {t:4d}/{steps}  loss {loss:.3f}  ({time.perf_counter()-t0:.1f}s)")
    return model


def gpu_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,temperature.gpu",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL,
            timeout=3).decode().strip()
        u, c, t = [x.strip() for x in out.split(",")]
        return f"util {u:>3}%  {c}MHz  {t}C"
    except Exception:
        return ""


def main():
    chars = sorted(set(CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in CORPUS], np.int64)
    V = len(chars)
    d, n_layer, block = 64, 2, 48
    print(f"=== 文字レベル GPT を CPU 学習 ===")
    print(f"vocab={V}  d_model={d}  layers={n_layer}  block={block}  corpus={len(CORPUS)}字")
    model = NumpyGPT(V, d, n_layer, block)
    train(model, data, steps=1000, B=16)

    # --- GPU にロード ---
    ctx = gd.Device(0).create_context()
    weights = {k: v.astype(np.float32) for k, v in model.p.items()}
    gt = GpuTransformer(ctx, weights, model.cfg)

    # --- GPU vs CPU の一致確認 ---
    print("\n=== GPU 推論 vs CPU 推論 の一致確認 ===")
    seq = data[:block].tolist()
    gpu_logits = gt.forward_logits(seq)
    cpu_logits, _ = model.forward(np.array(seq)[None])
    cpu_logits = cpu_logits[0, -1]
    print(f"ロジット最大誤差 : {np.abs(gpu_logits - cpu_logits).max():.2e}")
    print(f"次トークン予測    : GPU='{itos[int(gpu_logits.argmax())]}'  "
          f"CPU='{itos[int(cpu_logits.argmax())]}'")

    # --- GPU で文章生成 ---
    print("\n=== GPU 生成 (すべて手書き PTX カーネル) ===")
    prompt = "the gpu "
    ids = [stoi[c] for c in prompt]
    out = gt.generate(ids, n_new=180, temperature=0.6, top_k=8, seed=3)
    print(repr(prompt) + " ->")
    print("  " + "".join(itos[i] for i in out))

    ctx.destroy()

    # --- 超重い設定で GPU 飽和ベンチ ---
    print("\n=== 超重い設定で GPU 飽和ベンチ (ランダム重み) ===")
    bd, bL, bblock, bV = 256, 6, 256, 96
    bcfg = dict(vocab=bV, d_model=bd, n_layer=bL, n_head=1, block=bblock)
    r = np.random.default_rng(0)
    bw = {}
    bw["tok_emb"] = r.normal(0, .02, (bV, bd)); bw["pos_emb"] = r.normal(0, .02, (bblock, bd))
    for l in range(bL):
        pf = f"b{l}."
        bw[pf+"ln1_g"] = np.ones(bd); bw[pf+"ln1_b"] = np.zeros(bd)
        for w in "qkv":
            bw[pf+f"W{w}"] = r.normal(0, .02, (bd, bd)); bw[pf+f"b{w}"] = np.zeros(bd)
        bw[pf+"Wo"] = r.normal(0, .02, (bd, bd)); bw[pf+"bo"] = np.zeros(bd)
        bw[pf+"ln2_g"] = np.ones(bd); bw[pf+"ln2_b"] = np.zeros(bd)
        bw[pf+"W1"] = r.normal(0, .02, (bd, 4*bd)); bw[pf+"b1"] = np.zeros(4*bd)
        bw[pf+"W2"] = r.normal(0, .02, (4*bd, bd)); bw[pf+"b2"] = np.zeros(bd)
    bw["lnf_g"] = np.ones(bd); bw["lnf_b"] = np.zeros(bd)
    bw = {k: v.astype(np.float32) for k, v in bw.items()}

    ctx2 = gd.Device(0).create_context()
    big = GpuTransformer(ctx2, bw, bcfg)
    T = bblock
    seq = [i % bV for i in range(T)]
    big.forward_logits(seq)   # warmup
    # FLOP 概算 (行列積 2*M*N*K を合算, 全 T 位置)
    per = 0
    per += 3*(2*T*bd*bd)          # QKV
    per += 2*T*T*bd               # QK^T
    per += 2*T*bd*T               # PV
    per += 2*T*bd*bd              # Wo
    per += 2*T*(4*bd)*bd + 2*T*bd*(4*bd)   # FFN
    flop = bL*per + 2*T*bV*bd     # + logits
    reps = 30
    t0 = time.perf_counter()
    for _ in range(reps):
        big.forward_logits(seq)
    dt = (time.perf_counter()-t0)/reps
    print(f"設定: d={bd} layers={bL} block/seq={T} vocab={bV}")
    print(f"1 forward: {dt*1000:.2f} ms   {flop/dt/1e9:.0f} GFLOP/s   {gpu_util()}")
    print(f"注意行列 {T}x{T} を {bL} 層 ぶん GPU 上で計算 (O(T^2 d) の重い部分)")
    ctx2.destroy()


if __name__ == "__main__":
    main()
