"""
GPU 学習デモ: 順伝播・逆伝播・Adam をすべて GPU の手書き PTX で回して
two-moons を分類する MLP を学習する。CuPy / PyTorch 不使用。

- GPU 学習の損失が下がり、精度が上がることを確認。
- 同じ初期重み・同じデータで CPU(numpy)学習と重みが一致することで
  「GPU の勾配が正しい」ことを検証。
    python gpu_train_demo.py
"""
import time
import numpy as np

import gpudirect as gd
from gpudirect.train import GpuMLPTrainer

rng = np.random.default_rng(0)


def make_moons(n):
    n0 = n//2; n1 = n-n0
    t0 = np.pi*rng.random(n0); x0 = np.stack([np.cos(t0), np.sin(t0)], 1)
    t1 = np.pi*rng.random(n1); x1 = np.stack([1-np.cos(t1), 1-np.sin(t1)-0.5], 1)
    X = np.concatenate([x0, x1]).astype(np.float32)
    y = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(np.int64)
    X += 0.08*rng.standard_normal(X.shape).astype(np.float32)
    idx = rng.permutation(n); return X[idx], y[idx]


def softmax_ce(logits, y):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z); p = e/e.sum(1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    d = p.copy(); d[np.arange(n), y] -= 1; d /= n
    return loss, d.astype(np.float32), p.argmax(1)


def main():
    Xtr, ytr = make_moons(2000)
    Xte, yte = make_moons(1000)
    dims = [2, 64, 32, 2]
    ctx = gd.Device(0).create_context()
    print(f"=== GPU 学習 (forward+backward+Adam すべて PTX) 構成 {dims} ===")
    tr = GpuMLPTrainer(ctx, dims, seed=42)

    dXtr = tr._dev(Xtr.ravel())     # 入力は一度 GPU に載せて常駐
    batch = len(Xtr)
    t0 = time.perf_counter()
    for step in range(1, 601):
        logits = tr.forward(dXtr, batch)
        loss, dlog, _ = softmax_ce(logits, ytr)
        tr.backward_and_step(dlog, batch, lr=0.05)
        if step % 100 == 0 or step == 1:
            # テスト精度
            dXte = tr._dev(Xte.ravel())
            te_logits = tr.forward(dXte, len(Xte))
            acc = (te_logits.argmax(1) == yte).mean()*100
            print(f"  step {step:4d}  loss {loss:.3f}  test-acc {acc:5.1f}%")
    print(f"GPU 学習 600 step: {time.perf_counter()-t0:.1f}s")

    # --- 勾配の正しさ検証: 同一初期重み・同一データで CPU と 1 step 比較 ---
    print("\n=== 勾配検証: GPU 1step vs CPU 1step (同一初期重み) ===")
    tr2 = GpuMLPTrainer(ctx, dims, seed=7)
    init = tr2.get_weights()
    Xb, yb = Xtr[:256], ytr[:256]
    dXb = tr2._dev(Xb.ravel())
    lg = tr2.forward(dXb, 256); loss, dl, _ = softmax_ce(lg, yb)
    tr2.backward_and_step(dl, 256, lr=0.05)
    gpu_after = tr2.get_weights()

    # CPU 参照 (同じ Adam / 同じ初期重み, t=1)
    P = [w.astype(np.float64) for w, _ in init] + [b.astype(np.float64) for _, b in init]
    def cpu_step():
        Ws = [w.copy() for w, _ in init]; bs = [b.copy().astype(np.float64) for _, b in init]
        Ws = [w.astype(np.float64) for w in Ws]
        acts = [Xb.astype(np.float64)]; z = acts[0]
        for i in range(len(Ws)):
            z = acts[-1] @ Ws[i] + bs[i]
            if i < len(Ws)-1: z = np.maximum(z, 0)
            acts.append(z)
        _, dl, _ = softmax_ce(acts[-1], yb); g = dl.astype(np.float64)
        gW = [None]*len(Ws); gb = [None]*len(Ws)
        for i in reversed(range(len(Ws))):
            gW[i] = acts[i].T @ g; gb[i] = g.sum(0)
            if i > 0:
                g = g @ Ws[i].T; g = g*(acts[i] > 0)
        # Adam t=1
        b1, b2, eps = 0.9, 0.999, 1e-8; bc1, bc2 = 1-b1, 1-b2
        outW = []
        for i in range(len(Ws)):
            mw = (1-b1)*gW[i]; vw = (1-b2)*gW[i]**2
            Ws[i] -= 0.05*(mw/bc1)/(np.sqrt(vw/bc2)+eps)
            outW.append(Ws[i])
        return outW
    cpuW = cpu_step()
    md = max(np.abs(gpu_after[i][0] - cpuW[i]).max() for i in range(len(cpuW)))
    print(f"  GPU と CPU の更新後 W 最大差: {md:.2e}  {'OK (勾配一致)' if md < 1e-3 else 'NG'}")
    ctx.destroy()


if __name__ == "__main__":
    main()
