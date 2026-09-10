"""
AI を gpudirect で動かすデモ。

- 2 クラスの "two-moons" データを numpy で生成し、MLP を CPU で学習する。
- 学習した重みを GPU に載せ、推論(順伝播)を手書き PTX カーネルだけで実行。
- GPU の予測が CPU(numpy)と一致すること、テスト精度、決定境界を表示する。

CuPy / PyTorch は不使用。行列積も活性化も自作 PTX。
    python ai_demo.py
"""
import numpy as np

import time

import gpudirect as gd
from gpudirect.nn import GpuMLP, FusedMLP

rng = np.random.default_rng(0)


def make_moons(n):
    """two-moons: 半月 2 つを上下に配置した 2 クラス分類データ。"""
    n0 = n // 2
    n1 = n - n0
    t0 = np.pi * rng.random(n0)
    x0 = np.stack([np.cos(t0), np.sin(t0)], 1)
    t1 = np.pi * rng.random(n1)
    x1 = np.stack([1 - np.cos(t1), 1 - np.sin(t1) - 0.5], 1)
    X = np.concatenate([x0, x1]).astype(np.float32)
    y = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(np.int64)
    X += 0.08 * rng.standard_normal(X.shape).astype(np.float32)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def train_cpu(X, y, hidden=(32, 16), epochs=400, lr=0.1):
    """numpy だけで MLP を学習し、[(W,b), ...] を返す。ReLU + softmax CE。"""
    dims = [X.shape[1]] + list(hidden) + [2]
    Ws = [rng.standard_normal((dims[i], dims[i + 1])).astype(np.float32)
          * np.sqrt(2.0 / dims[i]) for i in range(len(dims) - 1)]
    bs = [np.zeros(dims[i + 1], np.float32) for i in range(len(dims) - 1)]
    Y = np.eye(2, dtype=np.float32)[y]
    n = len(X)
    for ep in range(epochs):
        # forward
        acts = [X]
        z = X
        for i, (W, b) in enumerate(zip(Ws, bs)):
            z = acts[-1] @ W + b
            if i < len(Ws) - 1:
                z = np.maximum(z, 0)
            acts.append(z)
        logits = acts[-1]
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        # backward
        g = (p - Y) / n
        for i in reversed(range(len(Ws))):
            a = acts[i]
            gW = a.T @ g
            gb = g.sum(0)
            if i > 0:
                g = g @ Ws[i].T
                g *= (acts[i] > 0)
            Ws[i] -= lr * gW
            bs[i] -= lr * gb
    return list(zip(Ws, bs))


def cpu_forward(weights, X):
    z = X
    for i, (W, b) in enumerate(weights):
        z = z @ W + b
        if i < len(weights) - 1:
            z = np.maximum(z, 0)
    return z


def main():
    print("=== データ生成 & CPU 学習 ===")
    Xtr, ytr = make_moons(2000)
    Xte, yte = make_moons(1000)
    weights = train_cpu(Xtr, ytr)
    cpu_logits = cpu_forward(weights, Xte)
    cpu_pred = cpu_logits.argmax(1)
    print(f"CPU テスト精度: {(cpu_pred == yte).mean()*100:.1f}%")
    arch = " -> ".join(str(W.shape[0]) for W, _ in weights) + " -> 2"
    print(f"ネット構成: {arch}  (全結合 + ReLU)")

    print("\n=== GPU 推論 (手書き PTX: matmul + bias_relu) ===")
    ctx = gd.Device(0).create_context()
    net = GpuMLP(ctx)
    net.load_layers(weights)

    batch = len(Xte)
    gpu_logits, gpu_pred = net.forward(Xte.ravel().tolist(), batch)
    gpu_pred = np.array(gpu_pred)
    gpu_logits = np.array(gpu_logits).reshape(batch, 2)

    acc = (gpu_pred == yte).mean() * 100
    agree = (gpu_pred == cpu_pred).mean() * 100
    max_diff = float(np.abs(gpu_logits - cpu_logits).max())
    print(f"GPU テスト精度   : {acc:.1f}%")
    print(f"CPU との予測一致 : {agree:.1f}%")
    print(f"ロジット最大誤差 : {max_diff:.2e}  (float32 の丸め程度)")
    n_layers = len(weights)
    print(f"起動カーネル数   : {n_layers*2} 回 (層ごとに matmul+bias_relu)")

    print("\n=== 融合カーネル (全層+ReLU+argmax を 1 起動で一気に) ===")
    fused = FusedMLP(ctx)
    fused.load_layers(weights)
    fx = Xte.ravel().tolist()
    fused.forward(fx, batch)                       # warmup
    f_logits, f_pred = fused.forward(fx, batch)
    f_pred = np.array(f_pred)
    f_logits = np.array(f_logits).reshape(batch, 2)
    print(f"融合 GPU 精度    : {(f_pred == yte).mean()*100:.1f}%")
    print(f"多起動版との一致 : {(f_pred == gpu_pred).mean()*100:.1f}%")
    print(f"CPU とのロジット差: {float(np.abs(f_logits - cpu_logits).max()):.2e}")
    print(f"起動カーネル数   : 1 回  ({n_layers*2} -> 1 に融合)")

    # 速度比較 (カーネル部のみ、複数回の最小値)
    def bench(fn_forward, reps=50):
        best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter()
            fn_forward()
            best = min(best, (time.perf_counter() - t0) * 1000)
        return best
    t_multi = bench(lambda: net.forward(fx, batch))
    t_fused = bench(lambda: fused.forward(fx, batch))
    print(f"\n1 推論あたり  多起動: {t_multi:.3f} ms   融合: {t_fused:.3f} ms"
          f"   ({t_multi/t_fused:.2f}x 高速)")

    # 決定境界を ASCII で描く (全部 GPU 推論)
    print("\n=== GPU が引いた決定境界 ===")
    W_, H_ = 56, 22
    xs = np.linspace(-1.6, 2.6, W_)
    ys = np.linspace(-1.4, 1.6, H_)
    grid = np.array([[x, y] for y in ys for x in xs], np.float32)
    _, gp = net.forward(grid.ravel().tolist(), len(grid))
    gp = np.array(gp).reshape(H_, W_)
    for row in gp[::-1]:
        print("  " + "".join("#" if v == 1 else "." for v in row))
    print("  ( '#' = クラス1 / '.' = クラス0  ・すべて GPU の PTX カーネルで推論 )")

    ctx.destroy()


if __name__ == "__main__":
    main()
