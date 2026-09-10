"""
GPU で Transformer(赤ちゃん)を学習する。順伝播・逆伝播・Adam すべて手書き PTX。
- CPU(numpy)との 1 ステップ速度を比較。
- GPU で大量学習して loss(perplexity)を下げ、生成サンプルで賢さを見る。
    python gpu_baby.py
"""
import time
import numpy as np

import gpudirect as gd
from gpudirect.train_gpt import GpuGPTTrainer
from gpudirect.transformer import GpuTransformer
from llm_demo import NumpyGPT

CORPUS = (
    "the gpu runs the kernel and the kernel runs the gpu. "
    "we write the ptx by hand and the driver builds the code. "
    "the tensor flows through the layers and the answer comes out. "
    "attention looks back at every token and picks what matters. "
    "hello, i am a tiny gpt running on the gpu. "
    "you write a kernel and i run it fast. "
) * 40


def main():
    chars = sorted(set(CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in CORPUS], np.int64)
    V, d, L, block = len(chars), 96, 3, 64
    cfg = dict(vocab=V, d_model=d, n_layer=L, n_head=1, block=block)
    print(f"vocab={V} d={d} layers={L} block={block}")

    base = NumpyGPT(V, d, L, block, seed=0)
    w0 = {k: v.astype(np.float32) for k, v in base.p.items()}
    ctx = gd.Device(0).create_context()

    rng = np.random.default_rng(0)
    def batch():
        i = int(rng.integers(0, len(data)-block-1))
        return list(data[i:i+block]), list(data[i+1:i+block+1])

    # --- 速度比較 (1 系列/step) ---
    print("\n=== 1 ステップ速度: CPU vs GPU ===")
    seq, tgt = batch()
    # CPU
    cpu = NumpyGPT(V, d, L, block, seed=0)
    cpu.loss_and_grad(np.array(seq)[None], np.array(tgt)[None])  # warm
    t0 = time.perf_counter(); R = 10
    for _ in range(R):
        loss, g = cpu.loss_and_grad(np.array(seq)[None], np.array(tgt)[None])
        for k in cpu.p: cpu.p[k] -= 1e-9*g[k]
    tcpu = (time.perf_counter()-t0)/R
    # GPU
    tr = GpuGPTTrainer(ctx, cfg, w0, lr=1.5e-3, seed=0)
    tr.step(seq, tgt)  # warm
    t0 = time.perf_counter()
    for _ in range(R): tr.step(*batch())
    tgpu = (time.perf_counter()-t0)/R
    print(f"CPU {tcpu*1000:7.1f} ms/step   GPU {tgpu*1000:6.1f} ms/step   -> {tcpu/tgpu:.0f}x 高速")

    # --- GPU 大量学習 ---
    print("\n=== GPU で育成 ===")
    def sample(seed_text="the ", n=140):
        gt = GpuTransformer(ctx, tr.export(), cfg)
        ids = [stoi[c] for c in seed_text if c in stoi] or [0]
        out = gt.generate(ids, n_new=n, temperature=0.5, top_k=6, seed=1)
        return "".join(itos[i] for i in out).replace("\n", " ")
    steps = 1200
    t0 = time.perf_counter(); last = 0
    for s in range(1, steps+1):
        last = tr.step(*batch())
        if s % 200 == 0 or s == 1:
            ppl = float(np.exp(last))
            print(f"  step {s:4d}  loss {last:.3f}  ppl {ppl:6.1f}  ({time.perf_counter()-t0:.1f}s)")
            print(f"     sample: {sample()}")
    print(f"\nGPU {steps} step 完了: {time.perf_counter()-t0:.1f}s")
    print(f"最終 sample('hello'): {sample('hello', 160)}")
    ctx.destroy()


if __name__ == "__main__":
    main()
