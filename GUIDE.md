# gpudirect 書き方 & 学習キット

「GPU を直接使う」を、手を動かしながら学ぶためのガイド。
上から順にやれば、**GPU 直叩き → 自作カーネル → 全ベンダ → ライブラリ連携** まで一周できます。

---

## 0. 準備

```
pip install gpudirect-0.4.0-py3-none-any.whl
pip install numpy        # fastnumpy と例で使う(任意)
```

まず GPU が見えるか:

```python
import gpudirect as gd
print(gd.devices())        # 全ベンダの GPU/iGPU 一覧
```

---

## 1. いちばん簡単: 全力で回す(仕組みを体感)

```
python -m gpudirect.turbo 5
```

GPU が 5 秒フル稼働します。`nvidia-smi -l 1` を別窓で見ると使用率が張り付くのが分かる。
「GPU に仕事を積むと動く」という感覚をまず掴む段階。

---

## 2. numpy 風に計算する(いちばん実用的)

```python
import numpy as np, gpudirect as gd
A = np.random.randn(512, 512).astype(np.float32)
B = np.random.randn(512, 512).astype(np.float32)

a, b = gd.array(A), gd.array(B)   # GPU に載る
c = a @ b + a * 2.0               # 計算は全部 GPU
print(c.numpy())                  # numpy で受け取る
```

`gd.array` / `@` / `+ - * /` / `relu` が使えます。まずはここで「GPU で計算した」を実感。

---

## 3. 自分で GPU カーネルを書く(核心)

GPU への命令(**PTX** = GPU 用アセンブリ)を自分で書いて、関数のように呼びます。
下は「c[i] = a[i] + b[i]」。

```python
import numpy as np, gpudirect as gd

PTX = r"""
.version 7.0
.target sm_75          // ← 自分の GPU の世代に合わせる(GTX16xx は sm_75)
.address_size 64
.visible .entry add(.param .u64 a,.param .u64 b,.param .u64 c,.param .u32 n){
  .reg .pred %p; .reg .b32 %r<6>; .reg .f32 %f<4>; .reg .b64 %rd<11>;
  ld.param.u64 %rd1,[a]; ld.param.u64 %rd2,[b]; ld.param.u64 %rd3,[c]; ld.param.u32 %r2,[n];
  mov.u32 %r3,%ctaid.x; mov.u32 %r4,%ntid.x; mov.u32 %r5,%tid.x;
  mad.lo.s32 %r1,%r3,%r4,%r5;             // i = blockIdx.x*blockDim.x + threadIdx.x
  setp.ge.s32 %p,%r1,%r2; @%p bra END;    // if (i>=n) return;
  cvta.to.global.u64 %rd4,%rd1; cvta.to.global.u64 %rd5,%rd2; cvta.to.global.u64 %rd6,%rd3;
  mul.wide.s32 %rd7,%r1,4;
  add.s64 %rd8,%rd4,%rd7; ld.global.f32 %f1,[%rd8];
  add.s64 %rd9,%rd5,%rd7; ld.global.f32 %f2,[%rd9];
  add.f32 %f3,%f1,%f2;
  add.s64 %rd10,%rd6,%rd7; st.global.f32 [%rd10],%f3;
END: ret; }
"""

g = gd.GPU(0)
n = 1000
a = g.to_gpu(np.arange(n, dtype=np.float32))
b = g.to_gpu(np.ones(n, np.float32))
c = g.empty(n, np.float32)

k = g.kernel(PTX, "add")                  # PTX を JIT して関数化
t = 256
k(grid=((n+t-1)//t, 1, 1), block=(t, 1, 1), args=[a, b, c, n])
print(c.get()[:5])                        # [1. 2. 3. 4. 5.]
```

### PTX の読み方(最小)
- `%tid.x / %ntid.x / %ctaid.x` = threadIdx / blockDim / blockIdx。この3つで担当要素 `i` を決める。
- `ld.global.f32 / st.global.f32` = メモリ読み書き。`mul.wide.s32 ...,4` は float(4byte)の番地計算。
- `setp` + `@%p bra` = if 文。範囲外スレッドは何もしない(はみ出し対策)。
- 1 スレッド = 1 要素、を大量のスレッドで並列に処理する、が基本形。

---

## 4. AMD / Intel / iGPU で動かす(全ベンダ)

NVIDIA 以外は OpenCL で、カーネルは **OpenCL C**(C 言語風)で書きます。書き味はほぼ同じ。

```python
import numpy as np, gpudirect as gd

gi = gd.GPU(1, backend="opencl")          # 番号は gd.devices() で確認
src = "__kernel void add(__global const float*a,__global const float*b,__global float*c,const int n){int i=get_global_id(0); if(i<n) c[i]=a[i]+b[i];}"
n = 1000
a, b = gi.to_gpu(np.arange(n, np.float32)), gi.to_gpu(np.ones(n, np.float32))
c = gi.empty(n, np.float32)
k = gi.kernel(src, "add")
k(n, args=[a, b, c, n])                    # OpenCL は global_size を渡す
print(c.get()[:5])
```

PTX(NVIDIA)と OpenCL C(その他)で言語は違うが、`to_gpu / kernel / get` の使い方は共通。

---

## 5. 他のライブラリとつなぐ

```python
import torch, gpudirect as gd
g = gd.GPU(0)
x = g.to_gpu(torch.arange(6))   # torch / cupy / pandas / tf / jax / numpy / list を受け取る
x.torch(); x.cupy(); x.pandas(); x.numpy(); x.to("jax")   # どの形でも返す
```

CUDA 配列は `__cuda_array_interface__` を公開するので、CuPy / Numba とはゼロコピーで繋がります。

---

## 6. ソースを読んで学ぶ(この順がおすすめ)

| 順 | ファイル | 学べること |
|---|---|---|
| 1 | `gpudirect/_driver.py` | Windows API で DLL を掴み、CUDA 関数を ctypes で呼ぶ最下層 |
| 2 | `gpudirect/__init__.py` | Device / Context / メモリ / PTX ロード / カーネル起動の土台 |
| 3 | `gpudirect/kernels/*.ptx` | 手書き PTX の実例(vecadd → matmul → matmul_reg の順) |
| 4 | `gpudirect/easy.py` | 汎用 API と、CUDA/OpenCL を同じ書き方にする統一層 |
| 5 | `gpudirect/opencl.py` | OpenCL.dll 直叩き(列挙・計算・飽和) |
| 6 | `gpudirect/fastnumpy.py` | 手書きカーネルを numpy 風 API に組み上げる例 |
| 7 | `gpudirect/interop.py` | 主要ライブラリと相互変換する仕組み |

### 練習問題
1. PTX の `add` を `mul`(掛け算)に変えてみる(`add.f32` → `mul.f32`)。
2. `c[i] = a[i]*2 + 1` を作る(`fma.rn.f32`)。
3. OpenCL 版で同じことをやって、NVIDIA と iGPU 両方で走らせる。
4. `fastnumpy` に自分の演算(例: `sub` はある。`exp` を足す)を追加する。

---

## つまずきポイント
- **`.target sm_XX` は GPU 世代に合わせる**(GTX 16xx = sm_75、RTX 30xx = sm_86 など)。合わないと JIT で失敗。
- PTX は行頭 `//` コメントで JIT が失敗することがある(gpudirect 側で自動除去済み)。
- `turbo` は全力稼働なので**発熱・電力に注意**。`Ctrl+C` で停止。
- 速度は大きな計算ほど GPU が有利。小さい配列や要素演算は転送で相殺されがち。
