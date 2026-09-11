# gpudirect

**GPU を、Python から直接使うためのライブラリです。**
CUDA Toolkit も PyTorch も CuPy も要りません。パソコンに最初から入っている
NVIDIA ドライバの `nvcuda.dll`(と、内蔵GPU用に `OpenCL.dll`)を `ctypes` で
直接呼び出すだけで動きます。pip で入れる追加パッケージはゼロです。

> 動作確認: Windows 10 / Python 3.10 / NVIDIA GTX 1650 + Intel HD Graphics 530

---

## import 一つで全部（v0.2.2〜）

```python
import gpudirect as gd
gd.devices()                       # 全ベンダの GPU/iGPU
g = gd.GPU(0)                      # 汎用API(easy)
g = gd.GPU(0, backend="opencl")   # AMD/Intel/iGPU
gd.saturate()                      # 一行で全GPU飽和(turbo)
c = gd.array(A) @ gd.array(B)      # numpy風(fastnumpy)
```

`easy` / `turbo` / `fastnumpy` / `transformer` / `train_gpt` などは
すべて `gpudirect` の中に統合済み。`import gpudirect` だけで到達できます
(numpy が要る機能に初めて触れた時だけ numpy を読み込む軽量設計)。

## これは何？

普通、GPU を使うには CUDA Toolkit(数GB)や PyTorch のような大きなものを入れて、
何層もの「翻訳係」を経由します:

```
あなたのコード → PyTorch → CUDAライブラリ → ドライバ → GPU
```

gpudirect は、この中間を全部とばして **ドライバの入口を直接つかみます**:

```
あなたのコード → (ctypes) → ドライバDLL → GPU
```

だから **軽い**(ドライバだけ)、**GPU に一番近い**、そして
GPU への命令(PTX という GPU 用アセンブリ)を **自分で書いて直接流し込めます**。

---

## インストール

```
pip install gpudirect-0.5.0-py3-none-any.whl
```

または同梱の MSI(`gpudirect-0.5.0.msi`)を実行するとローカルの Python に入ります。

一部の機能(下記 fastnumpy と AI デモ)だけ `numpy` が必要です:

```
pip install numpy
```

---

## ゼロコピー配列（v0.5.0〜）

**CPU と GPU が同じメモリを直接共有する**配列です(CUDA Unified Memory)。
普通は「CPU→GPU」「GPU→CPU」の転送(`cuMemcpyHtoD`/`DtoH`)が都度発生しますが、
ゼロコピー配列は 1 回確保すれば、あとは numpy のように直接読み書きするだけで
GPU 側からもそのまま見えます。転送コマンドを挟まない分、**繰り返し使うと実測で
2〜4倍速い**(このマシンでの計測、下記参照)。

```python
import numpy as np, gpudirect as gd
ctx = gd.Device(0).create_context()

z = ctx.zeros_shared(1_000_000, "f4")   # 1回確保、CPU/GPU で共有
z.np[:] = np.random.randn(1_000_000)    # numpy として直接書き込み(転送コマンド無し)
fn.launch(grid=..., block=..., args=[z.ptr, 1_000_000])  # GPU が直接読み書き
print(z.np[:5])                          # そのまま読める(転送コマンド無し)
```

実測(このマシン、400万要素、確保は1回のみ・使い回し):

| 方式 | 時間/回 |
|---|---|
| 通常(`cuMemcpyHtoD`＋`cuMemcpyDtoH`を毎回) | 28.7 ms |
| **ゼロコピー(直接読み書き)** | **7.1 ms(4.05倍)** |

カーネル実行を挟む実運用パターン(200万要素、書く→GPU計算→読む)でも **2.15倍**。

> 注意: GPU 側の処理が終わる前に CPU から読み書きすると競合するため、
> カーネル起動後は同期(既定の `sync=True`)を挟むこと。

## 主要ライブラリと連携（v0.4.0〜）

**PyTorch / CuPy / pandas / TensorFlow / JAX / numpy とそのままつながります。**
どのライブラリの配列も受け取り、どのライブラリの形でも返せます。

```python
import gpudirect as gd, torch, numpy as np
g = gd.GPU(0)

x = g.to_gpu(torch.arange(6))     # torch / cupy / tf / jax / pandas / numpy / list 何でも入力
x.torch()                          # → torch.Tensor
x.cupy()                           # → cupy.ndarray
x.pandas()                         # → pandas.Series / DataFrame
x.numpy()                          # → numpy
x.to("jax")                        # → 任意ライブラリを名前指定

# CUDA 配列は __cuda_array_interface__ を公開 → CuPy/Numba とゼロコピー連携
```

## 対応する GPU（v0.2.0〜）

**NVIDIA だけでなく、AMD・Intel の GPU と内蔵GPU(iGPU)でも計算できます。**
NVIDIA は CUDA(手書き PTX)、それ以外は OpenCL(手書き OpenCL C)で動きます。
どちらも同じ書き方で扱えます。

```python
import gpudirect.easy as ge
ge.devices()                       # 全ベンダの GPU/iGPU を一覧
g = ge.GPU(0)                      # 既定 = NVIDIA(CUDA)
g = ge.GPU(0, backend="opencl")    # AMD / Intel / iGPU(OpenCL)
```

## どのライブラリの配列とも噛み合う

入力は numpy 配列・list・bytes・バッファ・`__array__` を持つ物(torch の CPU テンソル等)・
`__dlpack__` を持つ物、なんでも受け取れます。出力(`GpuArray`/`CLArray`)は
`np.asarray(x)` でそのまま numpy として取り出せる(array-protocol 対応)ので、
他のライブラリとつなげられます。

```python
g = ge.GPU(0)
x = g.to_gpu([1, 2, 3])            # list でも
y = g.to_gpu(np_array)             # numpy でも
import numpy as np; np.asarray(x)  # numpy として取り出す
```

## 使い方は3段階

### 段階1: とにかく GPU を全力で回す（一行）

```
python -m gpudirect.turbo
```

見つかった **すべての GPU と内蔵GPU(iGPU)を同時に最大稼働**させます。
ベンチマーク・負荷試験・発熱チェック用。`Ctrl+C` で停止します。

```python
import gpudirect.turbo as turbo
turbo.saturate()      # Python から呼ぶ場合
```

### 段階2: numpy っぽく計算する（fastnumpy）

`fastnumpy` は **「numpy と同じ書き方で、計算を GPU にやらせる」** ための
おまけモジュールです。`+ - * / @`(行列積)や relu が GPU 上で走ります。
numpy を知っていれば、そのままの感覚で書けます。

```python
import gpudirect.fastnumpy as fnp   # gpudirect に統合済み

a = fnp.array([[1, 2], [3, 4]])   # GPU に配列を置く
b = fnp.ones((2, 2))
c = a @ b + a * 2.0               # 計算は全部 GPU 上で
print(c.numpy())                  # numpy に戻して受け取る
```

> ※ numpy 自体をとても速くしたもの、ではありません。
> 「numpy と同じ書き味で GPU を使える入口」だと思ってください。

### 段階3: 自分で GPU カーネルを書く（いちばん低レベル）

GPU への命令(PTX)を自分で書いて、関数のように呼べます。中身が全部見える一番下の層です。

```python
import numpy as np
import gpudirect.easy as ge

g = ge.GPU()                                # GPU をつかむ
a = g.to_gpu(np.arange(8, dtype=np.float32))  # numpy → GPU
out = g.empty(8, np.float32)
k = g.kernel(PTX_SOURCE, "my_kernel")       # 手書き PTX を関数化
k(grid=(1, 1, 1), block=(8, 1, 1), args=[a, out, 8])
print(out.get())                            # GPU → numpy
```

---

## 中に入っているもの

| モジュール | 何をするか |
|---|---|
| `gpudirect` | `nvcuda.dll`(CUDA)を直叩き。デバイス・メモリ・PTX/cubin・カーネル起動の土台 |
| `gpudirect.easy` | 汎用の入口。`GPU` / `GpuArray` / `Kernel`(段階3で使うやつ) |
| `gpudirect.turbo` | 全 GPU/iGPU を一行で最大飽和(段階1) |
| `gpudirect.opencl` | `OpenCL.dll` を直叩き。内蔵GPU含む全ベンダを列挙・稼働 |
| `fastnumpy` | numpy 風に GPU で計算(段階2) |

デモ: `gpu_general_demo.py`(汎用API / 自作カーネル)。

---

## 注意

- CUDA 経路は **NVIDIA 専用**です。Intel/AMD の内蔵GPUは OpenCL 経由で使います。
- `turbo` は GPU を全力で回すので、**発熱・消費電力に注意**(特にノートPC)。`Ctrl+C` で停止。
- これは学習・実験・軽量用途向けの「透明な道具」です。総合的な速度・機能は
  CuPy / PyTorch の方が上です。価値は **「GPU を最下層から全部自分の手で組んだ、
  中身が全部見える軽いライブラリ」** という点にあります。
- ドライバより下(BIOS・ハード直接)は原理的に触れません。到達点は「ドライバなしの次に直接」です。

MIT License.

---

## 学習キット

手を動かして学ぶ手順書 → **[GUIDE.md](GUIDE.md)**(GPU直叩き→自作カーネル→全ベンダ→ライブラリ連携)
