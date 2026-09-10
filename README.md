# gpudirect

**GPU を、Python から直接使うためのライブラリです。**
CUDA Toolkit も PyTorch も CuPy も要りません。パソコンに最初から入っている
NVIDIA ドライバの `nvcuda.dll`(と、内蔵GPU用に `OpenCL.dll`)を `ctypes` で
直接呼び出すだけで動きます。pip で入れる追加パッケージはゼロです。

> 動作確認: Windows 10 / Python 3.10 / NVIDIA GTX 1650 + Intel HD Graphics 530

---

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
pip install gpudirect-0.2.1-py3-none-any.whl
```

または同梱の MSI(`gpudirect-0.2.1.msi`)を実行するとローカルの Python に入ります。

一部の機能(下記 fastnumpy と AI デモ)だけ `numpy` が必要です:

```
pip install numpy
```

---

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
| `gpudirect.transformer` / `train_gpt` | 小さな GPT を GPU で「推論」＆「学習」。全部手書き PTX |

デモ: `gpu_general_demo.py`(汎用API)、`ai_demo.py`(ニューラルネット)、
`gpu_baby.py`(GPT を GPU で学習)など。

---

## 注意

- CUDA 経路は **NVIDIA 専用**です。Intel/AMD の内蔵GPUは OpenCL 経由で使います。
- `turbo` は GPU を全力で回すので、**発熱・消費電力に注意**(特にノートPC)。`Ctrl+C` で停止。
- これは学習・実験・軽量用途向けの「透明な道具」です。総合的な速度・機能は
  CuPy / PyTorch の方が上です。価値は **「GPU を最下層から全部自分の手で組んだ、
  中身が全部見える軽いライブラリ」** という点にあります。
- ドライバより下(BIOS・ハード直接)は原理的に触れません。到達点は「ドライバなしの次に直接」です。

MIT License.
