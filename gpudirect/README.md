# gpudirect

NVIDIA GPU を **pip 追加ゼロ**で直接使う自作ライブラリ。
NVIDIA ドライバ同梱の `nvcuda.dll`(CUDA Driver API)を、
**Windows API(`LoadLibraryW` / `GetProcAddress`)で自分でロード・シンボル解決**して叩く。

- CUDA Toolkit 不要 / CuPy 不要 / PyTorch 不要 / cffi 不要
- カーネルは **手書き PTX(アセンブリ)** か **cubin(SASS=機械語)** で投入
- 検証済み環境: GTX 1650 (sm_75) / ドライバ CUDA 13.2 / Python 3.10

## どこまで「直接」か
`nvcuda.dll` はドライバのユーザーモード入口で、Python から到達できる最下層。
これより下(MMIO レジスタ / GPU ページテーブル)は Ring0(カーネル)専用で、
ユーザープロセスからは触れない。ドライバ抜きで NVIDIA GPU を動かすことは
仕様非公開のため不可能。よって本ライブラリが「ドライバなしの次」に最も直接的。

ネイティブ関数ポインタの**呼び出し**は CPython では `ctypes` が唯一の手段
（ctypes = Python の FFI 本体）。DLL のロードとシンボル解決だけを
Windows API 側で明示的に握っている。ctypes の呼出オーバーヘッドは 1 回数µs で、
GPU の計算速度には影響しない(下のベンチのカーネル実測 0.78ms が本体)。

## 使い方
```python
import gpudirect as gd

ctx = gd.Device(0).create_context()

# アセンブリレベル: 手書き PTX をドライバ JIT で SASS 化
mod = ctx.load_ptx(open("gpudirect/kernels/vecadd.ptx", "rb").read())
fn  = mod.function("vecadd")

n = 3
a = ctx.to_device([1.0, 2.0, 3.0], "f4")
b = ctx.to_device([10.0, 20.0, 30.0], "f4")
c = ctx.malloc(n * 4); c.dtype = "f4"

fn.launch(grid=(1,1,1), block=(n,1,1), args=[a, b, c, n])
print(list(c.copy_to_host("f4", n)))   # [11.0, 22.0, 33.0]
```

機械語レベル(JIT なし)で cubin を直接ロード:
```python
mod = ctx.load_cubin("mykernel.cubin")   # 中身は sm_75 の SASS
```
> cubin は「コンパイル済み GPU 機械語」。生成には ptxas / nvcc が要る
> (この環境には未導入)。ドライバは JIT せずそのまま実行する。

## ベンチ (8,000,000 要素の float 加算)
| | 時間 |
|---|---|
| CPU (純 Python) | ~1365 ms |
| GPU 転送込み | ~108 ms |
| GPU カーネルのみ | **~0.78 ms** (対 CPU 約 1700×) |

## 構成
- `gpudirect/_driver.py` — Windows API 直呼びでの nvcuda.dll バインディング
- `gpudirect/__init__.py` — Device / Context / Module / Function / DeviceMemory
- `gpudirect/kernels/vecadd.ptx` — 手書き PTX の例
- `example.py` — 実行例 + ベンチ (`python example.py`)
