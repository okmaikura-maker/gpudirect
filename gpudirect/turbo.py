"""
gpudirect.turbo — 世界一かんたんな GPU/iGPU 直接最大飽和。一行で呼べる。

    import gpudirect.turbo as turbo
    turbo.saturate()        # 見つかった全 GPU/iGPU を同時に最大飽和(既定 10 秒)

シェルからも一行:
    python -m gpudirect.turbo         # 全 GPU/iGPU を 10 秒フル稼働
    python -m gpudirect.turbo 30      # 30 秒

NVIDIA も Intel/AMD の iGPU も、ドライバ同梱の OpenCL.dll 経由で直接叩く。
CUDA Toolkit / pyopencl / CuPy / PyTorch 一切不要。
"""
import sys
import threading

from . import opencl as _ocl


def devices():
    """見つかった GPU/iGPU の一覧を返す。"""
    return [d for d in _ocl.list_devices() if d["kind"] == "GPU"]


def saturate(seconds=10.0, which="all", verbose=True):
    """
    見つかった全 GPU/iGPU を seconds 秒あいだ同時に最大飽和させる。
    which="all"(既定) / "nvidia" / "intel" / "amd" / デバイス番号(int) で選択可。
    戻り値: 各デバイスの結果 dict のリスト。
    """
    devs = devices()
    if isinstance(which, int):
        devs = [devs[which]]
    elif which != "all":
        devs = [d for d in devs if which.lower() in (d["platform"]+d["name"]).lower()]
    if not devs:
        print("GPU/iGPU が見つかりませんでした。"); return []
    if verbose:
        print(f"最大飽和 {seconds:.0f}s -> " + ", ".join(f"{d['name']}" for d in devs))
    results = [None]*len(devs)

    def work(i, d):
        results[i] = _ocl.saturate(d["handle"], seconds=seconds)
        results[i]["name"] = d["name"]; results[i]["kind"] = d["kind"]

    ths = [threading.Thread(target=work, args=(i, d)) for i, d in enumerate(devs)]
    for t in ths: t.start()
    for t in ths: t.join()
    if verbose:
        for r in results:
            print(f"  {r['name']:32s} {r['gflops']:7.0f} GFLOP/s  ({r['launches']} launches)")
    return results


def _main():
    sec = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    print("=== gpudirect.turbo : 全 GPU/iGPU 直接最大飽和 ===")
    for i, d in enumerate(devices()):
        print(f"  [{i}] {d['name']} [{d['kind']}] via {d['platform']}  CU={d['cu']} {d['mhz']}MHz")
    print()
    saturate(sec)


if __name__ == "__main__":
    _main()
