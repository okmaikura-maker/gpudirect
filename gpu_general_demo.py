"""
汎用 GPU 直叩き API (gpudirect.easy) のデモ。
numpy 配列を GPU に載せ、自作 PTX カーネルを関数のように呼び、結果を numpy で回収。
どんな計算でも「PTX を書いて渡すだけ」で GPU で走る。
    python gpu_general_demo.py
"""
import numpy as np
import gpudirect.easy as ge

# --- 用途1: SAXPY  c = alpha*a + b  (float32) ---
SAXPY = """
.version 7.0
.target sm_75
.address_size 64
.visible .entry saxpy(.param .u64 a,.param .u64 b,.param .u64 c,.param .f32 alpha,.param .u32 n){
  .reg .pred %p; .reg .b32 %r<6>; .reg .f32 %f<5>; .reg .b64 %rd<11>;
  ld.param.u64 %rd1,[a]; ld.param.u64 %rd2,[b]; ld.param.u64 %rd3,[c];
  ld.param.f32 %f1,[alpha]; ld.param.u32 %r2,[n];
  mov.u32 %r3,%ctaid.x; mov.u32 %r4,%ntid.x; mov.u32 %r5,%tid.x; mad.lo.s32 %r1,%r3,%r4,%r5;
  setp.ge.s32 %p,%r1,%r2; @%p bra END;
  cvta.to.global.u64 %rd4,%rd1; cvta.to.global.u64 %rd5,%rd2; cvta.to.global.u64 %rd6,%rd3;
  mul.wide.s32 %rd7,%r1,4;
  add.s64 %rd8,%rd4,%rd7; add.s64 %rd9,%rd5,%rd7; add.s64 %rd10,%rd6,%rd7;
  ld.global.f32 %f2,[%rd8]; ld.global.f32 %f3,[%rd9];
  fma.rn.f32 %f4,%f1,%f2,%f3; st.global.f32 [%rd10],%f4;
END: ret; }
"""

# --- 用途2: 整数の二乗  out[i] = x[i]*x[i]  (int32) ---
SQUARE = """
.version 7.0
.target sm_75
.address_size 64
.visible .entry square(.param .u64 x,.param .u64 out,.param .u32 n){
  .reg .pred %p; .reg .b32 %r<8>; .reg .b64 %rd<8>;
  ld.param.u64 %rd1,[x]; ld.param.u64 %rd2,[out]; ld.param.u32 %r2,[n];
  mov.u32 %r3,%ctaid.x; mov.u32 %r4,%ntid.x; mov.u32 %r5,%tid.x; mad.lo.s32 %r1,%r3,%r4,%r5;
  setp.ge.s32 %p,%r1,%r2; @%p bra END;
  cvta.to.global.u64 %rd3,%rd1; cvta.to.global.u64 %rd4,%rd2;
  mul.wide.s32 %rd5,%r1,4; add.s64 %rd6,%rd3,%rd5; add.s64 %rd7,%rd4,%rd5;
  ld.global.u32 %r6,[%rd6]; mul.lo.s32 %r7,%r6,%r6; st.global.u32 [%rd7],%r7;
END: ret; }
"""


def launch1d(kernel, n, args):
    threads = 256
    kernel(grid=((n+threads-1)//threads, 1, 1), block=(threads, 1, 1), args=args)


def main():
    g = ge.GPU()
    print("device:", g.name)
    for k, v in g.info().items():
        print(f"  {k}: {v}")

    # 用途1: SAXPY (float32)
    n = 1_000_000
    a = g.to_gpu(np.arange(n, dtype=np.float32))
    b = g.to_gpu(np.ones(n, dtype=np.float32) * 5)
    c = g.empty(n, np.float32)
    saxpy = g.kernel(SAXPY, "saxpy")
    launch1d(saxpy, n, [a, b, c, np.float32(3.0), n])
    out = c.get()
    ref = 3.0 * np.arange(n) + 5
    print(f"\n[SAXPY]  c=3a+b   max誤差 {np.abs(out-ref).max():.2e}   例 {out[:5]}")

    # 用途2: int の二乗
    x = g.to_gpu(np.arange(10, dtype=np.int32))
    o = g.empty(10, np.int32)
    sq = g.kernel(SQUARE, "square")
    launch1d(sq, 10, [x, o, 10])
    print(f"[SQUARE] x^2 (int) -> {o.get().tolist()}")

    # 既存の自作カーネルファイルもそのまま使える(例: 行列積)
    mm = g.module(open("gpudirect/kernels/matmul_reg.ptx", "rb").read()).kernel("matmul_reg")
    M = K = N = 512
    A = g.to_gpu(np.random.randn(M, K).astype(np.float32))
    B = g.to_gpu(np.random.randn(K, N).astype(np.float32))
    C = g.empty((M, N), np.float32)
    mm(grid=((N+63)//64, (M+63)//64, 1), block=(16, 16, 1), args=[A, B, C, M, N, K])
    Cg = C.get()
    print(f"[MATMUL] {M}x{K}x{N}  max誤差 {np.abs(Cg - A.get()@B.get()).max():.2e}")

    print("\nどの用途も『PTX を書いて渡す→numpy で受け取る』だけ。GPU 直叩き汎用 API 完成。")
    g.close()


if __name__ == "__main__":
    main()
