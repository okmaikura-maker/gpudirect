"""
gpudirect.opencl — OpenCL.dll を ctypes 直叩きするクロスベンダ backend。
NVIDIA / AMD / Intel iGPU など、OpenCL が見える全デバイスを列挙・飽和できる。
CUDA Toolkit も pyopencl も不要(ドライバ同梱の OpenCL.dll のみ)。
"""
import ctypes
import time

_cl = ctypes.WinDLL("OpenCL")
P = ctypes.c_void_p
_SZ = ctypes.c_size_t

for _n, _a in [
    ("clGetPlatformIDs", [ctypes.c_uint, P, P]),
    ("clGetPlatformInfo", [P, ctypes.c_uint, _SZ, P, P]),
    ("clGetDeviceIDs", [P, ctypes.c_ulonglong, ctypes.c_uint, P, P]),
    ("clGetDeviceInfo", [P, ctypes.c_uint, _SZ, P, P]),
    ("clCreateContext", [P, ctypes.c_uint, P, P, P, P]),
    ("clCreateCommandQueue", [P, P, ctypes.c_ulonglong, P]),
    ("clCreateProgramWithSource", [P, ctypes.c_uint, P, P, P]),
    ("clBuildProgram", [P, ctypes.c_uint, P, ctypes.c_char_p, P, P]),
    ("clGetProgramBuildInfo", [P, P, ctypes.c_uint, _SZ, P, P]),
    ("clCreateKernel", [P, ctypes.c_char_p, P]),
    ("clCreateBuffer", [P, ctypes.c_ulonglong, _SZ, P, P]),
    ("clSetKernelArg", [P, ctypes.c_uint, _SZ, P]),
    ("clEnqueueNDRangeKernel", [P, P, ctypes.c_uint, P, P, P, ctypes.c_uint, P, P]),
    ("clFinish", [P]),
    ("clReleaseKernel", [P]), ("clReleaseProgram", [P]),
    ("clReleaseMemObject", [P]), ("clReleaseCommandQueue", [P]), ("clReleaseContext", [P]),
]:
    f = getattr(_cl, _n); f.restype = ctypes.c_int if _n.startswith(("clGet", "clBuild",
        "clSet", "clEnqueue", "clFinish", "clRelease")) else P
    f.argtypes = _a
# 返り値がハンドルのものを個別修正
for _n in ("clCreateContext", "clCreateCommandQueue", "clCreateProgramWithSource",
           "clCreateKernel", "clCreateBuffer"):
    getattr(_cl, _n).restype = P

CL_ALL = 0xFFFFFFFF
PNAME = 0x0902; DNAME = 0x102B; DTYPE = 0x1000; DCU = 0x1002; DCLK = 0x100C
CL_MEM_WRITE_ONLY = 1 << 1
_KIND = {4: "GPU", 2: "CPU", 8: "ACCEL", 1: "DEFAULT"}

# 8 本独立 FMA 列でパイプラインを埋める飽和カーネル
_BURN = b"""
__kernel void burn(__global float* out, const int iters){
  int i = get_global_id(0);
  float a0=i*1e-6f, a1=a0+0.1f, a2=a0+0.2f, a3=a0+0.3f;
  float a4=a0+0.4f, a5=a0+0.5f, a6=a0+0.6f, a7=a0+0.7f;
  for(int k=0;k<iters;k++){
    a0=a0*0.9f+0.1f; a1=a1*0.9f+0.1f; a2=a2*0.9f+0.1f; a3=a3*0.9f+0.1f;
    a4=a4*0.9f+0.1f; a5=a5*0.9f+0.1f; a6=a6*0.9f+0.1f; a7=a7*0.9f+0.1f;
  }
  out[i]=a0+a1+a2+a3+a4+a5+a6+a7;
}
"""


def _sinfo(fn, h, param):
    sz = _SZ(); fn(h, param, 0, None, ctypes.byref(sz))
    b = ctypes.create_string_buffer(sz.value); fn(h, param, sz.value, b, None)
    return b.value.decode(errors="replace")


def _uinfo(h, param):
    v = ctypes.c_uint(); _cl.clGetDeviceInfo(h, param, 4, ctypes.byref(v), None); return v.value


def list_devices():
    """OpenCL で見える全デバイス(NVIDIA/AMD/Intel iGPU 等)を列挙。"""
    out = []
    n = ctypes.c_uint(); _cl.clGetPlatformIDs(0, None, ctypes.byref(n))
    plats = (P*n.value)(); _cl.clGetPlatformIDs(n.value, plats, None)
    for pi in range(n.value):
        p = plats[pi]; pname = _sinfo(_cl.clGetPlatformInfo, p, PNAME)
        nd = ctypes.c_uint()
        if _cl.clGetDeviceIDs(p, CL_ALL, 0, None, ctypes.byref(nd)) != 0:
            continue
        devs = (P*nd.value)(); _cl.clGetDeviceIDs(p, CL_ALL, nd.value, devs, None)
        for di in range(nd.value):
            d = devs[di]; t = ctypes.c_ulonglong()
            _cl.clGetDeviceInfo(d, DTYPE, 8, ctypes.byref(t), None)
            out.append(dict(handle=devs[di], platform=pname,
                            name=_sinfo(_cl.clGetDeviceInfo, d, DNAME),
                            kind=_KIND.get(t.value, str(t.value)),
                            cu=_uinfo(d, DCU), mhz=_uinfo(d, DCLK)))
    return out


def saturate(device_handle, seconds=8.0, iters=3000, gsize=1 << 20):
    """指定 OpenCL デバイスを seconds 秒あいだ最大飽和させる。"""
    err = ctypes.c_int()
    dev = P(device_handle)
    ctx = _cl.clCreateContext(None, 1, ctypes.byref(dev), None, None, ctypes.byref(err))
    q = _cl.clCreateCommandQueue(ctx, dev, 0, ctypes.byref(err))
    src = ctypes.c_char_p(_BURN); ln = _SZ(len(_BURN))
    prog = _cl.clCreateProgramWithSource(ctx, 1, ctypes.byref(src), ctypes.byref(ln), ctypes.byref(err))
    if _cl.clBuildProgram(prog, 1, ctypes.byref(dev), None, None, None) != 0:
        log = _sinfo(lambda h, pa, s, b, x: _cl.clGetProgramBuildInfo(h, dev, pa, s, b, x), prog, 0x1183)
        raise RuntimeError("OpenCL build failed:\n"+log)
    kern = _cl.clCreateKernel(prog, b"burn", ctypes.byref(err))
    buf = _cl.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, gsize*4, None, ctypes.byref(err))
    _cl.clSetKernelArg(kern, 0, ctypes.sizeof(P), ctypes.byref(P(buf)))
    it = ctypes.c_int(iters); _cl.clSetKernelArg(kern, 1, 4, ctypes.byref(it))
    g = (_SZ*1)(gsize)
    launches = 0; t0 = time.perf_counter()
    while time.perf_counter()-t0 < seconds:
        for _ in range(3):
            _cl.clEnqueueNDRangeKernel(q, kern, 1, None, g, None, 0, None, None)
            launches += 1
        _cl.clFinish(q)
    dt = time.perf_counter()-t0
    for r, h in [(_cl.clReleaseMemObject, buf), (_cl.clReleaseKernel, kern),
                 (_cl.clReleaseProgram, prog), (_cl.clReleaseCommandQueue, q),
                 (_cl.clReleaseContext, ctx)]:
        r(P(h))
    flop = launches*gsize*iters*8*2
    return dict(seconds=dt, launches=launches, gflops=flop/dt/1e9)
