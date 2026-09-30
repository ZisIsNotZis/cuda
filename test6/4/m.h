// #include <cstdio>

template <class T>
__device__ __forceinline__ void swap(T &_, T &__) {
  const T t = _;
  _ = __;
  __ = t;
}

__device__ __forceinline__ void shf(int4 &_) {
  int lan;
  asm("mov.b32 %0,%laneid;" : "=r"(lan) :);
  if (lan % 2) {
    swap(_.x, _.y);
    swap(_.z, _.w);
  }
  if (lan / 2 % 2)
    swap(((long2 &)_).x, ((long2 &)_).y);
  _.y = __shfl_xor_sync(0xffffffff, _.y, 1);
  _.z = __shfl_xor_sync(0xffffffff, _.z, 2);
  _.w = __shfl_xor_sync(0xffffffff, _.w, 3);
  if (lan / 2 % 2)
    swap(((long2 &)_).x, ((long2 &)_).y);
  if (lan % 2) {
    swap(_.x, _.y);
    swap(_.z, _.w);
  }
}

struct e2828 : int4 {
  __device__ __forceinline__ void ld() {
    *(long2 *)this = make_long2(0, 0);
  }
  template <bool FIX = true, bool SHR = false, bool RED = false, bool TH = false, bool T = false>
  __device__ __forceinline__ void ld(const int4 *__restrict__ _, int Y = 1, int4 *__restrict__ tmp = NULL, int th = 0) {
    int lan;
    asm("mov.b32 %0,%laneid;" : "=r"(lan) :);
    if constexpr (TH && !SHR) {
      _ += lan / 2 * Y * 2 + lan % 2;
      if (!th) {
        if constexpr (RED)
          asm("ld.relaxed.gpu.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3},[%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(_));
        else
          *(int4 *)this = *_;
        tmp[lan] = *(int4 *)this;
      }
      __syncthreads();
      ld<FIX, true>(tmp);
    } else if constexpr (!FIX || !SHR || RED) {
      _ += (lan % 2 * 8 + lan / 4) * Y * 2 + lan / 2 % 2;
      if constexpr (RED)
        if constexpr (SHR)
          asm("cvta.to.shared.u64 %4,%4;"
              "ld.relaxed.cta.shared.v4.b32 {%0,%1,%2,%3},[%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(_));
        else
          asm("ld.relaxed.gpu.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3},[%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(_));
      else
        *(int4 *)this = *_;
    } else {
      _ += lan / 2 * Y * 2 + lan % 2;
      if constexpr (T)
        asm("cvta.to.shared.u64 %4,%4;"
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(_));
      else
        asm("cvta.to.shared.u64 %4,%4;"
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(_));
    }
  }
  __device__ __forceinline__ void mm(const e2828 &a, const e2828 &b) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%0,%2},{%4,%5,%6,%7},{%8,%9},{%0,%2};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%1,%3},{%4,%5,%6,%7},{%10,%11},{%1,%3};" : "+r"(x), "+r"(y), "+r"(z), "+r"(w) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w));
  }
  template <bool FIX = true, bool SHR = false, bool RED = false>
  __device__ __forceinline__ void st(int4 *_, int Y = 1) {
    int lan;
    asm("mov.b32 %0,%laneid;" : "=r"(lan) :);
    _ += (lan % 2 * 8 + lan / 4) * Y * 2 + lan / 2 % 2;
    if constexpr (FIX)
      shf(*this);
    if constexpr (RED)
      if constexpr (SHR)
        asm("cvta.to.shared.u64 %4,%4;"
            "red.relaxed.cta.shared.add.noftz.f16x2 [%4],%0;"
            "red.relaxed.cta.shared.add.noftz.f16x2 [%4+4],%1;"
            "red.relaxed.cta.shared.add.noftz.f16x2 [%4+8],%2;"
            "red.relaxed.cta.shared.add.noftz.f16x2 [%4+12],%3;" ::"r"(x),
            "r"(y), "r"(z), "r"(w), "l"(_));
      else
        asm("red.relaxed.gpu.global.add.noftz.f16x2 [%4],%0;"
            "red.relaxed.gpu.global.add.noftz.f16x2 [%4+4],%1;"
            "red.relaxed.gpu.global.add.noftz.f16x2 [%4+8],%2;"
            "red.relaxed.gpu.global.add.noftz.f16x2 [%4+12],%3;" ::"r"(x),
            "r"(y), "r"(z), "r"(w), "l"(_));
    else
      *_ = *(int4 *)this;
  };
};

template <int FX, int FY, int FZ, int DF0, int DF1, int X, int Y, int Z, int D0, int D1, bool BZ_, bool TX_, bool TY_, bool TZ_>
__global__ void __maxnreg__(255) m_ker(const int4 *__restrict__ XZ, const int4 *__restrict__ YZ, int4 *__restrict__ XY) {
  extern __shared__ __align__(16) int4 tmp[][32];
  constexpr int DF2 = 3 - DF0 - DF1;
  constexpr int D2 = 3 - D0 - D1;
  constexpr int F0 = !DF0 ? FX : (DF0 == 1 ? FY : FZ);
  constexpr int F1 = !DF1 ? FX : (DF1 == 1 ? FY : FZ);
  constexpr int F2 = !DF2 ? FX : (DF2 == 1 ? FY : FZ);
  constexpr int I0 = !D0 ? X : (D0 == 1 ? Y : Z);
  constexpr int I1 = !D1 ? X : (D1 == 1 ? Y : Z);
  constexpr int I2 = !D2 ? X : (D2 == 1 ? Y : Z);
  const int BX = gridDim.z;
  const int BY = gridDim.y;
  const int BZ = gridDim.x;
  const int bx = blockIdx.z;
  const int by = blockIdx.y;
  const int bz = blockIdx.x;
  const int TX = blockDim.z;
  const int TY = blockDim.y;
  const int TZ = blockDim.x / 32;
  const int tx = threadIdx.z;
  const int ty = threadIdx.y;
  const int tz = threadIdx.x / 32;
  const int BYTY16FZ = BY * TY * 16 * FZ;
  const int BXTX16FZ = BX * TX * 16 * FZ;
  const int BXTX16FY = BX * TX * 16 * FY;
  const int BYTY2 = BY * TY * 2;
  const int BZTZ2 = BZ * TZ * 2;
  const int FYYBYTY = FY * Y * BY * TY;
  const int FZZBZTZ = FZ * Z * BZ * TZ;
  XZ += (((bx * TX + tx) * 16 * FZ * Z * BZ + bz) * TZ + tz) * 2;
  XY += (((bx * TX + tx) * 16 * FY * Y * BY + by) * TY + ty) * 2;
  YZ += (((by * TY + ty) * 16 * FZ * Z * BZ + bz) * TZ + tz) * 2;
#pragma unroll
  for (int f0 = 0; f0 < F0; f0++)
#pragma unroll
    for (int f1 = 0; f1 < F1; f1++)
#pragma unroll
      for (int f2 = 0; f2 < F2; f2++) {
        const int fx = !DF0 ? f0 : (!DF1 ? f1 : f2);
        const int fy = DF0 == 1 ? f0 : (DF1 == 1 ? f1 : f2);
        const int fz = DF0 == 2 ? f0 : (DF1 == 2 ? f1 : f2);
        e2828 a0[I1][I2];
#pragma unroll
        for (int i0 = 0; i0 < I0; i0++) {
          e2828 a1[I2];
#pragma unroll
          for (int i1 = 0; i1 < I1; i1++) {
            e2828 a2;
#pragma unroll
            for (int i2 = 0; i2 < I2; i2++) {
              const int x = !D0 ? i0 : (!D1 ? i1 : i2);
              const int y = D0 == 1 ? i0 : (D1 == 1 ? i1 : i2);
              const int z = D0 == 2 ? i0 : (D1 == 2 ? i1 : i2);
              e2828 &yz = !D0 ? a0[i1][i2] : (D0 == 1 ? a1[i2] : a2);
              e2828 &xz = !D1 ? a0[i1][i2] : (D1 == 1 ? a1[i2] : a2);
              e2828 &xy = !D2 ? a0[i1][i2] : (D2 == 1 ? a1[i2] : a2);
              int4 *O = XY + (((fx * X + x) * BXTX16FY + fy) * Y + y) * BYTY2;
              int4 *o = tmp[tx * TY + ty];
              if (!x)
                yz.ld<true, false, false, TX_>(YZ + (((fy * Y + y) * BYTY16FZ + fz) * Z + z) * BZTZ2, FZZBZTZ, tmp[ty * TZ + tz], tx);
              if (!y)
                xz.ld<true, false, false, TY_>(XZ + (((fx * X + x) * BXTX16FZ + fz) * Z + z) * BZTZ2, FZZBZTZ, tmp[tx * TZ + tz], ty);
              if (!z)
                if constexpr (FZ == 1)
                  xy.ld();
                else if (!fz)
                  xy.ld();
                else
                  xy.ld<false, false, BZ_, TZ_>(O, FYYBYTY, o, tz);
              xy.mm(xz, yz);
              if constexpr (TZ_) {
                xy.st<false, true, true>(o);
                __syncthreads();
                if (tz)
                  continue;
                xy.ld<false, true, true>(o);
              }
              if (z == Z - 1) {
                if constexpr (FZ == 1)
                  shf(xy);
                else if (fz == FZ - 1)
                  shf(xy);
                xy.st<true, false, BZ_>(O, FYYBYTY);
              }
            }
          }
        }
      }
}

template <int N, int FX, int FY, int FZ, int DF0, int DF1, int X, int Y, int Z, int D0, int D1, int BX, int BY, int BZ>
inline void m(int4 *_) {
  constexpr int TX = N - FX - X - BX;
  constexpr int TY = N - FY - Y - BY;
  constexpr int TZ = N - FZ - Z - BZ;
  if constexpr(TX + TY + TZ <= 3) {
    m_ker<1 << FX, 1 << FY, 1 << FZ, DF0, DF1, 1 << X, 1 << Y, 1 << Z, D0, D1, (bool)BZ, (bool)TX, (bool)TY, (bool)TZ><<<dim3(1 << BZ, 1 << BY, 1 << BX), dim3(32 << TZ, 1 << TY, 1 << TX), max(max(TX ? 512 << TY + TZ : 0, TY ? 512 << TX + TZ : 0), TZ ? 512 << TX + TY : 0)>>>(_, _, _);
    // cudaDeviceSynchronize();
    // printf("%d,%d,%d %d,%d,%d %d,%d,%d,%d,%d %d,%d,%d,%d,%d %s\n", BX, BY, BZ, TX, TY, TZ, FX, FY, FZ, DF0, DF1, X, Y, Z, D0, D1, cudaGetErrorString(cudaGetLastError()));
  }
  if constexpr(FZ + Z + BZ < N)
    m<N, FX, FY, FZ, DF0, DF1, X, Y, Z, D0, D1, BX, BY, BZ + 1>(_);
}

template <int N, int FX, int FY, int FZ, int DF0, int DF1, int X, int Y, int Z, int D0, int D1, int BX, int BY>
inline void m(int4 *_) {
  m<N, FX, FY, FZ, DF0, DF1, X, Y, Z, D0, D1, BX, BY, 0>(_);
  if constexpr(FY + Y + BY < N)
    m<N, FX, FY, FZ, DF0, DF1, X, Y, Z, D0, D1, BX, BY + 1>(_);
}

template <int N, int FX, int FY, int FZ, int DF0, int DF1, int X, int Y, int Z, int D0, int D1, int BX>
inline void m(int4 *_) {
  m<N, FX, FY, FZ, DF0, DF1, X, Y, Z, D0, D1, BX, 0>(_);
  if constexpr(FX + X + BX < N)
    m<N, FX, FY, FZ, DF0, DF1, X, Y, Z, D0, D1, BX + 1>(_);
}

template <int N, int FX, int FY, int FZ, int DF0, int DF1, int X, int Y, int Z, int D0, int D1>
inline void m(int4 *_) {
  constexpr int DF2 = 3 - DF0 - DF1;
  constexpr int D2 = 3 - D0 - D1;
  constexpr int F2 = !DF2 ? FX : (DF2 == 1 ? FY : FZ);
  constexpr int I0 = !D0 ? X : (D0 == 1 ? Y : Z);
  constexpr int I1 = !D1 ? X : (D1 == 1 ? Y : Z);
  constexpr int I2 = !D2 ? X : (D2 == 1 ? Y : Z);
  constexpr bool I0_ = I0;
  constexpr bool I1_ = I1;
  constexpr bool I2_ = I2;
  if constexpr (I0_ >= I1_ && I1_ >= I2_ && (I1_ || (D1 > D2 && (I0_ || D0 > D1))) && (DF2 != D0 || F2 < 2) && I1 + I2 < 7)
    m<N, FX, FY, FZ, DF0, DF1, X, Y, Z, D0, D1, 0>(_);
}

template <int N, int FX, int FY, int FZ, int DF0, int DF1, int X, int Y, int Z>
inline void m(int4 *_) {
  m<N, FX, FY, FZ, DF0, DF1, X, Y, Z, 0, 1>(_);
  m<N, FX, FY, FZ, DF0, DF1, X, Y, Z, 0, 2>(_);
  m<N, FX, FY, FZ, DF0, DF1, X, Y, Z, 1, 0>(_);
  m<N, FX, FY, FZ, DF0, DF1, X, Y, Z, 1, 2>(_);
  m<N, FX, FY, FZ, DF0, DF1, X, Y, Z, 2, 0>(_);
  m<N, FX, FY, FZ, DF0, DF1, X, Y, Z, 2, 1>(_);
  if constexpr (FZ + Z < N)
    m<N, FX, FY, FZ, DF0, DF1, X, Y, Z + 1>(_);
}

template <int N, int FX, int FY, int FZ, int DF0, int DF1, int X, int Y>
inline void m(int4 *_) {
  m<N, FX, FY, FZ, DF0, DF1, X, Y, 0>(_);
  if constexpr (FY + Y < N)
    m<N, FX, FY, FZ, DF0, DF1, X, Y + 1>(_);
}

template <int N, int FX, int FY, int FZ, int DF0, int DF1, int X>
inline void m(int4 *_) {
  m<N, FX, FY, FZ, DF0, DF1, X, 0>(_);
  if constexpr (FX + X < N)
    m<N, FX, FY, FZ, DF0, DF1, X + 1>(_);
}

template <int N, int FX, int FY, int FZ, int DF0, int DF1>
inline void m(int4 *_) {
  constexpr int DF2 = 3 - DF0 - DF1;
  constexpr bool F0_ = !DF0 ? FX : (DF0 == 1 ? FY : FZ);
  constexpr bool F1_ = !DF1 ? FX : (DF1 == 1 ? FY : FZ);
  constexpr bool F2_ = !DF2 ? FX : (DF2 == 1 ? FY : FZ);
  if constexpr (F2_ >= F1_ && F1_ >= F0_ && (F1_ || (DF1 > DF0 && (F2_ || DF2 > DF1))))
    m<N, FX, FY, FZ, DF0, DF1, 0>(_);
}

template <int N, int FX, int FY, int FZ>
inline void m(int4 *_) {
  m<N, FX, FY, FZ, 0, 1>(_);
  m<N, FX, FY, FZ, 0, 2>(_);
  m<N, FX, FY, FZ, 1, 0>(_);
  m<N, FX, FY, FZ, 1, 2>(_);
  m<N, FX, FY, FZ, 2, 0>(_);
  m<N, FX, FY, FZ, 2, 1>(_);
}
