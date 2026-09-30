#include <assert.h>
#include <cstdio>

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
  template <bool SHR = false, bool FIX = true, bool RED = false, bool TH = false>
  __device__ __forceinline__ void ld(const int4 *__restrict__ _, int Y = 1, int4 *__restrict__ tmp = NULL, int th = 0) {
    int lan;
    asm("mov.b32 %0,%laneid;" : "=r"(lan) :);
    if constexpr (!SHR && TH) {
      _ += lan / 2 * 2 * Y + lan % 2;
      if (th) {
      } else if constexpr (RED) {
        asm("ld.relaxed.gpu.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3},[%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(_));
        tmp[lan] = *(int4 *)this;
      } else
        tmp[lan] = *_;
      __syncthreads();
      ld<true, FIX>(tmp);
    } else if constexpr (!SHR || !FIX || RED) {
      _ += (lan % 2 * 8 + lan / 4) * 2 * Y + lan / 2 % 2;
      if constexpr (!RED)
        *(int4 *)this = *_;
      else if constexpr (SHR)
        asm("{.reg .b64 a;"
            "cvta.to.shared.u64 a,%4;"
            "ld.relaxed.cta.shared.v4.b32 {%0,%1,%2,%3},[a];}" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(_));
      else
        asm("ld.relaxed.gpu.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3},[%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(_));
      if constexpr (FIX)
        shf(*this);
    } else {
      _ += lan / 2 * 2 * Y + lan % 2;
      asm("{.reg .b64 a;"
          "cvta.to.shared.u64 a,%4;"
          "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[a];}" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(_));
    }
  }
  __device__ __forceinline__ void mm(const e2828 &a, const e2828 &b) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%0,%2},{%4,%5,%6,%7},{%8,%9},{%0,%2};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%1,%3},{%4,%5,%6,%7},{%10,%11},{%1,%3};" : "+r"(x), "+r"(y), "+r"(z), "+r"(w) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w));
  }
  template <bool SHR = false, bool FIX = true, bool RED = false>
  __device__ __forceinline__ void st(int4 *_, int Y = 1) {
    int lan;
    asm("mov.b32 %0,%laneid;" : "=r"(lan) :);
    _ += (lan % 2 * 8 + lan / 4) * 2 * Y + lan / 2 % 2;
    if constexpr (FIX)
      shf(*this);
    if constexpr (!RED)
      *_ = *(int4 *)this;
    else if constexpr (SHR)
      asm("{.reg .b64 a;"
          "cvta.to.shared.u64 a,%4;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [a],%0;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [a+4],%1;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [a+8],%2;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [a+12],%3;}" ::"r"(x),
          "r"(y), "r"(z), "r"(w), "l"(_));
    else
      asm("red.relaxed.gpu.global.add.noftz.f16x2 [%4],%0;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%4+4],%1;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%4+8],%2;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%4+12],%3;" ::"r"(x),
          "r"(y), "r"(z), "r"(w), "l"(_));
  }
};

template <int DF0, int DF1, int FX, int FY, int FZ, int D0, int D1, int X, int Y, int Z, int DB0, int DB1, bool BZ_, int DT0, int DT1, bool TZ_, bool SX, bool SY, bool SZ>
__global__ void __maxnreg__(255) m_ker(const int4 *__restrict__ XZ, const int4 *__restrict__ YZ, int4 *__restrict__ XY) {
  extern __align__(16) __shared__ int4 tmp[][32];
  constexpr int DF2 = 3 - DF0 - DF1;
  constexpr int F0 = !DF0 ? FX : (DF0 == 1 ? FY : FZ);
  constexpr int F1 = !DF1 ? FX : (DF1 == 1 ? FY : FZ);
  constexpr int F2 = !DF2 ? FX : (DF2 == 1 ? FY : FZ);
  constexpr int D2 = 3 - D0 - D1;
  constexpr int I0 = !D0 ? X : (D0 == 1 ? Y : Z);
  constexpr int I1 = !D1 ? X : (D1 == 1 ? Y : Z);
  constexpr int I2 = !D2 ? X : (D2 == 1 ? Y : Z);
  const int B0 = gridDim.z;
  const int B1 = gridDim.y;
  const int B2 = gridDim.x;
  const int BX = !DB0 ? B0 : (!DB1 ? B1 : B2);
  const int BY = DB0 == 1 ? B0 : (DB1 == 1 ? B1 : B2);
  const int BZ = DB0 == 2 ? B0 : (DB1 == 2 ? B1 : B2);
  const int b0 = blockIdx.z;
  const int b1 = blockIdx.y;
  const int b2 = blockIdx.x;
  const int bx = !DB0 ? b0 : (!DB1 ? b1 : b2);
  const int by = DB0 == 1 ? b0 : (DB1 == 1 ? b1 : b2);
  const int bz = DB0 == 2 ? b0 : (DB1 == 2 ? b1 : b2);
  const int T0 = blockDim.z;
  const int T1 = blockDim.y;
  const int T2 = blockDim.x / 32;
  const int TX = !DT0 ? T0 : (!DT1 ? T1 : T2);
  const int TY = DT0 == 1 ? T0 : (DT1 == 1 ? T1 : T2);
  const int TZ = DT0 == 2 ? T0 : (DT1 == 2 ? T1 : T2);
  const int t0 = threadIdx.z;
  const int t1 = threadIdx.y;
  const int t2 = threadIdx.x / 32;
  const int tx = !DT0 ? t0 : (!DT1 ? t1 : t2);
  const int ty = DT0 == 1 ? t0 : (DT1 == 1 ? t1 : t2);
  const int tz = DT0 == 2 ? t0 : (DT1 == 2 ? t1 : t2);
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
  int4 *ox = tmp[ty * TZ + tz];
  int4 *oy = tmp[tx * TZ + tz];
  int4 *oz = tmp[tx * TY + ty];
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
              if (!x)
                yz.ld<false, true, false, SX>(YZ + (((fy * Y + y) * BYTY16FZ + fz) * Z + z) * BZTZ2, FZZBZTZ, ox, tx);
              if (!y)
                xz.ld<false, true, false, SY>(XZ + (((fx * X + x) * BXTX16FZ + fz) * Z + z) * BZTZ2, FZZBZTZ, oy, ty);
              if (!z)
                if constexpr (FZ == 1)
                  xy.ld();
                else if (!fz)
                  xy.ld();
                else
                  xy.ld<false, false, BZ_, SZ>(O, FYYBYTY, oz, tz);
              else
                xy.ld();
              xy.mm(xz, yz);
              if constexpr (TZ_) {
                xy.st<true, false, true>(oz);
                __syncthreads();
                if (tz)
                  continue;
                xy.ld<true, false, true>(oz);
              }
              if (z == Z - 1)
                if constexpr (FZ == 1)
                  xy.st<false, true, BZ_>(O, FYYBYTY);
                else if (fz == FZ - 1)
                  xy.st<false, true, BZ_>(O, FYYBYTY);
                else
                  xy.st<false, false, BZ_>(O, FYYBYTY);
            }
          }
        }
      }
}

template <int N, int DF0, int DF1, int FX, int FY, int FZ, int D0, int D1, int X, int Y, int Z, int DB0, int DB1, int BX, int BY, int BZ, int DT0, int DT1, bool SX, bool SY, bool SZ>
inline void m(int4 *_) {
  constexpr int TX = N - FX - X - BX;
  constexpr int TY = N - FY - Y - BY;
  constexpr int TZ = N - FZ - Z - BZ;
  constexpr int DB2 = 3 - DB0 - DB1;
  constexpr int B0 = !DB0 ? BX : (DB0 == 1 ? BY : BZ);
  constexpr int B1 = !DB1 ? BX : (DB1 == 1 ? BY : BZ);
  constexpr int B2 = !DB2 ? BX : (DB2 == 1 ? BY : BZ);
  constexpr int DT2 = 3 - DT0 - DT1;
  constexpr int T0 = !DT0 ? TX : (DT0 == 1 ? TY : TZ);
  constexpr int T1 = !DT1 ? TX : (DT1 == 1 ? TY : TZ);
  constexpr int T2 = !DT2 ? TX : (DT2 == 1 ? TY : TZ);
  const int SHR = max(max(SX ? 512 << TY + TZ : 0, SY ? 512 << TX + TZ : 0), TZ ? 512 << TX + TY : 0);
  m_ker<DF0, DF1, 1 << FX, 1 << FY, 1 << FZ, D0, D1, 1 << X, 1 << Y, 1 << Z, DB0, DB1, (bool)BZ, DT0, DT1, (bool)TZ, SX, SY, SZ><<<dim3(1 << B2, 1 << B1, 1 << B0), dim3(32 << T2, 1 << T1, 1 << T0), SHR>>>(_, _, _);
  cudaDeviceSynchronize();
  printf("%d,%d,%d,%d,%d %d,%d,%d,%d,%d %d,%d,%d,%d,%d %d,%d,%d,%d,%d %d,%d,%d %d %s\n", DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX, BY, BZ, DT0, DT1, TX, TY, TZ, SX, SY, SZ, SHR, cudaGetErrorString(cudaGetLastError()));
}

template <int N, int DF0, int DF1, int FX, int FY, int FZ, int D0, int D1, int X, int Y, int Z, int DB0, int DB1, int BX, int BY, int BZ, int DT0, int DT1, bool SX, bool SY>
inline void m(int4 *_) {
  m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX, BY, BZ, DT0, DT1, SX, SY, false>(_);
  // if constexpr (TZ > 1)
  //   m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX, BY, BZ, DT0, DT1, SX, SY, true>(_);
}

template <int N, int DF0, int DF1, int FX, int FY, int FZ, int D0, int D1, int X, int Y, int Z, int DB0, int DB1, int BX, int BY, int BZ, int DT0, int DT1, bool SX>
inline void m(int4 *_) {
  constexpr int TY = N - FY - Y - BY;
  m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX, BY, BZ, DT0, DT1, SX, false>(_);
  if constexpr (TY > 1)
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX, BY, BZ, DT0, DT1, SX, true>(_);
}

template <int N, int DF0, int DF1, int FX, int FY, int FZ, int D0, int D1, int X, int Y, int Z, int DB0, int DB1, int BX, int BY, int BZ, int DT0, int DT1>
inline void m(int4 *_) {
  constexpr int TX = N - FX - X - BX;
  constexpr int TY = N - FY - Y - BY;
  constexpr int TZ = N - FZ - Z - BZ;
  constexpr int DT2 = 3 - DT0 - DT1;
  constexpr int T0 = !DT0 ? TX : (DT0 == 1 ? TY : TZ);
  constexpr int T1 = !DT1 ? TX : (DT1 == 1 ? TY : TZ);
  constexpr int T2 = !DT2 ? TX : (DT2 == 1 ? TY : TZ);
  constexpr bool T0_ = T0;
  constexpr bool T1_ = T1;
  constexpr bool T2_ = T2;
  if constexpr (T2_ >= T1_ && T1_ >= T0_ && (T1_ || (DT1 > DT0 && (T2_ || DT2 > DT1))) && TX + TY + TZ <= 3) {
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX, BY, BZ, DT0, DT1, false>(_);
    if constexpr (TX > 1)
      m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX, BY, BZ, DT0, DT1, true>(_);
  }
}

template <int N, int DF0, int DF1, int FX, int FY, int FZ, int D0, int D1, int X, int Y, int Z, int DB0, int DB1, int BX, int BY, int BZ>
inline void m(int4 *_) {
  constexpr int DB2 = 3 - DB0 - DB1;
  constexpr int B0 = !DB0 ? BX : (DB0 == 1 ? BY : BZ);
  constexpr int B1 = !DB1 ? BX : (DB1 == 1 ? BY : BZ);
  constexpr int B2 = !DB2 ? BX : (DB2 == 1 ? BY : BZ);
  constexpr bool B0_ = B0;
  constexpr bool B1_ = B1;
  constexpr bool B2_ = B2;
  if constexpr (B2_ >= B1_ && B1_ >= B0_ && (B1_ || (DB1 > DB0 && (B2_ || DB2 > DB1)))) {
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX, BY, BZ, 0, 1>(_);
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX, BY, BZ, 0, 2>(_);
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX, BY, BZ, 1, 0>(_);
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX, BY, BZ, 1, 2>(_);
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX, BY, BZ, 2, 0>(_);
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX, BY, BZ, 2, 1>(_);
  }
  if constexpr (FZ + Z + BZ < N)
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX, BY, BZ + 1>(_);
}

template <int N, int DF0, int DF1, int FX, int FY, int FZ, int D0, int D1, int X, int Y, int Z, int DB0, int DB1, int BX, int BY>
inline void m(int4 *_) {
  m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX, BY, 0>(_);
  if constexpr (FY + Y + BY < N)
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX, BY + 1>(_);
}

template <int N, int DF0, int DF1, int FX, int FY, int FZ, int D0, int D1, int X, int Y, int Z, int DB0, int DB1, int BX>
inline void m(int4 *_) {
  m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX, 0>(_);
  if constexpr (FX + X + BX < N)
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, DB0, DB1, BX + 1>(_);
}

template <int N, int DF0, int DF1, int FX, int FY, int FZ, int D0, int D1, int X, int Y, int Z>
inline void m(int4 *_) {
  constexpr int DF2 = 3 - DF0 - DF1;
  constexpr int F2 = !DF2 ? FX : (DF2 == 1 ? FY : FZ);
  constexpr int D2 = 3 - D0 - D1;
  constexpr int I0 = !D0 ? X : (D0 == 1 ? Y : Z);
  constexpr int I1 = !D1 ? X : (D1 == 1 ? Y : Z);
  constexpr int I2 = !D2 ? X : (D2 == 1 ? Y : Z);
  if constexpr (I0 >= I1 && I1 >= I2 && (I1 || (D1 > D2 && (I0 || D0 > D1))) && (DF2 != D0 || !F2) && I1 + I2 <= 7) {
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, 0, 1, 0>(_);
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, 0, 2, 0>(_);
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, 1, 0, 0>(_);
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, 1, 2, 0>(_);
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, 2, 0, 0>(_);
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z, 2, 1, 0>(_);
  }
  if constexpr (FZ + Z < N)
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, Z + 1>(_);
}

template <int N, int DF0, int DF1, int FX, int FY, int FZ, int D0, int D1, int X, int Y>
inline void m(int4 *_) {
  m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y, 0>(_);
  if constexpr (FY + Y < N)
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, Y + 1>(_);
}

template <int N, int DF0, int DF1, int FX, int FY, int FZ, int D0, int D1, int X>
inline void m(int4 *_) {
  m<N, DF0, DF1, FX, FY, FZ, D0, D1, X, 0>(_);
  if constexpr (FX + X < N)
    m<N, DF0, DF1, FX, FY, FZ, D0, D1, X + 1>(_);
}

template <int N, int DF0, int DF1, int FX, int FY, int FZ>
inline void m(int4 *_) {
  constexpr int DF2 = 3 - DF0 - DF1;
  constexpr int F0 = !DF0 ? FX : (DF0 == 1 ? FY : FZ);
  constexpr int F1 = !DF1 ? FX : (DF1 == 1 ? FY : FZ);
  constexpr int F2 = !DF2 ? FX : (DF2 == 1 ? FY : FZ);
  constexpr bool F0_ = F0;
  constexpr bool F1_ = F1;
  constexpr bool F2_ = F2;
  if constexpr (F2_ >= F1_ && F1_ >= F0_ && (F1_ || (DF1 > DF0 && (F2_ || DF2 > DF1)))) {
    m<N, DF0, DF1, FX, FY, FZ, 0, 1, N - FX, N - FY, N - FZ>(_);
    m<N, DF0, DF1, FX, FY, FZ, 0, 2, N - FX, N - FY, N - FZ>(_);
    m<N, DF0, DF1, FX, FY, FZ, 1, 0, N - FX, N - FY, N - FZ>(_);
    m<N, DF0, DF1, FX, FY, FZ, 1, 2, N - FX, N - FY, N - FZ>(_);
    m<N, DF0, DF1, FX, FY, FZ, 2, 0, N - FX, N - FY, N - FZ>(_);
    m<N, DF0, DF1, FX, FY, FZ, 2, 1, N - FX, N - FY, N - FZ>(_);
  }
}

template <int N, int FX, int FY, int FZ>
inline void m(int4 *_) {
  m<N, 0, 1, FX, FY, FZ>(_);
  m<N, 0, 2, FX, FY, FZ>(_);
  m<N, 1, 0, FX, FY, FZ>(_);
  m<N, 1, 2, FX, FY, FZ>(_);
  m<N, 2, 0, FX, FY, FZ>(_);
  m<N, 2, 1, FX, FY, FZ>(_);
}
