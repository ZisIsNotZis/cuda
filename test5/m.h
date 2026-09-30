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
  template <bool FIX = true, int TH = 1, bool SHR = false, bool RED = false, bool T = false>
  __device__ __forceinline__ void ld(const int4 *__restrict__ _, int Y = 1, int4 *__restrict__ shr = NULL, int th = 0) {
    int lan;
    asm("mov.b32 %0,%laneid;" : "=r"(lan) :);
    if constexpr (TH > 1 && !SHR) {
      _ += lan / 2 * Y * 2 + lan % 2;
      if (!th) {
        if constexpr (RED)
          asm("ld.relaxed.gpu.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3},[%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(_));
        else
          *(int4 *)this = *_;
        shr[lan / 2 * Y * 2 + lan % 2] = *(int4 *)this;
      }
      __syncthreads();
      ld<FIX, 1, true>(shr);
    } else if constexpr (!FIX || !SHR || RED) {
      _ += (lan % 2 * 8 + lan / 4) * Y * 2 + lan / 2 % 2;
      if constexpr (RED)
        if constexpr (SHR)
          asm("cvta.to.shared.b64 %4,%4;"
              "ld.relaxed.cta.shared.v4.b32 {%0,%1,%2,%3},[%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(_));
        else
          asm("ld.relaxed.gpu.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3},[%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(_));
      else
        *(int4 *)this = *_;
    } else {
      _ += lan / 2 * Y * 2 + lan % 2;
      if constexpr (T)
        asm("cvta.to.shared.b64 %4,%4;"
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(_));
      else
        asm("cvta.to.shared.b64 %4,%4;"
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(_));
    }
  }
  __device__ __forceinline__ void mm(const e2828 &a, const e2828 &b) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%0,%2},{%4,%5,%6,%7},{%8,%9},{%0,%2};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%1,%3},{%4,%5,%6,%7},{%10,%11},{%1,%3};" : "+r"(x), "+r"(y), "+r"(z), "+r"(w) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w));
  }
  template <bool FIX = true, bool SHR = false, bool RED = false>
  __device__ __forceinline__ void st(int4 *_, int Y) {
    int lan;
    asm("mov.b32 %0,%laneid;" : "=r"(lan) :);
    _ += (lan % 2 * 8 + lan / 4) * Y * 2 + lan / 2 % 2;
    if constexpr (FIX)
      shf(*this);
    if constexpr (RED)
      if constexpr (SHR)
        asm("cvta.to.shared.b64 %4,%4;"
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

template <int NX, int NY, int NZ, int FX, int FY, int FZ, int DF0, int DF1, int D0, int D1>
__global__ void __maxnreg__(255) m_ker(const int4 *__restrict__ XZ, const int4 *__restrict__ YZ, int4 *__restrict__ XY) {
  constexpr int X = NX / FX;
  constexpr int Y = NY / FY;
  constexpr int Z = NZ / FZ;
  constexpr int DF2 = 3 - DF0 - DF1;
  constexpr int D2 = 3 - D0 - D1;
  constexpr int F0 = !DF0 ? FX : (DF0 == 1 ? FY : FZ);
  constexpr int F1 = !DF1 ? FX : (DF1 == 1 ? FY : FZ);
  constexpr int F2 = !DF2 ? FX : (DF2 == 1 ? FY : FZ);
  constexpr int I0 = !D0 ? X : (D0 == 1 ? Y : Z);
  constexpr int I1 = !D1 ? X : (D1 == 1 ? Y : Z);
  constexpr int I2 = !D2 ? X : (D2 == 1 ? Y : Z);
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
              if (!x)
                yz.ld(YZ + (((fy * Y + y) * 16 * FZ + fz) * Z + z) * 2, FZ * Z);
              if (!y)
                xz.ld(XZ + (((fx * X + x) * 16 * FZ + fz) * Z + z) * 2, FZ * Z);
              if (!z)
                if (fz)
                  xy.ld<false>(XY + (((fx * X + x) * 16 * FY + fy) * Y + y) * 2, FY * Y);
                else
                  xy.ld();
              xy.mm(xz, yz);
              if (z == Z - 1)
                if (fz != FZ - 1)
                  xy.st<false>(XY + (((fx * X + x) * 16 * FY + fy) * Y + y) * 2, FY * Y);
                else
                  xy.st(XY + (((fx * X + x) * 16 * FY + fy) * Y + y) * 2, FY * Y);
            }
          }
        }
      }
}

template <int NX, int NY, int NZ, int FX, int FY, int FZ, int DF0, int DF1, int D0, int D1>
inline void m(int4 *_) {
  constexpr int X = NX / FX;
  constexpr int Y = NY / FY;
  constexpr int Z = NZ / FZ;
  constexpr int DF2 = 3 - DF0 - DF1;
  constexpr int D2 = 3 - D0 - D1;
  constexpr int F2 = !DF2 ? FX : (DF2 == 1 ? FY : FZ);
  constexpr bool I0 = (!D0 ? X : (D0 == 1 ? Y : Z)) > 1;
  constexpr bool I1 = (!D1 ? X : (D1 == 1 ? Y : Z)) > 1;
  constexpr bool I2 = (!D2 ? X : (D2 == 1 ? Y : Z)) > 1;
  if constexpr (I0 >= I1 && I1 >= I2 && (I1 || (D1 > D2 && (I0 || D0 > D1))) && (DF2 != D0 || F2 < 2) && (I1 + 1) * I2 < 96)
    m_ker<NX, NY, NZ, FX, FY, FZ, DF0, DF1, D0, D1><<<1, 32>>>(_, _, _);
}

template <int NX, int NY, int NZ, int FX, int FY, int FZ, int DF0, int DF1>
inline void m(int4 *_) {
  constexpr int DF2 = 3 - DF0 - DF1;
  constexpr bool F0 = (!DF0 ? FX : (DF0 == 1 ? FY : FZ)) > 1;
  constexpr bool F1 = (!DF1 ? FX : (DF1 == 1 ? FY : FZ)) > 1;
  constexpr bool F2 = (!DF2 ? FX : (DF2 == 1 ? FY : FZ)) > 1;
  if constexpr (F2 >= F1 && F1 >= F0 && (F1 || (DF1 > DF0 && (F2 || DF2 > DF1)))) {
    m<NX, NY, NZ, FX, FY, FZ, DF0, DF1, 0, 1>(_);
    m<NX, NY, NZ, FX, FY, FZ, DF0, DF1, 0, 2>(_);
    m<NX, NY, NZ, FX, FY, FZ, DF0, DF1, 1, 0>(_);
    m<NX, NY, NZ, FX, FY, FZ, DF0, DF1, 1, 2>(_);
    m<NX, NY, NZ, FX, FY, FZ, DF0, DF1, 2, 0>(_);
    m<NX, NY, NZ, FX, FY, FZ, DF0, DF1, 2, 1>(_);
  }
}

template <int NX, int NY, int NZ, int FX, int FY, int FZ>
inline void m(int4 *_) {
  m<NX, NY, NZ, FX, FY, FZ, 0, 1>(_);
  m<NX, NY, NZ, FX, FY, FZ, 0, 2>(_);
  m<NX, NY, NZ, FX, FY, FZ, 1, 0>(_);
  m<NX, NY, NZ, FX, FY, FZ, 1, 2>(_);
  m<NX, NY, NZ, FX, FY, FZ, 2, 0>(_);
  m<NX, NY, NZ, FX, FY, FZ, 2, 1>(_);
}