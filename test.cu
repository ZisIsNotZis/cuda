#include <assert.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <stdio.h>

#define LEN(a) sizeof(a) / sizeof(*a)

template <int I, class F> __device__ __forceinline__ void For(F f) {
  f(I - 1);
  if constexpr (I > 1)
    For<I - 1>(f);
}

template <int I, class F> __device__ __forceinline__ auto At(int i, F f) {
  if (I - 1 == i) {
    return f(I - 1);
  }
  if constexpr (I > 2)
    return At<I - 1>(i, f);
  return f(0);
}

template <bool SYNC = true>
__device__ __forceinline__ void
cp(half *__restrict__ O, const half *__restrict__ A, unsigned AY, unsigned X,
   unsigned Y, unsigned th, unsigned TH) {
  if (!SYNC) {
    asm("cvta.to.shared.u64 %0,%0;" : "+l"(O) :);
  }
  th *= 8;
  TH *= 8;
  for (unsigned i_ = 0; i_ <= X * Y - 8; i_ += TH) {
    unsigned i = min(i_ + th, X * Y - 8);
    if (SYNC) {
      *(int4 *)(O + i) = *(int4 *)(A + i / Y * AY + i % Y);
      continue;
    }
    asm("cp.async.cg.shared.global[%0],[%1],16;" ::"l"(O + i),
        "l"(A + i / Y * AY + i % Y));
  }
}

template <class T> __device__ __forceinline__ void swap(T &a, T &b) {
  T t;
  t = a;
  a = b;
  b = t;
}

template <int N, class T>
__device__ __forceinline__ void shf(T (&_)[N], int shf = 0) {
  if (threadIdx.x & 1) {
    swap(_[0], _[1]);
    if constexpr (N > 3)
      swap(_[2], _[3]);
    if constexpr (N > 7) {
      swap(_[4], _[5]);
      swap(_[6], _[7]);
    }
  }
  if (N > 3 && threadIdx.x & 2) {
    swap(((long *)_)[0], ((long *)_)[1]);
    if constexpr (N > 7)
      swap(((long *)_)[2], ((long *)_)[3]);
  }
  if (N > 7 && threadIdx.x & 4) {
    swap(((int4 *)_)[0], ((int4 *)_)[1]);
  }
  for (int i = 1; i < N; i++) {
    _[i] = __shfl_xor_sync(0xffffffff, _[i], i << shf);
  }
  if (threadIdx.x & 1) {
    swap(_[0], _[1]);
    if constexpr (N > 3)
      swap(_[2], _[3]);
    if constexpr (N > 7) {
      swap(_[4], _[5]);
      swap(_[6], _[7]);
    }
  }
  if (N > 3 && threadIdx.x & 2) {
    swap(((long *)_)[0], ((long *)_)[1]);
    if constexpr (N > 7)
      swap(((long *)_)[2], ((long *)_)[3]);
  }
  if (N > 7 && threadIdx.x & 4) {
    swap(((int4 *)_)[0], ((int4 *)_)[1]);
  }
}

template <bool IF, class T> __device__ __forceinline__ void sort(T &A, T &B) {
  if constexpr (!IF) {
    T t = min(A, B);
    B = max(A, B);
    A = t;
  } else if (A > B) {
    T t = A;
    A = B;
    B = t;
  }
}

template <>
__device__ __forceinline__ void sort<false, half>(half &A, half &B) {
  half t = __hmin(A, B);
  B = __hmax(A, B);
  A = t;
}

template <>
__device__ __forceinline__ void sort<false, half2>(half2 &A, half2 &B) {
  half2 t = __hmin2(A, B);
  B = __hmax2(A, B);
  A = t;
}

template <>
__device__ __forceinline__ void sort<true, half2>(half2 &A, half2 &B) {
  half2 t = __hgt2(A, B);
  if (t.x) {
    half t = A.x;
    A.x = B.x;
    B.x = t;
  }
  if (t.y) {
    half t = A.y;
    A.y = B.y;
    B.y = t;
  }
}

template <bool IF, class T>
__device__ __forceinline__ void sort(T *A, T *B, int I, int MAX) {
  for (int i = MAX; i > 1; i--) {
    if (i > I) {
      continue;
    }
    sort<IF>(A[I - i], B[0]);
    sort<IF>(A[I - 1], B[i - 1]);
    for (int j = 1; j < i - 1; j++) {
      sort<IF>(A[I - i + j], A[I - 1]);
      sort<IF>(B[0], B[j]);
    }
  }
  sort<IF>(A[I - 1], B[0]);
}

template <bool IF, class T>
__device__ __forceinline__ void sort(T *A, int I, int N) {
  for (int i = N; i < I; i *= 2) {
    for (int j = 0; j < I; j += i * 2) {
      sort<IF>(A + j, A + j + i, i, I / 2);
    }
  }
}

template <int ALG, bool IF, class T>
__device__ __forceinline__ void sort(T *A, int I) {
  if constexpr (ALG == 0)
    for (int i = 0; i < I - 1; i++) {
      for (int j = i + 1; j < I; j++) {
        sort<IF>(A[i], A[j]);
      }
    }
  else if constexpr (ALG == 1)
    for (int i = 0; i < I - 1; i++) {
      for (int j = I - 1; j > i; j--) {
        sort<IF>(A[i], A[j]);
      }
    }
  else if constexpr (ALG == 2)
    for (int i = I - 1; i > 0; i--) {
      for (int j = 0; j < i; j++) {
        sort<IF>(A[i], A[j]);
      }
    }
  else if constexpr (ALG == 3)
    for (int i = 0; i < I - 1; i++) {
      for (int j = i - 1; j >= 0; j--) {
        sort<IF>(A[i], A[j]);
      }
    }
  else if constexpr (ALG == 4)
    for (int i = I - 1; i > 0; i--) {
      for (int j = 0; j < i; j++) {
        sort<IF>(A[j], A[j + 1]);
      }
    }
  else if constexpr (ALG == 5)
    for (int i = 0; i < I - 1; i++) {
      for (int j = I - 1; j > i; j--) {
        sort<IF>(A[j - 1], A[j]);
      }
    }
  else
    sort<IF>(A, I, 1);
}

template <bool IF, class T, int N>
__device__ __forceinline__ void sortsfl(T (&_)[N], int TH, int A, int B) {
  T a = __shfl_sync(0xffffffff, _[A % N], threadIdx.x / TH * TH + A / N);
  T b = __shfl_sync(0xffffffff, _[B % N], threadIdx.x / TH * TH + B / N);
  if constexpr (IF) {
    if (a > b) {
      T t = a;
      a = b;
      b = t;
    }
  } else if constexpr (std::is_same_v<T, half>) {
    T t = __hmin(a, b);
    b = __hmax(a, b);
    a = t;
  } else {
    T t = min(a, b);
    b = max(a, b);
    a = t;
  }
  if (threadIdx.x == threadIdx.x / TH * TH + A / N) {
    _[A % N] = a;
  }
  if (threadIdx.x == threadIdx.x / TH * TH + B / N) {
    _[B % N] = b;
  }
}

template <bool IF, class T, int N>
__device__ __forceinline__ void sortsfl(T (&_)[N], int TH, int A, int B, int I,
                                        int MAX) {
  for (int i = MAX; i > 1; i--) {
    if (i > I) {
      continue;
    }
    sortsfl<IF>(_, TH, A + I - i, B);
    sortsfl<IF>(_, TH, A + I - 1, B + i - 1);
    for (int j = 1; j < MAX - 1; j++) {
      if (j >= i - 1) {
        break;
      }
      sortsfl<IF>(_, TH, A + I - 1 - j, A + I - 1);
      sortsfl<IF>(_, TH, B, B + j);
    }
  }
  sortsfl<IF>(_, TH, A + I - 1, B);
}

template <int ALG, bool IF, class T, int N>
__device__ __forceinline__ void sortsfl(T (&_)[N]) {
  int I = N * 32;
  if constexpr (ALG == 0)
    for (int i = 0; i < I - 1; i++) {
      for (int j = i + 1; j < I; j++) {
        sortsfl<IF>(_, 32, i, j);
      }
    }
  else if constexpr (ALG == 1)
    for (int i = 0; i < I - 1; i++) {
      for (int j = I - 1; j > i; j--) {
        sortsfl<IF>(_, 32, i, j);
      }
    }
  else if constexpr (ALG == 2)
    for (int i = I - 1; i > 0; i--) {
      for (int j = 0; j < i; j++) {
        sortsfl<IF>(_, 32, i, j);
      }
    }
  else if constexpr (ALG == 3)
    for (int i = 0; i < I - 1; i++) {
      for (int j = i - 1; j >= 0; j--) {
        sortsfl<IF>(_, 32, i, j);
      }
    }
  else if constexpr (ALG == 4)
    for (int i = I - 1; i > 0; i--) {
      for (int j = 0; j < i; j++) {
        sortsfl<IF>(_, 32, j, j + 1);
      }
    }
  else if constexpr (ALG == 5)
    for (int i = 0; i < I - 1; i++) {
      for (int j = I - 1; j > i; j--) {
        sortsfl<IF>(_, 32, j - 1, j);
      }
    }
  else {
    sort<ALG - 6, IF>(_, N);
    for (int i = 1; i < 32; i *= 2) {
      sortsfl<IF>(_, i * 2, 0, i * N, i * N, 16 * N);
    }
  }
}

__device__ __forceinline__ void redux(int o, int *O) {
  asm volatile("{.shared .b64 share;.reg .b32 r;.reg .pred p;"
               "redux.sync.add.u32 %0,%0,-1;"
               "mov.b32 r,%laneid;"
               "setp.ne.b32 p,r,0;"
               "@p exit;"
               "mov.b32 r,%tid.x;"
               "setp.ne.b32 p,r,0;"
               "@!p st.shared.b32[share],0;"
               "bar.sync 0;"
               "red.relaxed.cta.shared.add.u32[share],%0;"
               "bar.sync 0;"
               "@p exit;"
               "ld.relaxed.cta.shared.b32 %0,[share];"
               "red.relaxed.gpu.global.add.u32[%1],%0;}" ::"r"(o),
               "l"(O));
}

template <int N, class T> struct tup {
  T _[N];

  __device__ __forceinline__ tup() {}

  __device__ __forceinline__ tup(T (&&x)[N]) : tup(*(tup<N, T> *)x) {}

  __device__ __forceinline__ T operator[](tup<N, T> x) {
    T t = x._[0];
    for (int i = 1; i < N; i++) {
      t = t * _[i] + x._[i];
    }
    return t;
  }

  __device__ __forceinline__ tup<N, T> operator[](T x) {
    tup<N, T> t;
    for (int i = N - 1; i >= 0; i--) {
      t._[i] = x % _[i];
      x /= _[i];
    }
    return t;
  }

  __device__ __forceinline__ tup<N, T> operator+(tup<N, T> x) {
    tup<N, T> t;
    for (int i = 0; i < N; i++) {
      t._[i] = _[i] + x._[i];
    }
    return t;
  }

  __device__ __forceinline__ T prod() {
    T t = 1;
    for (int i = 0; i < N; i++) {
      t *= _[i];
    }
    return t;
  }
};

template <bool PAK, bool IF, bool TR, class T, int N> union list {
  uint64_t pak[N * sizeof(T) / 8];
  T upak[N];

  __device__ __forceinline__ list() {
    for (int i = 0; i < LEN(pak); i++) {
      pak[i] = 0;
    }
  }

  __device__ __forceinline__ T operator[](int I) {
    if constexpr (PAK) {
      if constexpr (TR) {
        constexpr T MASK = (1 << 64 / N) - 1;
        const int BIT = 64 / N * I;
        T t = 0;
        for (int n = 0; n < LEN(pak); n++) {
          t |= ((T)(pak[n] >> 64 / N * n) & MASK) << BIT;
        }
        return t;
      } else {
        const int BIT = I * sizeof(T) * 8 % 64;
        I /= 8 / sizeof(T);
        if constexpr (IF) {
          return At<LEN(pak)>(I, [&](int i) { return pak[i]; }) >> BIT;
        } else
          return pak[I] >> BIT;
      }
    } else {
      if constexpr (IF) {
        return At<N>(I, [&](int i) { return upak[i]; });
      } else
        return upak[I];
    }
  }

  __device__ __forceinline__ void operator()(int I, T t) {
    if constexpr (PAK) {
      if constexpr (TR) {
        constexpr T MASK = (1 << 64 / N) - 1;
        const int BIT = 64 / N * I;
        const uint64_t AND = (-1ul ^ (uint64_t)MASK << BIT);
        for (int n = 0; n < LEN(pak); n++) {
          pak[n] = pak[n] & AND | (uint64_t)(t >> 64 / N * n & MASK) << BIT;
        }
      } else {
        constexpr T MASK = -1;
        const int BIT = I * sizeof(T) * 8 % 64;
        const uint64_t AND = (-1ul ^ (uint64_t)MASK << BIT);
        const uint64_t OR = (uint64_t)t << BIT;
        I /= 8 / sizeof(T);
        if constexpr (IF) {
          At<LEN(pak)>(I, [&](int i) {
            pak[i] = pak[i] & AND | OR;
            return 0;
          });
        } else
          pak[I] = pak[I] & AND | OR;
      }
    } else {
      if constexpr (IF) {
        At<N>(I, [&](int i) {
          upak[i] = t;
          return 0;
        });
      } else
        upak[I] = t;
    }
  };
};

struct e2828 {
  int _[4];

  __device__ __forceinline__ e2828() {
    for (int i = 0; i < 2; i++)
      ((long *)_)[i] = 0;
  }

  template <bool T = false, bool SHR = false>
  __device__ __forceinline__ e2828(const half *__restrict__ A, unsigned Y = 1) {
    if (SHR) {
      asm("cvta.to.shared %0,%0;" : "+l"(A) :);
      A += (threadIdx.x % 16 * Y * 2 + threadIdx.x / 16) * 8;
      if (T)
        asm("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16{%0,%1,%2,%3},[%4];"
            : "=r"(*_), "=r"(_[2]), "=r"(_[1]), "=r"(_[3])
            : "l"(A));
      else
        asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16{%0,%1,%2,%3},[%4];"
            : "=r"(*_), "=r"(_[1]), "=r"(_[2]), "=r"(_[3])
            : "l"(A));
    } else if (T) {
      *(int4 *)_ = ((int4 *)A)[tup{
          {2u, 4u, 2u, Y * 2u}}[{{threadIdx.x / 16, threadIdx.x % 4,
                                  threadIdx.x % 8 / 4, threadIdx.x % 16 / 8}}]];
      shf((half(&)[8])_, 2);
    } else {
      *(int4 *)_ = ((int4 *)A)[tup{{2u, 8u, Y * 2u}}[{
          {threadIdx.x % 2, threadIdx.x / 4, threadIdx.x % 4 / 2}}]];
      shf(_);
    }
  }

  __device__ __forceinline__ void mm(e2828 a, e2828 b) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%0,%2},{%4,%5,%6,%7}"
        ",{%8,%9},{%0,%2};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%1,%3},{%4,%5,%6,%7}"
        ",{%10,%11},{%1,%3};"
        : "+r"(_[0]), "+r"(_[1]), "+r"(_[2]), "+r"(_[3])
        : "r"(a._[0]), "r"(a._[1]), "r"(a._[2]), "r"(a._[3]), "r"(b._[0]),
          "r"(b._[1]), "r"(b._[2]), "r"(b._[3]));
  }

  __device__ __forceinline__ void relu() {
    half2 zero;
    *(int *)&zero = 0;
    for (int i = 0; i < 4; i++)
      ((half2 *)_)[i] = __hmax2(((half2 *)_)[i], zero);
  }

  template <bool T = false, bool SHR = false, bool RED = false>
  __device__ __forceinline__ void to(half *__restrict__ A, unsigned Y = 1) {
    if (T) {
      A += tup{{2u, 4u, 16u,
                Y * 2u}}[{{threadIdx.x / 16, threadIdx.x % 4,
                           threadIdx.x % 8 / 4, threadIdx.x % 16 / 8 * 8}}];
      shf((half(&)[8])_, 2);
    } else {
      A += tup{{2u, 8u, Y * 16u}}[{
          {threadIdx.x % 2, threadIdx.x / 4, threadIdx.x % 4 / 2 * 8}}];
      shf(_);
    }
    if (!RED)
      *(int4 *)A = *(int4 *)_;
    else if (SHR)
      asm volatile(
          "cvta.to.shared %0,%0;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0],%1;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+4],%2;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+8],%3;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+12],%4;" ::"l"(A),
          "r"(*_), "r"(_[1]), "r"(_[2]), "r"(_[3]));
    else
      asm volatile(
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0],%1;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+4],%2;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+8],%3;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+12],%4;" ::"l"(A),
          "r"(*_), "r"(_[1]), "r"(_[2]), "r"(_[3]));
  }
};

struct e828 {
  int _[2];

  template <bool T = false, bool SHR = false>
  __device__ __forceinline__ e828(const half *__restrict__ A, unsigned Y = 1) {
    if constexpr (SHR) {
      A += (threadIdx.x % 8 * Y * 2 + threadIdx.x / 8) * 8;
      if constexpr (T) {
        asm("ldmatrix.sync.aligned.m8n8.x2.trans.b16{%0,%1},[%4];"
            : "=r"(*_), "=r"(_[1])
            : "l"(A));
      } else {
        asm("ldmatrix.sync.aligned.m8n8.x2.b16{%0,%1},[%4];"
            : "=r"(*_), "=r"(_[1])
            : "l"(A));
      }
    } else {
      *(long *)_ = ((long *)A)[tup{{8u, Y * 2u, 2u}}[{
          {threadIdx.x / 4, threadIdx.x % 2, threadIdx.x % 4 / 2}}]];
      shf(_);
    }
  }

  __device__ __forceinline__ void to(half *__restrict__ A, unsigned Y = 1) {
    shf(_);
    ((long *)A)[tup{{8u, Y * 2u, 2u}}[{
        {threadIdx.x / 4, threadIdx.x % 2, threadIdx.x % 4 / 2}}]] = *(long *)_;
  }
};

struct e288 {
  int _[2];

  template <bool T = false, bool SHR = false>
  __device__ __forceinline__ e288(const half *__restrict__ A, unsigned Y = 1) {
    if constexpr (SHR) {
      A += threadIdx.x * Y * 8;
      if constexpr (T)
        asm("ldmatrix.sync.aligned.m8n8.x2.trans.b16{%0,%1},[%4];"
            : "=r"(*_), "=r"(_[1])
            : "l"(A));
      else
        asm("ldmatrix.sync.aligned.m8n8.x2.b16{%0,%1},[%4];"
            : "=r"(*_), "=r"(_[1])
            : "l"(A));
    } else {
      *(long *)_ = ((long *)A)[tup{{2u, 8u, Y, 2u}}[{
          {threadIdx.x % 4 / 2, threadIdx.x / 4, threadIdx.x % 2}}]];
      shf(_);
    }
  }

  __device__ __forceinline__ void mm(e2828 a, e828 b) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%0,%1},{%2,%3,%4,%5%"
        "1,{%6,%7},{%0,%1};"
        : "+r"(_[0]), "+r"(_[1])
        : "r"(a._[0]), "r"(a._[1]), "r"(a._[2]), "r"(a._[3]), "r"(b._[0]),
          "r"(b._[1]));
  }

  __device__ __forceinline__ void st(half *__restrict__ A, unsigned Y) {
    shf(_);
    ((long *)A)[tup{{2u, 8u, Y * 2u}}[{
        {threadIdx.x % 4 / 2, threadIdx.x / 4, threadIdx.x % 2}}]] = *(long *)_;
  }
};

struct e4848 {
  int _[16];

  __device__ __forceinline__ e4848() {
    for (int i = 0; i < 8; i++) {
      ((long *)_)[i] = 0;
    }
  }

  template <bool T = false, bool SHR = false>
  __device__ __forceinline__ e4848(const half *__restrict__ A, unsigned Y = 1) {
    if (SHR) {
      asm("cvta.to.shared %0,%0;" : "+l"(A) :);
      A += (threadIdx.x % 8 * Y * 4 + threadIdx.x / 8) * 8;
      for (int i = 0; i < 4; i++, A += Y * 32) {
        if (T)
          asm("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16{%0,%1,%2,%3},[%4]"
              ";"
              : "=r"(_[i]), "=r"(_[i + 4]), "=r"(_[i + 8]), "=r"(_[i + 12])
              : "l"(A));
        else
          asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16{%0,%1,%2,%3},[%4];"
              : "=r"(_[i * 4]), "=r"(_[i * 4 + 1]), "=r"(_[i * 4 + 2]),
                "=r"(_[i * 4 + 3])
              : "l"(A));
      }

    } else if (!T) {
      for (int i = 0; i < 2; i++) {
        ((long4 *)_)[i] = ((long4 *)A)[tup{{2u, 2u, 8u, Y * 2u}}[{
            {i, threadIdx.x % 4 / 2, threadIdx.x / 4, threadIdx.x % 2}}]];
        int t = _[i * 8 + 1];
        _[i * 8 + 1] = _[i * 8 + 4];
        _[i * 8 + 4] = _[i * 8 + 2];
        _[i * 8 + 2] = t;
        t = _[i * 8 + 3];
        _[i * 8 + 3] = _[i * 8 + 5];
        _[i * 8 + 5] = _[i * 8 + 6];
        _[i * 8 + 6] = t;
        shf(((long(*)[4])_)[i]);
      }
    }
  }

  __device__ __forceinline__ void mm(e4848 a, e4848 b) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%0,%4},{%16,%17,%20,"
        "%21},{%32,%33},{%0,%4};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%0,%4},{%18,%19,%22,"
        "%23},{%34,%35},{%0,%4};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%1,%5},{%16,%17,%20,"
        "%21},{%36,%37},{%1,%5};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%1,%5},{%18,%19,%22,"
        "%23},{%38,%39},{%1,%5};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%2,%6},{%16,%17,%20,"
        "%21},{%40,%41},{%2,%6};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%2,%6},{%18,%19,%22,"
        "%23},{%42,%43},{%2,%6};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%3,%7},{%16,%17,%20,"
        "%21},{%44,%45},{%3,%7};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%3,%7},{%18,%19,%22,"
        "%23},{%46,%47},{%3,%7};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%8,%12},{%24,%25,%"
        "128,%29},{%32,%33},{%8,%12};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%8,%12},{%26,%27,%"
        "130,%31},{%34,%35},{%8,%12};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%9,%13},{%24,%25,%"
        "128,%29},{%36,%37},{%9,%13};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%9,%13},{%26,%27,%"
        "130,%31},{%38,%39},{%9,%13};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%10,%14},{%24,%25,%"
        "128,%29},{%40,%41},{%10,%14};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%10,%14},{%26,%27,%"
        "130,%31},{%42,%43},{%10,%14};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%11,%15},{%24,%25,%"
        "128,%29},{%44,%45},{%11,%15};"
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16{%11,%15},{%26,%27,%"
        "130,%31},{%46,%47},{%11,%15};"
        : "+r"(_[0]), "+r"(_[1]), "+r"(_[2]), "+r"(_[3]), "+r"(_[4]),
          "+r"(_[5]), "+r"(_[6]), "+r"(_[7]), "+r"(_[8]), "+r"(_[9]),
          "+r"(_[10]), "+r"(_[11]), "+r"(_[12]), "+r"(_[13]), "+r"(_[14]),
          "+r"(_[15])
        : "r"(a._[0]), "r"(a._[1]), "r"(a._[2]), "r"(a._[3]), "r"(a._[4]),
          "r"(a._[5]), "r"(a._[6]), "r"(a._[7]), "r"(a._[8]), "r"(a._[9]),
          "r"(a._[10]), "r"(a._[11]), "r"(a._[12]), "r"(a._[13]), "r"(a._[14]),
          "r"(a._[15]), "r"(b._[0]), "r"(b._[1]), "r"(b._[2]), "r"(b._[3]),
          "r"(b._[4]), "r"(b._[5]), "r"(b._[6]), "r"(b._[7]), "r"(b._[8]),
          "r"(b._[9]), "r"(b._[10]), "r"(b._[11]), "r"(b._[12]), "r"(b._[13]),
          "r"(b._[14]), "r"(b._[15]));
  }

  __device__ __forceinline__ void relu() {
    half2 zero;
    *(int *)&zero = 0;
    for (int i = 0; i < 16; i++)
      ((half2 *)_)[i] = __hmax2(((half2 *)_)[i], zero);
  }

  template <bool SHR = false, bool RED = false>
  __device__ __forceinline__ void to(half *__restrict__ A, unsigned Y = 1) {
    A += tup{{2u, 2u, 8u, Y * 16u}}[{
        {0u, threadIdx.x % 4 / 2, threadIdx.x / 4, threadIdx.x % 2 * 8}}];
    shf(((long(*)[4])_)[0]);
    shf(((long(*)[4])_)[1]);
    int t = _[1];
    _[1] = _[4];
    _[4] = _[2];
    _[2] = t;
    t = _[3];
    _[3] = _[5];
    _[5] = _[6];
    _[6] = t;
    t = _[9];
    _[9] = _[12];
    _[12] = _[10];
    _[10] = t;
    t = _[11];
    _[11] = _[13];
    _[13] = _[14];
    _[14] = t;
    if (!RED) {
      ((long4 *)A)[0] = ((long4 *)_)[0];
      ((long4 *)A)[1] = ((long4 *)_)[1];
    } else if (SHR)
      asm volatile(
          "cvta.to.shared %0,%0;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0],%1;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+4],%2;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+8],%3;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+12],%4;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+16],%5;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+20],%6;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+24],%7;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+28],%8;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+32],%9;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+36],%10;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+40],%11;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+44],%12;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+48],%13;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+52],%14;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+56],%15;"
          "red.relaxed.cta.shared.add.noftz.f16x2 [%0+60],%16;" ::"l"(A),
          "r"(*_), "r"(_[1]), "r"(_[2]), "r"(_[3]), "r"(_[4]), "r"(_[5]),
          "r"(_[6]), "r"(_[7]), "r"(_[8]), "r"(_[9]), "r"(_[10]), "r"(_[11]),
          "r"(_[12]), "r"(_[13]), "r"(_[14]), "r"(_[15]));
    else
      asm volatile(
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0],%1;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+4],%2;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+8],%3;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+12],%4;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+16],%5;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+20],%6;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+24],%7;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+28],%8;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+32],%9;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+36],%10;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+40],%11;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+44],%12;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+48],%13;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+52],%14;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+56],%15;"
          "red.relaxed.gpu.global.add.noftz.f16x2 [%0+60],%16;" ::"l"(A),
          "r"(*_), "r"(_[1]), "r"(_[2]), "r"(_[3]), "r"(_[4]), "r"(_[5]),
          "r"(_[6]), "r"(_[7]), "r"(_[8]), "r"(_[9]), "r"(_[10]), "r"(_[11]),
          "r"(_[12]), "r"(_[13]), "r"(_[14]), "r"(_[15]));
  }
};

template <unsigned Z, unsigned X, unsigned Y, unsigned TX, unsigned TY,
          class T = e2828, bool TR = false, bool SHR = true>
__global__ void mm_ker(const half *__restrict__ A, const half *__restrict__ B,
                       half *__restrict__ O) {
  /**
   *[o]continuous memory access
   *[?]continuous share access
   *[x]idx order
   *[x]share load
   *[x]min/if/skip
   *[x]precise bar
   *  [x]more precise bar
   *[?]async load
   *[x]async store
   *[x]rot buf
   *[x]transpose
   *[]larger buffer
   *[x]larger atom
   *[x]bit op
   */
  unsigned th = tup{{TX, TY, 32u}}[{{threadIdx.z, threadIdx.y, threadIdx.x}}];
  unsigned TH = TX * TY * 32u;
  extern __shared__ __align__(16) half TMP[];
  auto cps = [&](unsigned z) {
    if constexpr (TR) {
      cp<false>(TMP + z % 2 * (X * TX + Y * TY) * T::len * T::len,
                A + tup{{Z, T::len, gridDim.z, X, TX,
                         T::len}}[{{z, 0u, blockIdx.z, 0u, 0u, 0u}}],
                gridDim.z * X * TX * T::len, T::len, X * TX * T::len, th, TH);
      cp<false>(TMP + (z % 2 * (X * TX + Y * TY) + X * TX) * T::len * T::len,
                B + tup{{Z, T::len, gridDim.y, Y, TY,
                         T::len}}[{{z, 0u, blockIdx.y, 0u, 0u, 0u}}],
                gridDim.y * Y * TY * T::len, T::len, Y * TY * T::len, th, TH);
    } else {
      cp<false>(TMP + z % 2 * (X * TX + Y * TY) * T::len * T::len,
                A + tup{{gridDim.z, X, TX, T::len, Z,
                         T::len}}[{{blockIdx.z, 0u, 0u, 0u, z, 0u}}],
                Z * T::len, X * TX * T::len, T::len, th, TH);
      cp<false>(TMP + (z % 2 * (X * TX + Y * TY) + X * TX) * T::len * T::len,
                B + tup{{gridDim.y, Y, TY, T::len, Z,
                         T::len}}[{{blockIdx.y, 0u, 0u, 0u, z, 0u}}],
                Z * T::len, Y * TY * T::len, T::len, th, TH);
    }
    asm("cp.async.commit_group;");
  };
  cps(0);
  T o[X][Y];
  for (unsigned z = 0; z < Z; z++) {
    if (z + 1 < Z) {
      cps(z + 1);
      asm("cp.async.wait_group 1;");
    } else {
      asm("cp.async.wait_all;");
    }
    __syncthreads();
    T a[X];
    for (unsigned x = 0; x < X; x++) {
      a[x] = {TMP + (z % 2 * (X * TX + Y * TY) + x * TX + threadIdx.z) *
                        T::len * T::len,
              1, TR, SHR};
    }
    for (unsigned y = 0; y < Y; y++) {
      T b{TMP + (z % 2 * (X * TX + Y * TY) + X * TX + y * TY + threadIdx.y) *
                    T::len * T::len,
          1, TR, SHR};
      for (unsigned x = 0; x < X; x++) {
        o[x][y].mm(a[x], b);
      }
    }
  }
  for (unsigned x = 0; x < X; x++) {
    for (unsigned y = 0; y < Y; y++) {
      o[x][y].to(O + tup{{gridDim.z, X, TX, T::len, gridDim.y, Y, TY,
                          T::len}}[{{blockIdx.z, x, threadIdx.z, 0u, blockIdx.y,
                                     y, threadIdx.y, 0u}}],
                 gridDim.y * Y * TY, false);
    }
  }
}

template <int AK, int BL, int BK>
__global__ void conv_ker(const half *__restrict__ A, const half *__restrict__ B,
                         half *__restrict__ O, int AL) {
  e2828 b[BL][BK][AK];
  for (int bl = 0; bl < BL; bl++) {
    for (int bk = 0; bk < BK; bk++) {
      for (int ak = 0; ak < AK; ak++) {
        b[bl][bk][ak] = {B + tup{{BL, BK, 16, AK, 16}}[{{bl, bk, 0, ak, 0}}],
                         AK};
      }
    }
  }
  const int OL = AL - BL + 1;
  e2828 o[BL][BK];
  for (int al = 0; al < BL; al++) {
    for (int ak = 0; ak < AK; ak++) {
      e2828 a{
          A + tup{{(int)gridDim.y, (int)blockDim.y, 16, AL, AK,
                   16}}[{{(int)blockIdx.y, (int)threadIdx.y, 0, al, ak, 0}}],
          AL * AK};
      for (int bl = 0; bl <= al; bl++) {
        for (int bk = 0; bk < BK; bk++) {
          o[al - bl][bk].mm(a, b[bl][bk][ak]);
        }
      }
    }
  }
  for (int al_ = BL; al_ < AL - BL; al_ += BL) {
    for (int i = 0; i < BL; i++) {
      int al = al_ + i;
      for (int bk = 0; bk < BK; bk++) {
        o[i % BL][bk].to(
            O + tup{{(int)gridDim.y, (int)blockDim.y, 16, OL, BK, 16}}[{
                    {(int)blockIdx.y, (int)threadIdx.y, 0, al - BL, bk, 0}}],
            OL * BK);
      }
      for (int ak = 0; ak < AK; ak++) {
        e2828 a{
            A + tup{{(int)gridDim.y, (int)blockDim.y, 16, AL, AK,
                     16}}[{{(int)blockIdx.y, (int)threadIdx.y, 0, al, ak, 0}}],
            AL * AK};
        for (int bl = 0; bl <= BL - 1; bl++) {
          for (int bk = 0; bk < BK; bk++) {
            o[(i - bl) % BL][bk].mm(a, b[bl][bk][ak]);
          }
        }
      }
    }
  }
  for (int al_ = -BL; al_ < 0; al_++) {
    int al = AL + al_;
    for (int bk = 0; bk < BK; bk++) {
      o[al_ % BL][bk].to(
          O + tup{{(int)gridDim.y, (int)blockDim.y, 16, OL, BK, 16}}[{
                  {(int)blockIdx.y, (int)threadIdx.y, 0, al - BL, bk, 0}}],
          OL * BK);
    }
    for (int ak = 0; ak < AK; ak++) {
      e2828 a{
          A + tup{{(int)gridDim.y, (int)blockDim.y, 16, AL, AK,
                   16}}[{{(int)blockIdx.y, (int)threadIdx.y, 0, al, ak, 0}}],
          AL * AK};
      for (int bl = BL + al_; bl <= BL - 1; bl++) {
        for (int bk = 0; bk < BK; bk++) {
          o[(al_ - bl) % BL][bk].mm(a, b[bl][bk][ak]);
        }
      }
    }
  }
  for (int bk = 0; bk < BK; bk++) {
    o[0][bk].to(
        O + tup{{(int)gridDim.y, (int)blockDim.y, 16, OL, BK,
                 16}}[{{(int)blockIdx.y, (int)threadIdx.y, 0, AL - BL, bk, 0}}],
        OL * BK);
  }
}

template <int AL, int AK, int BL, int BK>
__global__ void conv2_ker(const half *__restrict__ A,
                          const half *__restrict__ B, half *__restrict__ O) {
  e2828 b[BL][BL][BK][AK];
  for (int bx = 0; bx < BL; bx++) {
    for (int by = 0; by < BL; by++) {
      for (int bk = 0; bk < BK; bk++) {
        for (int ak = 0; ak < AK; ak++) {
          b[bx][by][bk][ak] = {
              B + tup{{BL, BL, BK, 16, AK, 16}}[{{bx, by, bk, 0, ak, 0}}], AK};
        }
      }
    }
  }
  const int OL = AL - BL + 1;
  const int OI = AL * AL - BL * BL;
  e2828 o[OI][BK];
  for (int al = 0; al < AL; al++) {
    if (al >= BL) {
      int ol = al - BL;
      int oi = ol * ol;
      for (int ox = 0; ox <= ol; ox++, oi++) {
        int oy = ol;
        for (int bk = 0; bk < BK; bk++) {
          o[oi % OI][bk].to(
              O + tup{{(int)gridDim.x, (int)blockDim.y, 16, OL, OL, BK, 16}}[{
                      {(int)blockIdx.x, (int)threadIdx.y, 0, ox, oy, bk, 0}}],
              OL * OL * BK);
        }
      }
      for (int oy = 0; oy < ol; oy++, oi++) {
        int ox = ol;
        for (int bk = 0; bk < BK; bk++) {
          o[oi % OI][bk].to(
              O + tup{{(int)gridDim.x, (int)blockDim.y, 16, OL, OL, BK, 16}}[{
                      {(int)blockIdx.x, (int)threadIdx.y, 0, ox, oy, bk, 0}}],
              OL * OL * BK);
        }
      }
    }
    for (int ax = 0; ax <= al; ax++) {
      int ay = al;
      for (int ak = 0; ak < AK; ak++) {
        e2828 a{A + tup{{(int)gridDim.x, (int)blockDim.y, 16, AL, AL, AK, 16}}[{
                        {(int)blockIdx.x, (int)threadIdx.y, 0, ax, ay, ak, 0}}],
                AL * AL * AK};
        for (int bx = max(BL - AL + ax, 0); bx <= min(ax, BL - 1); bx++) {
          for (int by = max(BL - AL + ay, 0); bx <= min(ay, BL - 1); by++) {
            int ox = ax - bx;
            int oy = ay - by;
            int ol = max(ox, oy);
            int oi = ol * ol + (ox <= oy ? ox : ol + oy);
            for (int bk = 0; bk < BK; bk++) {
              o[oi % OI][bk].mm(a, b[bx][by][bk][ak]);
            }
          }
        }
      }
    }
    for (int ay = 0; ay < al; ay++) {
      int ax = al;
      for (int ak = 0; ak < AK; ak++) {
        e2828 a{A + tup{{(int)gridDim.x, (int)blockDim.y, 16, AL, AL, AK, 16}}[{
                        {(int)blockIdx.x, (int)threadIdx.y, 0, ax, ay, ak, 0}}],
                AL * AL * AK};
        for (int bx = max(BL - AL + ax, 0); bx <= min(ax, BL - 1); bx++) {
          for (int by = max(BL - AL + ay, 0); bx <= min(ay, BL - 1); by++) {
            int ox = ax - bx;
            int oy = ay - by;
            int ol = max(ox, oy);
            int oi = ol * ol + (ox <= oy ? ox : 2 * ol - oy);
            for (int bk = 0; bk < BK; bk++) {
              o[oi % OI][bk].mm(a, b[bx][by][bk][ak]);
            }
          }
        }
      }
    }
  }
  int ol = AL - BL;
  int oi = ol * ol;
  for (int ox = 0; ox <= ol; ox++, oi++) {
    int oy = ol;
    for (int bk = 0; bk < BK; bk++) {
      o[oi % OI][bk].to(
          O + tup{{(int)gridDim.x, (int)blockDim.y, 16, OL, OL, BK, 16}}[{
                  {(int)blockIdx.x, (int)threadIdx.y, 0, ox, oy, bk, 0}}],
          OL * OL * BK);
    }
  }
  for (int oy = 0; oy < ol; oy++, oi++) {
    int ox = ol;
    for (int bk = 0; bk < BK; bk++) {
      o[oi % OI][bk].to(
          O + tup{{(int)gridDim.x, (int)blockDim.y, 16, OL, OL, BK, 16}}[{
                  {(int)blockIdx.x, (int)threadIdx.y, 0, ox, oy, bk, 0}}],
          OL * OL * BK);
    }
  }
}

template <int AL, int AK, int BL, int BK>
__global__ void conv2_ker_test(const half *__restrict__ A,
                               const half *__restrict__ B,
                               half *__restrict__ O) {
  const int OL = AL - BL + 1;
  const int OI = AL * AL - (AL - BL) * (AL - BL);
  e2828 b[BL][BL][BK][AK];
  for (int bx = 0; bx < BL; bx++) {
    for (int by = 0; by < BL; by++) {
      for (int bk = 0; bk < BK; bk++) {
        for (int ak = 0; ak < AK; ak++) {
          b[bx][by][bk][ak] = {
              B + tup{{BL, BL, BK, 16, AK, 16}}[{{bx, by, bk, 0, ak, 0}}], AK};
        }
      }
    }
  }
  e2828 o[OI][BK];
  for (int al = 0; al < AL; al++) {
    if (al >= BL) {
      int ol = al - BL;
      int oi = ol * ol;
      for (int ox = 0; ox <= ol; ox++, oi++) {
        int oy = ol;
        for (int bk = 0; bk < BK; bk++) {
          o[oi % OI][bk].to(
              O + tup{{(int)gridDim.x, (int)blockDim.y, 16, OL, OL, BK, 16}}[{
                      {(int)blockIdx.x, (int)threadIdx.y, 0, ox, oy, bk, 0}}],
              OL * OL * BK);
        }
      }
      for (int oy = 0; oy < ol; oy++, oi++) {
        int ox = ol;
        for (int bk = 0; bk < BK; bk++) {
          o[oi % OI][bk].to(
              O + tup{{(int)gridDim.x, (int)blockDim.y, 16, OL, OL, BK, 16}}[{
                      {(int)blockIdx.x, (int)threadIdx.y, 0, ox, oy, bk, 0}}],
              OL * OL * BK);
        }
      }
    }
    for (int ax = 0; ax <= al; ax++) {
      int ay = al;
      for (int ak = 0; ak <= AK; ak++) {
        e2828 a{A + tup{{(int)gridDim.x, (int)blockDim.y, 16, AL, AL, AK, 16}}[{
                        {(int)blockIdx.x, (int)threadIdx.y, 0, ax, ay, ak, 0}}],
                AL * AL * AK};
        for (int bx = max(BL - AL + ax, 0); bx <= min(ax, BL - 1); bx++) {
          for (int by = max(BL - AL + ay, 0); by <= min(ay, BL - 1); by++) {
            for (int bk = 0; bk < BK; bk++) {
              int ox = ax - bx;
              int oy = ay - by;
              int ol = max(ox, oy);
              int oi = ol * ol + (ox <= oy ? ox : ol + oy);
              o[oi % OI][bk].mm(a, b[bx][by][bk][ak]);
            }
          }
        }
      }
    }
    for (int ay = 0; ay < al; ay++) {
      int ax = al;
      for (int ak = 0; ak <= AK; ak++) {
        e2828 a{A + tup{{(int)gridDim.x, (int)blockDim.y, 16, AL, AL, AK, 16}}[{
                        {(int)blockIdx.x, (int)threadIdx.y, 0, ax, ay, ak, 0}}],
                AL * AL * AK};
        for (int bx = max(BL - AL + ax, 0); bx <= min(ax, BL - 1); bx++) {
          for (int by = max(BL - AL + ay, 0); by <= min(ay, BL - 1); by++) {
            for (int bk = 0; bk < BK; bk++) {
              int ox = ax - bx;
              int oy = ay - by;
              int ol = max(ox, oy);
              int oi = ol * ol + (ox <= oy ? ox : ol + oy);
              o[oi % OI][bk].mm(a, b[bx][by][bk][ak]);
            }
          }
        }
      }
    }
  }
  int ol = AL - BL;
  int oi = ol * ol;
  for (int ox = 0; ox <= ol; ox++, oi++) {
    int oy = ol;
    for (int bk = 0; bk < BK; bk++) {
      o[oi % OI][bk].to(
          O + tup{{(int)gridDim.x, (int)blockDim.y, 16, OL, OL, BK, 16}}[{
                  {(int)blockIdx.x, (int)threadIdx.y, 0, ox, oy, bk, 0}}],
          OL * OL * BK);
    }
  }
  for (int oy = 0; oy < ol; oy++, oi++) {
    int ox = ol;
    for (int bk = 0; bk < BK; bk++) {
      o[oi % OI][bk].to(
          O + tup{{(int)gridDim.x, (int)blockDim.y, 16, OL, OL, BK, 16}}[{
                  {(int)blockIdx.x, (int)threadIdx.y, 0, ox, oy, bk, 0}}],
          OL * OL * BK);
    }
  }
}

template <unsigned AK, unsigned BL, unsigned BK>
__global__ void convw_ker(const half *__restrict__ A,
                          const half *__restrict__ B, half *__restrict__ O,
                          unsigned AL) {
  unsigned OL = AL - BL + 1;
  e2828 b[BL][BK][AK];
  for (unsigned bl = 0; bl < BL; bl++) {
    for (unsigned bk = 0; bk < BK; bk++) {
      for (unsigned ak = 0; ak < AK; ak++) {
        b[bl][bk][ak] = {
            B + tup{{BL, BK, 16u, AK, 16u}}[{{bl, bk, 0u, ak, 0u}}], AK};
      }
    }
  }
  e2828 a[BL][AK];
  for (unsigned al = 0; al < BL - 1; al++) {
    for (unsigned ak = 0; ak < AK; ak++) {
      a[al][ak] = {A + tup{{gridDim.y, blockDim.y, 16u, AL, AK,
                            16u}}[{{blockIdx.y, threadIdx.y, 0u, al, ak, 0u}}],
                   AL * AK};
    }
  }
  for (unsigned ol_ = 0; ol_ < OL; ol_ += BL) {
    for (unsigned i = 0; i < BL; i++) {
      unsigned ol = ol_ + i;
      for (unsigned ak = 0; ak < BK; ak++) {
        a[(i - 1) % BL][ak] = {
            A + tup{{gridDim.y, blockDim.y, 16u, AL, AK, 16u}}[{
                    {blockIdx.y, threadIdx.y, 0u, BL - 1 + ol, ak, 0u}}],
            AL * AK};
      }
      for (unsigned bk = 0; bk < BK; bk++) {
        e2828 o;
        for (unsigned bl = 0; bl < BL; bl++) {
          for (unsigned ak = 0; ak < AK; ak++) {
            o.mm(a[(i + bl) % BL][ak], b[bl][bk][ak]);
          }
        }
        o.to(O + tup{{gridDim.y, blockDim.y, 16u, OL, BK,
                      16u}}[{{blockIdx.y, threadIdx.y, 0u, ol, bk, 0u}}],
             OL * BK);
      }
    }
  }
}

template <int AK, int BX, int BY, int BK>
__global__ void conv2w_ker(const half *__restrict__ A,
                           const half *__restrict__ B, half *__restrict__ O,
                           int AX, int AY) {
  // AY=256*3-1;
  int OX = AX - BX + 1;
  int OY = AY - BY + 1;
  e2828 b[BX][BY][BK][AK];
  for (int bx = 0; bx < BX; bx++) {
    for (int by = 0; by < BY; by++) {
      for (int bk = 0; bk < BK; bk++) {
        for (int ak = 0; ak < AK; ak++) {
          b[bx][by][bk][ak] = {
              B + tup{{BX, BK, 16, AK, 16}}[{{bx, bk, 0, ak, 0}}], AK};
        }
      }
    }
  }
  e2828 a[BX][BY][AK];
  for (int ax = 0; ax < BX - 1; ax++) {
    for (int ay = 0; ay < BY; ay++) {
      for (int ak = 0; ak < AK; ak++) {
        a[ax][ay][ak] = {
            A + tup{{(int)gridDim.y, (int)blockDim.y, 16, AX, AY, AK, 16}}[{
                    {(int)blockIdx.y, (int)threadIdx.y, 0, ax, ay, ak, 0}}],
            AX * AY * AK};
      }
    }
  }
  for (int ox_ = 0; ox_ < OX; ox_ += BX * 2) {
    for (int i = 0; i < BX * 2; i += 2) {
      for (int dir = 0; dir < 2; dir++) {
        int ox = ox_ + i + dir;
        int oy = dir ? OY - 1 : 0;
        int bx = BX - 1;
        for (int by = 0; by < BY; by++) {
          for (int ak = 0; ak < AK; ak++) {
            a[(i + dir - 1) % BX][((dir ? -1 : 0) + by) % BY][ak] = {
                A + tup{{(int)gridDim.y, (int)blockDim.y, 16, AX, AY, AK,
                         16}}[{{(int)blockIdx.y, (int)threadIdx.y, 0, bx + ox,
                                by + oy, ak, 0}}],
                AX * AY * AK};
          }
        }
        for (int bk = 0; bk < BK; bk++) {
          e2828 o;
          for (int bx = 0; bx < BX; bx++) {
            for (int by = 0; by < BY; by++) {
              for (int ak = 0; ak < AK; ak++) {
                o.mm(a[(i + dir + bx) % BX][by % BY][ak], b[bx][by][bk][ak]);
              }
            }
          }
          o.to(O + tup{{(int)gridDim.y, (int)blockDim.y, 16, OX, OY, BK, 16}}[{
                       {(int)blockIdx.y, (int)threadIdx.y, 0, ox, oy, bk, 0}}],
               OX * OY * BK);
        }
        for (int oy_ = dir ? OY - 2 : 1; dir ? oy_ >= 0 : oy_ < OY;
             oy_ += dir ? -BY : BY) {
          for (int j = 0; dir ? j > -BY : j < BY; j += dir ? -1 : 1) {
            int oy = oy_ + j;
            int by = dir ? 0 : BY - 1;
            for (int bx = 0; bx < BX; bx++) {
              for (int ak = 0; ak < AK; ak++) {
                a[(i + dir + bx) % BX][((dir ? -2 : 0) + j) % BY][ak] = {
                    A + tup{{(int)gridDim.y, (int)blockDim.y, 16, AX, AY, AK,
                             16}}[{{(int)blockIdx.y, (int)threadIdx.y, 0,
                                    bx + ox, by + oy, ak, 0}}],
                    AX * AY * AK};
              }
            }
            for (int bk = 0; bk < BK; bk++) {
              e2828 o;
              for (int bx = 0; bx < BX; bx++) {
                for (int by = 0; by < BY; by++) {
                  for (int ak = 0; ak < AK; ak++) {
                    o.mm(a[(i + dir + bx) % BX][((dir ? -2 : 1) + j + by) % BY]
                          [ak],
                         b[bx][by][bk][ak]);
                  }
                }
              }
              o.to(O + tup{{(int)gridDim.y, (int)blockDim.y, 16, OX, OY, BK,
                            16}}[{{(int)blockIdx.y, (int)threadIdx.y, 0, ox, oy,
                                   bk, 0}}],
                   OX * OY * BK);
            }
          }
        }
      }
    }
  }
}

template <int AX, int AY, int AZ, int WX, int WY, int WZ, int X = AX - WX + 1,
          int Y = AY - WY + 1, int Z = AZ - WZ + 1>
__global__ void conv3w_ker(const float (&A)[AX][AY][AZ],
                           const float (&W)[WX][WY][WZ], float (&O)[X][Y][Z]) {
  float w[WX][WY][WZ];
  for (int wx = 0; wx < WX; wx++) {
    for (int wy = 0; wy < WY; wy++) {
      for (int wz = 0; wz < WZ; wz++) {
        w[wx][wy][wz] = W[wx][wy][wz];
      }
    }
  }
  float a[WX][WY][WZ];
  for (int wx = 0; wx < WX - 1; wx++) {
    for (int wy = 0; wy < WY; wy++) {
      for (int wz = 0; wz < WZ; wz++) {
        a[wx][wy][wz] = A[wx][wy][wz];
      }
    }
  }
  for (int x_ = 0; x_ < X; x_ += 2) {
    for (int ydir = 0; ydir < 2; ydir++) {
      int x = x_ + ydir;
      int ybeg = ydir ? Y - 1 : 0;
      int yend = ydir ? 0 : Y - 1;
      int ystep = ydir ? -1 : 1;
      for (int y_ = ybeg; y_ * ystep <= yend * ystep; y_ += ystep * 2) {
        for (int zdir = 0; zdir < 2; zdir++) {
          int y = y_ + zdir * ystep;
          int zbeg = zdir ? Z - 1 : 0;
          int zend = zdir ? 0 : Z - 1;
          int zstep = zdir ? -1 : 1;
          for (int z = zbeg; z * zstep <= zend * zstep; z += zstep) {
            assert(x < X && y < Y && z < Z);
            printf("%d %d %d\n", x, y, z);
            if (z != zbeg) {
              for (int i = x; i < x + WX; i++) {
                for (int j = y; j * ystep < y * ystep + WY; j += ystep) {
                  // printf("%d %d\n",i,j);
                  // a[i % WX][j % WY][z % WZ]=A[i][j][z+WZ*zstep];
                }
              }
            } else if (y != ybeg) {
              for (int i = x; i < x + WX; i++) {
                for (int j = z; j * zstep < z * zstep + WZ; j += zstep) {
                  // a[i % WX][y % WY][j % WZ]=A[i][y+WY*ystep][j];
                }
              }
            } else {
              for (int i = y; i * ystep < y * ystep + WY; i += ystep) {
                for (int j = z; j * zstep < z * zstep + WZ; j += zstep) {
                  // a[(x+1)% WX][i % WY][j % WZ]=A[x+1][i][j];
                }
              }
            }
            float o;
            for (int wx = 0; wx < WX; wx++) {
              for (int wy = 0; wy < WY; wy++) {
                for (int wz = 0; wz < WZ; wz++) {
                  o += a[(x + wx) % WX][(y + wy) % WY][(z + wz) % WZ] *
                       w[wx][wy][wz];
                }
              }
            }
            O[x][y][z] = o;
          }
        }
      }
    }
  }
}

template <int AX_, int AY_, int WX, int WY>
__global__ void conv2w_ker_test(const half *__restrict__ A,
                                const half *__restrict__ W,
                                half *__restrict__ O, int AX, int AY) {
  AX = AX_ ? AX_ : AX;
  AY = AY_ ? AY_ : AY;
  int X = AX - WX + 1;
  int Y = AY - WY + 1;
  half w[WX][WY];
  for (int wx = 0; wx < WX; wx++) {
    for (int wy = 0; wy < WY; wy++) {
      w[wx][wy] = W[wx * WX + wy];
    }
  }
  half a[WX][WY];
  for (int wx = 0; wx < WX - 1; wx++) {
    for (int wy = 0; wy < WY; wy++) {
      a[wx][wy] = A[wx * AX + wy];
    }
  }
  for (int x_ = 0; x_ < X; x_ += WX * 2) {
    for (int wx_ = 0; wx_ < WX * 2; wx_ += 2) {
      for (int ydir = 0; ydir < 2; ydir++) {
        int x = x_ + wx_ + ydir;
        int ystep = 1 - 2 * ydir;
        auto f = [&](int y_) {
          for (int wy_ = 0; wy_ < WY; wy_++) {
            int y = (Y - 1) * ydir + (y_ + wy_) * ystep;
            if (y_ || wy_) {
              for (int i = 0; i < WX; i++) {
                a[(wx_ + ydir + i) % WX][((wy_ + 1) * ystep - ydir) % WY] =
                    A[(x + i) * AX + y + ystep];
              }
            } else {
              for (int i = 0; i < WY; i++) {
                a[(wx_ + ydir + 1) % WX][(i * ystep - ydir) % WY] =
                    A[(x + 1) * AX + y + i * ystep];
              }
            }
            half o;
            for (int wx = 0; wx < WX; wx++) {
              for (int wy = 0; wy < WY; wy++) {
                o += a[(wx_ + ydir + wx) % WX][(wy_ * ystep - ydir + wy) % WY] *
                     w[wx][wy];
              }
            }
            O[x * X + y] = o;
          }
        };
        f(0);
        for (int y_ = WY; y_ < Y; y_ += WY) {
          f(y_);
        }
      }
    }
  }
}

template <int N, int I, int H, int O>
__global__ void mlp_ker(const half *__restrict__ A, const half *__restrict__ V,
                        const half *__restrict__ W, half *__restrict__ AVW) {
  extern __shared__ __align__(16) half TMP[];
  cp(TMP, V, 0, 1, H * I * 256, threadIdx.y * 32 + threadIdx.x,
     blockDim.y * 32);
  cp(TMP + H * I * 256, W, 0, 1, O * H * 256, threadIdx.y * 32 + threadIdx.x,
     blockDim.y * 32);
  __syncthreads();
  e2828 v[H][I];
  for (int h = 0; h < H; h++) {
    for (int i = 0; i < I; i++) {
      v[h][i] = {TMP + tup{{H, 16, I, 16}}[{{h, 0, i, 0}}], I, false, true};
    }
  }
  e2828 w[O][H];
  for (int o = 0; o < O; o++) {
    for (int h = 0; h < H; h++) {
      w[o][h] = {TMP + tup{{O, 16, H, 16}}[{{o, 0, h, 0}}], H, false, true};
    }
  }
  for (int n = 0; n < N; n++) {
    e2828 a[I];
    for (int i = 0; i < I; i++) {
      a[i] = {A + tup{{0, (int)blockDim.y, N, 16, I,
                       16}}[{{(int)blockIdx.x, (int)threadIdx.y, n, 0, i, 0}}],
              I};
    }
    e2828 avw[O];
    for (int h = 0; h < H; h++) {
      e2828 av;
      for (int i = 0; i < I; i++) {
        av.mm(a[i], v[h][i]);
      }
      for (int i = 0; i < 8; i++) {
        ((short *)av._)[i] = max(((short *)av._)[i], 0);
      }
      for (int o = 0; o < O; o++) {
        avw[o].mm(av, w[o][h]);
      }
    }
    for (int o = 0; o < O; o++) {
      avw[o].to(
          AVW + tup{{0, (int)blockDim.y, N, 16, O,
                     16}}[{{(int)blockIdx.x, (int)threadIdx.y, n, 0, o, 0}}],
          O);
    }
  }
}

int main() {
  half *A;
  cudaMalloc(&A, 16 * 16 * 512);

  mmm_ker<16, 16, 16, 16><<<dim3(16, 1, 16), 32>>>(A, A, A, A);
  cudaDeviceSynchronize();
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
