#include <mma.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <iostream>

using namespace nvcuda;

template<int N, class T>
struct tup {
    T _[N];
    __device__ __forceinline__ tup() {}
    __device__ __forceinline__ tup(T(&&x)[N]): tup(*(tup<N, T>*)x) {}
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

template<int N, class T>
__device__ __forceinline__ void cp(const T* __restrict__ A, tup<N, unsigned> AX, tup<N, unsigned> Ax, T* __restrict__ B, tup<N, unsigned> BX, tup<N, unsigned> Bx, tup<N, unsigned> X, unsigned th, unsigned TH) {
    A += AX[Ax];
    B += BX[Bx];
    th *= 16 / sizeof(T);
    TH *= 16 / sizeof(T);
    unsigned max = X.prod() - 16 / sizeof(T);
    for (unsigned i = 0; i <= max; i += TH) {
        tup x = X[min(i + th, max)];
        *(int4*)(B + BX[x]) = *(int4*)(A + AX[x]);
    }
}
template<unsigned FX, unsigned FY, unsigned TX, unsigned TY, unsigned WX, unsigned WY, unsigned WZ, class T, class U>
__global__ void mm_ker(const T *__restrict__ A, const T *__restrict__ B, U *__restrict__ O, unsigned Z) {
    __shared__ T ba[WZ][FX][TX][WX];
    __shared__ T bb[WZ][FY][TY][WY];
    wmma::fragment <wmma::matrix_a, WX, WY, WZ, T, wmma::col_major> a;
    wmma::fragment <wmma::matrix_b, WX, WY, WZ, T, wmma::row_major> b[FY];
    wmma::fragment <wmma::accumulator, WX, WY, WZ, U> o[FX][FY];
    unsigned th = tup{{TX, TY, 32u}} [ {{threadIdx.x, threadIdx.y, threadIdx.z}}];
    unsigned TH = TX * TY * 32;
    for (unsigned z = 0; z < Z; z += WZ) {
        cp(A, {{Z, FX, gridDim.x, TX, WX}}, {{z, 0u, blockIdx.x, 0u, 0u}}, ***ba, {{WZ, FX, 1u, TX, WX}}, tup{{0u, 0u, 0u, 0u, 0u}}, {{WZ, FX, 1u, TX, WX}}, th, TH);
        cp(B, {{Z, FY, gridDim.y, TY, WY}}, {{z, 0u, blockIdx.y, 0u, 0u}}, ***bb, {{WZ, FY, 1u, TY, WY}}, tup{{0u, 0u, 0u, 0u, 0u}}, {{WZ, FY, 1u, TY, WY}}, th, TH);
        __syncthreads();
        for (unsigned x = 0; x < FX; x++) {
            wmma::load_matrix_sync(a, (*ba)[x][threadIdx.x], FX * TX * WX);
            for (unsigned y = 0; y < FY; y++) {
                if (x == 0) {
                    wmma::load_matrix_sync(b[y], (*bb)[y][threadIdx.y], FY * TY * WY);
                }
                wmma::mma_sync(o[x][y], a, b[y], o[x][y]);
            }
        }
        __syncthreads();
    }
    for (unsigned x = 0; x < FX; x++) {
        for (unsigned y = 0; y < FY; y++) {
            wmma::store_matrix_sync(O + tup{{FX, gridDim.x, TX, WX, FY, gridDim.y, TY, WY}} [ {{x, blockIdx.x, threadIdx.x, 0u, y, blockIdx.y, threadIdx.y, 0u}}], o[x][y], gridDim.y * TY * FY * WY, wmma::mem_row_major);
        }
    }
}

template<unsigned FX, unsigned FY, unsigned TX, unsigned TY, unsigned WX, unsigned WY, unsigned WZ, unsigned N = 2, class T, class U>
__global__ void mm_cg_ker(const T *__restrict__ A, const T *__restrict__ B, U *__restrict__ O, unsigned Z) {
    __align__(16) __shared__ T ba[2][WZ][FX][TX][WX];
    __align__(16) __shared__ T bb[2][WZ][FY][TY][WY];
    wmma::fragment <wmma::matrix_a, WX, WY, WZ, T, wmma::col_major> a;
    wmma::fragment <wmma::matrix_b, WX, WY, WZ, T, wmma::row_major> b[FY];
    wmma::fragment <wmma::accumulator, WX, WY, WZ, U> o[FX][FY];
    auto block = cooperative_groups::this_thread_block();
    auto cp = [&](unsigned z) {
        for (unsigned i = 0; i < WZ; i++) {
            cooperative_groups::memcpy_async(block, **ba[z & 1][i], A + tup{{Z, gridDim.x, FX*TX * WX}} [ {{z * WZ + i, blockIdx.x, 0u}}], FX * TX * WX);
            cooperative_groups::memcpy_async(block, **bb[z & 1][i], B + tup{{Z, gridDim.y, FY*TY * WY}} [ {{z * WZ + i, blockIdx.y, 0u}}], FY * TY * WY);
        }
    };
    cp(0);
    for (unsigned z = 0; z < Z / WZ; z++) {
        if (z + 1 < Z / WZ) {
            cp(z + 1);
            cooperative_groups::wait_prior<WZ * 2>(block);
        } else {
            cooperative_groups::wait_prior<0>(block);
        }
        for (unsigned x = 0; x < FX; x++) {
            wmma::load_matrix_sync(a, ba[z & 1][0][x][threadIdx.x], FX * TX * WX);
            for (unsigned y = 0; y < FY; y++) {
                if (x == 0) {
                    wmma::load_matrix_sync(b[y], bb[z & 1][0][y][threadIdx.y], FY * TY * WY);
                }
                wmma::mma_sync(o[x][y], a, b[y], o[x][y]);
            }
        }
    }
    for (unsigned x = 0; x < FX; x++) {
        for (unsigned y = 0; y < FY; y++) {
            wmma::store_matrix_sync(O + tup{{FX, gridDim.x, TX, WX, FY, gridDim.y, TY, WY}} [ {{x, blockIdx.x, threadIdx.x, 0u, y, blockIdx.y, threadIdx.y, 0u}}], o[x][y], FY * gridDim.y * TY * WY, wmma::mem_row_major);
        }
    }
}

template<int N, class T>
__device__ __forceinline__ void cp_async(const T* __restrict__ G, tup<N, unsigned> GX, tup<N, unsigned> Gx, T* __restrict__ S, tup<N, unsigned> SX, unsigned th, unsigned TH) {
    G += GX[Gx];
    th *= 16 / sizeof(T);
    TH *= 16 / sizeof(T);
    unsigned max = SX.prod() - 16 / sizeof(T);
    for (unsigned _i = 0; _i <= max; _i += TH) {
        unsigned i = min(_i + th, max);
        asm("{.reg .b64 %p; cvta.to.shared.u64 %p,%0; cp.async.cg.shared.global [%p],[%1],16;}" : : "l"(S+i), "l"(G+GX[SX[i]]));
    }
}
template<unsigned FX, unsigned FY, unsigned TX, unsigned TY, unsigned WX, unsigned WY, unsigned WZ, class T, class U>
__global__ void mm_async_ker(const T *__restrict__ A, const T *__restrict__ B, U *__restrict__ O, unsigned Z) {
    __align__(16) __shared__ T ba[2][WZ][FX][TX][WX];
    __align__(16) __shared__ T bb[2][WZ][FY][TY][WY];
    wmma::fragment <wmma::matrix_a, WX, WY, WZ, T, wmma::col_major> a;
    wmma::fragment <wmma::matrix_b, WX, WY, WZ, T, wmma::row_major> b[FY];
    wmma::fragment <wmma::accumulator, WX, WY, WZ, U> o[FX][FY];
    unsigned th = tup{{TX, TY, 32u}} [ {{threadIdx.x, threadIdx.y, threadIdx.z}}];
    unsigned TH = TX * TY * 32;
    auto cp = [&](unsigned z) {
        cp_async(A, tup{{Z, FX, gridDim.x, TX, WX}}, {{z * WZ, 0u, blockIdx.x, 0u, 0u}}, ***ba[z & 1], {{WZ, FX, 1u, TX, WX}}, th, TH);
        cp_async(B, tup{{Z, FY, gridDim.y, TY, WY}}, {{z * WZ, 0u, blockIdx.y, 0u, 0u}}, ***bb[z & 1], {{WZ, FY, 1u, TY, WY}}, th, TH);
        asm("cp.async.commit_group;");
    };
    cp(0);
    for (unsigned z = 0; z < Z / WZ; z++) {
        if (z + 1 < Z / WZ) {
            cp(z + 1);
            asm("cp.async.wait_group 1;");
        } else {
            asm("cp.async.wait_all;");
        }
        __syncthreads();
        for (unsigned x = 0; x < FX; x++) {
            wmma::load_matrix_sync(a, ba[z & 1][0][x][threadIdx.x], FX * TX * WX);
            for (unsigned y = 0; y < FY; y++) {
                if (x == 0) {
                    wmma::load_matrix_sync(b[y], bb[z & 1][0][y][threadIdx.y], FY * TY * WY);
                }
                wmma::mma_sync(o[x][y], a, b[y], o[x][y]);
            }
        }
    }
    for (unsigned x = 0; x < FX; x++) {
        for (unsigned y = 0; y < FY; y++) {
            wmma::store_matrix_sync(O + tup{{FX, gridDim.x, TX, WX, FY, gridDim.y, TY, WY}} [ {{x, blockIdx.x, threadIdx.x, 0u, y, blockIdx.y, threadIdx.y, 0u}}], o[x][y], FY * gridDim.y * TY * WY, wmma::mem_row_major);
        }
    }
}

template<class T>
__device__ __forceinline__ void cp_sync(T* __restrict__ B, const T* __restrict__ A, int X, int th, int TH) {
    th *= 16;
    TH *= 16;
    unsigned max = X - 16;
    for (unsigned _i = 0; _i <= max; _i += TH) {
        unsigned i = min(_i + th, max);
        *(int4*)((char*)B + i) = *(int4*)((char*)A + i);
    }
}
// ((AQK @ b.T).masksoftmax() @ b @ VW.T).relu()
// template<int TX, int X, int Y, int K>
// __global__ void satn_ker(const half* __restrict__ Amem, const half* __restrict__ QKmem, const half* __restrict__ VWmem, half* __restrict__ Omem) {
//     __shared__ half A[TX][X][16][K][16];
//     __shared__ half QK[K][16][K][16];
//     int th = threadIdx.x * 32 + threadIdx.y;
//     int TH = TX * 32;
//     cp_sync(****A, Amem + blockIdx.x * sizeof(A) / 2, sizeof(A), th, TH);
//     cp_sync(***QK, QKmem, sizeof(QK), th, TH);
//     __syncthreads();
//     wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aqk[X][K];
//     {
//         wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> qk[K][K];
//         for (int x = 0; x < X; x++) {
//             wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[K];
//             for (int k = 0; k < K; k++) {
//                 wmma::fragment <wmma::accumulator, 16, 16, 16, half> o;
//                 wmma::fill_fragment(o, 0);
//                 for (int i = 0; i < K; i++) {
//                     if (k == 0) wmma::load_matrix_sync(a[i], A[threadIdx.x][x][0][i], K * 16);
//                     if (x == 0) wmma::load_matrix_sync(qk[k][i], QK[k][0][i], K * 16);
//                     wmma::mma_sync(o, a[i], qk[k][i], o);
//                 }
//                 wmma::store_matrix_sync(***A[threadIdx.x], o, 16, wmma::mem_row_major);
//                 wmma::load_matrix_sync(aqk[x][k], ***A[threadIdx.x], 16);
//             }
//         }
//     }
//     half saqkbb[X][K][8];
//     {
//         half sexp[X];
//         for (int l = 0; l < (blockIdx.x + 1)*TX * X / Y; l++) {
//             __shared__ half B[Y][16][K][16];
//             cp_sync(***B, Amem + l * sizeof(B) / 2, sizeof(B), th, TH);
//             __syncthreads();
//             for (int y = 0; y < Y; y++) {
//                 wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[K];
//                 for (int x = 0; x < X; x++) {
//                     half aqkb[16];
//                     wmma::fragment<wmma::accumulator, 16, 16, 16, half> o;
//                     wmma::fill_fragment(o, 0);
//                     for (int k = 0; k < K; k++) {
//                         if (x == 0) wmma::load_matrix_sync(b[k], B[y][0][k], K * 16);
//                         wmma::mma_sync(o, aqk[x][k], b[k], o);
//                     }
//                     wmma::store_matrix_sync(***A[threadIdx.x], o, K * 16, wmma::mem_row_major);
//                     cp_sync(aqkb, *A[threadIdx.x][0][threadIdx.y / 2], sizeof(aqkb), 1, 1);
//                     for (int i = 0; i < 16; i++) {
//                         half exp = sexp[x] + hexp(aqkb[i]);
//                         half scale = sexp[x] / exp;
//                         half scale_ = half(1) - scale;
//                         sexp[x] = exp;
//                         for (int k = 0; k < K; k++) {
//                             for (int j = 0; j < 8; j++) {
//                                 saqkbb[x][k][j] = saqkbb[x][k][j] * scale + B[y][i][k][threadIdx.y % 2 * 8 + j] * scale_;
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
//     wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> vw[K][K];
//     {
//         cp_sync(***QK, VWmem, sizeof(QK), th, TH);
//         __syncthreads();
//         for (int k = 0; k < K; k++) {
//             for (int i = 0; i < K; i++) {
//                 wmma::load_matrix_sync(vw[k][i], QK[k][0][i], K * 16);
//             }
//         }
//     }
//     for (int x = 0; x < X; x++) {
//         wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aqkbb[K];
//         for (int k = 0; k < K; k++) {
//             wmma::fragment <wmma::accumulator, 16, 16, 16, half> o;
//             wmma::fill_fragment(o, 0);
//             for (int i = 0; i < K; i++) {
//                 if (k == 0) {
//                     cp_sync(&A[threadIdx.x][x][threadIdx.y / 2][k][threadIdx.y % 2 * 8], saqkbb[x][i], sizeof(**saqkbb), 1, 1);
//                     wmma::load_matrix_sync(aqkbb[i], A[threadIdx.x][x][0][k], K * 16);
//                 }
//                 wmma::mma_sync(o, aqkbb[i], vw[k][i], o);
//             }
//             wmma::store_matrix_sync(A[threadIdx.x][x][0][k], o, K * 16, wmma::mem_row_major);
//         }
//     }
//     cp_sync(Omem + blockIdx.x * sizeof(A) / 2, ****A, sizeof(A), th, TH);
// }
// template<int LT, int LQ, int LKV, int KQK, int KV>
// __global__ void atn_ker(const half* __restrict__ Qmem/*...LT LQ 16 KQK 16*/, const half* __restrict__ Kmem/*NKV LKV 16 KQK 16*/, const half* __restrict__ Vmem/*NKV LKV 16 KV 16*/, half* _restrict__ QKVmem/*...LT LQ 16 KV 16*/, int NKV) {
//     __shared__ half tmp[LT][16][16];
//     half qkv[LQ][KV][8];
//     half exp[LQ];
//     wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q[LQ][KQK];
//     for (int lq = 0; lq < LQ; lq++) {
//         for (int kqk = 0; kqk < KQK; kqk++) {
//             (int4*)&tmp[threadIdx.x][0][threadIdx.y * 8] = (int4*)&Q[tup{{NKV, LT, LQ, 16, KQK, 16}} [ {{blockIdx.x, threadIdx.x, lq, threadIdx.y / 2, kqk, threadIdx.y % 2 * 8}}]];
//             wmma::load_matrix_sync(q[lq][kqk], *tmp[threadIdx.x], 16);
//         }
//     }
//     for (int nkv = 0; nkv < NKV; nkv++) {
//         half v[LKV][KV][8];
//         for (int lkv = 0; lkv < LKV; lkv++) {
//             for (int kv = 0; kv < KV; kv++) {
//                 *(int4*)v[lkv][kv] = *(int4*)&V[tup{{NKV, LKV, 16, KV, 16}} [ {{blockIdx.x, lkv, threadIdx.y / 2, kv, threadIdx.y % 2 * 8}}]];
//             }
//         }
//         for (int lkv = 0; lkv < LKV; lkv++) {
//             wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k[KQK];
//             for (int kqk = 0; kqk < KQK; kqk++) {
//                 *((int4*)tmp)[threadIdx.y] = *(int4*)&K[tup{{NKV, LKV, 16, KQK, 16}} [ {{blockIdx.x, lkv, threadIdx.y / 2, kqk, threadIdx.y % 2 * 8}}]];
//                 wmma::load_matrix_sync(k[kqk], *tmp, 16);
//             }
//             for (int lq = 0; lq < LQ; lq++) {
//                 wmma::fragment<wmma::accumulator, 16, 16, 16, half> o;
//                 wmma::fill_fragment(o, 0);
//                 for (int kqk = 0; kqk < KQK; kqk++) {
//                     wmma::mma_sync(o, q[lq][kqk], k[kqk], o);
//                 }
//                 half qk[16];
//                 wmma::store_matrix_sync(*tmp[threadIdx.x], o, 16);
//                 cp(qk, tmp[threadIdx.x][threadIdx.y / 2], 16, 0, 1);
//                 for (int i = 0; i < 16; i++) {
//                     half expnew = exp[lq] + hexp(qk[i]);
//                     half scale = exp[lq] / expnew;
//                     exp[lq] = expnew;
//                     for (int kv = 0; kv < KV; kv++) {
//                         for (int j = 0; j < 8; j++) {
//                             qkv[lq][lv][j] = qkv[lq][kv][j] * scale + v[lkv][kv][j] * (half(1) - scale);
//                         }
//                     }
//                 }
//             }
//         }
//     }
//     for (int lq = 0; lq < LQ; lq++) {
//         for (int kv = 0; kv < KV; kv++) {
//             *(int4*)&O[tup{{0, LT, LQ, 16, KV, 16}} [ {{blockIdx.x, threadIdx.x, lq, threadIdx.y / 2, kv, threadIdx.y % 2 * 8}}]] = *(int4*)qkv[lq][kv];
//         }
//     }
// }

/*X gridDim.x T 16 Z 16*/
template<unsigned X, unsigned T, unsigned Z>
__global__ void atn_ker(const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, half* __restrict__ O) {
    __shared__ __align__(16) half tmp[T][256];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q[X][Z];
    for (unsigned x = 0; x < X; x++) {
        for (unsigned z = 0; z < Z; z++) {
            wmma::load_matrix_sync(q[x][z], &Q[tup{{X, gridDim.x, T, 16u, Z, 16u}} [ {{x, blockIdx.x, threadIdx.x, 0u, z, 0u}}]], Z * 16);
        }
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> o[X][Z];
    for (unsigned y = 0; y < X * gridDim.x; y++) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> qk[X][T];
        for (unsigned z = 0; z < Z; z++) {
            *(int4*)&tmp[threadIdx.x][threadIdx.y * 8] = *(int4*)&K[tup{{X * gridDim.x, T, 16u, Z, 16u}} [ {{y, threadIdx.x, threadIdx.y / 2, z, threadIdx.y % 2 * 8}}]];
            for (unsigned t = 0; t < T; t++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k;
                wmma::load_matrix_sync(k, tmp[t], 16);
                for (unsigned x = 0; x < X; x++) {
                    wmma::mma_sync(qk[x][t], q[x][z], k, qk[x][t]);
                }
            }
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qk_[X][T];
        for (unsigned x = 0; x < X; x++) {
            for (unsigned t = 0; t < T; t++) {
                wmma::store_matrix_sync(tmp[threadIdx.x], qk[x][t], 16, wmma::mem_row_major);
                wmma::load_matrix_sync(qk_[x][t], tmp[threadIdx.x], 16);
            }
        }
        for (unsigned z = 0; z < Z; z++) {
            *(int4*)&tmp[threadIdx.x][threadIdx.y * 8] = *(int4*)&V[tup{{X * gridDim.x, T, 16u, Z, 16u}} [ {{y, threadIdx.x, threadIdx.y / 2, z, threadIdx.y % 2 * 8}}]];
            for (unsigned t = 0; t < T; t++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> v;
                wmma::load_matrix_sync(v, tmp[t], 16);
                for (unsigned x = 0; x < X; x++) {
                    wmma::mma_sync(o[x][z], qk_[x][t], v, o[x][z]);
                }
            }
        }
    }
    for (unsigned x = 0; x < X; x++) {
        for (unsigned z = 0; z < Z; z++) {
            wmma::store_matrix_sync(&O[tup{{X, gridDim.x, T, 16u, Z, 16u}} [ {{x, blockIdx.x, threadIdx.x, 0u, z, 0u}}]], o[x][z], 16, wmma::mem_row_major);
        }
    }
}
template<unsigned X, unsigned T, unsigned Z>
__global__ void atn2_ker(const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, half* __restrict__ O) {
    __shared__ __align__(16) half tmp[T][256];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q[X][Z];
    for (unsigned x = 0; x < X; x++) {
        for (unsigned z = 0; z < Z; z++) {
            wmma::load_matrix_sync(q[x][z], &Q[tup{{X, gridDim.x, T, 16u, Z, 16u}} [ {{x, blockIdx.x, threadIdx.x, 0u, z, 0u}}]], Z * 16);
        }
    }
    half sum[X][Z];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> o[X][Z];
    for (unsigned y = 0; y < X * gridDim.x * T; y++) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> qk[X];
        for (unsigned z = 0; z < Z; z++) {
            if (threadIdx.x == 0) {
                ((int4*)tmp)[threadIdx.y] = ((int4*)K)[tup{{X * gridDim.x * T, 16u, Z, 2u}} [ {{y, threadIdx.y / 2, z, threadIdx.y % 2}}]];
            }
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k;
            wmma::load_matrix_sync(k, *tmp, 16);
            for (unsigned x = 0; x < X; x++) {
                wmma::mma_sync(qk[x], q[x][z], k, qk[x]);
                for (int i = 0; i < qk[x].num_elements; i++) {
                    qk[x].x[i] = hexp(qk[x].x[i]);
                    sum[x][z] += qk[x].x[i];
                }
            }
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qk_[X];
        for (unsigned x = 0; x < X; x++) {
            wmma::store_matrix_sync(tmp[threadIdx.x], qk[x], 16, wmma::mem_row_major);
            wmma::load_matrix_sync(qk_[x], tmp[threadIdx.x], 16);
        }
        for (unsigned z = 0; z < Z; z++) {
            if (threadIdx.x == 0) {
                ((int4*)tmp)[threadIdx.y] = ((int4*)V)[tup{{X * gridDim.x * T, 16u, Z, 2u}} [ {{y, threadIdx.y / 2, z, threadIdx.y % 2}}]];
            }
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> v;
            wmma::load_matrix_sync(v, *tmp, 16);
            for (unsigned x = 0; x < X; x++) {
                wmma::mma_sync(o[x][z], qk_[x], v, o[x][z]);
            }
        }
    }
    for (unsigned x = 0; x < X; x++) {
        for (unsigned z = 0; z < Z; z++) {
            for (int i = 0; i < o[x][z].num_elements; i++) {
                o[x][z].x[i] = o[x][z].x[i] / sum[x][z];
            }
            wmma::store_matrix_sync(&O[tup{{X, gridDim.x, T, 16u, Z, 16u}} [ {{x, blockIdx.x, threadIdx.x, 0u, z, 0u}}]], o[x][z], 16, wmma::mem_row_major);
        }
    }
}

/**
 * a0a0c0a3 a1b1b2a3
 * 
 * a2c1c2d2 d0d1c3d3
 */

int main() {
    const unsigned N = 16384, Z = 1;
    half *A;
    cudaMalloc(&A,  N * 16 * Z * 16 * 2);
    // for(int i=0;i<10;i++)
    {
        const unsigned X = 2, T = 32;
        atn2_ker <X, T, Z> <<< N / X / T, dim3(T, 32) >>> (A, A, A, A);
        cudaDeviceSynchronize();
        std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
    }
    // {
    //     const unsigned X = 1, T = 12;
    //     atn_ker <X, T, Z> <<< N / X / T, dim3(T, 32) >>> (A, A, A, A);
    //     cudaDeviceSynchronize();
    //     std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
    // }
    // {
    //     const unsigned X = 1, T = 8;
    //     atn_ker <X, T, Z> <<< N / X / T, dim3(T, 32) >>> (A, A, A, A);
    //     cudaDeviceSynchronize();
    //     std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
    // }
    // {
    //     const unsigned X = 2, T = 8;
    //     atn_ker <X, T, Z> <<< N / X / T, dim3(T, 32) >>> (A, A, A, A);
    //     cudaDeviceSynchronize();
    //     std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
    // }
    // {
    //     const unsigned X = 4, T = 8;
    //     atn_ker <X, T, Z> <<< N / X / T, dim3(T, 32) >>> (A, A, A, A);
    //     cudaDeviceSynchronize();
    //     std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
    // }
    // {
    //     const unsigned X = 8, T = 8;
    //     atn_ker <X, T, Z> <<< N / X / T, dim3(T, 32) >>> (A, A, A, A);
    //     cudaDeviceSynchronize();
    //     std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
    // }

}
