#include <cuda_fp16.h>

__device__ __forceinline__ void shf_bra(int (&i)[4], int (&o)[4]) {
	int y = threadIdx.y & 3;
	*(int4*)o = *(int4*)i;
	asm("{.reg .pred %p;"
	    "setp.ne.s32 %p,%8,0;"
	    "@%p bra L1;"
	    "shfl.sync.bfly.b32 %1,%4,1,31,-1;"
	    "shfl.sync.bfly.b32 %2,%4,2,31,-1;"
	    "shfl.sync.bfly.b32 %3,%4,3,31,-1;"
	    "bra L0;"
	    "L1:"
	    "setp.ne.s32 %p,%8,1;"
	    "@%p bra L2;"
	    "shfl.sync.bfly.b32 %0,%5,1,31,-1;"
	    "shfl.sync.bfly.b32 %3,%5,2,31,-1;"
	    "shfl.sync.bfly.b32 %2,%5,3,31,-1;"
	    "bra L0;"
	    "L2:"
	    "setp.ne.s32 %p,%8,2;"
	    "@%p bra L3;"
	    "shfl.sync.bfly.b32 %3,%6,1,31,-1;"
	    "shfl.sync.bfly.b32 %0,%6,2,31,-1;"
	    "shfl.sync.bfly.b32 %1,%6,3,31,-1;"
	    "bra L0;"
	    "L3:"
	    "shfl.sync.bfly.b32 %2,%7,1,31,-1;"
	    "shfl.sync.bfly.b32 %1,%7,2,31,-1;"
	    "shfl.sync.bfly.b32 %0,%7,3,31,-1;"
	    "setp.eq.s32 %p,%8,0;"
	    "L0:}"
	    :"=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3])
	    :"r"(i[0]), "r"(i[1]), "r"(i[2]), "r"(i[3]), "r"(y));
}
__device__ __forceinline__ void shf(int (&i)[4], int (&o)[4]) {
	int y = threadIdx.y & 3;
	*(int4*)o = *(int4*)i;
	asm("{.reg .pred %p;"
	    "setp.eq.s32 %p,%8,0;"
	    "@%p shfl.sync.bfly.b32 %1,%4,1,31,-1;"
	    "@%p shfl.sync.bfly.b32 %2,%4,2,31,-1;"
	    "@%p shfl.sync.bfly.b32 %3,%4,3,31,-1;"
	    "setp.eq.s32 %p,%8,1;"
	    "@%p shfl.sync.bfly.b32 %0,%5,1,31,-1;"
	    "@%p shfl.sync.bfly.b32 %3,%5,2,31,-1;"
	    "@%p shfl.sync.bfly.b32 %2,%5,3,31,-1;"
	    "setp.eq.s32 %p,%8,2;"
	    "@%p shfl.sync.bfly.b32 %3,%6,1,31,-1;"
	    "@%p shfl.sync.bfly.b32 %0,%6,2,31,-1;"
	    "@%p shfl.sync.bfly.b32 %1,%6,3,31,-1;"
	    "setp.eq.s32 %p,%8,3;"
	    "@%p shfl.sync.bfly.b32 %2,%7,1,31,-1;"
	    "@%p shfl.sync.bfly.b32 %1,%7,2,31,-1;"
	    "@%p shfl.sync.bfly.b32 %0,%7,3,31,-1;}"
	    :"=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3])
	    :"r"(i[0]), "r"(i[1]), "r"(i[2]), "r"(i[3]), "r"(y));
}
template<class T>__device__ __forceinline__ void swap(T& a, T& b) {
	T t;
	t = a;
	a = b;
	b = t;
}
template<class N>__device__ __forceinline__ void shf_swap(int (&_)[N]) {
	if (threadIdx.y & 1) {
		swap(_[0], _[1]);
		swap(_[2], _[3]);
	}
	if (N >= 4 && threadIdx.y & 2) {
		swap(((long*)_)[0], ((long*)_)[1]);
	}
	for (int i = 1; i < N; i++) {
		_[i] = __shfl_xor_sync(0xffffffff, _[i], i);
	}
	if (N >= 4 && threadIdx.y & 1) {
		swap(_[0], _[1]);
		swap(_[2], _[3]);
	}
	if (threadIdx.y & 2) {
		swap(((long*)_)[0], ((long*)_)[1]);
	}
}
__device__ __forceinline__ void shf_switch(int (&i)[4], int (&o)[4]) {
	int y = threadIdx.y & 3;
	*(int4*)o = *(int4*)i;
	switch (y) {
	case 0:
		o[1] = __shfl_xor_sync(0xffffffff, i[0], 1);
		o[2] = __shfl_xor_sync(0xffffffff, i[0], 2);
		o[3] = __shfl_xor_sync(0xffffffff, i[0], 3);
		break;
	case 1:
		o[0] = __shfl_xor_sync(0xffffffff, i[1], 1);
		o[3] = __shfl_xor_sync(0xffffffff, i[1], 2);
		o[2] = __shfl_xor_sync(0xffffffff, i[1], 3);
		break;
	case 2:
		o[3] = __shfl_xor_sync(0xffffffff, i[2], 1);
		o[0] = __shfl_xor_sync(0xffffffff, i[2], 2);
		o[1] = __shfl_xor_sync(0xffffffff, i[2], 3);
		break;
	default:
		o[2] = __shfl_xor_sync(0xffffffff, i[3], 1);
		o[1] = __shfl_xor_sync(0xffffffff, i[3], 2);
		o[0] = __shfl_xor_sync(0xffffffff, i[3], 3);
	}
}
__device__ __forceinline__ void shf_if(int (&i)[4], int (&o)[4]) {
	int y = threadIdx.y & 3;
	*(int4*)o = *(int4*)i;
	if (y == 0) {
		o[1] = __shfl_xor_sync(0xffffffff, i[0], 1);
		o[2] = __shfl_xor_sync(0xffffffff, i[0], 2);
		o[3] = __shfl_xor_sync(0xffffffff, i[0], 3);
	} else if (y == 1) {
		o[0] = __shfl_xor_sync(0xffffffff, i[1], 1);
		o[3] = __shfl_xor_sync(0xffffffff, i[1], 2);
		o[2] = __shfl_xor_sync(0xffffffff, i[1], 3);
	} else if (y == 2) {
		o[3] = __shfl_xor_sync(0xffffffff, i[2], 1);
		o[0] = __shfl_xor_sync(0xffffffff, i[2], 2);
		o[1] = __shfl_xor_sync(0xffffffff, i[2], 3);
	} else {
		o[2] = __shfl_xor_sync(0xffffffff, i[3], 1);
		o[1] = __shfl_xor_sync(0xffffffff, i[3], 2);
		o[0] = __shfl_xor_sync(0xffffffff, i[3], 3);
	}
}
__device__ __forceinline__ void shf_for(int (&i)[4], int (&o)[4]) {
	int y = threadIdx.y & 3;
	*(int4*)o = *(int4*)i;
	for (int x = 0; x < 4; x++) {
		if (y == x) {
			for (int j = 1; j < 4; j++) {
				o[x ^ j] = __shfl_xor_sync(0xffffffff, i[x], j);
			}
		}
		break;
	}
}
__device__ __forceinline__ void shf_local(int (&i)[4], int (&o)[4]) {
	int y = threadIdx.y & 3;
	*(int4*)o = *(int4*)i;
	for (int j = 1; j < 4; j++) {
		o[y ^ j] = __shfl_xor_sync(0xffffffff, i[y], j);
	}
}
struct e2828 {
	int _[4];
	__device__ __forceinline__ void ld(const half* __restrict__ A, int Y) {
		*(int4*)_ = ((int4*)A)[(threadIdx.y << 3 & 16 | threadIdx.y >> 1 & 14) * Y | threadIdx.y & 1];
		shf_swap(_);
	}
	__device__ __forceinline__ void ld(const half* __restrict__ A, int Y, int4* __restrict__ TMP) {
		if (!threadIdx.x) {
			TMP[threadIdx.y] = ((int4*)A)[(threadIdx.y >> 1) * Y | threadIdx.y & 1];
		}
		__syncthreads();
		ld((half*)TMP, 1);
	}
	__device__ __forceinline__ void zero() {
		((long*)_)[0] = 0;
		((long*)_)[1] = 0;
	}
	__device__ __forceinline__ void st(half* __restrict__ A, int Y) {
		shf_swap(_);
		((int4*)A)[(threadIdx.y << 3 & 16 | threadIdx.y >> 1 & 14) * Y | threadIdx.y & 1] = *(int4*)_;
	}
	__device__ __forceinline__ void mm(e2828 a, e2828 b) {
		asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%2},{%4,%5,%6,%7},{%8,%9},{%0,%2};"
		    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%1,%3},{%4,%5,%6,%7},{%10,%11},{%1,%3};":
		    "=r"(_[0]), "=r"(_[1]), "=r"(_[2]), "=r"(_[3]):
		    "r"(a._[0]), "r"(a._[1]), "r"(a._[2]), "r"(a._[3]),
		    "r"(b._[0]), "r"(b._[1]), "r"(b._[2]), "r"(b._[3]));
	}
};
template<int X, int Y, int Z>__global__ void ker(const half * __restrict__ A, const half * __restrict__ B, half * __restrict__ O) {
	e2828 a, b, o;
	for (int x = 0; x < X; x++) {
		for (int y = 0; y < Y; y++) {
			o.zero();
			for (int z = 0; z < Z; z++) {
				a.ld(A + (x * 16 * Z + z) * 16, Z);
				b.ld(B + (y * 16 * Z + z) * 16, Z);
				o.mm(a, b);
			}
			o.st(O + (x * 16 * Y + y) * 16, Y);
		}
	}
}
int main() {
	half *A, *B, *O;
	cudaMalloc(&A, 64 * 16 * 64 * 16 * 2);
	cudaMalloc(&B, 64 * 16 * 64 * 16 * 2);
	cudaMalloc(&O, 64 * 16 * 64 * 16 * 2);
	ker<64, 64, 64> <<< 1, 32>>>(A, B, O);
}