/*template<class T, unsigned N, unsigned X, unsigned K> __global__ void atn_ker(const half * __restrict__ A, const half * __restrict__ QK, const half * __restrict__ VW, half * __restrict__ AQKBBVW) {
	extern __shared__ __align__(16) half TMP[];
	T aqk[X][K];
	for (unsigned i = 0; i < K; i++) {
		T qk[K];
		for (unsigned x = 0; x < X; x++) {
			T a;
			a.ld(A + tup{{X, gridDim.zz, blockDim.zz, N, K, N}} [ {{x, blockIdx.zz, threadIdx.zz, 0u, i, 0u}}], K, NULL);
			for (unsigned k = 0; k < K; k++) {
				if (x == 0) {
					qk[k].ld(QK + tup{{K, N, K, N}} [ {{k, 0u, i, 0u}}], K, TMP);
				}
				if (i == 0) {
					aqk[x][k].zero();
				}
				aqk[x][k].mm(a, qk[k]);
			}
		}
	}
	T vw[K][K];
	for (unsigned k = 0; k < K; k++) {
		for (unsigned i = 0; i < K; i++) {
			vw[k][i].ld(VW + tup{{K, N, K, N}} [ {{k, 0u, i, 0u}}], K, TMP);
		}
	}
	T aqkbbvw[X][K];
	for (unsigned y = 0; y < X * gridDim.zz * blockDim.zz; y++) {
		T b[K];
		for (unsigned x = 0; x < X; x++) {
			T aqkb;
			aqkb.zero();
			for (unsigned i = 0; i < K; i++) {
				if (x == 0) {
					b[i].ld(A + tup{{X*gridDim.zz * blockDim.zz, N, K, N}} [ {{y, 0u, i, 0u}}], K, TMP);
				}
				aqkb.mm(aqk[x][i], b[i]);
			}
			for (unsigned i = 0; i < K; i++) {
				T aqkbb;
				aqkbb.zero();
				aqkbb.mm(aqkb, b[i]);
				for (unsigned k = 0; k < K; k++) {
					aqkbbvw[x][k].mm(aqkbb, vw[k][i]);
				}
			}
		}
	}
	for (unsigned x = 0; x < X; x++) {
		for (unsigned k = 0; k < K; k++) {
			aqkbbvw[x][k].st(AQKBBVW + tup{{X, gridDim.zz, blockDim.zz, N, K, N}} [ {{x, blockIdx.zz, threadIdx.zz, 0u, k, 0u}}], K);
		}
	}
}*/
template<unsigned NX_, unsigned NY_, unsigned NZ_, unsigned X = 4, unsigned Y = 4, unsigned Z = 1, unsigned TX = 4, unsigned TY = 4, int MOD = 4, int BND = 256, int SHARE = 0>__global__ void __launch_bounds__(BND) mm_ker(const half*__restrict__ A_, const half*__restrict__ B_, half*__restrict__ O_, unsigned NX, unsigned NY, unsigned NZ) {
	extern __shared__ __align__(16) int4 TMP[][32];
	if (NX_)NX = NX_;
	if (NY_)NY = NY_;
	if (NZ_)NZ = NZ_;
	if constexpr(MOD / 2 == 0) {
		for (unsigned ny = 0; ny < ny; ny++) {
			for (unsigned nz = 0; nz < NZ; nz++) {
				e2828 b[Y][Z];
				for (unsigned y = 0; y < Y; y++) {
					for (unsigned z = 0; z < Z; z++) {
						b[y][z] = {B(0), NZ * Z};
					}
				}
				for (unsigned nx = 0; nx < nx; nx++) {
					for (unsigned x = 0; x < X; x++) {
						if constexpr(MOD % 2 == 0) {
							e2828 o[Y];
							for (unsigned z = 0; z < Z; z++) {
								e2828 a{A(0), NZ * Z};
								for (unsigned y = 0; y < Y; y++) {
									if (z == 0)o[y] = {O(0), NY*Y*gridDim.y * blockDim.y};
									o[y].mm(b[y][z], a);
									if (z == Z - 1)o[y].to(O(0), NY * Y * gridDim.y * blockDim.y);
								}
							}
						} else {
							e2828 a[Z];
							for (unsigned y = 0; y < Y; y++) {
								e2828 o{O(0), NY*Y*gridDim.y * blockDim.y};
								for (unsigned z = 0; z < Z; z++) {
									if (y == 0)a[z] = {A(0), NZ * Z};
									o.mm(b[y][z], a[z]);
								}
								o.to(O(0), NY * Y * gridDim.y * blockDim.y);
							}
						}
					}
				}
			}
		}
	} else if constexpr(MOD / 2 == 1) {
		for (unsigned nx = 0; nx < NX; nx++) {
			for (unsigned nz = 0; nz < NZ; nz++) {
				e2828 a[X][Z];
				for (unsigned x = 0; x < X; x++) {
					for (unsigned z = 0; z < Z; z++) {
						a[x][z] = {A(0), NZ * Z};
					}
				}
				for (unsigned ny = 0; ny < NY; ny++) {
					for (unsigned y = 0; y < Y; y++) {
						if constexpr(MOD % 2 == 0) {
							e2828 o[X];
							for (unsigned z = 0; z < Z; z++) {
								e2828 b{B(0), NZ * Z};
								for (unsigned x = 0; x < X; x++) {
									if (z == 0)o[x] = {O(0), NY * Y * gridDim.y * blockDim.y};
									o[x].mm(a[x][z], b);
									if (z == Z - 1)o[x].to(O(0), NY * Y * gridDim.y * blockDim.y);
								}
							}
						} else {
							e2828 b[Z];
							for (unsigned x = 0; x < X; x++) {
								e2828 o{O(0), NY * Y * gridDim.y * blockDim.y};
								for (unsigned z = 0; z < Z; z++) {
									if (x == 0)b[z] = {B(0), NZ * Z};
									o.mm(a[x][z], b[z]);
								}
								o.to(O(0), NY * Y * gridDim.y * blockDim.y);
							}
						}
					}
				}
			}
		}
	} else {
		if constexpr(SHARE == 2) {
			const unsigned nx = 0, ny = 0, nz = 0, z = 0;
			for (unsigned i = 0; i < X; i += TY) {
				unsigned x = min(i + threadIdx.y, X - 1);
				cp_async_cg_shared_global(TMP[(z % 2 * X + x) * TX + threadIdx.z][threadIdx.x], *(int4*)A(threadIdx.x));
			}
			for (unsigned i = 0; i < Y; i += TX) {
				unsigned y = min(i + threadIdx.z, Y - 1);
				cp_async_cg_shared_global(TMP[2 * X * TX + (z % 2 * Y + y) * TY + threadIdx.y][threadIdx.x], *(int4*)B(threadIdx.x));
			}
			cp_async_commit_group();
		}
		for (unsigned nx = 0; nx < NX; nx++) {
			for (unsigned ny = 0; ny < NY; ny++) {
				e2828 o[X][Y];
				for (unsigned nz = 0; nz < NZ; nz++) {
					for (unsigned z = 0; z < Z; z++) {
						if constexpr(SHARE == 2) {
							if (z + 1 < Z || nz + 1 < NZ || ny + 1 < NY || nx + 1 < NX) {
								unsigned tmp[3];
								if (z + 1 < Z) {
									z++;
								} else {
									tmp[0] = z;
									z = 0;
									if (nz + 1 < NZ) {
										nz++;
									} else {
										tmp[1] = nz;
										nz = 0;
										if (ny + 1 < NY) {
											ny++;
										} else {
											tmp[2] = ny;
											ny = 0;
											nx++;
										}
									}
								}
								for (unsigned i = 0; i < X; i += TY) {
									unsigned x = min(i + threadIdx.y, X - 1);
									cp_async_cg_shared_global(TMP[(z % 2 * X + x) * TX + threadIdx.z][threadIdx.x], *(int4*)A(threadIdx.x));
								}
								for (unsigned i = 0; i < Y; i += TX) {
									unsigned y = min(i + threadIdx.z, Y - 1);
									cp_async_cg_shared_global(TMP[2 * X * TX + (z % 2 * Y + y) * TY + threadIdx.y][threadIdx.x], *(int4*)B(threadIdx.x));
								}
								cp_async_commit_group();
								cp_async_wait_group_1();
								__syncthreads();
								if (z > 0) {
									z--;
								} else {
									z = tmp[0];
									if (nz > 0) {
										nz--;
									} else {
										nz = tmp[1];
										if (ny > 0) {
											ny--;
										} else {
											ny = tmp[2];
											nx--;
										}
									}
								}
							} else {
								cp_async_wait_all();
							}
							if constexpr(MOD % 2 == 0) {
								e2828 a[X];
								for (unsigned y = 0; y < Y; y++) {
									e2828 b{(half*)TMP[2 * X * TX + (z % 2 * Y + y) * TY + threadIdx.y], 1};
									for (unsigned x = 0; x < X; x++) {
										if (y == 0)a[x] = {(half*)TMP[(z % 2 * X + x) * TX + threadIdx.z], 1};
										o[x][y].mm(a[x], b);
									}
								}
							} else {
								e2828 b[Y];
								for (unsigned x = 0; x < X; x++) {
									e2828 a{(half*)TMP[(z % 2 * X + x) * TX + threadIdx.z], 1};
									for (unsigned y = 0; y < Y; y++) {
										if (x == 0)b[y] = {(half*)TMP[2 * X * TX + (z % 2 * Y + y) * TY + threadIdx.y], 1};
										o[x][y].mm(a, b[y]);
									}
								}
							}
						} else if constexpr(SHARE == 1) {
							for (unsigned i = 0; i < X; i += TY) {
								unsigned x = min(i + threadIdx.y, X - 1);
								TMP[x * TX + threadIdx.z][threadIdx.x] = *(int4*)A(threadIdx.x);
							}
							for (unsigned i = 0; i < Y; i += TX) {
								unsigned y = min(i + threadIdx.z, Y - 1);
								TMP[X * TX + y * TY + threadIdx.y][threadIdx.x] = *(int4*)B(threadIdx.x);
							}
							__syncthreads();
							if constexpr(MOD % 2 == 0) {
								e2828 a[X];
								for (unsigned y = 0; y < Y; y++) {
									e2828 b{(half*)TMP[X * TX + y * TY + threadIdx.y], 1};
									for (unsigned x = 0; x < X; x++) {
										if (y == 0)a[x] = {(half*)TMP[x * TX + threadIdx.z], 1};
										o[x][y].mm(a[x], b);
									}
								}
							} else {
								e2828 b[Y];
								for (unsigned x = 0; x < X; x++) {
									e2828 a{(half*)TMP[x * TX + threadIdx.z], 1};
									for (unsigned y = 0; y < Y; y++) {
										if (x == 0)b[y] = {(half*)TMP[X * TX + y * TY + threadIdx.y], 1};
										o[x][y].mm(a, b[y]);
									}
								}
							}
						} else if constexpr(MOD % 2 == 0) {
							e2828 a[X];
							for (unsigned y = 0; y < Y; y++) {
								e2828 b{B(0), NZ * Z};
								for (unsigned x = 0; x < X; x++) {
									if (y == 0)a[x] = {A(0), NZ * Z};
									o[x][y].mm(a[x], b);
								}
							}
						} else {
							e2828 b[Y];
							for (unsigned x = 0; x < X; x++) {
								e2828 a{A(0), NZ * Z};
								for (unsigned y = 0; y < Y; y++) {
									if (x == 0)b[y] = {B(0), NZ * Z};
									o[x][y].mm(a, b[y]);
								}
							}
						}
					}
				}
				for (unsigned x = 0; x < X; x++) {
					for (unsigned y = 0; y < Y; y++) {
						o[x][y].to(O(0), NY * Y * gridDim.y * blockDim.y);
					}
				}
			}
		}
	}
}