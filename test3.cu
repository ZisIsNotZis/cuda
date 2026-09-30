#include<stdio.h>
__global__ void ker() {
	asm("barrier.cta.sync %0,%1;"::"r"(threadIdx.x), "r"(blockDim.y*32));
}
int main() {
	ker <<< 1, dim3(2, 2, 32)>>>();
	cudaDeviceSynchronize();
	printf("%s", cudaGetErrorString(cudaGetLastError()));
}