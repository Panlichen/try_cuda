#include<cuda/kernel.cuh>

__global__ void kernel(int *a, int *b, int *c) {
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
 
	if (threadID < N)
	{
		c[threadID] = (a[threadID] + b[threadID]) / 2;
	}
}

void kernel_wrapper(int *a, int *b, int *c, dim3 grid, dim3 block) {
	kernel <<<grid, block >>> (a, b, c);
}

void kernel_wrapper(int *a, int *b, int *c, dim3 grid, dim3 block, 
	int share_mem_size, cudaStream_t& stream) {
	kernel <<<grid, block, share_mem_size, stream >>> (a, b, c);
}