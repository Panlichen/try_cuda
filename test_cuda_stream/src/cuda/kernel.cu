#define N (1024*1024)  
#define FULL_DATA_SIZE N*20  

extern "C"
__global__ void kernel(int* a, int *b, int*c)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
 
	if (threadID < N)
	{
		c[threadID] = (a[threadID] + b[threadID]) / 2;
	}
}