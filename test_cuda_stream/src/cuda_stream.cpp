#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda/kernel.cuh>
 
int main()
{
	//获取设备属性
	cudaDeviceProp prop;
	int deviceID;
	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&prop, deviceID);
 
	//检查设备是否支持重叠功能
	if (!prop.deviceOverlap)
	{
		printf("No device will handle overlaps. so no speed up from stream.\n");
		return 0;
	}
 
	//启动计时器
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
 
	//创建一个CUDA流
	cudaStream_t stream;
	cudaStreamCreate(&stream);
 
	int *host_a, *host_b, *host_c;
	int *dev_a, *dev_b, *dev_c;
 
	//在GPU上分配内存
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));
 
	//在CPU上分配页锁定内存
	cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
 
	//主机上的内存赋值
	for (int i = 0; i < FULL_DATA_SIZE; i++)
	{
		host_a[i] = i;
		host_b[i] = FULL_DATA_SIZE - i;
	}
 
	for (int i = 0; i < FULL_DATA_SIZE; i += N)
	{
		cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
 
		dim3 grid(FULL_DATA_SIZE / 1024);
		dim3 block(1024);
		kernel_wrapper(dev_a, dev_b, dev_c, grid, block, 0, stream);
 
		cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream);
	}
 
	// wait until gpu execution finish
	cudaStreamSynchronize(stream);
 
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
 
	std::cout << "消耗时间： " << elapsedTime << std::endl;
 
	//输出前10个结果
	for (int i = 0; i < 10; i++)
	{
		std::cout << host_c[i] << std::endl;
	}
 
	// free stream and mem
	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);
 
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
 
	cudaStreamDestroy(stream);
	return 0;
}