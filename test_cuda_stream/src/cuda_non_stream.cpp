#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda/kernel.cuh>
 
int main()
{
	//启动计时器
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
 
	int *host_a, *host_b, *host_c;
	int *dev_a, *dev_b, *dev_c;
 
	//在GPU上分配内存
	cudaMalloc((void**)&dev_a, FULL_DATA_SIZE * sizeof(int));
	cudaMalloc((void**)&dev_b, FULL_DATA_SIZE * sizeof(int));
	cudaMalloc((void**)&dev_c, FULL_DATA_SIZE * sizeof(int));
 
	//在CPU上分配可分页内存
	host_a = (int*)malloc(FULL_DATA_SIZE * sizeof(int));
	host_b = (int*)malloc(FULL_DATA_SIZE * sizeof(int));
	host_c = (int*)malloc(FULL_DATA_SIZE * sizeof(int));
 
	//主机上的内存赋值
	for (int i = 0; i < FULL_DATA_SIZE; i++)
	{
		host_a[i] = i;
		host_b[i] = FULL_DATA_SIZE - i;
	}
 
	//从主机到设备复制数据
	cudaMemcpy(dev_a, host_a, FULL_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, FULL_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
 
	dim3 grid(FULL_DATA_SIZE / 1024);
	dim3 block(1024);
	// kernel <<<grid, block >>> (dev_a, dev_b, dev_c);
	kernel_wrapper(dev_a, dev_b, dev_c, grid, block);
 
	//数据拷贝回主机
	cudaMemcpy(host_c, dev_c, FULL_DATA_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
 
	//计时结束
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
 
	std::cout << "消耗时间： " << elapsedTime << std::endl;
 
	//输出前10个结果
	for (int i = 0; i < 10; i++)
	{
		std::cout << host_c[i] << std::endl;
	}
 
	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);
 
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
 
	return 0;
}