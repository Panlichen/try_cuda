#define N (1024*1024)  
#define FULL_DATA_SIZE N*20  

extern "C"
void kernel_wrapper(int *a, int *b, int *c, dim3 grid, dim3 block);
void kernel_wrapper(int *a, int *b, int *c, dim3 grid, dim3 block, 
                    int share_mem_size, cudaStream_t& stream);