#include_directories ("${PROJECT_SOURCE_DIR}")

# nvcc flags -g for debug
#set(CUDA_NVCC_FLAGS -O3;-G;-g)

#set(CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20;-G;-g)
#set(CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52;-G;-g)

#cuda_add_library(gpu SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
cuda_add_library(gpu STATIC foo.cu)
# target_include_directories(project PRIVATE
#     $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
#     $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
# )
