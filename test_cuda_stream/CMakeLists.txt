# CMakeLists.txt for G4CU project

project(project)

# required cmake version
cmake_minimum_required(VERSION 3.10)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CUDA_PATH /usr/local/cuda)

# packages
find_package(CUDA REQUIRED)

include_directories(project PRIVATE
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CUDA_PATH}/include>
)

add_subdirectory(src/cuda)
add_subdirectory(src)