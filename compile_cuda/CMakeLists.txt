# CMakeLists.txt for G4CU project

project(project)

# required cmake version
cmake_minimum_required(VERSION 2.8)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# packages
find_package(CUDA)

include_directories(project PRIVATE
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

add_subdirectory(src/cuda)
add_subdirectory(src)