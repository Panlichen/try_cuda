project(project)

cmake_minimum_required(VERSION 2.8)

add_subdirectory(cuda)

set(EXTRA_LIBS ${EXTRA_LIBS} gpu)

add_executable(non_stream cuda_non_stream.cpp)
add_executable(steam cuda_stream.cpp)

target_link_libraries(non_stream, ${EXTRA_LIBS})
target_link_libraries(stream, ${EXTRA_LIBS})