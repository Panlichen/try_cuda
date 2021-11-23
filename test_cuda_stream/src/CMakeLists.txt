project(project)

set(EXTRA_LIBS ${EXTRA_LIBS} gpu)

cuda_add_executable(non_stream cuda_non_stream.cpp)
cuda_add_executable(stream cuda_stream.cpp)

target_link_libraries(non_stream ${EXTRA_LIBS})
target_link_libraries(stream ${EXTRA_LIBS})