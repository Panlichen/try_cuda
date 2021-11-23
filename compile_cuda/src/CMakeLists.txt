# ${EXTRA_LIBS} gpu are both values, 
# and will be joined as a semicolon-separated list
set (EXTRA_LIBS ${EXTRA_LIBS} gpu)

ADD_EXECUTABLE(project main.cpp)

target_link_libraries(project ${EXTRA_LIBS})

# target_include_directories(project PRIVATE
#     $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
#     $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
# )