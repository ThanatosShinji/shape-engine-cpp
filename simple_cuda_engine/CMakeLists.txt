
include(FindCUDAToolkit)
project(cuda_engine)
enable_language(CUDA)
file(GLOB SRCS "*.cpp")
file(GLOB HEAS "*.h")
file(GLOB CUS "*.cu" "*.cuh")
link_directories(${CUDAToolkit_LIBRARY_DIR}/..)
string(APPEND CMAKE_CUDA_FLAGS " --ptxas-options=-v")

add_executable(${PROJECT_NAME} ${SRCS} ${HEAS} ${CUS})
target_link_libraries(${PROJECT_NAME} shape_engine::shape_engine CUDA::cudart_static CUDA::cublas cudnn)