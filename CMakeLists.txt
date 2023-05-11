# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)

project ("shape_engine_cpp")

option(ENABLE_CPP "enable simple_inference_engine " ON)
option(ENABLE_CUDA "enable simple_cuda_engine " OFF)
option(ENABLE_SYCL "enable simple_sycl_engine " OFF)

# 包含子项目。
add_subdirectory ("shape_engine")
if(ENABLE_CPP)
add_subdirectory("simple_inference_engine")
endif()
if(ENABLE_CUDA)
add_subdirectory("simple_cuda_engine")
endif()
if(ENABLE_SYCL)
add_subdirectory("simple_sycl_engine")
endif()
