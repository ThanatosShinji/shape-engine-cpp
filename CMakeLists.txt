# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)

project ("shape_engine_cpp")

# 包含子项目。
add_subdirectory ("shape_engine")

add_subdirectory("simple_inference_engine")

add_subdirectory("simple_cuda_engine")
