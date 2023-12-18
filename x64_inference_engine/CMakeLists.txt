project(x64_inference_engine LANGUAGES CXX)

add_subdirectory(lib)
file(GLOB HERs "*.h")
file(GLOB SRCs "*.cpp")
add_executable(x64_inference_engine ${SRCs})
target_link_libraries(${PROJECT_NAME} shape_engine::shape_engine jblas::jblas)
