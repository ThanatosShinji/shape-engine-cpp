project(simple_inference_engine LANGUAGES CXX)

add_executable(simple_inference_engine "simple_inference_engine.cpp" "simple_inference_engine.h")
target_link_libraries(${PROJECT_NAME} shape_engine::shape_engine)
