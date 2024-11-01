cmake_minimum_required(VERSION 3.10)
project(mlp CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "86")

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-arch=sm_86)

# CPU executable
add_executable(mlp_cpu src/mlp_cpu.cpp)
target_include_directories(mlp_cpu PRIVATE ./src/)

# cuDNN executable
cuda_add_executable(mlp_cudnn ./src/mlp_cudnn.cu)
set_target_properties(mlp_cudnn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(mlp_cudnn PRIVATE ./src/)
target_link_libraries(mlp_cudnn cuda cudart cudnn cublas)

# cuDNN iso executable
cuda_add_executable(mlp_cudnn_iso ./src/mlp_cudnn_iso.cu)
set_target_properties(mlp_cudnn_iso PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(mlp_cudnn_iso PRIVATE ./src/)
target_link_libraries(mlp_cudnn_iso cuda cudart cudnn cublas)

# CUDA Basic executable
cuda_add_executable(mlp_cuda_basic ./src/mlp_cuda_basic.cu)
set_target_properties(mlp_cudnn_iso PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(mlp_cuda_basic PRIVATE ./src/)
target_link_libraries(mlp_cudnn_iso cuda cudart)
