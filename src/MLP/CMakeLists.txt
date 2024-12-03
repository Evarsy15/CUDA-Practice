# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project name and C++ standard
project(MLP VERSION 1.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Allocate Source & Header
set(SRCS main.cpp mlp.cpp mlp_cpu.cpp mlp_cuda.cu)
set(HDRS mlp.h mlp_cpu.h mlp_cuda.h common.h)

# Register Source and Header files
add_executable(${PROJECT_NAME} ${SRCS} ${HDRS})
