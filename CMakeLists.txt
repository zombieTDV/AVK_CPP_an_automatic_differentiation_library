# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(program)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find Eigen package (if installed)
find_package(Eigen3 REQUIRED)

# List all source files
set(SOURCES
    main.cpp
    src/tensor/TensorBase.cpp
    src/tensor/Tensor0D.cpp
    src/tensor/Tensor1D.cpp
    src/tensor/Tensor2D.cpp
    src/tensor/Tensor3D.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}        # For main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include # For tensor headers
    ${EIGEN3_INCLUDE_DIR}              # For Eigen headers
) 