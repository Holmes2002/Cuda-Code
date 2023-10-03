cmake_minimum_required(VERSION 3.17)
project(main)
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
add_executable(main main.cpp inference.cu)
target_compile_features(main PUBLIC cxx_std_11)
set(CMAKE_CUDA_ARCHITECTURES 75)  # Replace XX with the desired CUDA architecture
target_link_libraries(main PRIVATE ${TORCH_LIBRARIES})