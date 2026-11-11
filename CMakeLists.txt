cmake_minimum_required(VERSION 3.20)
project(NbodyVulkan LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Архитектура Ada Lovelace (RTX 4090 = sm_89)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(Vulkan REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(nbody_vulkan
    main.cpp
    draw.cpp
    calc_switch.cpp
    generator.cpp
    openmp.cpp
    cuda.cu
    init.cu
)

target_link_libraries(nbody_vulkan PRIVATE
    Vulkan::Vulkan
    OpenMP::OpenMP_CXX
    CUDA::cudart
    glfw
)