cmake_minimum_required(VERSION 3.20)
project(NbodyVulkan LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

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

# Настройка флагов компиляции
target_compile_options(nbody_vulkan PRIVATE
    # Флаги для C++ (CPU): O3, Zen 4 (AVX-512), быстрая математика и отчет о векторизации
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=znver4 -ffast-math -fopt-info-vec-optimized -DNDEBUG>

    # Флаги для CUDA (GPU + Host): O3, fast math для GPU и проброс флагов в g++
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -DNDEBUG --compiler-options=-O3,-march=znver4,-ffast-math>
)

# Включение LTO (Link Time Optimization)
set_property(TARGET nbody_vulkan PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

target_link_libraries(nbody_vulkan PRIVATE
    Vulkan::Vulkan
    OpenMP::OpenMP_CXX
    CUDA::cudart
    glfw
)
