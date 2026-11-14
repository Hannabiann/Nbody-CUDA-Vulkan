cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;100-real;120")
endif()

project(Nbody_CUDA VERSION 4.2.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Статическая линковка рантайма CUDA (чтобы бинарник запускался даже на ПК без драйвера NVIDIA)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(Nbody_CUDA
    main.cpp
    draw.cpp
    calc_switch.cpp
    generator.cpp
    openmp.cpp
    init.cu
    cuda.cu
)

target_include_directories(Nbody_CUDA PRIVATE src)

# Флаги оптимизации для C++ (CPU) и NVCC (GPU)
target_compile_options(Nbody_CUDA PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -ffast-math -DNDEBUG -march=native -ffast-math -fopt-info-vec-optimized -funroll-loops -DNDEBUG>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -DNDEBUG --compiler-options=-O3,-march=native,-ffast-math>
)

target_link_libraries(Nbody_CUDA PRIVATE
    Vulkan::Vulkan
    glfw
    OpenMP::OpenMP_CXX
    ${CMAKE_DL_LIBS}
)