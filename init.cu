#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "global.h"
#include <iostream>

int nDevices = 0;
bool hasCuda = false;

void cudaInitDevices()
{
    cudaError_t err = cudaGetDeviceCount(&nDevices);
    if (err != cudaSuccess || nDevices <= 0) {
        hasCuda = false;
        button = 1; // Автоматически переключаем на CPU (OpenMP)
        std::cout << "CUDA GPU not found or unavailable! Automatically running on CPU (OpenMP)." << std::endl;
        return;
    }

    hasCuda = true;
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "GPU # " << i << " - " << prop.name << std::endl;
    }
}