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
    int bestDevice = 0;
    long long bestScore = -1;

    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;

        // Оценка вычислительной мощности: число SM * архитектурный вес * пиковая частота
        int clockKhz = 1000;
        cudaDeviceGetAttribute(&clockKhz, cudaDevAttrClockRate, i);
        long long score = (long long)prop.multiProcessorCount * (prop.major * 10 + prop.minor) * clockKhz;

        std::cout << "GPU #" << i << ": " << prop.name
            << " | SM " << prop.major << "." << prop.minor
            << " (" << prop.multiProcessorCount << " SMs)" << std::endl;

        if (score > bestScore) {
            bestScore = score;
            bestDevice = i;
        }
    }

    cudaSetDevice(bestDevice);

    cudaDeviceProp activeProp;
    cudaGetDeviceProperties(&activeProp, bestDevice);
    std::cout << "--> Selected CUDA GPU #" << bestDevice << ": " << activeProp.name
        << " (Using architecture sm_" << activeProp.major << activeProp.minor << ")" << std::endl;
}