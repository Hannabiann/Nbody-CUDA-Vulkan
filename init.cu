#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

//CUDA device initialize
int nDevices;

void cudaInitDevices() 
{
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) 
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		std::cout << "GPU # " << i;
		std::cout << " - " << prop.name;
	}
}