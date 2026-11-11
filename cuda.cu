#ifdef _WIN32
#include <windows.h>
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "global.h"
#include <cstdio>
#include <cmath>

struct Vec3 {
    float x, y, z;
};

// Девайсные указатели для хранения физических параметров на GPU
float* dev_POSX = nullptr;
float* dev_POSY = nullptr;
float* dev_POSZ = nullptr;
float* dev_POSM = nullptr;
float* dev_POSR = nullptr;

// Ресурсы CUDA-Vulkan Interop
cudaExternalMemory_t extVulkanMem = nullptr;
Vec3* dev_vkVertices = nullptr;

const int BLOCK_SIZE = 256;

// ============================================================================
// CUDA ЯДРО 1: Force (Прямой кэшированный доступ)
// ============================================================================
__global__ void NbodyForceKernelOptimized(
    const float* __restrict__ posx,
    const float* __restrict__ posy,
    const float* __restrict__ posz,
    const float* __restrict__ posm,
    float* __restrict__ posr,
    int N, int grav, float EPS)
{
    unsigned int first = blockIdx.x * blockDim.x + threadIdx.x;
    if (first >= N) return;

    float AX_M = 0.0f;
    float AY_M = 0.0f;
    float AZ_M = 0.0f;

    float fx = posx[first];
    float fy = posy[first];
    float fz = posz[first];

#pragma unroll 16
    for (unsigned int next = 0; next < N; next++) {
        float det_X = posx[next] - fx;
        float det_Y = posy[next] - fy;
        float det_Z = posz[next] - fz;

        float norm = det_X * det_X + det_Y * det_Y + det_Z * det_Z;
        float ext = norm * norm + EPS * EPS;

        float inv_dist = rsqrtf(ext);
        float den = (grav * posm[next]) * (inv_dist * inv_dist * inv_dist);

        AX_M += det_X * den;
        AY_M += det_Y * den;
        AZ_M += det_Z * den;
    }

    posr[first] += AX_M;
    posr[first + N] += AY_M;
    posr[first + 2 * N] += AZ_M;
}

// ============================================================================
// CUDA ЯДРО 2: Tile (Разделяемая память)
// ============================================================================
__global__ void NbodyTileKernelOptimized(
    const float* __restrict__ posx,
    const float* __restrict__ posy,
    const float* __restrict__ posz,
    const float* __restrict__ posm,
    float* __restrict__ posr,
    int N, int grav, float EPS)
{
    unsigned int first = blockIdx.x * blockDim.x + threadIdx.x;

    float AX_M = 0.0f;
    float AY_M = 0.0f;
    float AZ_M = 0.0f;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    if (first < N) {
        fx = posx[first];
        fy = posy[first];
        fz = posz[first];
    }

    __shared__ float sh_x[BLOCK_SIZE];
    __shared__ float sh_y[BLOCK_SIZE];
    __shared__ float sh_z[BLOCK_SIZE];
    __shared__ float sh_m[BLOCK_SIZE];

    unsigned int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (unsigned int tile = 0; tile < numTiles; tile++) {
        unsigned int idx = tile * BLOCK_SIZE + threadIdx.x;

        if (idx < N) {
            sh_x[threadIdx.x] = posx[idx];
            sh_y[threadIdx.x] = posy[idx];
            sh_z[threadIdx.x] = posz[idx];
            sh_m[threadIdx.x] = posm[idx];
        }
        else {
            sh_x[threadIdx.x] = 0.0f;
            sh_y[threadIdx.x] = 0.0f;
            sh_z[threadIdx.x] = 0.0f;
            sh_m[threadIdx.x] = 0.0f;
        }

        __syncthreads();

        if (first < N) {
#pragma unroll 16
            for (unsigned int j = 0; j < BLOCK_SIZE; j++) {
                if (tile * BLOCK_SIZE + j >= N) break;

                float det_X = sh_x[j] - fx;
                float det_Y = sh_y[j] - fy;
                float det_Z = sh_z[j] - fz;

                float norm = det_X * det_X + det_Y * det_Y + det_Z * det_Z;
                float ext = norm * norm + EPS * EPS;

                float inv_dist = rsqrtf(ext);
                float den = (grav * sh_m[j]) * (inv_dist * inv_dist * inv_dist);

                AX_M += det_X * den;
                AY_M += det_Y * den;
                AZ_M += det_Z * den;
            }
        }

        __syncthreads();
    }

    if (first < N) {
        posr[first] += AX_M;
        posr[first + N] += AY_M;
        posr[first + 2 * N] += AZ_M;
    }
}

// ============================================================================
// CUDA ЯДРО 3: Интеграция позиций + Прямая запись в буфер Vulkan (Zero-Copy)
// ============================================================================
__global__ void IntegratePositionsKernel(
    float* __restrict__ posx,
    float* __restrict__ posy,
    float* __restrict__ posz,
    const float* __restrict__ posr,
    Vec3* __restrict__ vk_verts,
    int N)
{
    unsigned int first = blockIdx.x * blockDim.x + threadIdx.x;
    if (first >= N) return;

    float nx = posx[first] + posr[first];
    float ny = posy[first] + posr[first + N];
    float nz = posz[first] + posr[first + 2 * N];

    posx[first] = nx;
    posy[first] = ny;
    posz[first] = nz;

    if (vk_verts != nullptr) {
        vk_verts[first] = { nx, ny, nz };
    }
}

// ============================================================================
// ВНЕШНИЙ ИНТЕРФЕЙС УПРАВЛЕНИЯ ПАМЯТЬЮ И КРОССПЛАТФОРМЕННЫЙ INTEROP
// ============================================================================

extern "C" void GPU_UnregisterVulkanBuffer()
{
    if (dev_vkVertices != nullptr) {
        cudaFree(dev_vkVertices);
        dev_vkVertices = nullptr;
    }
    if (extVulkanMem != nullptr) {
        cudaDestroyExternalMemory(extVulkanMem);
        extVulkanMem = nullptr;
    }
}

#ifdef _WIN32
extern "C" void GPU_RegisterVulkanBuffer(void* win32Handle, size_t allocSize, const void* initialHostData, size_t dataSize)
{
    if (win32Handle == nullptr) return;
    GPU_UnregisterVulkanBuffer();

    cudaExternalMemoryHandleDesc memHandleDesc = {};
    memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    memHandleDesc.handle.win32.handle = win32Handle;
    memHandleDesc.size = allocSize;
#else
extern "C" void GPU_RegisterVulkanBuffer(int fd, size_t allocSize, const void* initialHostData, size_t dataSize)
{
    if (fd < 0) return;
    GPU_UnregisterVulkanBuffer();

    cudaExternalMemoryHandleDesc memHandleDesc = {};
    memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    memHandleDesc.handle.fd = fd;
    memHandleDesc.size = allocSize;
#endif

    if (cudaImportExternalMemory(&extVulkanMem, &memHandleDesc) != cudaSuccess) {
        printf("Failed to import Vulkan external memory into CUDA!\n");
        return;
    }

    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = dataSize;
    bufferDesc.flags = 0;

    void* devPtr = nullptr;
    if (cudaExternalMemoryGetMappedBuffer(&devPtr, extVulkanMem, &bufferDesc) != cudaSuccess) {
        printf("Failed to get mapped CUDA buffer from Vulkan memory!\n");
        return;
    }

    dev_vkVertices = static_cast<Vec3*>(devPtr);

    if (initialHostData != nullptr && dataSize > 0) {
        cudaMemcpy(dev_vkVertices, initialHostData, dataSize, cudaMemcpyHostToDevice);
    }
}

extern "C" void GPU_UpdateVulkanBufferFromHost(const void* hostData, size_t dataSize)
{
    if (dev_vkVertices != nullptr && hostData != nullptr) {
        cudaMemcpy(dev_vkVertices, hostData, dataSize, cudaMemcpyHostToDevice);
    }
}

extern "C" void GPU_Malloc()
{
    if (dev_POSX != nullptr) return;
    cudaMalloc(&dev_POSX, N * sizeof(float));
    cudaMalloc(&dev_POSY, N * sizeof(float));
    cudaMalloc(&dev_POSZ, N * sizeof(float));
    cudaMalloc(&dev_POSM, N * sizeof(float));
    cudaMalloc(&dev_POSR, N * 3 * sizeof(float));
}

extern "C" void GPU_Free()
{
    if (dev_POSX == nullptr) return;
    cudaFree(dev_POSX); dev_POSX = nullptr;
    cudaFree(dev_POSY); dev_POSY = nullptr;
    cudaFree(dev_POSZ); dev_POSZ = nullptr;
    cudaFree(dev_POSM); dev_POSM = nullptr;
    cudaFree(dev_POSR); dev_POSR = nullptr;
}

extern "C" void GPU_UploadDataToDevice()
{
    if (dev_POSX == nullptr) return;
    cudaMemcpy(dev_POSX, POSX, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_POSY, POSY, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_POSZ, POSZ, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_POSM, POSM, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_POSR, POSR, N * 3 * sizeof(float), cudaMemcpyHostToDevice);
}

extern "C" void GPU_DownloadDataFromDevice()
{
    if (dev_POSX == nullptr) return;
    cudaMemcpy(POSX, dev_POSX, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(POSY, dev_POSY, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(POSZ, dev_POSZ, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(POSR, dev_POSR, N * 3 * sizeof(float), cudaMemcpyDeviceToHost);
}

extern "C" void GPU_Math(int current_grav, float current_eps)
{
    if (dev_POSX == nullptr) return;

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (button == 2) {
        NbodyForceKernelOptimized << <dimGrid, dimBlock >> > (dev_POSX, dev_POSY, dev_POSZ, dev_POSM, dev_POSR, N, current_grav, current_eps);
    }
    else if (button == 3) {
        NbodyTileKernelOptimized << <dimGrid, dimBlock >> > (dev_POSX, dev_POSY, dev_POSZ, dev_POSM, dev_POSR, N, current_grav, current_eps);
    }

    IntegratePositionsKernel << <dimGrid, dimBlock >> > (dev_POSX, dev_POSY, dev_POSZ, dev_POSR, dev_vkVertices, N);
    // Убрали cudaDeviceSynchronize() отсюда — ядра запускаются асинхронно за ~5 мкс!
}

extern "C" void GPU_Sync()
{
    if (button != 1 && dev_POSX != nullptr) {
        cudaDeviceSynchronize();
    }
}