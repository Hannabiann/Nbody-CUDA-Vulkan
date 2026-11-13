#ifdef _WIN32
#include <windows.h>
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "global.h"
#include <cstdio>
#include <cmath>
#include <vector>

struct Vec3 {
    float x, y, z;
};

// Векторизованный буфер: .x = X, .y = Y, .z = Z, .w = Mass (читается за 1 инструкцию LDG.E.128)
float4* dev_POS = nullptr;
float* dev_POSR = nullptr;

// Ресурсы CUDA-Vulkan Interop
cudaExternalMemory_t extVulkanMem = nullptr;
Vec3* dev_vkVertices = nullptr;

const int BLOCK_SIZE = 128;

// ============================================================================
// CUDA ЯДРО 1: Force (128-битное векторное чтение float4)
// ============================================================================
__global__ void NbodyForceKernelOptimized(
    const float4* __restrict__ pos,
    float* __restrict__ posr,
    int N, int grav, float EPS, float dt)
{
    unsigned int first = blockIdx.x * blockDim.x + threadIdx.x;
    if (first >= N) return;

    float AX_M = 0.0f;
    float AY_M = 0.0f;
    float AZ_M = 0.0f;

    float4 myPos = pos[first];
    const float eps2 = EPS * EPS;
    const float grav_dt = grav * dt;

#pragma unroll 16
    for (unsigned int next = 0; next < N; next++) {
        float4 other = pos[next];

        float det_X = other.x - myPos.x;
        float det_Y = other.y - myPos.y;
        float det_Z = other.z - myPos.z;

        float norm = det_X * det_X + det_Y * det_Y + det_Z * det_Z;
        float ext = norm + eps2;

        float inv_dist = rsqrtf(ext);
        float den = (grav_dt * other.w) * (inv_dist * inv_dist * inv_dist);

        AX_M += det_X * den;
        AY_M += det_Y * den;
        AZ_M += det_Z * den;
    }

    posr[first] += AX_M;
    posr[first + N] += AY_M;
    posr[first + 2 * N] += AZ_M;
}

// ============================================================================
// CUDA ЯДРО 2: Tile (Векторизованная Shared Memory float4)
// ============================================================================
__global__ void NbodyTileKernelOptimized(
    const float4* __restrict__ pos,
    float* __restrict__ posr,
    int N, int grav, float EPS, float dt)
{
    unsigned int first = blockIdx.x * blockDim.x + threadIdx.x;

    float AX_M = 0.0f;
    float AY_M = 0.0f;
    float AZ_M = 0.0f;

    float4 myPos = { 0.0f, 0.0f, 0.0f, 0.0f };
    if (first < N) {
        myPos = pos[first];
    }

    // Единый векторный массив в разделяемой памяти (читается через LDS.128 broadcast)
    __shared__ float4 sh_pos[BLOCK_SIZE];

    unsigned int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const float eps2 = EPS * EPS;
    const float grav_dt = grav * dt;

    for (unsigned int tile = 0; tile < numTiles; tile++) {
        unsigned int idx = tile * BLOCK_SIZE + threadIdx.x;

        if (idx < N) {
            sh_pos[threadIdx.x] = pos[idx];
        }
        else {
            sh_pos[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        __syncthreads();

        if (first < N) {
#pragma unroll 16
            for (unsigned int j = 0; j < BLOCK_SIZE; j++) {
                float4 other = sh_pos[j];

                float det_X = other.x - myPos.x;
                float det_Y = other.y - myPos.y;
                float det_Z = other.z - myPos.z;

                float norm = det_X * det_X + det_Y * det_Y + det_Z * det_Z;
                float ext = norm + eps2;

                float inv_dist = rsqrtf(ext);
                float den = (grav_dt * other.w) * (inv_dist * inv_dist * inv_dist);

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
// CUDA ЯДРО 3: Интеграция позиций с шагом dt + Zero-Copy запись в буфер Vulkan
// ============================================================================
__global__ void IntegratePositionsKernel(
    float4* __restrict__ pos,
    const float* __restrict__ posr,
    Vec3* __restrict__ vk_verts,
    int N, float dt)
{
    unsigned int first = blockIdx.x * blockDim.x + threadIdx.x;
    if (first >= N) return;

    float4 p = pos[first];
    p.x += posr[first] * dt;
    p.y += posr[first + N] * dt;
    p.z += posr[first + 2 * N] * dt;
    pos[first] = p;

    if (vk_verts != nullptr) {
        vk_verts[first] = { p.x, p.y, p.z };
    }
}

// ============================================================================
// ВНЕШНИЙ ИНТЕРФЕЙС УПРАВЛЕНИЯ ПАМЯТЬЮ И INTEROP
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
    if (dev_POS != nullptr) return;
    cudaMalloc(&dev_POS, N * sizeof(float4));
    cudaMalloc(&dev_POSR, N * 3 * sizeof(float));
}

extern "C" void GPU_Free()
{
    if (dev_POS == nullptr) return;
    cudaFree(dev_POS);  dev_POS = nullptr;
    cudaFree(dev_POSR); dev_POSR = nullptr;
}

extern "C" void GPU_UploadDataToDevice()
{
    if (dev_POS == nullptr) return;
    std::vector<float4> hostPos(N);
    for (int i = 0; i < N; i++) {
        hostPos[i] = make_float4(POSX[i], POSY[i], POSZ[i], POSM[i]);
    }
    cudaMemcpy(dev_POS, hostPos.data(), N * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_POSR, POSR, N * 3 * sizeof(float), cudaMemcpyHostToDevice);
}

extern "C" void GPU_DownloadDataFromDevice()
{
    if (dev_POS == nullptr) return;
    std::vector<float4> hostPos(N);
    cudaMemcpy(hostPos.data(), dev_POS, N * sizeof(float4), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        POSX[i] = hostPos[i].x;
        POSY[i] = hostPos[i].y;
        POSZ[i] = hostPos[i].z;
    }
    cudaMemcpy(POSR, dev_POSR, N * 3 * sizeof(float), cudaMemcpyDeviceToHost);
}

extern "C" void GPU_Math(int current_grav, float current_eps, float dt)
{
    if (dev_POS == nullptr) return;

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (button == 2) {
        NbodyForceKernelOptimized << <dimGrid, dimBlock >> > (dev_POS, dev_POSR, N, current_grav, current_eps, dt);
    }
    else if (button == 3) {
        NbodyTileKernelOptimized << <dimGrid, dimBlock >> > (dev_POS, dev_POSR, N, current_grav, current_eps, dt);
    }

    IntegratePositionsKernel << <dimGrid, dimBlock >> > (dev_POS, dev_POSR, dev_vkVertices, N, dt);

    cudaDeviceSynchronize();
}