#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "global.h"
#include <cstdio>
#include <cmath>
#include <vector>

extern float getXAngle();
extern float getYAngle();
extern float getZoom();
extern float getWarp();
extern float getRotVelX();
extern float getRotVelY();

struct Vec3 {
    float x, y, z;
};

// Параметры релятивистской камеры для расчета локального времени каждой звезды
struct RelCamParams {
    float cx, sx, cy, sy;
    float invZoom;
    float warp;
    float rotVelX, rotVelY;
    int active;
};

// Вычисляет локальный шаг времени звезды dt_i и сдвиг относительности одновременности
__host__ __device__ inline float getLocalRelativisticDt(
    float x, float y, float z,
    float baseDt,
    const RelCamParams& cam,
    float& outSimultaneityShift)
{
    if (!cam.active) {
        outSimultaneityShift = 0.0f;
        return baseDt;
    }

    // Координаты звезды относительно камеры
    float x1 = x * cam.cy + z * cam.sy;
    float z1 = -x * cam.sy + z * cam.cy;
    float y2 = y * cam.cx - z1 * cam.sx;
    float z2 = y * cam.sx + z1 * cam.cx;

    float nx = x1 * cam.invZoom;
    float ny = -y2 * cam.invZoom;
    float nz = z2 * cam.invZoom;
    float r2 = nx * nx + ny * ny;

    // 1. Продольный и поперечный эффект Доплера при приближении / удалении
    float zoomTimeFactor = 1.0f;
    if (cam.warp > 0.0f) {
        // Внутри конуса аберрации (центр экрана) время ускоряется, на краях — замедляется (1/gamma)
        float cone = 1.0f - 1.65f * r2;
        zoomTimeFactor = expf(cam.warp * cone * 1.30f);
    }
    else if (cam.warp < 0.0f) {
        // При уходе от галактики на 99.99999% c фотоны еле догоняют камеру — время замирает
        float w = -cam.warp;
        float lag = 1.0f / (1.0f + 0.45f * r2);
        zoomTimeFactor = expf(-w * 3.10f * lag);
    }

    // 2. Вращение камеры: встречный край галактики ускоряется, уходящий край замирает
    float rotProj = nx * cam.rotVelX + ny * cam.rotVelY;
    float rotTimeFactor = expf(rotProj * 1.75f);

    float factor = zoomTimeFactor * rotTimeFactor;
    if (factor < 0.03f) factor = 0.03f;
    if (factor > 3.80f) factor = 3.80f;

    // 3. Относительность одновременности и запаздывание света по глубине (nz)
    outSimultaneityShift = (factor - 1.0f) * 2.5f - nz * cam.warp * 3.5f;

    return baseDt * factor;
}

float4* dev_POS = nullptr;
float* dev_POSR = nullptr;

cudaExternalMemory_t extVulkanMem = nullptr;
Vec3* dev_vkVertices = nullptr;

const int BLOCK_SIZE = 128;

__global__ void NbodyForceKernelOptimized(
    const float4* __restrict__ pos,
    float* __restrict__ posr,
    int N, int grav, float EPS, float dt,
    RelCamParams cam)
{
    unsigned int first = blockIdx.x * blockDim.x + threadIdx.x;
    if (first >= N) return;

    float AX_M = 0.0f, AY_M = 0.0f, AZ_M = 0.0f;
    float4 myPos = pos[first];
    const float eps2 = EPS * EPS;

    float dummyShift = 0.0f;
    float local_dt = getLocalRelativisticDt(myPos.x, myPos.y, myPos.z, dt, cam, dummyShift);
    const float grav_dt = grav * local_dt;

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

__global__ void NbodyTileKernelOptimized(
    const float4* __restrict__ pos,
    float* __restrict__ posr,
    int N, int grav, float EPS, float dt,
    RelCamParams cam)
{
    unsigned int first = blockIdx.x * blockDim.x + threadIdx.x;

    float AX_M = 0.0f, AY_M = 0.0f, AZ_M = 0.0f;
    float4 myPos = { 0.0f, 0.0f, 0.0f, 0.0f };
    float local_dt = dt;

    if (first < N) {
        myPos = pos[first];
        float dummyShift = 0.0f;
        local_dt = getLocalRelativisticDt(myPos.x, myPos.y, myPos.z, dt, cam, dummyShift);
    }

    __shared__ float4 sh_pos[BLOCK_SIZE];

    unsigned int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const float eps2 = EPS * EPS;
    const float grav_dt = grav * local_dt;

    for (unsigned int tile = 0; tile < numTiles; tile++) {
        unsigned int idx = tile * BLOCK_SIZE + threadIdx.x;
        sh_pos[threadIdx.x] = (idx < N) ? pos[idx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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

__global__ void IntegratePositionsKernel(
    float4* __restrict__ pos,
    const float* __restrict__ posr,
    Vec3* __restrict__ vk_verts,
    int N, float dt,
    RelCamParams cam)
{
    unsigned int first = blockIdx.x * blockDim.x + threadIdx.x;
    if (first >= N) return;

    float4 p = pos[first];
    float vx = posr[first];
    float vy = posr[first + N];
    float vz = posr[first + 2 * N];

    float simultaneityShift = 0.0f;
    float local_dt = getLocalRelativisticDt(p.x, p.y, p.z, dt, cam, simultaneityShift);

    p.x += vx * local_dt;
    p.y += vy * local_dt;
    p.z += vz * local_dt;
    pos[first] = p;

    if (vk_verts != nullptr) {
        // В буфер рендеринга записываем положение с учетом релятивистской неодновременности и задержки света
        vk_verts[first] = {
            p.x + vx * simultaneityShift,
            p.y + vy * simultaneityShift,
            p.z + vz * simultaneityShift
        };
    }
}

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

    if (cudaImportExternalMemory(&extVulkanMem, &memHandleDesc) != cudaSuccess) return;

    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = dataSize;
    bufferDesc.flags = 0;

    void* devPtr = nullptr;
    if (cudaExternalMemoryGetMappedBuffer(&devPtr, extVulkanMem, &bufferDesc) != cudaSuccess) return;

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

    const float PI = 3.1415926535f;
    float radX = getXAngle() * PI / 180.0f;
    float radY = getYAngle() * PI / 180.0f;
    float warp = getWarp();
    float rx = getRotVelX();
    float ry = getRotVelY();

    RelCamParams cam;
    cam.cx = std::cos(radX);
    cam.sx = std::sin(radX);
    cam.cy = std::cos(radY);
    cam.sy = std::sin(radY);
    cam.invZoom = 1.0f / std::max(getZoom(), 1.0f);
    cam.warp = warp;
    cam.rotVelX = rx;
    cam.rotVelY = ry;
    cam.active = (std::fabs(warp) > 0.005f || std::fabs(rx) > 0.005f || std::fabs(ry) > 0.005f) ? 1 : 0;

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (button == 2) {
        NbodyForceKernelOptimized << <dimGrid, dimBlock >> > (dev_POS, dev_POSR, N, current_grav, current_eps, dt, cam);
    }
    else if (button == 3) {
        NbodyTileKernelOptimized << <dimGrid, dimBlock >> > (dev_POS, dev_POSR, N, current_grav, current_eps, dt, cam);
    }

    IntegratePositionsKernel << <dimGrid, dimBlock >> > (dev_POS, dev_POSR, dev_vkVertices, N, dt, cam);

    cudaDeviceSynchronize();
}