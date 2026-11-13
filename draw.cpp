#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#endif

#include "global.h"
#include <vulkan/vulkan.h>
#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#endif

#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

extern VkDevice device;
extern VkPhysicalDevice physicalDevice;
extern GLFWwindow* window;
extern size_t g_vertexCount;

#ifdef _WIN32
extern PFN_vkGetMemoryWin32HandleKHR fpGetMemoryWin32HandleKHR;
extern "C" void GPU_RegisterVulkanBuffer(void* win32Handle, size_t allocSize, const void* initialHostData, size_t dataSize);
#else
extern PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;
extern "C" void GPU_RegisterVulkanBuffer(int fd, size_t allocSize, const void* initialHostData, size_t dataSize);
#endif

extern "C" {
    void GPU_UnregisterVulkanBuffer();
    void GPU_UpdateVulkanBufferFromHost(const void* hostData, size_t dataSize);
}

VkBuffer vertexBuffer = VK_NULL_HANDLE;
VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
VkDeviceSize vertexBufferAllocSize = 0;
void* pMappedPosData = nullptr; // Используется, когда CUDA недоступна

VkBuffer colorBuffer = VK_NULL_HANDLE;
VkDeviceMemory colorBufferMemory = VK_NULL_HANDLE;

static float x_angle = 0.0f, y_angle = 0.0f;
static float view_zoom = 6000.0f;
static float fps = 0.0f, time_f = 0.0f, timebase = 0.0f;
static int frames = 0;

struct Vec3 { float x, y, z; };

static std::vector<Vec3> cpuPositions;
static std::vector<Vec3> cpuColors;
static bool bufferCreated = false;
static bool dataChanged = true;

bool checkDataChanged() { return dataChanged; }
void clearDataChangedFlag() { dataChanged = false; }
float getXAngle() { return x_angle; }
float getYAngle() { return y_angle; }
float getZoom() { return view_zoom; }
VkBuffer getColorBuffer() { return colorBuffer; }
void* getCpuVerticesPointer() { return cpuPositions.empty() ? nullptr : cpuPositions.data(); }

#ifdef _WIN32
HANDLE getVulkanBufferMemoryHandle() {
    if (fpGetMemoryWin32HandleKHR == nullptr) return nullptr;

    VkMemoryGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.memory = vertexBufferMemory;
    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;

    HANDLE handle = nullptr;
    if (fpGetMemoryWin32HandleKHR(device, &handleInfo, &handle) != VK_SUCCESS) {
        return nullptr;
    }
    return handle;
}
#else
int getVulkanBufferMemoryFd() {
    if (fpGetMemoryFdKHR == nullptr) return -1;

    VkMemoryGetFdInfoKHR fdInfo{};
    fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fdInfo.memory = vertexBufferMemory;
    fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd = -1;
    if (fpGetMemoryFdKHR(device, &fdInfo, &fd) != VK_SUCCESS) {
        return -1;
    }
    return fd;
}
#endif

void MousePressedMove(GLFWwindow* window, double xpos, double ypos) {
    static double lastX = 0, lastY = 0;
    static bool firstMouse = true;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
        float sensitivity = 0.2f;
        x_angle += (float)((ypos - lastY) * sensitivity);
        y_angle += (float)((xpos - lastX) * sensitivity);
        if (x_angle > 89.0f) x_angle = 89.0f;
        if (x_angle < -89.0f) x_angle = -89.0f;
        lastX = xpos; lastY = ypos;
    }
    else { firstMouse = true; }
}

void MouseScroll(GLFWwindow* window, double xoffset, double yoffset) {
    if (yoffset < 0)
        view_zoom *= 1.15f;
    else if (yoffset > 0)
        view_zoom /= 1.15f;

    if (view_zoom < 500.0f) view_zoom = 500.0f;
}

void keyboardPressed(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        switch (key) {
        case GLFW_KEY_1:
            button = 1;
            dataChanged = true;
            break;
        case GLFW_KEY_2:
            if (hasCuda) {
                button = 2;
                dataChanged = true;
            }
            else {
                std::cout << "[CUDA Unavailable] Running in CPU (OpenMP) mode.\n";
            }
            break;
        case GLFW_KEY_3:
            if (hasCuda) {
                button = 3;
                dataChanged = true;
            }
            else {
                std::cout << "[CUDA Unavailable] Running in CPU (OpenMP) mode.\n";
            }
            break;
        }
    }
}

VkDeviceSize helperCreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory, bool isExternal) {
#ifdef _WIN32
    const VkExternalMemoryHandleTypeFlags extHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
#else
    const VkExternalMemoryHandleTypeFlags extHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkExternalMemoryBufferCreateInfo externalMemoryBufferCreateInfo{};
    externalMemoryBufferCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    externalMemoryBufferCreateInfo.handleTypes = extHandleType;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = isExternal ? &externalMemoryBufferCreateInfo : nullptr;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkExportMemoryAllocateInfoKHR exportMemoryAllocateInfoKHR{};
    exportMemoryAllocateInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    exportMemoryAllocateInfoKHR.handleTypes = extHandleType;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = isExternal ? &exportMemoryAllocateInfoKHR : nullptr;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = UINT32_MAX;

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            allocInfo.memoryTypeIndex = i;
            break;
        }
    }
    vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
    vkBindBufferMemory(device, buffer, bufferMemory, 0);

    return memRequirements.size;
}

void createVertexBuffer() {
    cpuPositions.clear(); cpuColors.clear();
    cpuPositions.reserve(N + 200); cpuColors.reserve(N + 200);

    int N_large = (N * 3) / 4;
    int bulgeLarge = N_large / 7;
    int bulgeSmall = (N - N_large) / 7;

    for (int i = 0; i < N; i++) {
        cpuPositions.push_back({ POSX[i], POSY[i], POSZ[i] });

        if (i < N_large) {
            if (i < bulgeLarge)
                cpuColors.push_back({ 1.0f, 0.85f, 0.45f });
            else
                cpuColors.push_back({ 0.55f, 0.80f, 1.0f });
        }
        else {
            if (i - N_large < bulgeSmall)
                cpuColors.push_back({ 1.0f, 0.70f, 0.35f });
            else
                cpuColors.push_back({ 0.75f, 0.95f, 0.90f });
        }
    }

    const float AXIS_LENGTH = 3800.0f; const float AXIS_OFFSET = 2000.0f;
    const float MARKER_SIZE = 100.0f; const float MARKER_STEP = 500.0f;

    cpuPositions.push_back({ -AXIS_LENGTH, -AXIS_OFFSET, -AXIS_OFFSET }); cpuColors.push_back({ 1.0f, 0.0f, 0.0f });
    cpuPositions.push_back({ AXIS_LENGTH, -AXIS_OFFSET, -AXIS_OFFSET });  cpuColors.push_back({ 1.0f, 0.0f, 0.0f });
    cpuPositions.push_back({ -AXIS_OFFSET, -AXIS_LENGTH, -AXIS_OFFSET }); cpuColors.push_back({ 0.0f, 1.0f, 0.0f });
    cpuPositions.push_back({ -AXIS_OFFSET, AXIS_LENGTH, -AXIS_OFFSET });  cpuColors.push_back({ 0.0f, 1.0f, 0.0f });
    cpuPositions.push_back({ -AXIS_OFFSET, -AXIS_OFFSET, -AXIS_LENGTH }); cpuColors.push_back({ 0.0f, 0.0f, 1.0f });
    cpuPositions.push_back({ -AXIS_OFFSET, -AXIS_OFFSET, AXIS_LENGTH });  cpuColors.push_back({ 0.0f, 0.0f, 1.0f });

    for (float pos = -AXIS_LENGTH + MARKER_STEP; pos < AXIS_LENGTH; pos += MARKER_STEP) {
        if (fabs(pos) < 0.1f) continue;
        cpuPositions.push_back({ pos, -AXIS_OFFSET - MARKER_SIZE, -AXIS_OFFSET }); cpuColors.push_back({ 1.0f, 0.0f, 0.0f });
        cpuPositions.push_back({ pos, -AXIS_OFFSET + MARKER_SIZE, -AXIS_OFFSET }); cpuColors.push_back({ 1.0f, 0.0f, 0.0f });
        cpuPositions.push_back({ -AXIS_OFFSET - MARKER_SIZE, pos, -AXIS_OFFSET }); cpuColors.push_back({ 0.0f, 1.0f, 0.0f });
        cpuPositions.push_back({ -AXIS_OFFSET + MARKER_SIZE, pos, -AXIS_OFFSET }); cpuColors.push_back({ 0.0f, 1.0f, 0.0f });
        cpuPositions.push_back({ -AXIS_OFFSET - MARKER_SIZE, -AXIS_OFFSET, pos }); cpuColors.push_back({ 0.0f, 0.0f, 1.0f });
        cpuPositions.push_back({ -AXIS_OFFSET + MARKER_SIZE, -AXIS_OFFSET, pos }); cpuColors.push_back({ 0.0f, 0.0f, 1.0f });
    }

    g_vertexCount = cpuPositions.size();

    if (vertexBuffer != VK_NULL_HANDLE) {
        if (hasCuda) {
            GPU_UnregisterVulkanBuffer();
        }
        else if (pMappedPosData != nullptr) {
            vkUnmapMemory(device, vertexBufferMemory);
            pMappedPosData = nullptr;
        }
        vkDestroyBuffer(device, vertexBuffer, nullptr); vkFreeMemory(device, vertexBufferMemory, nullptr);
        vkDestroyBuffer(device, colorBuffer, nullptr); vkFreeMemory(device, colorBufferMemory, nullptr);
    }

    VkDeviceSize posSize = sizeof(Vec3) * g_vertexCount;
    VkDeviceSize colorSize = sizeof(Vec3) * g_vertexCount;

    if (hasCuda) {
        // Режим с поддержкой CUDA: буфер в DEVICE_LOCAL памяти и экспорт в CUDA
        vertexBufferAllocSize = helperCreateBuffer(
            posSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            vertexBuffer,
            vertexBufferMemory,
            true
        );

#ifdef _WIN32
        HANDLE win32Handle = getVulkanBufferMemoryHandle();
        if (win32Handle != nullptr) {
            GPU_RegisterVulkanBuffer(win32Handle, (size_t)vertexBufferAllocSize, cpuPositions.data(), (size_t)posSize);
            CloseHandle(win32Handle);
        }
#else
        int fd = getVulkanBufferMemoryFd();
        if (fd != -1) {
            GPU_RegisterVulkanBuffer(fd, (size_t)vertexBufferAllocSize, cpuPositions.data(), (size_t)posSize);
        }
#endif
    }
    else {
        // Режим без CUDA: обычный HOST_VISIBLE буфер для прямого доступа с CPU
        vertexBufferAllocSize = helperCreateBuffer(
            posSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vertexBuffer,
            vertexBufferMemory,
            false
        );
        vkMapMemory(device, vertexBufferMemory, 0, posSize, 0, &pMappedPosData);
        std::memcpy(pMappedPosData, cpuPositions.data(), (size_t)posSize);
    }

    helperCreateBuffer(
        colorSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        colorBuffer,
        colorBufferMemory,
        false
    );
    void* pColorData = nullptr;
    vkMapMemory(device, colorBufferMemory, 0, colorSize, 0, &pColorData);
    std::memcpy(pColorData, cpuColors.data(), (size_t)colorSize);
    vkUnmapMemory(device, colorBufferMemory);

    bufferCreated = true;
    dataChanged = true;
}

void updateVertexBuffer() {
    if (!bufferCreated) { createVertexBuffer(); return; }

    // Если работает CUDA (кнопки 2 или 3), вершины обновляются напрямую в VRAM
    if (hasCuda && button != 1) return;

    if (hasCuda) {
        // CPU OpenMP на компьютере с CUDA
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            cpuPositions[i].x = POSX[i];
            cpuPositions[i].y = POSY[i];
            cpuPositions[i].z = POSZ[i];
        }
        GPU_UpdateVulkanBufferFromHost(cpuPositions.data(), N * sizeof(Vec3));
    }
    else if (pMappedPosData != nullptr) {
        // CPU OpenMP на компьютере без CUDA (прямая запись в память Vulkan)
        Vec3* posData = static_cast<Vec3*>(pMappedPosData);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            posData[i].x = POSX[i];
            posData[i].y = POSY[i];
            posData[i].z = POSZ[i];
        }
    }
}

void markDataChanged() { dataChanged = true; }

void drawFrame() {
    frames++;
    static auto lastTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
    lastTime = currentTime; time_f += deltaTime * 1000.0f;

    if (time_f - timebase >= 1000.0f) {
        fps = frames * 1000.0f / (time_f - timebase); timebase = time_f; frames = 0;
        const char* name_method = (button == 2) ? "CUDA Force" : (button == 3) ? "CUDA Tile" : "OpenMP";
        char res[100];
        std::snprintf(res, sizeof(res), "Nbody V4.1 | Method: %s | FPS: %.3f", name_method, fps);
        glfwSetWindowTitle(window, res);
    }
    updateVertexBuffer();
}

void cleanupVertexBuffer() {
    if (hasCuda) {
        GPU_UnregisterVulkanBuffer();
    }
    else if (pMappedPosData != nullptr && vertexBufferMemory != VK_NULL_HANDLE) {
        vkUnmapMemory(device, vertexBufferMemory);
        pMappedPosData = nullptr;
    }

    if (vertexBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
        vertexBuffer = VK_NULL_HANDLE;
    }

    if (colorBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, colorBuffer, nullptr);
        vkFreeMemory(device, colorBufferMemory, nullptr);
        colorBuffer = VK_NULL_HANDLE;
    }
}