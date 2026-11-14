#ifdef _WIN32
#define NOMINMAX
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
#include <algorithm>

extern VkDevice device;
extern VkPhysicalDevice physicalDevice;
extern GLFWwindow* window;
extern size_t g_vertexCount;
extern void resetSimulation();
extern void toggleFullscreen();
extern void getStarColor(int idx, float& r, float& g, float& b);

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
void* pMappedPosData = nullptr;

VkBuffer colorBuffer = VK_NULL_HANDLE;
VkDeviceMemory colorBufferMemory = VK_NULL_HANDLE;

static float x_angle = 0.0f, y_angle = 0.0f;
static float view_zoom = 6000.0f;
static float target_zoom = 6000.0f;

// Плавные (интерполированные) и целевые значения релятивистской скорости
static double warp_drive = 0.0, warp_target = 0.0;
static double rot_vel_x = 0.0, rot_target_x = 0.0;
static double rot_vel_y = 0.0, rot_target_y = 0.0;
static double current_beta = 0.0;
static double current_gamma = 1.0;

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
float getWarp() { return (float)warp_drive; }
float getGamma() { return (float)current_gamma; }
float getRotVelX() { return (float)rot_vel_x; }
float getRotVelY() { return (float)rot_vel_y; }
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

static void recalculateRelativity() {
    double rotMag = std::sqrt(rot_vel_x * rot_vel_x + rot_vel_y * rot_vel_y);
    double driveMag = std::clamp(std::max(std::fabs(warp_drive), rotMag), 0.0, 1.0);

    if (driveMag < 0.005) {
        current_beta = 0.0;
        current_gamma = 1.0;
        return;
    }

    // Нелинейное отображение [0..1] -> beta вплоть до 0.9999999 (99.99999% скорости света)
    // При driveMag = 1.0: 1 - beta = 10^-7 => beta = 0.9999999, gamma = 2236.07
    double logOneMinusBeta = -7.0 * std::pow(driveMag, 1.35);
    double oneMinusBeta = std::pow(10.0, logOneMinusBeta);
    current_beta = 1.0 - oneMinusBeta;
    if (current_beta > 0.9999999) current_beta = 0.9999999;

    current_gamma = 1.0 / std::sqrt(1.0 - current_beta * current_beta);
}

void updateWindowTitle() {
    const char* name_method = (button == 2) ? "CUDA Force" : (button == 3) ? "CUDA Tile" : "OpenMP";
    const char* state_str = isPaused ? " [PAUSED]" : "";
    char res[200];
    if (current_beta > 0.05) {
        double peakCmbTemp = 2.725 * std::sqrt((1.0 + current_beta) / (1.0 - current_beta));
        const char* dirSign = (warp_drive < -0.01) ? "-" : "+";
        std::snprintf(res, sizeof(res),
            APP_TITLE " | %s | v = %s%.5f%% c (gamma: %.0fx, CMB: %.0fK) | FPS: %.1f%s",
            name_method, dirSign, current_beta * 100.0, current_gamma, peakCmbTemp, fps, state_str);
    }
    else {
        std::snprintf(res, sizeof(res), APP_TITLE " | Method: %s | Speed: %.2fx | FPS: %.3f%s",
            name_method, simSpeed, fps, state_str);
    }
    glfwSetWindowTitle(window, res);
}

void MousePressedMove(GLFWwindow* window, double xpos, double ypos) {
    static double lastX = 0, lastY = 0;
    static bool firstMouse = true;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
        double dx = xpos - lastX;
        double dy = ypos - lastY;

        float sensitivity = 0.2f;
        x_angle += (float)(dy * sensitivity);
        y_angle += (float)(dx * sensitivity);
        if (x_angle > 89.0f) x_angle = 89.0f;
        if (x_angle < -89.0f) x_angle = -89.0f;

        // Накапливаем целевой импульс вращения (плавно догоняется в drawFrame)
        rot_target_x = std::clamp(rot_target_x + dx * 0.010, -1.0, 1.0);
        rot_target_y = std::clamp(rot_target_y - dy * 0.010, -1.0, 1.0);

        lastX = xpos; lastY = ypos;
    }
    else { firstMouse = true; }
}

void MouseScroll(GLFWwindow* window, double xoffset, double yoffset) {
    // Плавный шаг зума без верхнего ограничения на отдаление
    float zoomFactor = std::pow(1.12f, (float)(-yoffset));
    target_zoom *= zoomFactor;
    if (target_zoom < 50.0f) target_zoom = 50.0f;

    // При смене направления прокрутки мягко гасим противоположный импульс
    if ((yoffset > 0 && warp_target < 0.0) || (yoffset < 0 && warp_target > 0.0)) {
        warp_target *= 0.25;
    }
    // Один щелчок дает мягкий импульс (0.20), а быстрая прокрутка разгоняет до 1.0 (99.99999% c)
    warp_target = std::clamp(warp_target + yoffset * 0.20, -1.0, 1.0);
}

void keyboardPressed(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) {
            float step = (simSpeed < 1.0f) ? 0.10f : 0.25f;
            simSpeed = std::round((simSpeed + step) * 100.0f) / 100.0f;
            if (simSpeed > 10.0f) simSpeed = 10.0f;
            updateWindowTitle();
        }
        else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) {
            float step = (simSpeed > 1.0f) ? 0.25f : 0.10f;
            simSpeed = std::round((simSpeed - step) * 100.0f) / 100.0f;
            if (simSpeed < 0.10f) simSpeed = 0.10f;
            updateWindowTitle();
        }
    }

    if (action == GLFW_PRESS) {
        int newButton = button;
        switch (key) {
        case GLFW_KEY_1:
            newButton = 1;
            break;
        case GLFW_KEY_2:
            if (hasCuda) newButton = 2;
            else std::cout << "[CUDA Unavailable] Running in CPU (OpenMP) mode.\n";
            break;
        case GLFW_KEY_3:
            if (hasCuda) newButton = 3;
            else std::cout << "[CUDA Unavailable] Running in CPU (OpenMP) mode.\n";
            break;
        case GLFW_KEY_0:
        case GLFW_KEY_KP_0:
            simSpeed = 1.0f;
            updateWindowTitle();
            break;
        case GLFW_KEY_SPACE:
            isPaused = !isPaused;
            updateWindowTitle();
            break;
        case GLFW_KEY_R:
            vkDeviceWaitIdle(device);
            resetSimulation();
            break;
        case GLFW_KEY_G:
            showAxes = !showAxes;
            break;
        case GLFW_KEY_F11:
        case GLFW_KEY_F:
            toggleFullscreen();
            break;
        }

        if (newButton != button) {
            vkDeviceWaitIdle(device);
            button = newButton;
            dataChanged = true;
            updateWindowTitle();
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
    cpuPositions.reserve(N + 800); cpuColors.reserve(N + 800);

    for (int i = 0; i < N; i++) {
        cpuPositions.push_back({ POSX[i], POSY[i], POSZ[i] });
        float r, g, b;
        getStarColor(i, r, g, b);
        cpuColors.push_back({ r, g, b });
    }

    const float AXIS_LENGTH = 3800.0f; const float AXIS_OFFSET = 2000.0f;
    const float MARKER_SIZE = 100.0f;  const float MARKER_STEP = 500.0f;
    const int   AXIS_SEGS = 80; // Разбиваем оси на 80 сегментов, чтобы они плавно изгибались при сжатии/расширении пространства

    for (int s = 0; s < AXIS_SEGS; s++) {
        float t0 = -AXIS_LENGTH + (2.0f * AXIS_LENGTH * s) / AXIS_SEGS;
        float t1 = -AXIS_LENGTH + (2.0f * AXIS_LENGTH * (s + 1)) / AXIS_SEGS;

        // Ось X (красная)
        cpuPositions.push_back({ t0, -AXIS_OFFSET, -AXIS_OFFSET }); cpuColors.push_back({ 1.0f, 0.0f, 0.0f });
        cpuPositions.push_back({ t1, -AXIS_OFFSET, -AXIS_OFFSET }); cpuColors.push_back({ 1.0f, 0.0f, 0.0f });
        // Ось Y (зеленая)
        cpuPositions.push_back({ -AXIS_OFFSET, t0, -AXIS_OFFSET }); cpuColors.push_back({ 0.0f, 1.0f, 0.0f });
        cpuPositions.push_back({ -AXIS_OFFSET, t1, -AXIS_OFFSET }); cpuColors.push_back({ 0.0f, 1.0f, 0.0f });
        // Ось Z (синяя)
        cpuPositions.push_back({ -AXIS_OFFSET, -AXIS_OFFSET, t0 }); cpuColors.push_back({ 0.0f, 0.0f, 1.0f });
        cpuPositions.push_back({ -AXIS_OFFSET, -AXIS_OFFSET, t1 }); cpuColors.push_back({ 0.0f, 0.0f, 1.0f });
    }

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
    if (isPaused) return;
    if (hasCuda && button != 1) return;

    if (hasCuda) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            cpuPositions[i].x = POSX[i];
            cpuPositions[i].y = POSY[i];
            cpuPositions[i].z = POSZ[i];
        }
        GPU_UpdateVulkanBufferFromHost(cpuPositions.data(), N * sizeof(Vec3));
    }
    else if (pMappedPosData != nullptr) {
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
    lastTime = currentTime;
    time_f += deltaTime * 1000.0f;

    float dt = std::min(deltaTime, 0.1f);

    // Плавная логарифмическая интерполяция зума (одинаково гладкая на любых дистанциях)
    view_zoom += (target_zoom - view_zoom) * std::min(1.0f, dt * 12.0f);

    // Плавный разгон текущих скоростей к целевым (убирает рывок в момент щелчка колеса)
    double smoothRate = std::min(1.0, (double)dt * 10.0);
    warp_drive += (warp_target - warp_drive) * smoothRate;
    rot_vel_x += (rot_target_x - rot_vel_x) * smoothRate;
    rot_vel_y += (rot_target_y - rot_vel_y) * smoothRate;

    // Экспоненциальное затухание целевого импульса после остановки колеса/мыши
    warp_target *= std::exp(-dt * 2.2);
    rot_target_x *= std::exp(-dt * 4.0);
    rot_target_y *= std::exp(-dt * 4.0);

    if (std::fabs(warp_drive) < 0.002 && std::fabs(warp_target) < 0.002) warp_drive = 0.0;
    if (std::fabs(rot_vel_x) < 0.002 && std::fabs(rot_target_x) < 0.002) rot_vel_x = 0.0;
    if (std::fabs(rot_vel_y) < 0.002 && std::fabs(rot_target_y) < 0.002) rot_vel_y = 0.0;

    recalculateRelativity();

    if (time_f - timebase >= 1000.0f) {
        fps = frames * 1000.0f / (time_f - timebase); timebase = time_f; frames = 0;
        updateWindowTitle();
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