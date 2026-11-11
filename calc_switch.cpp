#include "global.h"
#include <cstdio>
#include <iostream>

extern void OMP_Math();
extern void drawFrame();
extern void markDataChanged();

// ================= ИСПРАВЛЕНО ДЛЯ ЛИНКОВКИ С CUDA =================
extern "C" {
    void GPU_Malloc();
    void GPU_Math(int current_grav, float current_eps);
    void GPU_Free();
    void GPU_UploadDataToDevice();
    void GPU_DownloadDataFromDevice();
}
// ==================================================================

int button = 3;
bool prepGPU = false;

void CallNextFrame(float deltaTime)
{
    if (button != 1) {
        if (!prepGPU) {
            GPU_Malloc();
            GPU_UploadDataToDevice();
            prepGPU = true;
        }
        GPU_Math(grav, EPS);
    }
    else {
        if (prepGPU) {
            GPU_DownloadDataFromDevice();
            GPU_Free();
            prepGPU = false;
        }
        OMP_Math();
    }
    drawFrame();
}
