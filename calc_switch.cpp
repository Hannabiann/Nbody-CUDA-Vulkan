#include "global.h"
#include <cstdio>
#include <iostream>

extern void OMP_Math();
extern void drawFrame();
extern void markDataChanged();
extern void generateWorld();
extern void createVertexBuffer();

extern "C" {
    void GPU_Malloc();
    void GPU_Math(int current_grav, float current_eps, float dt);
    void GPU_Free();
    void GPU_UploadDataToDevice();
    void GPU_DownloadDataFromDevice();
}

int button = 3;
bool prepGPU = false;
bool isPaused = false;
bool showAxes = true;
float simSpeed = 1.0f;

void resetSimulation()
{
    generateWorld();
    if (hasCuda && prepGPU) {
        GPU_UploadDataToDevice();
    }
    createVertexBuffer();
}

void CallNextFrame(float deltaTime)
{
    if (!hasCuda) {
        button = 1;
    }

    if (!isPaused) {
        if (hasCuda && button != 1) {
            if (!prepGPU) {
                GPU_Malloc();
                GPU_UploadDataToDevice();
                prepGPU = true;
            }
            GPU_Math(grav, EPS, simSpeed);
        }
        else {
            if (prepGPU) {
                GPU_DownloadDataFromDevice();
                GPU_Free();
                prepGPU = false;
            }
            OMP_Math();
        }
    }
    drawFrame();
}