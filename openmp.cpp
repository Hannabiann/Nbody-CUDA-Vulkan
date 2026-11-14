#include "global.h"
#include <omp.h>
#include <cmath>
#include <algorithm>

extern float getXAngle();
extern float getYAngle();
extern float getZoom();
extern float getWarp();
extern float getRotVelX();
extern float getRotVelY();

void OMP_Math()
{
    const float eps2 = EPS * EPS;
    const float baseDt = simSpeed;

    const float PI = 3.1415926535f;
    float radX = getXAngle() * PI / 180.0f;
    float radY = getYAngle() * PI / 180.0f;
    float cx = std::cos(radX), sx = std::sin(radX);
    float cy = std::cos(radY), sy = std::sin(radY);
    float invZoom = 1.0f / std::max(getZoom(), 1.0f);
    float warp = getWarp();
    float rx = getRotVelX();
    float ry = getRotVelY();
    bool relActive = (std::fabs(warp) > 0.005f || std::fabs(rx) > 0.005f || std::fabs(ry) > 0.005f);

    // Шаг 1: Расчет сил и обновление скоростей
#pragma omp parallel for schedule(static)
    for (int first = 0; first < N; first++) {
        float AX_M = 0.0f;
        float AY_M = 0.0f;
        float AZ_M = 0.0f;

        const float fx = POSX[first];
        const float fy = POSY[first];
        const float fz = POSZ[first];

        float local_dt = baseDt;
        if (relActive) {
            float x1 = fx * cy + fz * sy;
            float z1 = -fx * sy + fz * cy;
            float y2 = fy * cx - z1 * sx;
            float nx = x1 * invZoom;
            float ny = -y2 * invZoom;
            float r2 = nx * nx + ny * ny;

            float zoomTimeFactor = 1.0f;
            if (warp > 0.0f) {
                zoomTimeFactor = std::exp(warp * (1.0f - 1.65f * r2) * 1.30f);
            }
            else if (warp < 0.0f) {
                zoomTimeFactor = std::exp(warp * 3.10f / (1.0f + 0.45f * r2));
            }
            float rotTimeFactor = std::exp((nx * rx + ny * ry) * 1.75f);
            float factor = std::clamp(zoomTimeFactor * rotTimeFactor, 0.03f, 3.80f);
            local_dt = baseDt * factor;
        }

        const float grav_dt = grav * local_dt;

        for (int next = 0; next < N; next++) {
            float det_X = POSX[next] - fx;
            float det_Y = POSY[next] - fy;
            float det_Z = POSZ[next] - fz;

            float norm = det_X * det_X + det_Y * det_Y + det_Z * det_Z;
            float ext = norm + eps2;

            float inv_dist = 1.0f / std::sqrt(ext);
            float den = (grav_dt * POSM[next]) * (inv_dist * inv_dist * inv_dist);

            AX_M += det_X * den;
            AY_M += det_Y * den;
            AZ_M += det_Z * den;
        }

        POSR[first] += AX_M;
        POSR[first + N] += AY_M;
        POSR[first + 2 * N] += AZ_M;
    }

    // Шаг 2: Интеграция позиций с локальным релятивистским временем каждой звезды
#pragma omp parallel for schedule(static)
    for (int first = 0; first < N; first++) {
        float fx = POSX[first];
        float fy = POSY[first];
        float fz = POSZ[first];

        float local_dt = baseDt;
        if (relActive) {
            float x1 = fx * cy + fz * sy;
            float z1 = -fx * sy + fz * cy;
            float y2 = fy * cx - z1 * sx;
            float nx = x1 * invZoom;
            float ny = -y2 * invZoom;
            float r2 = nx * nx + ny * ny;

            float zoomTimeFactor = 1.0f;
            if (warp > 0.0f) {
                zoomTimeFactor = std::exp(warp * (1.0f - 1.65f * r2) * 1.30f);
            }
            else if (warp < 0.0f) {
                zoomTimeFactor = std::exp(warp * 3.10f / (1.0f + 0.45f * r2));
            }
            float rotTimeFactor = std::exp((nx * rx + ny * ry) * 1.75f);
            float factor = std::clamp(zoomTimeFactor * rotTimeFactor, 0.03f, 3.80f);
            local_dt = baseDt * factor;
        }

        POSX[first] = fx + POSR[first] * local_dt;
        POSY[first] = fy + POSR[first + N] * local_dt;
        POSZ[first] = fz + POSR[first + 2 * N] * local_dt;
    }
}