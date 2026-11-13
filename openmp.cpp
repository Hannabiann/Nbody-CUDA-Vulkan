#include "global.h"
#include <omp.h>
#include <cmath>

void OMP_Math()
{
    const float eps2 = EPS * EPS;
    const float dt = simSpeed;
    const float grav_dt = grav * dt;

    // Шаг 1: Расчет сил и обновление скоростей с учетом шага времени dt
#pragma omp parallel for schedule(static)
    for (int first = 0; first < N; first++) {
        float AX_M = 0.0f;
        float AY_M = 0.0f;
        float AZ_M = 0.0f;

        const float fx = POSX[first];
        const float fy = POSY[first];
        const float fz = POSZ[first];

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

    // Шаг 2: Интеграция позиций с учетом dt
#pragma omp parallel for schedule(static)
    for (int first = 0; first < N; first++) {
        POSX[first] += POSR[first] * dt;
        POSY[first] += POSR[first + N] * dt;
        POSZ[first] += POSR[first + 2 * N] * dt;
    }
}