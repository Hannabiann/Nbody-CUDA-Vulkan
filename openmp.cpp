#include "global.h"
#include <omp.h>
#include <cmath>

// CPU_OpenMP
void OMP_Math()
{
    const float eps2 = EPS * EPS;

    // Шаг 1: Расчет сил и обновление скоростей
#pragma omp parallel for schedule(static)
    for (int first = 0; first < N; first++) {
        float AX_M = 0.0f;
        float AY_M = 0.0f;
        float AZ_M = 0.0f;

        // Кэшируем координаты текущей частицы в регистры (как в cuda.cu)
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
            float den = (grav * POSM[next]) * (inv_dist * inv_dist * inv_dist);

            AX_M += det_X * den;
            AY_M += det_Y * den;
            AZ_M += det_Z * den;
        }

        POSR[first] += AX_M;
        POSR[first + N] += AY_M;
        POSR[first + 2 * N] += AZ_M;
    }

    // Шаг 2: Интеграция позиций
#pragma omp parallel for schedule(static)
    for (int first = 0; first < N; first++) {
        POSX[first] += POSR[first];
        POSY[first] += POSR[first + N];
        POSZ[first] += POSR[first + 2 * N];
    }
}