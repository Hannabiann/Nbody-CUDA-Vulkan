#include "global.h"
#include <omp.h>
#include <cmath>

// CPU_OpenMP
void OMP_Math()
{
    // Шаг 1: Расчет сил и обновление скоростей (ПО СТАРЫМ КООРДИНАТАМ)
#pragma omp parallel
    {
        float AX_M, AY_M, AZ_M, det_X, det_Y, det_Z, den, norm, ext;

#pragma omp for schedule(static)
        for (int first = 0; first < N; first++) {
            AX_M = 0.0f;
            AY_M = 0.0f;
            AZ_M = 0.0f;

            // Внутренний цикл считает по фиксированным массивам POSX, POSY, POSZ
#pragma omp simd reduction(+:AX_M,AY_M,AZ_M)
            for (int next = 0; next < N; next++) {
                det_X = POSX[next] - POSX[first];
                det_Y = POSY[next] - POSY[first];
                det_Z = POSZ[next] - POSZ[first];

                norm = det_X * det_X + det_Y * det_Y + det_Z * det_Z;
                ext = norm * norm + EPS * EPS;
                den = grav * POSM[next] / (sqrtf(ext) * ext);

                AX_M += det_X * den;
                AY_M += det_Y * den;
                AZ_M += det_Z * den;
            }

            // Накапливаем ускорение в компоненты скорости (аналог VELX[ia] += ax на GPU)
            POSR[first] += AX_M; // Скорость по X
            POSR[first + N] += AY_M; // Скорость по Y
            POSR[first + 2 * N] += AZ_M; // Скорость по Z
        }
    }

    // Шаг 2: Интеграция позиций (В отдельном цикле, как IntegratePositionsKernel на GPU)
    // Разделение исключает влияние измененных координат на расчет гравитации
#pragma omp parallel for schedule(static)
    for (int first = 0; first < N; first++) {
        POSX[first] += POSR[first];
        POSY[first] += POSR[first + N];
        POSZ[first] += POSR[first + 2 * N];
    }
}
