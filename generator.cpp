#include "global.h"
#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <cmath>
#include <omp.h>

int N = 16000;
const int grav = 2;
const float EPS = 200.0f;

float* POSX = nullptr;
float* POSY = nullptr;
float* POSZ = nullptr;
float* POSR = nullptr;
float* POSM = nullptr;

static void spawnSpiralGalaxy(
    int startIdx, int count,
    float centerX, float centerY, float centerZ,
    float bulkVx, float bulkVy, float bulkVz,
    float diskRadius, float totalMass,
    float tiltX_deg, float tiltY_deg,
    int numArms, std::mt19937& gen)
{
    if (count <= 0) return;

    const float PI = 3.1415926535f;
    float radX = tiltX_deg * PI / 180.0f;
    float radY = tiltY_deg * PI / 180.0f;
    float cx = std::cos(radX), sx = std::sin(radX);
    float cy = std::cos(radY), sy = std::sin(radY);

    // 45% массы — центральная ЧД, 35% — симметричное круглое ядро (для трения при слиянии),
    // 20% — легкие спиральные рукава (не разваливают сами себя)
    float bhMass = totalMass * 0.45f;
    float bulgeMass = totalMass * 0.35f;
    float diskMass = totalMass * 0.20f;

    int bulgeCount = count / 5;
    int diskCount = count - bulgeCount;

    float bulgeStarMass = (bulgeCount > 1) ? (bulgeMass / (bulgeCount - 1)) : bulgeMass;
    float diskStarMass = (diskCount > 0) ? (diskMass / diskCount) : diskMass;

    // Временные локальные координаты в плоскости диска до 3D-поворота
    std::vector<float> lx(count, 0.0f);
    std::vector<float> ly(count, 0.0f);
    std::vector<float> lz(count, 0.0f);

    // Частица №0 — центральная черная дыра в (0, 0, 0) локальной системы
    POSM[startIdx] = bhMass;

    std::uniform_real_distribution<float> dist_u(0.0f, 1.0f);
    std::normal_distribution<float> dist_norm(0.0f, 1.0f);

    float bulgeRadius = diskRadius * 0.16f;

    // Шаг 1: Генерируем локальные координаты и массы всех звезд галактики
    for (int i = 1; i < count; i++) {
        int idx = startIdx + i;
        float r, theta, y_local, pMass;

        if (i < bulgeCount) {
            // Строго симметричное круглое ядро (без вытянутого бара, чтобы не рвало спирали)
            float u = dist_u(gen);
            r = bulgeRadius * (0.05f + 0.95f * std::pow(u, 0.55f));
            theta = dist_u(gen) * 2.0f * PI;
            y_local = dist_norm(gen) * (bulgeRadius * 0.18f * (1.0f - 0.5f * (r / bulgeRadius)));
            pMass = bulgeStarMass * (0.9f + 0.2f * dist_u(gen));
        }
        else {
            // Четкие логарифмические спиральные рукава
            float u = dist_u(gen);
            r = diskRadius * (0.12f + 0.88f * std::pow(u, 0.80f));

            int arm = i % numArms;
            float armBaseAngle = (2.0f * PI * arm) / numArms;

            float normR = (r - 0.12f * diskRadius) / (0.88f * diskRadius);
            float spiralTwist = 8.5f * std::pow(normR, 0.65f);
            float armWidth = 0.07f + 0.06f * normR;
            float angleNoise = dist_norm(gen) * armWidth;

            theta = armBaseAngle + spiralTwist + angleNoise;

            float thickness = diskRadius * 0.010f * (1.1f - 0.6f * normR);
            y_local = dist_norm(gen) * thickness;
            pMass = diskStarMass * (0.9f + 0.2f * dist_u(gen));
        }

        lx[i] = r * std::cos(theta);
        lz[i] = r * std::sin(theta);
        ly[i] = y_local;
        POSM[idx] = pMass;
    }

    // Шаг 2: Вычисляем ТОЧНУЮ круговую скорость для каждой звезды из реального поля тяжести
    // этой галактики (с тем же EPS, что в CUDA/OpenMP), чтобы диск не разваливался до удара
    const float eps2 = EPS * EPS;

    // Положение и скорость центральной точки №0
    POSX[startIdx] = centerX;
    POSY[startIdx] = centerY;
    POSZ[startIdx] = centerZ;
    POSR[startIdx] = bulkVx;
    POSR[startIdx + N] = bulkVy;
    POSR[startIdx + 2 * N] = bulkVz;

#pragma omp parallel for schedule(static)
    for (int i = 1; i < count; i++) {
        int idx = startIdx + i;
        float fx = lx[i];
        float fy = ly[i];
        float fz = lz[i];

        float r_plane = std::sqrt(fx * fx + fz * fz);
        float ax = 0.0f, az = 0.0f;

        // Суммируем реальное притяжение от всех точек своей галактики
        for (int j = 0; j < count; j++) {
            float dx = lx[j] - fx;
            float dy = ly[j] - fy;
            float dz = lz[j] - fz;

            float ext = dx * dx + dy * dy + dz * dz + eps2;
            float inv_dist = 1.0f / std::sqrt(ext);
            float den = (grav * POSM[startIdx + j]) * (inv_dist * inv_dist * inv_dist);

            ax += dx * den;
            az += dz * den;
        }

        // Проекция ускорения на радиус к центру галактики: a_inward = -(a . r) / |r|
        float a_inward = (r_plane > 1e-4f) ? -(ax * fx + az * fz) / r_plane : 0.0f;
        float v_orb = (a_inward > 0.0f) ? std::sqrt(r_plane * a_inward) : 0.0f;

        // Касательный вектор скорости в плоскости диска (вращение хвостами назад)
        float inv_r = (r_plane > 1e-4f) ? (1.0f / r_plane) : 0.0f;
        float lvx = v_orb * (fz * inv_r);
        float lvz = -v_orb * (fx * inv_r);
        float lvy = 0.0f;

        // Поворот координат и скоростей диска в 3D (вокруг X и Y)
        float x1 = fx;
        float y1 = fy * cx - fz * sx;
        float z1 = fy * sx + fz * cx;

        float vx1 = lvx;
        float vy1 = lvy * cx - lvz * sx;
        float vz1 = lvy * sx + lvz * cx;

        float x2 = x1 * cy + z1 * sy;
        float y2 = y1;
        float z2 = -x1 * sy + z1 * cy;

        float vx2 = vx1 * cy + vz1 * sy;
        float vy2 = vy1;
        float vz2 = -vx1 * sy + vz1 * cy;

        POSX[idx] = centerX + x2;
        POSY[idx] = centerY + y2;
        POSZ[idx] = centerZ + z2;

        POSR[idx] = bulkVx + vx2;
        POSR[idx + N] = bulkVy + vy2;
        POSR[idx + 2 * N] = bulkVz + vz2;
    }
}

void generateWorld()
{
    delete[] POSX; delete[] POSY; delete[] POSZ;
    delete[] POSR; delete[] POSM;

    POSX = new float[N];
    POSY = new float[N];
    POSZ = new float[N];
    POSR = new float[N * 3];
    POSM = new float[N];

    std::memset(POSR, 0, N * 3 * sizeof(float));

    std::mt19937 gen(1337);

    int N_large = (N * 3) / 4;
    int N_small = N - N_large;

    float massLarge = 2700.0f;
    float massSmall = 900.0f;

    // Начальная скорость в 4.5 раза ниже второй космической (скорости убегания):
    // галактики плавно разгоняются друг к другу собственным притяжением и сливаются в центре
    float vxLarge = 0.08f, vyLarge = 0.06f, vzLarge = 0.02f;
    float vxSmall = -vxLarge * 3.0f;
    float vySmall = -vyLarge * 3.0f;
    float vzSmall = -vzLarge * 3.0f;

    // 1. Большая галактика (Андромеда)
    spawnSpiralGalaxy(
        0, N_large,
        -1100.0f, -250.0f, 0.0f,
        vxLarge, vyLarge, vzLarge,
        1900.0f, massLarge,
        20.0f, -15.0f,
        2, gen
    );

    // 2. Малая галактика (Млечный Путь)
    spawnSpiralGalaxy(
        N_large, N_small,
        3300.0f, 750.0f, 0.0f,
        vxSmall, vySmall, vzSmall,
        1150.0f, massSmall,
        -45.0f, 25.0f,
        2, gen
    );

    printf("Generated %d particles (Large galaxy: %d, Small galaxy: %d)\n", N, N_large, N_small);
}