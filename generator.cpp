#include "global.h"
#include <iostream>
#include <random>
#include <cstring>
#include <cmath>

int N = 16000;
const int grav = 3;       // Ускоренная гравитация
const float EPS = 220.0f; // Оптимальное сглаживание для сохранения четких спиралей

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

    // 75% массы — в центральной сверхмассивной черной дыре/ядре,
    // чтобы рукава вращались стабильно и не распадались сами по себе
    float coreMass = totalMass * 0.75f;
    float diskMass = totalMass * 0.25f;
    float starMass = (count > 1) ? (diskMass / (count - 1)) : totalMass;

    // Частица №0 — центральное сверхмассивное ядро
    POSX[startIdx] = centerX;
    POSY[startIdx] = centerY;
    POSZ[startIdx] = centerZ;
    POSM[startIdx] = coreMass;
    POSR[startIdx] = bulkVx;
    POSR[startIdx + N] = bulkVy;
    POSR[startIdx + 2 * N] = bulkVz;

    std::uniform_real_distribution<float> dist_u(0.0f, 1.0f);
    std::normal_distribution<float> dist_norm(0.0f, 1.0f);

    float eps2 = EPS * EPS;
    int bulgeCount = count / 7; // ~14% звезд образуют плотный центральный балдж/бар

    for (int i = 1; i < count; i++) {
        int idx = startIdx + i;
        float r, theta, ly;

        if (i < bulgeCount) {
            // --- ЦЕНТРАЛЬНЫЙ БАЛДЖ И ПЕРЕМЫЧКА (БАР) ---
            float u = dist_u(gen);
            r = diskRadius * 0.18f * std::pow(u, 0.6f);
            // Вытягиваем балдж вдоль линии начала спиральных рукавов (эффект бара)
            int barSide = i % 2;
            theta = barSide * PI + dist_norm(gen) * 0.45f;
            ly = dist_norm(gen) * (diskRadius * 0.035f * (1.0f - r / (diskRadius * 0.2f)));
        }
        else {
            // --- ЧЕТКИЕ ЛОГАРИФМИЧЕСКИЕ СПИРАЛЬНЫЕ РУКАВА ---
            float u = dist_u(gen);
            // Звезды начинаются от края бара (0.12 R) до края диска (1.0 R)
            r = diskRadius * (0.12f + 0.88f * std::pow(u, 0.8f));

            int arm = i % numArms;
            float armBaseAngle = (2.0f * PI * arm) / numArms;

            // Закрутка 8.5 радиан (~1.35 полного оборота) с нелинейным шагом как у реальных галактик
            float normR = (r - 0.12f * diskRadius) / (0.88f * diskRadius);
            float spiralTwist = 8.5f * std::pow(normR, 0.65f);

            // Узкий профиль рукава: звезды плотно жмутся к линии спирали
            float armWidth = 0.08f + 0.07f * normR;
            float angleNoise = dist_norm(gen) * armWidth;

            theta = armBaseAngle + spiralTwist + angleNoise;

            // Очень тонкий диск по оси Y
            float thickness = diskRadius * 0.012f * (1.1f - 0.6f * normR);
            ly = dist_norm(gen) * thickness;
        }

        float lx = r * std::cos(theta);
        float lz = r * std::sin(theta);

        // Масса внутри радиуса r (ядро + часть диска)
        float enclosedDiskFraction = std::pow(r / diskRadius, 1.35f);
        if (enclosedDiskFraction > 1.0f) enclosedDiskFraction = 1.0f;
        float enclosedMass = coreMass + diskMass * enclosedDiskFraction;

        // Точная круговая орбитальная скорость в потенциале Пламмера
        float ext = r * r + eps2;
        float v_orb = r * std::sqrt((grav * enclosedMass) / (ext * std::sqrt(ext)));

        // Вращение в ту сторону, куда закручены рукава (хвостами назад)
        float lvx = v_orb * std::sin(theta);
        float lvz = -v_orb * std::cos(theta);
        float lvy = 0.0f;

        // Поворот диска в 3D (вокруг X, затем вокруг Y)
        float x1 = lx;
        float y1 = ly * cx - lz * sx;
        float z1 = ly * sx + lz * cx;

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

        POSM[idx] = starMass * (0.85f + 0.3f * dist_u(gen));
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

    // Пропорция числа звезд 3:1 (Андромеда и Млечный Путь)
    int N_large = (N * 3) / 4;
    int N_small = N - N_large;

    float massLarge = 2100.0f;
    float massSmall = 700.0f;

    // Задаем скорость так, чтобы галактики прошли по касательной спиральной орбите
    float vxLarge = 0.38f, vyLarge = 0.18f, vzLarge = 0.08f;
    float vxSmall = -vxLarge * 3.0f;
    float vySmall = -vyLarge * 3.0f;
    float vzSmall = -vzLarge * 3.0f;

    // 1. Большая галактика (Андромеда: 2 мощных рукава, радиус 2100)
    spawnSpiralGalaxy(
        0, N_large,
        -1500.0f, -450.0f, 0.0f,
        vxLarge, vyLarge, vzLarge,
        2100.0f, massLarge,
        22.0f, -15.0f,
        2, gen
    );

    // 2. Малая галактика (Млечный Путь: 2 рукава, радиус 1200)
    spawnSpiralGalaxy(
        N_large, N_small,
        4500.0f, 1350.0f, 0.0f,
        vxSmall, vySmall, vzSmall,
        1200.0f, massSmall,
        -50.0f, 30.0f,
        2, gen
    );

    printf("Generated %d particles (Large galaxy: %d, Small galaxy: %d)\n", N, N_large, N_small);
}