#include "global.h"
#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <omp.h>

int N = 16000;
const int grav = 2;
const float EPS = 200.0f;

float* POSX = nullptr;
float* POSY = nullptr;
float* POSZ = nullptr;
float* POSR = nullptr;
float* POSM = nullptr;

// Внутренний массив цветов звезд (заполняется в генераторе по типу звезды)
static std::vector<float> starColors;

void getStarColor(int idx, float& r, float& g, float& b)
{
    if (idx * 3 + 2 < (int)starColors.size()) {
        r = starColors[idx * 3 + 0];
        g = starColors[idx * 3 + 1];
        b = starColors[idx * 3 + 2];
    }
    else {
        r = 1.0f; g = 1.0f; b = 1.0f;
    }
}

struct GalaxyProfile {
    bool isBarred;       // true = Млечный Путь (с баром и 4 рукавами), false = Андромеда (M31)
    float diskRadius;
    float totalMass;
    float tiltX_deg;
    float tiltY_deg;
};

static void spawnRealisticGalaxy(
    int startIdx, int count,
    float centerX, float centerY, float centerZ,
    float bulkVx, float bulkVy, float bulkVz,
    const GalaxyProfile& prof,
    std::mt19937& gen)
{
    if (count <= 0) return;

    const float PI = 3.1415926535f;
    float radX = prof.tiltX_deg * PI / 180.0f;
    float radY = prof.tiltY_deg * PI / 180.0f;
    float cx = std::cos(radX), sx = std::sin(radX);
    float cy = std::cos(radY), sy = std::sin(radY);

    // Распределение массы: 50% — центральное ядро/гало (гарантирует стабильность рукавов и бара),
    // 30% — сфероидальный балдж, 20% — спиральный диск
    float bhMass = prof.totalMass * 0.50f;
    float bulgeMass = prof.totalMass * 0.30f;
    float diskMass = prof.totalMass * 0.20f;

    int bulgeCount = (int)(count * 0.22f); // 22% звезд — балдж и бар
    int haloCount = (int)(count * 0.06f); // 6% звезд — сферическое 3D-гало
    int diskCount = count - bulgeCount - haloCount;

    float bulgeStarMass = (bulgeCount > 1) ? (bulgeMass / (bulgeCount - 1)) : bulgeMass;
    float diskStarMass = (diskCount + haloCount > 0) ? (diskMass / (diskCount + haloCount)) : diskMass;

    std::vector<float> lx(count, 0.0f);
    std::vector<float> ly(count, 0.0f);
    std::vector<float> lz(count, 0.0f);

    // Частица №0 — сверхмассивный центр галактики (яркая бело-золотая точка)
    POSM[startIdx] = bhMass;
    starColors[startIdx * 3 + 0] = 1.0f;
    starColors[startIdx * 3 + 1] = 0.95f;
    starColors[startIdx * 3 + 2] = 0.80f;

    std::uniform_real_distribution<float> dist_u(0.0f, 1.0f);
    std::normal_distribution<float> dist_norm(0.0f, 1.0f);
    // Гамма-распределение дает настоящий экспоненциальный диск без резкого обрыва на краях
    std::gamma_distribution<float> dist_expDisk(2.2f, 0.26f);

    float R = prof.diskRadius;
    float bulgeRadius = R * 0.18f;
    float barLength = R * 0.26f;
    float barAngle = 0.45f; // Угол наклона центральной перемычки Млечного Пути

    for (int i = 1; i < count; i++) {
        int idx = startIdx + i;
        float x_local = 0.0f, y_local = 0.0f, z_local = 0.0f;
        float pMass = diskStarMass;
        float cr = 1.0f, cg = 1.0f, cb = 1.0f;

        if (i < bulgeCount) {
            // --- 1. ЦЕНТРАЛЬНЫЙ БАЛДЖ И БАР (ПЕРЕМЫЧКА) ---
            float u = dist_u(gen);
            bool inBar = prof.isBarred && (i > bulgeCount / 2);

            if (inBar) {
                // Звезды центральной перемычки (бара) Млечного Пути
                float along = (dist_u(gen) * 2.0f - 1.0f) * barLength;
                float across = dist_norm(gen) * (barLength * 0.22f * (1.0f - 0.5f * std::fabs(along) / barLength));
                x_local = along * std::cos(barAngle) - across * std::sin(barAngle);
                z_local = along * std::sin(barAngle) + across * std::cos(barAngle);
                y_local = dist_norm(gen) * (R * 0.022f);
                // Носим массу легкой звезды диска, чтобы несимметричный бар не разрывал орбиты
                pMass = diskStarMass;
            }
            else {
                // Сферически-эллиптический балдж (профиль Хернквиста, плотный к центру)
                float r = bulgeRadius * std::pow(u, 0.5f);
                float theta = dist_u(gen) * 2.0f * PI;
                float z_cos = dist_u(gen) * 2.0f - 1.0f;
                float sin_t = std::sqrt(std::max(0.0f, 1.0f - z_cos * z_cos));

                x_local = r * sin_t * std::cos(theta);
                z_local = r * sin_t * std::sin(theta);
                y_local = r * z_cos * 0.55f; // Сплюснутый сфероид
                pMass = bulgeStarMass * (0.9f + 0.2f * dist_u(gen));
            }

            // Цвет балджа: от раскаленного бело-желтого в самом центре до глубокого янтарного
            float r_norm = std::min(1.0f, std::sqrt(x_local * x_local + z_local * z_local) / barLength);
            cr = 1.0f;
            cg = 0.88f - 0.25f * r_norm + dist_norm(gen) * 0.04f;
            cb = 0.60f - 0.35f * r_norm + dist_norm(gen) * 0.04f;
        }
        else if (i < bulgeCount + haloCount) {
            // --- 2. СФЕРИЧЕСКОЕ ГАЛО И ШАРОВЫЕ СКОПЛЕНИЯ ---
            float u = dist_u(gen);
            float r = R * (0.15f + 0.95f * std::pow(u, 0.7f));
            float theta = dist_u(gen) * 2.0f * PI;
            float z_cos = dist_u(gen) * 2.0f - 1.0f;
            float sin_t = std::sqrt(std::max(0.0f, 1.0f - z_cos * z_cos));

            x_local = r * sin_t * std::cos(theta);
            z_local = r * sin_t * std::sin(theta);
            y_local = r * z_cos * 0.38f;
            pMass = diskStarMass * 0.8f;

            // Тусклые старые звезды гало (теплый кремово-золотистый оттенок)
            cr = 0.75f; cg = 0.65f; cb = 0.50f;
        }
        else {
            // --- 3. СПИРАЛЬНЫЙ ДИСК (МЛЕЧНЫЙ ПУТЬ vs АНДРОМЕДА) ---
            float r_frac = dist_expDisk(gen);
            if (r_frac > 1.35f) r_frac = 0.25f + 0.75f * dist_u(gen);

            float minR = prof.isBarred ? barLength * 0.85f : R * 0.10f;
            float r = minR + (R - minR) * r_frac;

            // Для Андромеды (M31): знаменитое «Кольцо огня» на радиусе ~0.55 R
            bool inAndromedaRing = (!prof.isBarred && dist_u(gen) < 0.18f);
            if (inAndromedaRing) {
                r = R * (0.54f + dist_norm(gen) * 0.045f);
            }

            float normR = std::max(0.001f, (r - minR) / (R - minR));
            float theta = 0.0f;
            bool isMainArm = true;

            if (prof.isBarred) {
                // МЛЕЧНЫЙ ПУТЬ: 4 рукава (2 главных рукава Персея/Щита-Центавра + 2 вторичных)
                float armPick = dist_u(gen);
                int armIdx = 0;
                if (armPick < 0.36f) { armIdx = 0; isMainArm = true; }
                else if (armPick < 0.72f) { armIdx = 2; isMainArm = true; }
                else if (armPick < 0.86f) { armIdx = 1; isMainArm = false; }
                else { armIdx = 3; isMainArm = false; }

                // Главные рукава выходят прямо из концов бара
                float baseAngle = barAngle + armIdx * (PI * 0.5f);
                // Логарифмическая закрутка
                float twist = 4.6f * std::log(1.0f + 2.6f * normR);
                theta = baseAngle + twist;
            }
            else {
                // АНДРОМЕДА (M31): 2 плотно закрученных рукава с ветвящимися «перьями» (spurs)
                int armIdx = i % 2;
                float baseAngle = armIdx * PI;
                float twist = 6.2f * std::log(1.0f + 3.0f * normR);
                theta = baseAngle + twist;

                // Боковые ответвления («перья») на внешних витках
                if (normR > 0.35f && (i % 7 == 0)) {
                    theta += 0.25f * std::sin(normR * 18.0f);
                }
            }

            // Асимметричный профиль волны плотности:
            // узкий плотный внутренний хребет и широкий размытый шлейф позади рукава
            float rawNoise = dist_norm(gen);
            float armSpread = (isMainArm ? 0.075f : 0.11f) + 0.06f * normR;
            float asymmetricOffset = (rawNoise > 0.0f) ? (rawNoise * armSpread * 1.7f) : (rawNoise * armSpread * 0.6f);

            // 12% звезд диска — фоновые звезды между рукавами
            bool isInterArm = (dist_u(gen) < 0.12f);
            if (isInterArm) {
                asymmetricOffset = (dist_u(gen) * 2.0f - 1.0f) * PI;
            }

            theta += asymmetricOffset;

            float thickness = R * 0.011f * (1.1f - 0.5f * std::min(1.0f, normR));
            x_local = r * std::cos(theta);
            z_local = r * std::sin(theta);
            y_local = dist_norm(gen) * thickness;
            pMass = diskStarMass * (0.85f + 0.3f * dist_u(gen));

            // --- ЦВЕТА ЗВЕЗДНЫХ ПОПУЛЯЦИЙ В РУКАВАХ ---
            float distFromRidge = std::fabs(rawNoise);
            if (!isInterArm && distFromRidge < 0.32f && dist_u(gen) < 0.09f) {
                // 1. Ярко-розовые/рубиновые узелки H-II туманностей вдоль самого хребта рукава
                cr = 1.00f;
                cg = 0.28f + 0.10f * dist_u(gen);
                cb = 0.62f + 0.15f * dist_u(gen);
                y_local *= 0.4f; // Туманности лежат строго в тонкой плоскости диска
            }
            else if (!isInterArm && distFromRidge < 0.85f) {
                // 2. Молодые горячие голубые гиганты (O/B-звезды) в рукавах
                if (prof.isBarred) {
                    // Млечный Путь — сапфирово-голубые рукава
                    cr = 0.45f + 0.15f * distFromRidge;
                    cg = 0.72f + 0.12f * distFromRidge;
                    cb = 1.00f;
                }
                else {
                    // Андромеда — чуть более лазурно-фиолетовый/серебристый оттенок
                    cr = 0.55f + 0.20f * distFromRidge;
                    cg = 0.78f + 0.12f * distFromRidge;
                    cb = 1.00f;
                }
            }
            else {
                // 3. Звезды шлейфа и межрукавного диска (желтовато-белые звезды типа Солнца)
                cr = 0.85f;
                cg = 0.82f;
                cb = 0.76f;
            }
        }

        lx[i] = x_local;
        ly[i] = y_local;
        lz[i] = z_local;
        POSM[idx] = pMass;

        starColors[idx * 3 + 0] = std::clamp(cr, 0.0f, 1.0f);
        starColors[idx * 3 + 1] = std::clamp(cg, 0.0f, 1.0f);
        starColors[idx * 3 + 2] = std::clamp(cb, 0.0f, 1.0f);
    }

    // Шаг 2: Честный параллельный расчет круговой скорости для каждой звезды
    const float eps2 = EPS * EPS;

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

        float a_inward = (r_plane > 1e-4f) ? -(ax * fx + az * fz) / r_plane : 0.0f;
        float v_orb = (a_inward > 0.0f) ? std::sqrt(r_plane * a_inward) : 0.0f;

        float inv_r = (r_plane > 1e-4f) ? (1.0f / r_plane) : 0.0f;
        float lvx = v_orb * (fz * inv_r);
        float lvz = -v_orb * (fx * inv_r);
        float lvy = 0.0f;

        // 3D-поворот галактики
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
    starColors.assign(N * 3, 1.0f);

    std::memset(POSR, 0, N * 3 * sizeof(float));

    std::mt19937 gen(1337);

    // 70% звезд — гигантская Андромеда (M31), 30% — Млечный Путь (пересеченная спираль с баром)
    int N_andromeda = (N * 7) / 10;
    int N_milkyWay = N - N_andromeda;

    float massRatio = (float)N_andromeda / (float)N_milkyWay;
    float massMilkyWay = 1100.0f;
    float massAndromeda = massMilkyWay * massRatio;

    // Сближение по плавной орбите захвата (центр масс неподвижен в 0,0,0)
    float vxM31 = 0.09f, vyM31 = 0.05f, vzM31 = 0.03f;
    float vxMW = -vxM31 * massRatio;
    float vyMW = -vyM31 * massRatio;
    float vzMW = -vzM31 * massRatio;

    // 1. Галактика Андромеды (M31): без бара, 2 плотных рукава + Кольцо огня, радиус 2100
    GalaxyProfile m31Profile = {
        false,          // Классическая спираль SA(s)b
        2100.0f,        // Радиус диска
        massAndromeda,  // Масса
        22.0f, -15.0f   // 3D-наклон
    };
    spawnRealisticGalaxy(
        0, N_andromeda,
        -1200.0f, -250.0f, 0.0f,
        vxM31, vyM31, vzM31,
        m31Profile, gen
    );

    // 2. Млечный Путь: центральный бар + 4 рукава (2 главных, 2 вторичных), радиус 1300
    GalaxyProfile mwProfile = {
        true,           // Спираль с перемычкой SBbc
        1300.0f,        // Радиус диска
        massMilkyWay,   // Масса
        -42.0f, 28.0f   // 3D-наклон
    };
    spawnRealisticGalaxy(
        N_andromeda, N_milkyWay,
        1200.0f * massRatio, 250.0f * massRatio, 0.0f,
        vxMW, vyMW, vzMW,
        mwProfile, gen
    );

    printf("Generated %d particles (Andromeda M31: %d, Milky Way: %d)\n", N, N_andromeda, N_milkyWay);
}