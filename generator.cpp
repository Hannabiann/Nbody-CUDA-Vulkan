#include "global.h"
#include <iostream>
#include <random>
#include <cstring> // Для std::memset

const int N = 16000;
const int grav = 5000000;
const float EPS = 3E5f;

// Координаты точек
float POSX[N], POSY[N], POSZ[N], POSR[N * 3];
// Массы точек
float POSM[N];

// Оптимизированный однопоточный генератор мира
void generateWorld()
{
    // Вихрь Мерсенна — инициализируется один раз, работает быстро и выдает качественный рандом
    std::mt19937 gen(1337);

    // Равномерные распределения float (работают быстрее и правильнее, чем rand() % X)
    std::uniform_real_distribution<float> dist_coord(0.0f, 1000.0f);
    std::uniform_real_distribution<float> dist_mass(1.0f, 1001.0f);

    // Быстрое обнуление скоростей через векторные инструкции процессора
    std::memset(POSR, 0, sizeof(POSR));

    const int particles_per_zone = N / 8;

    // Обычный последовательный цикл, компилятор сам векторизует математику
    for (int i = 0; i < N; i++) {
        int zone = i / particles_per_zone;
        if (zone > 7) zone = 7;

        // Вычисляем смещения зон через битовые маски БЕЗ использования switch-case
        float signX = (zone & 2) ? 1000.0f : -2000.0f;
        float signY = (zone & 4) ? -2000.0f : 1000.0f;
        float signZ = (zone & 1) ? -2000.0f : 1000.0f;

        // Генерация координат
        POSX[i] = dist_coord(gen) + signX;
        POSY[i] = dist_coord(gen) + signY;
        POSZ[i] = dist_coord(gen) + signZ;

        // Генерация массы
        POSM[i] = dist_mass(gen);
    }

    printf("Generated %d particles\n", N);
}
