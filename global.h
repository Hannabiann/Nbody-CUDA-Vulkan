#ifndef GLOBAL_H
#define GLOBAL_H

extern int N;
extern const int h, w, grav;
extern const float EPS;
extern int button;
extern bool hasCuda; // Флаг доступности CUDA
extern float* POSX, * POSY, * POSZ, * POSR;
extern float* POSM;

#endif