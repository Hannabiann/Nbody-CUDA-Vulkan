#ifndef GLOBAL_H
#define GLOBAL_H

#define APP_VER_MAJOR 4
#define APP_VER_MINOR 2
#define APP_VER_PATCH 0
#define APP_TITLE "NBody V4.2.0"

extern int N;
extern const int h, w, grav;
extern const float EPS;
extern int button;
extern bool hasCuda;
extern bool isPaused;
extern bool showAxes;
extern float simSpeed;
extern float* POSX, * POSY, * POSZ, * POSR;
extern float* POSM;

#endif