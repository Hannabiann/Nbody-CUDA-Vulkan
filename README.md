# Nbody-CUDA-Vulkan

This is an evolution of the [Nbody-CUDA-GLUT project](https://github.com/Hannabiann/Nbody-CUDA-GLUT), optimized and rewritten for the Vulkan graphics API with the help of AI.

<img width="2258" height="1029" alt="Nbody-CUDA-Vulkan" src="https://github.com/user-attachments/assets/e04ebdbf-4296-414f-b7fe-5ad65e693fa6" />

## Bench (use fast-math, v4.1)
__Windows__

| Device/Mode | N=16000 (FPS) | N=64000 (FPS) |
| :--- | :---: | :---: |
| Ryzen 9800x3d (AVX-512) | ~ 210 | ~ 13 |
| Ryzen 9800x3d (AVX-2) | ~ 109 | ~ 7 |
| Ryzen 7600 (AVX-512) | ~ 93 | ~ 6 |
| Ryzen 7600 (AVX-2) | ~ 73 | ~ 5 |
| RTX 4060 (Force) | ~ 680 | ~ 81 |
| RTX 4060 (Tile) | ~ 1084 | ~ 81 |
| RTX 4090 (Force) | ~ 1300 | ~ 265 |
| RTX 4090 (Tile) | ~ 2390 | ~ 408 |

__Linux (CachyOS)__
| Device/Mode | N=16000 (FPS) | N=64000 (FPS) |
| :--- | :---: | :---: |
| Ryzen 7700 (AVX-512) | ~ 130 | ~ 8 |
| Ryzen 7700 (AVX-2) | ~ 109 | ~ 7 |
| RTX 4070 (Force) | ~ 1075 | ~ 93 |
| RTX 4070 (Tile) | ~ 1720 | ~ 157 |
