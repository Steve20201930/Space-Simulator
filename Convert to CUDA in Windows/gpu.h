//
//  gpu.h
//  Space Simulator App — CUDA backend interface
//

#pragma once

// One Velocity-Verlet acceleration/velocity update step on the GPU.
// nx/ny/nz/mass are read-only; vx..az are updated in place.
void gpu_nbody_step(int N,
    const float* nx, const float* ny, const float* nz,
    const float* mass,
    float* vx, float* vy, float* vz,
    float* ax, float* ay, float* az,
    float dt, float G);

void gpu_shutdown();
