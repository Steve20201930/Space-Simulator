//
//  gpu.cu
//  Space Simulator App — CUDA port of GPU.metal `comp`
//

#include "gpu.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define BLOCK    256
#define IN_ARRS  4      // nx, ny, nz, mass          (host -> device only)
#define OUT_ARRS 6      // vx, vy, vz, ax, ay, az    (in-out)
#define NARR     (IN_ARRS + OUT_ARRS)

static inline void ck(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) {
        fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

// ---------------------------------------------------------------- kernel
// 1:1 with the Metal `comp` kernel, plus shared-memory tiling.
__global__ void comp(const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ pz,
    const float* __restrict__ pm,
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    float* __restrict__ ax,
    float* __restrict__ ay,
    float* __restrict__ az,
    float dt, float G, int N)
{
    __shared__ float sx[BLOCK], sy[BLOCK], sz[BLOCK], sm[BLOCK];

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = threadIdx.x;

    // Out-of-range threads still have to reach every __syncthreads() below,
    // so they run the loop with dummy data and just skip the final store.
    float myx = 0.0f, myy = 0.0f, myz = 0.0f;
    if (id < N) { myx = px[id]; myy = py[id]; myz = pz[id]; }

    float newax = 0.0f, neway = 0.0f, newaz = 0.0f;

    for (int tile = 0; tile < N; tile += BLOCK) {
        const int j = tile + t;
        if (j < N) { sx[t] = px[j]; sy[t] = py[j]; sz[t] = pz[j]; sm[t] = pm[j]; }
        else { sx[t] = 0.0f;  sy[t] = 0.0f;  sz[t] = 0.0f;  sm[t] = 0.0f; }
        __syncthreads();

        const int lim = (N - tile < BLOCK) ? (N - tile) : BLOCK;
#pragma unroll 4
        for (int k = 0; k < lim; k++) {
            if (tile + k == id) continue;          // == `if(i==id)continue;`

            const float dx = sx[k] - myx;
            const float dy = sy[k] - myy;
            const float dz = sz[k] - myz;

            const float r2 = dx * dx + dy * dy + dz * dz + 1e-9f;
            const float r = rsqrtf(r2);
            const float r_inv3 = r * r * r;
            const float cF = G * sm[k] * r_inv3;

            newax += dx * cF;
            neway += dy * cF;
            newaz += dz * cF;
        }
        __syncthreads();
    }

    if (id >= N) return;

    // NOTE: 0.5f, not 0.5 — a double literal here would drag the whole
    // expression onto the FP64 path (1/32 rate on GeForce).
    vx[id] += 0.5f * (ax[id] + newax) * dt;
    vy[id] += 0.5f * (ay[id] + neway) * dt;
    vz[id] += 0.5f * (az[id] + newaz) * dt;

    ax[id] = newax;
    ay[id] = neway;
    az[id] = newaz;
}

// ------------------------------------------------------------ host side
// Metal used MTLResourceStorageModeShared (unified memory, plain memcpy).
// On a discrete GPU that becomes a real PCIe transfer, so all 10 arrays are
// packed into one allocation: 1 H2D copy + 1 D2H copy per step instead of 22.
static float* d_all = nullptr;   // device, NARR * cap floats
static float* h_all = nullptr;   // pinned host staging
static int    s_cap = 0;

static void ensure_capacity(int N)
{
    if (N <= s_cap) return;

    const int cap = N + 500;     // same growth policy as s_lastCapacity
    if (d_all) cudaFree(d_all);
    if (h_all) cudaFreeHost(h_all);

    ck(cudaMalloc((void**)&d_all, sizeof(float) * NARR * (size_t)cap), "cudaMalloc");
    ck(cudaHostAlloc((void**)&h_all, sizeof(float) * NARR * (size_t)cap,
        cudaHostAllocDefault), "cudaHostAlloc");
    s_cap = cap;
}

void gpu_nbody_step(int N,
    const float* nx, const float* ny, const float* nz,
    const float* mass,
    float* vx, float* vy, float* vz,
    float* ax, float* ay, float* az,
    float dt, float G)
{
    if (N <= 0) return;
    ensure_capacity(N);

    const size_t n = (size_t)N;   // arrays are packed with stride N, not cap,
    // so nothing unused is ever transferred.

    const float* in[IN_ARRS] = { nx, ny, nz, mass };
    float* io[OUT_ARRS] = { vx, vy, vz, ax, ay, az };

    for (int a = 0; a < IN_ARRS; a++)
        memcpy(h_all + (size_t)a * n, in[a], sizeof(float) * n);
    for (int a = 0; a < OUT_ARRS; a++)
        memcpy(h_all + (size_t)(IN_ARRS + a) * n, io[a], sizeof(float) * n);

    ck(cudaMemcpy(d_all, h_all, sizeof(float) * NARR * n,
        cudaMemcpyHostToDevice), "H2D");

    float* d_nx = d_all + 0 * n;
    float* d_ny = d_all + 1 * n;
    float* d_nz = d_all + 2 * n;
    float* d_m = d_all + 3 * n;
    float* d_vx = d_all + 4 * n;
    float* d_vy = d_all + 5 * n;
    float* d_vz = d_all + 6 * n;
    float* d_ax = d_all + 7 * n;
    float* d_ay = d_all + 8 * n;
    float* d_az = d_all + 9 * n;

    const int blocks = (N + BLOCK - 1) / BLOCK;
    comp << <blocks, BLOCK >> > (d_nx, d_ny, d_nz, d_m,
        d_vx, d_vy, d_vz,
        d_ax, d_ay, d_az,
        dt, G, N);
    ck(cudaGetLastError(), "kernel launch");

    // cudaMemcpy is synchronous — this is the waitUntilCompleted equivalent.
    ck(cudaMemcpy(h_all + (size_t)IN_ARRS * n,
        d_all + (size_t)IN_ARRS * n,
        sizeof(float) * OUT_ARRS * n,
        cudaMemcpyDeviceToHost), "D2H");

    for (int a = 0; a < OUT_ARRS; a++)
        memcpy(io[a], h_all + (size_t)(IN_ARRS + a) * n, sizeof(float) * n);
}

void gpu_shutdown()
{
    if (d_all) { cudaFree(d_all);     d_all = nullptr; }
    if (h_all) { cudaFreeHost(h_all); h_all = nullptr; }
    s_cap = 0;
}
