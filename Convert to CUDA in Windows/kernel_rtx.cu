//
//  kernel_rtx.cu
//  Space Simulator CUDA
//
//  ← 原文件: RTX.metal (逐行移植, 函数顺序 / 公式 / 判断分支都和原来一致)
//       kernel void correct_vector(...)  →  __global__ void correct_vector_kernel(...)
//       kernel void RTXPath(...)         →  __global__ void RTXPath_kernel(...)
//
//  Metal → CUDA 对照(本文件用到的部分):
//     device const T*  x [[buffer(n)]]    →  const T* x          (kernel 参数, encoder 按 n 号槽位绑定)
//     thread T&        x                  →  T& x                (device 端引用参数, 一样用)
//     uint2 pos [[thread_position_in_grid]] → uint2 pos          (普通参数; nvcc 模式 kernel 第一行用
//                                                          gpuGridId2D() 自己算, host 只能传占位值)
//     fast::sqrt(x)                       →  gpu_sqrt(x)         (CUDA 没有 __sqrtf, 见 gpu_common.h)
//     fast::pow(v,2) / pow(v,2)           →  fast_pow2(v) = v*v  (数学等价; 位级不完全一致, 肉眼看不出)
//     fast::acos(x) / fast::asin(x)       →  acosf(x) / asinf(x) (CUDA 没有快速三角函数内置, 需要更快可换多项式近似)
//     rsqrt(x)                            →  rsqrtf(x)
//     abs(float)                          →  fabsf(float)
//     float3 a = float3(0.0f); a += b;     →  CUDA 的 float3 没有 +=, 拆成 .x/.y/.z 三个分量累加
//     ulong                             →  unsigned long long  (Windows 的 unsigned long 只有 32 位!)
//     long                              →  long long           (同上, Metal 的 long 永远 64 位)
//     struct metalBVHNode / RTXV / ...    →  定义搬到 gpu_common.h(host 和 device 共用同一份布局)
//
//  ⚠️ BVH 遍历是递归的(和原版一样)。Metal 靠 maxCallStackDepth = 32,
//     CUDA 靠 gpuInit() 里的 cudaDeviceSetLimit(cudaLimitStackSize, 32KB)。
//
/* 这两个 kernel 文件需要 gpu_common.h 里补的 CUDA 内置类型/函数
   (uint / clamp / max / min / gpu_sqrt / grid 线程号)。
   ⚠️ 必须**无条件** define: 这些兼容层 nvcc 模式下同样需要 —— CUDA 里既没有全局
   uint, 也没有 clamp, 更没有 __sqrtf。原来写成 #ifndef __CUDACC__ 包着, 等于
   nvcc 编的时候一条都没生效, 直接满屏 undefined identifier。
   main.cpp 里仍然绝对不要定义它(见 gpu_common.h 的说明)。 */
#define GPU_KERNEL_UNIT

#include "gpu_common.h"
#include "gpu_cuda.h"

/* 让 kernel 源码保持和原文一样的书写位置(见文件头对照表):
   MTL_DEV / MTL_THREAD / MTL_CONST 在两种模式下都是空的标记宏, 只做"这里原来是
   Metal 的哪个存储类别限定符"的记号。

   ⚠️⚠️ 这三行绝对不许写回 #define device / #define thread / #define constant!
   CUDA 的 __device__ 是这样展开的(crt/host_defines.h):
       #define __device__      __location__(device)
       #define __location__(a) __attribute__((a))      (MSVC 下是 __declspec(a))
   预处理器展开 __device__ 之后还会再扫一遍替换列表; 那时 MTL_DEV 要是个空宏,
   __attribute__((MTL_DEV)) 就被吃成 __attribute__(()), 于是所有用 GPU_D
   (= __device__ __forceinline__) 写的小函数会悄悄变成 __host__,
   nvcc 报 "calling a __host__ function from a __global__ function"。
   连带 CUDA 自己的 min/max(float,float) 也一起失效, 现场表现成"几十个函数找不到、
   类型不认识", 完全联不到这三行 #define 上头。constant 同理会毁掉 __constant__,
   MTL_THREAD 一并换成 MTL_THREAD 保持一致。 */
#define MTL_DEV
#define MTL_THREAD
#define MTL_CONST

typedef CudaRTXV        RTXV;          /* 原 RTX.metal: struct RTXV */
typedef CudaProjection  projection;    /* 原 RTX.metal: struct projection */
typedef CudaCamera      camera;        /* 原 RTX.metal: struct camera */
/* 原 RTX.metal: struct metalBVHNode —— 定义在 gpu_common.h, host 填 / device 读 */

/* 快速数学辅助 */
GPU_D float fast_pow2(float v){ return v*v; }        /* 原 fast::pow(v,2) */

/* ===================== 原 RTX.metal: kernel void correct_vector ===================== */
__global__ void correct_vector_kernel(
                      MTL_DEV const RTXV *OriginVector,
                      MTL_DEV const RTXV *Matrix,
                      MTL_DEV const int *Width,
                      MTL_DEV RTXV *Result,
                      int height,                       /* CUDA 的 grid 向上取整, 用来自行越界(原本 Metal 不用) */
                      uint2 pos
                      )
{
#ifdef __CUDACC__
    pos = gpuGridId2D();        /* host 传的是占位值, 真实线程号只能在 device 端读 */
#endif
    if(pos.x >= (unsigned int)*Width || pos.y >= (unsigned int)height)return;
    uint index = pos.y * *Width + pos.x;
    Result[index].x = OriginVector[index].x * Matrix[0].x
                    + OriginVector[index].y * Matrix[0].y
                    + OriginVector[index].z * Matrix[0].z;
    Result[index].y = OriginVector[index].x * Matrix[1].x
                    + OriginVector[index].y * Matrix[1].y
                    + OriginVector[index].z * Matrix[1].z;
    Result[index].z = OriginVector[index].x * Matrix[2].x
                    + OriginVector[index].y * Matrix[2].y
                    + OriginVector[index].z * Matrix[2].z;
}

/* ===================== 原 RTX.metal: 向量小工具 ===================== */
GPU_D float VectorDotMul(RTXV A, RTXV B){
    return A.x * B.x + A.y * B.y + A.z * B.z;
}
GPU_D float VectorLength(RTXV A){
    return gpu_sqrt(A.x * A.x + A.y * A.y + A.z * A.z);
}
GPU_D RTXV VectorUnit(RTXV A){
    float abs = VectorLength(A);
    RTXV Result = {
        A.x / abs, A.y / abs, A.z / abs
    };
    return Result;
}
GPU_D RTXV VectorMul(RTXV A, float k){
    RTXV R = {A.x * k, A.y * k, A.z * k};
    return R;
}
GPU_D RTXV VectorPlus(RTXV A, RTXV B){
    RTXV R = {A.x + B.x, A.y + B.y, A.z + B.z};
    return R;
}
GPU_D RTXV RTXVectorCalcM(RTXV v, RTXV O, RTXV P, float radius, float VLength){
    /*
     Formula:
     OE = OP * v / |v|
     OM = v/|v| * (OE - sqrt(r^2 - (OP^2 - OE^2)))
     */
    RTXV M;
    RTXV vUnit = VectorUnit(v);
    RTXV OP = {P.x-O.x, P.y-O.y, P.z-O.z};
    float OE = VectorDotMul(OP, vUnit);
    RTXV EP = VectorPlus(OP, VectorMul(vUnit, -OE));
    float delta = radius * radius - VectorDotMul(EP, EP);
    if(delta < 0){
        RTXV miss = {0,0,0,false,0};
        return miss;
    }

    RTXV OM = VectorMul(vUnit, OE - gpu_sqrt(delta));
    M = VectorPlus(OM, O);
    M.OM = OE - gpu_sqrt(delta);
    return M;
}
GPU_D RTXV RTXAngularCalcM(RTXV v, RTXV O, RTXV P, float radius, float VLength){
    RTXV M;
    float OP_v = (P.x - O.x) * v.x + (P.y - O.y) * v.y + (P.z - O.z) * v.z;
    float OP = gpu_sqrt(
                          fast_pow2(P.x - O.x) +
                          fast_pow2(P.y - O.y) +
                          fast_pow2(P.z - O.z));
    float real_angle = acosf(clamp(OP_v / (OP * VLength),-1.0f,1.0f));
    float tan_angle = asinf( radius / OP );
    if(real_angle > tan_angle){
        RTXV miss = {0,0,0,false,0};
        return miss;
    }
    float OM = OP * cosf(real_angle) - gpu_sqrt(fast_pow2(radius) - fast_pow2(OP * sinf(real_angle)));
    M.x = v.x * OM / VLength + O.x;
    M.y = v.y * OM / VLength + O.y;
    M.z = v.z * OM / VLength + O.z;
    M.OM = OM;
    return M;
}

/* ===================== 原 RTX.metal: struct metalBVHNode → gpu_common.h ===================== */
/* (定义搬走了, 这里只是把名字留着, 让下面的代码和原文一样) */
typedef CudaBVHNode metalBVHNode;

/* ===================== 原 RTX.metal: bool RTXBVHRayAABB ===================== */
GPU_D bool RTXBVHRayAABB(MTL_DEV const metalBVHNode &nextBVH, MTL_THREAD const RTXV &vector, MTL_THREAD const RTXV &camera){
    bool intersect = false;
    float t_enter, t_exit;
    if((camera.x >= nextBVH.C1.x && camera.x <= nextBVH.C2.x) && (camera.y >= nextBVH.C1.y && camera.y <= nextBVH.C2.y) && (camera.z >= nextBVH.C1.z && camera.z <= nextBVH.C2.z)){
        return true;
    }
    if(vector.x == 0 && (camera.x < nextBVH.C1.x || camera.x > nextBVH.C2.x))return false;
    if(vector.y == 0 && (camera.y < nextBVH.C1.y || camera.y > nextBVH.C2.y))return false;
    if(vector.z == 0 && (camera.z < nextBVH.C1.z || camera.z > nextBVH.C2.z))return false;
    /* x-axis */
    float x_tmin = -INFINITY;
    float x_tmax = INFINITY;
    if(vector.x != 0){
        float x_t1 = (nextBVH.C1.x - camera.x) / vector.x;
        float x_t2 = (nextBVH.C2.x - camera.x) / vector.x;

        x_tmin = min(x_t1, x_t2);
        x_tmax = max(x_t1, x_t2);
    }
    /* y-axis */
    float y_tmin = -INFINITY;
    float y_tmax = INFINITY;
    if(vector.y != 0){
        float y_t1 = (nextBVH.C1.y - camera.y) / vector.y;
        float y_t2 = (nextBVH.C2.y - camera.y) / vector.y;

        y_tmin = min(y_t1, y_t2);
        y_tmax = max(y_t1, y_t2);
    }
    /* z-axis */
    float z_tmin = -INFINITY;
    float z_tmax = INFINITY;
    if(vector.z != 0){
        float z_t1 = (nextBVH.C1.z - camera.z) / vector.z;
        float z_t2 = (nextBVH.C2.z - camera.z) / vector.z;

        z_tmin = min(z_t1, z_t2);
        z_tmax = max(z_t1, z_t2);
    }

    t_enter = max(max(x_tmin, y_tmin), z_tmin);
    t_exit = min(min(x_tmax, y_tmax), z_tmax);
    if(t_enter <= t_exit && t_exit >= 0)intersect = true;

    return intersect;
}

/* ===================== 原 RTX.metal: void RTXBVHRayForIntersect ===================== */
GPU_D void RTXBVHRayForIntersect(MTL_DEV const projection *parti, MTL_DEV const metalBVHNode &rBVH, const RTXV vector, MTL_THREAD const RTXV &camera, MTL_DEV const metalBVHNode *BVHCollector, const int BVHCollectorSize, MTL_THREAD float &nearestOM, MTL_THREAD int &nearestPlanet, MTL_THREAD RTXV &M, const float vectorLength){
    if(rBVH.left == -1 && rBVH.right == -1){
        for(int i = 0;i < rBVH.count;i++){
            long long p = rBVH.indices[i];
            RTXV Mcandi = RTXVectorCalcM(vector, RTXV{camera.x, camera.y, camera.z, true, 0}, RTXV{parti[p].x, parti[p].y, parti[p].z, true, 0}, parti[p].r, vectorLength);
            if(Mcandi.intersect == false){
                continue;
            }else{
                if(Mcandi.OM > 0 && Mcandi.OM < nearestOM){
                    nearestOM = Mcandi.OM;
                    nearestPlanet = (int)p;
                    M = Mcandi;
                }
            }
        }
        return ;
    }
    if(rBVH.left != -1){
        if(RTXBVHRayAABB(BVHCollector[rBVH.left], vector, camera)){
            RTXBVHRayForIntersect(parti, BVHCollector[rBVH.left], vector, camera, BVHCollector, BVHCollectorSize, nearestOM, nearestPlanet, M, vectorLength);
        }
    }
    if(rBVH.right != -1){
        if(RTXBVHRayAABB(BVHCollector[rBVH.right], vector, camera)){
            RTXBVHRayForIntersect(parti, BVHCollector[rBVH.right], vector, camera, BVHCollector, BVHCollectorSize, nearestOM, nearestPlanet, M, vectorLength);
        }
    }
    return ;
}

/* ===================== 原 RTX.metal: void RTXBVHRayForShadow ===================== */
GPU_D void RTXBVHRayForShadow(MTL_DEV const projection *parti, MTL_DEV const metalBVHNode &rBVH, const RTXV vector, MTL_THREAD const RTXV &camera, MTL_DEV const metalBVHNode *BVHCollector, const int BVHCollectorSize, MTL_THREAD const int nearestPlanet, MTL_THREAD const RTXV &M, MTL_THREAD const RTXV &MS, MTL_THREAD const float &MSLength, MTL_THREAD float &k, MTL_THREAD int sunIndex, MTL_THREAD float SSAOI){

    if(k == SSAOI)return;
    if(rBVH.left == -1 && rBVH.right == -1){
        for(int i = 0;i < rBVH.count;i++){
            long long p = rBVH.indices[i];
            if(p == nearestPlanet || p == sunIndex)continue;
            RTXV Shadow = RTXVectorCalcM(MS, M, RTXV{parti[p].x, parti[p].y, parti[p].z, true, 0}, parti[p].r, MSLength);
            if(Shadow.intersect && Shadow.OM < MSLength && Shadow.OM > 0){
                k = SSAOI;
                break;
            }
        }
        return ;
    }
    if(rBVH.left != -1){
        if(RTXBVHRayAABB(BVHCollector[rBVH.left], vector, camera)){
            RTXBVHRayForShadow(parti, BVHCollector[rBVH.left], vector, camera, BVHCollector, BVHCollectorSize, nearestPlanet, M, MS, MSLength, k, sunIndex, SSAOI);
        }
    }
    if(rBVH.right != -1){
        if(RTXBVHRayAABB(BVHCollector[rBVH.right], vector, camera)){
            RTXBVHRayForShadow(parti, BVHCollector[rBVH.right], vector, camera, BVHCollector, BVHCollectorSize, nearestPlanet, M, MS, MSLength, k, sunIndex, SSAOI);
        }
    }
    return ;
}

/* ===================== 原 RTX.metal: void StellarHaloDetect ===================== */
GPU_D void StellarHaloDetect(MTL_DEV const projection *parti, MTL_DEV const int *suns, long long suns_size, const RTXV vector, const RTXV camera, MTL_THREAD uint8_t &R, MTL_THREAD uint8_t &G, MTL_THREAD uint8_t &B){
    for(long long i = 0;i < suns_size;i++){
        long long p = suns[i];

        //Check if intersect with halo of the stellar
        RTXV SC = {
            parti[p].x - camera.x, parti[p].y - camera.y, parti[p].z - camera.z
        };
        float SCLength = VectorLength(SC);
        float VLength = VectorLength(vector);
        RTXV VUnit = VectorMul(vector, 1/VLength);
        float CTLength = VectorDotMul(SC, VUnit);
        if(CTLength <= 0)continue;
        float STLength = gpu_sqrt(fmaxf(0.0f, SCLength * SCLength - CTLength * CTLength));
        float I = 0.0;
        if(STLength > parti[p].r && STLength < 8 * parti[p].r){
            float q = STLength / parti[p].r;
            I += (1 - q * q / 64) / (0.8 * q * q + 0.2);
        }else if(STLength <= parti[p].r && STLength > 0){
            float q = STLength / parti[p].r;
            I = 1.5 - 0.5 * q * q;
        }
        /* 和 Metal 一样: 超出 0~255 时 uint8_t 转换都会绕回, 行为保持一致 */
        R = (uint8_t)(parti[p].c1 * I);
        G = (uint8_t)(parti[p].c2 * I);
        B = (uint8_t)(parti[p].c3 * I);
    }
}

/* ===================== 原 RTX.metal: void getTextureByCube ===================== */
GPU_D void getTextureByCube(MTL_DEV const projection &planet, MTL_DEV const uint8_t *texture, int textureID, int S, const RTXV vector, MTL_THREAD int &c1, MTL_THREAD int &c2, MTL_THREAD int &c3, MTL_THREAD int &c4){
    if(textureID < 0)return;
    float inclination = clamp(planet.inclination, -90.0f, 90.0f) * 3.1415926 / 180.0;

    RTXV rotVector = {
        vector.x * cosf(planet.rotation) + vector.y * sinf(planet.rotation) * sinf(inclination) + vector.z * sinf(planet.rotation) * cosf(inclination),
        vector.y * cosf(inclination) - vector.z * sinf(inclination),
        -vector.x * sinf(planet.rotation) + vector.y * sinf(inclination) * cosf(planet.rotation) + vector.z * cosf(planet.rotation) * cosf(planet.rotation)
    };

    rotVector.x = -rotVector.x;

    int i, j, face;
    //px
    if(rotVector.x > 0 && rotVector.x >= fabsf(rotVector.y) && rotVector.x >= fabsf(rotVector.z)){
        float t = 1 / rotVector.x;

        face = 0;
        i = (int)((1 - t * rotVector.z) * 0.5 * S);
        j = (int)((1 - t * rotVector.y) * 0.5 * S);
    }
    //nx
    else if(rotVector.x <= 0 && -rotVector.x >= fabsf(rotVector.y) && -rotVector.x >= fabsf(rotVector.z)){
        float t = -1 / rotVector.x;

        face = 1;
        i = (int)((1 + t * rotVector.z) * 0.5 * S);
        j = (int)((1 - t * rotVector.y) * 0.5 * S);
    }
    //py
    else if(rotVector.y > 0 && rotVector.y >= fabsf(rotVector.x) && rotVector.y >= fabsf(rotVector.z)){
        float t = 1 / rotVector.y;

        face = 2;
        i = (int)((1 + t * rotVector.x) * 0.5 * S);
        j = (int)((1 + t * rotVector.z) * 0.5 * S);
    }
    //ny
    else if(rotVector.y <= 0 && -rotVector.y >= fabsf(rotVector.x) && -rotVector.y >= fabsf(rotVector.z)){
        float t = -1 / rotVector.y;

        face = 3;
        i = (int)((1 + t * rotVector.x) * 0.5 * S);
        j = (int)((1 - t * rotVector.z) * 0.5 * S);
    }
    //pz
    else if(rotVector.z > 0 && rotVector.z >= fabsf(rotVector.x) && rotVector.z >= fabsf(rotVector.y)){
        float t = 1 / rotVector.z;

        face = 4;
        i = (int)((1 + t * rotVector.x) * 0.5 * S);
        j = (int)((1 - t * rotVector.y) * 0.5 * S);
    }
    //nz
    else if(rotVector.z <= 0 && -rotVector.z >= fabsf(rotVector.x) && -rotVector.z >= fabsf(rotVector.y)){
        float t = -1 / rotVector.z;

        face = 5;
        i = (int)((1 - t * rotVector.x) * 0.5 * S);
        j = (int)((1 - t * rotVector.y) * 0.5 * S);
    }
    else return;

    i = clamp(i, 0, S-1);
    j = clamp(j, 0, S-1);

    /* 原来写的是 ulong: Metal 的 ulong 是 64 位, Windows 的 unsigned long 是 32 位,
       贴图一大(6 面 * S * S * 4)就会溢出 → 必须 unsigned long long */
    unsigned long long p = (unsigned long long)textureID + ((unsigned long long)face * S * S + j * S + i) * 4;

    c1 = texture[p];
    c2 = texture[p + 1];
    c3 = texture[p + 2];
    c4 = texture[p + 3];
}

/* ===================== 原 RTX.metal: kernel void RTXPath ===================== */
__global__ void RTXPath_kernel(
                    MTL_DEV const int *Width,
                    MTL_DEV const RTXV *RTXVector,
                    MTL_DEV const float *RTXVectorLength,
                    MTL_DEV const int *suns,
                    MTL_DEV const long long *suns_size,
                    MTL_DEV const projection *parti,
                    MTL_DEV const long long *parti_size,
                    MTL_DEV const camera *cam,

                    MTL_DEV const metalBVHNode *metalBVHCollector,
                    MTL_DEV const int *metalBVHCollectorSize,
                    MTL_DEV const int *RTXReflect,

                    MTL_DEV const int *textureIndex,
                    MTL_DEV const uint8_t *texture,
                    MTL_DEV const int *S,

                    MTL_DEV const bool *hasSky,
                    MTL_DEV const uint8_t *skyTexture,
                    MTL_DEV const int *skyS,

                    MTL_DEV const projection *sky,

                    MTL_DEV uint8_t *R,
                    MTL_DEV uint8_t *G,
                    MTL_DEV uint8_t *B,
                    int height,                            /* CUDA 越界判断用(Metal 不需要) */
                    uint2 pos
                    )
{
#ifdef __CUDACC__
    pos = gpuGridId2D();        /* host 传的是占位值, 真实线程号只能在 device 端读 */
#endif
    if(pos.x >= (unsigned int)*Width || pos.y >= (unsigned int)height)return;
    uint index = pos.y * *Width + pos.x;

    /* Execute BVH & Synchronize Calculate intersect*/
    float nearestOM = INFINITY;
    int nearestPlanet = -1;
    RTXV M;
    RTXBVHRayForIntersect(parti, metalBVHCollector[0], RTXVector[index], RTXV{cam->x, cam->y, cam->z, true, 0}, metalBVHCollector, *metalBVHCollectorSize, nearestOM, nearestPlanet, M, RTXVectorLength[index]);

    bool isStellar = false;
    for(long long i = 0;i < *suns_size;i++){
        if(suns[i] == nearestPlanet){
            isStellar = true;
            break;
        }
    }
    if(nearestPlanet == -1 || isStellar){
        int RR = 0, RG = 0, RB = 0, RA = 0;
        if(*hasSky && nearestPlanet == -1){
            getTextureByCube(*sky, skyTexture, 0, *skyS, RTXVector[index], RR, RG, RB, RA);
        }
        uint8_t SR = 0, SG = 0, SB = 0;
        StellarHaloDetect(parti, suns, *suns_size, RTXVector[index], RTXV{cam->x, cam->y, cam->z, true, 0}, SR, SG, SB);
        R[index] = (uint8_t)min(255, RR + SR);
        G[index] = (uint8_t)min(255, RG + SG);
        B[index] = (uint8_t)min(255, RB + SB);
        return ;
    }

    /* Light Magnitude & Shadow Calc*/
    RTXV Npm = {
        (M.x - parti[nearestPlanet].x)/parti[nearestPlanet].r,
        (M.y - parti[nearestPlanet].y)/parti[nearestPlanet].r,
        (M.z - parti[nearestPlanet].z)/parti[nearestPlanet].r,
    };
    float I = 0;
    float SSAOI = 0.10;
    for(int i = 0;i < *suns_size;i++){
        RTXV MS = {
            parti[suns[i]].x - M.x,
            parti[suns[i]].y - M.y,
            parti[suns[i]].z - M.z
        };
        float MSLength = gpu_sqrt(
                                    fast_pow2(MS.x) +
                                    fast_pow2(MS.y) +
                                    fast_pow2(MS.z)
                                    );
        RTXV Nps = {
            MS.x/MSLength,
            MS.y/MSLength,
            MS.z/MSLength
        };
        float k = 1.0;
        RTXBVHRayForShadow(parti, metalBVHCollector[0], MS, M, metalBVHCollector, *metalBVHCollectorSize, nearestPlanet, M, MS, MSLength, k, suns[i], SSAOI);
        I += max(Npm.x * Nps.x + Npm.y * Nps.y + Npm.z * Nps.z, 0.0f) * (1 - SSAOI) * k + SSAOI;

        //Texture Renderer
        int c1 = parti[nearestPlanet].c1;
        int c2 = parti[nearestPlanet].c2;
        int c3 = parti[nearestPlanet].c3;
        int c4 = parti[nearestPlanet].c4;
        if(textureIndex[nearestPlanet] >= 0){
            getTextureByCube(parti[nearestPlanet], texture, textureIndex[nearestPlanet], S[nearestPlanet], Npm, c1, c2, c3, c4);
        }

        /* 原来是 R[index] += clamp(...)：Metal 里 uint8_t += float 会先按 float 相加再截断成 uint8_t,
           CUDA 必须写成同样的“浮点相加 → 再转 uint8_t”, 不能先把加数量化成整数。
           另外 R/G/B 每帧由 gpuBufferNew 清 0(对应 Metal 新 buffer 读为 0), 所以这里才是累加。
           注: 原文写的是 0.05(double), 这里用 0.05f 让 clamp 走 float 重载, 结果最后都要截断成 uint8, 看不出差别。 */
        R[index] = (uint8_t)((float)R[index] + clamp(((parti[suns[i]].c1 - c1) * 0.05f + c1) * I, 0.0f, 255.0f));
        G[index] = (uint8_t)((float)G[index] + clamp(((parti[suns[i]].c2 - c2) * 0.05f + c2) * I, 0.0f, 255.0f));
        B[index] = (uint8_t)((float)B[index] + clamp(((parti[suns[i]].c3 - c3) * 0.05f + c3) * I, 0.0f, 255.0f));
    }

    if(*RTXReflect > 1){
        RTXV MrPre = M;
        RTXV Mr;
        RTXV DirectVector = RTXVector[index];
        projection PreP = parti[nearestPlanet];
        float WR = 1.0;
        float RRefPre = R[index], GRefPre = G[index], BRefPre = B[index];
        float RR = R[index], GG = G[index], BB = B[index];
        for(int r = 1;r < *RTXReflect;r++){
            WR *= PreP.reflection;

            RTXV N = VectorMul(
                               RTXV{MrPre.x - PreP.x, MrPre.y - PreP.y, MrPre.z - PreP.z, true, 0},
                               1 / PreP.r
                               );
            RTXV ReflectVector = VectorPlus(DirectVector, VectorMul(N, VectorDotMul(DirectVector, N) * (-2)));
            DirectVector = ReflectVector;
            float nearestOMRef = INFINITY;
            int nearestPlanetRef = -1;
            RTXBVHRayForIntersect(parti, metalBVHCollector[0], ReflectVector, MrPre, metalBVHCollector, *metalBVHCollectorSize, nearestOMRef, nearestPlanetRef, Mr, VectorLength(ReflectVector));
            bool isStellarRef = false;
            for(long long i = 0;i < *suns_size;i++){
                if(suns[i] == nearestPlanetRef){
                    isStellarRef = true;
                    break;
                }
            }
            if(nearestPlanetRef == -1 || isStellarRef){
                uint8_t SR = 0, SG = 0, SB = 0;
                StellarHaloDetect(parti, suns, *suns_size, ReflectVector, MrPre, SR, SG, SB);
                RR += ((float)SR - RRefPre) * WR;
                GG += ((float)SG - GRefPre) * WR;
                BB += ((float)SB - BRefPre) * WR;
                break;
            }
            MrPre = Mr;
            PreP = parti[nearestPlanetRef];
            /* Light Magnitude */
            RTXV NpmRef = {
                (Mr.x - parti[nearestPlanetRef].x)/parti[nearestPlanetRef].r,
                (Mr.y - parti[nearestPlanetRef].y)/parti[nearestPlanetRef].r,
                (Mr.z - parti[nearestPlanetRef].z)/parti[nearestPlanetRef].r,
            };

            /* 原来这里是 float3 RefColor = float3(0.0f); RefColor += ...
               CUDA 的 float3 没有 += 运算符, 拆成三个分量 */
            float RefR = 0.0f, RefG = 0.0f, RefB = 0.0f;
            for(int i = 0;i < *suns_size;i++){
                float PSRef = gpu_sqrt(
                                         fast_pow2(parti[suns[i]].x - parti[nearestPlanetRef].x) + fast_pow2(parti[suns[i]].y - parti[nearestPlanetRef].y) + fast_pow2(parti[suns[i]].z - parti[nearestPlanetRef].z)
                                         );
                RTXV NpsRef = {
                    (parti[suns[i]].x - parti[nearestPlanetRef].x)/PSRef,
                    (parti[suns[i]].y - parti[nearestPlanetRef].y)/PSRef,
                    (parti[suns[i]].z - parti[nearestPlanetRef].z)/PSRef,
                };
                float MSLengthRef = gpu_sqrt(
                                         fast_pow2(parti[suns[i]].x - Mr.x) +
                                         fast_pow2(parti[suns[i]].y - Mr.y) +
                                         fast_pow2(parti[suns[i]].z - Mr.z)
                                         );
                RTXV MSRef = {
                    parti[suns[i]].x - Mr.x,
                    parti[suns[i]].y - Mr.y,
                    parti[suns[i]].z - Mr.z,
                };
                float k = 1.0;
                float Ir = max(
                               VectorDotMul(NpmRef, NpsRef) * (1.0f - SSAOI) + SSAOI,
                               0.0f
                               );
                RTXBVHRayForShadow(parti, metalBVHCollector[0], MSRef, Mr, metalBVHCollector, *metalBVHCollectorSize, nearestPlanetRef, Mr, MSRef, MSLengthRef, k, suns[i], SSAOI);
                Ir += max((NpmRef.x * NpsRef.x + NpmRef.y * NpsRef.y + NpmRef.z * NpsRef.z) * (1 - SSAOI) * k + SSAOI, 0.0f);
                if(k < 1.0f)Ir = SSAOI;

                //Texture Renderer
                int c1Ref = parti[nearestPlanetRef].c1;
                int c2Ref = parti[nearestPlanetRef].c2;
                int c3Ref = parti[nearestPlanetRef].c3;
                int c4Ref = parti[nearestPlanetRef].c4;
                if(textureIndex[nearestPlanetRef] >= 0){
                    getTextureByCube(parti[nearestPlanetRef], texture, textureIndex[nearestPlanetRef], S[nearestPlanetRef], NpmRef, c1Ref, c2Ref, c3Ref, c4Ref);
                }

                RefR += (float)((parti[suns[i]].c1 - c1Ref) * 0.05 + c1Ref) * Ir;
                RefG += (float)((parti[suns[i]].c2 - c2Ref) * 0.05 + c2Ref) * Ir;
                RefB += (float)((parti[suns[i]].c3 - c3Ref) * 0.05 + c3Ref) * Ir;
            }
            RR += (RefR - RRefPre) * WR;
            GG += (RefG - GRefPre) * WR;
            BB += (RefB - BRefPre) * WR;

            RRefPre = RefR;
            GRefPre = RefG;
            BRefPre = RefB;
        }
        R[index] = (uint8_t)clamp(RR, 0.0f, 255.0f);
        G[index] = (uint8_t)clamp(GG, 0.0f, 255.0f);
        B[index] = (uint8_t)clamp(BB, 0.0f, 255.0f);
    }
}

/* ===================== 启动(≈ [enc dispatchThreads:threadsPerThreadgroup:]) ===================== */
void gpuDispatchCorrectVector(GpuEncoder* enc, int width, int height){
    if(!enc || width <= 0 || height <= 0)return;
    MTL_DEV const RTXV *OriginVector = (MTL_DEV const RTXV*)gpuArgPtr(enc, 0);
    MTL_DEV const RTXV *Matrix       = (MTL_DEV const RTXV*)gpuArgPtr(enc, 1);
    MTL_DEV const int  *Width        = (MTL_DEV const int*)gpuArgPtr(enc, 2);
    MTL_DEV RTXV       *Result       = (MTL_DEV RTXV*)gpuArgPtr(enc, 3);

#ifdef __CUDACC__
    /* 原来: groupSize = MIN(sqrt(maxTotalThreadsPerThreadgroup), sqrt(w*h))  →  16x16 等价 */
    dim3 block(GPU_RTX_BLOCK, GPU_RTX_BLOCK);
    dim3 grid((width  + GPU_RTX_BLOCK - 1)/GPU_RTX_BLOCK,
              (height + GPU_RTX_BLOCK - 1)/GPU_RTX_BLOCK);
    correct_vector_kernel<<<grid, block>>>(OriginVector, Matrix, Width, Result, height,
                                           gpuGridIdPlaceholder2D());
#else
    /* CPU 仿真: grid 向上取整到 GPU_RTX_BLOCK 的整数倍(和 CUDA 一致),
       多出来的线程靠 kernel 开头那行越界 return 退出 —— 对拍时正好一起验 */
    int gx = ((width  + GPU_RTX_BLOCK - 1)/GPU_RTX_BLOCK)*GPU_RTX_BLOCK;
    int gy = ((height + GPU_RTX_BLOCK - 1)/GPU_RTX_BLOCK)*GPU_RTX_BLOCK;
    for(int y = 0;y < gy;y++)
        for(int x = 0;x < gx;x++)
            correct_vector_kernel(OriginVector, Matrix, Width, Result, height, make_uint2(x, y));
#endif
}

void gpuDispatchRTXPath(GpuEncoder* enc, int width, int height){
    if(!enc || width <= 0 || height <= 0)return;
    MTL_DEV const int *Width                = (MTL_DEV const int*)gpuArgPtr(enc, 0);
    MTL_DEV const RTXV *RTXVector           = (MTL_DEV const RTXV*)gpuArgPtr(enc, 1);
    MTL_DEV const float *RTXVectorLength    = (MTL_DEV const float*)gpuArgPtr(enc, 2);
    MTL_DEV const int *suns                 = (MTL_DEV const int*)gpuArgPtr(enc, 3);
    MTL_DEV const long long *suns_size      = (MTL_DEV const long long*)gpuArgPtr(enc, 4);
    MTL_DEV const projection *parti         = (MTL_DEV const projection*)gpuArgPtr(enc, 5);
    MTL_DEV const long long *parti_size     = (MTL_DEV const long long*)gpuArgPtr(enc, 6);
    MTL_DEV const camera *cam               = (MTL_DEV const camera*)gpuArgPtr(enc, 7);
    MTL_DEV const metalBVHNode *metalBVHCollector = (MTL_DEV const metalBVHNode*)gpuArgPtr(enc, 8);
    MTL_DEV const int *metalBVHCollectorSize      = (MTL_DEV const int*)gpuArgPtr(enc, 9);
    MTL_DEV const int *RTXReflect           = (MTL_DEV const int*)gpuArgPtr(enc, 10);
    MTL_DEV const int *textureIndex         = (MTL_DEV const int*)gpuArgPtr(enc, 11);
    MTL_DEV const uint8_t *texture          = (MTL_DEV const uint8_t*)gpuArgPtr(enc, 12);
    MTL_DEV const int *S                    = (MTL_DEV const int*)gpuArgPtr(enc, 13);
    MTL_DEV const bool *hasSky              = (MTL_DEV const bool*)gpuArgPtr(enc, 14);
    MTL_DEV const uint8_t *skyTexture       = (MTL_DEV const uint8_t*)gpuArgPtr(enc, 15);
    MTL_DEV const int *skyS                 = (MTL_DEV const int*)gpuArgPtr(enc, 16);
    MTL_DEV const projection *sky           = (MTL_DEV const projection*)gpuArgPtr(enc, 17);
    MTL_DEV uint8_t *R                      = (MTL_DEV uint8_t*)gpuArgPtr(enc, 18);
    MTL_DEV uint8_t *G                      = (MTL_DEV uint8_t*)gpuArgPtr(enc, 19);
    MTL_DEV uint8_t *B                      = (MTL_DEV uint8_t*)gpuArgPtr(enc, 20);

#ifdef __CUDACC__
    dim3 block(GPU_RTX_BLOCK, GPU_RTX_BLOCK);
    dim3 grid((width  + GPU_RTX_BLOCK - 1)/GPU_RTX_BLOCK,
              (height + GPU_RTX_BLOCK - 1)/GPU_RTX_BLOCK);
    RTXPath_kernel<<<grid, block>>>(Width, RTXVector, RTXVectorLength, suns, suns_size,
                                    parti, parti_size, cam,
                                    metalBVHCollector, metalBVHCollectorSize, RTXReflect,
                                    textureIndex, texture, S,
                                    hasSky, skyTexture, skyS, sky,
                                    R, G, B, height,
                                    gpuGridIdPlaceholder2D());
#else
    /* CPU 仿真: grid 向上取整(同 CUDA), 越界线程由 kernel 自己 return */
    int gx = ((width  + GPU_RTX_BLOCK - 1)/GPU_RTX_BLOCK)*GPU_RTX_BLOCK;
    int gy = ((height + GPU_RTX_BLOCK - 1)/GPU_RTX_BLOCK)*GPU_RTX_BLOCK;
    for(int y = 0;y < gy;y++)
        for(int x = 0;x < gx;x++)
            RTXPath_kernel(Width, RTXVector, RTXVectorLength, suns, suns_size,
                           parti, parti_size, cam,
                           metalBVHCollector, metalBVHCollectorSize, RTXReflect,
                           textureIndex, texture, S,
                           hasSky, skyTexture, skyS, sky,
                           R, G, B, height, make_uint2(x, y));
#endif
}
