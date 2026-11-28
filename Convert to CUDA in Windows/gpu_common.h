//
//  gpu_common.h
//  Space Simulator CUDA
//
//  host(main.cpp) 和 device(.cu) 共用的一套结构体。
//  字段顺序 / 类型 / 默认值和原来 GPU.metal、RTX.metal 里的完全一致，
//  这样 main.cpp 里“填结构体”的代码基本不用动，GPU 端按同一布局读就行。
//
//  ⚠️ 必须注意的 ABI 坑（Windows 上最容易踩）:
//     Metal 里的 long 永远是 64 位；Windows(MSVC) 里 C++ 的 long 只有 32 位。
//     所以这里（以及 main.cpp 里对应的地方）统一改成 long long，
//     否则 GPU 读到的 indices / suns_size / parti_size 全部错位。
//
#ifndef GPU_COMMON_H
#define GPU_COMMON_H

#include <cstdint>
#include <cstddef>
#include <cmath>

/* ------------------------------------------------------------------ */
/* CUDA 关键字兼容:                                                    */
/* 用 nvcc 编译时走 CUDA 分支;                                         */
/* 用普通 C++(MSVC / mac 上对拍用的 clang++)编译 .cu 时走 CPU 分支,   */
/* 这样同一份 kernel 源码能在 mac 上跑 CPU 数值对拍验证。              */
/* ------------------------------------------------------------------ */
#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define GPU_D __device__ __forceinline__      // device 端小函数
#else
    #define __global__
    #define __device__
    #define __host__
    #define __forceinline__
    #define GPU_D inline                          // CPU 模式下就是普通 inline

    /* -------------------------------------------------------------
       ⚠️ 下面这块"给 CPU 模式补 CUDA 内置向量类型/快速数学"的代码, 只在编译
       kernel(.cu) 时才需要, 所以用 GPU_KERNEL_UNIT 包起来。
       main.cpp 里千万 define 它: main.cpp 有 using namespace std;,
       再往全局命名空间塞 max/min/clamp 会和 <algorithm> 里的 std::max /
       std::min / std::clamp 撞名, MSVC 会直接报 call to 'max' is ambiguous。
       kernel 文件(kernel_comp.cu / kernel_rtx.cu)顶部会自己 #define 它。
       ------------------------------------------------------------- */
    #if defined(GPU_KERNEL_UNIT)
    #include <algorithm>
    #include <math.h>        /* 全局命名空间的 sqrtf/sinf/cosf/acosf/asinf/fabsf */

    /* CUDA 内置类型 */
    struct uint2 { unsigned int x, y; };
    struct uint3 { unsigned int x, y, z; };
    struct dim3 {
        unsigned int x, y, z;
        dim3(unsigned int X = 1, unsigned int Y = 1, unsigned int Z = 1) : x(X), y(Y), z(Z) {}
    };

    static inline uint2 make_uint2(unsigned int x, unsigned int y){ uint2 v; v.x = x; v.y = y; return v; }

    /* CUDA 的向量类型(浮点/字节), CPU 模式下自己补一份 */
    struct float2 { float x, y; };
    struct float3 { float x, y, z; };
    struct float4 { float x, y, z, w; };
    struct uchar4 { unsigned char x, y, z, w; };
    static inline float2 make_float2(float x, float y){ float2 v{x,y}; return v; }
    static inline float3 make_float3(float x, float y, float z){ float3 v{x,y,z}; return v; }
    static inline float4 make_float4(float x, float y, float z, float w){ float4 v{x,y,z,w}; return v; }
    static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w){ uchar4 v{x,y,z,w}; return v; }

    /* CUDA 快速数学内置函数 → 标准数学函数(CPU 模式下数值会略有差别, 只用于对拍逻辑) */
    static inline float gpu_rsqrtf(float x){ return 1.0f / sqrtf(x); }
    static inline float gpu__powf(float a, float b){ return (float)pow((double)a, (double)b); }
    #define rsqrtf(x)   gpu_rsqrtf(x)
    #define __powf(a,b) gpu__powf(a,b)
    #endif   /* GPU_KERNEL_UNIT(CPU 模式专用块) */
#endif

/* ================================================================== */
/* 两种模式(nvcc / CPU 对拍)都要的兼容层 —— 同样只在编 kernel 时打开。   */
/*                                                                    */
/* 这些原来全被放在 #ifndef __CUDACC__ 里面, 结果是 nvcc 编的时候一个都  */
/* 没生效, 满屏 undefined identifier。逐条说明为什么 nvcc 也需要:        */
/*   uint    —— CUDA 头里没有全局 uint typedef(只有 cuda::std 里的);     */
/*   clamp   —— CUDA Math API 根本没有 clamp(只有 cuda::std::clamp);     */
/*   gpu_sqrt—— CUDA 里没有 __sqrtf 这个东西(只有 sqrtf/__fsqrt_rn);     */
/*   max/min —— 新版 CUDA 有 max(float,float), 但 9.x/10.x 只有整数版,   */
/*              自带模板兜底; 实参完全匹配时 CUDA 自己的非模板重载优先级  */
/*              更高, 所以和它们共存不会产生二义性。                     */
/* ================================================================== */
#if defined(GPU_KERNEL_UNIT)

    /* CUDA 内置的 uint。重复 typedef 同一个类型在 C++ 里是合法的,
       所以就算以后某个 Toolkit 自己定义了 uint 也不会打架。 */
    typedef unsigned int uint;

    /* Metal 的 clamp(x,lo,hi): CUDA 没有, 自己补三份重载。
       万一某个 CUDA Toolkit 版本哪天自带了 clamp 并且打架,
       在项目属性 → CUDA C/C++ → Preprocessor Definitions 里加
       GPU_DEFINE_OWN_CLAMP=0 可以关掉这里的实现。 */
    #if !defined(GPU_DEFINE_OWN_CLAMP) || GPU_DEFINE_OWN_CLAMP
    GPU_D float  clamp(float v, float lo, float hi){ return v < lo ? lo : (v > hi ? hi : v); }
    GPU_D double clamp(double v, double lo, double hi){ return v < lo ? lo : (v > hi ? hi : v); }
    GPU_D int    clamp(int v, int lo, int hi){ return v < lo ? lo : (v > hi ? hi : v); }
    #endif

    /* Metal 的 max()/min() */
    template <typename T> GPU_D T max(T a, T b){ return a < b ? b : a; }
    template <typename T> GPU_D T min(T a, T b){ return a < b ? a : b; }

    /* 快速开方。原文件里写的是 __sqrtf(x), 但 CUDA 里**不存在** __sqrtf
       (grep 遍 CUDA 12.6 头文件: 只有 sqrtf / __fsqrt_rn / rsqrtf / __powf)。
       对应 Metal 的 fast::sqrt(), 这里用正确舍入的硬件开方 __fsqrt_rn;
       要更快可以把 CUDA 编译选项 Optimization → 打开 fast math, 或换 rsqrtf 近似。 */
    #ifdef __CUDACC__
    GPU_D float gpu_sqrt(float x){ return __fsqrt_rn(x); }
    #else
    static inline float gpu_sqrt(float x){ return sqrtf(x); }
    #endif

    /* ---------------- grid 线程号(≈ Metal 的 [[thread_position_in_grid]]) ----------------
       ⚠️ 这是本文件最容易踩的坑: CUDA 的 blockIdx/threadIdx 在
       device_launch_parameters.h 里声明成 extern const, 只有 nvcc 把它们翻译成
       特殊寄存器读取, 而且**只能在 device 代码里用**。在 host 函数(下面的
       gpuDispatch* 启动函数)里读它, 编译器不报错, 但链接时必炸:
           unresolved external symbol threadIdx / blockIdx
       所以 kernel 保留 pos/gid/id 参数(CPU 对拍时由 host 的二重循环传值),
       nvcc 模式下由 kernel 自己第一行调 gpuGridId1D()/gpuGridId2D() 覆盖,
       host 侧只传 gpuGridIdPlaceholder1D()/2D() 这个占位值。 */
    #ifdef __CUDACC__
    GPU_D unsigned int gpuGridId1D(void){ return blockIdx.x*blockDim.x + threadIdx.x; }
    GPU_D uint2 gpuGridId2D(void){
        return make_uint2(blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y);
    }
    #endif
    /* host 侧占位值: nvcc 模式下 kernel 第一行就会覆盖掉它, 数值本身没有意义 */
    static inline unsigned int gpuGridIdPlaceholder1D(void){ return 0u; }
    static inline uint2 gpuGridIdPlaceholder2D(void){ return make_uint2(0u, 0u); }
#endif  /* GPU_KERNEL_UNIT(两种模式共用块) */

#ifndef INFINITY
#define INFINITY (1.0f/0.0f)
#endif

/* ------------------------------------------------------------------ */
/* 原 RTX.metal 里的 struct RTXV / projection / camera / metalBVHNode */
/* ------------------------------------------------------------------ */
struct CudaRTXV{                       // 原 struct RTXV (Metal) / MetalRTXV (main.mm)
    float x, y, z;
    bool  intersect = true;
    float OM = 0;
};

struct CudaProjection{                 // 原 struct projection (RTX.metal) / rtx_metal_projection (main.mm)
    float x, y, z;
    int   c1, c2, c3, c4;
    float r;
    float reflection  = 0.5;

    float w_velocity  = 2 * 3.1415926 / 86400;
    float rotation    = 0;
    float inclination = 0;
};

struct CudaCamera{                     // 原 struct camera (RTX.metal) / metalcamera (main.mm)
    float x, y, z;
    float yaw, pitch;
    float fov;
    float vx, vy, vz;
};

struct CudaBVHNode{                    // 原 struct metalBVHNode
    CudaRTXV C1, C2;
    long long indices[7];              // ← Metal 的 long(8B) 必须写成 long long
    int left  = -1;
    int right = -1;
    int count = 0;
};

/* 布局检查: host 和 device 用的是同一个定义, 这两处断言保证不会被编译器改布局 */
static_assert(sizeof(CudaRTXV)       == 20,  "CudaRTXV 布局变了 (应为 3*float + bool + float = 20B)");
static_assert(sizeof(CudaProjection) == 48,  "CudaProjection 布局变了 (应为 12 * 4B = 48B)");
static_assert(sizeof(CudaCamera)     == 36,  "CudaCamera 布局变了 (应为 9 * 4B = 36B)");
static_assert(sizeof(CudaBVHNode)    == 112, "CudaBVHNode 布局变了 (应为 2*20 + 7*8 + 3*4 -> 112B)");

/* host 端 main.cpp 里保留原来的名字, 这样 RTXRenderMetal() 内部代码不用改 */
typedef CudaRTXV        MetalRTXV;
typedef CudaProjection  rtx_metal_projection;
typedef CudaCamera      metalcamera;
typedef CudaBVHNode     metalBVHNode;

#endif /* GPU_COMMON_H */
