//
//  gpu_cuda.cu
//  Space Simulator CUDA
//
//  原来 main.mm 里这几行 Metal 初始化:
//
//      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
//      id<MTLCommandQueue> queuea = [device newCommandQueue];
//      id<MTLLibrary> lib = [device newDefaultLibrary];
//      id<MTLFunction> func = [lib newFunctionWithName:@"comp"];
//      id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:nil];
//      ... correct_vectorPSO / RTXPathPSO(maxCallStackDepth = 32)
//
//  在 CUDA 里全部由 gpuInit() 取代:
//    - kernel 由 nvcc 直接编进可执行文件, 没有 runtime 编译 / PSO 这一步;
//    - Metal 的 commandQueue/commandBuffer 对应 CUDA 的 default stream, gpuCommit() 就是同步点;
//    - Metal 的 maxCallStackDepth = 32(给递归 BVH 遍历用的)对应 cudaLimitStackSize。
//
#include "gpu_cuda.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

/* ------------------------------------------------------------------ */
/*                          CUDA 真实实现                              */
/* ------------------------------------------------------------------ */

#define GPU_CHECK(call)                                                        \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if(err__ != cudaSuccess){                                              \
            fprintf(stderr, "[CUDA] %s:%d  %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err__));                                \
            gpuReady = 0;                                                      \
            return;                                                            \
        }                                                                      \
    } while(0)

#define GPU_CHECK_R(call, ret)                                                 \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if(err__ != cudaSuccess){                                              \
            fprintf(stderr, "[CUDA] %s:%d  %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err__));                                \
            gpuReady = 0;                                                      \
            return (ret);                                                      \
        }                                                                      \
    } while(0)

/* setBytes: 用的 pinned 暂存区, GPU 直接读, 不需要每帧 memcpy */
#define GPU_BYTESLOT_SIZE 512
static char*  g_byteslotHost = nullptr;   // host 侧地址(写)
static char*  g_byteslotDev  = nullptr;   // device 侧地址(kernel 参数读)

static int    gpuReady = 0;
static int    g_maxThreads = GPU_RTX_BLOCK * GPU_RTX_BLOCK;
static char   g_deviceName[256] = "none";

int gpuInit(void){
    if(gpuReady)return 1;

    int count = 0;
    if(cudaGetDeviceCount(&count) != cudaSuccess || count <= 0){
        fprintf(stderr, "[CUDA] 没找到可用的 CUDA 设备, 走 CPU 渲染路径\n");
        gpuReady = 0;
        return 0;
    }

    /* 选显存最大的那块卡(多卡机器上核显/亮机卡容易被选中) */
    int best = 0;
    size_t bestMem = 0;
    for(int i = 0;i < count;i++){
        cudaDeviceProp p;
        if(cudaGetDeviceProperties(&p, i) != cudaSuccess)continue;
        if(p.major < 5)continue;                       // Maxwell 以前不支持需要的特性
        if(p.totalGlobalMem > bestMem){ bestMem = p.totalGlobalMem; best = i; }
    }
    if(cudaSetDevice(best) != cudaSuccess){ gpuReady = 0; return 0; }

    cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop, best) != cudaSuccess){ gpuReady = 0; return 0; }

    /* ≈ [device newComputePipelineStateWithDescriptor: maxCallStackDepth:32]
       RTXPath 里 BVH 遍历是递归的, CUDA 默认栈(1KB)不够, 这里调大。 */
    size_t stackBytes = 32 * 1024;
    if(cudaDeviceSetLimit(cudaLimitStackSize, stackBytes) != cudaSuccess){
        fprintf(stderr, "[CUDA] 设置 stack size 失败, BVH 递归可能溢出\n");
    }

    /* 相当于 newCommandQueue: CUDA 用 default stream, 不需要显式创建 */
    if(cudaFree(0) != cudaSuccess){ gpuReady = 0; return 0; }

    /* setBytes: 暂存区(pinned + mapped, host/device 共用) */
    if(cudaHostAlloc(&g_byteslotHost, GPU_ARG_SLOTS * GPU_BYTESLOT_SIZE,
                     cudaHostAllocMapped) != cudaSuccess){
        gpuReady = 0; return 0;
    }
    if(cudaHostGetDevicePointer((void**)&g_byteslotDev, g_byteslotHost, 0) != cudaSuccess){
        gpuReady = 0; return 0;
    }

    g_maxThreads = prop.maxThreadsPerBlock;
    snprintf(g_deviceName, sizeof(g_deviceName), "%s (sm_%d%d, %zu MB)",
             prop.name, prop.major, prop.minor, (size_t)(prop.totalGlobalMem >> 20));

    gpuReady = 1;
    printf("[CUDA] device = %s\n", g_deviceName);
    return 1;
}

void gpuQuit(void){
    if(g_byteslotHost){ cudaFreeHost(g_byteslotHost); g_byteslotHost = nullptr; g_byteslotDev = nullptr; }
    cudaDeviceReset();
    gpuReady = 0;
}

const char* gpuDeviceName(void){ return g_deviceName; }
int  gpuIsReady(void){ return gpuReady; }
int  gpuMaxThreadsPerThreadgroup(void){ return g_maxThreads; }

/* ------------------------- buffer ------------------------- */
GpuBuffer* gpuBufferNew(size_t bytes, GpuMemKind kind){
    if(bytes == 0)bytes = 4;                       // 0 字节 cudaMalloc 会返回 nullptr
    GpuBuffer* b = (GpuBuffer*)calloc(1, sizeof(GpuBuffer));
    if(!b)return nullptr;
    b->size = bytes;
    b->kind = (int)kind;

    if(kind == GPU_MEM_SHARED){
        /* ≈ MTLResourceStorageModeShared: host 和 device 指向同一块物理内存,
           所以 main.cpp 里 memcpy(xxx.contents, ...) / (float*)xxx.contents 这种写法可以原样保留 */
        if(cudaHostAlloc(&b->host, bytes, cudaHostAllocMapped) != cudaSuccess){ free(b); return nullptr; }
        if(cudaHostGetDevicePointer(&b->dev, b->host, 0) != cudaSuccess){ cudaFreeHost(b->host); free(b); return nullptr; }
        memset(b->host, 0, bytes);
    }else{
        /* ≈ MTLResourceStorageModePrivate: 纯显存, 大块只读数据(贴图 / 光线向量)用这个 */
        if(cudaMalloc(&b->dev, bytes) != cudaSuccess){ free(b); return nullptr; }
        b->host = nullptr;
        /* Metal 新 buffer 是可读为 0 的, CUDA cudaMalloc 不清零, 这里必须自己清:
           RTXPath 里 R[index] += ... 依赖这个初始 0 */
        if(cudaMemset(b->dev, 0, bytes) != cudaSuccess){ cudaFree(b->dev); free(b); return nullptr; }
    }
    return b;
}

GpuBuffer* gpuBufferNewWithBytes(const void* src, size_t bytes, GpuMemKind kind){
    GpuBuffer* b = gpuBufferNew(bytes, kind);
    if(!b)return nullptr;
    if(src){
        if(kind == GPU_MEM_SHARED)memcpy(b->host, src, bytes);
        else if(cudaMemcpy(b->dev, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess){ gpuBufferFree(b); return nullptr; }
    }
    return b;
}

void gpuBufferFree(GpuBuffer* buf){
    if(!buf)return;
    if(buf->kind == GPU_MEM_SHARED){
        if(buf->host)cudaFreeHost(buf->host);
    }else{
        if(buf->dev)cudaFree(buf->dev);
    }
    free(buf);
}

void gpuBufferResize(GpuBuffer** buf, size_t bytes, GpuMemKind kind){
    if(*buf && (*buf)->size >= bytes)return;
    if(*buf)gpuBufferFree(*buf);
    *buf = gpuBufferNew(bytes, kind);
}

void* gpuBufferContents(GpuBuffer* buf){
    if(!buf)return nullptr;
    return buf->host;                              // DEVICE 类型没有 host 指针, 返回 nullptr
}

void gpuUpload(GpuBuffer* buf, const void* src, size_t bytes){
    if(!buf || !src)return;
    if(bytes > buf->size)bytes = buf->size;
    if(buf->kind == GPU_MEM_SHARED)memcpy(buf->host, src, bytes);
    else cudaMemcpy(buf->dev, src, bytes, cudaMemcpyHostToDevice);
}

void gpuDownload(GpuBuffer* buf, void* dst, size_t bytes){
    if(!buf || !dst)return;
    if(bytes > buf->size)bytes = buf->size;
    if(buf->kind == GPU_MEM_SHARED){ gpuCommit(); memcpy(dst, buf->host, bytes); }
    else cudaMemcpy(dst, buf->dev, bytes, cudaMemcpyDeviceToHost);
}

/* ------------------------- encoder ------------------------- */
void gpuEncoderInit(GpuEncoder* enc){
    if(!enc)return;
    for(int i = 0;i < GPU_ARG_SLOTS;i++){ enc->buf[i] = nullptr; enc->val[i] = nullptr; enc->valSize[i] = 0; }
}

void gpuSetBuffer(GpuEncoder* enc, int indexAt, GpuBuffer* buf){
    if(!enc || indexAt < 0 || indexAt >= GPU_ARG_SLOTS)return;
    enc->buf[indexAt] = buf;
    enc->val[indexAt] = nullptr;
}

void gpuSetBytes(GpuEncoder* enc, int indexAt, const void* src, size_t bytes){
    if(!enc || indexAt < 0 || indexAt >= GPU_ARG_SLOTS || !src)return;
    if(bytes > GPU_BYTESLOT_SIZE)bytes = GPU_BYTESLOT_SIZE;
    memcpy(g_byteslotHost + (size_t)indexAt * GPU_BYTESLOT_SIZE, src, bytes);
    enc->val[indexAt] = (const void*)(g_byteslotDev + (size_t)indexAt * GPU_BYTESLOT_SIZE);
    enc->valSize[indexAt] = bytes;
}

void* gpuArgPtr(GpuEncoder* enc, int indexAt){
    if(!enc || indexAt < 0 || indexAt >= GPU_ARG_SLOTS)return nullptr;
    if(enc->buf[indexAt])return enc->buf[indexAt]->dev;
    return (void*)enc->val[indexAt];
}

void gpuCommit(void){
    cudaDeviceSynchronize();
}

#else  /* !__CUDACC__ ------------------------------------------------ */
/* ------------------------------------------------------------------ */
/*  CPU 仿真实现(mac 上数值对拍用; Windows 用 nvcc 编译时不会进这里) */
/*  device 指针 == host 指针, 所有内存都是普通 malloc。                */
/* ------------------------------------------------------------------ */
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define GPU_BYTESLOT_SIZE 512
static char* g_byteslot[GPU_ARG_SLOTS];
static int   gpuReady = 0;

int gpuInit(void){
    if(gpuReady)return 1;
    for(int i = 0;i < GPU_ARG_SLOTS;i++)g_byteslot[i] = (char*)calloc(1, GPU_BYTESLOT_SIZE);
    gpuReady = 1;
    printf("[CPU-EMU] CUDA kernel 以普通 C++ 方式在 CPU 上逐线程执行(对拍模式)\n");
    return 1;
}
void gpuQuit(void){
    for(int i = 0;i < GPU_ARG_SLOTS;i++){ free(g_byteslot[i]); g_byteslot[i] = nullptr; }
    gpuReady = 0;
}
const char* gpuDeviceName(void){ return "cpu-emulation"; }
int  gpuIsReady(void){ return gpuReady; }
int  gpuMaxThreadsPerThreadgroup(void){ return 256; }

GpuBuffer* gpuBufferNew(size_t bytes, GpuMemKind kind){
    if(bytes == 0)bytes = 4;
    GpuBuffer* b = (GpuBuffer*)calloc(1, sizeof(GpuBuffer));
    b->size = bytes; b->kind = (int)kind;
    b->host = (char*)calloc(1, bytes);
    b->dev  = b->host;                             // CPU 模式下 device == host
    return b;
}
GpuBuffer* gpuBufferNewWithBytes(const void* src, size_t bytes, GpuMemKind kind){
    GpuBuffer* b = gpuBufferNew(bytes, kind);
    if(b && src)memcpy(b->host, src, bytes > b->size ? b->size : bytes);
    return b;
}
void gpuBufferFree(GpuBuffer* buf){ if(!buf)return; free(buf->host); free(buf); }
void gpuBufferResize(GpuBuffer** buf, size_t bytes, GpuMemKind kind){
    if(*buf && (*buf)->size >= bytes)return;
    if(*buf)gpuBufferFree(*buf);
    *buf = gpuBufferNew(bytes, kind);
}
void* gpuBufferContents(GpuBuffer* buf){ return buf ? buf->host : nullptr; }
void gpuUpload(GpuBuffer* buf, const void* src, size_t bytes){
    if(!buf || !src)return; if(bytes > buf->size)bytes = buf->size; memcpy(buf->host, src, bytes);
}
void gpuDownload(GpuBuffer* buf, void* dst, size_t bytes){
    if(!buf || !dst)return; if(bytes > buf->size)bytes = buf->size; memcpy(dst, buf->host, bytes);
}
void gpuEncoderInit(GpuEncoder* enc){
    for(int i = 0;i < GPU_ARG_SLOTS;i++){ enc->buf[i] = nullptr; enc->val[i] = nullptr; enc->valSize[i] = 0; }
}
void gpuSetBuffer(GpuEncoder* enc, int indexAt, GpuBuffer* buf){
    if(indexAt < 0 || indexAt >= GPU_ARG_SLOTS)return;
    enc->buf[indexAt] = buf; enc->val[indexAt] = nullptr;
}
void gpuSetBytes(GpuEncoder* enc, int indexAt, const void* src, size_t bytes){
    if(indexAt < 0 || indexAt >= GPU_ARG_SLOTS || !src)return;
    if(bytes > GPU_BYTESLOT_SIZE)bytes = GPU_BYTESLOT_SIZE;
    memcpy(g_byteslot[indexAt], src, bytes);
    enc->val[indexAt] = g_byteslot[indexAt];
    enc->valSize[indexAt] = bytes;
}
void* gpuArgPtr(GpuEncoder* enc, int indexAt){
    if(indexAt < 0 || indexAt >= GPU_ARG_SLOTS)return nullptr;
    if(enc->buf[indexAt])return enc->buf[indexAt]->dev;
    return (void*)enc->val[indexAt];
}
void gpuCommit(void){}
#endif /* __CUDACC__ */
