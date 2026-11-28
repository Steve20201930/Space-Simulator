//
//  gpu_cuda.h
//  Space Simulator CUDA
//
//  替代原来的 #import <Metal/Metal.h>。
//  目标: 让 main.cpp 里调用 GPU 的那两个函数(position_movement_METAL /
//  RTXRenderMetal)改动越少越好, 所以这里刻意做成和 Metal 那套 API 一一对应的形状:
//
//   Metal                                              CUDA 版
//   -------------------------------------------------  ------------------------------------------------------
//   MTLCreateSystemDefaultDevice() + newCommandQueue   gpuInit()                       (默认流, 不需要 queue)
//   newDefaultLibrary / newFunctionWithName / PSO      不需要(nvcc 直接把 kernel 编进可执行文件)
//   id<MTLBuffer> = [device newBufferWithLength:]      GpuBuffer* = gpuBufferNew(bytes, kind)
//   id<MTLBuffer> = [device newBufferWithBytes:src]    GpuBuffer* = gpuBufferNewWithBytes(src, bytes, kind)
//   buf.contents                                       gpuBufferContents(buf)          (kind=SHARED 时可直接读写)
//   [enc setBuffer:b offset:0 atIndex:i]               gpuSetBuffer(&enc, i, b)
//   [enc setBytes:&v length:sizeof(v) atIndex:i]       gpuSetBytes(&enc, i, &v, sizeof(v))
//   [enc dispatchThreads:g threadsPerThreadgroup:t]    gpuDispatchComp(&enc,N) / gpuDispatchCorrectVector(&enc,w,h) /
//                                                      gpuDispatchRTXPath(&enc,w,h)
//   [cmd commit] + [cmd waitUntilCompleted]            gpuCommit()
//
#ifndef GPU_CUDA_H
#define GPU_CUDA_H

#include <cstddef>

/* buffer 类型, 对应 MTLResourceStorageMode 的两个常用档 */
typedef enum GpuMemKind{
    GPU_MEM_DEVICE = 0,   // 纯显存(相当于 MTLResourceStorageModePrivate): 只能用 gpuUpload/gpuDownload 进出
    GPU_MEM_SHARED = 1    // host/device 共享同一块内存(相当于 MTLResourceStorageModeShared): contents 可直接读写
} GpuMemKind;

/* 相当于 id<MTLBuffer> */
typedef struct GpuBuffer{
    void*       dev;      // device 侧指针(kernel 参数用这个)
    void*       host;     // host 侧指针(= buf.contents); 只有 GPU_MEM_SHARED 才有值,
                          // GPU_MEM_DEVICE 是纯显存, host == nullptr, 只能用 gpuUpload/gpuDownload 进出
    size_t      size;
    int         kind;
} GpuBuffer;

/* 相当于 id<MTLComputeCommandEncoder>, atIndex: 的槽位数 */
#define GPU_ARG_SLOTS 32

/* block 大小(≈ Metal 里自己算的 threadsPerThreadgroup):
   comp       一维 256 线程/block           (原来: pipeline.maxTotalThreadsPerThreadgroup)
   其它三个    16x16 = 256 线程/block        (原来: MIN(sqrt(maxTotal...), sqrt(w*h)))
   CUDA 里 block 大小固定写死更稳, 因为 grid 是向上取整的, kernel 里都有越界 return。 */
#define GPU_COMP_BLOCK     256
#define GPU_SHADER_BLOCK   16
#define GPU_RTX_BLOCK      16

typedef struct GpuEncoder{
    GpuBuffer* buf[GPU_ARG_SLOTS];      // setBuffer: 进来的显存 buffer
    const void* val[GPU_ARG_SLOTS];     // setBytes:  进来的常量(放在 pinned 暂存区, GPU 直接读)
    size_t     valSize[GPU_ARG_SLOTS];
} GpuEncoder;

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------- 初始化 / 关闭 ---------------- */
int   gpuInit(void);                    // 返回 1 = GPU 可用, 0 = 不可用(调用处要能退回 CPU 路径)
void  gpuQuit(void);
const char* gpuDeviceName(void);
int   gpuIsReady(void);
/* 对应 pipeline.maxTotalThreadsPerThreadgroup, 只是用来给调用处算 threadgroup 大小 */
int   gpuMaxThreadsPerThreadgroup(void);

/* ---------------- buffer(≈ id<MTLBuffer>) ---------------- */
GpuBuffer* gpuBufferNew(size_t bytes, GpuMemKind kind);                  // 新 buffer 一律清零(Metal 也是这个行为,
                                                                          // RTXPath 里 R[] += ... 依赖初始为 0)
GpuBuffer* gpuBufferNewWithBytes(const void* src, size_t bytes, GpuMemKind kind);
void  gpuBufferFree(GpuBuffer* buf);
void  gpuBufferResize(GpuBuffer** buf, size_t bytes, GpuMemKind kind);   // 容量不够时重建(保留原内容无意义, 直接新建)
void* gpuBufferContents(GpuBuffer* buf);                                  // ≈ buf.contents
void  gpuUpload (GpuBuffer* buf, const void* src, size_t bytes);          // host → device
void  gpuDownload(GpuBuffer* buf, void* dst, size_t bytes);               // device → host

/* ---------------- encoder(≈ id<MTLComputeCommandEncoder>) ---------------- */
void  gpuEncoderInit(GpuEncoder* enc);                                    // 全部槽位清空
void  gpuSetBuffer(GpuEncoder* enc, int indexAt, GpuBuffer* buf);         // ≈ [enc setBuffer:offset:atIndex:]
void  gpuSetBytes (GpuEncoder* enc, int indexAt, const void* src, size_t bytes); // ≈ [enc setBytes:length:atIndex:]

/* ---------------- 提交 ---------------- */
void  gpuCommit(void);                                                    // ≈ [cmd commit] + waitUntilCompleted

/* ---------------- kernel 启动(实现在 kernel_comp.cu / kernel_rtx.cu) ----------------
   indexAt 的编号和原来 GPU.metal / RTX.metal 里 [[buffer(n)]] 的编号完全一致,
   调用处 setBuffer/setBytes 的顺序可以和原来一样照抄。 */
void  gpuDispatchComp(GpuEncoder* enc, long long N);                                  // GPU.metal: comp
void  gpuDispatchShader(GpuEncoder* enc, unsigned int w, unsigned int h);              // GPU.metal: shader
void  gpuDispatchCorrectVector(GpuEncoder* enc, int width, int height);                // RTX.metal: correct_vector
void  gpuDispatchRTXPath(GpuEncoder* enc, int width, int height);                      // RTX.metal: RTXPath

/* ---------------- 内部用: 取某个槽位对应的 device 指针 ---------------- */
void* gpuArgPtr(GpuEncoder* enc, int indexAt);

#ifdef __cplusplus
}
#endif

#endif /* GPU_CUDA_H */
