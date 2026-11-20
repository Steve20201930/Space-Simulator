//
//  RTXMetalKernel.metal
//  Space Simulator App
//
//  Created by Steve William on 11/20/26.
//

#include <metal_stdlib>
using namespace metal;

// Matches the 16-byte-aligned CPU POD structures in RTXRenderMetal_v2.inc.
struct RTXMetalBody {
    float4 centerRadius;   // xyz: position relative to camera, w: radius
    float4 colorEmissive;  // rgb: 0..1, w: self-emissive flag
};

struct RTXMetalNode {
    float4 bmin;
    float4 bmax;
    int left;
    int right;
    int bodyIndex; // -1 means an internal BVH node
    int padding;
};

struct RTXMetalParams {
    float4 right;
    float4 up;
    float4 forward;
    float4 sunPosition; // camera-relative position of the light center
    float focal;
    uint width;
    uint height;
    uint bodyCount;
    int sunIndex;        // -1 means no sun
    float ambient;
    uint enableShadows;
    uint padding;
};

// Slab test, also handles zero ray-direction components.
static inline bool intersectsBox(float3 origin, float3 ray, float3 lo, float3 hi, float maxT) {
    float nearT = 0.0f;
    float farT = maxT;
    for (int k = 0; k < 3; ++k) {
        if (fabs(ray[k]) < 1e-10f) {
            if (origin[k] < lo[k] || origin[k] > hi[k]) return false;
        } else {
            float a = (lo[k] - origin[k]) / ray[k];
            float b = (hi[k] - origin[k]) / ray[k];
            if (a > b) { float temp = a; a = b; b = temp; }
            nearT = max(nearT, a);
            farT = min(farT, b);
            if (nearT > farT) return false;
        }
    }
    return true;
}

// Same geometry as OM = OF - sqrt(r*r - FP*FP), without acos/asin.
// Returns the first positive ray hit; handles origins inside spheres.
static inline float intersectSphere(float3 origin, float3 ray, float3 center, float radius, float maxT) {
    float3 d = center - origin;
    float OF = dot(d, ray);
    float3 perp = d - OF * ray;
    float FP2 = dot(perp, perp);
    float rr = radius * radius;
    if (FP2 > rr) return -1.0f;
    float MF = sqrt(max(rr - FP2, 0.0f));
    float t = OF - MF;
    if (t <= 0.0f) t = OF + MF;
    return (t > 0.0f && t < maxT) ? t : -1.0f;
}

struct RTXHit {
    float distance;
    int bodyIndex;
};

static inline RTXHit traceNearest(float3 origin,
                                  float3 ray,
                                  device const RTXMetalBody *bodies,
                                  device const RTXMetalNode *nodes,
                                  float maxDistance,
                                  int excludeIndex) {
    RTXHit best = {maxDistance, -1};
    int stack[32];
    int count = 0;
    stack[count++] = 0;
    while (count > 0) {
        int nodeIndex = stack[--count];
        RTXMetalNode node = nodes[nodeIndex];
        if (!intersectsBox(origin, ray, node.bmin.xyz, node.bmax.xyz, best.distance)) continue;
        if (node.bodyIndex >= 0) {
            int idx = node.bodyIndex;
            if (idx == excludeIndex) continue;
            RTXMetalBody b = bodies[idx];
            float t = intersectSphere(origin, ray, b.centerRadius.xyz,
                                      b.centerRadius.w, best.distance);
            if (t > 0.0f) {
                best.distance = t;
                best.bodyIndex = idx;
            }
        } else {
            // The CPU builds a balanced median-split BVH, depth << 32.
            stack[count++] = node.left;
            stack[count++] = node.right;
        }
    }
    return best;
}

// Shadow ray only needs to know whether *any* non-emissive blocker is present.
static inline bool traceShadow(float3 origin,
                               float3 ray,
                               float maxDistance,
                               device const RTXMetalBody *bodies,
                               device const RTXMetalNode *nodes,
                               int excludeIndex,
                               int sunIndex) {
    int stack[32];
    int count = 0;
    stack[count++] = 0;
    while (count > 0) {
        RTXMetalNode node = nodes[stack[--count]];
        if (!intersectsBox(origin, ray, node.bmin.xyz, node.bmax.xyz, maxDistance)) continue;
        if (node.bodyIndex >= 0) {
            int idx = node.bodyIndex;
            if (idx == excludeIndex || idx == sunIndex) continue;
            RTXMetalBody b = bodies[idx];
            if (intersectSphere(origin, ray, b.centerRadius.xyz,
                                b.centerRadius.w, maxDistance) > 0.0f) {
                return true;
            }
        } else {
            stack[count++] = node.left;
            stack[count++] = node.right;
        }
    }
    return false;
}

kernel void RTXMetalKernel(device const RTXMetalBody *bodies [[buffer(0)]],
                           device const RTXMetalNode *nodes [[buffer(1)]],
                           device uint *pixels [[buffer(2)]],
                           constant RTXMetalParams &p [[buffer(3)]],
                           uint2 tid [[thread_position_in_grid]]) {
    if (tid.x >= p.width || tid.y >= p.height) return;
    uint pixelIndex = tid.y * p.width + tid.x;

    float sx = ((float(tid.x) + 0.5f) - float(p.width) * 0.5f);
    float sy = float(p.height) * 0.5f - (float(tid.y) + 0.5f);
    float3 rd = normalize(sx * p.right.xyz + sy * p.up.xyz + p.focal * p.forward.xyz);

    RTXHit hit = traceNearest(float3(0.0f), rd, bodies, nodes, 1e30f, -1);
    if (hit.bodyIndex < 0) {
        pixels[pixelIndex] = 0xFF000000u; // opaque black in RGBA32 byte layout
        return;
    }

    RTXMetalBody body = bodies[hit.bodyIndex];
    float illumination = 1.0f;
    if (body.colorEmissive.w < 0.5f) {
        illumination = p.ambient;
        if (p.sunIndex >= 0) {
            float3 M = rd * hit.distance;
            float3 normal = normalize(M - body.centerRadius.xyz);
            float3 toSun = p.sunPosition.xyz - M;
            float sunDistance = length(toSun);
            if (sunDistance > 0.0f) {
                float3 L = toSun / sunDistance;
                float lambert = max(dot(normal, L), 0.0f);
                if (lambert > 0.0f) {
                    bool occluded = false;
                    if (p.enableShadows != 0u) {
                        float eps = max(body.centerRadius.w * 1e-5f, 1.0f);
                        float3 shiftedM = M + normal * eps;
                        float3 shiftedToSun = p.sunPosition.xyz - shiftedM;
                        float shiftedDistance = length(shiftedToSun);
                        if (shiftedDistance > 0.0f) {
                            occluded = traceShadow(shiftedM,
                                                   shiftedToSun / shiftedDistance,
                                                   shiftedDistance,
                                                   bodies, nodes, hit.bodyIndex, p.sunIndex);
                        }
                    }
                    float fullI = p.ambient + (1.0f - p.ambient) * lambert;

                    illumination = occluded ? fullI * 0.1f : fullI;
                }
            }
        }
    }

    float3 rgb = clamp(body.colorEmissive.xyz * illumination, 0.0f, 1.0f);
    uint r = uint(rgb.r * 255.0f + 0.5f);
    uint g = uint(rgb.g * 255.0f + 0.5f);
    uint b = uint(rgb.b * 255.0f + 0.5f);
    // On Apple Silicon little-endian: memory bytes are R,G,B,A; SDL_PIXELFORMAT_RGBA32.
    pixels[pixelIndex] = r | (g << 8) | (b << 16) | 0xFF000000u;
}
