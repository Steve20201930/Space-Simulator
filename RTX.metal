//
//  RTX.metal
//  Space Simulator App
//
//  Created by Steve William on 11/21/26.
//

#include <metal_stdlib>
using namespace metal;

struct RTXV{
    float x, y, z;
    bool intersect = true;
    float OM = 0;
};
struct projection{
    float x,y,z;
    int c1,c2,c3,c4;
    float r;
};
struct camera{
    float x,y,z;
    float yaw,pitch;
    float fov;
    float vx,vy,vz;
};

kernel void correct_vector(
                      device const RTXV *OriginVector,
                      device const RTXV *Matrix,
                      device const int *Width,
                      device RTXV *Result,
                      uint2 pos [[thread_position_in_grid]]
                      )
{
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
float VectorDotMul(RTXV A, RTXV B){
    return A.x * B.x + A.y * B.y + A.z * B.z;
}
float VectorLength(RTXV A){
    return fast::sqrt(A.x * A.x + A.y * A.y + A.z * A.z);
}
RTXV VectorUnit(RTXV A){
    float abs = VectorLength(A);
    RTXV Result = {
        A.x / abs, A.y / abs, A.z / abs
    };
    return Result;
}
RTXV VectorMul(RTXV A, float k){
    return {A.x * k, A.y * k, A.z * k};
}
RTXV VectorPlus(RTXV A, RTXV B){
    return {A.x + B.x, A.y + B.y, A.z + B.z};
}
RTXV RTXVectorCalcM(RTXV v, RTXV O, RTXV P, float radius, float VLength){
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
        return {0,0,0,false};
    }
    
    RTXV OM = VectorMul(vUnit, OE - fast::sqrt(delta));
    M = VectorPlus(OM, O);
    M.OM = OE - fast::sqrt(delta);
    return M;
}
RTXV RTXAngularCalcM(RTXV v, RTXV O, RTXV P, float radius, float VLength){
    RTXV M;
    float OP_v = (P.x - O.x) * v.x + (P.y - O.y) * v.y + (P.z - O.z) * v.z;
    float OP = fast::sqrt(
                          fast::pow(P.x - O.x,2) +
                          fast::pow(P.y - O.y,2) +
                          fast::pow(P.z - O.z,2));
    float real_angle = fast::acos(clamp(OP_v / (OP * VLength),-1.0,1.0));
    float tan_angle = fast::asin( radius / OP );
    if(real_angle > tan_angle){
        return {0,0,0,false};
    }
    float OM = OP * fast::cos(real_angle) - fast::sqrt(pow(radius, 2) - fast::pow(OP * fast::sin(real_angle), 2));
    M.x = v.x * OM / VLength + O.x;
    M.y = v.y * OM / VLength + O.y;
    M.z = v.z * OM / VLength + O.z;
    M.OM = OM;
    return M;
}
struct metalBVHNode{
    RTXV C1, C2;
    long indices[7];
    int left = -1, right = -1;
    int count = 0;
};
bool RTXBVHRayAABB(device const metalBVHNode &nextBVH, thread const RTXV &vector, thread const RTXV &camera){
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
void RTXBVHRayForIntersect(device const projection *parti, device const metalBVHNode &rBVH, const RTXV vector, thread const RTXV &camera, device const metalBVHNode *BVHCollector, const int BVHCollectorSize, thread float &nearestOM, thread int &nearestPlanet, thread RTXV &M, const float vectorLength){
    if(rBVH.left == -1 && rBVH.right == -1){
        for(int i = 0;i < rBVH.count;i++){
            long p = rBVH.indices[i];
            RTXV Mcandi = RTXVectorCalcM(vector, {camera.x, camera.y, camera.z}, {parti[p].x, parti[p].y, parti[p].z}, parti[p].r, vectorLength);
            if(Mcandi.intersect == false){
                continue;
            }else{
                if(Mcandi.OM > 0 && Mcandi.OM < nearestOM){
                    nearestOM = Mcandi.OM;
                    nearestPlanet = p;
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
void RTXBVHRayForShadow(device const projection *parti, device const metalBVHNode &rBVH, const RTXV vector, thread const RTXV &camera, device const metalBVHNode *BVHCollector, const int BVHCollectorSize, thread const int nearestPlanet, thread const RTXV &M, thread const RTXV &MS, thread const float &MSLength, thread float &k, thread int sunIndex){
    if(k == 0.02)return ;
    if(rBVH.left == -1 && rBVH.right == -1){
        for(int i = 0;i < rBVH.count;i++){
            long p = rBVH.indices[i];
            if(p == nearestPlanet || p == sunIndex)continue;
            RTXV Shadow = RTXVectorCalcM(MS, M, {parti[p].x, parti[p].y, parti[p].z}, parti[p].r, MSLength);
            if(Shadow.intersect && Shadow.OM < MSLength && Shadow.OM > 0){
                k = 0.02;
                break;
            }
        }
        return ;
    }
    if(rBVH.left != -1){
        if(RTXBVHRayAABB(BVHCollector[rBVH.left], vector, camera)){
            RTXBVHRayForShadow(parti, BVHCollector[rBVH.left], vector, camera, BVHCollector, BVHCollectorSize, nearestPlanet, M, MS, MSLength, k, sunIndex);
        }
    }
    if(rBVH.right != -1){
        if(RTXBVHRayAABB(BVHCollector[rBVH.right], vector, camera)){
            RTXBVHRayForShadow(parti, BVHCollector[rBVH.right], vector, camera, BVHCollector, BVHCollectorSize, nearestPlanet, M, MS, MSLength, k, sunIndex);
        }
    }
    return ;
}
void StellarHaloDetect(device const projection *parti, device const int *suns, long suns_size, const RTXV vector, const RTXV camera, device uint8_t &R, device uint8_t &G, device uint8_t &B){
    for(long i = 0;i < suns_size;i++){
        long p = suns[i];
        
        //Check if intersect with halo of the stellar
        RTXV SC = {
            parti[p].x - camera.x, parti[p].y - camera.y, parti[p].z - camera.z
        };
        float SCLength = VectorLength(SC);
        float VLength = VectorLength(vector);
        RTXV VUnit = VectorMul(vector, 1/VLength);
        float CTLength = VectorDotMul(SC, VUnit);
        if(CTLength <= 0)continue;
        float STLength = sqrt(max(0.0,SCLength * SCLength - CTLength * CTLength));
        float I = 0.0;
        if(STLength > parti[p].r && STLength < 8 * parti[p].r){
            float q = STLength / parti[p].r;
            I += (1 - q * q / 64) / (0.8 * q * q + 0.2);
        }else if(STLength <= parti[p].r && STLength > 0){
            float q = STLength / parti[p].r;
            I = 1.5 - 0.5 * q * q;
        }
        R = parti[p].c1 * I;
        G = parti[p].c2 * I;
        B = parti[p].c3 * I;
    }
}
kernel void RTXPath(
                    device const int *Width,
                    device const RTXV *RTXVector,
                    device const float *RTXVectorLength,
                    device const int *suns,
                    device const long *suns_size,
                    device const projection *parti,
                    device const long *parti_size,
                    device const camera *cam,
                    
                    device const metalBVHNode *metalBVHCollector,
                    device const int *metalBVHCollectorSize,
                    
                    device uint8_t *R,
                    device uint8_t *G,
                    device uint8_t *B,
                    uint2 pos [[thread_position_in_grid]]
                    )
{
    uint index = pos.y * *Width + pos.x;
    
    /* Execute BVH & Synchronize Calculate intersect*/
    float nearestOM = INFINITY;
    int nearestPlanet = -1;
    RTXV M;
    RTXBVHRayForIntersect(parti, metalBVHCollector[0], RTXVector[index], {cam->x, cam->y, cam->z}, metalBVHCollector, *metalBVHCollectorSize, nearestOM, nearestPlanet, M, RTXVectorLength[index]);
    
    bool isStellar = false;
    for(long i = 0;i < *suns_size;i++){
        if(suns[i] == nearestPlanet){
            isStellar = true;
            break;
        }
    }
    if(nearestPlanet == -1 || isStellar){
        StellarHaloDetect(parti, suns, *suns_size, RTXVector[index], {cam->x, cam->y, cam->z}, R[index], G[index], B[index]);
        return ;
    }
    /* Light Magnitude & Shadow Calc*/
    RTXV Npm = {
        (M.x - parti[nearestPlanet].x)/parti[nearestPlanet].r,
        (M.y - parti[nearestPlanet].y)/parti[nearestPlanet].r,
        (M.z - parti[nearestPlanet].z)/parti[nearestPlanet].r,
    };
    float I = 0;
    for(int i = 0;i < *suns_size;i++){
        RTXV MS = {
            parti[suns[i]].x - M.x,
            parti[suns[i]].y - M.y,
            parti[suns[i]].z - M.z
        };
        float MSLength = fast::sqrt(
                                    MS.x * MS.x +
                                    MS.y * MS.y +
                                    MS.z * MS.z
                                    );
        RTXV Nps = {
            MS.x/MSLength,
            MS.y/MSLength,
            MS.z/MSLength
        };
        float k = 1.0;
        RTXBVHRayForShadow(parti, metalBVHCollector[0], MS, M, metalBVHCollector, *metalBVHCollectorSize, nearestPlanet, M, MS, MSLength, k, suns[i]);
        I += max(Npm.x * Nps.x + Npm.y * Nps.y + Npm.z * Nps.z, 0.0) * k;
    }
    R[index] = clamp(parti[nearestPlanet].c1 * I,0.0f,255.0f);
    G[index] = clamp(parti[nearestPlanet].c2 * I,0.0f,255.0f);
    B[index] = clamp(parti[nearestPlanet].c3 * I,0.0f,255.0f);
}
