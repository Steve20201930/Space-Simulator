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
    float reflection = 0.5;
    
    float w_velocity = 2 * 3.1415926 / 86400;
    float rotation = 0;
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
void RTXBVHRayForShadow(device const projection *parti, device const metalBVHNode &rBVH, const RTXV vector, thread const RTXV &camera, device const metalBVHNode *BVHCollector, const int BVHCollectorSize, thread const int nearestPlanet, thread const RTXV &M, thread const RTXV &MS, thread const float &MSLength, thread float &k, thread int sunIndex, thread float SSAOI){
    
    if(k == SSAOI)return;
    if(rBVH.left == -1 && rBVH.right == -1){
        for(int i = 0;i < rBVH.count;i++){
            long p = rBVH.indices[i];
            if(p == nearestPlanet || p == sunIndex)continue;
            RTXV Shadow = RTXVectorCalcM(MS, M, {parti[p].x, parti[p].y, parti[p].z}, parti[p].r, MSLength);
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

void StellarHaloDetect(device const projection *parti, device const int *suns, long suns_size, const RTXV vector, const RTXV camera, thread uint8_t &R, thread uint8_t &G, thread uint8_t &B){
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
void getTextureByCube(device const projection &planet, device const uint8_t *texture, int textureID, int S, const RTXV vector, thread int &c1, thread int &c2, thread int &c3, thread int &c4){
    if(textureID < 0)return;
    
    RTXV rotVector = {
        vector.x * cos(planet.rotation) - vector.z * sin(planet.rotation),
        vector.y,
        vector.x * sin(planet.rotation) + vector.z * cos(planet.rotation)
    };
    
    int i, j, face;
    //px
    if(rotVector.x > 0 && rotVector.x >= abs(rotVector.y) && rotVector.x >= abs(rotVector.z)){
        float t = 1 / rotVector.x;
        
        face = 0;
        i = int((1 - t * rotVector.z) * 0.5 * S);
        j = int((1 - t * rotVector.y) * 0.5 * S);
    }
    //nx
    else if(rotVector.x <= 0 && -rotVector.x >= abs(rotVector.y) && -rotVector.x >= abs(rotVector.z)){
        float t = -1 / rotVector.x;

        face = 1;
        i = int((1 + t * rotVector.z) * 0.5 * S);
        j = int((1 - t * rotVector.y) * 0.5 * S);
    }
    //py
    else if(rotVector.y > 0 && rotVector.y >= abs(rotVector.x) && rotVector.y >= abs(rotVector.z)){
        float t = 1 / rotVector.y;

        face = 2;
        i = int((1 + t * rotVector.x) * 0.5 * S);
        j = int((1 + t * rotVector.z) * 0.5 * S);
    }
    //ny
    else if(rotVector.y <= 0 && -rotVector.y >= abs(rotVector.x) && -rotVector.y >= abs(rotVector.z)){
        float t = -1 / rotVector.y;

        face = 3;
        i = int((1 + t * rotVector.x) * 0.5 * S);
        j = int((1 - t * rotVector.z) * 0.5 * S);
    }
    //pz
    else if(rotVector.z > 0 && rotVector.z >= abs(rotVector.x) && rotVector.z >= abs(rotVector.y)){
        float t = 1 / rotVector.z;

        face = 4;
        i = int((1 + t * rotVector.x) * 0.5 * S);
        j = int((1 - t * rotVector.y) * 0.5 * S);
    }
    //nz
    else if(rotVector.z <= 0 && -rotVector.z >= abs(rotVector.x) && -rotVector.z >= abs(rotVector.y)){
        float t = -1 / rotVector.z;

        face = 5;
        i = int((1 - t * rotVector.x) * 0.5 * S);
        j = int((1 - t * rotVector.y) * 0.5 * S);
    }
    else return;
    
    i = clamp(i, 0, S-1);
    j = clamp(j, 0, S-1);
    
    ulong p = ulong(textureID) + (ulong(face) * S * S + j * S + i) * 4;
    
    c1 = texture[p];
    c2 = texture[p + 1];
    c3 = texture[p + 2];
    c4 = texture[p + 3];
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
                    device const int *RTXReflect,
                    
                    device const int *textureIndex,
                    device const uint8_t *texture,
                    device const int *S,
                    
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
        uint8_t SR = 0, SG = 0, SB = 0;
        StellarHaloDetect(parti, suns, *suns_size, RTXVector[index], {cam->x, cam->y, cam->z}, SR, SG, SB);
        R[index] = SR;
        G[index] = SG;
        B[index] = SB;
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
        RTXBVHRayForShadow(parti, metalBVHCollector[0], MS, M, metalBVHCollector, *metalBVHCollectorSize, nearestPlanet, M, MS, MSLength, k, suns[i], SSAOI);
        I += max(Npm.x * Nps.x + Npm.y * Nps.y + Npm.z * Nps.z, 0.0) * (1 - SSAOI) * k + SSAOI;
        
        //Texture Renderer
        int c1 = parti[nearestPlanet].c1;
        int c2 = parti[nearestPlanet].c2;
        int c3 = parti[nearestPlanet].c3;
        int c4 = parti[nearestPlanet].c4;
        if(textureIndex[nearestPlanet] >= 0){
            getTextureByCube(parti[nearestPlanet], texture, textureIndex[nearestPlanet], S[nearestPlanet], Npm, c1, c2, c3, c4);
        }
        
        R[index] += clamp(
                         ((parti[suns[i]].c1 - c1) * 0.05 + c1) * I
                         ,0.0f,255.0f);
        G[index] += clamp(
                          ((parti[suns[i]].c2 - c2) * 0.05 + c2) * I
                          ,0.0f,255.0f);
        B[index] += clamp(
                          ((parti[suns[i]].c3 - c3) * 0.05 + c3) * I
                          ,0.0f,255.0f);
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
                               {MrPre.x - PreP.x, MrPre.y - PreP.y, MrPre.z - PreP.z},
                               1 / PreP.r
                               );
            RTXV ReflectVector = VectorPlus(DirectVector, VectorMul(N, VectorDotMul(DirectVector, N) * (-2)));
            DirectVector = ReflectVector;
            float nearestOMRef = INFINITY;
            int nearestPlanetRef = -1;
            RTXBVHRayForIntersect(parti, metalBVHCollector[0], ReflectVector, MrPre, metalBVHCollector, *metalBVHCollectorSize, nearestOMRef, nearestPlanetRef, Mr, VectorLength(ReflectVector));
            bool isStellarRef = false;
            for(long i = 0;i < *suns_size;i++){
                if(suns[i] == nearestPlanetRef){
                    isStellarRef = true;
                    break;
                }
            }
            if(nearestPlanetRef == -1 || isStellarRef){
                uint8_t SR = 0, SG = 0, SB = 0;
                StellarHaloDetect(parti, suns, *suns_size, ReflectVector, MrPre, SR, SG, SB);
                RR += (float(SR) - RRefPre) * WR;
                GG += (float(SG) - GRefPre) * WR;
                BB += (float(SB) - BRefPre) * WR;
                break;
            }
            MrPre = Mr;
            PreP = parti[nearestPlanetRef];
            /* Light Magnitude */
            RTXV NpmRef = {
                (Mr.x - parti[nearestPlanetRef].x)/parti[nearestPlanetRef].r,
                (Mr.y - parti[nearestPlanetRef].y)/parti[nearestPlanetRef].r,
                (Mr.z - parti[nearestPlanetRef].z)/parti[nearestPlanetRef].r
            };
            
            float3 RefColor = float3(0.0f);
            for(int i = 0;i < *suns_size;i++){
                float PSRef = fast::sqrt(
                                         pow(parti[suns[i]].x - parti[nearestPlanetRef].x, 2) + pow(parti[suns[i]].y - parti[nearestPlanetRef].y, 2) + pow(parti[suns[i]].z - parti[nearestPlanetRef].z, 2)
                                         );
                RTXV NpsRef = {
                    (parti[suns[i]].x - parti[nearestPlanetRef].x)/PSRef,
                    (parti[suns[i]].y - parti[nearestPlanetRef].y)/PSRef,
                    (parti[suns[i]].z - parti[nearestPlanetRef].z)/PSRef,
                };
                float MSLengthRef = fast::sqrt(
                                         pow((parti[suns[i]].x - Mr.x), 2) +
                                         pow((parti[suns[i]].y - Mr.y), 2) +
                                         pow((parti[suns[i]].z - Mr.z), 2)
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
                Ir += max((NpmRef.x * NpsRef.x + NpmRef.y * NpsRef.y + NpmRef.z * NpsRef.z) * (1 - SSAOI) * k + SSAOI, 0.0);
                if(k < 1.0f)Ir = SSAOI;
                
                //Texture Renderer
                int c1Ref = parti[nearestPlanetRef].c1;
                int c2Ref = parti[nearestPlanetRef].c2;
                int c3Ref = parti[nearestPlanetRef].c3;
                int c4Ref = parti[nearestPlanetRef].c4;
                if(textureIndex[nearestPlanetRef] >= 0){
                    getTextureByCube(parti[nearestPlanetRef], texture, textureIndex[nearestPlanetRef], S[nearestPlanetRef], NpmRef, c1Ref, c2Ref, c3Ref, c4Ref);
                }
                
                RefColor += float3((parti[suns[i]].c1 - c1Ref) * 0.05 + c1Ref,
                (parti[suns[i]].c2 - c2Ref) * 0.05 + c2Ref,
                (parti[suns[i]].c3 - c3Ref) * 0.05 + c3Ref) * Ir;
            }
            RR += (RefColor.x - RRefPre) * WR;
            GG += (RefColor.y - GRefPre) * WR;
            BB += (RefColor.z - BRefPre) * WR;
            
            RRefPre = RefColor.x;
            GRefPre = RefColor.y;
            BRefPre = RefColor.z;
        }
        R[index] = uint8_t(clamp(RR, 0.0f, 255.0f));
        G[index] = uint8_t(clamp(GG, 0.0f, 255.0f));
        B[index] = uint8_t(clamp(BB, 0.0f, 255.0f));
    }
}
