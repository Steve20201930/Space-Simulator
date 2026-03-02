//
//  pos2.mm
//  Space Simulator App
//
//  Created by Steve William on 2/27/26.
//

#include <metal_stdlib>
using namespace metal;
kernel void comp(device const float* newx[[buffer(0)]],
                 device const float* newy[[buffer(1)]],
                 device const float* newz[[buffer(2)]],
                 device float* vx[[buffer(3)]],
                 device float* vy[[buffer(4)]],
                 device float* vz[[buffer(5)]],
                 device float* ax[[buffer(6)]],
                 device float* ay[[buffer(7)]],
                 device float* az[[buffer(8)]],
                 device float* dt[[buffer(9)]],
                 device float* mass[[buffer(10)]],
                 device float* radius[[buffer(11)]],
                 device long* N[[buffer(12)]],
                 device const float* G[[buffer(13)]],
                 device const bool* SpaceShipMode[[buffer(14)]],
                 uint id [[thread_position_in_grid]]
                 ){
    
    float newax=0;
    float neway=0;
    float newaz=0;
    for(long i=0;i<*N;i++){
        if(i==id)continue;
        float dx=newx[i]-newx[id];
        float dy=newy[i]-newy[id];
        float dz=newz[i]-newz[id];
//        float r=sqrt(dx*dx+dy*dy+dz*dz+1e-9f);
//        float F=*G*mass[id]*mass[i]/(r*r);
        float r2 = dx*dx + dy*dy + dz*dz + 1e-9f;
        float r = rsqrt(r2);
        float r_inv3 = r*r*r;
        float cF = *G * mass[i] * r_inv3;
        
        newax+=dx*cF;
        neway+=dy*cF;
        newaz+=dz*cF;
    }
    vx[id]+=0.5*(ax[id]+newax)* *dt;
    vy[id]+=0.5*(ay[id]+neway)* *dt;
    vz[id]+=0.5*(az[id]+newaz)* *dt;
    
    ax[id]=newax;
    ay[id]=neway;
    az[id]=newaz;
}
kernel void shader(texture2d<float, access::write> outTexture [[texture(0)]],
                   constant float2& pos [[buffer(0)]],
                   constant float4& color[[buffer(1)]],
                   device const float* lx[[buffer(2)]],
                   device const float* ly[[buffer(3)]],
                   device const float* lz[[buffer(4)]],
                   device const float* r[[buffer(5)]],
                   uint2 gid[[thread_position_in_grid]]){
    float nx=((float)gid.x/ *r)-1.0f;
    float ny=((float)gid.y/ *r)-1.0f;
    float nz=sqrt(1.0-(nx*nx+ny*ny));
    
    float intensity=nx**lx+ny**ly+nz**lz;
    if(intensity<0)intensity=abs(intensity*0.03);
    float4 final_color=float4(color.rgb*intensity,color.a);
    uint2 final_pos=uint2(pos.x+(float)gid.x-*r,pos.y+(float)gid.y-*r);
    outTexture.write(final_color,final_pos);
}
