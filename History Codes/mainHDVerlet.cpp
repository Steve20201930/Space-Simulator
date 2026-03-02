#include <iostream>
#include <unistd.h>
#include <vector>
#include <SDL3/SDL.h>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <SDL3_ttf/SDL_ttf.h>
using namespace std;
TTF_Font* globalFont=nullptr;
const double G=1.0;
double dt=0.01;
const double ess=1e-12;
int width=1512;
int height=900;
struct particle{
    string name;
    double x,y,z;
    double vx,vy,vz;
    double ax,ay,az;
    double mass;
    double radius;
    int c1,c2,c3,c4;
};
struct camera{
    double x,y,z;
    double yaw,pitch;
    double fov;
    double vx,vy,vz;
    int target;
};
struct projection{
    double sx,sy,sz;
    int c1,c2,c3,c4;
    double r;
}res;
double Screen(particle& p,camera& cam){
    double dx=p.x-cam.x;
    double dy=p.y-cam.y;
    double dz=p.z-cam.z;
    double r=sqrt(dx*dx+dy*dy+dz*dz);

    double a=cam.yaw*(M_PI/180),b=cam.pitch*(M_PI/180);
    double fov=cam.fov*(M_PI/180);

    double x=dx*cos(a)-dz*sin(a);
    double z1=dx*sin(a)+dz*cos(a);
    double y=dy*cos(b)-z1*sin(b);
    double z=dy*sin(b)+z1*cos(b);

    double s_x=(x/(z*tan(0.5*fov))+1)*width/2;
    double s_y=(1-y/(z*tan(0.5*fov)))*height/2;

    res.sx=s_x;
    res.sy=s_y;
    res.sz=z;
    return -1;
}
void drawCircle2(SDL_Renderer* ren,projection p,double proj){
    double perspective=proj/p.sz;
    int r=(int)(p.r*perspective);
    if(r>500)r=500;
    // for(int j=-r;j<=r;j++){
    //     for(int i=-r;i<=r;i++){
    //         if(j*j+i*i<=r*r){
    //             SDL_RenderPoint(ren,p.sx+i,p.sy+j);
    //         }
    //     }
    // }
    for(int j=-r;j<=r;j++){
        int len=(int)sqrt(r*r-j*j);
        SDL_RenderLine(ren,p.sx-len,p.sy+j,p.sx+len,p.sy+j);
    }
}
int findFocus(vector<particle>& planet,camera& cam){
    double min_dist=50.0;
    int best=-1;
    for(int i=0;i<planet.size();i++){
        Screen(planet[i],cam);
        if(res.sz>1e-3&&res.sz<200){
            double dist_to_center=sqrt(pow(res.sx-width/2,2)+pow(res.sy-height/2,2));
            if(dist_to_center<min_dist){
                min_dist=dist_to_center;
                best=i;
            }
        }
    }
    return best;
}
// void draw(SDL_Renderer* ren,particle& p,camera& cam){
//     int sx=worldToScreen(p.x,cam,'x');
//     int sy=worldToScreen(p.y,cam,'y');
//     SDL_RenderPoint(ren,sx,sy);
// }
// void drawCircle(SDL_Renderer* ren,particle& p,camera& cam){
//     int sx=worldToScreen(p.x,cam,'x');
//     int sy=worldToScreen(p.y,cam,'y');
//     double dz=cam.z-p.z;
//     if (dz <= 2) return;

//     const double proj=200.0;
//     double perspective = proj / dz;

//     int r=(int)(p.radius*perspective);
//     for(int y=-r;y<=r;y++){
//         for(int x=-r;x<=r;x++){
//             if(x*x+y*y<=r*r){
//                 SDL_RenderPoint(ren,sx+x,sy+y);
//             }
//         }
//     }
// }
int main()
{
    vector<particle> planet;
    vector<double> newx,newy,newz,newax,neway,newaz;
    bool pause=false;
    // planet.push_back({"Star",0,0,0,2,0,0,1000,10,245,101,5,255});
    // planet.push_back({"Planet",100,0,0,2,0,0,1000,10,5,197,245,255});
    again:
    cout <<"Quantity of your stars: ";
    int n;
    cin>>n;
    if(n==-2){
        double newdt;
        cin>>newdt;
        dt=newdt;
        goto again;
    }if(n==-3){
        pause=true;
        goto again;
    }
    //Storage
    newx.resize(n);
    newy.resize(n);
    newz.resize(n);
    newax.resize(n);
    neway.resize(n);
    newaz.resize(n);
    //Storage
    for(int i=0;i<n;i++){
        string name;
        double x,y,z;
        double vx,vy,vz;
        double ax,ay,az;
        double mass;
        double radius;
        int c1,c2,c3,c4;
        cin>>name>>x>>y>>z>>vx>>vy>>vz>>ax>>ay>>az>>mass>>radius>>c1>>c2>>c3>>c4;
        planet.push_back({name,x,y,z,vx,vy,vz,ax,ay,az,mass,radius,c1,c2,c3,c4});
    }

    camera cam={0,0,-200,0,0,90,0,0,0,-1};
    SDL_Window* win=SDL_CreateWindow("Gravity system",width,height,SDL_WINDOW_RESIZABLE);
    SDL_Renderer* ren=SDL_CreateRenderer(win,nullptr);
    SDL_SetWindowPosition(win,SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED);
    SDL_SetWindowRelativeMouseMode(win, true);
    const double mousesensitivity=0.1;
    /*Initial movement*/
    double acceleration=0.5;
    double friction=0.94;
    /*Initial movement*/
    SDL_Event event;
    bool running=true;
    bool camera_locked=false;
    bool mouse_locked=true;
    int lastX,lastY;
    double proj=150.0;
    while(running){
        SDL_PollEvent(&event);
        SDL_GetWindowSize(win,&width,&height);
        switch(event.type){
            case SDL_EVENT_QUIT:{
                running=false;
                break;
            }case SDL_EVENT_KEY_DOWN:{
                if(event.key.key==SDLK_UP){
                    dt*=10;
                }else if(event.key.key==SDLK_DOWN){
                    dt*=0.1;
                }else if(event.key.key==SDLK_P&&acceleration<2000.0){
                    acceleration*=10;
                }else if(event.key.key==SDLK_O&&acceleration>0.2){
                    acceleration*=0.1;
                }else if(event.key.key==SDLK_TAB){
                    pause=!pause;
                }else if(event.key.key==SDLK_F&&camera_locked==false&&pause==true){
                    cam.x=planet[cam.target].x;
                    cam.y=planet[cam.target].y;
                    cam.z=planet[cam.target].z;
                    cam.vx=0;cam.vy=0;cam.vz=0;
                    camera_locked=true;
                }else if(event.key.key==SDLK_LEFT&&proj>1.0){
                    proj-=1.0;
                }else if(event.key.key==SDLK_RIGHT&&proj<200.0){
                    proj+=1.0;
                }else if(event.key.key==SDLK_ESCAPE){
                    mouse_locked=false;
                    SDL_SetWindowRelativeMouseMode(win, false);
                }
                break;
            }
            case SDL_EVENT_MOUSE_MOTION:{
                if(!mouse_locked)break;
                if(cam.yaw>360)cam.yaw-=360;
                if(cam.yaw<0)cam.yaw+=360;
                if(cam.pitch>90)cam.pitch=90;
                if(cam.pitch<-90)cam.pitch=-90;

                cam.yaw   += event.motion.xrel * mousesensitivity;
                cam.pitch -= event.motion.yrel * mousesensitivity;
            }
            case SDL_EVENT_MOUSE_BUTTON_DOWN:{
                if(!mouse_locked){
                    mouse_locked=true;
                    SDL_SetWindowRelativeMouseMode(win, true);
                }
            }
        }
        SDL_SetRenderDrawColor(ren,0,0,0,255);
        SDL_RenderClear(ren);
        /*COMPUTE MOVEMENT*/
        const bool* keystate=SDL_GetKeyboardState(NULL);
        double rad_yaw=cam.yaw*(M_PI/180);
        double rad_pitch=cam.pitch*(M_PI/180);
        
        double fx=sin(rad_yaw)*cos(rad_pitch);
        double fy=sin(rad_pitch);
        double fz=cos(rad_yaw)*cos(rad_pitch);
        double rx=cos(rad_yaw);
        double ry=0;
        double rz=-sin(rad_yaw);
        double ux=sin(rad_yaw)*sin(-rad_pitch);
        double uy=cos(rad_pitch);
        double uz=cos(rad_yaw)*sin(-rad_pitch);

        //escape locked planet
        if (camera_locked==true&&(keystate[SDL_SCANCODE_W] || keystate[SDL_SCANCODE_A] || keystate[SDL_SCANCODE_S] || keystate[SDL_SCANCODE_D] || keystate[SDL_SCANCODE_SPACE] || keystate[SDL_SCANCODE_LSHIFT])){
            camera_locked=false;
        }
        //escape locked planet
        if(keystate[SDL_SCANCODE_W]){
            cam.vx+=fx*acceleration;
            cam.vy+=fy*acceleration;
            cam.vz+=fz*acceleration;
        }if(keystate[SDL_SCANCODE_S]){
            cam.vx-=fx*acceleration;
            cam.vy-=fy*acceleration;
            cam.vz-=fz*acceleration;
        }if(keystate[SDL_SCANCODE_A]){
            cam.vx -= rx * acceleration;
            cam.vz -= rz * acceleration;
        }if(keystate[SDL_SCANCODE_D]){
            cam.vx += rx * acceleration;
            cam.vz += rz * acceleration;
        }if (keystate[SDL_SCANCODE_SPACE]) {
            cam.vx += ux * acceleration;
            cam.vy += uy * acceleration;
            cam.vz += uz * acceleration;
        }if (keystate[SDL_SCANCODE_LSHIFT]) {
            cam.vx -= ux * acceleration;
            cam.vy -= uy * acceleration;
            cam.vz -= uz * acceleration;
        }
        cam.vx *= friction; cam.vy *= friction; cam.vz *= friction;
        cam.x += cam.vx; cam.y += cam.vy; cam.z += cam.vz;
        /*COMPUTE MOVEMENT*/
        //COMPUTE POSITION t+dt
        if(pause==false){
            for(int cur=0;cur<planet.size();cur++){
                newx[cur]=planet[cur].x+planet[cur].vx*dt+0.5*planet[cur].ax*dt*dt;
                newy[cur]=planet[cur].y+planet[cur].vy*dt+0.5*planet[cur].ay*dt*dt;
                newz[cur]=planet[cur].z+planet[cur].vz*dt+0.5*planet[cur].az*dt*dt;
            }
            #pragma omp parallel for
            for(int cur=0;cur<planet.size();cur++){
                newax[cur]=0;
                neway[cur]=0;
                newaz[cur]=0;
                for(int comp=0;comp<planet.size();comp++){
                    if(cur==comp)continue;
                    double dx=newx[comp]-newx[cur];
                    double dy=newy[comp]-newy[cur];
                    double dz=newz[comp]-newz[cur];
                    double r=sqrt(dx*dx+dy*dy+dz*dz+ess);
                    double F=G*planet[cur].mass*planet[comp].mass/(r*r);

                    newax[cur]+=F*dx/(r*planet[cur].mass);
                    neway[cur]+=F*dy/(r*planet[cur].mass);
                    newaz[cur]+=F*dz/(r*planet[cur].mass);
                }
                planet[cur].vx+=0.5*(planet[cur].ax+newax[cur])*dt;
                planet[cur].vy+=0.5*(planet[cur].ay+neway[cur])*dt;
                planet[cur].vz+=0.5*(planet[cur].az+newaz[cur])*dt;
                planet[cur].x=newx[cur];
                planet[cur].y=newy[cur];
                planet[cur].z=newz[cur];
                planet[cur].ax=newax[cur];
                planet[cur].ay=neway[cur];
                planet[cur].az=newaz[cur];
        
                // cout <<rate<<' '<<planet[cur].name<<':'<<planet[cur].x<<' '<<planet[cur].y<<' '<<planet[cur].z<<' '<<planet[cur].vx<<' '<<planet[cur].vy<<' '<<planet[cur].vz<<' '<<planet[cur].ax<<' '<<planet[cur].ay<<' '<<planet[cur].az<<endl;
            }
        }else{
            if(camera_locked==false)cam.target=findFocus(planet,cam);
        }
        // sort(planet.begin(), planet.end(),
        //     [&](const particle& a, const particle& b){
        //         return a.z < b.z; // z 大 = 更远，先画
        //     }
        // );
        vector<projection> rendering;
        //DRAW
        for(int cur=0;cur<planet.size();cur++){
            // drawCircle(ren,planet[cur],cam);
            if(cur==cam.target&&camera_locked==true){
                cam.x=planet[cur].x;
                cam.y=planet[cur].y;
                cam.z=planet[cur].z;
                continue;
            }
            Screen(planet[cur],cam);
            if(res.sz>1e-3)rendering.push_back({res.sx,res.sy,res.sz,planet[cur].c1,planet[cur].c2,planet[cur].c3,planet[cur].c4,planet[cur].radius});
        }
        sort(rendering.begin(),rendering.end(),[](const projection& a,const projection&b){
            return a.sz>b.sz;
        });
        for(int cur=0;cur<rendering.size();cur++){
            SDL_SetRenderDrawColor(ren,rendering[cur].c1,rendering[cur].c2,rendering[cur].c3,rendering[cur].c4);
            drawCircle2(ren,rendering[cur],proj);
        }
        cout <<"dt"<<dt<<" a"<<acceleration<<" proj"<<proj<<" yaw"<<cam.yaw<<" pitch"<<cam.pitch<<" x:"<<cam.x<<" y:"<<cam.y<<" z:"<<cam.z<<endl;
        SDL_RenderPresent(ren);
    }
    return 0;
}