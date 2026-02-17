#include <iostream>
#include <unistd.h>
#include <vector>
#include <SDL3/SDL.h>
#include <SDL3_ttf/SDL_ttf.h>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <sstream>
using namespace std;
double G=1.0;
double dt=0.01;
const double ess=1e-12;
int width=1512;
int height=900;
double rocheK=1.26;
int disintegrate_num=1;
struct particle{
    string name;
    double x,y,z;
    double vx,vy,vz;
    double ax,ay,az;
    double mass;
    double radius;
    int c1,c2,c3,c4;
    bool merged;
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
void disintegrate4(vector<particle>& p,vector<particle>& roc,double rocheR,double r,int cur){
    if(p[cur].radius<=disintegrate_num)return;
    const double K=2.0;
    int genp=(int)(rocheR/r)*K;
    if(genp>10)genp=10;
    //Original 
    double p0x=p[cur].mass*p[cur].vx;
    double p0y=p[cur].mass*p[cur].vy;
    double p0z=p[cur].mass*p[cur].vz;
    double m0=p[cur].mass;
    double r0=p[cur].mass;
    //Pieces
    double frag_r=disintegrate_num;
    double frag_m=p[cur].mass*pow(frag_r/p[cur].radius,3);

    double pfragx=0,pfragy=0,pfragz=0;
    for(int i=0;i<genp;i++){
        double ux=2.0*rand()/RAND_MAX-1.0; //[-1.0,1.0]
        double uy=2.0*rand()/RAND_MAX-1.0; //[-1.0,1.0]
        double uz=2.0*rand()/RAND_MAX-1.0; //[-1.0,1.0]
        double len=sqrt(ux*ux+uy*uy+uz*uz);
        if(len<1e-6)continue; //Vector 0
        ux/=len; uy/=len; uz/=len;

        double newx=p[cur].x+p[cur].radius*ux;
        double newy=p[cur].y+p[cur].radius*uy;
        double newz=p[cur].z+p[cur].radius*uz;

        double esc=0.3*sqrt(G*p[cur].mass/(p[cur].radius+ess));
        double vx=p[cur].vx+ux*esc*(0.8+0.4*rand()/RAND_MAX);
        double vy=p[cur].vy+uy*esc*(0.8+0.4*rand()/RAND_MAX);
        double vz=p[cur].vz+uz*esc*(0.8+0.4*rand()/RAND_MAX);

        pfragx+=vx*frag_m; pfragy+=vy*frag_m; pfragy+=vz*frag_m;
        string name="G";
        #pragma omp critical
        {
            roc.push_back({p[cur].name+name,newx,newy,newz,vx,vy,vz,0,0,0,frag_m,frag_r,p[cur].c1,p[cur].c2,p[cur].c3,p[cur].c4,false});
        }
    }
    double mt=p[cur].mass-frag_m*genp;
    double rt=p[cur].radius*pow(mt/m0,1.0/3.0)*0.99;
    // double rt=p[cur].radius-genp*frag_r;
    p[cur].vx=(p0x-pfragx)/mt;
    p[cur].vy=(p0y-pfragy)/mt;
    p[cur].vz=(p0z-pfragz)/mt;
    p[cur].mass=mt;
    p[cur].radius=rt;
}
void merge(vector<particle>& p,int cur,int comp){
    string new_name=p[cur].name;
    double new_x=(p[cur].mass*p[cur].x+p[comp].mass*p[comp].x)/(p[cur].mass+p[comp].mass);
    double new_y=(p[cur].mass*p[cur].y+p[comp].mass*p[comp].y)/(p[cur].mass+p[comp].mass);
    double new_z=(p[cur].mass*p[cur].z+p[comp].mass*p[comp].z)/(p[cur].mass+p[comp].mass);
    double new_vx=(p[cur].mass*p[cur].vx+p[comp].mass*p[comp].vx)/(p[cur].mass+p[comp].mass);
    double new_vy=(p[cur].mass*p[cur].vy+p[comp].mass*p[comp].vy)/(p[cur].mass+p[comp].mass);
    double new_vz=(p[cur].mass*p[cur].vz+p[comp].mass*p[comp].vz)/(p[cur].mass+p[comp].mass);
    int c1=(int)((p[cur].mass*p[cur].c1+p[comp].mass*p[comp].c1)/(p[cur].mass+p[comp].mass));
    int c2=(int)((p[cur].mass*p[cur].c2+p[comp].mass*p[comp].c2)/(p[cur].mass+p[comp].mass));
    int c3=(int)((p[cur].mass*p[cur].c3+p[comp].mass*p[comp].c3)/(p[cur].mass+p[comp].mass));
    #pragma omp critical
    {
        p.push_back({new_name,new_x,new_y,new_z,new_vx,new_vy,new_vz,0,0,0,(p[cur].mass+p[comp].mass),pow(pow(p[cur].radius,3)+pow(p[comp].radius,3),1.0/3.0),c1,c2,c3,255,false});
    }   
}
void execute(vector<particle>& p,string command,bool& pause,bool& roche){
    stringstream ssin(command);
    int n;
    ssin>>n;
    switch(n){
        case -3:{
            pause=!pause;
        }
        case -4:{
            roche=!roche;
        }
        case -5:{
            double newrocheK;
            ssin>>newrocheK;
            rocheK=newrocheK;
        }
        case -6:{
            double newG;
            ssin>>newG;
            G=newG;
        } 
    }
    for(int i=0;i<n;i++){
        string name;
        double x,y,z;
        double vx,vy,vz;
        double ax,ay,az;
        double mass;
        double radius;
        int c1,c2,c3,c4;
        ssin>>name>>x>>y>>z>>vx>>vy>>vz>>ax>>ay>>az>>mass>>radius>>c1>>c2>>c3>>c4;
        p.push_back({name,x,y,z,vx,vy,vz,ax,ay,az,mass,radius,c1,c2,c3,c4,false});
    }
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
    vector<particle> roc;

    vector<double> newx,newy,newz,newax,neway,newaz;
    bool pause=false;
    bool roche=false;
    // planet.push_back({"Star",0,0,0,2,0,0,1000,10,245,101,5,255});
    // planet.push_back({"Planet",100,0,0,2,0,0,1000,10,5,197,245,255});

    camera cam={0,0,-200,0,0,90,0,0,0,-1};
    SDL_Window* win=SDL_CreateWindow("Gravity system",width,height,SDL_WINDOW_RESIZABLE);
    SDL_Renderer* ren=SDL_CreateRenderer(win,nullptr);
    SDL_SetWindowPosition(win,SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED);
    SDL_SetWindowRelativeMouseMode(win, true);
    TTF_Font* font=TTF_OpenFont("Arial.ttf",24);
    //Initial TTF Font
    SDL_Color textColor={255,255,255,255};
    //Initial TTF Font
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
    //Inital input
    bool isTyping=false;
    string commandbuf="";
    vector<string> currentbuf;
    int line=0;
    int fontsize=24;
    int initialT=0.7*height;
    int iniline=initialT;
    while(running){
        // if(event.type==SDL_EVENT_WINDOW_RESIZED){
        //     int initialT=0.7*height;
        //     int iniline=initialT;
        // }
        newx.resize(planet.size());
        newy.resize(planet.size());
        newz.resize(planet.size());
        newax.resize(planet.size());
        neway.resize(planet.size());
        newaz.resize(planet.size());
        SDL_PollEvent(&event);
        SDL_GetWindowSize(win,&width,&height);
        switch(event.type){
            case SDL_EVENT_QUIT:{
                running=false;
                break;
            }case SDL_EVENT_KEY_DOWN:{
                if(isTyping==false){
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
                    }else if(event.key.key==SDLK_T){
                        isTyping=true;
                        commandbuf="";
                        currentbuf.clear();
                        line=0;
                        currentbuf.push_back(">");
                        SDL_StartTextInput(win);
                    }
                }
                if(event.key.key==SDLK_ESCAPE){
                    if(isTyping==true){
                        isTyping=false;
                        SDL_StopTextInput(win);
                    }else{
                        mouse_locked=false;
                        SDL_SetWindowRelativeMouseMode(win, false);
                    }
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
        if(isTyping==false){
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
        }
        cam.vx *= friction; cam.vy *= friction; cam.vz *= friction;
        cam.x += cam.vx; cam.y += cam.vy; cam.z += cam.vz;
        /*Console Input*/
        if(isTyping==true){
            cout <<commandbuf<<endl;
            if(event.type==SDL_EVENT_TEXT_INPUT){
                commandbuf+=event.text.text;
                currentbuf[line]+=event.text.text;
                cout <<commandbuf<<endl;
            }else if(event.type==SDL_EVENT_KEY_DOWN){
                if(event.key.key==SDLK_BACKSPACE&&currentbuf[line].size()>1){
                    commandbuf.pop_back();
                    currentbuf[line].pop_back();
                }
                if((SDL_KMOD_LSHIFT&event.key.mod)&&event.key.key==SDLK_RETURN){
                    execute(planet,commandbuf,pause,roche);
                    isTyping=false;
                    SDL_StopTextInput(win);
                    continue;
                }else if(event.key.key==SDLK_RETURN){
                    if(initialT+line*0.5*fontsize>height-fontsize*0.5){
                        iniline-=(0.5*fontsize);
                    }
                    line+=1;
                    commandbuf+=" ";
                    currentbuf.push_back(">");
                }
                if(SDL_KMOD_GUI&&event.key.key==SDLK_V){
                    if(SDL_HasClipboardText()){
                        char* text=SDL_GetClipboardText();
                        commandbuf+=text;
                        currentbuf[line]+=text;
                        SDL_free(text);
                    }
                }
                
            }
            SDL_SetRenderDrawColor(ren,255,255,255,255);
            for(int i=0;i<=line;i++){
                SDL_RenderDebugText(ren,50,iniline+i*0.5*fontsize,currentbuf[i].c_str());
            }
        }
        //COMPUTE POSITION t+dt
        if(pause==false){
            for(int cur=0;cur<planet.size();cur++){
                newx[cur]=planet[cur].x+planet[cur].vx*dt+0.5*planet[cur].ax*dt*dt;
                newy[cur]=planet[cur].y+planet[cur].vy*dt+0.5*planet[cur].ay*dt*dt;
                newz[cur]=planet[cur].z+planet[cur].vz*dt+0.5*planet[cur].az*dt*dt;
            }
            int fsize=planet.size();
            #pragma omp parallel for
            for(int cur=0;cur<fsize;cur++){
                newax[cur]=0;
                neway[cur]=0;
                newaz[cur]=0;
                for(int comp=0;comp<fsize;comp++){
                    if(cur==comp)continue;
                    double dx=newx[comp]-newx[cur];
                    double dy=newy[comp]-newy[cur];
                    double dz=newz[comp]-newz[cur];
                    double r=sqrt(dx*dx+dy*dy+dz*dz+ess);
                    double F=G*planet[cur].mass*planet[comp].mass/(r*r);

                    newax[cur]+=F*dx/(r*planet[cur].mass);
                    neway[cur]+=F*dy/(r*planet[cur].mass);
                    newaz[cur]+=F*dz/(r*planet[cur].mass);

                    //Compute roche limit
                     if(roche==true&&planet[cur].radius>disintegrate_num){
                        double rocheR=rocheK*planet[cur].radius*pow(2*planet[comp].mass/planet[cur].mass,1.0/3.0);
                        if(r>rocheR)continue;
                        disintegrate4(planet,roc,rocheR,r,cur);
                    }
                    //Compute merge limit
                    if(roche==true&&planet[cur].merged==false&&planet[comp].merged==false){
                        if((planet[cur].radius+planet[comp].radius)<r)continue;
                        merge(planet,cur,comp);
                        planet[cur].merged=true;
                        planet[comp].merged=true;
                    }
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
                
                // cout <<' '<<planet[cur].name<<':'<<planet[cur].x<<' '<<planet[cur].y<<' '<<planet[cur].z<<' '<<planet[cur].vx<<' '<<planet[cur].vy<<' '<<planet[cur].vz<<' '<<planet[cur].ax<<' '<<planet[cur].ay<<' '<<planet[cur].az<<' '<<planet[cur].mass<<' '<<planet[cur].radius<<endl;
            }
        }else{
            if(camera_locked==false)cam.target=findFocus(planet,cam);
        }
        if(roche==true){
            //Merge function
            int fsize=planet.size();
            for(int i=0;i<fsize;i++){
                if(planet[i].merged==true){
                    planet.erase(planet.begin()+i);
                    i--;
                    fsize--;
                }
            }
            //Roche function
            planet.insert(planet.end(),roc.begin(),roc.end());
            roc.clear();
        }
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
            if(res.sz>1e-1)rendering.push_back({res.sx,res.sy,res.sz,planet[cur].c1,planet[cur].c2,planet[cur].c3,planet[cur].c4,planet[cur].radius});
        }
        sort(rendering.begin(),rendering.end(),[](const projection& a,const projection&b){
            return a.sz>b.sz;
        });
        for(int cur=0;cur<rendering.size();cur++){
            SDL_SetRenderDrawColor(ren,rendering[cur].c1,rendering[cur].c2,rendering[cur].c3,rendering[cur].c4);
            drawCircle2(ren,rendering[cur],proj);
        }
        cout <<"Num"<<planet.size()<<" dt"<<dt<<" a"<<acceleration<<" proj"<<proj<<" yaw"<<cam.yaw<<" pitch"<<cam.pitch<<" x:"<<cam.x<<" y:"<<cam.y<<" z:"<<cam.z<<endl;
        SDL_RenderPresent(ren);
    }
    return 0;
}
