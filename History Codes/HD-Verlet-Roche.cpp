#include <iostream>
#include <unistd.h>
#include <vector>
#include <SDL3/SDL.h>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <sstream>
#include <fstream>
// #include "spaceshipmode.h"
using namespace std;
double G=6.674e-11;
double dt=0.01;
const double ess=1e-12;
int width=1512;
int height=900;
double rocheK=1.26;
int disintegrate_num=1;
double disK=2.0;
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
}cam;
struct projection{
    double sx,sy,sz;
    int c1,c2,c3,c4;
    double r;
}res;
double Screen(particle& p,camera& cam);
void drawCircle2(SDL_Renderer* ren,projection p,double proj);
void drawCircle3(SDL_Renderer* ren,projection p,double fov_deg);
int findFocus(vector<particle>& planet,camera& cam);
void disintegrate4(vector<particle>& p,vector<particle>& roc,double rocheR,double r,int cur,double dx,double dy,double dz,double K);
void merge(vector<particle>& p,int cur,int comp);
void execute(vector<particle>& p,string command,bool& pause,bool& roche,bool& debug,bool& spaceshipmode,camera &cam,string& camera_lockin,bool& camera_locked);
void save(vector<particle> p);
void camera_movement(camera &cam,double friction,double acceleration,bool& camera_locked,string camera_lockin,vector<particle> p);
void position_movement(vector<particle> &planet,vector<double> &newx,vector<double> &newy,vector<double> &newz,vector<double> &newax,vector<double> &neway,vector<double> &newaz,bool roche,vector<particle> &roc);
string toScientific(double val,int precision);

int main()
{
    vector<particle> planet;
    vector<particle> roc;

    vector<double> newx,newy,newz,newax,neway,newaz;
    bool pause=false;
    bool roche=false;
    // planet.push_back({"Star",0,0,0,2,0,0,1000,10,245,101,5,255});
    // planet.push_back({"Planet",100,0,0,2,0,0,1000,10,5,197,245,255});
    //Initial
    cam={0,0,-200,0,0,60,0,0,0};
    SDL_Window* win=SDL_CreateWindow("Gravity system",width,height,SDL_WINDOW_RESIZABLE);
    SDL_Renderer* ren=SDL_CreateRenderer(win,nullptr);
    SDL_SetWindowPosition(win,SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED);
    SDL_SetWindowRelativeMouseMode(win, true);
    const double mousesensitivity=0.1;
    /*Initial movement*/
    double acceleration=0.5;
    double friction=0.94;
    /*Initial movement*/
    float fps=0;
    Uint64 frame_count=0;
    Uint64 last_time=SDL_GetTicks();
    SDL_Event event;
    bool running=true;
    bool camera_locked=false;
    string camera_lockin="";
    bool mouse_locked=true;
    bool debug=true;
    bool spaceshipmode=false;
    //Inital input
    bool isTyping=false;
    string commandbuf="";
    vector<string> currentbuf;
    int line=0;
    int fontsize=24;
    int initialT=0.7*height;
    int iniline=initialT;
    while(running){
         if(event.type==SDL_EVENT_WINDOW_RESIZED){
             int initialT=0.7*height;
             int iniline=initialT;
         }
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
                    }else if(event.key.key==SDLK_P&&acceleration<20000000.0){
                        acceleration*=10;
                    }else if(event.key.key==SDLK_O&&acceleration>0.2){
                        acceleration*=0.1;
                    }else if(event.key.key==SDLK_TAB){
                        pause=!pause;
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
        if(isTyping==false)camera_movement(cam,friction,acceleration,camera_locked,camera_lockin,planet);
        
        //COMPUTE POSITION t+dt
        if(pause==false)position_movement(planet,newx,newy,newz,newax,neway,newaz,roche,roc);
        if(roche==true){
            //Merge function
            long fsize=planet.size();
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
        //DRAW
        
        vector<projection> rendering;
        for(int cur=0;cur<planet.size();cur++){
            // drawCircle(ren,planet[cur],cam);
            if(planet[cur].name==camera_lockin&&camera_locked==true){
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
//            drawCircle2(ren,rendering[cur],proj);
            drawCircle3(ren,rendering[cur],cam.fov);
        }
        /*Console Input*/
        if(isTyping==true){
            if(event.type==SDL_EVENT_TEXT_INPUT){
                commandbuf+=event.text.text;
                currentbuf[line]+=event.text.text;
            }else if(event.type==SDL_EVENT_KEY_DOWN){
                if(event.key.key==SDLK_BACKSPACE&&currentbuf[line].size()>1){
                    commandbuf.pop_back();
                    currentbuf[line].pop_back();
                }
                if((SDL_KMOD_LSHIFT&event.key.mod)&&event.key.key==SDLK_RETURN){
                    if(initialT+line*0.5*fontsize>height-fontsize*0.5){
                        iniline-=(0.5*fontsize);
                    }
                    line+=1;
                    commandbuf+=" ";
                    currentbuf.push_back(">");
                }else if(event.key.key==SDLK_RETURN){
                    execute(planet,commandbuf,pause,roche,debug,spaceshipmode,cam,camera_lockin,camera_locked);
                    isTyping=false;
                    SDL_StopTextInput(win);
                    continue;
                }
                if((SDL_KMOD_GUI&event.key.mod)&&event.key.key==SDLK_V){
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
        /*Console Input*/
        /*Debug*/
        if(debug==true){
            frame_count++;
            Uint64 current_time=SDL_GetTicks();
            if(current_time-last_time>=1000){
                fps=frame_count/((current_time-last_time)/1000.f);
                frame_count=0;
                last_time=current_time;
            }
            string debug_buf1="X:"+to_string(cam.x)+" Y:"+to_string(cam.y)+" Z:"+to_string(cam.z)+"  YAW:"+to_string(cam.yaw)+" PITCH:"+to_string(cam.pitch)+"    Quantity:"+to_string(planet.size());
            string debug_buf2="RunSpeed(dt/s):"+to_string(dt)+" acceleration:"+to_string(acceleration)+" FOV:"+to_string(cam.fov)+" fps:"+to_string(fps);
            string debug_buf3="RocheLimit:["+to_string(roche)+"] "+to_string(rocheK);
            string debug_buf4="G:"+toScientific(G, 4);
            if(pause==true)debug_buf1+="  [PAUSE]";
            SDL_SetRenderDrawColor(ren,255,255,255,255);
            SDL_RenderDebugText(ren,20,10,debug_buf1.c_str());
            SDL_RenderDebugText(ren,20,22,debug_buf2.c_str());
            SDL_RenderDebugText(ren,20,34,debug_buf3.c_str());
            SDL_RenderDebugText(ren,20,46,debug_buf4.c_str());
        }
        /*Debug*/
        SDL_RenderPresent(ren);
    }
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
/*
    
 
 
            Function
 
 
 
 */
void camera_movement(camera &cam,double friction,double acceleration,bool& camera_locked,string camera_lockin,vector<particle> p){
    const bool* keystate=SDL_GetKeyboardState(NULL);
    double rad_yaw=cam.yaw*(M_PI/180);
    double rad_pitch=cam.pitch*(M_PI/180);
    
    double fx=sin(rad_yaw)*cos(rad_pitch);
    double fy=sin(rad_pitch);
    double fz=cos(rad_yaw)*cos(rad_pitch);
    double rx=cos(rad_yaw);
    double rz=-sin(rad_yaw);
    double ux=sin(rad_yaw)*sin(-rad_pitch);
    double uy=cos(rad_pitch);
    double uz=cos(rad_yaw)*sin(-rad_pitch);
    
    if(camera_locked==true&&(keystate[SDL_SCANCODE_W]||keystate[SDL_SCANCODE_S]||keystate[SDL_SCANCODE_A]||keystate[SDL_SCANCODE_D]||keystate[SDL_SCANCODE_SPACE]||keystate[SDL_SCANCODE_LSHIFT]))camera_locked=false;
    if(camera_locked==true){
        cam.vx=0; cam.vy=0; cam.vz=0;
        for(int i=0;i<p.size();i++){
            if(p[i].name==camera_lockin){
                cam.x=p[i].x; cam.y=p[i].y; cam.z=p[i].z;
                return;
            }
        }
    }
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
}
void position_movement(vector<particle> &planet,vector<double> &newx,vector<double> &newy,vector<double> &newz,vector<double> &newax,vector<double> &neway,vector<double> &newaz,bool roche,vector<particle> &roc){

    for(int cur=0;cur<planet.size();cur++){
        newx[cur]=planet[cur].x+planet[cur].vx*dt+0.5*planet[cur].ax*dt*dt;
        newy[cur]=planet[cur].y+planet[cur].vy*dt+0.5*planet[cur].ay*dt*dt;
        newz[cur]=planet[cur].z+planet[cur].vz*dt+0.5*planet[cur].az*dt*dt;
    }
    long fsize=planet.size();
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
                disintegrate4(planet,roc,rocheR,r,cur,dx,dy,dz,disK);
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
}
void save(vector<particle> p){
//    const char* homeDir=getenv("HOME");
    string x="/Users/stevewilliam/1.txt";
    ofstream file("/Users/stevewilliam/1.txt");
    file<<p.size()<<endl;
    for(int i=0;i<p.size();i++){
        file<<p[i].name<<' '
            <<p[i].x<<' '<<p[i].y<<' '<<p[i].z<<' '
            <<p[i].vx<<' '<<p[i].vy<<' '<<p[i].vz<<' '
            <<p[i].ax<<' '<<p[i].ay<<' '<<p[i].az<<' '
            <<p[i].mass<<' '<<p[i].radius<<' '
            <<p[i].c1<<' '<<p[i].c2<<' '<<p[i].c3<<' '<<p[i].c4<<endl;
    }
    file.close();
}
void execute(vector<particle>& p,string command,bool& pause,bool& roche,bool& debug,bool& spaceshipmode,camera& cam,string& camera_lockin,bool& camera_locked){
    stringstream ssin(command);
    string n;
    ssin>>n;
    
    if(n=="-3"||n=="pause"){
        pause=!pause;
        return;
    }
    if(n=="-4"||n=="roche"){
        roche=!roche;
        return;
    }
    if(n=="-5"||n=="newrochek"){
        double newrocheK;
        ssin>>newrocheK;
        rocheK=newrocheK;
        return;
    }
    if(n=="-6"||n=="newg"){
        double newG;
        ssin>>newG;
        G=newG;
        return;
    }
    if(n=="-7"||n=="newdnum"){
        double newdisK;
        ssin>>newdisK;
        disK=newdisK;
        return;
    }
    if(n=="-8"||n=="newdsize"){
        double newdisnum;
        ssin>>newdisnum;
        disintegrate_num=newdisnum;
        return;
    }
    if(n=="debug"){
        debug=!debug;
        return;
    }
    if(n=="clear"){
        p.clear();
        return;
    }
    if(n=="spaceshipmode"){
        spaceshipmode=!spaceshipmode;
        return;
    }
    if(n=="goto"){
        string goname;
        ssin>>goname;
        for(int i=0;i<p.size();i++){
            if(goname==p[i].name){
                cam.x=p[i].x;
                cam.y=p[i].y;
                cam.z=p[i].z;
                cam.vx=0;
                cam.vy=0;
                cam.vz=0;
            }
        }
        return;
    }
    if(n=="target"){
        string target;
        ssin>>target;
        camera_lockin=target;
        for(int i=0;i<p.size();i++){
            if(target==p[i].name){
                camera_locked=true;
                break;
            }
        }
        return;
    }
    if(n=="save"){
        save(p);
        return;
    }
    
    for(int i=0;i<stoi(n);i++){
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
double Screen(particle& p,camera& cam){
    double dx=p.x-cam.x;
    double dy=p.y-cam.y;
    double dz=p.z-cam.z;

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
void drawCircle3(SDL_Renderer* ren,projection p,double fov_deg){
    double fov_rad=fov_deg*(M_PI/180.0);
    double focal_length=height/(2.0*tan(fov_rad*0.5));
    int r=(int)(p.r*focal_length/p.sz);
    if(r<=1)r=1;
    if(r>2000)r=2000;
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
void disintegrate4(vector<particle>& p,vector<particle>& roc,double rocheR,double r,int cur,double dx,double dy,double dz,double K){
    if(p[cur].radius<=disintegrate_num)return;
    int genp=(int)K*(rocheR/r)*dt;
    // double expectgen=K*(rocheR-r)*dt;
    // int genp=(int)(-r+400);
    // int genp=(int)expectgen;
    //Original
    double p0x=p[cur].mass*p[cur].vx;
    double p0y=p[cur].mass*p[cur].vy;
    double p0z=p[cur].mass*p[cur].vz;
    double m0=p[cur].mass;
    //Pieces
    double frag_r=disintegrate_num;
    double frag_m=p[cur].mass*pow(frag_r/p[cur].radius,3);

    double pfragx=0,pfragy=0,pfragz=0;
    for(int i=0;i<genp;i++){
        double uni=sqrt(dx*dx+dy*dy+dz*dz);
        double udx=dx/uni;
        double udy=dy/uni;
        double udz=dz/uni;

        double ux=2.0*rand()/RAND_MAX-1.0; //[-1.0,1.0]
        double uy=2.0*rand()/RAND_MAX-1.0; //[-1.0,1.0]
        double uz=2.0*rand()/RAND_MAX-1.0; //[-1.0,1.0]
        double len=sqrt(ux*ux+uy*uy+uz*uz);
        if(len<1e-6)continue; //Vector 0
        ux/=len; uy/=len; uz/=len;
        ux+=udx; uy+=udy; uz+=udz;

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
string toScientific(double val,int precision){
    stringstream ss;
    ss<<scientific<<setprecision(precision)<<val;
    return ss.str();
}
