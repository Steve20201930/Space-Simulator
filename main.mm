#include <iostream>
#include <unistd.h>
#include <vector>
#include <SDL3/SDL.h>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <sstream>
#include <fstream>
#include <random>
#include <iomanip>
#include <algorithm>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <SDL3_ttf/SDL_ttf.h>
#import <simd/simd.h>
#import <Metal/Metal.h>
using namespace std;

double G=6.674e-11;
double dt=0.01;
const double ess=1e-12;
int width=1512;
int height=900;
//int width=800;
//int height=600;
double rocheK=1.26;
int disintegrate_num=1;
double disK=2.0;
int RTXReflect = 1;
struct particle{
    string name;
    double x,y,z;
    double vx,vy,vz;
    double ax,ay,az;
    double mass;
    double radius;
    int c1,c2,c3,c4;
    bool stellar;
    bool merged;
    double w = 2 * M_PI / 86400;
};
struct camera{
    double x,y,z;
    double yaw,pitch;
    double fov;
    double vx,vy,vz;
}cam;
struct projection{
    string name;
    double x,y,z;
    double sx,sy,sz;
    int c1,c2,c3,c4;
    double r;
    bool stellar;
}res;
struct SpaceShip{
    bool initial=false;
    string name="SpaceX";
    double x,y,z;
    double vx=0,vy=0,vz=0;
    double ax=0,ay=0,az=0;
    double yaw=0,pitch=0;
    
    double mass=120000;
    int engine_count=9;
    double single_engine_thrust=2745000;
    double single_engine_consumption=0.1;
    float fuel=1200000;
    float throttle=0.0;
    
    bool target=false;
    string target_name="";
    float distance=0;
}myship;
id<MTLDevice> device=MTLCreateSystemDefaultDevice();
id<MTLCommandQueue> queuea=[device newCommandQueue];

id<MTLLibrary> lib=[device newDefaultLibrary];
id<MTLFunction> func=[lib newFunctionWithName:@"comp"];
id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:func error:nil];
bool isNumber(string& str);
double Screen(particle& p,camera& cam);
void drawCircle3(SDL_Renderer* ren,projection p,double fov_deg,bool Target);
int findFocus(vector<particle>& planet,camera& cam);
void disintegrate4(vector<particle>& p,vector<particle>& roc,double rocheR,double r,int cur,double dx,double dy,double dz,double K);
void merge(vector<particle>& p,int cur,int comp);
void execute(vector<particle>& p,string command,bool& pause,bool& roche,bool& debug,bool& SpaceShipMode,camera &cam,string& camera_lockin,bool& camera_locked,SDL_Window* win,bool& shader,bool& isMetal,bool& AutoDTime,long& second,float& fps,int& perform, bool& rtx, bool& stand, string& standon);
void save(vector<particle> p);
void camera_movement(camera &cam,double friction,double acceleration,bool& camera_locked,string camera_lockin,vector<particle> p, bool &stand, string standon, bool pause);
void position_movement(vector<particle> &planet,vector<double> &newx,vector<double> &newy,vector<double> &newz,vector<double> &newax,vector<double> &neway,vector<double> &newaz,bool roche,vector<particle> &roc,bool SpaceShipMode);
void position_movement_METAL(vector<particle> &planet,vector<double> &newx,vector<double> &newy,vector<double> &newz,vector<double> &newax,vector<double> &neway,vector<double> &newaz,bool roche,vector<particle> &roc,bool SpaceShipMode);
string toScientific(double val,int precision);
void YourSpaceShip(camera& cam,SDL_Event& event,SDL_Window* win,SDL_Renderer* ren,bool& pause,int iniline);
void InMyShip(camera& cam,SDL_Event& event,SDL_Window* win,SDL_Renderer* ren,int iniline,bool& targetisTyping,vector<particle> p,bool pause,bool debug,TTF_Font* font);
void AutoOrbitSystem(bool first_lockedC,vector<particle> p);
void DrawTrajectorySystem(SDL_Renderer* ren,double proj,vector<particle> p, camera& cam);
void DrawCircleShader(SDL_Renderer* ren,projection p,double fov_deg,bool Target,double lx,double ly,double lz,double stellar_intensity);
void DrawTTF(SDL_Renderer* ren, TTF_Font* font, const char* text, float centerX, float centerY, SDL_Color col,float maxWidth,float x_scale,float y_scale);
void DrawStellarShader(SDL_Renderer* ren, projection p, double fov_deg);
/*RTX Render Method*/
struct RTXV{
    double x, y, z;
    bool intersect = true;
    double OM = 0;
};
vector<vector<RTXV>> RTXOriginVector(height, vector<RTXV>(width, {0, 0, 0}));
vector<vector<double>> RTXVectorLength(height, vector<double>(width, 0));
vector<vector<RTXV>> RTXVector(height, vector<RTXV>(width, {0, 0, 0}));
void RTXInitial();
void RTXRender(SDL_Renderer* ren, const vector<projection> &planet, const vector<particle> &parti, const vector<int> &suns);
void RTXRenderMetal(SDL_Renderer* ren, const vector<projection> &planet, const vector<particle> &parti, const vector<int> &suns);
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
    SDL_Window* win=SDL_CreateWindow("Space Sim ver2.3",width,height,SDL_WINDOW_RESIZABLE);
    SDL_Renderer* ren=SDL_CreateRenderer(win,nullptr);
    SDL_SetWindowPosition(win,SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED);
    SDL_SetWindowRelativeMouseMode(win, true);
    TTF_Init();
    const char* base = SDL_GetBasePath();
    string full_path=string(base)+"font.ttf";
    TTF_Font* font=TTF_OpenFont(full_path.c_str(),40);
    const double mousesensitivity=0.1;
    long second=1.0;
    int perform=1;
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
    bool SpaceShipMode=false;
    bool AutoDTime=true;
    long accelerate=1;
    bool stand = false;
    string standon = "";
    //Inital input&SpaceShip Initial input
    bool isTyping=false;
    bool shipisTyping=false;
    bool targetisTyping=false;
    string commandbuf="";
    vector<string> currentbuf;
    int line=0;
    int fontsize=24;
    //Initial Computing Methods
    bool isMetal=false;
    //Renderer
    bool shader=false;
    bool rtx = false;
    //Initial RTX parameters
    int last_width=-1, last_height=-1;

    while(running){
        int initialT=0.7*height;
        int iniline=initialT;
        newx.resize(planet.size());
        newy.resize(planet.size());
        newz.resize(planet.size());
        newax.resize(planet.size());
        neway.resize(planet.size());
        newaz.resize(planet.size());
        SDL_PollEvent(&event);
        SDL_GetWindowSize(win,&width,&height);
        if(last_width != width || last_height != height){
            RTXInitial();
            last_width = width;
            last_height = height;
        }
        
        switch(event.type){
            case SDL_EVENT_QUIT:{
                running=false;
                break;
            }case SDL_EVENT_KEY_DOWN:{
                if(isTyping==false&&shipisTyping==false&&targetisTyping==false){
                    if(event.key.key==SDLK_UP){
                        if(AutoDTime&&accelerate<9999999999999999){
                            second+=accelerate;
                            accelerate+=5;
                        }
                        else dt*=10;
                    }else if(event.key.key==SDLK_DOWN){
                        if(AutoDTime&&accelerate<9999999999999999){
                            second-=accelerate;
                            accelerate+=5;
                        }
                        else dt*=0.1;
                    }else if(event.key.key==SDLK_LEFT){
                        cam.fov=max(20.0,cam.fov-10.0);
                    }else if(event.key.key==SDLK_RIGHT){
                        cam.fov=min(160.0,cam.fov+10.0);
                    }else if(event.key.key==SDLK_P){
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
                        if(SpaceShipMode)pause=false;
                    }else{
                        mouse_locked=false;
                        SDL_SetWindowRelativeMouseMode(win, false);
                    }
                }
                break;
            }
            case SDL_EVENT_KEY_UP:{
                if(isTyping==false&&shipisTyping==false&&targetisTyping==false){
                    if(event.key.key==SDLK_UP){
                        accelerate=1;
                    }
                    if(event.key.key==SDLK_DOWN){
                        accelerate=1;
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
        
        //COMPUTE POSITION t+dt
        if(pause==false){
            if(isMetal==true)position_movement_METAL(planet,newx,newy,newz,newax,neway,newaz,roche,roc,SpaceShipMode);
            else position_movement(planet,newx,newy,newz,newax,neway,newaz,roche,roc,SpaceShipMode);
        }
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
        
        //RENDER
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
            if(res.sz>1e-1){
                double fl=height/(2.0*tan(cam.fov*(M_PI/180.0)*0.5));
                double rr=planet[cur].radius*fl/res.sz;
                if(rr<1)rr=1; if(rr>2000)rr=2000;
                if(res.sx+rr<0||res.sx-rr>width||res.sy+rr<0||res.sy-rr>height) continue;
                rendering.push_back({planet[cur].name,planet[cur].x,planet[cur].y,planet[cur].z,res.sx,res.sy,res.sz,planet[cur].c1,planet[cur].c2,planet[cur].c3,planet[cur].c4,planet[cur].radius,planet[cur].stellar});
            }
        }
        sort(rendering.begin(),rendering.end(),[](const projection& a,const projection&b){
            return a.sz>b.sz;
        });
        bool target=false;
        if(rtx == true){
            vector<int> suns;
            for(int i = 0;i < planet.size();i++){
                if(planet[i].stellar){
                    suns.push_back(i);
                    if(perform < 3){
                        break;
                    }
                }
            }
            if(isMetal)RTXRenderMetal(ren, rendering, planet, suns);
            else RTXRender(ren, rendering, planet, suns);
            
            for(int i = 0;i < rendering.size();i++){
                if(rendering[i].stellar)DrawStellarShader(ren, rendering[i], cam.fov);
            }
        }else{
            for(int cur=0;cur<rendering.size();cur++){
                if(SpaceShipMode&&myship.target&&myship.target_name==rendering[cur].name)target=true;
                else target=false;
                if(!shader){
                    SDL_SetRenderDrawColor(ren,rendering[cur].c1,rendering[cur].c2,rendering[cur].c3,rendering[cur].c4);
                    drawCircle3(ren,rendering[cur],cam.fov,target);
                }
                if(shader){
                    if(rendering[cur].stellar) {
                        DrawStellarShader(ren, rendering[cur], cam.fov);
                        //                    SDL_SetRenderDrawColor(ren,rendering[cur].c1,rendering[cur].c2,rendering[cur].c3,rendering[cur].c4);
                        //                    DrawStellarShader(ren, rendering[cur], cam.fov,target);
                        //                    drawCircle3(ren,rendering[cur],cam.fov,target);
                        continue;
                    }
                    if(sqrt((cam.x-rendering[cur].x)*(cam.x-rendering[cur].x)+(cam.y-rendering[cur].y)*(cam.y-rendering[cur].y)+(cam.z-rendering[cur].z))>1e16&&perform==1){
                        SDL_SetRenderDrawColor(ren,rendering[cur].c1,rendering[cur].c2,rendering[cur].c3,rendering[cur].c4);
                        drawCircle3(ren,rendering[cur],cam.fov,target);
                        continue;
                    }
                    DrawCircleShader(ren, rendering[cur], cam.fov, target, 0, 0, 0, 0);
                    double stellar_intensity=0;
                    double lx = 0, ly = 0, lz = 0;
                    double dr=ess;
                    for(int i=0;i<planet.size();i++){
                        if(planet[i].stellar==true){
                            double drx=planet[i].x-rendering[cur].x;
                            double dry=planet[i].y-rendering[cur].y;
                            double drz=planet[i].z-rendering[cur].z;
                            dr=sqrt(drx*drx+dry*dry+drz*drz)+ess;
                            //IDK WHY BUT THIS VECTOR IS GOOD TO WORK
                            
                            double a=cam.yaw*(M_PI/180.0);
                            double b=cam.pitch*(M_PI/180.0);
                            double rx = drx * cos(a) - drz * sin(a);
                            double rz1 = drx * sin(a) + drz * cos(a);
                            double ry = dry * cos(b) - rz1 * sin(b);
                            double rz = dry * sin(b) + rz1 * cos(b);
                            double view_dist = sqrt(rx * rx + ry * ry + rz * rz) + ess;
                            double k = 0.05;
                            
                            float d = dr / planet[i].radius;
                            double I0 = 1000000.0;
                            double physical_intensity = I0 / (d * d + 1.0);
                            float scale = 2.0f;
                            stellar_intensity = max(1.0,log10(physical_intensity * scale + 1.0) / log10(I0 * scale + 1.0));
                            
                            lx += stellar_intensity * rx / view_dist;
                            ly += stellar_intensity * -ry / view_dist;
                            lz += stellar_intensity * -rz / view_dist;
                            SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_ADD);
                            DrawCircleShader(ren, rendering[cur], cam.fov, target, lx, ly, lz, stellar_intensity);
                            if(perform<=2){
                                break;
                            }
                        }
                    }
                    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_NONE);
                }
            }
        }
        /*COMPUTE MOVEMENT*/
        if(isTyping==false){
            if(SpaceShipMode==false)camera_movement(cam,friction,acceleration,camera_locked,camera_lockin,planet,stand,standon, pause);
            if(SpaceShipMode==true){
                if(myship.initial==false){
                    shipisTyping=true;
                    YourSpaceShip(cam, event, win, ren, pause, iniline);
                }else{
                    shipisTyping=false;
                    InMyShip(cam, event, win, ren, iniline, targetisTyping, planet, pause, debug, font);
                }
                
            }
        }
        /*Console Input*/
        if(isTyping==true){
            if(SpaceShipMode)pause=true;
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
                    execute(planet,commandbuf,pause,roche,debug,SpaceShipMode,cam,camera_lockin,camera_locked,win,shader,isMetal,AutoDTime,second,fps,perform,rtx, stand,standon);
                    isTyping=false;
                    SDL_StopTextInput(win);
                    if(SpaceShipMode)pause=false;
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
            if(AutoDTime)dt=second/fps;
            string debug_buf1="X:"+to_string(cam.x)+" Y:"+to_string(cam.y)+" Z:"+to_string(cam.z)+"  YAW:"+to_string(cam.yaw)+" PITCH:"+to_string(cam.pitch)+"    Quantity:"+to_string(planet.size());
            if(AutoDTime){
                debug_buf1+=" Scale:";
                long minutes=second/60;
                long hours=second/3600;
                long days=second/86400;
                long years=second/31536000;
                if(years!=0)debug_buf1+=to_string(years)+"y ";
                if(days!=0)debug_buf1+=to_string(days%365)+"d ";
                if(hours!=0)debug_buf1+=to_string(hours%24)+"hr ";
                if(minutes!=0)debug_buf1+=to_string(minutes%60)+"min ";
                debug_buf1+=to_string(second%60)+"s /s";
            }
            string debug_buf2="RunSpeed(dt/s):"+to_string(dt)+" acceleration:"+to_string(acceleration)+" FOV:"+to_string(cam.fov)+" fps:"+to_string(fps);
            string debug_buf3="RocheLimit:["+to_string(roche)+"] "+to_string(rocheK);
            string debug_buf4="G:"+toScientific(G, 4);
            string debug_buf5="SpaceShipMode:"+to_string(SpaceShipMode);
            string debug_buf6="Shader: "+to_string(shader)+" RTX: "+to_string(rtx);
            if(pause==true)debug_buf1+="  [PAUSE]";
            SDL_SetRenderDrawColor(ren,255,255,255,255);
            SDL_RenderDebugText(ren,20,10,debug_buf1.c_str());
            SDL_RenderDebugText(ren,20,22,debug_buf2.c_str());
            SDL_RenderDebugText(ren,20,34,debug_buf3.c_str());
            SDL_RenderDebugText(ren,20,46,debug_buf4.c_str());
            SDL_RenderDebugText(ren,20,58,debug_buf5.c_str());
            SDL_RenderDebugText(ren,20,70,debug_buf6.c_str());
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
double standon_theta = 0;
void camera_movement(camera &cam,double friction,double acceleration,bool& camera_locked,string camera_lockin,vector<particle> p, bool &stand, string standon, bool pause){
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
    if(stand==true&&(keystate[SDL_SCANCODE_W]||keystate[SDL_SCANCODE_S]||keystate[SDL_SCANCODE_A]||keystate[SDL_SCANCODE_D]||keystate[SDL_SCANCODE_SPACE]||keystate[SDL_SCANCODE_LSHIFT]))stand=false;
    if(camera_locked==true){
        cam.vx=0; cam.vy=0; cam.vz=0;
        for(int i=0;i<p.size();i++){
            if(p[i].name==camera_lockin){
                cam.x=p[i].x; cam.y=p[i].y; cam.z=p[i].z;
                return;
            }
        }
    }
    if(stand==true){
        cam.vx=0; cam.vy=0; cam.vz=0;
        for(int i=0;i<p.size();i++){
            if(p[i].name==standon){
                if(!pause)standon_theta += p[i].w * dt;
                cam.x=p[i].x + sin(standon_theta) * p[i].radius * 0.85; cam.y=p[i].y; cam.z=p[i].z + cos(standon_theta) * p[i].radius * 0.85;
                cam.yaw = 180 * (standon_theta - int(standon_theta / (2 * M_PI)) * 2 * M_PI) / M_PI;
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
void position_movement(vector<particle> &planet,vector<double> &newx,vector<double> &newy,vector<double> &newz,vector<double> &newax,vector<double> &neway,vector<double> &newaz,bool roche,vector<particle> &roc,bool SpaceShipMode){
    if(SpaceShipMode){
//        myship.ax=0; myship.ay=0; myship.az=0;
        planet.push_back({myship.name,myship.x,myship.y,myship.z,myship.vx,myship.vy,myship.vz,myship.ax,myship.ay,myship.az,myship.mass,1,0,0,0,0,false});
        newx.resize(planet.size());
        newy.resize(planet.size());
        newz.resize(planet.size());
        newax.resize(planet.size());
        neway.resize(planet.size());
        newaz.resize(planet.size());
    }
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
            
            //Target Name
            if(SpaceShipMode&&cur==fsize-1&&myship.target&&myship.target_name==planet[comp].name){
                myship.distance=r;
            }
            //Compute roche limit
            if(roche==true&&planet[cur].radius>disintegrate_num){
                double rocheR=rocheK*planet[cur].radius*pow(2*planet[comp].mass/planet[cur].mass,1.0/3.0);
                if(r>rocheR)continue;
                disintegrate4(planet,roc,rocheR,r,cur,dx,dy,dz,disK);
            }
            //Compute merge limit
            if(roche==true&&planet[cur].merged==false&&planet[comp].merged==false){
                if(r>(planet[comp].radius-planet[cur].radius))continue;
                merge(planet,cur,comp);
                planet[cur].merged=true;
                planet[comp].merged=true;
            }
        }
        planet[cur].vx+=0.5*(planet[cur].ax+newax[cur])*dt; planet[cur].vy+=0.5*(planet[cur].ay+neway[cur])*dt; planet[cur].vz+=0.5*(planet[cur].az+newaz[cur])*dt;
        planet[cur].x=newx[cur]; planet[cur].y=newy[cur]; planet[cur].z=newz[cur];
        planet[cur].ax=newax[cur]; planet[cur].ay=neway[cur]; planet[cur].az=newaz[cur];
        
    }
    if(SpaceShipMode){
        myship.x=planet[fsize-1].x; myship.y=planet[fsize-1].y; myship.z=planet[fsize-1].z;
        myship.vx=planet[fsize-1].vx; myship.vy=planet[fsize-1].vy; myship.vz=planet[fsize-1].vz;
        myship.ax=planet[fsize-1].ax; myship.ay=planet[fsize-1].ay; myship.az=planet[fsize-1].az;
        planet.pop_back();
    }
}

void position_movement_METAL(vector<particle> &planet,vector<double> &newx,vector<double> &newy,vector<double> &newz,vector<double> &newax,vector<double> &neway,vector<double> &newaz,bool roche,vector<particle> &roc,bool SpaceShipMode){
    
    if(SpaceShipMode){
//        myship.ax=0; myship.ay=0; myship.az=0;
        planet.push_back({myship.name,myship.x,myship.y,myship.z,myship.vx,myship.vy,myship.vz,myship.ax,myship.ay,myship.az,myship.mass,1,0,0,0,0,false});
        newx.resize(planet.size());
        newy.resize(planet.size());
        newz.resize(planet.size());
        newax.resize(planet.size());
        neway.resize(planet.size());
        newaz.resize(planet.size());
    }
    
    for(int cur=0;cur<planet.size();cur++){
        newx[cur]=planet[cur].x+planet[cur].vx*dt+0.5*planet[cur].ax*dt*dt;
        newy[cur]=planet[cur].y+planet[cur].vy*dt+0.5*planet[cur].ay*dt*dt;
        newz[cur]=planet[cur].z+planet[cur].vz*dt+0.5*planet[cur].az*dt*dt;
        planet[cur].x=newx[cur];
        planet[cur].y=newy[cur];
        planet[cur].z=newz[cur];
    }
    
    /*Scale*/
    double maxCoord = 1.0, maxMass = 1.0;
    for(int i = 0;i < planet.size();i++){
        maxCoord=max({maxCoord, fabs(planet[i].x), fabs(planet[i].y), fabs(planet[i].z)});
        maxMass = max(maxMass, planet[i].mass);
    }
    double Ls = max(1.0, maxCoord / 1e6);
    double Ms = max(1.0, maxMass / 1e6);
    float nG = (float)(G*Ms/(Ls*Ls));
    
    long N=planet.size();
    float ndt=dt;
    
    if(N<=0)return;
    
    static id<MTLBuffer> s_bufnx, s_bufny, s_bufnz, s_bufvx, s_bufvy, s_bufvz,s_bufax, s_bufay, s_bufaz, s_bufmass, s_bufradius;
    static long s_lastCapacity = 0;
    
    if (N > s_lastCapacity) {
        long newCapacity = N + 500;
        s_bufnx = [device newBufferWithLength:sizeof(float)*newCapacity options:MTLResourceStorageModeShared];
        s_bufny = [device newBufferWithLength:sizeof(float)*newCapacity options:MTLResourceStorageModeShared];
        s_bufnz = [device newBufferWithLength:sizeof(float)*newCapacity options:MTLResourceStorageModeShared];
        s_bufvx = [device newBufferWithLength:sizeof(float)*newCapacity options:MTLResourceStorageModeShared];
        s_bufvy = [device newBufferWithLength:sizeof(float)*newCapacity options:MTLResourceStorageModeShared];
        s_bufvz = [device newBufferWithLength:sizeof(float)*newCapacity options:MTLResourceStorageModeShared];
        s_bufax = [device newBufferWithLength:sizeof(float)*newCapacity options:MTLResourceStorageModeShared];
        s_bufay = [device newBufferWithLength:sizeof(float)*newCapacity options:MTLResourceStorageModeShared];
        s_bufaz = [device newBufferWithLength:sizeof(float)*newCapacity options:MTLResourceStorageModeShared];
        s_bufmass = [device newBufferWithLength:sizeof(float)*newCapacity options:MTLResourceStorageModeShared];
        s_bufradius = [device newBufferWithLength:sizeof(float)*newCapacity options:MTLResourceStorageModeShared];
        s_lastCapacity = newCapacity;
    }
    
    vector<float> nx(N), ny(N), nz(N), vx(N), vy(N), vz(N), ax(N), ay(N), az(N), mass(N), radius(N);
    
    for(int i=0;i<planet.size();i++){
        nx[i]=(float)(planet[i].x/Ls);
        ny[i]=(float)(planet[i].y/Ls);
        nz[i]=(float)(planet[i].z/Ls);
        vx[i]=planet[i].vx;
        vy[i]=planet[i].vy;
        vz[i]=planet[i].vz;
        ax[i]=planet[i].ax;
        ay[i]=planet[i].ay;
        az[i]=planet[i].az;
        mass[i]=(float)(planet[i].mass/Ms);
        radius[i]=planet[i].radius;
    }
    
    memcpy(s_bufnx.contents, nx.data(), sizeof(float)*N);
    memcpy(s_bufny.contents, ny.data(), sizeof(float)*N);
    memcpy(s_bufnz.contents, nz.data(), sizeof(float)*N);
    memcpy(s_bufvx.contents, vx.data(), sizeof(float)*N);
    memcpy(s_bufvy.contents, vy.data(), sizeof(float)*N);
    memcpy(s_bufvz.contents, vz.data(), sizeof(float)*N);
    memcpy(s_bufax.contents, ax.data(), sizeof(float)*N);
    memcpy(s_bufay.contents, ay.data(), sizeof(float)*N);
    memcpy(s_bufaz.contents, az.data(), sizeof(float)*N);
    memcpy(s_bufmass.contents, mass.data(), sizeof(float)*N);
    memcpy(s_bufradius.contents, radius.data(), sizeof(float)*N);
    
    id<MTLCommandBuffer> cmd=[queuea commandBuffer];
    id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
    
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:s_bufnx offset:0 atIndex:0];
    [enc setBuffer:s_bufny offset:0 atIndex:1];
    [enc setBuffer:s_bufnz offset:0 atIndex:2];
    [enc setBuffer:s_bufvx offset:0 atIndex:3];
    [enc setBuffer:s_bufvy offset:0 atIndex:4];
    [enc setBuffer:s_bufvz offset:0 atIndex:5];
    [enc setBuffer:s_bufax offset:0 atIndex:6];
    [enc setBuffer:s_bufay offset:0 atIndex:7];
    [enc setBuffer:s_bufaz offset:0 atIndex:8];
    [enc setBytes:&ndt length:sizeof(float) atIndex:9];
    [enc setBuffer:s_bufmass offset:0 atIndex:10];
    [enc setBuffer:s_bufradius offset:0 atIndex:11];
    [enc setBytes:&N length:sizeof(long) atIndex:12];
    [enc setBytes:&nG length:sizeof(float) atIndex:13];
    [enc setBytes:&SpaceShipMode length:sizeof(bool) atIndex:14];
    
    MTLSize gridSize=MTLSizeMake(N,1,1);
    NSUInteger maxThreads = pipeline.maxTotalThreadsPerThreadgroup;
    MTLSize threadGroupSize=MTLSizeMake(maxThreads,1,1);
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    
    float* rvx=(float*)s_bufvx.contents;
    float* rvy=(float*)s_bufvy.contents;
    float* rvz=(float*)s_bufvz.contents;
    float* rax=(float*)s_bufax.contents;
    float* ray=(float*)s_bufay.contents;
    float* raz=(float*)s_bufaz.contents;
    
    for(int i=0;i<N;i++){
        //Backput Position
//        planet[i].x=nx[i];
//        planet[i].y=ny[i];
//        planet[i].z=nz[i];
        //Backput Velocity
        planet[i].vx=rvx[i];
        planet[i].vy=rvy[i];
        planet[i].vz=rvz[i];
        //Backput Accelerate
        planet[i].ax=rax[i];
        planet[i].ay=ray[i];
        planet[i].az=raz[i];
    }
    if(SpaceShipMode){
        for(int cur=0;cur<N-1;cur++){
            if(SpaceShipMode&&myship.target&&myship.target_name==planet[cur].name){
                double dx=myship.x-planet[cur].x;
                double dy=myship.y-planet[cur].y;
                double dz=myship.z-planet[cur].z;
                double r=sqrt(dx*dx+dy*dy+dz*dz+ess);
                myship.distance=r;
            }
        }
        myship.x=planet[N-1].x; myship.y=planet[N-1].y; myship.z=planet[N-1].z;
        myship.vx=planet[N-1].vx; myship.vy=planet[N-1].vy; myship.vz=planet[N-1].vz;
        myship.ax=planet[N-1].ax; myship.ay=planet[N-1].ay; myship.az=planet[N-1].az;
        planet.pop_back();
    }
}

void save(vector<particle> p){
    string path=string(getenv("HOME"))+"/Documents/running.txt";
    ofstream file(path);
    file<<p.size()<<endl;
    for(int i=0;i<p.size();i++){
        file<<p[i].name<<' '
            <<p[i].x<<' '<<p[i].y<<' '<<p[i].z<<' '
            <<p[i].vx<<' '<<p[i].vy<<' '<<p[i].vz<<' '
            <<p[i].ax<<' '<<p[i].ay<<' '<<p[i].az<<' '
            <<p[i].mass<<' '<<p[i].radius<<' '
            <<p[i].c1<<' '<<p[i].c2<<' '<<p[i].c3<<' '<<p[i].c4<<' ';
        if(p[i].stellar==true){
            file<<1<<endl;
        }else{
            file<<0<<endl;
        }
    }
    file.close();
}
void execute(vector<particle>& p,string command,bool& pause,bool& roche,bool& debug,bool& SpaceShipMode,camera& cam,string& camera_lockin,bool& camera_locked,SDL_Window* win,bool& shader,bool& isMetal,bool& AutoDTime,long& second,float& fps,int& perform, bool& rtx, bool& stand, string& standon){
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
        SpaceShipMode=false;
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
    if(n=="standon"){
        string standk;
        ssin>>standk;
        for(int i = 0;i < p.size();i++){
            if(standk==p[i].name){
                stand=true;
                standon=standk;
                break;
            }
        }
    }
    if(n=="save"){
        save(p);
        return;
    }
    if(n=="spaceshipmode"){
        SpaceShipMode=!SpaceShipMode;
        myship.x=cam.x; myship.y=cam.y; myship.z=cam.z;
        return;
    }
    if(n=="clearmyship"){
        SpaceShipMode=false;
        myship.initial=false;
        return;
    }
    if(n=="in"){
        string method;
        ssin>>method;
        if(method=="Metal"){
            isMetal=true;
        }
    }
    if(n=="out"){
        string method;
        ssin>>method;
        if(method=="Metal"){
            isMetal=false;
        }
    }
    if(n=="perform"){
        string per;
        ssin>>per;
        if(per=="1"){
            perform=1;
        }else if(per=="2"){
            perform=2;
        }else if(per=="3"){
            perform=3;
        }
    }
    if(n=="sel"){
        string name;
        ssin>>name;
        if(name=="solar"||name=="solar2"||name=="solar3"){
            p.push_back({"Sun", 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.9885e30, 695700000, 255, 210, 26, 255, true});
            p.push_back({"Mercury", 6.98179e10, 0, 0, 0, 0, 47870, 0, 0, 0, 3.3022e23, 2440000, 194, 194, 194, 255, false});
            p.push_back({"Venus", 1.08939e11, 0, 0, 0, 0, 35020, 0, 0, 0, 4.8676e24, 6051000, 255, 236, 59, 255, false});
            p.push_back({"Earth", 1.521e11, 0, 0, 0, 0, 29780, 0, 0, 0, 5.97237e24, 6.371e6, 28, 115, 232, 255, false});
            p.push_back({"Mars", 2.49261e11, 0, 0, 0, 0, 24080, 0, 0, 0, 6.4169e23, 3.3895e6, 245, 91, 22, 255, false});
            p.push_back({"Jupiter", 8.165208e11, 0, 0, 0, 0, 13070, 0, 0, 0, 1.8981e27, 6.9886e7, 255, 213, 125, 255, false});
            p.push_back({"Saturnus", 1.353572956e12, 0, 0, 0, 0, 9690, 0, 0, 0, 5.6846e26, 5.8232e7, 219, 167, 61, 255, false});
            p.push_back({"Uranus", 3.004419704e12, 0, 0, 0, 0, 6810, 0, 0, 0, 8.6810e25, 2.5559e7, 215, 248, 250, 255, false});
            p.push_back({"Neptune", 4.55394649e12, 0, 0, 0, 0, 5430, 0, 0, 0, 1.0243e26, 2.4764e7, 65, 105, 225, 255, false});
            p.push_back({"Moon", 1.51736896e11, 0, 0, 0, 0, 30802, 0, 0, 0, 7.3477e22, 1.7371e6, 194, 194, 194, 255, false});
            if(name=="solar2"||name=="solar3"){
                p.push_back({"MarsI",2.4925188e11,0,0,0,0,26229.2183,0,0,0,1.8e15,1.11e4,160,160,160,255,false});
                p.push_back({"MarsII",2.4923754e11,0,0,0,0,25428.3773,0,0,0,1.48e15,6.2e3,163,163,163,255,false});
                p.push_back({"J1",8.160991e11,0,0,0,0,30366.991,0,0,0,8.94e22,1.8213e6,164,164,164,255,false});
                p.push_back({"J2",8.158499e11,0,0,0,0,26783.347,0,0,0,4.799844e22,1.5608e6,165,165,165,255,false});
                p.push_back({"J3",8.154504e11,0,0,0,0,23926.737,0,0,0,1.4819e23,2.6431e6,166,166,166,255,false});
                p.push_back({"J4",8.146381e11,0,0,0,0,19771.37,0,0,0,1.075938e23,2.4103e6,168,168,168,255,false});
                p.push_back({"J5",8.1633944e11,0,0,0,0,39445.13,0,0,0,2.08e18,8.35e4,170,170,170,255,false});
                p.push_back({"J6",8.034388e11,0,0,0,0,16175.526,0,0,0,4.2e18,1.414e5,172,172,172,255,false});
                p.push_back({"J7",8.034388e11,0,0,0,0,16348.221,0,0,0,8.7e17,4.3e4,174,174,174,255,false});
                if(name=="solar3"){
                    double s_x = 1.353572956e12, s_y = 0, s_z = 0;
                    double s_vx = 0, s_vy = 0, s_vz = 9690;
                    double s_m = 5.6846e26, s_r = 5.8232e7;
                    double tilt = 26.73 * (M_PI / 180.0);
                    int ring_count = 10000;
                    random_device rd;
                    mt19937 gen(rd());
                    uniform_real_distribution<> dis(0.0, 1.0);
                    for(int i=0;i<ring_count;i++){
                        //Ring's radius
                        double norm_dist = dis(gen);
                        double r_orbit = s_r * (1.3 + 1.2 * norm_dist);
                        double angle = dis(gen) * 2.0 * M_PI;
                        //Ring's texture
                        double ring_pattern = sin(norm_dist * 20.0) * 0.5 + 0.5;
                        double gap_shading = (norm_dist > 0.65 && norm_dist < 0.7) ? 0.2 : 1.0;
                        double final_shade = (0.6 + 0.4 * ring_pattern) * gap_shading;
                        //Ring's thickness
                        double rx_0 = cos(angle) * r_orbit;
                        double rz_0 = sin(angle) * r_orbit;
                        double ry_0 = (dis(gen) - 0.5) * s_r * 0.01;

                        double v_mag = sqrt(G * s_m / r_orbit);
                        double rvx_0 = -sin(angle) * v_mag;
                        double rvz_0 = cos(angle) * v_mag;
                        double rvy_0 = 0;
                        //Axis X Changing
//                        double rx = rx_0;
//                        double ry = ry_0 * cos(tilt) - rz_0 * sin(tilt);
//                        double rz = ry_0 * sin(tilt) + rz_0 * cos(tilt);
//                        
//                        double rvx = rvx_0;
//                        double rvy = rvy_0 * cos(tilt) - rvz_0 * sin(tilt);
//                        double rvz = rvy_0 * sin(tilt) + rvz_0 * cos(tilt);
                        double rx = rx_0 * cos(tilt) - ry_0 * sin(tilt);
                        double ry = rx_0 * sin(tilt) + ry_0 * cos(tilt);
                        double rz = rz_0;
                        
                        double rvx = rvx_0 * cos(tilt) - rvy_0 * sin(tilt);
                        double rvy = rvx_0 * sin(tilt) + rvy_0 * cos(tilt);
                        double rvz = rvz_0;
                        //Color
                        int cr = (int)(219 * final_shade);
                        int cg = (int)(167 * final_shade);
                        int cb = (int)(100 * final_shade);
                        
                        p.push_back({"Ring_Part", s_x + rx, s_y + ry, s_z + rz, s_vx + rvx, s_vy + rvy, s_vz + rvz,0, 0, 0, 1e15, 300000.0, (Uint8)cr, (Uint8)cg, (Uint8)cb, 255, false});
                    }
                    
                    int asteroid_count = 2000;
                    double G = 6.67430e-11;
                    double sun_m = 1.9885e30;

                    for(int i = 0; i < asteroid_count; i++) {
                        //2.1AU-3.3AU
                        double r_orbit = (3.1e11 + dis(gen) * 1.7e11);
                        double angle = dis(gen) * 2.0 * M_PI;
                        
                        //Random orbit
                        double vertical_dist = (dis(gen) - 0.5) * 1e10;

                        double ax = cos(angle) * r_orbit;
                        double az = sin(angle) * r_orbit;
                        double ay = vertical_dist;

                        double v_mag = sqrt(G * sun_m / r_orbit);
                        double avx = -sin(angle) * v_mag;
                        double avz = cos(angle) * v_mag;
                        double avy = (dis(gen) - 0.5) * 500;//Random speed

                        int gray = 100 + (int)(dis(gen) * 60);
                        
                        p.push_back({"Asteroid", ax, ay, az, avx, avy, avz, 0, 0, 0, 1e16, 400000.0, (Uint8)gray, (Uint8)gray, (Uint8)gray, 255, false});
                    }
                }
                
            }
        }
        if(name=="J2000"){
            p.push_back({"Sun",0,0,0,0,0,0,0,0,0,1.9885e30,695700000,255,210,26,255,true});
            p.push_back({"Mercury",-2.521097e10,4.403542e10,5.190143e09,-4.790189e04,-2.621833e04,2.323756e03,0,0,0,3.3022e23,2440000,194,194,194,255,false});
            p.push_back({"Venus",-1.075055e11,-3.366520e10,5.706055e09,1.074483e04,-3.519233e04,-1.055433e03,0,0,0,4.8676e24,6051000,255,236,59,255,false});
            p.push_back({"Earth",-2.650443e10,1.446262e11,-3.866346e04,-2.978611e04,-5.455937e03,3.497548e-01,0,0,0,5.97237e24,6.371e6,28,115,232,255,false});
            p.push_back({"Moon",-2.687302e10,1.449407e11,8.615146e07,-3.058929e04,-6.237986e03,8.651896e01,0,0,0,7.3477e22,1.7371e6,194,194,194,255,false});
            p.push_back({"Mars",2.079950e11,-3.143009e09,-5.201802e09,1.164770e03,2.649227e04,5.327515e02,0,0,0,6.4169e23,3.3895e6,245,91,22,255,false});
            p.push_back({"Jupiter",5.982191e11,4.391086e11,-1.523254e10,-7.901937e03,1.109340e04,1.306729e02,0,0,0,1.8981e27,6.9886e7,255,213,125,255,false});
            p.push_back({"Saturn",9.586095e11,9.825652e11,-5.522900e10,-7.049246e03,6.727673e03,1.775769e02,0,0,0,5.6846e26,5.8232e7,219,167,61,255,false});
            p.push_back({"Uranus",2.158746e12,-2.055373e12,-3.559704e10,4.663791e03,4.626879e03,-1.463484e02,0,0,0,8.6810e25,2.5559e7,215,248,250,255,false});
            p.push_back({"Neptune",2.514009e12,-3.739228e12,1.901638e10,4.466817e03,3.077484e03,-1.669711e02,0,0,0,1.0243e26,2.4764e7,65,105,225,255,false});
        }
        if(name=="milkyway1"){
            string filename = "milkyway_lite.txt";
            const char* base = SDL_GetBasePath();
            ifstream file(string(base) + filename);
            if(!file.is_open()){
                return;
            }
            string name,x,y,z;
            double vx,vy,vz,ax,ay,az,mass,radius;
            int c1,c2,c3,c4;
            bool stellar;
            while(file>>name>>x>>y>>z>>vx>>vy>>vz>>ax>>ay>>az>>mass>>radius>>c1>>c2>>c3>>c4>>stellar){
                p.push_back({name,stod(x),stod(y),stod(z),vx,vy,vz,ax,ay,az,mass,radius,c1,c2,c3,c4,stellar,false});
            }
            file.close();
        }
        if(name=="milkyway2"){
            string filename = "milkyway_medium.txt";
            const char* base = SDL_GetBasePath();
            ifstream file(string(base) + filename);
            if(!file.is_open()){
                return;
            }
            string name,x,y,z;
            double vx,vy,vz,ax,ay,az,mass,radius;
            int c1,c2,c3,c4;
            bool stellar;
            while(file>>name>>x>>y>>z>>vx>>vy>>vz>>ax>>ay>>az>>mass>>radius>>c1>>c2>>c3>>c4>>stellar){
                p.push_back({name,stod(x),stod(y),stod(z),vx,vy,vz,ax,ay,az,mass,radius,c1,c2,c3,c4,stellar,false});
            }
            file.close();
        }
        if(name=="milkyway3"){
            string filename = "milkyway_full.txt";
            const char* base = SDL_GetBasePath();
            ifstream file(string(base) + filename);
            if(!file.is_open()){
                return;
            }
            string name,x,y,z;
            double vx,vy,vz,ax,ay,az,mass,radius;
            int c1,c2,c3,c4;
            bool stellar;
            while(file>>name>>x>>y>>z>>vx>>vy>>vz>>ax>>ay>>az>>mass>>radius>>c1>>c2>>c3>>c4>>stellar){
                p.push_back({name,stod(x),stod(y),stod(z),vx,vy,vz,ax,ay,az,mass,radius,c1,c2,c3,c4,stellar,false});
            }
            file.close();
        }
        return;
    }
    if(n=="shader"){
        shader=!shader;
        rtx = false;
    }
    if(n=="rtx"){
        rtx = !rtx;
        shader = false;
    }
    if(n=="autodt"){
        AutoDTime=!AutoDTime;
        second=dt*fps;
    }
    if(isNumber(n)==false)return;
    
    for(int i=0;i<stoi(n);i++){
        string name;
        string x,y,z;
        double vx,vy,vz;
        double ax,ay,az;
        double mass;
        double radius;
        int c1,c2,c3,c4;
        bool stellar;
        ssin>>name>>x>>y>>z>>vx>>vy>>vz>>ax>>ay>>az>>mass>>radius>>c1>>c2>>c3>>c4>>stellar;
        if(name=="")return;
        if(x=="~")x=to_string(cam.x);
        if(y=="~")y=to_string(cam.y);
        if(z=="~")z=to_string(cam.z);
        p.push_back({name,stod(x),stod(y),stod(z),vx,vy,vz,ax,ay,az,mass,radius,c1,c2,c3,c4,stellar});
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

void drawCircle3(SDL_Renderer* ren,projection p,double fov_deg,bool Target){
    double fov_rad=fov_deg*(M_PI/180.0);
    double focal_length=height/(2.0*tan(fov_rad*0.5));
    int r=(int)(p.r*focal_length/p.sz);
    if(r<=1)r=1;
    if(r>2000)r=2000;
    for(int j=-r;j<=r;j++){
        int len=(int)sqrt(r*r-j*j);
        SDL_RenderLine(ren,p.sx-len,p.sy+j,p.sx+len,p.sy+j);
    }
    if(Target){
        int t_cir=r+20;
        int t_cir_l=r+19;
        for(int i=-t_cir;i<=t_cir;i++){
            for(int j=-t_cir;j<=t_cir;j++){
                if(i*i+j*j<t_cir*t_cir&&i*i+j*j>t_cir_l*t_cir_l){
                    SDL_SetRenderDrawColor(ren, 0, 255, 0, 255);
                    SDL_RenderPoint(ren,p.sx+i,p.sy+j);
                }
            }
        }
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
bool isNumber(string& str) {
    return !str.empty() && all_of(str.begin(), str.end(), [](unsigned char c) {
        return isdigit(c) || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+';
    });
}
void YourSpaceShip(camera& cam,SDL_Event& event,SDL_Window* win,SDL_Renderer* ren,bool& pause,int iniline){
    /*Initialize*/
    pause=true;
    const int command_count=7;
    string command[command_count+1]={"Name ","Mass(kg, without fuel) ","Engine Count ","Single Engine Thrust(N) ","Single Engine Consumption(kg/s) ","Fuel(kg) ","Velocity(m/s) ","Finish"};
    static int cur_command=1;
    static string commandbuf="";
    static string currentbuf=">";
    static bool first_input=true;
    if(first_input){
        first_input=!first_input;
        myship.vx=0; myship.vy=0; myship.vz=0;
        SDL_StartTextInput(win);
    }
    
    //Render
    SDL_SetRenderDrawColor(ren,255,255,255,255);
    int spaceline=200;
    command[cur_command-1]="->"+command[cur_command-1];
    float velocity=sqrt(myship.vx*myship.vx+myship.vy*myship.vy+myship.vz*myship.vz);
    SDL_RenderDebugText(ren, 10, spaceline+12*0, (command[0]+myship.name).c_str());
    SDL_RenderDebugText(ren, 10, spaceline+12*1, (command[1]+to_string(myship.mass)).c_str());
    SDL_RenderDebugText(ren, 10, spaceline+12*2, (command[2]+to_string(myship.engine_count)).c_str());
    SDL_RenderDebugText(ren, 10, spaceline+12*3, (command[3]+to_string(myship.single_engine_thrust)).c_str());
    SDL_RenderDebugText(ren, 10, spaceline+12*4, (command[4]+to_string(myship.single_engine_consumption)).c_str());
    SDL_RenderDebugText(ren, 10, spaceline+12*5, (command[5]+to_string(myship.fuel)).c_str());
    SDL_RenderDebugText(ren, 10, spaceline+12*6, (command[6]+to_string(velocity)).c_str());
    SDL_RenderDebugText(ren, 10, spaceline+12*7, command[7].c_str());
    
    if(event.type==SDL_EVENT_TEXT_INPUT){
        currentbuf+=event.text.text;
        commandbuf+=event.text.text;
    }else if(event.type==SDL_EVENT_KEY_DOWN){
        if(event.key.key==SDLK_BACKSPACE&&currentbuf.size()>1){
            currentbuf.pop_back();
            commandbuf.pop_back();
        }
        else if(event.key.key==SDLK_RETURN){
            if(cur_command==(command_count+1)){
                myship.initial=true;
                myship.mass+=myship.fuel;
                myship.x=cam.x; myship.y=cam.y; myship.z=cam.z;
                myship.yaw=(int)cam.yaw;
                myship.pitch=(int)cam.pitch;
                SDL_StopTextInput(win);
                pause=false;
                first_input=true;
                return;
            }
            switch(cur_command){
                case 1:{
                    myship.name=commandbuf;
                    break;
                }case 2:{
                    if(!isNumber(commandbuf))return;
                    myship.mass=stod(commandbuf);
                    break;
                }case 3:{
                    if(!isNumber(commandbuf))return;
                    myship.engine_count=stod(commandbuf);
                    break;
                }case 4:{
                    if(!isNumber(commandbuf))return;
                    myship.single_engine_thrust=stod(commandbuf);
                    break;
                }case 5:{
                    if(!isNumber(commandbuf))return;
                    myship.single_engine_consumption=stod(commandbuf);
                    break;
                }case 6:{
                    if(!isNumber(commandbuf))return;
                    myship.fuel=stod(commandbuf);
                    break;
                }case 7:{
                    if(!isNumber(commandbuf))return;
                    double v=stod(commandbuf);
                    double rady=cam.yaw*(M_PI/180.0);
                    double radp=cam.pitch*(M_PI/180.0);
                    myship.vx=sin(rady)*cos(radp)*v;
                    myship.vy=sin(radp)*v;
                    myship.vz=cos(rady)*cos(radp)*v;
                    break;
                }
            }
            commandbuf="";
            currentbuf=">";
            cur_command++;
        }
        else if(event.key.key==SDLK_UP&&cur_command>1)cur_command--;
        else if(event.key.key==SDLK_DOWN&&cur_command<command_count+1)cur_command++;
    }
    
    SDL_RenderDebugText(ren, 50, iniline, currentbuf.c_str());
    /*Initialize*/
}
float max_velocity=0;
float max_fuel=0;
float max_distance=0;
float max_throttle=1.0;
void InMyShip(camera& cam,SDL_Event& event,SDL_Window* win,SDL_Renderer* ren,int iniline,bool& targetisTyping,vector<particle> p,bool pause,bool debug,TTF_Font* font){
    cam.x=myship.x; cam.y=myship.y; cam.z=myship.z;
    /*Parameter*/
    static bool AutoOrbit=false;
    static bool DrawTrajectory=false;
    if(debug==true){
        const int para_count=9;
        int spaceline=200;
        string show[para_count]={"","Velocity(m/s):","Acceleration(m/s^2):","Throttle:","Mass/Fuel(kg):","Yaw/Pitch:","Target:","AutoOrbit:","Trajectory"};
        float velocity=sqrt(myship.vx*myship.vx+myship.vy*myship.vy+myship.vz*myship.vz);
        float acceleration=sqrt(myship.ax*myship.ax+myship.ay*myship.ay+myship.az*myship.az);
        float total_pushforce=myship.engine_count*myship.single_engine_thrust*myship.throttle;
        if(myship.fuel<=0)total_pushforce=0;
        SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
        SDL_RenderDebugText(ren,10,spaceline+12*0,myship.name.c_str());
        SDL_RenderDebugText(ren,10,spaceline+12*1,(show[1]+to_string(velocity)).c_str());
        SDL_RenderDebugText(ren,10,spaceline+12*2,(show[2]+to_string(acceleration)).c_str());
        SDL_RenderDebugText(ren,10,spaceline+12*3,(show[3]+to_string(myship.throttle)+'/'+to_string(total_pushforce)+'N').c_str());
        SDL_RenderDebugText(ren,10,spaceline+12*4,(show[4]+to_string(myship.mass)+'/'+to_string(myship.fuel)).c_str());
        SDL_RenderDebugText(ren,10,spaceline+12*6,(show[5]+to_string(myship.yaw)+'/'+to_string(myship.pitch)).c_str());
        SDL_RenderDebugText(ren,10,spaceline+12*7,(show[6]+myship.target_name+'/'+to_string(myship.distance)).c_str());
        SDL_RenderDebugText(ren,10,spaceline+12*8,(show[7]+to_string(AutoOrbit)).c_str());
        SDL_RenderDebugText(ren,10,spaceline+12*9,(show[8]+to_string(DrawTrajectory)).c_str());
    }
    //DashBoard
    float dash_r=(50.0f/900.0f)*height;
    float dash_minus_r=(20.0f/900.0f)*height;
    float dash_x=width*0.6;
    float dash_y=height*0.75;
    
    float v=(float)sqrt(myship.vx*myship.vx+myship.vy*myship.vy+myship.vz*myship.vz);
    max_velocity=(max_velocity>=v)?max_velocity:v;
    float fuel=(float)myship.fuel;
    max_fuel=(max_fuel>=fuel)?max_fuel:fuel;
    float distance=(float)myship.distance;
    max_distance=(max_distance>=distance)?max_distance:distance;
    float throttle=(float)myship.throttle;
    max_throttle=1.0;
    
    SDL_SetRenderDrawColor(ren, 90, 90, 90, 255);
    for(int y=-dash_r;y<=dash_r;y++){
        for(int x=-dash_r;x<=dash_r;x++){
            if(y*y+x*x<=dash_r*dash_r&&y*y+x*x>=dash_r*dash_r*0.95&&y<=(dash_r-dash_minus_r)){
                //Velocity
                SDL_RenderPoint(ren,dash_x+x+0*3*dash_r,dash_y+y+0);
                //Fuel
                SDL_RenderPoint(ren,dash_x+x+1*3*dash_r,dash_y+y+0);
                //Distance
                SDL_RenderPoint(ren,dash_x+x+2*3*dash_r,dash_y+y+0);
                //Throttle
                SDL_RenderPoint(ren,dash_x+x+3*3*dash_r,dash_y+y+0);
            }
        }
    }
    
    float x0=-sqrt(dash_r*dash_r-(dash_r-dash_minus_r)*(dash_r-dash_minus_r));
    float y0=-(dash_r-dash_minus_r);
    float a0=atan2(y0,x0);
    
    float v_theta=(v/max_velocity)*(2*M_PI-2*acos((dash_r-dash_minus_r)/dash_r));
    float fuel_theta=(fuel/max_fuel)*(2*M_PI-2*acos((dash_r-dash_minus_r)/dash_r));
    float distance_theta=(1-distance/max_distance)*(2*M_PI-2*acos((dash_r-dash_minus_r)/dash_r));
    float thr_theta=(sqrt(throttle*throttle)/max_throttle)*(2*M_PI-2*acos((dash_r-dash_minus_r)/dash_r));
    SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
    for(int y=-dash_r;y<=dash_r;y++){
        for(int x=-dash_r;x<=dash_r;x++){
            if(x*x+y*y<=dash_r*dash_r&&x*x+y*y>=dash_r*dash_r*0.85){
                //Velocity
                float at=atan2(y,x);
                float diff=a0-at;
                if(at>a0)diff+=2*M_PI;
                if(diff<=v_theta)SDL_RenderPoint(ren,dash_x+x+0*3*dash_r,dash_y-y+0);
                if(diff<=fuel_theta)SDL_RenderPoint(ren,dash_x+x+1*3*dash_r,dash_y-y+0);
                if(diff<=distance_theta)SDL_RenderPoint(ren,dash_x+x+2*3*dash_r,dash_y-y+0);
                if(diff<=thr_theta)SDL_RenderPoint(ren,dash_x+x+3*3*dash_r,dash_y-y+0);
            }
        }
    }
    //TTF Initial
    SDL_Color white={255,255,255,255};
    
    char speed_buf[16],speed_label[]="SPEED";
    char fuel_buf[16],fuel_label[]="FUEL";
    char dis_buf[8],dis_label[]="DISTANCE";
    char thr_buf[8],thr_label[]="THROTTLE";
    SDL_snprintf(speed_buf, sizeof(speed_buf), "%d", (int)v);
    SDL_snprintf(fuel_buf, sizeof(fuel_buf), "%d", (int)fuel);
    SDL_snprintf(dis_buf, sizeof(dis_buf), "%d", (int)((1-distance/max_distance)*100));
    SDL_snprintf(thr_buf, sizeof(thr_buf), "%d", (int)((throttle/max_throttle)*100));
    
    DrawTTF(ren,font,speed_buf,dash_x,dash_y,white,dash_r*1.7,1,1);
    DrawTTF(ren,font,speed_label,dash_x,dash_y+dash_r*0.8,white,100,0.6,0.46);
    DrawTTF(ren,font,fuel_buf,dash_x+1*3*dash_r,dash_y,white,dash_r*1.5,1,1);
    DrawTTF(ren,font,fuel_label,dash_x+1*3*dash_r,dash_y+dash_r*0.8,white,100,0.6,0.46);
    DrawTTF(ren,font,dis_buf,dash_x+2*3*dash_r,dash_y,white,dash_r*1.8,1,1);
    DrawTTF(ren,font,dis_label,dash_x+2*3*dash_r,dash_y+dash_r*0.8,white,100,0.6,0.46);
    DrawTTF(ren,font,thr_buf,dash_x+3*3*dash_r,dash_y,white,dash_r*1.8,1,1);
    DrawTTF(ren,font,thr_label,dash_x+3*3*dash_r,dash_y+dash_r*0.8,white,100,0.6,0.46);
    
    /*Parameter*/
    /*Header*/
    particle heading;
    double rady=myship.yaw*(M_PI/180.0);
    double radp=myship.pitch*(M_PI/180.0);
    heading.x=myship.x+sin(rady)*cos(radp)*1000;
    heading.y=myship.y+sin(radp)*1000;
    heading.z=myship.z+cos(rady)*cos(radp)*1000;
    Screen(heading,cam);
    if(res.sz>0){
        SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
        SDL_RenderLine(ren,res.sx-5,res.sy,res.sx+5,res.sy);
        SDL_RenderLine(ren,res.sx,res.sy-5,res.sx,res.sy+5);
    }
    /*Header*/
    /*Controller*/
    const bool* keystate = SDL_GetKeyboardState(NULL);
    //Straight
    if(targetisTyping==false&&pause==false){
        static bool TOGA=false;
        
        if(TOGA==true&&!keystate[SDL_SCANCODE_I]&&!keystate[SDL_SCANCODE_K])TOGA=false;
        if(keystate[SDL_SCANCODE_I]&&myship.throttle<0.99&&TOGA==false)myship.throttle+=0.01;
        if(keystate[SDL_SCANCODE_K]&&myship.throttle>-0.99&&TOGA==false)myship.throttle-=0.01;
        if(abs(myship.throttle)<0.001&&TOGA==false&&(keystate[SDL_SCANCODE_I]||keystate[SDL_SCANCODE_K])){
            myship.throttle=0.0;
            TOGA=true;
        }
        double used=sqrt(myship.engine_count*myship.single_engine_consumption*myship.throttle*dt*myship.engine_count*myship.single_engine_consumption*myship.throttle*dt);
        if((myship.fuel-used)>0.0){
            double F=myship.engine_count*myship.single_engine_thrust*myship.throttle;
            double a=F/myship.mass;
            myship.ax+=sin(rady)*cos(radp)*a;
            myship.ay+=sin(radp)*a;
            myship.az+=cos(rady)*cos(radp)*a;
            myship.fuel-=used;
            myship.mass-=used;
        }
        //Heading
        double turning_speed=1.5;
        if(keystate[SDL_SCANCODE_S]&&myship.pitch>-89.0)myship.pitch-=turning_speed;
        if(keystate[SDL_SCANCODE_W]&&myship.pitch<89.0)myship.pitch+=turning_speed;
        if(keystate[SDL_SCANCODE_A])myship.yaw-=turning_speed;
        if(keystate[SDL_SCANCODE_D])myship.yaw+=turning_speed;
        if(myship.yaw>=360)myship.yaw-=360;
        if(myship.yaw<0)myship.yaw+=360;
    }
    /*Controller*/
    /*Function*/
    static bool first_lockedZ=false;
    static bool first_lockedE=false;
    static bool first_lockedC=false;
    static bool locked=false;
    static string commandbuf="";
    static string currentbuf="Target:";
    if(keystate[SDL_SCANCODE_Q]&&targetisTyping==false){
        targetisTyping=true;
        SDL_StartTextInput(win);
    }
    if(first_lockedE==true&&!keystate[SDL_SCANCODE_E])first_lockedE=false;
    if(keystate[SDL_SCANCODE_E]&&first_lockedE==false){
        if(myship.target==true){
            locked=!locked;
            first_lockedE=true;
        }else{
            locked=false;
        }
    }
    if(first_lockedC==true&&!keystate[SDL_SCANCODE_C])first_lockedC=false;
    if(keystate[SDL_SCANCODE_C]&&first_lockedC==false){
        if(myship.target==true){
            AutoOrbit=!AutoOrbit;
            first_lockedC=true;
        }else{
            AutoOrbit=false;
        }
    }
    if(first_lockedZ==true&&!keystate[SDL_SCANCODE_Z])first_lockedZ=false;
    if(keystate[SDL_SCANCODE_Z]&&first_lockedZ==false){
        DrawTrajectory=!DrawTrajectory;
        first_lockedZ=true;
    }

    if(DrawTrajectory){
        static double proj=100.0;
        if(event.type==SDL_EVENT_KEY_DOWN&&event.key.key==SDLK_U&&proj>1.0)proj/=10.0;
        if(event.type==SDL_EVENT_KEY_DOWN&&event.key.key==SDLK_O&&proj<1000000.0)proj*=10.0;
        DrawTrajectorySystem(ren,proj,p,cam);
    }
    if(AutoOrbit){
        AutoOrbitSystem(first_lockedC,p);
    }
    if(locked==true){
        for(int i=0;i<p.size();i++){
            if(p[i].name==myship.target_name){
                double dx=p[i].x-myship.x;
                double dy=p[i].y-myship.y;
                double dz=p[i].z-myship.z;
                if(myship.distance<0.1)break;
                double targetYaw=atan2(dx,dz)*(180.0/M_PI);
                double targetPitch=asin(dy/myship.distance)*(180.0/M_PI);
                myship.yaw=targetYaw;
                myship.pitch=targetPitch;
                break;
            }
        }
    }
    if(targetisTyping==true){
        if(event.type==SDL_EVENT_TEXT_INPUT){
            commandbuf+=event.text.text;
            currentbuf+=event.text.text;
        }if(event.type==SDL_EVENT_KEY_DOWN){
            if(event.key.key==SDLK_BACKSPACE&&currentbuf.size()>7){
                commandbuf.pop_back();
                currentbuf.pop_back();
            }
            if(event.key.key==SDLK_RETURN){
                for(int i=0;i<p.size();i++){
                    if(p[i].name==commandbuf){
                        myship.target=true;
                        myship.target_name=commandbuf;
                        commandbuf="";
                        currentbuf="Target:";
                        break;
                    }
                    else{
                        myship.target=false;
                        myship.target_name="";
                    }
                }
                targetisTyping=false;
                SDL_StopTextInput(win);
            }
            if(event.key.key==SDLK_ESCAPE){
                targetisTyping=false;
                SDL_StopTextInput(win);
            }
        }
        SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
        SDL_RenderDebugText(ren,30,height*0.7,currentbuf.c_str());
    }
    /*Function*/
}
void AutoOrbitSystem(bool first_lockedC,vector<particle> p){
    static double shortest=0;
    double M=0;
    double pvx; double pvy; double pvz;
    if(first_lockedC||shortest>myship.distance)shortest=myship.distance;
    if(shortest<=myship.distance){
        int cur=0;
        for(int i=0;i<p.size();i++){
            if(myship.target_name==p[i].name){
                cur=i;
                break;
            }
        }
        M=p[cur].mass;
        pvx=p[cur].vx; pvy=p[cur].vy; pvz=p[cur].vz;
        double OrbitV=sqrt(G*M/myship.distance);
        //Distance-Velocity Heading

        double dx=p[cur].x-myship.x;
        double dy=p[cur].y-myship.y;
        double dz=p[cur].z-myship.z;
        double dist=dx*dx+dy*dy+dz*dz;
        //Relatively Velocity
        double rvx=myship.vx-pvx;
        double rvy=myship.vy-pvy;
        double rvz=myship.vz-pvz;
        double k=(rvx*dx+rvy*dy+rvz*dz)/(dist+ess);
        
        //New Vector
        double a=rvx-k*dx;
        double b=rvy-k*dy;
        double c=rvz-k*dz;
        double tmag=sqrt(a*a+b*b+c*c+ess);
        
        //Each Heading Velocity
        double tagvx=(a/tmag)*OrbitV+pvx;
        double tagvy=(b/tmag)*OrbitV+pvy;
        double tagvz=(c/tmag)*OrbitV+pvz;
        
        double dvx=tagvx-myship.vx;
        double dvy=tagvy-myship.vy;
        double dvz=tagvz-myship.vz;
        double dv=sqrt(dvx*dvx+dvy*dvy+dvz*dvz);
        //Checking
        if(dv>1.0){
            double tag_yaw=atan2(dvx,dvz)*(180.0/M_PI);
            if(tag_yaw<0)tag_yaw+=360;
            double tag_pitch=asin(dvy/(dv+ess))*(180.0/M_PI);
            
            double turning_speed=1.5;
            
            double y_diff=tag_yaw-myship.yaw;
            if(y_diff>180)y_diff-=360;
            if(y_diff<-180)y_diff+=360;
            if(abs(y_diff)>1.0){
                myship.yaw+=(y_diff>0)?turning_speed:-turning_speed;
            }
            double p_diff=tag_pitch-myship.pitch;
            if(abs(p_diff)>1.0){
                myship.pitch+=(p_diff>0)?turning_speed:-turning_speed;
            }
            double sensi=100.0;
            if(abs(y_diff)<10.0&&abs(p_diff)<10.0){
                myship.throttle=(dv>sensi)?1.0:(dv/sensi);
            }else{
                myship.throttle=0.0;
            }
        }else{
            myship.throttle=0.0;
        }
    }
}
void DrawTrajectorySystem(SDL_Renderer* ren,double proj,vector<particle> p, camera& cam){
    const int steps=100;
    double v=sqrt(myship.vx*myship.vx+myship.vy*myship.vy+myship.vz*myship.vz);
    
    double dt_pre=proj/v;
    
    double cvx=myship.vx, cvy=myship.vy, cvz=myship.vz;
    double cx=myship.x, cy=myship.y, cz=myship.z;
    SDL_SetRenderDrawColor(ren, 0, 255, 0, 255);
    for(int i=0;i<steps;i++){
        double ax=0, ay=0, az=0;
        for(int j=0;j<p.size();j++){
            double dx=p[j].x-cx;
            double dy=p[j].y-cy-100;
            double dz=p[j].z-cz;
            double r=sqrt(dx*dx+dy*dy+dz*dz);
            double F=(G*p[j].mass)/(r*r);
            
            ax+=F*dx/r;
            ay+=F*dy/r;
            az+=F*dz/r;
        }
        cvx+=ax*dt_pre; cvy+=ay*dt_pre; cvz+=az*dt_pre;
        cx+=cvx*dt_pre; cy+=cvy*dt_pre; cz+=cvz*dt_pre;
        
        particle temp={"",cx,cy-500,cz,0,0,0,0,0,0,0,0.01,0,255,0,255,false};
        Screen(temp,cam);
        projection pro={"",0,0,0,res.sx,res.sy,res.sz,0,255,0,150,0.01};
        if(res.sz>0.1)drawCircle3(ren,pro,cam.fov,false);
    }
}
void DrawCircleShader(SDL_Renderer* ren,projection p,double fov_deg,bool Target,double lx,double ly,double lz,double stellar_intensity){
    static double rotation_time = 0;
    rotation_time += dt * 0.0005;
    
    double fov_rad=fov_deg*(M_PI/180.0);
    double focal_length=height/(2.0*tan(fov_rad*0.5));
    int r=(int)(p.r*focal_length/(p.sz+ess));
    if(r<=1)r=1;
    if(r>2000)r=2000;
    int start_i=(p.sx-r>=0)?-r:-p.sx;
    int end_i=(p.sx+r<=width)?r:width-p.sx;
    int start_j=(p.sy-r>=0)?-r:-p.sy;
    int end_j=(p.sy+r<=height)?r:height-p.sy;
    
    for(int i=start_i;i<=end_i;i++){
        for(int j=start_j;j<=end_j;j++){
            if(i*i+j*j<=r*r){
                double nx=(double)i/r;
                double ny=(double)j/r;
                double nz=sqrt(1.0-(nx*nx+ny*ny));

                double intensity=nx*lx+ny*ly+nz*lz;
                intensity*=stellar_intensity;
                if(intensity<0)intensity=abs(intensity*0.03);
                
                SDL_SetRenderDrawColor(ren, intensity*p.c1, intensity*p.c2, intensity*p.c3, p.c4);
                SDL_RenderPoint(ren,p.sx+i,p.sy+j);
            }
        }
    }
    
    if(Target){
        int t_cir=r+20;
        int t_cir_l=r+19;
        for(int i=-t_cir;i<=t_cir;i++){
            for(int j=-t_cir;j<=t_cir;j++){
                if(i*i+j*j<t_cir*t_cir&&i*i+j*j>t_cir_l*t_cir_l){
                    SDL_SetRenderDrawColor(ren, 0, 255, 0, 255);
                    SDL_RenderPoint(ren,p.sx+i,p.sy+j);
                }
            }
        }
    }
}
void DrawTTF(SDL_Renderer* ren, TTF_Font* font, const char* text, float centerX, float centerY, SDL_Color col,float maxWidth,float x_scale,float y_scale){
    if(!font||!text||strlen(text)==0)return;
    //Turning to image
    SDL_Surface* surf=TTF_RenderText_Blended(font, text, 0, col);
    if(!surf)return;
    //Send to GPU
    SDL_Texture* tex=SDL_CreateTextureFromSurface(ren, surf);
    if(!tex){
        SDL_Log("SDL_ttf Error: %s", SDL_GetError());
        SDL_DestroySurface(surf);
        return;
    }
    //Calculate scale
    float inscale=1.0f;
    if(surf->w>maxWidth){
        inscale=maxWidth/surf->w;
    }
    //Position calculate
    SDL_FRect dstRect;
    dstRect.w=(float)surf->w*inscale*x_scale;
    dstRect.h=(float)surf->h*inscale*y_scale;
    dstRect.x=centerX-(dstRect.w/2.0f);
    dstRect.y=centerY-(dstRect.h/2.0f);
    //Draw
    SDL_RenderTexture(ren,tex,NULL,&dstRect);
    
    SDL_DestroySurface(surf);
    SDL_DestroyTexture(tex);
}
void DrawStellarShader(SDL_Renderer* ren, projection p, double fov_deg) {

    double fov_rad = fov_deg * (M_PI / 180.0);
    double focal_length = height / (2.0 * tan(fov_rad * 0.5));

    double r = p.r * focal_length / (p.sz + 1e-6);
    if (r <= 0 || p.sz <= 0) return;

    int glow_limit = r * 8;

    int start_i = max(-glow_limit, (int)-p.sx);
    int end_i   = min(glow_limit,  (int)(width - p.sx));
    int start_j = max(-glow_limit, (int)-p.sy);
    int end_j   = min(glow_limit,  (int)(height - p.sy));

    double inv_r2 = 1.0 / (r * r);
    double inv_glow = 1.0 / glow_limit;

    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_ADD);

    for (int j = start_j; j <= end_j; j++) {
        int jj = j * j;
        for (int i = start_i; i <= end_i; i++) {
            int dist_sq = i * i + jj;

            if (dist_sq > glow_limit * glow_limit) continue;
            double norm_dist_sq = dist_sq * inv_r2;
            double intensity;

            if (norm_dist_sq < 1.0) {

                intensity = 1.5 - 0.5 * norm_dist_sq;
            } else {

                intensity = 1.0 / (norm_dist_sq * 0.8 + 0.2);


                double fade = 1.0 - (dist_sq * inv_glow * inv_glow);
                if (fade < 0) continue;

                intensity *= fade;
            }

            if (intensity <= 0.01) continue;

            double intensity_clamped = min(intensity, 2.0);

            int final_r = (int)(p.c1 * intensity_clamped);
            int final_g = (int)(p.c2 * intensity_clamped);
            int final_b = (int)(p.c3 * intensity_clamped);

            if (final_r > 255) final_r = 255;
            if (final_g > 255) final_g = 255;
            if (final_b > 255) final_b = 255;

            SDL_SetRenderDrawColor(ren, final_r, final_g, final_b, 255);
            SDL_RenderPoint(ren, p.sx + i, p.sy + j);
        }
    }

    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_NONE);
}
void RTXInitial(){
    RTXOriginVector.assign(height, vector<RTXV>(width, {0,0,0}));
    RTXVector.assign(height, vector<RTXV>(width, {0,0,0}));
    RTXVectorLength.assign(height, vector<double>(width, 0));
    double OOm = height/(2*tan(cam.fov * M_PI / 360.0));
    for(int i = height/2;i > -height/2;i--){
        for(int j = -width/2;j < width/2;j++){
            RTXOriginVector[height/2-i][j+width/2].x = j;
            RTXOriginVector[height/2-i][j+width/2].y = i;
            RTXOriginVector[height/2-i][j+width/2].z = OOm;
            RTXVectorLength[height/2-i][j+width/2] = sqrt(j * j + i * i + OOm * OOm);
        }
    }
}

RTXV RTXVectorCalcM(RTXV v, RTXV O, RTXV P, double radius, double VLength){
    //                double dx = planet[p].x - cam.x;
    //                double dy = planet[p].y - cam.y;
    //                double dz = planet[p].z - cam.z;
    //
    //                double vx = RTXVector[i][j].x;
    //                double vy = RTXVector[i][j].y;
    //                double vz = RTXVector[i][j].z;
    //
    //                double len = RTXVectorLength[i][j];
    //
    //                double OP_v = dx * vx + dy * vy + dz * vz;
    //                double OF = OP_v / len;
    //
    //                double cx = dy * vz - dz * vy;
    //                double cy = dz * vx - dx * vz;
    //                double cz = dx * vy - dy * vx;
    //
    //                double FP2 = (cx*cx + cy*cy + cz*cz) / (len*len);
    //
    //                double r2 = planet[p].r * planet[p].r;
    //
    //                if(FP2 > r2){
    //                    continue;
    //                }
    //
    //                double MF = sqrt(max(0.0, r2 - FP2));
    //
    //                double OM = OF - MF;
    //
    //                if(OM < 0){
    //                    OM = OF + MF;
    //                }
    //
    //                if(OM < 0){
    //                    continue;
    //                }
    //
    //                if(OM < nearestOM){
    //
    //                    nearestOM = OM;
    //                    nearestPlanet = p;
    //
    //                    M.x = cam.x + vx * OM / len;
    //                    M.y = cam.y + vy * OM / len;
    //                    M.z = cam.z + vz * OM / len;
    //                }
    return RTXV{0,0,0};
}
RTXV RTXAngularCalcM(RTXV v, RTXV O, RTXV P, double radius, double VLength){
    RTXV M;
    double OP_v = (P.x - O.x) * v.x + (P.y - O.y) * v.y + (P.z - O.z) * v.z;
    double OP = sqrt(
                     pow(P.x - O.x,2) +
                     pow(P.y - O.y,2) +
                     pow(P.z - O.z,2));
    double real_angle = acos(clamp(OP_v / (OP * VLength),-1.0,1.0));
    double tan_angle = asin( radius / OP );
    if(real_angle > tan_angle){
        return {0,0,0,false};
    }
    double OM = OP * cos(real_angle) - sqrt(pow(radius, 2) - pow(OP * sin(real_angle), 2));
    M.x = v.x * OM / VLength + O.x;
    M.y = v.y * OM / VLength + O.y;
    M.z = v.z * OM / VLength + O.z;
    M.OM = OM;
    return M;
}
void RTXRender(SDL_Renderer* ren, const vector<projection> &planet, const vector<particle> &parti, const vector<int> &suns){
    /* Correct Ray Vector */
    RTXV Matrix[3] = {
        {cos(cam.yaw * M_PI/ 180)                              , -sin(cam.pitch * M_PI/ 180) * sin(cam.yaw * M_PI/ 180), cos(cam.pitch * M_PI/ 180) * sin(cam.yaw * M_PI/ 180)},
        {0                                                     , cos(cam.pitch * M_PI/ 180)                            , sin(cam.pitch * M_PI/ 180)                           },
        {-sin(cam.yaw * M_PI/ 180)                             , -sin(cam.pitch * M_PI/ 180) * cos(cam.yaw * M_PI/ 180), cos(cam.pitch * M_PI/ 180) * cos(cam.yaw * M_PI/ 180)}
    };
    #pragma omp parallel for
    for(int i = height - 1;i >= 0;i--){
        for(int j = 0;j < width;j++){
            RTXVector[i][j].x = RTXOriginVector[i][j].x * Matrix[0].x + RTXOriginVector[i][j].y * Matrix[0].y + RTXOriginVector[i][j].z * Matrix[0].z;
            RTXVector[i][j].y = RTXOriginVector[i][j].x * Matrix[1].x + RTXOriginVector[i][j].y * Matrix[1].y + RTXOriginVector[i][j].z * Matrix[1].z;
            RTXVector[i][j].z = RTXOriginVector[i][j].x * Matrix[2].x + RTXOriginVector[i][j].y * Matrix[2].y + RTXOriginVector[i][j].z * Matrix[2].z;
        }
    }
    
    /* Ray out */
    vector<vector<Uint8>> Rbuf(height, vector<Uint8>(width, 0));
    vector<vector<Uint8>> Gbuf(height, vector<Uint8>(width, 0));
    vector<vector<Uint8>> Bbuf(height, vector<Uint8>(width, 0));
    #pragma omp parallel for
    for(int i = height - 1;i >= 0;i--){
        for(int j = 0;j < width;j++){
            double nearestOM = INFINITY;
            int nearestPlanet = -1;
            RTXV M;
            for(int p = 0;p < planet.size();p++){
                /* Angle detect whether intersect*/
                /* Angular Method */
                RTXV Mcandi = RTXAngularCalcM(RTXVector[i][j], {cam.x, cam.y, cam.z}, {planet[p].x, planet[p].y, planet[p].z}, planet[p].r, RTXVectorLength[i][j]);
                if(Mcandi.intersect == false){
                    continue;
                }else{
                    if(Mcandi.OM > 0 && Mcandi.OM < nearestOM){
                        nearestOM = Mcandi.OM;
                        nearestPlanet = p;
                        M = Mcandi;
                    }
                }
                /* Vector Method */
            }
            if(nearestPlanet == -1){
                continue;
            }
            /* Light Magnitude */
            RTXV Npm = {(M.x - planet[nearestPlanet].x)/planet[nearestPlanet].r, (M.y - planet[nearestPlanet].y)/planet[nearestPlanet].r, (M.z - planet[nearestPlanet].z)/planet[nearestPlanet].r};
            double PS = sqrt(
                             pow(parti[suns[0]].x - planet[nearestPlanet].x, 2) +
                             pow(parti[suns[0]].y - planet[nearestPlanet].y, 2) +
                             pow(parti[suns[0]].z - planet[nearestPlanet].z, 2)
                             );
            RTXV Nps = {(parti[suns[0]].x - planet[nearestPlanet].x)/PS, (parti[suns[0]].y - planet[nearestPlanet].y)/PS, (parti[suns[0]].z - planet[nearestPlanet].z)/PS};
            double I = max(Npm.x * Nps.x + Npm.y * Nps.y + Npm.z * Nps.z, 0.0);
            
            /* Shadow Calc */
            double MSLength = sqrt(
                             pow((parti[suns[0]].x - M.x), 2) +
                             pow((parti[suns[0]].y - M.y), 2) +
                             pow((parti[suns[0]].z - M.z), 2)
                             );
            
            RTXV MS = {parti[suns[0]].x - M.x, parti[suns[0]].y - M.y, parti[suns[0]].z - M.z};
            for(int p = 0;p < planet.size();p++){
                if(p == nearestPlanet)continue;
                RTXV Shadow = RTXAngularCalcM(MS, M, {planet[p].x, planet[p].y, planet[p].z}, planet[p].r, MSLength);
                if(Shadow.intersect && Shadow.OM < MSLength && Shadow.OM > 0){
                    I = 0.1 * I;
                    break;
                }
            }
            
            /* RGB Calc */
            Uint8 R = planet[nearestPlanet].c1 * I;
            Uint8 G = planet[nearestPlanet].c2 * I;
            Uint8 B = planet[nearestPlanet].c3 * I;
            Rbuf[i][j] = R;
            Gbuf[i][j] = G;
            Bbuf[i][j] = B;
        }
    }
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            /* Render */
            SDL_SetRenderDrawColor(ren, Rbuf[i][j], Gbuf[i][j], Bbuf[i][j], 255);
            SDL_RenderPoint(ren, j, i);
        }
    }
}
void RTXRenderMetal(SDL_Renderer *ren,
                    const std::vector<projection> &planet,
                    const std::vector<particle> &parti,
                    const std::vector<int> &suns) {
    (void)planet; // Full parti scene is required for consistent camera/shadow rays.

    struct alignas(16) RTXMetalFloat4 {
        float x, y, z, w;
    };
    struct alignas(16) RTXMetalBodyCPU {
        RTXMetalFloat4 centerRadius;
        RTXMetalFloat4 colorEmissive;
    };
    struct alignas(16) RTXMetalNodeCPU {
        RTXMetalFloat4 bmin, bmax;
        int32_t left, right, bodyIndex, padding;
    };
    struct alignas(16) RTXMetalParamsCPU {
        RTXMetalFloat4 right, up, forward, sunPosition;
        float focal;
        uint32_t width, height, bodyCount;
        int32_t sunIndex;
        float ambient;
        uint32_t enableShadows;
        uint32_t padding;
    };
    
    // Same arguments as RTXRender(); 'planet' is intentionally not used.
    // BVH and Shadow Rays need the complete physical scene, not just visible projections.

    if (!ren || width <= 0 || height <= 0) return;

    static id<MTLDevice> gpu = nil;
    static id<MTLCommandQueue> gpuQueue = nil;
    static id<MTLComputePipelineState> gpuPipeline = nil;
    static id<MTLBuffer> bodyBuffer = nil;
    static id<MTLBuffer> nodeBuffer = nil;
    static id<MTLBuffer> pixelBuffer = nil;
    static NSUInteger maxBodies = 0, maxNodes = 0, maxPixels = 0;
    static SDL_Texture *texture = nullptr;
    static int textureW = 0, textureH = 0;
    static bool initAttempted = false;
    static SDL_Renderer *lastRenderer = nullptr;
    static_assert(sizeof(RTXMetalBodyCPU) == 32, "RTXMetalBody CPU/GPU size mismatch");
    static_assert(sizeof(RTXMetalNodeCPU) == 48, "RTXMetalNode CPU/GPU size mismatch");
    static_assert(sizeof(RTXMetalParamsCPU) == 96, "RTXMetalParams CPU/GPU size mismatch");

    if (!initAttempted) {
        initAttempted = true;
        gpu = MTLCreateSystemDefaultDevice();
        if (!gpu) {
            std::cerr << "RTXMetal: Metal device unavailable\n";
            return;
        }
        gpuQueue = [gpu newCommandQueue];
        id<MTLLibrary> library = [gpu newDefaultLibrary];
        if (!library) {
            std::cerr << "RTXMetal: No default Metal library. Add RTXMetalKernel_v2.metal to the Xcode target.\n";
            return;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"RTXMetalKernel"];
        if (!function) {
            std::cerr << "RTXMetal: Kernel RTXMetalKernel not found in default library.\n";
            return;
        }
        NSError *error = nil;
        gpuPipeline = [gpu newComputePipelineStateWithFunction:function error:&error];
        if (!gpuPipeline) {
            std::cerr << "RTXMetal: failed to create pipeline: "
                      << [[error localizedDescription] UTF8String] << "\n";
            return;
        }
    }
    if (!gpuPipeline || !gpuQueue) return;

    if (!texture || lastRenderer != ren || textureW != width || textureH != height) {
        if (texture) SDL_DestroyTexture(texture);
        texture = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGBA32,
                                    SDL_TEXTUREACCESS_STREAMING, width, height);
        if (!texture) {
            std::cerr << "RTXMetal: SDL_CreateTexture: " << SDL_GetError() << "\n";
            textureW = textureH = 0;
            return;
        }
        textureW = width;
        textureH = height;
        lastRenderer = ren;
    }

    const double yaw = cam.yaw * (M_PI / 180.0);
    const double pitch = cam.pitch * (M_PI / 180.0);
    const double cy = cos(yaw), sy = sin(yaw);
    const double cp = cos(pitch), sp = sin(pitch);
    // Columns of the same Ry(yaw) * Rx(-pitch) matrix as the CPU RTX renderer.
    double rx = cy, ry = 0.0, rz = -sy;
    double ux = -sp * sy, uy = cp, uz = -sp * cy;
    double fx = cp * sy, fy = sp, fz = cp * cy;

    std::vector<RTXMetalBodyCPU> bodies;
    bodies.reserve(parti.size());
    std::vector<int> originalToGPU(parti.size(), -1);
    for (size_t i = 0; i < parti.size(); ++i) {
        const particle &p = parti[i];
        if (!std::isfinite(p.radius) || p.radius <= 0) continue;
        double dx = p.x - cam.x;
        double dy = p.y - cam.y;
        double dz = p.z - cam.z;
        if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)) continue;
        // Discard objects that cannot be represented safely by GPU float coordinates.
        if (!std::isfinite(float(dx)) || !std::isfinite(float(dy)) ||
            !std::isfinite(float(dz)) || !std::isfinite(float(p.radius)) ||
            float(p.radius) <= 0.0f) continue;

        RTXMetalBodyCPU body = {};
        body.centerRadius = {float(dx), float(dy), float(dz), float(p.radius)};
        body.colorEmissive = {float(std::clamp(p.c1, 0, 255)) / 255.0f,
                              float(std::clamp(p.c2, 0, 255)) / 255.0f,
                              float(std::clamp(p.c3, 0, 255)) / 255.0f,
                              p.stellar ? 1.0f : 0.0f};
        originalToGPU[i] = int(bodies.size());
        bodies.push_back(body);
    }
    if (bodies.empty()) return;

    // Pick one stellar body, as your current CPU implementation does.
    int sunIndex = -1;
    if (!suns.empty() && suns[0] >= 0 && size_t(suns[0]) < originalToGPU.size())
        sunIndex = originalToGPU[size_t(suns[0])];

    // Build a balanced BVH each frame; works for all spheres, no name-based special cases.
    std::vector<int> order(bodies.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = int(i);
    std::vector<RTXMetalNodeCPU> nodes;
    nodes.reserve(2 * bodies.size());

    std::function<int(int,int)> buildBVH = [&](int first, int last) -> int {
        int nodeId = int(nodes.size());
        nodes.emplace_back();
        RTXMetalNodeCPU node = {};
        node.left = -1;
        node.right = -1;
        node.bodyIndex = -1;

        float lo[3] = {INFINITY, INFINITY, INFINITY};
        float hi[3] = {-INFINITY, -INFINITY, -INFINITY};
        float clo[3] = {INFINITY, INFINITY, INFINITY};
        float chi[3] = {-INFINITY, -INFINITY, -INFINITY};
        for (int k = first; k < last; ++k) {
            const auto &s = bodies[size_t(order[k])].centerRadius;
            float c[3] = {s.x, s.y, s.z};
            for (int a = 0; a < 3; ++a) {
                lo[a] = std::min(lo[a], c[a] - s.w);
                hi[a] = std::max(hi[a], c[a] + s.w);
                clo[a] = std::min(clo[a], c[a]);
                chi[a] = std::max(chi[a], c[a]);
            }
        }
        // One floating point step outward ensures AABB encloses the quantized spheres.
        node.bmin = {std::nextafter(lo[0], -std::numeric_limits<float>::infinity()),
                     std::nextafter(lo[1], -std::numeric_limits<float>::infinity()),
                     std::nextafter(lo[2], -std::numeric_limits<float>::infinity()), 0};
        node.bmax = {std::nextafter(hi[0], std::numeric_limits<float>::infinity()),
                     std::nextafter(hi[1], std::numeric_limits<float>::infinity()),
                     std::nextafter(hi[2], std::numeric_limits<float>::infinity()), 0};

        if (last - first == 1) {
            node.bodyIndex = order[first];
        } else {
            int axis = 0;
            if (chi[1] - clo[1] > chi[axis] - clo[axis]) axis = 1;
            if (chi[2] - clo[2] > chi[axis] - clo[axis]) axis = 2;
            int mid = first + (last - first) / 2;
            std::nth_element(order.begin() + first, order.begin() + mid, order.begin() + last,
                [&](int a, int b) {
                    const RTXMetalFloat4 &aa = bodies[size_t(a)].centerRadius;
                    const RTXMetalFloat4 &bb = bodies[size_t(b)].centerRadius;
                    float va = axis == 0 ? aa.x : axis == 1 ? aa.y : aa.z;
                    float vb = axis == 0 ? bb.x : axis == 1 ? bb.y : bb.z;
                    return va < vb;
                });
            node.left = buildBVH(first, mid);
            node.right = buildBVH(mid, last);
        }
        nodes[size_t(nodeId)] = node;
        return nodeId;
    };
    buildBVH(0, int(order.size()));

    NSUInteger nBodiesBytes = bodies.size() * sizeof(RTXMetalBodyCPU);
    NSUInteger nNodesBytes = nodes.size() * sizeof(RTXMetalNodeCPU);
    NSUInteger nPixelBytes = size_t(width) * size_t(height) * sizeof(uint32_t);
    if (nBodiesBytes > maxBodies) {
        bodyBuffer = [gpu newBufferWithLength:nBodiesBytes options:MTLResourceStorageModeShared];
        maxBodies = nBodiesBytes;
    }
    if (nNodesBytes > maxNodes) {
        nodeBuffer = [gpu newBufferWithLength:nNodesBytes options:MTLResourceStorageModeShared];
        maxNodes = nNodesBytes;
    }
    if (nPixelBytes > maxPixels) {
        pixelBuffer = [gpu newBufferWithLength:nPixelBytes options:MTLResourceStorageModeShared];
        maxPixels = nPixelBytes;
    }
    if (!bodyBuffer || !nodeBuffer || !pixelBuffer) {
        std::cerr << "RTXMetal: GPU buffer allocation failed\n";
        return;
    }
    std::memcpy([bodyBuffer contents], bodies.data(), nBodiesBytes);
    std::memcpy([nodeBuffer contents], nodes.data(), nNodesBytes);

    RTXMetalParamsCPU params = {};
    params.right = {float(rx),float(ry),float(rz),0};
    params.up = {float(ux),float(uy),float(uz),0};
    params.forward = {float(fx),float(fy),float(fz),0};
    params.focal = float(height / (2.0 * tan(cam.fov * M_PI / 360.0)));
    params.width = uint32_t(width);
    params.height = uint32_t(height);
    params.bodyCount = uint32_t(bodies.size());
    params.sunIndex = sunIndex;
    params.ambient = 0.03f;  // Set to 0.0f for strictly zero light in shadow.
    params.enableShadows = 1;
    if (sunIndex >= 0) params.sunPosition = bodies[size_t(sunIndex)].centerRadius;

    id<MTLCommandBuffer> command = [gpuQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:gpuPipeline];
    [encoder setBuffer:bodyBuffer offset:0 atIndex:0];
    [encoder setBuffer:nodeBuffer offset:0 atIndex:1];
    [encoder setBuffer:pixelBuffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    NSUInteger threads = std::min(NSUInteger(128), gpuPipeline.maxTotalThreadsPerThreadgroup);
    MTLSize threadsPerGroup = MTLSizeMake(threads, 1, 1);
    MTLSize gridSize = MTLSizeMake(NSUInteger(width), NSUInteger(height), 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        std::cerr << "RTXMetal: Compute command failed\n";
        return;
    }

    if (!SDL_UpdateTexture(texture, nullptr, [pixelBuffer contents], width * int(sizeof(uint32_t)))) {
        std::cerr << "RTXMetal: SDL_UpdateTexture: " << SDL_GetError() << "\n";
        return;
    }
    SDL_RenderTexture(ren, texture, nullptr, nullptr);
}
