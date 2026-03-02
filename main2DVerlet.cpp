#include <iostream>
#include <unistd.h>
#include <vector>
#include <SDL3/SDL.h>
#include <cmath>
using namespace std;
const double G=1.0;
const double dt=0.01;
const double ess=1e-12;
const int weight=1512;
const int height=970;
struct particle{
    string name;
    double x,y;
    double vx,vy;
    double ax,ay;
    double mass;
    double radius;
    int c1,c2,c3,c4;
};
struct camera{
    double x,y;
    double scale;
};
int worldToScreen(double w,camera& cam,char c){
    if(c=='x'){
        return (int)((w-cam.x)*cam.scale+weight/2);
    }else{
        return (int)(height/2-(w-cam.y)*cam.scale);
    }
}
void draw(SDL_Renderer* ren,particle& p,camera& cam){
    int sx=worldToScreen(p.x,cam,'x');
    int sy=worldToScreen(p.y,cam,'y');
    SDL_RenderPoint(ren,sx,sy);
}
void drawCircle(SDL_Renderer* ren,particle& p,camera& cam){
    int sx=worldToScreen(p.x,cam,'x');
    int sy=worldToScreen(p.y,cam,'y');
    int r=(int)(p.radius*cam.scale);
    for(int y=-r;y<=r;y++){
        for(int x=-r;x<=r;x++){
            if(x*x+y*y<=r*r){
                SDL_RenderPoint(ren,sx+x,sy+y);
            }
        }
    }
}
int main()
{
    vector<particle> planet;
    vector<double> newx,newy,newax,neway;
    // planet.push_back({"Star",0,0,0,2,0,0,1000,10,245,101,5,255});
    // planet.push_back({"Planet",100,0,0,2,0,0,1000,10,5,197,245,255});
    cout <<"Quantity of your stars: ";
    int n;
    cin>>n;
    //Storage
    newx.resize(n);
    newy.resize(n);
    newax.resize(n);
    neway.resize(n);
    //Storage
    for(int i=0;i<n;i++){
        string name;
        double x,y;
        double vx,vy;
        double ax,ay;
        double mass;
        double radius;
        int c1,c2,c3,c4;
        cin>>name>>x>>y>>vx>>vy>>ax>>ay>>mass>>radius>>c1>>c2>>c3>>c4;
        planet.push_back({name,x,y,vx,vy,ax,ay,mass,radius,c1,c2,c3,c4});
    }

    camera cam={0,0,1};
    SDL_Window* win=SDL_CreateWindow("Gravity system",weight,height,SDL_WINDOW_RESIZABLE);
    SDL_Renderer* ren=SDL_CreateRenderer(win,nullptr);
    SDL_SetWindowPosition(win,SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED);

    SDL_Event event;
    bool running=true;
    bool dragging=false;
    int lastX,lastY;
    long long rate=10000;
    while(running){
        SDL_PollEvent(&event);
        switch(event.type){
            case SDL_EVENT_QUIT:{
                running=false;
                break;
            }
            case SDL_EVENT_MOUSE_BUTTON_DOWN:{
                dragging=true;
                lastX=event.button.x;
                lastY=event.button.y;
                break;
            }
            case SDL_EVENT_MOUSE_BUTTON_UP:{
                dragging=false;
                break;
            }
            case SDL_EVENT_MOUSE_WHEEL:{
                if(event.wheel.y>0) cam.scale*=1.1;
                if(event.wheel.y<0) cam.scale*=0.9;
                break;
            }
            case SDL_EVENT_KEY_DOWN:{
                if(event.key.key==SDLK_UP&&rate>1){
                    rate*=0.1;
                }else if(event.key.key==SDLK_DOWN&&rate<100000){
                    rate*=10;
                }
            }
        }
        if(dragging==true&&event.type==SDL_EVENT_MOUSE_MOTION){
            int ddx=event.button.x-lastX;
            int ddy=event.button.y-lastY;
            cam.x-=ddx/cam.scale;
            cam.y+=ddy/cam.scale;
            lastX=event.button.x;
            lastY=event.button.y;
        }
        SDL_SetRenderDrawColor(ren,0,0,0,255);
        SDL_RenderClear(ren);
        //COMPUTE POSITION t+dt
        for(int cur=0;cur<planet.size();cur++){
            newx[cur]=planet[cur].x+planet[cur].vx*dt+0.5*planet[cur].ax*dt*dt;
            newy[cur]=planet[cur].y+planet[cur].vy*dt+0.5*planet[cur].ay*dt*dt;
        }
        for(int cur=0;cur<planet.size();cur++){
            newax[cur]=0;
            neway[cur]=0;
            for(int comp=0;comp<planet.size();comp++){
                if(cur==comp)continue;
                double dx=newx[comp]-newx[cur];
                double dy=newy[comp]-newy[cur];
                double r=sqrt(dx*dx+dy*dy+ess);
                double F=G*planet[cur].mass*planet[comp].mass/(r*r);

                newax[cur]+=F*dx/(r*planet[cur].mass);
                neway[cur]+=F*dy/(r*planet[cur].mass);
            }
            planet[cur].vx+=0.5*(planet[cur].ax+newax[cur])*dt;
            planet[cur].vy+=0.5*(planet[cur].ay+neway[cur])*dt;
            planet[cur].x=newx[cur];
            planet[cur].y=newy[cur];
            planet[cur].ax=newax[cur];
            planet[cur].ay=neway[cur];
            SDL_SetRenderDrawColor(ren,planet[cur].c1,planet[cur].c2,planet[cur].c3,planet[cur].c4);
            drawCircle(ren,planet[cur],cam);
            cout <<rate<<' '<<planet[cur].name<<':'<<planet[cur].x<<' '<<planet[cur].y<<' '<<planet[cur].vx<<' '<<planet[cur].vy<<' '<<planet[cur].ax<<' '<<planet[cur].ay<<endl;
        }

        
        usleep(rate);
        SDL_RenderPresent(ren);
    }
    return 0;
}