#define _USE_MATH_DEFINES   // MSVC: must come before any <cmath>/<math.h> include
#ifdef _WIN32
#define NOMINMAX          // keep windows.h from defining min/max as macros
#endif
#include <iostream>
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
#include "gpu.h"
#include <SDL3_ttf/SDL_ttf.h>
/* ---- CUDA 移植(新增): 光线追踪子系统需要的三个头文件 ----
   SDL_image    : 读 Texture/<行星名>/{px,nx,py,ny,pz,nz}.png 立方体贴图
   filesystem   : 判断某个行星有没有配套的贴图目录
   gpu_cuda.h   : 替代原来 Metal 的 buffer/encoder/dispatch 封装
   gpu_common.h : host 与 kernel 共用的结构体(布局必须完全一致)
   注意: 这两个头文件在工程根目录, 所以工程属性里要有 $(ProjectDir) 在
   AdditionalIncludeDirectories 中(VS 默认只有源文件目录, 这里显式加上)。 */
/* ---- CUDA 移植(改): 这里不再无脑依赖 SDL3_image 的开发包。
   网上下的 SDL3_image-3.4.8-win32-x64.zip 是"运行时版", 里面只有 SDL3_image.dll,
   没有 include\SDL3_image\SDL_image.h, 也没有 SDL3_image.lib, 所以 MSVC 会报
   "C1083: 无法打开包括文件 SDL3_image/SDL_image.h".
   现在默认改用工程目录里那份单文件 stb_image.h 读 png(功能上只要 png, 够用),
   不需要 SDL3_image 的头文件和 .lib。真想用 SDL3_image 的话,
   在工程属性 -> C/C++ -> 预处理器定义里加 SS_USE_SDL_IMAGE, 并链接 SDL3_image.lib 即可。 ---- */
#ifdef SS_USE_SDL_IMAGE
#include <SDL3_image/SDL_image.h>
#else
#define STB_IMAGE_IMPLEMENTATION   /* 只在这一个编译单元里展开实现 */
#include "stb_image.h"
#include <cstring>
#endif

#include <filesystem>
#include "gpu_common.h"
#include "gpu_cuda.h"
using namespace std;

/* 统一取图片: 把 png/jpg/bmp/tga 读成 SDL_Surface, 内存布局固定 4 字节/像素(字节序 R,G,B,A),
   和原来 IMG_Load 出来的 SDL_PIXELFORMAT_RGBA32 完全一致,
   所以下面 getTextureByCube / 背景星空里 tex->pixels + j*pitch + i*4 的取像素代码不用改。 */
/* 判断一组立方体面片是不是 6 张都成功读到了(读到一半失败要当成没有贴图,
   否则后面 RTXRender / RTXRenderMetal 里 tex->pixels 就是空指针解引用直接崩)。 */
static bool SS_SurfaceSetOK(SDL_Surface* const* faces, int n){
    for(int i = 0;i < n;i++){
        if(!faces[i] || faces[i]->pixels == nullptr || faces[i]->w <= 0 || faces[i]->h <= 0)return false;
    }
    return true;
}
static void SS_FreeSurfaceSet(SDL_Surface** faces, int n){
    for(int i = 0;i < n;i++){
        if(faces[i]){ SDL_DestroySurface(faces[i]); faces[i] = nullptr; }
    }
}
static SDL_Surface* SS_LoadImage(const char* path){
#ifdef SS_USE_SDL_IMAGE
    return IMG_Load(path);
#else
    if(!path || !*path)return nullptr;
    int w = 0, h = 0, chn = 0;
    stbi_uc* px = stbi_load(path, &w, &h, &chn, 4);   /* 强制 4 通道 RGBA */
    if(!px || w <= 0 || h <= 0){
        if(px)stbi_image_free(px);
        return nullptr;
    }
    SDL_Surface* sfc = SDL_CreateSurface(w, h, SDL_PIXELFORMAT_RGBA32);   /* SDL3 没有 SDL_CreateRGBSurfaceWithFormat, 用 SDL_CreateSurface */
    if(!sfc){
        stbi_image_free(px);
        return nullptr;
    }
    for(int y = 0;y < h;y++){
        memcpy((Uint8*)sfc->pixels + (size_t)y * sfc->pitch, px + (size_t)y * w * 4, (size_t)w * 4);
    }
    stbi_image_free(px);
    return sfc;
#endif
}

double G = 6.674e-11;
double dt = 0.01;
const double ess = 1e-12;
int width = 1512;
int height = 900;
//int width=800;
//int height=600;
double rocheK = 1.26;
int disintegrate_num = 1;
double disK = 2.0;
int RTXReflect = 1;   /* CUDA 移植(新增): 光线追踪的反射次数, 命令 reflection N */
struct particle {
    string name;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
    double mass;
    double radius;
    int c1, c2, c3, c4;
    bool stellar;
    bool merged;

    /* ---- CUDA 移植(新增): 光线追踪/自转/立方体贴图用到的字段.
       全部带默认值(NSDMI), 所以工程里原来那几十处
       push_back({"name", x,y,z, ... , stellar}) 一位都不用改。 ---- */
    double T = 24;                 // 自转周期(小时), 命令 rotationT <name> <T>
    double inclination = 0;        // 轨道倾角 -90 < x < 90 (度)
    double reflection = 0.5;       // 反射率 0~1, 命令 reflectionvalue <name> <v>

    SDL_Surface* texture[6] = {};  // px nx py ny pz nz, 由 texture 命令 IMG_Load 进来
    bool hasTexture = false;

    double w_velocity = 2 * M_PI / (T * 3600);   // 角速度 (RAD/s)
    double rotation = 0;                        // 当前自转角度 (RAD)
};
struct camera {
    double x, y, z;
    double yaw, pitch;
    double fov;
    double vx, vy, vz;
}cam;
struct projection {
    string name;
    double x, y, z;
    double sx, sy, sz;
    int c1, c2, c3, c4;
    double r;
    bool stellar;
}res;
struct SpaceShip {
    bool initial = false;
    string name = "SpaceX";
    double x, y, z;
    double vx = 0, vy = 0, vz = 0;
    double ax = 0, ay = 0, az = 0;
    double yaw = 0, pitch = 0;

    double mass = 120000;
    int engine_count = 9;
    double single_engine_thrust = 2745000;
    double single_engine_consumption = 0.1;
    float fuel = 1200000;
    float throttle = 0.0;

    bool target = false;
    string target_name = "";
    float distance = 0;
}myship;
bool isNumber(string& str);
double Screen(particle& p, camera& cam);
void drawCircle3(SDL_Renderer* ren, projection p, double fov_deg, bool Target);
int findFocus(vector<particle>& planet, camera& cam);
void disintegrate4(vector<particle>& p, vector<particle>& roc, double rocheR, double r, int cur, double dx, double dy, double dz, double K);
void merge(vector<particle>& p, int cur, int comp);
void execute(vector<particle>& p, string command, bool& pause, bool& roche, bool& debug, bool& SpaceShipMode, camera& cam, string& camera_lockin, bool& camera_locked, SDL_Window* win, bool& shader, bool& isMetal, bool& AutoDTime, long long& second, float& fps, int& perform, bool& rtx, bool& debug_bvh, int& debug_bvh_max);
void save(vector<particle> p);
void camera_movement(camera& cam, double friction, double acceleration, bool& camera_locked, string camera_lockin, vector<particle> p);
void position_movement(vector<particle>& planet, vector<double>& newx, vector<double>& newy, vector<double>& newz, vector<double>& newax, vector<double>& neway, vector<double>& newaz, bool roche, vector<particle>& roc, bool SpaceShipMode);
void position_movement_CUDA(vector<particle>& planet, vector<double>& newx, vector<double>& newy, vector<double>& newz, vector<double>& newax, vector<double>& neway, vector<double>& newaz, bool roche, vector<particle>& roc, bool SpaceShipMode);
string toScientific(double val, int precision);
void YourSpaceShip(camera& cam, SDL_Event& event, SDL_Window* win, SDL_Renderer* ren, bool& pause, int iniline);
void InMyShip(camera& cam, SDL_Event& event, SDL_Window* win, SDL_Renderer* ren, int iniline, bool& targetisTyping, vector<particle> p, bool pause, bool debug, TTF_Font* font);
void AutoOrbitSystem(bool first_lockedC, vector<particle> p);
void DrawTrajectorySystem(SDL_Renderer* ren, double proj, vector<particle> p, camera& cam);
void DrawCircleShader(SDL_Renderer* ren, projection p, double fov_deg, bool Target, double lx, double ly, double lz, double stellar_intensity);
void DrawTTF(SDL_Renderer* ren, TTF_Font* font, const char* text, float centerX, float centerY, SDL_Color col, float maxWidth, float x_scale, float y_scale);
void DrawStellarShader(SDL_Renderer* ren, projection p, double fov_deg);
/* ===================== CUDA 移植(新增): 光线追踪子系统的全局状态 =====================
   以下整块来自 ver3.0 (原 Metal 版 main.mm), host 侧 BVH 与向量运算部分。 ---- */
struct RTXV{
    double x, y, z;
    bool intersect = true;
    double OM = 0;
};
vector<vector<RTXV>> RTXOriginVector(height, vector<RTXV>(width, {0, 0, 0}));
vector<vector<double>> RTXVectorLength(height, vector<double>(width, 0));
vector<vector<RTXV>> RTXVector(height, vector<RTXV>(width, {0, 0, 0}));
struct BVHNode{
    RTXV C1, C2;
    vector<long> indices;
    int left = -1, right = -1;
};
vector<BVHNode> BVHCollector;
void RTXInitial(int rtxheight = height, int rtxwidth = width);
void RTXRender(SDL_Renderer* ren, const vector<projection> &planet, const vector<particle> &parti, vector<int> suns);
void RTXRenderMetal(SDL_Renderer* ren, const vector<projection> &planet, const vector<particle> &parti, const vector<int> &suns);
void DrawBVHDebug(SDL_Renderer* ren, const vector<BVHNode> &BVHCollector, const camera &cam, int maxDepth = 3);
struct SysSetting{
    bool SSAA = false;
    int SSAA_multi = 2;
    bool Texture = false;
    bool rotationable = true;
    int follow = -1;
    bool hasbackground = false;
    particle background;
    
    bool textureChanged = true;
    bool backgroundChanged = true;
}ss;
/* =========================================================================== */
int main()
{
    vector<particle> planet;
    vector<particle> roc;

    vector<double> newx, newy, newz, newax, neway, newaz;
    bool pause = false;
    bool roche = false;
    // planet.push_back({"Star",0,0,0,2,0,0,1000,10,245,101,5,255});
    // planet.push_back({"Planet",100,0,0,2,0,0,1000,10,5,197,245,255});
    //Initial
    cam = { 0,0,-200,0,0,60,0,0,0 };
    SDL_Window* win = SDL_CreateWindow("<Gravity System> version->2.2.5", width, height, SDL_WINDOW_RESIZABLE);
    SDL_Renderer* ren = SDL_CreateRenderer(win, nullptr);
    SDL_SetWindowPosition(win, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    SDL_SetWindowRelativeMouseMode(win, true);
    TTF_Init();
    const char* base = SDL_GetBasePath();
    string full_path = string(base) + "font.ttf";
    TTF_Font* font = TTF_OpenFont(full_path.c_str(), 40);
    const double mousesensitivity = 0.1;
    bool shader = false;
    int perform = 1;
    /*Initial movement*/
    double acceleration = 0.5;
    double friction = 0.94;
    /*Initial movement*/
    float fps = 0;
    Uint64 frame_count = 0;
    Uint64 last_time = SDL_GetTicks();
    SDL_Event event;
    bool running = true;
    bool camera_locked = false;
    string camera_lockin = "";
    bool mouse_locked = true;
    bool debug = true;
    bool SpaceShipMode = false;
    bool AutoDTime = true;
    long long accelerate = 1;
    long long second = 1;
    //Inital input&SpaceShip Initial input
    bool isTyping = false;
    bool shipisTyping = false;
    bool targetisTyping = false;
    string commandbuf = "";
    vector<string> currentbuf;
    int line = 0;
    int fontsize = 24;
    //Initial Computing Methods
    bool isMetal = false;
    /* ---- CUDA 移植(新增): 光线追踪开关与 RTX 缓冲区的尺寸缓存 ---- */
    bool rtx = false;              /* 命令 rtx 切换 */
    bool debug_bvh = false;        /* 命令 debugbvh <深度> 切换 */
    int  debug_bvh_max = 3;
    int  last_width = -1, last_height = -1, last_fov = -1, last_ssaa = -1, last_ssaamulti = -1;

    while (running) {
        int initialT = 0.7 * height;
        int iniline = initialT;
        newx.resize(planet.size());
        newy.resize(planet.size());
        newz.resize(planet.size());
        newax.resize(planet.size());
        neway.resize(planet.size());
        newaz.resize(planet.size());
        SDL_GetWindowSize(win, &width, &height);

        /* ---- CUDA 移植(新增): RTXOriginVector / RTXVectorLength / RTXVector 是按
           像素建的表, 窗口尺寸、FOV 或 SSAA 一变就必须重建, 否则越界或画面错位。 ---- */
        if (last_width != width || last_height != height || last_fov != (int)cam.fov
            || last_ssaa != (int)ss.SSAA || last_ssaamulti != ss.SSAA_multi) {
            if (ss.SSAA)RTXInitial(height * ss.SSAA_multi, width * ss.SSAA_multi);
            else RTXInitial();
            last_width = width;
            last_height = height;
            last_fov = (int)cam.fov;
            last_ssaa = (int)ss.SSAA;
            last_ssaamulti = ss.SSAA_multi;
        }

        /*Mouse: drain EVERY queued motion event, not one per frame.
          A 1000Hz mouse queues ~1000 of these per second; consuming one per
          frame lets the backlog - and the lag - grow without bound.
          Draining them first also stops keyboard events from being stuck
          behind the motion flood. */
        SDL_PumpEvents();
        double mdx = 0, mdy = 0;
        {
            SDL_Event mev[64];
            int n;
            do {
                n = SDL_PeepEvents(mev, 64, SDL_GETEVENT,
                    SDL_EVENT_MOUSE_MOTION, SDL_EVENT_MOUSE_MOTION);
                for (int i = 0;i < n;i++) {
                    mdx += mev[i].motion.xrel;
                    mdy += mev[i].motion.yrel;
                }
            } while (n == 64);
        }
        if (mouse_locked) {
            cam.yaw += mdx * mousesensitivity;
            cam.pitch -= mdy * mousesensitivity;
            if (cam.yaw > 360)cam.yaw -= 360;
            if (cam.yaw < 0)cam.yaw += 360;
            if (cam.pitch > 90)cam.pitch = 90;
            if (cam.pitch < -90)cam.pitch = -90;
        }

        SDL_PollEvent(&event);

        switch (event.type) {
        case SDL_EVENT_QUIT: {
            running = false;
            break;
        }case SDL_EVENT_KEY_DOWN: {
            if (isTyping == false && shipisTyping == false && targetisTyping == false) {
                if (event.key.key == SDLK_UP) {
                    if (AutoDTime && accelerate < 9999999999999999) {
                        second += accelerate;
                        accelerate += 5;
                    }
                    else dt *= 10;
                }
                else if (event.key.key == SDLK_DOWN) {
                    if (AutoDTime && accelerate < 9999999999999999) {
                        second -= accelerate;
                        accelerate += 5;
                    }
                    else dt *= 0.1;
                }
                else if (event.key.key == SDLK_LEFT) {
                    cam.fov = max(20.0, cam.fov - 10.0);
                }
                else if (event.key.key == SDLK_RIGHT) {
                    cam.fov = min(160.0, cam.fov + 10.0);
                }
                else if (event.key.key == SDLK_P) {
                    acceleration *= 10;
                }
                else if (event.key.key == SDLK_O && acceleration > 0.2) {
                    acceleration *= 0.1;
                }
                else if (event.key.key == SDLK_TAB) {
                    pause = !pause;
                }
                else if (event.key.key == SDLK_T) {
                    isTyping = true;
                    commandbuf = "";
                    currentbuf.clear();
                    line = 0;
                    currentbuf.push_back(">");
                    SDL_StartTextInput(win);
                }
            }
            if (event.key.key == SDLK_ESCAPE) {
                if (isTyping == true) {
                    isTyping = false;
                    SDL_StopTextInput(win);
                    if (SpaceShipMode)pause = false;
                }
                else {
                    mouse_locked = false;
                    SDL_SetWindowRelativeMouseMode(win, false);
                }
            }
            break;
        }
        case SDL_EVENT_KEY_UP: {
            if (isTyping == false && shipisTyping == false && targetisTyping == false) {
                if (event.key.key == SDLK_UP) {
                    accelerate = 1;
                }
                if (event.key.key == SDLK_DOWN) {
                    accelerate = 1;
                }
            }
            break;
        }
        case SDL_EVENT_MOUSE_MOTION: {
            break;   /*handled above, in bulk*/
        }
        case SDL_EVENT_MOUSE_BUTTON_DOWN: {
            if (!mouse_locked) {
                mouse_locked = true;
                SDL_SetWindowRelativeMouseMode(win, true);
            }
            break;   /*was missing - the MOTION case used to fall through*/
        }
        }
        SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
        SDL_RenderClear(ren);

        //COMPUTE POSITION t+dt
        if (pause == false) {
            if (isMetal == true)position_movement_CUDA(planet, newx, newy, newz, newax, neway, newaz, roche, roc, SpaceShipMode);
            else position_movement(planet, newx, newy, newz, newax, neway, newaz, roche, roc, SpaceShipMode);
        }
        if (roche == true) {
            //Merge function
            long fsize = planet.size();
            for (int i = 0;i < fsize;i++) {
                if (planet[i].merged == true) {
                    planet.erase(planet.begin() + i);
                    i--;
                    fsize--;
                }
            }
            //Roche function
            planet.insert(planet.end(), roc.begin(), roc.end());
            roc.clear();
        }

        /* ---- CUDA 移植(新增): 自转推进。texture 画出来的球面靠 rotation 采样贴图,
           没有这一段贴图就是静止的。pause 或 rotation 命令关掉时不推进。 ---- */
        if (!pause && ss.rotationable) {
            for (int i = 0;i < planet.size();i++) {
                planet[i].rotation += planet[i].w_velocity * dt;
                if (planet[i].rotation > (2 * M_PI))planet[i].rotation -= (2 * M_PI);
                if (planet[i].rotation < (-2 * M_PI))planet[i].rotation += (2 * M_PI);
            }
        }

        //RENDER
        vector<projection> rendering;
        for (int cur = 0;cur < planet.size();cur++) {
            // drawCircle(ren,planet[cur],cam);
            if (planet[cur].name == camera_lockin && camera_locked == true) {
                cam.x = planet[cur].x;
                cam.y = planet[cur].y;
                cam.z = planet[cur].z;
                continue;
            }
            Screen(planet[cur], cam);
            if (res.sz > 1e-1) {
                rendering.push_back({ planet[cur].name,planet[cur].x,planet[cur].y,planet[cur].z,res.sx,res.sy,res.sz,planet[cur].c1,planet[cur].c2,planet[cur].c3,planet[cur].c4,planet[cur].radius,planet[cur].stellar });
            }
        }
        sort(rendering.begin(), rendering.end(), [](const projection& a, const projection& b) {
            return a.sz > b.sz;
            });
        /* ---- CUDA 移植(新增): rtx 分支。suns = 发光体(恒星)下标列表, perform<3 时
           只取第一个恒星(和 ver3.0 一致, 省性能)。isMetal 决定光线追踪本身跑在
           CUDA 上还是 CPU 上; 两条路径输出的是同一套像素。 ---- */
        if (rtx == true && planet.size() > 0) {
            vector<int> suns;
            for (int i = 0;i < planet.size();i++) {
                if (planet[i].stellar) {
                    suns.push_back(i);
                    if (perform < 3)break;
                }
            }
            if (isMetal)RTXRenderMetal(ren, rendering, planet, suns);
            else RTXRender(ren, rendering, planet, suns);
            if (debug_bvh)DrawBVHDebug(ren, BVHCollector, cam, debug_bvh_max);
        }
        else {   /* CUDA 移植: 下面这段原来没有缩进外层, 就是本工程 v2.2.5 的圆点渲染 */
        bool target = false;
        for (int cur = 0;cur < rendering.size();cur++) {
            if (SpaceShipMode && myship.target && myship.target_name == rendering[cur].name)target = true;
            else target = false;
            if (!shader) {
                SDL_SetRenderDrawColor(ren, rendering[cur].c1, rendering[cur].c2, rendering[cur].c3, rendering[cur].c4);
                drawCircle3(ren, rendering[cur], cam.fov, target);
            }
            if (shader) {
                if (rendering[cur].stellar) {
                    DrawStellarShader(ren, rendering[cur], cam.fov);
                    //                    SDL_SetRenderDrawColor(ren,rendering[cur].c1,rendering[cur].c2,rendering[cur].c3,rendering[cur].c4);
                    //                    DrawStellarShader(ren, rendering[cur], cam.fov,target);
                    //                    drawCircle3(ren,rendering[cur],cam.fov,target);
                    continue;
                }
                if (sqrt((cam.x - rendering[cur].x) * (cam.x - rendering[cur].x) + (cam.y - rendering[cur].y) * (cam.y - rendering[cur].y) + (cam.z - rendering[cur].z)) > 1e16 && perform == 1) {
                    SDL_SetRenderDrawColor(ren, rendering[cur].c1, rendering[cur].c2, rendering[cur].c3, rendering[cur].c4);
                    drawCircle3(ren, rendering[cur], cam.fov, target);
                    continue;
                }
                DrawCircleShader(ren, rendering[cur], cam.fov, target, 0, 0, 0, 0);
                double stellar_intensity = 0;
                double lx = 0, ly = 0, lz = 0;
                double dr = ess;
                for (int i = 0;i < planet.size();i++) {
                    if (planet[i].stellar == true) {
                        double drx = planet[i].x - rendering[cur].x;
                        double dry = planet[i].y - rendering[cur].y;
                        double drz = planet[i].z - rendering[cur].z;
                        dr = sqrt(drx * drx + dry * dry + drz * drz) + ess;
                        //IDK WHY BUT THIS VECTOR IS GOOD TO WORK

                        double a = cam.yaw * (M_PI / 180.0);
                        double b = cam.pitch * (M_PI / 180.0);
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
                        stellar_intensity = max(1.0, log10(physical_intensity * scale + 1.0) / log10(I0 * scale + 1.0));

                        lx += stellar_intensity * rx / view_dist;
                        ly += stellar_intensity * -ry / view_dist;
                        lz += stellar_intensity * -rz / view_dist;
                        SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_ADD);
                        DrawCircleShader(ren, rendering[cur], cam.fov, target, lx, ly, lz, stellar_intensity);
                        if (perform <= 2) {
                            break;
                        }
                    }
                }
                SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_NONE);
            }

        }
        }   /* end of 非 rtx 渲染分支 */
        /*COMPUTE MOVEMENT*/
        if (isTyping == false) {
            if (SpaceShipMode == false)camera_movement(cam, friction, acceleration, camera_locked, camera_lockin, planet);
            if (SpaceShipMode == true) {
                if (myship.initial == false) {
                    shipisTyping = true;
                    YourSpaceShip(cam, event, win, ren, pause, iniline);
                }
                else {
                    shipisTyping = false;
                    InMyShip(cam, event, win, ren, iniline, targetisTyping, planet, pause, debug, font);
                }

            }
        }
        /*Console Input*/
        if (isTyping == true) {
            if (SpaceShipMode)pause = true;
            if (event.type == SDL_EVENT_TEXT_INPUT) {
                commandbuf += event.text.text;
                currentbuf[line] += event.text.text;
            }
            else if (event.type == SDL_EVENT_KEY_DOWN) {
                if (event.key.key == SDLK_BACKSPACE && currentbuf[line].size() > 1) {
                    commandbuf.pop_back();
                    currentbuf[line].pop_back();
                }
                if ((SDL_KMOD_LSHIFT & event.key.mod) && event.key.key == SDLK_RETURN) {
                    if (initialT + line * 0.5 * fontsize > height - fontsize * 0.5) {
                        iniline -= (0.5 * fontsize);
                    }
                    line += 1;
                    commandbuf += " ";
                    currentbuf.push_back(">");
                }
                else if (event.key.key == SDLK_RETURN) {
                    execute(planet, commandbuf, pause, roche, debug, SpaceShipMode, cam, camera_lockin, camera_locked, win, shader, isMetal, AutoDTime, second, fps, perform, rtx, debug_bvh, debug_bvh_max);
                    isTyping = false;
                    SDL_StopTextInput(win);
                    if (SpaceShipMode)pause = false;
                    continue;
                }
                if ((SDL_KMOD_GUI & event.key.mod) && event.key.key == SDLK_V) {
                    if (SDL_HasClipboardText()) {
                        char* text = SDL_GetClipboardText();
                        commandbuf += text;
                        currentbuf[line] += text;
                        SDL_free(text);
                    }
                }

            }
            SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
            for (int i = 0;i <= line;i++) {
                SDL_RenderDebugText(ren, 50, iniline + i * 0.5 * fontsize, currentbuf[i].c_str());
            }
        }
        /*Console Input*/
        /*Debug*/
        if (debug == true) {
            frame_count++;
            Uint64 current_time = SDL_GetTicks();
            if (current_time - last_time >= 1000) {
                fps = frame_count / ((current_time - last_time) / 1000.f);
                frame_count = 0;
                last_time = current_time;
            }
            if (AutoDTime)dt = second / fps;
            string debug_buf1 = "X:" + to_string(cam.x) + " Y:" + to_string(cam.y) + " Z:" + to_string(cam.z) + "  YAW:" + to_string(cam.yaw) + " PITCH:" + to_string(cam.pitch) + "    Quantity:" + to_string(planet.size());
            if (AutoDTime) {
                debug_buf1 += " Scale:";
                long long minutes = second / 60;
                long long hours = second / 3600;
                long long days = second / 86400;
                long long years = second / 31536000;
                if (years != 0)debug_buf1 += to_string(years) + "y ";
                if (days != 0)debug_buf1 += to_string(days % 365) + "d ";
                if (hours != 0)debug_buf1 += to_string(hours % 24) + "hr ";
                if (minutes != 0)debug_buf1 += to_string(minutes % 60) + "min ";
                debug_buf1 += to_string(second % 60) + "s /s";
            }
            string debug_buf2 = "RunSpeed(dt/s):" + to_string(dt) + " acceleration:" + to_string(acceleration) + " FOV:" + to_string(cam.fov) + " fps:" + to_string(fps);
            string debug_buf3 = "RocheLimit:[" + to_string(roche) + "] " + to_string(rocheK);
            string debug_buf4 = "G:" + toScientific(G, 4);
            string debug_buf5 = "SpaceShipMode:" + to_string(SpaceShipMode);
            if (pause == true)debug_buf1 += "  [PAUSE]";
            SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
            SDL_RenderDebugText(ren, 20, 10, debug_buf1.c_str());
            SDL_RenderDebugText(ren, 20, 22, debug_buf2.c_str());
            SDL_RenderDebugText(ren, 20, 34, debug_buf3.c_str());
            SDL_RenderDebugText(ren, 20, 46, debug_buf4.c_str());
            SDL_RenderDebugText(ren, 20, 58, debug_buf5.c_str());
        }
        /*Debug*/
        SDL_RenderPresent(ren);
    }
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    gpu_shutdown();
    gpuQuit();   /* CUDA 移植(新增): 释放 gpu_cuda.cu 里的常量暂存区 */
    return 0;
}
/*



            Function



 */
void camera_movement(camera& cam, double friction, double acceleration, bool& camera_locked, string camera_lockin, vector<particle> p) {
    const bool* keystate = SDL_GetKeyboardState(NULL);
    double rad_yaw = cam.yaw * (M_PI / 180);
    double rad_pitch = cam.pitch * (M_PI / 180);

    double fx = sin(rad_yaw) * cos(rad_pitch);
    double fy = sin(rad_pitch);
    double fz = cos(rad_yaw) * cos(rad_pitch);
    double rx = cos(rad_yaw);
    double rz = -sin(rad_yaw);
    double ux = sin(rad_yaw) * sin(-rad_pitch);
    double uy = cos(rad_pitch);
    double uz = cos(rad_yaw) * sin(-rad_pitch);

    if (camera_locked == true && (keystate[SDL_SCANCODE_W] || keystate[SDL_SCANCODE_S] || keystate[SDL_SCANCODE_A] || keystate[SDL_SCANCODE_D] || keystate[SDL_SCANCODE_SPACE] || keystate[SDL_SCANCODE_LSHIFT]))camera_locked = false;
    if (camera_locked == true) {
        cam.vx = 0; cam.vy = 0; cam.vz = 0;
        for (int i = 0;i < p.size();i++) {
            if (p[i].name == camera_lockin) {
                cam.x = p[i].x; cam.y = p[i].y; cam.z = p[i].z;
                return;
            }
        }
    }
    if (keystate[SDL_SCANCODE_W]) {
        cam.vx += fx * acceleration;
        cam.vy += fy * acceleration;
        cam.vz += fz * acceleration;
    }if (keystate[SDL_SCANCODE_S]) {
        cam.vx -= fx * acceleration;
        cam.vy -= fy * acceleration;
        cam.vz -= fz * acceleration;
    }if (keystate[SDL_SCANCODE_A]) {
        cam.vx -= rx * acceleration;
        cam.vz -= rz * acceleration;
    }if (keystate[SDL_SCANCODE_D]) {
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
void position_movement(vector<particle>& planet, vector<double>& newx, vector<double>& newy, vector<double>& newz, vector<double>& newax, vector<double>& neway, vector<double>& newaz, bool roche, vector<particle>& roc, bool SpaceShipMode) {
    if (SpaceShipMode) {
        //        myship.ax=0; myship.ay=0; myship.az=0;
        planet.push_back({ myship.name,myship.x,myship.y,myship.z,myship.vx,myship.vy,myship.vz,myship.ax,myship.ay,myship.az,myship.mass,1,0,0,0,0,false });
        newx.resize(planet.size());
        newy.resize(planet.size());
        newz.resize(planet.size());
        newax.resize(planet.size());
        neway.resize(planet.size());
        newaz.resize(planet.size());
    }
    for (int cur = 0;cur < planet.size();cur++) {
        newx[cur] = planet[cur].x + planet[cur].vx * dt + 0.5 * planet[cur].ax * dt * dt;
        newy[cur] = planet[cur].y + planet[cur].vy * dt + 0.5 * planet[cur].ay * dt * dt;
        newz[cur] = planet[cur].z + planet[cur].vz * dt + 0.5 * planet[cur].az * dt * dt;
    }
    long fsize = planet.size();
#pragma omp parallel for
    for (int cur = 0;cur < fsize;cur++) {
        newax[cur] = 0;
        neway[cur] = 0;
        newaz[cur] = 0;
        for (int comp = 0;comp < fsize;comp++) {
            if (cur == comp)continue;
            double dx = newx[comp] - newx[cur];
            double dy = newy[comp] - newy[cur];
            double dz = newz[comp] - newz[cur];
            double r = sqrt(dx * dx + dy * dy + dz * dz + ess);
            double F = G * planet[cur].mass * planet[comp].mass / (r * r);

            newax[cur] += F * dx / (r * planet[cur].mass);
            neway[cur] += F * dy / (r * planet[cur].mass);
            newaz[cur] += F * dz / (r * planet[cur].mass);

            //Target Name
            if (SpaceShipMode && cur == fsize - 1 && myship.target && myship.target_name == planet[comp].name) {
                myship.distance = r;
            }
            //Compute roche limit
            if (roche == true && planet[cur].radius > disintegrate_num) {
                double rocheR = rocheK * planet[cur].radius * pow(2 * planet[comp].mass / planet[cur].mass, 1.0 / 3.0);
                if (r > rocheR)continue;
                disintegrate4(planet, roc, rocheR, r, cur, dx, dy, dz, disK);
            }
            //Compute merge limit
            if (roche == true && planet[cur].merged == false && planet[comp].merged == false) {
                if (r > (planet[comp].radius - planet[cur].radius))continue;
                merge(planet, cur, comp);
                planet[cur].merged = true;
                planet[comp].merged = true;
            }
        }
        planet[cur].vx += 0.5 * (planet[cur].ax + newax[cur]) * dt; planet[cur].vy += 0.5 * (planet[cur].ay + neway[cur]) * dt; planet[cur].vz += 0.5 * (planet[cur].az + newaz[cur]) * dt;
        planet[cur].x = newx[cur]; planet[cur].y = newy[cur]; planet[cur].z = newz[cur];
        planet[cur].ax = newax[cur]; planet[cur].ay = neway[cur]; planet[cur].az = newaz[cur];

    }
    if (SpaceShipMode) {
        myship.x = planet[fsize - 1].x; myship.y = planet[fsize - 1].y; myship.z = planet[fsize - 1].z;
        myship.vx = planet[fsize - 1].vx; myship.vy = planet[fsize - 1].vy; myship.vz = planet[fsize - 1].vz;
        myship.ax = planet[fsize - 1].ax; myship.ay = planet[fsize - 1].ay; myship.az = planet[fsize - 1].az;
        planet.pop_back();
    }
}

void position_movement_CUDA(vector<particle>& planet, vector<double>& newx, vector<double>& newy, vector<double>& newz, vector<double>& newax, vector<double>& neway, vector<double>& newaz, bool roche, vector<particle>& roc, bool SpaceShipMode) {

    if (SpaceShipMode) {
        //        myship.ax=0; myship.ay=0; myship.az=0;
        planet.push_back({ myship.name,myship.x,myship.y,myship.z,myship.vx,myship.vy,myship.vz,myship.ax,myship.ay,myship.az,myship.mass,1,0,0,0,0,false });
        newx.resize(planet.size());
        newy.resize(planet.size());
        newz.resize(planet.size());
        newax.resize(planet.size());
        neway.resize(planet.size());
        newaz.resize(planet.size());
    }

    for (int cur = 0;cur < planet.size();cur++) {
        newx[cur] = planet[cur].x + planet[cur].vx * dt + 0.5 * planet[cur].ax * dt * dt;
        newy[cur] = planet[cur].y + planet[cur].vy * dt + 0.5 * planet[cur].ay * dt * dt;
        newz[cur] = planet[cur].z + planet[cur].vz * dt + 0.5 * planet[cur].az * dt * dt;
        planet[cur].x = newx[cur];
        planet[cur].y = newy[cur];
        planet[cur].z = newz[cur];
    }

    /*Scale*/
    double maxCoord = 1.0, maxMass = 1.0;
    for (int i = 0;i < planet.size();i++) {
        maxCoord = max({ maxCoord, fabs(planet[i].x), fabs(planet[i].y), fabs(planet[i].z) });
        maxMass = max(maxMass, planet[i].mass);
    }
    double Ls = max(1.0, maxCoord / 1e6);
    double Ms = max(1.0, maxMass / 1e6);
    float nG = (float)(G * Ms / (Ls * Ls));

    long N = planet.size();
    float ndt = dt;

    if (N <= 0)return;

    vector<float> nx(N), ny(N), nz(N), vx(N), vy(N), vz(N), ax(N), ay(N), az(N), mass(N);

    for (int i = 0;i < planet.size();i++) {
        nx[i] = (float)(planet[i].x / Ls);
        ny[i] = (float)(planet[i].y / Ls);
        nz[i] = (float)(planet[i].z / Ls);
        vx[i] = planet[i].vx;
        vy[i] = planet[i].vy;
        vz[i] = planet[i].vz;
        ax[i] = planet[i].ax;
        ay[i] = planet[i].ay;
        az[i] = planet[i].az;
        mass[i] = (float)(planet[i].mass / Ms);
    }

    gpu_nbody_step((int)N,
        nx.data(), ny.data(), nz.data(), mass.data(),
        vx.data(), vy.data(), vz.data(),
        ax.data(), ay.data(), az.data(),
        ndt, nG);

    for (int i = 0;i < N;i++) {
        //Backput Velocity
        planet[i].vx = vx[i];
        planet[i].vy = vy[i];
        planet[i].vz = vz[i];
        //Backput Accelerate
        planet[i].ax = ax[i];
        planet[i].ay = ay[i];
        planet[i].az = az[i];
    }
    if (SpaceShipMode) {
        for (int cur = 0;cur < N - 1;cur++) {
            if (SpaceShipMode && myship.target && myship.target_name == planet[cur].name) {
                double dx = myship.x - planet[cur].x;
                double dy = myship.y - planet[cur].y;
                double dz = myship.z - planet[cur].z;
                double r = sqrt(dx * dx + dy * dy + dz * dz + ess);
                myship.distance = r;
            }
        }
        myship.x = planet[N - 1].x; myship.y = planet[N - 1].y; myship.z = planet[N - 1].z;
        myship.vx = planet[N - 1].vx; myship.vy = planet[N - 1].vy; myship.vz = planet[N - 1].vz;
        myship.ax = planet[N - 1].ax; myship.ay = planet[N - 1].ay; myship.az = planet[N - 1].az;
        planet.pop_back();
    }
}

void save(vector<particle> p) {
#ifdef _WIN32
    const char* home = getenv("USERPROFILE");
#else
    const char* home = getenv("HOME");
#endif
    string path = string(home ? home : ".") + "/Documents/running.txt";
    ofstream file(path);
    file << p.size() << endl;
    for (int i = 0;i < p.size();i++) {
        file << p[i].name << ' '
            << p[i].x << ' ' << p[i].y << ' ' << p[i].z << ' '
            << p[i].vx << ' ' << p[i].vy << ' ' << p[i].vz << ' '
            << p[i].ax << ' ' << p[i].ay << ' ' << p[i].az << ' '
            << p[i].mass << ' ' << p[i].radius << ' '
            << p[i].c1 << ' ' << p[i].c2 << ' ' << p[i].c3 << ' ' << p[i].c4 << ' ';
        if (p[i].stellar == true) {
            file << 1 << endl;
        }
        else {
            file << 0 << endl;
        }
    }
    file.close();
}
void execute(vector<particle>& p, string command, bool& pause, bool& roche, bool& debug, bool& SpaceShipMode, camera& cam, string& camera_lockin, bool& camera_locked, SDL_Window* win, bool& shader, bool& isMetal, bool& AutoDTime, long long& second, float& fps, int& perform, bool& rtx, bool& debug_bvh, int& debug_bvh_max) {
    stringstream ssin(command);
    string n;
    ssin >> n;

    if (n == "-3" || n == "pause") {
        pause = !pause;
        return;
    }
    if (n == "-4" || n == "roche") {
        roche = !roche;
        return;
    }
    if (n == "-5" || n == "newrochek") {
        double newrocheK;
        ssin >> newrocheK;
        rocheK = newrocheK;
        return;
    }
    if (n == "-6" || n == "newg") {
        double newG;
        ssin >> newG;
        G = newG;
        return;
    }
    if (n == "-7" || n == "newdnum") {
        double newdisK;
        ssin >> newdisK;
        disK = newdisK;
        return;
    }
    if (n == "-8" || n == "newdsize") {
        double newdisnum;
        ssin >> newdisnum;
        disintegrate_num = newdisnum;
        return;
    }
    if (n == "debug") {
        debug = !debug;
        return;
    }
    if (n == "clear") {
        p.clear();
        SpaceShipMode = false;
        ss.Texture = false;   /* CUDA 移植: 行星都没了, 贴图状态也要清掉 */
        return;
    }
    if (n == "goto") {
        string goname;
        ssin >> goname;
        for (int i = 0;i < p.size();i++) {
            if (goname == p[i].name) {
                cam.x = p[i].x;
                cam.y = p[i].y;
                cam.z = p[i].z;
                cam.vx = 0;
                cam.vy = 0;
                cam.vz = 0;
            }
        }
        return;
    }
    if (n == "target") {
        string target;
        ssin >> target;
        camera_lockin = target;
        for (int i = 0;i < p.size();i++) {
            if (target == p[i].name) {
                camera_locked = true;
                break;
            }
        }
        return;
    }
    if (n == "save") {
        save(p);
        return;
    }
    if (n == "spaceshipmode") {
        SpaceShipMode = !SpaceShipMode;
        myship.x = cam.x; myship.y = cam.y; myship.z = cam.z;
        return;
    }
    if (n == "clearmyship") {
        SpaceShipMode = false;
        myship.initial = false;
        return;
    }
    if (n == "in") {
        string method;
        ssin >> method;
        if (method == "Metal" || method == "CUDA") {
            isMetal = true;
        }
    }
    if (n == "out") {
        string method;
        ssin >> method;
        if (method == "Metal" || method == "CUDA") {
            isMetal = false;
        }
    }
    if (n == "perform") {
        string per;
        ssin >> per;
        if (per == "1") {
            perform = 1;
        }
        else if (per == "2") {
            perform = 2;
        }
        else if (per == "3") {
            perform = 3;
        }
    }
    if (n == "sel") {
        string name;
        ssin >> name;
        if (name == "solar" || name == "solar2" || name == "solar3") {
            p.push_back({ "Sun", 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.9885e30, 695700000, 255, 210, 26, 255, true });
            p.push_back({ "Mercury", 6.98179e10, 0, 0, 0, 0, 47870, 0, 0, 0, 3.3022e23, 2440000, 194, 194, 194, 255, false });
            p.push_back({ "Venus", 1.08939e11, 0, 0, 0, 0, 35020, 0, 0, 0, 4.8676e24, 6051000, 255, 236, 59, 255, false });
            p.push_back({ "Earth", 1.521e11, 0, 0, 0, 0, 29780, 0, 0, 0, 5.97237e24, 6.371e6, 28, 115, 232, 255, false });
            p.push_back({ "Mars", 2.49261e11, 0, 0, 0, 0, 24080, 0, 0, 0, 6.4169e23, 3.3895e6, 245, 91, 22, 255, false });
            p.push_back({ "Jupiter", 8.165208e11, 0, 0, 0, 0, 13070, 0, 0, 0, 1.8981e27, 6.9886e7, 255, 213, 125, 255, false });
            p.push_back({ "Saturnus", 1.353572956e12, 0, 0, 0, 0, 9690, 0, 0, 0, 5.6846e26, 5.8232e7, 219, 167, 61, 255, false });
            p.push_back({ "Uranus", 3.004419704e12, 0, 0, 0, 0, 6810, 0, 0, 0, 8.6810e25, 2.5559e7, 215, 248, 250, 255, false });
            p.push_back({ "Neptune", 4.55394649e12, 0, 0, 0, 0, 5430, 0, 0, 0, 1.0243e26, 2.4764e7, 65, 105, 225, 255, false });
            p.push_back({ "Moon", 1.51736896e11, 0, 0, 0, 0, 30802, 0, 0, 0, 7.3477e22, 1.7371e6, 194, 194, 194, 255, false });
            if (name == "solar2" || name == "solar3") {
                p.push_back({ "MarsI",2.4925188e11,0,0,0,0,26229.2183,0,0,0,1.8e15,1.11e4,160,160,160,255,false });
                p.push_back({ "MarsII",2.4923754e11,0,0,0,0,25428.3773,0,0,0,1.48e15,6.2e3,163,163,163,255,false });
                p.push_back({ "J1",8.160991e11,0,0,0,0,30366.991,0,0,0,8.94e22,1.8213e6,164,164,164,255,false });
                p.push_back({ "J2",8.158499e11,0,0,0,0,26783.347,0,0,0,4.799844e22,1.5608e6,165,165,165,255,false });
                p.push_back({ "J3",8.154504e11,0,0,0,0,23926.737,0,0,0,1.4819e23,2.6431e6,166,166,166,255,false });
                p.push_back({ "J4",8.146381e11,0,0,0,0,19771.37,0,0,0,1.075938e23,2.4103e6,168,168,168,255,false });
                p.push_back({ "J5",8.1633944e11,0,0,0,0,39445.13,0,0,0,2.08e18,8.35e4,170,170,170,255,false });
                p.push_back({ "J6",8.034388e11,0,0,0,0,16175.526,0,0,0,4.2e18,1.414e5,172,172,172,255,false });
                p.push_back({ "J7",8.034388e11,0,0,0,0,16348.221,0,0,0,8.7e17,4.3e4,174,174,174,255,false });
                if (name == "solar3") {
                    double s_x = 1.353572956e12, s_y = 0, s_z = 0;
                    double s_vx = 0, s_vy = 0, s_vz = 9690;
                    double s_m = 5.6846e26, s_r = 5.8232e7;
                    double tilt = 26.73 * (M_PI / 180.0);
                    int ring_count = 10000;
                    random_device rd;
                    mt19937 gen(rd());
                    uniform_real_distribution<> dis(0.0, 1.0);
                    for (int i = 0;i < ring_count;i++) {
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
                        double rx = rx_0;
                        double ry = ry_0 * cos(tilt) - rz_0 * sin(tilt);
                        double rz = ry_0 * sin(tilt) + rz_0 * cos(tilt);

                        double rvx = rvx_0;
                        double rvy = rvy_0 * cos(tilt) - rvz_0 * sin(tilt);
                        double rvz = rvy_0 * sin(tilt) + rvz_0 * cos(tilt);
                        //Color
                        int cr = (int)(219 * final_shade);
                        int cg = (int)(167 * final_shade);
                        int cb = (int)(100 * final_shade);

                        p.push_back({ "Ring_Part", s_x + rx, s_y + ry, s_z + rz, s_vx + rvx, s_vy + rvy, s_vz + rvz,0, 0, 0, 1e15, 300000.0, (Uint8)cr, (Uint8)cg, (Uint8)cb, 255, false });
                    }

                    int asteroid_count = 2000;
                    double G = 6.67430e-11;
                    double sun_m = 1.9885e30;

                    for (int i = 0; i < asteroid_count; i++) {
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

                        p.push_back({ "Asteroid", ax, ay, az, avx, avy, avz, 0, 0, 0, 1e16, 400000.0, (Uint8)gray, (Uint8)gray, (Uint8)gray, 255, false });
                    }
                }

            }
        }
        if (name == "J2000") {
            p.push_back({ "Sun",0,0,0,0,0,0,0,0,0,1.9885e30,695700000,255,210,26,255,true });
            p.push_back({ "Mercury",-2.521097e10,4.403542e10,5.190143e09,-4.790189e04,-2.621833e04,2.323756e03,0,0,0,3.3022e23,2440000,194,194,194,255,false });
            p.push_back({ "Venus",-1.075055e11,-3.366520e10,5.706055e09,1.074483e04,-3.519233e04,-1.055433e03,0,0,0,4.8676e24,6051000,255,236,59,255,false });
            p.push_back({ "Earth",-2.650443e10,1.446262e11,-3.866346e04,-2.978611e04,-5.455937e03,3.497548e-01,0,0,0,5.97237e24,6.371e6,28,115,232,255,false });
            p.push_back({ "Moon",-2.687302e10,1.449407e11,8.615146e07,-3.058929e04,-6.237986e03,8.651896e01,0,0,0,7.3477e22,1.7371e6,194,194,194,255,false });
            p.push_back({ "Mars",2.079950e11,-3.143009e09,-5.201802e09,1.164770e03,2.649227e04,5.327515e02,0,0,0,6.4169e23,3.3895e6,245,91,22,255,false });
            p.push_back({ "Jupiter",5.982191e11,4.391086e11,-1.523254e10,-7.901937e03,1.109340e04,1.306729e02,0,0,0,1.8981e27,6.9886e7,255,213,125,255,false });
            p.push_back({ "Saturn",9.586095e11,9.825652e11,-5.522900e10,-7.049246e03,6.727673e03,1.775769e02,0,0,0,5.6846e26,5.8232e7,219,167,61,255,false });
            p.push_back({ "Uranus",2.158746e12,-2.055373e12,-3.559704e10,4.663791e03,4.626879e03,-1.463484e02,0,0,0,8.6810e25,2.5559e7,215,248,250,255,false });
            p.push_back({ "Neptune",2.514009e12,-3.739228e12,1.901638e10,4.466817e03,3.077484e03,-1.669711e02,0,0,0,1.0243e26,2.4764e7,65,105,225,255,false });
        }
        if (name == "milkyway1") {
            string filename = "milkyway_lite.txt";
            const char* base = SDL_GetBasePath();
            ifstream file(string(base) + filename);
            if (!file.is_open()) {
                return;
            }
            string name, x, y, z;
            double vx, vy, vz, ax, ay, az, mass, radius;
            int c1, c2, c3, c4;
            bool stellar;
            while (file >> name >> x >> y >> z >> vx >> vy >> vz >> ax >> ay >> az >> mass >> radius >> c1 >> c2 >> c3 >> c4 >> stellar) {
                p.push_back({ name,stod(x),stod(y),stod(z),vx,vy,vz,ax,ay,az,mass,radius,c1,c2,c3,c4,stellar,false });
            }
            file.close();
        }
        if (name == "milkyway2") {
            string filename = "milkyway_medium.txt";
            const char* base = SDL_GetBasePath();
            ifstream file(string(base) + filename);
            if (!file.is_open()) {
                return;
            }
            string name, x, y, z;
            double vx, vy, vz, ax, ay, az, mass, radius;
            int c1, c2, c3, c4;
            bool stellar;
            while (file >> name >> x >> y >> z >> vx >> vy >> vz >> ax >> ay >> az >> mass >> radius >> c1 >> c2 >> c3 >> c4 >> stellar) {
                p.push_back({ name,stod(x),stod(y),stod(z),vx,vy,vz,ax,ay,az,mass,radius,c1,c2,c3,c4,stellar,false });
            }
            file.close();
        }
        if (name == "milkyway3") {
            string filename = "milkyway_full.txt";
            const char* base = SDL_GetBasePath();
            ifstream file(string(base) + filename);
            if (!file.is_open()) {
                return;
            }
            string name, x, y, z;
            double vx, vy, vz, ax, ay, az, mass, radius;
            int c1, c2, c3, c4;
            bool stellar;
            while (file >> name >> x >> y >> z >> vx >> vy >> vz >> ax >> ay >> az >> mass >> radius >> c1 >> c2 >> c3 >> c4 >> stellar) {
                p.push_back({ name,stod(x),stod(y),stod(z),vx,vy,vz,ax,ay,az,mass,radius,c1,c2,c3,c4,stellar,false });
            }
            file.close();
        }
        return;
    }
    if (n == "shader") {
        shader = !shader;
        rtx = false;   /* CUDA 移植: 两个渲染器互斥(ver3.0 行为) */
    }
    if (n == "autodt") {
        AutoDTime = !AutoDTime;
        second = dt * fps;
    }

    /* ================= CUDA 移植(新增): 光线追踪相关命令 =================
       ver3.0 里有而 v2.2.5 没有的那几条。跳过 ver3.0 的 follow / unfollow
       (本工程没有对应的“跟随视角”玩法) 和 stand / standon(同理)。 */
    if(n=="rtx"){
        rtx = !rtx;
        shader = false;
    }

    /* 反射次数 */
    if(n=="reflection"){
        string ref;
        ssin>>ref;
        if(ref == "")return;
        RTXReflect = stoi(ref);
    }

    /* 单个行星的反射率 */
    if(n=="reflectionvalue"){
        string name;
        ssin>>name;
        string value;
        ssin>>value;
        for(int i = 0;i < p.size();i++){
            if(p[i].name == name){
                p[i].reflection = stoi(value);
                break;
            }
        }
    }

    /* 把 BVH 的格子画出来调试用: debugbvh <最大深度> */
    if(n=="debugbvh"){
        string value;
        ssin>>value;
        if(value == ""){
            debug_bvh = false;
        }else{
            debug_bvh = true;
            debug_bvh_max = stoi(value);
        }
    }

    /* 超采样: ssaa <倍数> 打开, 光 ssaa 关闭 */
    if(n=="ssaa"){
        string value;
        ssin>>value;
        if(value == ""){
            ss.SSAA = false;
        }else{
            ss.SSAA = true;
            ss.SSAA_multi = stoi(value);
        }
    }

    /* 立方体贴图: texture 全量开关 / texture <行星名> */
    if(n=="texture"){
        ss.textureChanged = true;
        string value;
        ssin>>value;
        if(value == "")ss.Texture = !ss.Texture;
        if(ss.Texture && value == ""){
            for(long i = 0;i < p.size();i++){
                if(!filesystem::exists(string(SDL_GetBasePath()) + "Texture/" + p[i].name))continue;
                string eachPath[6] = {
                    string(SDL_GetBasePath()) + "Texture/" + p[i].name + "/px.png",
                    string(SDL_GetBasePath()) + "Texture/" + p[i].name + "/nx.png",
                    string(SDL_GetBasePath()) + "Texture/" + p[i].name + "/py.png",
                    string(SDL_GetBasePath()) + "Texture/" + p[i].name + "/ny.png",
                    string(SDL_GetBasePath()) + "Texture/" + p[i].name + "/pz.png",
                    string(SDL_GetBasePath()) + "Texture/" + p[i].name + "/nz.png",
                };
                for(int t = 0;t < 6;t++){
                    p[i].texture[t] = SS_LoadImage(eachPath[t].c_str());
                }
                /* 六张面片必须全在, 缺一张就当这个行星没有贴图(否则 GPU 侧取像素会空指针崩) */
                if(SS_SurfaceSetOK(p[i].texture, 6)){
                    p[i].hasTexture = true;
                }else{
                    fprintf(stderr, "[texture] %s 的贴图没读全(需要 Texture/%s/{px,nx,py,ny,pz,nz}.png), 按无贴图处理\n",
                            p[i].name.c_str(), p[i].name.c_str());
                    SS_FreeSurfaceSet(p[i].texture, 6);
                    p[i].hasTexture = false;
                }
            }
        }else if(ss.Texture && value != ""){
            for(long i = 0;i < p.size();i++){
                if(filesystem::exists(string(SDL_GetBasePath()) + "Texture/" + value) && p[i].name == value){
                    string eachPath[6] = {
                        string(SDL_GetBasePath()) + "Texture/" + p[i].name + "/px.png",
                        string(SDL_GetBasePath()) + "Texture/" + p[i].name + "/nx.png",
                        string(SDL_GetBasePath()) + "Texture/" + p[i].name + "/py.png",
                        string(SDL_GetBasePath()) + "Texture/" + p[i].name + "/ny.png",
                        string(SDL_GetBasePath()) + "Texture/" + p[i].name + "/pz.png",
                        string(SDL_GetBasePath()) + "Texture/" + p[i].name + "/nz.png",
                    };
                    for(int t = 0;t < 6;t++){
                        p[i].texture[t] = SS_LoadImage(eachPath[t].c_str());
                    }
                    if(SS_SurfaceSetOK(p[i].texture, 6)){
                        p[i].hasTexture = true;
                    }else{
                        fprintf(stderr, "[texture] %s 的贴图没读全(需要 Texture/%s/{px,nx,py,ny,pz,nz}.png), 按无贴图处理\n",
                                p[i].name.c_str(), p[i].name.c_str());
                        SS_FreeSurfaceSet(p[i].texture, 6);
                        p[i].hasTexture = false;
                    }
                }
            }
        }else if(!ss.Texture){
            for(long i = 0;i < p.size();i++){
                p[i].hasTexture = false;
            }
        }
    }

    /* 自转: rotation 总开关 / rotation <行星名> <角速度> */
    if(n=="rotation"){
        string value;
        ssin>>value;
        if(value==""){
            ss.rotationable = !ss.rotationable;
        }else{
            string rv;
            ssin>>rv;
            for(int i = 0;i < p.size();i++){
                if(p[i].name == value){
                    p[i].w_velocity = stod(rv);
                    break;
                }
            }
        }
    }

    /* 自转周期(小时) */
    if(n=="rotationT"){
        string value;
        ssin>>value;
        string rv;
        ssin>>rv;
        for(int i = 0;i < p.size();i++){
            if(p[i].name == value){
                p[i].T = stod(rv);
                break;
            }
        }
    }

    /* 轨道倾角 */
    if(n=="inclination"){
        string value;
        ssin>>value;
        string rv;
        ssin>>rv;
        for(int i = 0;i < p.size();i++){
            if(p[i].name == value){
                p[i].inclination = stod(rv);
                break;
            }
        }
    }

    /* 天空盒背景 */
    if(n=="background"){
        ss.backgroundChanged = true;
        ss.hasbackground = !ss.hasbackground;
        if(ss.hasbackground){
            string eachPath[6] = {
                string(SDL_GetBasePath()) + "Texture/Background/px.png",
                string(SDL_GetBasePath()) + "Texture/Background/nx.png",
                string(SDL_GetBasePath()) + "Texture/Background/py.png",
                string(SDL_GetBasePath()) + "Texture/Background/ny.png",
                string(SDL_GetBasePath()) + "Texture/Background/pz.png",
                string(SDL_GetBasePath()) + "Texture/Background/nz.png",
            };
            for(int t = 0;t < 6;t++){
                ss.background.texture[t] = SS_LoadImage(eachPath[t].c_str());
            }
            if(SS_SurfaceSetOK(ss.background.texture, 6)){
                ss.background.hasTexture = true;
            }else{
                fprintf(stderr, "[background] 背景星空没读到(需要 Texture/Background/{px,nx,py,ny,pz,nz}.png), 保持无背景\n");
                SS_FreeSurfaceSet(ss.background.texture, 6);
                ss.background.hasTexture = false;
                ss.hasbackground = false;   /* 关键: 没图就不能让 RTXRender 去碰 tex->pixels */
            }
        }
    }
    /* ==================================================================== */
    if (isNumber(n) == false)return;

    for (int i = 0;i < stoi(n);i++) {
        string name;
        string x, y, z;
        double vx, vy, vz;
        double ax, ay, az;
        double mass;
        double radius;
        int c1, c2, c3, c4;
        bool stellar;
        ssin >> name >> x >> y >> z >> vx >> vy >> vz >> ax >> ay >> az >> mass >> radius >> c1 >> c2 >> c3 >> c4 >> stellar;
        if (name == "")return;
        if (x == "~")x = to_string(cam.x);
        if (y == "~")y = to_string(cam.y);
        if (z == "~")z = to_string(cam.z);
        p.push_back({ name,stod(x),stod(y),stod(z),vx,vy,vz,ax,ay,az,mass,radius,c1,c2,c3,c4,stellar });
    }
}
void merge(vector<particle>& p, int cur, int comp) {
    string new_name = p[cur].name;
    double new_x = (p[cur].mass * p[cur].x + p[comp].mass * p[comp].x) / (p[cur].mass + p[comp].mass);
    double new_y = (p[cur].mass * p[cur].y + p[comp].mass * p[comp].y) / (p[cur].mass + p[comp].mass);
    double new_z = (p[cur].mass * p[cur].z + p[comp].mass * p[comp].z) / (p[cur].mass + p[comp].mass);
    double new_vx = (p[cur].mass * p[cur].vx + p[comp].mass * p[comp].vx) / (p[cur].mass + p[comp].mass);
    double new_vy = (p[cur].mass * p[cur].vy + p[comp].mass * p[comp].vy) / (p[cur].mass + p[comp].mass);
    double new_vz = (p[cur].mass * p[cur].vz + p[comp].mass * p[comp].vz) / (p[cur].mass + p[comp].mass);
    int c1 = (int)((p[cur].mass * p[cur].c1 + p[comp].mass * p[comp].c1) / (p[cur].mass + p[comp].mass));
    int c2 = (int)((p[cur].mass * p[cur].c2 + p[comp].mass * p[comp].c2) / (p[cur].mass + p[comp].mass));
    int c3 = (int)((p[cur].mass * p[cur].c3 + p[comp].mass * p[comp].c3) / (p[cur].mass + p[comp].mass));
#pragma omp critical
    {
        p.push_back({ new_name,new_x,new_y,new_z,new_vx,new_vy,new_vz,0,0,0,(p[cur].mass + p[comp].mass),pow(pow(p[cur].radius,3) + pow(p[comp].radius,3),1.0 / 3.0),c1,c2,c3,255,false });
    }
}
double Screen(particle& p, camera& cam) {
    double dx = p.x - cam.x;
    double dy = p.y - cam.y;
    double dz = p.z - cam.z;

    double a = cam.yaw * (M_PI / 180), b = cam.pitch * (M_PI / 180);
    double fov = cam.fov * (M_PI / 180);

    double x = dx * cos(a) - dz * sin(a);
    double z1 = dx * sin(a) + dz * cos(a);
    double y = dy * cos(b) - z1 * sin(b);
    double z = dy * sin(b) + z1 * cos(b);

    double s_x = (x / (z * tan(0.5 * fov)) + 1) * width / 2;
    double s_y = (1 - y / (z * tan(0.5 * fov))) * height / 2;

    res.sx = s_x;
    res.sy = s_y;
    res.sz = z;
    return -1;
}

void drawCircle3(SDL_Renderer* ren, projection p, double fov_deg, bool Target) {
    double fov_rad = fov_deg * (M_PI / 180.0);
    double focal_length = height / (2.0 * tan(fov_rad * 0.5));
    int r = (int)(p.r * focal_length / p.sz);
    if (r <= 1)r = 1;
    if (r > 2000)r = 2000;
    for (int j = -r;j <= r;j++) {
        int len = (int)sqrt(r * r - j * j);
        SDL_RenderLine(ren, p.sx - len, p.sy + j, p.sx + len, p.sy + j);
    }
    if (Target) {
        int t_cir = r + 20;
        int t_cir_l = r + 19;
        for (int i = -t_cir;i <= t_cir;i++) {
            for (int j = -t_cir;j <= t_cir;j++) {
                if (i * i + j * j<t_cir * t_cir && i * i + j * j>t_cir_l * t_cir_l) {
                    SDL_SetRenderDrawColor(ren, 0, 255, 0, 255);
                    SDL_RenderPoint(ren, p.sx + i, p.sy + j);
                }
            }
        }
    }
}
int findFocus(vector<particle>& planet, camera& cam) {
    double min_dist = 50.0;
    int best = -1;
    for (int i = 0;i < planet.size();i++) {
        Screen(planet[i], cam);
        if (res.sz > 1e-3 && res.sz < 200) {
            double dist_to_center = sqrt(pow(res.sx - width / 2, 2) + pow(res.sy - height / 2, 2));
            if (dist_to_center < min_dist) {
                min_dist = dist_to_center;
                best = i;
            }
        }
    }
    return best;
}
void disintegrate4(vector<particle>& p, vector<particle>& roc, double rocheR, double r, int cur, double dx, double dy, double dz, double K) {
    if (p[cur].radius <= disintegrate_num)return;
    int genp = (int)K * (rocheR / r) * dt;
    // double expectgen=K*(rocheR-r)*dt;
    // int genp=(int)(-r+400);
    // int genp=(int)expectgen;
    //Original
    double p0x = p[cur].mass * p[cur].vx;
    double p0y = p[cur].mass * p[cur].vy;
    double p0z = p[cur].mass * p[cur].vz;
    double m0 = p[cur].mass;
    //Pieces
    double frag_r = disintegrate_num;
    double frag_m = p[cur].mass * pow(frag_r / p[cur].radius, 3);

    double pfragx = 0, pfragy = 0, pfragz = 0;
    for (int i = 0;i < genp;i++) {
        double uni = sqrt(dx * dx + dy * dy + dz * dz);
        double udx = dx / uni;
        double udy = dy / uni;
        double udz = dz / uni;

        double ux = 2.0 * rand() / RAND_MAX - 1.0; //[-1.0,1.0]
        double uy = 2.0 * rand() / RAND_MAX - 1.0; //[-1.0,1.0]
        double uz = 2.0 * rand() / RAND_MAX - 1.0; //[-1.0,1.0]
        double len = sqrt(ux * ux + uy * uy + uz * uz);
        if (len < 1e-6)continue; //Vector 0
        ux /= len; uy /= len; uz /= len;
        ux += udx; uy += udy; uz += udz;

        double newx = p[cur].x + p[cur].radius * ux;
        double newy = p[cur].y + p[cur].radius * uy;
        double newz = p[cur].z + p[cur].radius * uz;

        double esc = 0.3 * sqrt(G * p[cur].mass / (p[cur].radius + ess));
        double vx = p[cur].vx + ux * esc * (0.8 + 0.4 * rand() / RAND_MAX);
        double vy = p[cur].vy + uy * esc * (0.8 + 0.4 * rand() / RAND_MAX);
        double vz = p[cur].vz + uz * esc * (0.8 + 0.4 * rand() / RAND_MAX);

        pfragx += vx * frag_m; pfragy += vy * frag_m; pfragy += vz * frag_m;
        string name = "G";
#pragma omp critical
        {
            roc.push_back({ p[cur].name + name,newx,newy,newz,vx,vy,vz,0,0,0,frag_m,frag_r,p[cur].c1,p[cur].c2,p[cur].c3,p[cur].c4,false });
        }
    }
    double mt = p[cur].mass - frag_m * genp;
    double rt = p[cur].radius * pow(mt / m0, 1.0 / 3.0) * 0.99;
    // double rt=p[cur].radius-genp*frag_r;
    p[cur].vx = (p0x - pfragx) / mt;
    p[cur].vy = (p0y - pfragy) / mt;
    p[cur].vz = (p0z - pfragz) / mt;
    p[cur].mass = mt;
    p[cur].radius = rt;
}
string toScientific(double val, int precision) {
    stringstream ss;
    ss << scientific << setprecision(precision) << val;
    return ss.str();
}
bool isNumber(string& str) {
    return !str.empty() && all_of(str.begin(), str.end(), [](unsigned char c) {
        return isdigit(c) || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+';
        });
}
void YourSpaceShip(camera& cam, SDL_Event& event, SDL_Window* win, SDL_Renderer* ren, bool& pause, int iniline) {
    /*Initialize*/
    pause = true;
    const int command_count = 7;
    string command[command_count + 1] = { "Name ","Mass(kg, without fuel) ","Engine Count ","Single Engine Thrust(N) ","Single Engine Consumption(kg/s) ","Fuel(kg) ","Velocity(m/s) ","Finish" };
    static int cur_command = 1;
    static string commandbuf = "";
    static string currentbuf = ">";
    static bool first_input = true;
    if (first_input) {
        first_input = !first_input;
        myship.vx = 0; myship.vy = 0; myship.vz = 0;
        SDL_StartTextInput(win);
    }

    //Render
    SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
    int spaceline = 200;
    command[cur_command - 1] = "->" + command[cur_command - 1];
    float velocity = sqrt(myship.vx * myship.vx + myship.vy * myship.vy + myship.vz * myship.vz);
    SDL_RenderDebugText(ren, 10, spaceline + 12 * 0, (command[0] + myship.name).c_str());
    SDL_RenderDebugText(ren, 10, spaceline + 12 * 1, (command[1] + to_string(myship.mass)).c_str());
    SDL_RenderDebugText(ren, 10, spaceline + 12 * 2, (command[2] + to_string(myship.engine_count)).c_str());
    SDL_RenderDebugText(ren, 10, spaceline + 12 * 3, (command[3] + to_string(myship.single_engine_thrust)).c_str());
    SDL_RenderDebugText(ren, 10, spaceline + 12 * 4, (command[4] + to_string(myship.single_engine_consumption)).c_str());
    SDL_RenderDebugText(ren, 10, spaceline + 12 * 5, (command[5] + to_string(myship.fuel)).c_str());
    SDL_RenderDebugText(ren, 10, spaceline + 12 * 6, (command[6] + to_string(velocity)).c_str());
    SDL_RenderDebugText(ren, 10, spaceline + 12 * 7, command[7].c_str());

    if (event.type == SDL_EVENT_TEXT_INPUT) {
        currentbuf += event.text.text;
        commandbuf += event.text.text;
    }
    else if (event.type == SDL_EVENT_KEY_DOWN) {
        if (event.key.key == SDLK_BACKSPACE && currentbuf.size() > 1) {
            currentbuf.pop_back();
            commandbuf.pop_back();
        }
        else if (event.key.key == SDLK_RETURN) {
            if (cur_command == (command_count + 1)) {
                myship.initial = true;
                myship.mass += myship.fuel;
                myship.x = cam.x; myship.y = cam.y; myship.z = cam.z;
                myship.yaw = (int)cam.yaw;
                myship.pitch = (int)cam.pitch;
                SDL_StopTextInput(win);
                pause = false;
                first_input = true;
                return;
            }
            switch (cur_command) {
            case 1: {
                myship.name = commandbuf;
                break;
            }case 2: {
                if (!isNumber(commandbuf))return;
                myship.mass = stod(commandbuf);
                break;
            }case 3: {
                if (!isNumber(commandbuf))return;
                myship.engine_count = stod(commandbuf);
                break;
            }case 4: {
                if (!isNumber(commandbuf))return;
                myship.single_engine_thrust = stod(commandbuf);
                break;
            }case 5: {
                if (!isNumber(commandbuf))return;
                myship.single_engine_consumption = stod(commandbuf);
                break;
            }case 6: {
                if (!isNumber(commandbuf))return;
                myship.fuel = stod(commandbuf);
                break;
            }case 7: {
                if (!isNumber(commandbuf))return;
                double v = stod(commandbuf);
                double rady = cam.yaw * (M_PI / 180.0);
                double radp = cam.pitch * (M_PI / 180.0);
                myship.vx = sin(rady) * cos(radp) * v;
                myship.vy = sin(radp) * v;
                myship.vz = cos(rady) * cos(radp) * v;
                break;
            }
            }
            commandbuf = "";
            currentbuf = ">";
            cur_command++;
        }
        else if (event.key.key == SDLK_UP && cur_command > 1)cur_command--;
        else if (event.key.key == SDLK_DOWN && cur_command < command_count + 1)cur_command++;
    }

    SDL_RenderDebugText(ren, 50, iniline, currentbuf.c_str());
    /*Initialize*/
}
float max_velocity = 0;
float max_fuel = 0;
float max_distance = 0;
float max_throttle = 1.0;
void InMyShip(camera& cam, SDL_Event& event, SDL_Window* win, SDL_Renderer* ren, int iniline, bool& targetisTyping, vector<particle> p, bool pause, bool debug, TTF_Font* font) {
    cam.x = myship.x; cam.y = myship.y; cam.z = myship.z;
    /*Parameter*/
    static bool AutoOrbit = false;
    static bool DrawTrajectory = false;
    if (debug == true) {
        const int para_count = 9;
        int spaceline = 200;
        string show[para_count] = { "","Velocity(m/s):","Acceleration(m/s^2):","Throttle:","Mass/Fuel(kg):","Yaw/Pitch:","Target:","AutoOrbit:","Trajectory" };
        float velocity = sqrt(myship.vx * myship.vx + myship.vy * myship.vy + myship.vz * myship.vz);
        float acceleration = sqrt(myship.ax * myship.ax + myship.ay * myship.ay + myship.az * myship.az);
        float total_pushforce = myship.engine_count * myship.single_engine_thrust * myship.throttle;
        if (myship.fuel <= 0)total_pushforce = 0;
        SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
        SDL_RenderDebugText(ren, 10, spaceline + 12 * 0, myship.name.c_str());
        SDL_RenderDebugText(ren, 10, spaceline + 12 * 1, (show[1] + to_string(velocity)).c_str());
        SDL_RenderDebugText(ren, 10, spaceline + 12 * 2, (show[2] + to_string(acceleration)).c_str());
        SDL_RenderDebugText(ren, 10, spaceline + 12 * 3, (show[3] + to_string(myship.throttle) + '/' + to_string(total_pushforce) + 'N').c_str());
        SDL_RenderDebugText(ren, 10, spaceline + 12 * 4, (show[4] + to_string(myship.mass) + '/' + to_string(myship.fuel)).c_str());
        SDL_RenderDebugText(ren, 10, spaceline + 12 * 6, (show[5] + to_string(myship.yaw) + '/' + to_string(myship.pitch)).c_str());
        SDL_RenderDebugText(ren, 10, spaceline + 12 * 7, (show[6] + myship.target_name + '/' + to_string(myship.distance)).c_str());
        SDL_RenderDebugText(ren, 10, spaceline + 12 * 8, (show[7] + to_string(AutoOrbit)).c_str());
        SDL_RenderDebugText(ren, 10, spaceline + 12 * 9, (show[8] + to_string(DrawTrajectory)).c_str());
    }
    //DashBoard
    float dash_r = (50.0f / 900.0f) * height;
    float dash_minus_r = (20.0f / 900.0f) * height;
    float dash_x = width * 0.6;
    float dash_y = height * 0.75;

    float v = (float)sqrt(myship.vx * myship.vx + myship.vy * myship.vy + myship.vz * myship.vz);
    max_velocity = (max_velocity >= v) ? max_velocity : v;
    float fuel = (float)myship.fuel;
    max_fuel = (max_fuel >= fuel) ? max_fuel : fuel;
    float distance = (float)myship.distance;
    max_distance = (max_distance >= distance) ? max_distance : distance;
    float throttle = (float)myship.throttle;
    max_throttle = 1.0;

    SDL_SetRenderDrawColor(ren, 90, 90, 90, 255);
    for (int y = -dash_r;y <= dash_r;y++) {
        for (int x = -dash_r;x <= dash_r;x++) {
            if (y * y + x * x <= dash_r * dash_r && y * y + x * x >= dash_r * dash_r * 0.95 && y <= (dash_r - dash_minus_r)) {
                //Velocity
                SDL_RenderPoint(ren, dash_x + x + 0 * 3 * dash_r, dash_y + y + 0);
                //Fuel
                SDL_RenderPoint(ren, dash_x + x + 1 * 3 * dash_r, dash_y + y + 0);
                //Distance
                SDL_RenderPoint(ren, dash_x + x + 2 * 3 * dash_r, dash_y + y + 0);
                //Throttle
                SDL_RenderPoint(ren, dash_x + x + 3 * 3 * dash_r, dash_y + y + 0);
            }
        }
    }

    float x0 = -sqrt(dash_r * dash_r - (dash_r - dash_minus_r) * (dash_r - dash_minus_r));
    float y0 = -(dash_r - dash_minus_r);
    float a0 = atan2(y0, x0);

    float v_theta = (v / max_velocity) * (2 * M_PI - 2 * acos((dash_r - dash_minus_r) / dash_r));
    float fuel_theta = (fuel / max_fuel) * (2 * M_PI - 2 * acos((dash_r - dash_minus_r) / dash_r));
    float distance_theta = (1 - distance / max_distance) * (2 * M_PI - 2 * acos((dash_r - dash_minus_r) / dash_r));
    float thr_theta = (sqrt(throttle * throttle) / max_throttle) * (2 * M_PI - 2 * acos((dash_r - dash_minus_r) / dash_r));
    SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
    for (int y = -dash_r;y <= dash_r;y++) {
        for (int x = -dash_r;x <= dash_r;x++) {
            if (x * x + y * y <= dash_r * dash_r && x * x + y * y >= dash_r * dash_r * 0.85) {
                //Velocity
                float at = atan2(y, x);
                float diff = a0 - at;
                if (at > a0)diff += 2 * M_PI;
                if (diff <= v_theta)SDL_RenderPoint(ren, dash_x + x + 0 * 3 * dash_r, dash_y - y + 0);
                if (diff <= fuel_theta)SDL_RenderPoint(ren, dash_x + x + 1 * 3 * dash_r, dash_y - y + 0);
                if (diff <= distance_theta)SDL_RenderPoint(ren, dash_x + x + 2 * 3 * dash_r, dash_y - y + 0);
                if (diff <= thr_theta)SDL_RenderPoint(ren, dash_x + x + 3 * 3 * dash_r, dash_y - y + 0);
            }
        }
    }
    //TTF Initial
    SDL_Color white = { 255,255,255,255 };

    char speed_buf[16], speed_label[] = "SPEED";
    char fuel_buf[16], fuel_label[] = "FUEL";
    char dis_buf[8], dis_label[] = "DISTANCE";
    char thr_buf[8], thr_label[] = "THROTTLE";
    SDL_snprintf(speed_buf, sizeof(speed_buf), "%d", (int)v);
    SDL_snprintf(fuel_buf, sizeof(fuel_buf), "%d", (int)fuel);
    SDL_snprintf(dis_buf, sizeof(dis_buf), "%d", (int)((1 - distance / max_distance) * 100));
    SDL_snprintf(thr_buf, sizeof(thr_buf), "%d", (int)((throttle / max_throttle) * 100));

    DrawTTF(ren, font, speed_buf, dash_x, dash_y, white, dash_r * 1.7, 1, 1);
    DrawTTF(ren, font, speed_label, dash_x, dash_y + dash_r * 0.8, white, 100, 0.6, 0.46);
    DrawTTF(ren, font, fuel_buf, dash_x + 1 * 3 * dash_r, dash_y, white, dash_r * 1.5, 1, 1);
    DrawTTF(ren, font, fuel_label, dash_x + 1 * 3 * dash_r, dash_y + dash_r * 0.8, white, 100, 0.6, 0.46);
    DrawTTF(ren, font, dis_buf, dash_x + 2 * 3 * dash_r, dash_y, white, dash_r * 1.8, 1, 1);
    DrawTTF(ren, font, dis_label, dash_x + 2 * 3 * dash_r, dash_y + dash_r * 0.8, white, 100, 0.6, 0.46);
    DrawTTF(ren, font, thr_buf, dash_x + 3 * 3 * dash_r, dash_y, white, dash_r * 1.8, 1, 1);
    DrawTTF(ren, font, thr_label, dash_x + 3 * 3 * dash_r, dash_y + dash_r * 0.8, white, 100, 0.6, 0.46);

    /*Parameter*/
    /*Header*/
    particle heading;
    double rady = myship.yaw * (M_PI / 180.0);
    double radp = myship.pitch * (M_PI / 180.0);
    heading.x = myship.x + sin(rady) * cos(radp) * 1000;
    heading.y = myship.y + sin(radp) * 1000;
    heading.z = myship.z + cos(rady) * cos(radp) * 1000;
    Screen(heading, cam);
    if (res.sz > 0) {
        SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
        SDL_RenderLine(ren, res.sx - 5, res.sy, res.sx + 5, res.sy);
        SDL_RenderLine(ren, res.sx, res.sy - 5, res.sx, res.sy + 5);
    }
    /*Header*/
    /*Controller*/
    const bool* keystate = SDL_GetKeyboardState(NULL);
    //Straight
    if (targetisTyping == false && pause == false) {
        static bool TOGA = false;

        if (TOGA == true && !keystate[SDL_SCANCODE_I] && !keystate[SDL_SCANCODE_K])TOGA = false;
        if (keystate[SDL_SCANCODE_I] && myship.throttle < 0.99 && TOGA == false)myship.throttle += 0.01;
        if (keystate[SDL_SCANCODE_K] && myship.throttle > -0.99 && TOGA == false)myship.throttle -= 0.01;
        if (abs(myship.throttle) < 0.001 && TOGA == false && (keystate[SDL_SCANCODE_I] || keystate[SDL_SCANCODE_K])) {
            myship.throttle = 0.0;
            TOGA = true;
        }
        double used = sqrt(myship.engine_count * myship.single_engine_consumption * myship.throttle * dt * myship.engine_count * myship.single_engine_consumption * myship.throttle * dt);
        if ((myship.fuel - used) > 0.0) {
            double F = myship.engine_count * myship.single_engine_thrust * myship.throttle;
            double a = F / myship.mass;
            myship.ax += sin(rady) * cos(radp) * a;
            myship.ay += sin(radp) * a;
            myship.az += cos(rady) * cos(radp) * a;
            myship.fuel -= used;
            myship.mass -= used;
        }
        //Heading
        double turning_speed = 1.5;
        if (keystate[SDL_SCANCODE_S] && myship.pitch > -89.0)myship.pitch -= turning_speed;
        if (keystate[SDL_SCANCODE_W] && myship.pitch < 89.0)myship.pitch += turning_speed;
        if (keystate[SDL_SCANCODE_A])myship.yaw -= turning_speed;
        if (keystate[SDL_SCANCODE_D])myship.yaw += turning_speed;
        if (myship.yaw >= 360)myship.yaw -= 360;
        if (myship.yaw < 0)myship.yaw += 360;
    }
    /*Controller*/
    /*Function*/
    static bool first_lockedZ = false;
    static bool first_lockedE = false;
    static bool first_lockedC = false;
    static bool locked = false;
    static string commandbuf = "";
    static string currentbuf = "Target:";
    if (keystate[SDL_SCANCODE_Q] && targetisTyping == false) {
        targetisTyping = true;
        SDL_StartTextInput(win);
    }
    if (first_lockedE == true && !keystate[SDL_SCANCODE_E])first_lockedE = false;
    if (keystate[SDL_SCANCODE_E] && first_lockedE == false) {
        if (myship.target == true) {
            locked = !locked;
            first_lockedE = true;
        }
        else {
            locked = false;
        }
    }
    if (first_lockedC == true && !keystate[SDL_SCANCODE_C])first_lockedC = false;
    if (keystate[SDL_SCANCODE_C] && first_lockedC == false) {
        if (myship.target == true) {
            AutoOrbit = !AutoOrbit;
            first_lockedC = true;
        }
        else {
            AutoOrbit = false;
        }
    }
    if (first_lockedZ == true && !keystate[SDL_SCANCODE_Z])first_lockedZ = false;
    if (keystate[SDL_SCANCODE_Z] && first_lockedZ == false) {
        DrawTrajectory = !DrawTrajectory;
        first_lockedZ = true;
    }

    if (DrawTrajectory) {
        static double proj = 100.0;
        if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_U && proj > 1.0)proj /= 10.0;
        if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_O && proj < 1000000.0)proj *= 10.0;
        DrawTrajectorySystem(ren, proj, p, cam);
    }
    if (AutoOrbit) {
        AutoOrbitSystem(first_lockedC, p);
    }
    if (locked == true) {
        for (int i = 0;i < p.size();i++) {
            if (p[i].name == myship.target_name) {
                double dx = p[i].x - myship.x;
                double dy = p[i].y - myship.y;
                double dz = p[i].z - myship.z;
                if (myship.distance < 0.1)break;
                double targetYaw = atan2(dx, dz) * (180.0 / M_PI);
                double targetPitch = asin(dy / myship.distance) * (180.0 / M_PI);
                myship.yaw = targetYaw;
                myship.pitch = targetPitch;
                break;
            }
        }
    }
    if (targetisTyping == true) {
        if (event.type == SDL_EVENT_TEXT_INPUT) {
            commandbuf += event.text.text;
            currentbuf += event.text.text;
        }if (event.type == SDL_EVENT_KEY_DOWN) {
            if (event.key.key == SDLK_BACKSPACE && currentbuf.size() > 7) {
                commandbuf.pop_back();
                currentbuf.pop_back();
            }
            if (event.key.key == SDLK_RETURN) {
                for (int i = 0;i < p.size();i++) {
                    if (p[i].name == commandbuf) {
                        myship.target = true;
                        myship.target_name = commandbuf;
                        commandbuf = "";
                        currentbuf = "Target:";
                        break;
                    }
                    else {
                        myship.target = false;
                        myship.target_name = "";
                    }
                }
                targetisTyping = false;
                SDL_StopTextInput(win);
            }
            if (event.key.key == SDLK_ESCAPE) {
                targetisTyping = false;
                SDL_StopTextInput(win);
            }
        }
        SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
        SDL_RenderDebugText(ren, 30, height * 0.7, currentbuf.c_str());
    }
    /*Function*/
}
void AutoOrbitSystem(bool first_lockedC, vector<particle> p) {
    static double shortest = 0;
    double M = 0;
    double pvx; double pvy; double pvz;
    if (first_lockedC || shortest > myship.distance)shortest = myship.distance;
    if (shortest <= myship.distance) {
        int cur = 0;
        for (int i = 0;i < p.size();i++) {
            if (myship.target_name == p[i].name) {
                cur = i;
                break;
            }
        }
        M = p[cur].mass;
        pvx = p[cur].vx; pvy = p[cur].vy; pvz = p[cur].vz;
        double OrbitV = sqrt(G * M / myship.distance);
        //Distance-Velocity Heading

        double dx = p[cur].x - myship.x;
        double dy = p[cur].y - myship.y;
        double dz = p[cur].z - myship.z;
        double dist = dx * dx + dy * dy + dz * dz;
        //Relatively Velocity
        double rvx = myship.vx - pvx;
        double rvy = myship.vy - pvy;
        double rvz = myship.vz - pvz;
        double k = (rvx * dx + rvy * dy + rvz * dz) / (dist + ess);

        //New Vector
        double a = rvx - k * dx;
        double b = rvy - k * dy;
        double c = rvz - k * dz;
        double tmag = sqrt(a * a + b * b + c * c + ess);

        //Each Heading Velocity
        double tagvx = (a / tmag) * OrbitV + pvx;
        double tagvy = (b / tmag) * OrbitV + pvy;
        double tagvz = (c / tmag) * OrbitV + pvz;

        double dvx = tagvx - myship.vx;
        double dvy = tagvy - myship.vy;
        double dvz = tagvz - myship.vz;
        double dv = sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
        //Checking
        if (dv > 1.0) {
            double tag_yaw = atan2(dvx, dvz) * (180.0 / M_PI);
            if (tag_yaw < 0)tag_yaw += 360;
            double tag_pitch = asin(dvy / (dv + ess)) * (180.0 / M_PI);

            double turning_speed = 1.5;

            double y_diff = tag_yaw - myship.yaw;
            if (y_diff > 180)y_diff -= 360;
            if (y_diff < -180)y_diff += 360;
            if (abs(y_diff) > 1.0) {
                myship.yaw += (y_diff > 0) ? turning_speed : -turning_speed;
            }
            double p_diff = tag_pitch - myship.pitch;
            if (abs(p_diff) > 1.0) {
                myship.pitch += (p_diff > 0) ? turning_speed : -turning_speed;
            }
            double sensi = 100.0;
            if (abs(y_diff) < 10.0 && abs(p_diff) < 10.0) {
                myship.throttle = (dv > sensi) ? 1.0 : (dv / sensi);
            }
            else {
                myship.throttle = 0.0;
            }
        }
        else {
            myship.throttle = 0.0;
        }
    }
}
void DrawTrajectorySystem(SDL_Renderer* ren, double proj, vector<particle> p, camera& cam) {
    const int steps = 100;
    double v = sqrt(myship.vx * myship.vx + myship.vy * myship.vy + myship.vz * myship.vz);

    double dt_pre = proj / v;

    double cvx = myship.vx, cvy = myship.vy, cvz = myship.vz;
    double cx = myship.x, cy = myship.y, cz = myship.z;
    SDL_SetRenderDrawColor(ren, 0, 255, 0, 255);
    for (int i = 0;i < steps;i++) {
        double ax = 0, ay = 0, az = 0;
        for (int j = 0;j < p.size();j++) {
            double dx = p[j].x - cx;
            double dy = p[j].y - cy - 100;
            double dz = p[j].z - cz;
            double r = sqrt(dx * dx + dy * dy + dz * dz);
            double F = (G * p[j].mass) / (r * r);

            ax += F * dx / r;
            ay += F * dy / r;
            az += F * dz / r;
        }
        cvx += ax * dt_pre; cvy += ay * dt_pre; cvz += az * dt_pre;
        cx += cvx * dt_pre; cy += cvy * dt_pre; cz += cvz * dt_pre;

        particle temp = { "",cx,cy - 500,cz,0,0,0,0,0,0,0,0.01,0,255,0,255,false };
        Screen(temp, cam);
        projection pro = { "",0,0,0,res.sx,res.sy,res.sz,0,255,0,150,0.01 };
        if (res.sz > 0.1)drawCircle3(ren, pro, cam.fov, false);
    }
}
void DrawCircleShader(SDL_Renderer* ren, projection p, double fov_deg, bool Target, double lx, double ly, double lz, double stellar_intensity) {
    static double rotation_time = 0;
    rotation_time += dt * 0.0005;

    double fov_rad = fov_deg * (M_PI / 180.0);
    double focal_length = height / (2.0 * tan(fov_rad * 0.5));
    int r = (int)(p.r * focal_length / (p.sz + ess));
    if (r <= 1)r = 1;
    if (r > 2000)r = 2000;
    int start_i = (p.sx - r >= 0) ? -r : -p.sx;
    int end_i = (p.sx + r <= width) ? r : width - p.sx;
    int start_j = (p.sy - r >= 0) ? -r : -p.sy;
    int end_j = (p.sy + r <= height) ? r : height - p.sy;

#pragma omp parallel for
    for (int i = start_i;i <= end_i;i++) {
        for (int j = start_j;j <= end_j;j++) {
            if (i * i + j * j <= r * r) {
                double nx = (double)i / r;
                double ny = (double)j / r;
                double nz = sqrt(1.0 - (nx * nx + ny * ny));

                double intensity = nx * lx + ny * ly + nz * lz;
                intensity *= stellar_intensity;
                if (intensity < 0)intensity = abs(intensity * 0.03);

                SDL_SetRenderDrawColor(ren, intensity * p.c1, intensity * p.c2, intensity * p.c3, p.c4);
                SDL_RenderPoint(ren, p.sx + i, p.sy + j);
            }
        }
    }

    if (Target) {
        int t_cir = r + 20;
        int t_cir_l = r + 19;
        for (int i = -t_cir;i <= t_cir;i++) {
            for (int j = -t_cir;j <= t_cir;j++) {
                if (i * i + j * j<t_cir * t_cir && i * i + j * j>t_cir_l * t_cir_l) {
                    SDL_SetRenderDrawColor(ren, 0, 255, 0, 255);
                    SDL_RenderPoint(ren, p.sx + i, p.sy + j);
                }
            }
        }
    }
}
void DrawTTF(SDL_Renderer* ren, TTF_Font* font, const char* text, float centerX, float centerY, SDL_Color col, float maxWidth, float x_scale, float y_scale) {
    if (!font || !text || strlen(text) == 0)return;
    //Turning to image
    SDL_Surface* surf = TTF_RenderText_Blended(font, text, 0, col);
    if (!surf)return;
    //Send to GPU
    SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, surf);
    if (!tex) {
        SDL_Log("SDL_ttf Error: %s", SDL_GetError());
        SDL_DestroySurface(surf);
        return;
    }
    //Calculate scale
    float inscale = 1.0f;
    if (surf->w > maxWidth) {
        inscale = maxWidth / surf->w;
    }
    //Position calculate
    SDL_FRect dstRect;
    dstRect.w = (float)surf->w * inscale * x_scale;
    dstRect.h = (float)surf->h * inscale * y_scale;
    dstRect.x = centerX - (dstRect.w / 2.0f);
    dstRect.y = centerY - (dstRect.h / 2.0f);
    //Draw
    SDL_RenderTexture(ren, tex, NULL, &dstRect);

    SDL_DestroySurface(surf);
    SDL_DestroyTexture(tex);
}
void DrawStellarShader(SDL_Renderer* ren, projection p, double fov_deg) {

    double fov_rad = fov_deg * (M_PI / 180.0);
    double focal_length = height / (2.0 * tan(fov_rad * 0.5));

    double r = p.r * focal_length / (p.sz + 1e-6);
    if (r <= 0 || p.sz <= 0) return;

    int glow_limit = r * 8; // 稍微降一点，性能更稳

    int start_i = max(-glow_limit, (int)-p.sx);
    int end_i = min(glow_limit, (int)(width - p.sx));
    int start_j = max(-glow_limit, (int)-p.sy);
    int end_j = min(glow_limit, (int)(height - p.sy));

    double inv_r2 = 1.0 / (r * r);
    double inv_glow = 1.0 / glow_limit;

    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_ADD);

    for (int j = start_j; j <= end_j; j++) {
        int jj = j * j;
        for (int i = start_i; i <= end_i; i++) {
            int dist_sq = i * i + jj;
            // 圆形裁剪（减少约 20~30% 像素）
            if (dist_sq > glow_limit * glow_limit) continue;
            double norm_dist_sq = dist_sq * inv_r2;
            double intensity;

            if (norm_dist_sq < 1.0) {
                // 内核（避免 sqrt）
                intensity = 1.5 - 0.5 * norm_dist_sq;
            }
            else {
                // 外光晕（完全平方版本）
                intensity = 1.0 / (norm_dist_sq * 0.8 + 0.2);

                // 软边（不用 sqrt）
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

/* ==========================================================================
   CUDA 移植(新增代码): 下面这一整块来自 ver3.0 (Metal 版 main.mm) 的光线追踪子系统,
   逐行搬运, 只做了 Metal→CUDA 的接口替换(gpuBufferNew / gpuSetBuffer /
   gpuDispatchCorrectVector / gpuDispatchRTXPath / gpuCommit) 与 long→long long。
   RTXRender        = CPU 参考实现(没有 N 卡 / isMetal 关时走它)
   RTXRenderMetal   = CUDA 实现(名字沿用原来的 Metal, 便于和原文件逐行对齐)
   ========================================================================== */
void RTXInitial(int rtxheight, int rtxwidth){
    RTXOriginVector.assign(rtxheight, vector<RTXV>(rtxwidth, {0,0,0}));
    RTXVector.assign(rtxheight, vector<RTXV>(rtxwidth, {0,0,0}));
    RTXVectorLength.assign(rtxheight, vector<double>(rtxwidth, 0));
    double OOm = rtxheight/(2*tan(cam.fov * M_PI / 360.0));
    for(int i = rtxheight/2;i > -rtxheight/2;i--){
        for(int j = -rtxwidth/2;j < rtxwidth/2;j++){
            RTXOriginVector[rtxheight/2-i][j+rtxwidth/2].x = j;
            RTXOriginVector[rtxheight/2-i][j+rtxwidth/2].y = i;
            RTXOriginVector[rtxheight/2-i][j+rtxwidth/2].z = OOm;
            RTXVectorLength[rtxheight/2-i][j+rtxwidth/2] = sqrt(j * j + i * i + OOm * OOm);
        }
    }
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
double VectorDotMul(RTXV A, RTXV B){
    return A.x * B.x + A.y * B.y + A.z * B.z;
}
double VectorLength(RTXV A){
    return sqrt(A.x * A.x + A.y * A.y + A.z * A.z);
}
RTXV VectorUnit(RTXV A){
    double abs = VectorLength(A);
    RTXV Result = {
        A.x / abs, A.y / abs, A.z / abs
    };
    return Result;
}
RTXV VectorMul(RTXV A, double k){
    return {A.x * k, A.y * k, A.z * k};
}
RTXV VectorPlus(RTXV A, RTXV B){
    return {A.x + B.x, A.y + B.y, A.z + B.z};
}
RTXV RTXVectorCalcM(RTXV v, RTXV O, RTXV P, double radius, double VLength){
    /*
     Formula:
     OE = OP * v / |v|
     OM = v/|v| * (OE - sqrt(r^2 - (OP^2 - OE^2)))
     */
    RTXV M;
    RTXV vUnit = VectorUnit(v);
    RTXV OP = {P.x-O.x, P.y-O.y, P.z-O.z};
    double OE = VectorDotMul(OP, vUnit);
    RTXV EP = VectorPlus(OP, VectorMul(vUnit, -OE));
    double delta = radius * radius - VectorDotMul(EP, EP);
    if(delta < 0){
        return {0,0,0,false};
    }
    
    RTXV OM = VectorMul(vUnit, OE - sqrt(delta));
    M = VectorPlus(OM, O);
    M.OM = OE - sqrt(delta);
    return M;
}
void RTXBVHAABB(const vector<particle> &parti, BVHNode &currentBVH, RTXV &C1, RTXV &C2, RTXV *Delta = {}){
    C1 = {
        parti[currentBVH.indices[0]].x - parti[currentBVH.indices[0]].radius,
        parti[currentBVH.indices[0]].y - parti[currentBVH.indices[0]].radius,
        parti[currentBVH.indices[0]].z - parti[currentBVH.indices[0]].radius
    };
    C2 = {
        parti[currentBVH.indices[0]].x + parti[currentBVH.indices[0]].radius,
        parti[currentBVH.indices[0]].y + parti[currentBVH.indices[0]].radius,
        parti[currentBVH.indices[0]].z + parti[currentBVH.indices[0]].radius
    };
    for(int i = 1;i < currentBVH.indices.size();i++){
        long p = currentBVH.indices[i];
        C1.x = (C1.x < parti[p].x - parti[p].radius)? C1.x : parti[p].x - parti[p].radius;
        C1.y = (C1.y < parti[p].y - parti[p].radius)? C1.y : parti[p].y - parti[p].radius;
        C1.z = (C1.z < parti[p].z - parti[p].radius)? C1.z : parti[p].z - parti[p].radius;
        C2.x = (C2.x > parti[p].x + parti[p].radius)? C2.x : parti[p].x + parti[p].radius;
        C2.y = (C2.y > parti[p].y + parti[p].radius)? C2.y : parti[p].y + parti[p].radius;
        C2.z = (C2.z > parti[p].z + parti[p].radius)? C2.z : parti[p].z + parti[p].radius;
    }
    if(Delta){
        *Delta = {
            C2.x - C1.x,
            C2.y - C1.y,
            C2.z - C1.z
        };
    }
}
void RTXBVHSplit(BVHNode &mainBVH,BVHNode &leftBVH, BVHNode &rightBVH, const vector<particle> &parti){
    RTXV delta;
    RTXBVHAABB(parti, mainBVH, mainBVH.C1, mainBVH.C2, &delta);
    RTXV C1 = mainBVH.C1, C2 = mainBVH.C2;
    
    double maximum = max(max(delta.x,delta.y), delta.z);
    double fence = 0;
    if(maximum == delta.x){
        fence = C1.x + 0.5 * delta.x;
        leftBVH.C1 = {C1.x, C1.y, C1.z};
        leftBVH.C2 = {fence, C2.y, C2.z};
        rightBVH.C1 = {fence, C1.y, C1.z};
        rightBVH.C2 = {C2.x, C2.y, C2.z};
    }
    else if(maximum == delta.y){
        fence = C1.y + 0.5 * delta.y;
        leftBVH.C1 = {C1.x, C1.y, C1.z};
        leftBVH.C2 = {C2.x, fence, C2.z};
        rightBVH.C1 = {C1.x, fence, C1.z};
        rightBVH.C2 = {C2.x, C2.y, C2.z};
    }
    else if(maximum == delta.z){
        fence = C1.z + 0.5 * delta.z;
        leftBVH.C1 = {C1.x, C1.y, C1.z};
        leftBVH.C2 = {C2.x, C2.y, fence};
        rightBVH.C1 = {C1.x, C1.y, fence};
        rightBVH.C2 = {C2.x, C2.y, C2.z};
    }
    
    
    for(int i = 0;i < mainBVH.indices.size();i++){
        long p = mainBVH.indices[i];
        if(maximum == delta.x){
            if(parti[p].x < fence){
                leftBVH.indices.push_back(p);
            }else{
                rightBVH.indices.push_back(p);
            }
        }
        else if(maximum == delta.y){
            if(parti[p].y < fence){
                leftBVH.indices.push_back(p);
            }else{
                rightBVH.indices.push_back(p);
            }
        }
        else if(maximum == delta.z){
            if(parti[p].z < fence){
                leftBVH.indices.push_back(p);
            }else{
                rightBVH.indices.push_back(p);
            }
        }
    }

    /* ---- CUDA 移植修复(原 Metal 版 RTX.metal/main.mm 就存在的越界 bug) ----
       上面用的是"最宽轴几何中点"当 fence。行星(或者一堆粒子)密集在恒星一侧时,
       所有下标会被分到同一边, 另一边为空 -> 下面的 if 进不去 -> mainBVH 被当成叶子
       整个塞进 BVHCollector。CPU 版无所谓(vector<long> 变长), 但 GPU 版的
       metalBVHNode::indices 只有 7 个槽位, count 却写了真实个数(比如 10),
       kernel 里 for(i<count) 会读到 indices[7..] —— 也就是 CudaBVHNode 里紧跟在
       indices 后面的 left/right, 值是 -1, 于是拿它当下标访问 parti[-1]:
       mac 上 ASan 直接报 heap-buffer-overflow, 真 N 卡上就是 illegal memory access。
       这里补一个按最宽轴排序取中位数的兜底切分, 保证叶子节点最多 7 个粒子。 ---- */
    if(mainBVH.indices.size() > 7 && (leftBVH.indices.empty() || rightBVH.indices.empty())){
        int axis = (maximum == delta.x) ? 0 : ((maximum == delta.y) ? 1 : 2);
        auto coord = [&](long idx)->double{
            if(axis == 0)return parti[idx].x;
            if(axis == 1)return parti[idx].y;
            return parti[idx].z;
        };
        vector<long> sorted = mainBVH.indices;
        stable_sort(sorted.begin(), sorted.end(), [&](long a, long b){ return coord(a) < coord(b); });
        size_t half = sorted.size() / 2;
        leftBVH.indices.assign(sorted.begin(), sorted.begin() + half);
        rightBVH.indices.assign(sorted.begin() + half, sorted.end());
        RTXBVHAABB(parti, leftBVH, leftBVH.C1, leftBVH.C2);
        RTXBVHAABB(parti, rightBVH, rightBVH.C1, rightBVH.C2);
    }

    
    if(!leftBVH.indices.empty() && !rightBVH.indices.empty()){
        RTXBVHAABB(parti, leftBVH, leftBVH.C1, leftBVH.C2);
        mainBVH.left = BVHCollector.size();
        RTXBVHAABB(parti, rightBVH, rightBVH.C1, rightBVH.C2);
        mainBVH.right = BVHCollector.size() + 1;
        if(BVHCollector.empty()){
            mainBVH.left = 1;
            mainBVH.right = 2;
            BVHCollector.push_back(mainBVH);
        }
        BVHCollector.push_back(leftBVH);
        BVHCollector.push_back(rightBVH);
    }
    if(BVHCollector.empty()){
        BVHCollector.push_back(mainBVH);
    }
}

void RTXBVH(BVHNode &mainBVH, const vector<particle> &parti){
    BVHNode leftBVH, rightBVH;
    RTXBVHSplit(mainBVH, leftBVH, rightBVH, parti);
    if(mainBVH.left == -1 && mainBVH.right == -1){
        return ;
    }
    
    int L = mainBVH.left;
    int R = mainBVH.right;
    
    if(leftBVH.indices.size() > 7){
        RTXBVH(leftBVH, parti);
    }if(rightBVH.indices.size() > 7){
        RTXBVH(rightBVH, parti);
    }
    
    BVHCollector[L] = leftBVH;
    BVHCollector[R] = rightBVH;
    return ;
}

bool RTXBVHRayAABB(const BVHNode &nextBVH, const RTXV &vector, const RTXV &camera){
    bool intersect = false;
    double t_enter, t_exit;
    if((camera.x >= nextBVH.C1.x && camera.x <= nextBVH.C2.x) && (camera.y >= nextBVH.C1.y && camera.y <= nextBVH.C2.y) && (camera.z >= nextBVH.C1.z && camera.z <= nextBVH.C2.z)){
        return true;
    }
    if(vector.x == 0 && (camera.x < nextBVH.C1.x || camera.x > nextBVH.C2.x))return false;
    if(vector.y == 0 && (camera.y < nextBVH.C1.y || camera.y > nextBVH.C2.y))return false;
    if(vector.z == 0 && (camera.z < nextBVH.C1.z || camera.z > nextBVH.C2.z))return false;
    /* x-axis */
    double x_tmin = -INFINITY;
    double x_tmax = INFINITY;
    if(vector.x != 0){
        double x_t1 = (nextBVH.C1.x - camera.x) / vector.x;
        double x_t2 = (nextBVH.C2.x - camera.x) / vector.x;
        
        x_tmin = min(x_t1, x_t2);
        x_tmax = max(x_t1, x_t2);
    }
    /* y-axis */
    double y_tmin = -INFINITY;
    double y_tmax = INFINITY;
    if(vector.y != 0){
        double y_t1 = (nextBVH.C1.y - camera.y) / vector.y;
        double y_t2 = (nextBVH.C2.y - camera.y) / vector.y;
        
        y_tmin = min(y_t1, y_t2);
        y_tmax = max(y_t1, y_t2);
    }
    /* z-axis */
    double z_tmin = -INFINITY;
    double z_tmax = INFINITY;
    if(vector.z != 0){
        double z_t1 = (nextBVH.C1.z - camera.z) / vector.z;
        double z_t2 = (nextBVH.C2.z - camera.z) / vector.z;
        
        z_tmin = min(z_t1, z_t2);
        z_tmax = max(z_t1, z_t2);
    }

    t_enter = max({x_tmin, y_tmin, z_tmin});
    t_exit = min({x_tmax, y_tmax, z_tmax});
    if(t_enter <= t_exit && t_exit >= 0)intersect = true;
    
    return intersect;
}
void RTXBVHRay(vector<long> &indexa, const BVHNode &rBVH, const RTXV &vector, const RTXV &camera){
    if(rBVH.left == -1 && rBVH.right == -1){
        for(int i = 0;i < rBVH.indices.size();i++){
            indexa.push_back(rBVH.indices[i]);
        }
        return ;
    }
    if(rBVH.left != -1){
        if(RTXBVHRayAABB(BVHCollector[rBVH.left], vector, camera)){
            RTXBVHRay(indexa, BVHCollector[rBVH.left], vector, camera);
        }
    }
    if(rBVH.right != -1){
        if(RTXBVHRayAABB(BVHCollector[rBVH.right], vector, camera)){
            RTXBVHRay(indexa, BVHCollector[rBVH.right], vector, camera);
        }
    }
    return ;
}
void StellarHaloDetect(const vector<particle> &parti, const vector<int> &suns, const RTXV vector, const RTXV camera, Uint8 &R, Uint8 &G, Uint8 &B){
    for(int i = 0;i < suns.size();i++){
        long p = suns[i];
        
        //Check if intersect with halo of the stellar
        RTXV SC = {
            parti[p].x - camera.x, parti[p].y - camera.y, parti[p].z - camera.z
        };
        double SCLength = VectorLength(SC);
        double VLength = VectorLength(vector);
        RTXV VUnit = VectorMul(vector, 1/VLength);
        double CTLength = VectorDotMul(SC, VUnit);
        if(CTLength <= 0)continue;
        double STLength = sqrt(max(0.0,SCLength * SCLength - CTLength * CTLength));
        double I = 0.0;
        if(STLength > parti[p].radius && STLength < 8 * parti[p].radius){
            double q = STLength / parti[p].radius;
            I += (1 - q * q / 64) / (0.8 * q * q + 0.2);
        }else if(STLength <= parti[p].radius && STLength > 0){
            double q = STLength / parti[p].radius;
            I = 1.5 - 0.5 * q * q;
        }
        R = (Uint8)clamp(parti[p].c1 * I, 0.0, 255.0);
        G = (Uint8)clamp(parti[p].c2 * I, 0.0, 255.0);
        B = (Uint8)clamp(parti[p].c3 * I, 0.0, 255.0);
    }
}
void getTextureByCube(const particle &planet, const RTXV &vector, int &c1, int &c2, int &c3, int &c4){
    if(planet.hasTexture == false || !SS_SurfaceSetOK(planet.texture, 6))return;
    double inclination = clamp(planet.inclination, -90.0, 90.0) * M_PI / 180.0;
    
    RTXV rotVector = {
        vector.x * cos(planet.rotation) + vector.y * sin(planet.rotation) * sin(inclination) + vector.z * sin(planet.rotation) * cos(inclination),
        vector.y * cos(inclination) - vector.z * sin(inclination),
        -vector.x * sin(planet.rotation) + vector.y * sin(inclination) * cos(planet.rotation) + vector.z * cos(planet.rotation) * cos(inclination)
    };
    
    rotVector.x = -rotVector.x;
    
    int i, j;
    //px
    if(rotVector.x > 0 && rotVector.x >= abs(rotVector.y) && rotVector.x >= abs(rotVector.z)){
        double t = 1 / rotVector.x;
        i = int((1 - t * rotVector.z) * 0.5 * (planet.texture[0]->w));
        j = int((1 - t * rotVector.y) * 0.5 * (planet.texture[0]->h));
        SDL_Surface *tex = planet.texture[0];
        const Uint8 *pixels = (const Uint8*)tex->pixels;
        int p = j * tex->pitch + i * 4;
        c1 = pixels[p];
        c2 = pixels[p + 1];
        c3 = pixels[p + 2];
        c4 = pixels[p + 3];
    }
    //nx
    else if(rotVector.x <= 0 && -rotVector.x >= abs(rotVector.y) && -rotVector.x >= abs(rotVector.z)){
        double t = -1 / rotVector.x;
        i = int((1 + t * rotVector.z) * 0.5 * planet.texture[1]->w);
        j = int((1 - t * rotVector.y) * 0.5 * planet.texture[1]->h);
        SDL_Surface *tex = planet.texture[1];
        const Uint8 *pixels = (const Uint8*)tex->pixels;
        int p = j * tex->pitch + i * 4;
        c1 = pixels[p];
        c2 = pixels[p + 1];
        c3 = pixels[p + 2];
        c4 = pixels[p + 3];
    }
    //py
    else if(rotVector.y > 0 && rotVector.y >= abs(rotVector.x) && rotVector.y >= abs(rotVector.z)){
        double t = 1 / rotVector.y;
        i = int((t * rotVector.x + 1) * 0.5 * (planet.texture[2]->w));
        j = int((t * rotVector.z + 1) * 0.5 * (planet.texture[2]->h));
        SDL_Surface *tex = planet.texture[2];
        const Uint8 *pixels = (const Uint8*)tex->pixels;
        int p = j * tex->pitch + i * 4;
        c1 = pixels[p];
        c2 = pixels[p + 1];
        c3 = pixels[p + 2];
        c4 = pixels[p + 3];
    }
    //ny
    else if(rotVector.y <= 0 && -rotVector.y >= abs(rotVector.x) && -rotVector.y >= abs(rotVector.z)){
        double t = -1 / rotVector.y;
        i = int((1 + t * rotVector.x) * 0.5 * (planet.texture[3]->w));
        j = int((1 - t * rotVector.z) * 0.5 * (planet.texture[3]->h));
        SDL_Surface *tex = planet.texture[3];
        const Uint8 *pixels = (const Uint8*)tex->pixels;
        int p = j * tex->pitch + i * 4;
        c1 = pixels[p];
        c2 = pixels[p + 1];
        c3 = pixels[p + 2];
        c4 = pixels[p + 3];
    }
    //pz
    else if(rotVector.z > 0 && rotVector.z >= abs(rotVector.x) && rotVector.z >= abs(rotVector.y)){
        double t = 1 / rotVector.z;
        i = int((1 + t * rotVector.x) * 0.5 * (planet.texture[4]->w));
        j = int((1 - t * rotVector.y) * 0.5 * (planet.texture[4]->h));
        SDL_Surface *tex = planet.texture[4];
        const Uint8 *pixels = (const Uint8*)tex->pixels;
        int p = j * tex->pitch + i * 4;
        c1 = pixels[p];
        c2 = pixels[p + 1];
        c3 = pixels[p + 2];
        c4 = pixels[p + 3];
    }
    //nz
    else if(rotVector.z <= 0 && -rotVector.z >= abs(rotVector.x) && -rotVector.z >= abs(rotVector.y)){
        double t = -1 / rotVector.z;
        i = int((1 - t * rotVector.x) * 0.5 * (planet.texture[5]->w));
        j = int((1 - t * rotVector.y) * 0.5 * (planet.texture[5]->h));
        SDL_Surface *tex = planet.texture[5];
        const Uint8 *pixels = (const Uint8*)tex->pixels;
        int p = j * tex->pitch + i * 4;
        c1 = pixels[p];
        c2 = pixels[p + 1];
        c3 = pixels[p + 2];
        c4 = pixels[p + 3];
    }
}
void RTXRender(SDL_Renderer* ren, const vector<projection> &planet, const vector<particle> &parti, vector<int> suns){
    int rtxheight = height;
    int rtxwidth = width;
    if(ss.SSAA){
        rtxheight *= ss.SSAA_multi;
        rtxwidth *= ss.SSAA_multi;
    }
    /* Correct Ray Vector */
    RTXV Matrix[3] = {
        {cos(cam.yaw * M_PI/ 180)                              , -sin(cam.pitch * M_PI/ 180) * sin(cam.yaw * M_PI/ 180), cos(cam.pitch * M_PI/ 180) * sin(cam.yaw * M_PI/ 180)},
        {0                                                     , cos(cam.pitch * M_PI/ 180)                            , sin(cam.pitch * M_PI/ 180)                           },
        {-sin(cam.yaw * M_PI/ 180)                             , -sin(cam.pitch * M_PI/ 180) * cos(cam.yaw * M_PI/ 180), cos(cam.pitch * M_PI/ 180) * cos(cam.yaw * M_PI/ 180)}
    };
    #pragma omp parallel for
    for(int i = rtxheight - 1;i >= 0;i--){
        for(int j = 0;j < rtxwidth;j++){
            RTXVector[i][j].x = RTXOriginVector[i][j].x * Matrix[0].x + RTXOriginVector[i][j].y * Matrix[0].y + RTXOriginVector[i][j].z * Matrix[0].z;
            RTXVector[i][j].y = RTXOriginVector[i][j].x * Matrix[1].x + RTXOriginVector[i][j].y * Matrix[1].y + RTXOriginVector[i][j].z * Matrix[1].z;
            RTXVector[i][j].z = RTXOriginVector[i][j].x * Matrix[2].x + RTXOriginVector[i][j].y * Matrix[2].y + RTXOriginVector[i][j].z * Matrix[2].z;
        }
    }
    
    /* Construct BVH */
    BVHNode mainBVH;
    BVHCollector.clear();
    for(int p = 0;p < parti.size();p++){
        mainBVH.indices.push_back(p);
    }
    RTXBVH(mainBVH, parti);
    
    double SSAOI = 0.10;
    /* Ray out */
    vector<vector<Uint8>> Rbuf(rtxheight, vector<Uint8>(rtxwidth, 0));
    vector<vector<Uint8>> Gbuf(rtxheight, vector<Uint8>(rtxwidth, 0));
    vector<vector<Uint8>> Bbuf(rtxheight, vector<Uint8>(rtxwidth, 0));
    
    if(suns.empty()){
        suns.push_back(0);
    }
    
    #pragma omp parallel for
    for(int i = rtxheight - 1;i >= 0;i--){
        for(int j = 0;j < rtxwidth;j++){
            /* Execute BVH */
            vector<long> index;
            RTXBVHRay(index, BVHCollector[0], RTXVector[i][j], {cam.x, cam.y, cam.z});
            
            double nearestOM = INFINITY;
            long nearestPlanet = -1;
            RTXV M;
            for(int n = 0;n < index.size();n++){
                long p = index[n];
                /* Angle detect whether intersect*/
                /* Angular Method */
                RTXV Mcandi = RTXVectorCalcM(RTXVector[i][j], {cam.x, cam.y, cam.z}, {parti[p].x, parti[p].y, parti[p].z}, parti[p].radius, RTXVectorLength[i][j]);
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
            auto it = find(suns.begin(), suns.end(), nearestPlanet);
            if(nearestPlanet == -1 || it != suns.end()){
                int R = 0, G = 0, B = 0, A = 0;
                if(ss.hasbackground && nearestPlanet == -1){
                    getTextureByCube(ss.background, RTXVector[i][j], R, G, B, A);
                }
                Uint8 Rh = 0, Gh = 0, Bh = 0;
                StellarHaloDetect(parti, suns, RTXVector[i][j], {cam.x, cam.y, cam.z}, Rh, Gh, Bh);
                Rbuf[i][j] = min(255, R + Rh);
                Gbuf[i][j] = min(255, G + Gh);
                Bbuf[i][j] = min(255, B + Bh);
                continue;
            }
            /* Light Magnitude */
            RTXV Npm = {(M.x - parti[nearestPlanet].x)/parti[nearestPlanet].radius, (M.y - parti[nearestPlanet].y)/parti[nearestPlanet].radius, (M.z - parti[nearestPlanet].z)/parti[nearestPlanet].radius};
            double PS = sqrt(
                             pow(parti[suns[0]].x - parti[nearestPlanet].x, 2) +
                             pow(parti[suns[0]].y - parti[nearestPlanet].y, 2) +
                             pow(parti[suns[0]].z - parti[nearestPlanet].z, 2)
                             );
            RTXV Nps = {(parti[suns[0]].x - parti[nearestPlanet].x)/PS, (parti[suns[0]].y - parti[nearestPlanet].y)/PS, (parti[suns[0]].z - parti[nearestPlanet].z)/PS};
            double I = max(
                           (Npm.x * Nps.x + Npm.y * Nps.y + Npm.z * Nps.z) * (1 - SSAOI) + SSAOI
                           , 0.0);
            
            /* Shadow Calc */
            double MSLength = sqrt(
                             pow((parti[suns[0]].x - M.x), 2) +
                             pow((parti[suns[0]].y - M.y), 2) +
                             pow((parti[suns[0]].z - M.z), 2)
                             );
            
            RTXV MS = {parti[suns[0]].x - M.x, parti[suns[0]].y - M.y, parti[suns[0]].z - M.z};
            
            /* Execute Shadow BVH */
            vector<long> shadow_index;
            RTXBVHRay(shadow_index, BVHCollector[0], MS, M);
            for(int n = 0;n < shadow_index.size();n++){
                long p = shadow_index[n];
                if(p == nearestPlanet || p == suns[0])continue;
                RTXV Shadow = RTXVectorCalcM(MS, M, {parti[p].x, parti[p].y, parti[p].z}, parti[p].radius, MSLength);
                if(Shadow.intersect && Shadow.OM < MSLength && Shadow.OM > 0){
                    I = SSAOI;
                    break;
                }
            }

            /* RGB Calc */
            /* Define RGB */
            int c1 = parti[nearestPlanet].c1;
            int c2 = parti[nearestPlanet].c2;
            int c3 = parti[nearestPlanet].c3;
            int c4 = parti[nearestPlanet].c4;
            
            int c1Sun = parti[suns[0]].c1;
            int c2Sun = parti[suns[0]].c2;
            int c3Sun = parti[suns[0]].c3;
            int c4Sun = parti[suns[0]].c4;
            
            if(ss.Texture){
                getTextureByCube(parti[nearestPlanet], Npm, c1, c2, c3, c4);
            }
            
            Uint8 R = ((c1Sun - c1) * 0.05 + c1) * I;
            Uint8 G = ((c2Sun - c2) * 0.05 + c2) * I;
            Uint8 B = ((c3Sun - c3) * 0.05 + c3) * I;
            
            if(RTXReflect > 1){
                vector<RTXV> Mr(RTXReflect);
                Mr[0] = M;
                vector<RTXV> DirectVector(RTXReflect);
                DirectVector[0] = RTXVector[i][j];
                vector<vector<long>> indexr(RTXReflect-1);
                vector<particle> PreP(RTXReflect);
                PreP[0] = parti[nearestPlanet];
                double WR = 1.0;
                double RRefPre = R, GRefPre = G, BRefPre = B;
                double RR = R, GG = G, BB = B;
                for(int r = 1;r < RTXReflect;r++){
                    WR *= PreP[r-1].reflection;
                    
                    RTXV N = VectorMul({Mr[r-1].x - PreP[r-1].x, Mr[r-1].y - PreP[r-1].y, Mr[r-1].z - PreP[r-1].z}, 1/PreP[r-1].radius);
                    
                    RTXV ReflectVector = VectorUnit(VectorPlus(DirectVector[r-1], VectorMul(N, VectorDotMul(DirectVector[r-1], N) * (-2))));

                    DirectVector[r] = ReflectVector;
                    RTXBVHRay(indexr[r - 1], BVHCollector[0], ReflectVector, Mr[r - 1]);
                    
                    double nearestOMRef = INFINITY;
                    long nearestPlanetRef = -1;
                    for(int n = 0;n < indexr[r-1].size();n++){
                        long p = indexr[r - 1][n];
                        RTXV Mcandi = RTXVectorCalcM(ReflectVector, Mr[r - 1], {parti[p].x, parti[p].y, parti[p].z}, parti[p].radius, VectorLength(ReflectVector));
                        if(Mcandi.intersect == false){
                            continue;
                        }else{
                            if(Mcandi.OM > 0 && Mcandi.OM < nearestOMRef){
                                nearestOMRef = Mcandi.OM;
                                nearestPlanetRef = p;
                                Mr[r] = Mcandi;
                            }
                        }
                    }
                    auto it = find(suns.begin(), suns.end(), nearestPlanetRef);
                    if(nearestPlanetRef == -1 || it != suns.end()){
                        Uint8 SR = 0, SG = 0, SB = 0;
                        StellarHaloDetect(parti, suns, ReflectVector, Mr[r - 1], SR, SG, SB);
                        RR += (double(SR) - RRefPre) * WR;
                        GG += (double(SG) - GRefPre) * WR;
                        BB += (double(SB) - BRefPre) * WR;
                        break;
                    }
                    PreP[r] = parti[nearestPlanetRef];
                    /* Light Magnitude */
                    RTXV NpmRef = {
                        (Mr[r].x - parti[nearestPlanetRef].x)/parti[nearestPlanetRef].radius,
                        (Mr[r].y - parti[nearestPlanetRef].y)/parti[nearestPlanetRef].radius,
                        (Mr[r].z - parti[nearestPlanetRef].z)/parti[nearestPlanetRef].radius,
                    };
                    double PSRef = sqrt(
                                        pow(parti[suns[0]].x - parti[nearestPlanetRef].x, 2) + pow(parti[suns[0]].y - parti[nearestPlanetRef].y, 2) + pow(parti[suns[0]].z - parti[nearestPlanetRef].z, 2)
                                        );
                    RTXV NpsRef = {
                        (parti[suns[0]].x - parti[nearestPlanetRef].x)/PSRef,
                        (parti[suns[0]].y - parti[nearestPlanetRef].y)/PSRef,
                        (parti[suns[0]].z - parti[nearestPlanetRef].z)/PSRef,
                    };
                    double Ir = max(
                                    (NpmRef.x * NpsRef.x + NpmRef.y * NpsRef.y + NpmRef.z * NpsRef.z) * (1 - SSAOI) + SSAOI, 0.0
                                    );
                    /* Shadow Calc */
                    double MSLengthRef = sqrt(
                                              pow((parti[suns[0]].x - Mr[r].x), 2) +
                                              pow((parti[suns[0]].y - Mr[r].y), 2) +
                                              pow((parti[suns[0]].z - Mr[r].z), 2)
                                              );
                    
                    RTXV MSRef = {
                        parti[suns[0]].x - Mr[r].x,
                        parti[suns[0]].y - Mr[r].y,
                        parti[suns[0]].z - Mr[r].z,
                    };
                    
                    /* Execute Shadow BVH */
                    vector<long> shadow_indexRef;
                    RTXBVHRay(shadow_indexRef, BVHCollector[0], MSRef, Mr[r]);
                    for(int n = 0;n < shadow_indexRef.size();n++){
                        long p = shadow_indexRef[n];
                        if(p == nearestPlanetRef || p == suns[0])continue;
                        RTXV ShadowRef = RTXVectorCalcM(MSRef, Mr[r], {parti[p].x, parti[p].y, parti[p].z}, parti[p].radius, MSLengthRef);
                        if(ShadowRef.intersect && ShadowRef.OM < MSLengthRef && ShadowRef.OM > 0){
                            Ir = SSAOI;
                            break;
                        }
                    }
                    
                    int c1Ref = parti[nearestPlanetRef].c1;
                    int c2Ref = parti[nearestPlanetRef].c2;
                    int c3Ref = parti[nearestPlanetRef].c3;
                    int c4Ref = parti[nearestPlanetRef].c4;
                    if(ss.Texture){
                        getTextureByCube(parti[nearestPlanetRef], NpmRef, c1Ref, c2Ref, c3Ref, c4Ref);
                    }
                    
                    Uint8 RRefSun = ((c1Sun - c1Ref) * 0.05 + c1Ref) * Ir;
                    Uint8 GRefSun = ((c2Sun - c2Ref) * 0.05 + c2Ref) * Ir;
                    Uint8 BRefSun = ((c3Sun - c3Ref) * 0.05 + c3Ref) * Ir;
                    
                    RR += (RRefSun - RRefPre) * WR;
                    GG += (GRefSun - GRefPre) * WR;
                    BB += (BRefSun - BRefPre) * WR;
                    
                    RRefPre = RRefSun; GRefPre = GRefSun; BRefPre = BRefSun;
                }
                R = Uint8(clamp(RR, 0.0, 255.0));
                G = Uint8(clamp(GG, 0.0, 255.0));
                B = Uint8(clamp(BB, 0.0, 255.0));
            }
            
            
            /* RGB Pass*/
            Rbuf[i][j] = R;
            Gbuf[i][j] = G;
            Bbuf[i][j] = B;
        }
    }
//    for(int i = 0;i < height;i++){
//        for(int j = 0;j < width;j++){
//            /* Render */
//            SDL_SetRenderDrawColor(ren, Rbuf[i][j], Gbuf[i][j], Bbuf[i][j], 255);
//            SDL_RenderPoint(ren, j, i);
//        }
//    }
    static SDL_Texture* tex = nullptr;
    static int tw = 0, th = 0;

    if (!tex || tw != width || th != height) {
        if (tex) SDL_DestroyTexture(tex);

        tex = SDL_CreateTexture(
            ren, SDL_PIXELFORMAT_RGBA32,
            SDL_TEXTUREACCESS_STREAMING, width, height
        );

        tw = width;
        th = height;
    }
    
    vector<Uint8> rgb(width * height * 4);
    
    #pragma omp parallel for
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            int p = (i * width + j) * 4;
            if(ss.SSAA){
                int R=0, G=0, B=0;
                for(int z1 = 0;z1 < ss.SSAA_multi;z1++){
                    for(int z2 = 0;z2 < ss.SSAA_multi;z2++){
                        R += Rbuf[i * ss.SSAA_multi + z1][j * ss.SSAA_multi + z2];
                        G += Gbuf[i * ss.SSAA_multi + z1][j * ss.SSAA_multi + z2];
                        B += Bbuf[i * ss.SSAA_multi + z1][j * ss.SSAA_multi + z2];
                    }
                }
                rgb[p] = R/(ss.SSAA_multi * ss.SSAA_multi);
                rgb[p+1] = G/(ss.SSAA_multi * ss.SSAA_multi);
                rgb[p+2] = B/(ss.SSAA_multi * ss.SSAA_multi);
                rgb[p+3] = 255;
            }else{
                rgb[p] = Rbuf[i][j];
                rgb[p+1] = Gbuf[i][j];
                rgb[p+2] = Bbuf[i][j];
                rgb[p+3] = 255;
            }
        }
    }
    
    SDL_UpdateTexture(tex, nullptr, rgb.data(), width * 4);
    SDL_FRect dst = {0, 0, (float)width, (float)height};
    SDL_RenderTexture(ren, tex, nullptr, &dst);
}
/* ---- CUDA 移植: 这 4 个结构体(MetalRTXV / rtx_metal_projection / metalcamera / metalBVHNode)
     搬到 gpu_common.h 了(改名 CudaRTXV / CudaProjection / CudaCamera / CudaBVHNode,
     并在头文件里 typedef 回原来这四个名字), 因为 host 要填、device 也要读, 必须共用同一份定义,
     否则两边对齐方式不一致就会读到垃圾数据。
     头文件里有 static_assert 锁死布局: 20B / 48B / 36B / 112B。
     唯一改动: metalBVHNode::indices 从 long 改成 long long(Windows 的 long 只有 32 位)。 ---- */
void RTXRenderMetal(SDL_Renderer* ren, const vector<projection> &planet, const vector<particle> &parti, const vector<int> &suns){
    static const bool s_gpuOk = (gpuInit() == 1);   /* CUDA移植: 首次调用时初始化 GPU(替代 Metal device/queue/PSO) */
    /* CUDA 移植(本机新增): 这台机器没有可用的 CUDA 设备时自动退回 CPU 光线追踪,
       画面一样, 只是慢, 不会像原来那样静默出一张全黑的图。 */
    if (!s_gpuOk) { RTXRender(ren, planet, parti, suns); return; }
    int rtxheight = height;
    int rtxwidth = width;
    if(ss.SSAA){
        rtxheight *= ss.SSAA_multi;
        rtxwidth *= ss.SSAA_multi;
    }
    /* Initial BVH */
    BVHNode mainBVH;
    BVHCollector.clear();
    for(int p = 0;p < parti.size();p++){
        mainBVH.indices.push_back(p);
    }
    RTXBVH(mainBVH, parti);
    
    vector<metalBVHNode> metalBVHCollector(BVHCollector.size());
    for(int i = 0;i < BVHCollector.size();i++){
        int size = BVHCollector[i].indices.size();
        int maxcollector = min(size, 7);
        metalBVHCollector[i].C1 = {
            float(BVHCollector[i].C1.x - cam.x),
            float(BVHCollector[i].C1.y - cam.y),
            float(BVHCollector[i].C1.z - cam.z)
        };
        metalBVHCollector[i].C2 = {
            float(BVHCollector[i].C2.x - cam.x),
            float(BVHCollector[i].C2.y - cam.y),
            float(BVHCollector[i].C2.z - cam.z)
        };
        metalBVHCollector[i].left = BVHCollector[i].left;
        metalBVHCollector[i].right = BVHCollector[i].right;
        bool leaf = BVHCollector[i].left == -1 && BVHCollector[i].right == -1;
        /* CUDA 移植修复: count 永远不能超过 indices[] 的长度 7(见上面 RTXBVHSplit 的注释)。
           正常情况下 RTXBVHSplit 的兜底切分已经保证叶子 <= 7 个, 这里是第二道保险,
           宁可少画一个粒子也决不能让 kernel 越界读显存。 */
        metalBVHCollector[i].count = leaf ? maxcollector : 0;
        for(int j = size;j < 7;j++){
            metalBVHCollector[i].indices[j] = 0;   /* 显式清零, 免得留垃圾值 */
        }
        for(int j = 0;j < maxcollector;j++){
            metalBVHCollector[i].indices[j] = BVHCollector[i].indices[j];
        }
    }

    
    /* Correct Ray Vector */
    MetalRTXV Matrix[3] = {
        {float(cos(cam.yaw * M_PI/ 180)), float(-sin(cam.pitch * M_PI/ 180) * sin(cam.yaw * M_PI/ 180)),float(cos(cam.pitch * M_PI/ 180) * sin(cam.yaw * M_PI/ 180))},
        {0, float(cos(cam.pitch * M_PI/ 180)), float(sin(cam.pitch * M_PI/ 180))},
        {float(-sin(cam.yaw * M_PI/ 180)), float(-sin(cam.pitch * M_PI/ 180) * cos(cam.yaw * M_PI/ 180)), float(cos(cam.pitch * M_PI/ 180) * cos(cam.yaw * M_PI/ 180))}
    };
    vector<MetalRTXV> flat_RTXOriginVector(rtxwidth * rtxheight);
    vector<float> flat_RTXVectorLength(rtxwidth * rtxheight);
    #pragma omp parallel for
    for(int i = 0;i < rtxheight;i++){
        for(int j = 0;j < rtxwidth;j++){
            flat_RTXOriginVector[i*rtxwidth + j] = {
                float(RTXOriginVector[i][j].x), float(RTXOriginVector[i][j].y), float(RTXOriginVector[i][j].z),
                RTXOriginVector[i][j].intersect,
                float(RTXOriginVector[i][j].OM)
            };
            flat_RTXVectorLength[i*rtxwidth + j] = float(RTXVectorLength[i][j]);
        }
    }
    
    /* CUDA 移植: 原来这里是 [queuea commandBuffer], 两个 encoder 挂在同一个 commandBuffer 上,
       靠 Metal 保证先后顺序。CUDA 用 default stream, 顺序同样有序, 提交点在两个 dispatch 之后
       各调一次 gpuCommit()(= [cmdBuf commit] + waitUntilCompleted)。 */
    
    /* Correcting Ray Vector */
    /* = [device newBufferWithBytes:... options:MTLResourceStorageModeShared]
       这些都是 kernel 只读的大块数据, 用 GPU_MEM_DEVICE(纯显存)就够;
       gpuBufferNew 内部一律清零(Metal 新 buffer 读出来就是 0, CUDA 不会, 必须自己清) */
    GpuBuffer* RTXOriginVectorBuf = gpuBufferNewWithBytes(flat_RTXOriginVector.data(), flat_RTXOriginVector.size()*sizeof(MetalRTXV), GPU_MEM_DEVICE);
    GpuBuffer* MatrixBuf          = gpuBufferNewWithBytes(Matrix, sizeof(Matrix), GPU_MEM_DEVICE);
    GpuBuffer* WidthBuf           = gpuBufferNewWithBytes(&rtxwidth, sizeof(rtxwidth), GPU_MEM_DEVICE);
    GpuBuffer* RTXVectorBuf       = gpuBufferNew(flat_RTXOriginVector.size()*sizeof(MetalRTXV), GPU_MEM_DEVICE);
    
    /* = [cmdBuf computeCommandEncoder] + [correct_vectorEnc setBuffer:... atIndex:n]
       atIndex 编号和 kernel_rtx.cu 里 correct_vector 的 [[buffer(n)]] 完全一致 */
    GpuEncoder correctVectorEnc;
    gpuEncoderInit(&correctVectorEnc);
    gpuSetBuffer(&correctVectorEnc, 0, RTXOriginVectorBuf);
    gpuSetBuffer(&correctVectorEnc, 1, MatrixBuf);
    gpuSetBuffer(&correctVectorEnc, 2, WidthBuf);
    gpuSetBuffer(&correctVectorEnc, 3, RTXVectorBuf);
    
    /* 原来:
         NSUInteger threadGroupSize = MIN(sqrt(correct_vectorPSO.maxTotalThreadsPerThreadgroup), sqrt(w*h));
       CUDA 里 block 固定 16x16=256(GPU_RTX_BLOCK), grid 向上取整, kernel 自己越界退出 */
    gpuDispatchCorrectVector(&correctVectorEnc, rtxwidth, rtxheight);
    gpuCommit();
    /* Correcting Ray Vector */
    /* Light Path Tracing */
    vector<Uint8> Sample(rtxheight * rtxwidth);
    
    /* CUDA移植: 原来是 rtx_metal_projection rtx_parti[parti.size()]; —— 变长数组(VLA),
       MSVC 不支持, 换成 vector; 下面建 buffer 时传 .data() / .size()*sizeof(...) */
    vector<rtx_metal_projection> rtx_parti(parti.size());

    #pragma omp parallel for
    for(long i = 0;i < parti.size();i++){
        rtx_parti[i] = {    float(parti[i].x - cam.x), float(parti[i].y - cam.y), float(parti[i].z - cam.z),
                            parti[i].c1, parti[i].c2, parti[i].c3, parti[i].c4,
                            float(parti[i].radius), (float)parti[i].reflection, (float)parti[i].w_velocity, (float)parti[i].rotation, (float)parti[i].inclination
                        };
    }
    metalcamera MetalCam = {
        0, 0, 0,
        float(cam.yaw), float(cam.pitch),
        float(cam.fov),
        0, 0, 0
    };
    
    /* Texture Renderer */
    vector<int> textureIndex(parti.size(), -1);
    
    /* Texture Background */
    bool hasSky = ss.hasbackground;
    vector<Uint8> skyTexture;
    int skyS = 0;
    if(ss.hasbackground && !SS_SurfaceSetOK(ss.background.texture, 6)){
        fprintf(stderr, "[background] 背景贴图不完整, 本次渲染按无背景处理\n");
        ss.hasbackground = false;
    }
    if(ss.hasbackground){
        for(int j = 0;j < 6;j++){
            SDL_Surface *tex = ss.background.texture[j];
            const Uint8 *pixels = (const Uint8*)tex->pixels;
            skyTexture.insert(
                              skyTexture.end(),
                              pixels,
                              pixels + tex->pitch * tex->h
                              );
        }
        skyS = ss.background.texture[0]->w;
    }
    if(skyTexture.empty())skyTexture.push_back(0);
    rtx_metal_projection sky = {};
    sky.rotation = ss.background.rotation;
    sky.inclination = ss.background.inclination;
    
    /* = [device newBufferWithBytes:... options:MTLResourceStorageModeShared] -> 纯显存就够 */
    GpuBuffer* RTXVectorLengthBuf   = gpuBufferNewWithBytes(flat_RTXVectorLength.data(), flat_RTXVectorLength.size()*sizeof(float), GPU_MEM_DEVICE);
    GpuBuffer* sunsBuf              = gpuBufferNewWithBytes(suns.data(), suns.size()*sizeof(int), GPU_MEM_DEVICE);
    GpuBuffer* partiBuf             = gpuBufferNewWithBytes(rtx_parti.data(), rtx_parti.size()*sizeof(rtx_metal_projection), GPU_MEM_DEVICE);
    GpuBuffer* cameraBuf            = gpuBufferNewWithBytes(&MetalCam, sizeof(MetalCam), GPU_MEM_DEVICE);
    GpuBuffer* metalBVHCollectorBuf = gpuBufferNewWithBytes(metalBVHCollector.data(), metalBVHCollector.size()*sizeof(metalBVHNode), GPU_MEM_DEVICE);
    
//    id<MTLBuffer> textureIndexBuf = [device newBufferWithBytes:textureIndex.data() length:textureIndex.size()*sizeof(int) options:MTLResourceStorageModeShared];
//    id<MTLBuffer> textureBuf = [device newBufferWithBytes:texturePixels.data() length:texturePixels.size()*sizeof(Uint8) options:MTLResourceStorageModeShared];
//    id<MTLBuffer> SBuf = [device newBufferWithBytes:S.data() length:S.size()*sizeof(int) options:MTLResourceStorageModeShared];
//    id<MTLBuffer> skyTextureBuf = [device newBufferWithBytes:skyTexture.data() length:skyTexture.size()*sizeof(Uint8) options:MTLResourceStorageModeShared];
    static GpuBuffer *textureIndexBuf = nullptr;
    static GpuBuffer *textureBuf      = nullptr;
    static GpuBuffer *SBuf            = nullptr;
    static GpuBuffer *skyTextureBuf   = nullptr;
    /* CUDA移植: 这四张图是 static 缓存的(和原来一样, 只在 textureChanged 时重建)。
       加一条保护: 粒子数变了就强制重建。因为 textureIndex/S 的长度是当时的 parti.size(),
       粒子变多以后 kernel 里 parti[i] 会读到 buffer 外面 —— Metal 一般只是读到脏数据,
       CUDA 读越界是直接 illegal address 崩溃。 */
    static long long s_texturePartiSize = -1;
    if((long long)parti.size() != s_texturePartiSize){
        s_texturePartiSize = (long long)parti.size();
        ss.textureChanged = true;
    }
    if(ss.textureChanged || !textureBuf){
        if(textureIndexBuf){ gpuBufferFree(textureIndexBuf); textureIndexBuf = nullptr; }
        if(textureBuf){ gpuBufferFree(textureBuf); textureBuf = nullptr; }
        if(SBuf){ gpuBufferFree(SBuf); SBuf = nullptr; }
        vector<Uint8> texturePixels;
        vector<int> S(parti.size(), -1);
        if(ss.Texture){
            for(long i = 0;i < parti.size();i++){
                /* parti 是 const 的, 这里不改 hasTexture, 只在打包时把不完整的当没有贴图 */
                if(parti[i].hasTexture && SS_SurfaceSetOK(parti[i].texture, 6)){
                    textureIndex[i] = texturePixels.size();
                    for(int j = 0;j < 6;j++){
                        SDL_Surface *tex = parti[i].texture[j];
                        const Uint8 *pixels = (const Uint8*)tex->pixels;
                        texturePixels.insert(
                                         texturePixels.end(),
                                         pixels,
                                         pixels + tex->pitch * tex->h
                                         );
                    }
                    S[i] = parti[i].texture[0]->w;
                }
            }
        }
        if(texturePixels.empty()) texturePixels.push_back(0);
        
        textureIndexBuf = gpuBufferNewWithBytes(textureIndex.data(), textureIndex.size()*sizeof(int), GPU_MEM_DEVICE);
        textureBuf      = gpuBufferNewWithBytes(texturePixels.data(), texturePixels.size()*sizeof(Uint8), GPU_MEM_DEVICE);
        SBuf            = gpuBufferNewWithBytes(S.data(), S.size()*sizeof(int), GPU_MEM_DEVICE);
        ss.textureChanged = false;
    }
    if(ss.backgroundChanged || !skyTextureBuf){
        if(skyTextureBuf){ gpuBufferFree(skyTextureBuf); skyTextureBuf = nullptr; }
        skyTextureBuf = gpuBufferNewWithBytes(skyTexture.data(), skyTexture.size()*sizeof(Uint8), GPU_MEM_DEVICE);
        ss.backgroundChanged = false;
    }

    /* R/G/B 三个平面最后要被 host 读回来打包成 SDL 纹理 -> 用 GPU_MEM_SHARED,
       这样原来 Uint8 *R = (Uint8*)RBuf.contents 的写法可以原样保留 */
    GpuBuffer* RBuf = gpuBufferNew(Sample.size()*sizeof(Uint8), GPU_MEM_SHARED);
    GpuBuffer* GBuf = gpuBufferNew(Sample.size()*sizeof(Uint8), GPU_MEM_SHARED);
    GpuBuffer* BBuf = gpuBufferNew(Sample.size()*sizeof(Uint8), GPU_MEM_SHARED);
    
    
    GpuEncoder rtxPathEnc;
    gpuEncoderInit(&rtxPathEnc);
    gpuSetBuffer(&rtxPathEnc, 0, WidthBuf);
    gpuSetBuffer(&rtxPathEnc, 1, RTXVectorBuf);
    gpuSetBuffer(&rtxPathEnc, 2, RTXVectorLengthBuf);
    gpuSetBuffer(&rtxPathEnc, 3, sunsBuf);
    long long sunsSize = suns.size();            /* CUDA移植: long long(Metal 的 long = 64 位) */
    gpuSetBytes(&rtxPathEnc, 4, &sunsSize, sizeof(sunsSize));
    gpuSetBuffer(&rtxPathEnc, 5, partiBuf);
    long long partiSize = parti.size();         /* CUDA移植: long long, 同上 */
    gpuSetBytes(&rtxPathEnc, 6, &partiSize, sizeof(partiSize));
    gpuSetBuffer(&rtxPathEnc, 7, cameraBuf);
    gpuSetBuffer(&rtxPathEnc, 8, metalBVHCollectorBuf);
    int metalBVHCollectorSize = metalBVHCollector.size();
    gpuSetBytes(&rtxPathEnc, 9, &metalBVHCollectorSize, sizeof(metalBVHCollectorSize));
    gpuSetBytes(&rtxPathEnc, 10, &RTXReflect, sizeof(RTXReflect));
    gpuSetBuffer(&rtxPathEnc, 11, textureIndexBuf);
    gpuSetBuffer(&rtxPathEnc, 12, textureBuf);
    gpuSetBuffer(&rtxPathEnc, 13, SBuf);
    gpuSetBytes(&rtxPathEnc, 14, &hasSky, sizeof(hasSky));
    gpuSetBuffer(&rtxPathEnc, 15, skyTextureBuf);
    gpuSetBytes(&rtxPathEnc, 16, &skyS, sizeof(skyS));
    gpuSetBytes(&rtxPathEnc, 17, &sky, sizeof(sky));
    gpuSetBuffer(&rtxPathEnc, 18, RBuf);
    gpuSetBuffer(&rtxPathEnc, 19, GBuf);
    gpuSetBuffer(&rtxPathEnc, 20, BBuf);
    
    /* = [RTXPathEnc dispatchThreads:MTLSizeMake(rtxwidth, rtxheight, 1)
         threadsPerThreadgroup:16x16]  (threadGroupSize 那三行删掉了, CUDA 里 block 固定 16x16) */
    gpuDispatchRTXPath(&rtxPathEnc, rtxwidth, rtxheight);
    /* Light Path Tracing */
    
    gpuCommit();                    /* = [cmdBuf commit]; [cmdBuf waitUntilCompleted]; */
    
    /* Result take back */
    Uint8 *R = (Uint8*)gpuBufferContents(RBuf);          /* = (Uint8*)RBuf.contents(SHARED buffer 的 host 指针) */
    Uint8 *G = (Uint8*)gpuBufferContents(GBuf);
    Uint8 *B = (Uint8*)gpuBufferContents(BBuf);
    /* SDL Draw */
    static SDL_Texture* tex = nullptr;
    static int tw = 0, th = 0;

    if (!tex || tw != width || th != height) {
        if (tex) SDL_DestroyTexture(tex);

        tex = SDL_CreateTexture(
            ren, SDL_PIXELFORMAT_RGBA32,
            SDL_TEXTUREACCESS_STREAMING, width, height
        );

        tw = width;
        th = height;
    }
    vector<Uint8> rgb(width * height * 4);
    
    #pragma omp parallel for
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            int pixel = i * width + j;
            int p = pixel * 4;
            
            if(ss.SSAA){
                int Rb=0, Gb=0, Bb=0;
                for(int z1 = 0;z1 < ss.SSAA_multi;z1++){
                    for(int z2 = 0;z2 < ss.SSAA_multi;z2++){
                        int index = (i * ss.SSAA_multi + z1) * rtxwidth + (j * ss.SSAA_multi + z2);
                        Rb += R[index];
                        Gb += G[index];
                        Bb += B[index];
                    }
                }
                rgb[p] = Rb/(ss.SSAA_multi * ss.SSAA_multi);
                rgb[p+1] = Gb/(ss.SSAA_multi * ss.SSAA_multi);
                rgb[p+2] = Bb/(ss.SSAA_multi * ss.SSAA_multi);
                rgb[p+3] = 255;
            }else{
                rgb[p] = R[pixel];
                rgb[p+1] = G[pixel];
                rgb[p+2] = B[pixel];
                rgb[p+3] = 225;
            }
        }
    }
    
    SDL_UpdateTexture(tex, nullptr, rgb.data(), width * 4);
    SDL_FRect dst = {0, 0, (float)width, (float)height};
    SDL_RenderTexture(ren, tex, nullptr, &dst);
    /* CUDA 移植: Metal 的 id<MTLBuffer> 由 ARC 自动回收, CUDA 必须自己 free。
       下面这些是每帧新建的, 不释放的话显存几秒钟就漏光。
       (textureIndexBuf / textureBuf / SBuf / skyTextureBuf 是 static 缓存, 不能在这里释放) */
    gpuBufferFree(RTXOriginVectorBuf);
    gpuBufferFree(MatrixBuf);
    gpuBufferFree(WidthBuf);
    gpuBufferFree(RTXVectorBuf);
    gpuBufferFree(RTXVectorLengthBuf);
    gpuBufferFree(sunsBuf);
    gpuBufferFree(partiBuf);
    gpuBufferFree(cameraBuf);
    gpuBufferFree(metalBVHCollectorBuf);
    gpuBufferFree(RBuf);
    gpuBufferFree(GBuf);
    gpuBufferFree(BBuf);
}
void DrawBVHDebug(SDL_Renderer* ren, const vector<BVHNode> &BVHCollector, const camera &cam, int maxDepth){

    if(BVHCollector.empty()) return;

    auto ProjectToScreen = [&](const RTXV &p, float &sx, float &sy)->bool{

        double dx = p.x - cam.x;
        double dy = p.y - cam.y;
        double dz = p.z - cam.z;

        double yaw = cam.yaw * M_PI / 180.0;
        double pitch = cam.pitch * M_PI / 180.0;
        double fov = cam.fov * M_PI / 180.0;

        // yaw
        double x1 = dx * cos(yaw) - dz * sin(yaw);
        double z1 = dx * sin(yaw) + dz * cos(yaw);

        // pitch
        double y2 = dy * cos(pitch) - z1 * sin(pitch);
        double z2 = dy * sin(pitch) + z1 * cos(pitch);

        if(z2 <= 0.01) return false;

        sx = float((x1 / (z2 * tan(fov / 2.0)) + 1.0) * width * 0.5);
        sy = float((1.0 - y2 / (z2 * tan(fov / 2.0))) * height * 0.5);

        return true;
    };

    auto DrawOneBox = [&](const BVHNode &node){

        RTXV p[8] = {
            {node.C1.x, node.C1.y, node.C1.z},
            {node.C2.x, node.C1.y, node.C1.z},
            {node.C1.x, node.C2.y, node.C1.z},
            {node.C2.x, node.C2.y, node.C1.z},
            {node.C1.x, node.C1.y, node.C2.z},
            {node.C2.x, node.C1.y, node.C2.z},
            {node.C1.x, node.C2.y, node.C2.z},
            {node.C2.x, node.C2.y, node.C2.z}
        };

        const int edge[12][2] = {
            {0,1},{0,2},{1,3},{2,3},
            {4,5},{4,6},{5,7},{6,7},
            {0,4},{1,5},{2,6},{3,7}
        };

        float sx[8], sy[8];
        bool ok[8];

        for(int i = 0; i < 8; i++){
            ok[i] = ProjectToScreen(p[i], sx[i], sy[i]);
        }

        for(int i = 0; i < 12; i++){
            int a = edge[i][0];
            int b = edge[i][1];

            if(ok[a] && ok[b]){
                SDL_RenderLine(ren, sx[a], sy[a], sx[b], sy[b]);
            }
        }
    };

    auto DFS = [&](auto&& self, int idx, int depth)->void{

        if(idx < 0 || idx >= BVHCollector.size()) return;
        if(depth > maxDepth) return;

        if(depth == 0) SDL_SetRenderDrawColor(ren, 255, 60, 60, 255);
        else if(depth == 1) SDL_SetRenderDrawColor(ren, 255, 220, 60, 255);
        else if(depth == 2) SDL_SetRenderDrawColor(ren, 60, 255, 120, 255);
        else SDL_SetRenderDrawColor(ren, 80, 180, 255, 255);

        DrawOneBox(BVHCollector[idx]);

        self(self, BVHCollector[idx].left, depth + 1);
        self(self, BVHCollector[idx].right, depth + 1);
    };

    DFS(DFS, 0, 0);
}
