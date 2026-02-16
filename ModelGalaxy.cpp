#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <iomanip>
#include <fstream>

using namespace std;

double G = 1.0;

struct Config {
    string name;
    double x, y, z, vx, vy, vz, mass, radius;
    int r, g, b;
};

void setRandomColor(int &r, int &g, int &b, mt19937 &gen) {
    uniform_int_distribution<int> dist(80, 255);
    r = dist(gen); g = dist(gen); b = dist(gen);
}

void saveToFile(const vector<Config>& bodies) {
    ofstream outFile("output.txt");
    outFile << bodies.size() << endl;
    outFile << fixed << setprecision(4);
    for (const auto& b : bodies) {
        outFile << b.name << " " << b.x << " " << b.y << " " << b.z << " "
                << b.vx << " " << b.vy << " " << b.vz << " 0.0 0.0 0.0 "
                << b.mass << " " << b.radius << " "
                << b.r << " " << b.g << " " << b.b << " 255" << endl;
    }
}

int main() {
    again:
    int mode, n; unsigned int seed;
    cout << "Advanced Models:\n1. Spiral (Thick)\n2. Disk (Tilted)\n3. Cluster (Sphere)\n4. Kuiper Belt (Ring)\n5. Cosmic Filament (String)\n6. Galaxy Collision (3D Cross)\n7. Pure Chaos\nChoice: ";
    cin >> mode;
    if(mode==0){
        double newG;
        cin>>newG;
        G=newG;
        goto again;
    }
    cout << "Quantity: "; cin >> n;
    cout << "Seed (0 for random): "; cin >> seed;

    if (seed == 0) seed = random_device{}();
    mt19937 gen(seed);
    normal_distribution<double> gauss(0, 1.0); // 用于生成自然厚度
    vector<Config> bodies;

    if (mode == 1) { // 1. 螺旋星系 (带厚度核球)
        double sun_m = 60000.0;
        bodies.push_back({"Center", 0,0,0,0,0,0, sun_m, 18, 255, 255, 200});
        for (int i = 0; i < n; i++) {
            double r = 50.0 + (double)i * (650.0 / n);
            double angle = r * 0.08 + (uniform_real_distribution<double>(0, 0.6)(gen));
            double thick = 15.0 * exp(-r/200.0); // 中心厚边缘薄
            double y = gauss(gen) * thick; 
            double v = sqrt(G * sun_m / r);
            int cr, cg, cb; setRandomColor(cr, cg, cb, gen);
            bodies.push_back({"S_"+to_string(i), r*cos(angle), y, r*sin(angle), -sin(angle)*v, gauss(gen)*0.1, cos(angle)*v, 10, 2, cr, cg, cb});
        }
    }
    else if (mode == 2) { // 2. 原行星盘 (带扰动和厚度)
        double sun_m = 25000.0;
        bodies.push_back({"Sun", 0,0,0,0,0,0, sun_m, 20, 255, 220, 30});
        for (int i = 0; i < n; i++) {
            double r = uniform_real_distribution<double>(70, 550)(gen);
            double a = uniform_real_distribution<double>(0, 2*M_PI)(gen);
            double y = gauss(gen) * 2.0; // 较薄的盘
            double v = sqrt(G * sun_m / r);
            int cr, cg, cb; setRandomColor(cr, cg, cb, gen);
            bodies.push_back({"P_"+to_string(i), r*cos(a), y, r*sin(a), -sin(a)*v, gauss(gen)*0.05, cos(a)*v, 15, 3, cr, cg, cb});
        }
    }
    else if (mode == 4) { // 4. 柯伊伯带 (中空环状)
        double sun_m = 40000.0;
        bodies.push_back({"Sun", 0,0,0,0,0,0, sun_m, 15, 255, 255, 100});
        for (int i = 0; i < n; i++) {
            double r = uniform_real_distribution<double>(350, 450)(gen); // 限制半径范围
            double a = uniform_real_distribution<double>(0, 2*M_PI)(gen);
            double y = gauss(gen) * 8.0;
            double v = sqrt(G * sun_m / r);
            int cr, cg, cb; setRandomColor(cr, cg, cb, gen);
            bodies.push_back({"K_"+to_string(i), r*cos(a), y, r*sin(a), -sin(a)*v, gauss(gen)*0.2, cos(a)*v, 5, 2, cr, cg, cb});
        }
    }
    else if (mode == 5) { // 5. 星系长城 (丝状结构)
        for (int i = 0; i < n; i++) {
            double x = (double)i * (1000.0 / n) - 500.0;
            double y = gauss(gen) * 20.0;
            double z = gauss(gen) * 20.0;
            int cr, cg, cb; setRandomColor(cr, cg, cb, gen);
            bodies.push_back({"W_"+to_string(i), x, y, z, 1.0, 0, 0, 50, 3, cr, cg, cb});
        }
    }
    else if (mode == 6) { // 6. 星系大碰撞 (立体交叉)
        auto make_galaxy = [&](double ox, double oy, double oz, double vx, double vy, double vz, double tilt, string p) {
            double sm = 20000.0;
            bodies.push_back({p+"BH", ox, oy, oz, vx, vy, vz, sm, 12, 255, 255, 255});
            for (int i = 0; i < n/2; i++) {
                double r = uniform_real_distribution<double>(30, 200)(gen);
                double a = uniform_real_distribution<double>(0, 2*M_PI)(gen);
                double v = sqrt(G * sm / r);
                double lx = r*cos(a), lz = r*sin(a);
                // 倾斜变换
                double rx = lx;
                double ry = lz * sin(tilt);
                double rz = lz * cos(tilt);
                int cr, cg, cb; setRandomColor(cr, cg, cb, gen);
                bodies.push_back({p+to_string(i), ox+rx, oy+ry, oz+rz, vx, vy-sin(a)*v*sin(tilt), vz+cos(a)*v*cos(tilt), 10, 2, cr, cg, cb});
            }
        };
        make_galaxy(-250, 0, 0, 4, 0, 0, M_PI/4.0, "G1_");
        make_galaxy(250, 0, 0, -4, 0, 0, -M_PI/4.0, "G2_");
    }
    else { // 3 & 7 模式 (球状星团 & 杂乱)
        for (int i = 0; i < n; i++) {
            double r = (mode == 3) ? uniform_real_distribution<double>(0, 200)(gen) : 500;
            double x = gauss(gen) * r, y = gauss(gen) * r, z = gauss(gen) * r;
            double v = (mode == 3) ? 0 : 8.0;
            int cr, cg, cb; setRandomColor(cr, cg, cb, gen);
            bodies.push_back({"B_"+to_string(i), x, y, z, gauss(gen)*v, gauss(gen)*v, gauss(gen)*v, 100, 3, cr, cg, cb});
        }
    }

    saveToFile(bodies);
    cout << "Model " << mode << " generated. Check output.txt!" << endl;
    return 0;
}