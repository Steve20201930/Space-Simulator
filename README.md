# Stellar-Roche-Sim 🪐 

[English](#english) | [简体中文](#chinese)

---

<a name="english"></a>
## English Version

A high-performance, real-time 3D N-body gravity simulator built with **C++20** and **SDL3**. This project simulates complex orbital mechanics, celestial collisions, and most notably, the **Roche Limit**—the critical threshold where a celestial body is "peeled" and torn apart by tidal forces.

### ✨ Key Features
* **Directional Tidal Disintegration (Vector Biasing)**: Unlike random explosions, fragments are "peeled" with a bias toward the massive attractor using a custom **Vector Biasing** algorithm, creating realistic tidal streams and accretion disk formations.
* **OpenMP-Powered Physics**: Multi-core parallel computing to handle thousands of simultaneous gravitational interactions ($O(N^2)$ complexity) efficiently.
* **Momentum-Preserving Mergers**: Inelastic collision system that merges planets upon contact, strictly conserving mass and momentum.
* **3D Camera & Flight Engine**: 
    * **6-DOF Movement**: Flight-simulator style controls (Yaw/Pitch/Roll).
    * **Target Tracking**: Lock the camera onto a specific planet to follow its journey.
    * **Perspective Scaling**: Real-time adjustment of projection focal length.

### 🎮 Controls
| Key | Action |
| :--- | :--- |
| **W/S/A/D** | Fly Forward / Backward / Left / Right |
| **Space / L-Shift** | Fly Up / Down |
| **TAB** | Toggle Simulation Pause |
| **T** | Open Console (Paste commands/data from clipboard) |
| **F** | (While Paused) Lock Camera to Focus Target |
| **Arrow Up / Down** | Scale Time Step ($dt$) |
| **Arrow Left / Right** | Adjust Projection/Focal Length |
| **0 (Zero)** | Save Current State to `Running_output.txt` |
| **ESC** | Unlock Mouse / Close Console |

### 🛠️ Developer Console (T-Mode)
Press **T** to enter commands:
* `-3`: Toggle Pause.
* `-4`: Toggle Roche Limit disintegration.
* `-5 [value]`: Set Roche Constant $K$ (Default: 1.26).
* `-6 [value]`: Set Gravitational Constant $G$.

### 🚀 Compilation
```bash
g++ -O3 -fopenmp main.cpp -o stellar_sim -lSDL3
