# Stellar-Roche-Sim 🪐 

[English](#english) | [简体中文](#chinese)

---

<a name="english"></a>
## English Version

A high-performance, real-time 3D N-body gravity simulator built with **C++20** and **SDL3**. This project simulates complex orbital mechanics, celestial collisions, and most notably, the **Roche Limit**—the critical threshold where a celestial body is "peeled" and torn apart by tidal forces.

### ✨ Key Features
* **Directional Tidal Disintegration (Vector Biasing)**: Unlike random explosions, fragments are "peeled" with a bias toward the massive attractor, creating realistic tidal streams.
* **OpenMP-Powered Physics**: Multi-core parallel computing for $O(N^2)$ gravity interactions.
* **Momentum-Preserving Mergers**: Inelastic collisions that strictly conserve mass and momentum.
* **3D Camera & Flight Engine**: 6-DOF controls with target tracking (Lock-on) and real-time perspective scaling.

### 🎮 Controls
| Key | Action |
| :--- | :--- |
| **W/S/A/D/Space/L-Shift** | 6-DOF Movement (Fly) |
| **TAB** | Toggle Pause |
| **T** | Open Console (Paste data) |
| **F** | (Paused) Lock Camera to Target |
| **Arrow Up/Down** | Scale Time Step ($dt$) |
| **0 (Zero)** | Save State to `Running_output.txt` |

### 🚀 Compilation
```bash
g++ -O3 -fopenmp main.cpp -o stellar_sim -lSDL3
