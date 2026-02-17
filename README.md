# Stellar-Roche-Sim 🪐

A high-performance, real-time 3D N-body gravity simulator built with **C++20** and **SDL3**. This project simulates complex orbital mechanics, celestial collisions, and most notably, the **Roche Limit**—the critical threshold where a celestial body is "peeled" and torn apart by tidal forces.



## ✨ Key Features

* **Directional Tidal Disintegration (Vector Biasing)**: 
    Unlike random explosions, this simulator uses a custom **Vector Biasing** algorithm. Fragments are "peeled" from the planet's surface with a bias toward the massive attractor, creating realistic tidal streams and accretion disk formations.
* **OpenMP-Powered Physics**: 
    Utilizes multi-core parallel computing to handle thousands of simultaneous gravitational interactions ($O(N^2)$ complexity) without stuttering.
* **Momentum-Preserving Mergers**: 
    An inelastic collision system that merges planets upon contact, strictly conserving mass and momentum while blending colors based on mass ratios.
* **3D Camera & Flight Engine**: 
    * **6-DOF Control**: Fly through the void with flight-simulator style movement.
    * **Target Tracking (Lock-on)**: Pause the simulation and press **F** to lock the camera to a planet, tracking its trajectory perfectly.
    * **Perspective Scaling**: Real-time adjustment of projection focal length for cinematic depth.
* **Numerical Stability**: 
    Implements a robust **Velocity Verlet** integration scheme to ensure energy conservation and stable orbits even at varying time steps.

## 🛠️ Tech Stack

* **Language**: C++20 (STL, Math)
* **Graphics**: SDL3 (Direct Point & Scanline Rendering)
* **Parallelism**: OpenMP
* **Physics**: Newtonian Gravitation + Roche Limit Theory

## 🎮 Controls

| Key | Action |
| :--- | :--- |
| **W / S / A / D** | Fly Forward / Backward / Strafe Left / Right |
| **Space / L-Shift** | Fly Up / Down |
| **TAB** | Toggle Simulation Pause |
| **T** | Open Console (Paste commands/data from clipboard) |
| **F** | (While Paused) Lock Camera to Focus Target |
| **Arrow Up / Down** | Scale Time Step ($dt$) |
| **Arrow Left / Right** | Adjust Projection/Focal Length |
| **O / P** | Adjust Camera Movement Sensitivity |
| **0 (Zero)** | Save Current State to `Running_output.txt` |
| **ESC** | Unlock Mouse / Close Console |

## 💻 Developer Console (T-Mode)

The simulation supports real-time command injection. Press **T**, then type or paste:
* `-3`: Toggle Pause.
* `-4`: Toggle Roche Limit disintegration.
* `-5 [value]`: Set Roche Constant $K$ (Default: 1.26).
* `-6 [value]`: Set Gravitational Constant $G$.
* `[N] [Data...]`: Batch spawn $N$ planets using the standard data format.

## 🚀 Getting Started

### Prerequisites
* A C++ compiler with **OpenMP** support (GCC 11+ or Clang).
* **SDL3** development libraries.

### Compilation
Compile using the following command:

```bash
g++ -O3 -fopenmp main.cpp -o stellar_sim -lSDL3
