# Stellar-Roche-Sim 🪐

A high-performance, real-time 3D N-body gravity simulator built with **C++** and **SDL3**. This project simulates orbital mechanics, celestial collisions, and the **Roche Limit**—the threshold where a celestial body disintegrates due to tidal forces from a more massive object.



## ✨ Key Features

* **Dynamic Roche Limit Disintegration**: Planets realistically fragment into smaller pieces when they wander too close to massive objects (like black holes), simulating the formation of planetary rings.
* **OpenMP-Powered Physics**: Utilizes multi-core parallel computing to handle thousands of gravitational interactions ($O(N^2)$ complexity) efficiently.
* **Momentum-Preserving Mergers**: An inelastic collision system where planets merge upon contact, conserving mass and momentum.
* **3D Camera Engine**: 
    * **6-DOF Control**: Fly through your simulation using flight-simulator style controls.
    * **Perspective Projection**: Custom mathematical mapping from 3D space to 2D screen coordinates.
    * **Target Tracking**: Lock the camera onto a specific planet to follow its journey through the void.
* **Numerical Stability**: Uses an optimized Velocity Verlet integration method for reliable physics across various time steps.

## 🛠️ Tech Stack

* **Language**: C++20
* **Graphics**: SDL3 (Simple DirectMedia Layer)
* **Parallelism**: OpenMP
* **Math**: Custom 3D Projection & Newtonian Mechanics

## 🎮 Controls

| Key | Action |
| :--- | :--- |
| **W / S** | Move Forward / Backward |
| **A / D** | Strafe Left / Right |
| **Space / L-Shift** | Fly Up / Down |
| **TAB** | Toggle Simulation Pause |
| **Arrow Up / Down** | Scale Time Step ($dt$) |
| **O / P** | Adjust Camera Movement Speed |
| **F** | Lock Camera to Current Target |
| **ESC** | Release Mouse Cursor |

## 🚀 Getting Started

### Prerequisites

* A C++ compiler with **OpenMP** support (GCC or Clang recommended).
* **SDL3** development libraries.

### Compilation
<img width="1624" height="997" alt="Screenshot 2026-02-17 at 12 09 58" src="https://github.com/user-attachments/assets/8ad15031-ec47-491f-b5df-c9263e5ecdfd" />
<img width="1624" height="997" alt="Screenshot 2026-02-17 at 12 10 58" src="https://github.com/user-attachments/assets/6ab85ee2-23a1-42f4-9470-2b422f1450b2" />


Compile using the following command (adjust for your OS):

```bash
g++ -O3 -fopenmp main.cpp -o stellar_sim -lSDL3
