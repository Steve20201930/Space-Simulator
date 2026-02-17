# 🌌 3D N-Body Gravity Engine  
## With Roche Limit & Parallel Acceleration  
### 三维 N 体引力引擎（含洛希极限与并行加速）

---

# 🔬 Abstract | 摘要

### English

This project implements a real-time three-dimensional N-body gravity simulation engine written in C++.  
The system models Newtonian gravitational interaction, tidal disintegration based on Roche limit theory, and momentum-conserving merging behavior.

The simulation integrates physics computation, OpenMP parallel acceleration, and real-time rendering using SDL.

It serves as:

- A physics sandbox
- A computational astrophysics experiment
- A rendering & simulation engine prototype

---

### 中文

本项目实现了一个基于 C++ 的实时三维 N 体引力模拟引擎。  
系统包含牛顿万有引力模型、基于洛希极限的潮汐解体机制，以及动量守恒的行星合并机制。

本项目融合了：

- 数值物理计算
- OpenMP 并行加速
- SDL 实时图形渲染

可作为：

- 物理沙盒系统
- 计算天体物理实验平台
- 图形与模拟引擎原型

---

# 🎥 Demo | 演示效果

## Rendering Example

<img width="1624" height="997" alt="Screenshot 2026-02-17 at 16 41 47" src="https://github.com/user-attachments/assets/dbd03f8b-b3a5-4322-9ba5-3f3d4463f8a2" />
<img width="1624" height="997" alt="Screenshot 2026-02-17 at 16 42 40" src="https://github.com/user-attachments/assets/e75635b7-5c24-44ef-ada3-671115fa6140" />
<img width="1624" height="997" alt="Screenshot 2026-02-17 at 12 10 58" src="https://github.com/user-attachments/assets/143990d6-d0a4-49f9-a68c-88c4c0148dc0" />

---

# ✨ Core Features | 核心功能

### Simulation

- Real-time N-body gravitational interaction
- Velocity Verlet integration
- Roche limit tidal fragmentation
- Momentum & mass conserving merge system
- Dynamic particle generation

### Rendering

- 3D camera system (yaw/pitch)
- Perspective projection
- Depth sorting
- Real-time rendering pipeline

### System

- OpenMP parallel force calculation
- Console command execution
- State save system
- Adjustable physics parameters

---

# 🧮 Physical Model | 物理模型

## 1️⃣ Newtonian Gravity

\[
F = G \frac{m_1 m_2}{r^2}
\]

Acceleration:

\[
a = \frac{F}{m}
\]

Integration method:

Velocity Verlet scheme.

---

## 2️⃣ Roche Limit Model

\[
R_{roche} = k R \left(\frac{2M}{m}\right)^{1/3}
\]

When:

\[
r < R_{roche}
\]

The body undergoes fragmentation.

Fragment properties:

- Directional dispersion
- Partial escape velocity
- Momentum conservation

---

## 3️⃣ Collision & Merging

When:

\[
r < R_1 + R_2
\]

New body:

- Mass conserved
- Momentum conserved
- Volume conserved

Radius:

\[
R = (R_1^3 + R_2^3)^{1/3}
\]

---

# 📊 Algorithm Complexity | 算法复杂度分析

Let N be number of particles.

### Force Computation

Double loop:

\[
O(N^2)
\]

Parallelized using OpenMP.

---

### Sorting (Depth Rendering)

\[
O(N \log N)
\]

---

### Fragmentation

Worst case additional:

\[
O(N)
\]

---

### Total Per Frame Complexity

\[
O(N^2)
\]

Future optimization suggestion:

- Barnes-Hut Tree → O(N log N)
- GPU acceleration
- Spatial partitioning

---

# ⚙ Controls | 操作方式

## Camera

| Key | Action |
|------|--------|
| W/S | Forward / Backward |
| A/D | Left / Right |
| SPACE | Up |
| LSHIFT | Down |
| Mouse | Look |
| TAB | Pause |
| F | Lock camera |
| ← / → | Adjust projection |

---

## Console

Press:

```
T
```

Execute:

```
Shift + Enter
```

---

# 🏗 System Architecture | 系统架构

```
Particle System
    ├── Position Update
    ├── Force Calculation (OpenMP)
    ├── Roche Check
    ├── Merge Check
    └── State Update

Rendering Pipeline
    ├── Camera Transform
    ├── Projection
    ├── Depth Sort
    └── Draw Circle Rasterization
```

---

# 🧠 Scientific Perspective | 科研视角说明

This engine demonstrates:

- Emergent behavior in multi-body gravitational systems
- Non-linear instability in tidal interactions
- Momentum conservation under discrete fragmentation
- Real-time computational astrophysics modeling

Potential extensions:

- Softening length modeling
- Relativistic correction
- Accretion disk simulation
- Multi-star system stability study

---

# 🚀 Build Instructions | 编译方式

## macOS / Linux

```
g++ main.cpp -o gravity -lSDL3 -fopenmp
```

## Windows (MinGW)

```
g++ main.cpp -o gravity.exe -lSDL3 -fopenmp
```

Make sure SDL3 development library is installed.

---

# 📁 Project Structure

```
main.cpp
README.md
demo/
Running_output.txt
```

---

# 🔭 Future Work | 未来方向

- Barnes-Hut optimization
- CUDA / Metal GPU acceleration
- True 3D sphere rendering
- GUI overlay
- Performance benchmarking
- Large-scale simulation support

---

# 📜 License

Educational & Research Use.

---

# 👨‍💻 Author

Independent physics engine experiment project.
