# 🌌 3D Gravity System with Roche Limit  
### 基于 SDL3 + OpenMP 的三维引力模拟系统

---

## 📖 Introduction | 项目介绍

### English

This project is a real-time **3D N-body gravity simulator** written in C++.  
It uses:

- SDL (for rendering and input handling)
- OpenMP (for parallel acceleration)

The simulator includes:

- Newtonian gravity
- Roche limit tidal disintegration
- Planet merging
- Free 3D camera system
- Interactive console command input
- Save system

It is designed as a physics sandbox and experimental gravity engine.

---

### 中文

这是一个使用 C++ 编写的实时 **三维 N 体引力模拟系统**。

项目使用：

- SDL（图形渲染与输入系统）
- OpenMP（并行计算加速）

系统支持：

- 牛顿万有引力
- 洛希极限潮汐解体
- 行星合并
- 自由三维摄像机
- 内置控制台命令输入
- 状态保存功能

这是一个偏物理模拟方向的引力沙盒系统。

---

# ✨ Features | 功能特性

### English

- Real-time N-body gravity simulation
- Roche limit tidal fragmentation
- Planet merging with momentum conservation
- OpenMP parallel acceleration
- 3D camera movement
- Perspective projection rendering
- Depth sorting
- Save simulation state to file
- Built-in command console

---

### 中文

- 实时 N 体引力模拟
- 洛希极限解体机制
- 动量守恒的行星合并
- OpenMP 并行加速
- 三维摄像机控制
- 透视投影渲染
- 深度排序绘制
- 状态保存功能
- 内置命令控制台

---

# 🧮 Physics Model | 物理模型

## Gravity | 万有引力

Newton’s law of gravitation:

F = G * m1 * m2 / r^2


Velocity Verlet integration is used for motion updates.

使用 Velocity Verlet 方法进行时间积分更新。

---

## Roche Limit | 洛希极限

Roche limit formula used:

R_roche = k * R * (2M/m)^(1/3)


When a body enters the Roche region:

- It may fragment into smaller particles
- Total momentum is conserved
- Mass gradually decreases

当进入洛希区域时：

- 行星会碎裂
- 动量守恒
- 质量逐渐损失

---

## Merging | 合并机制

If two bodies overlap:

- Mass is conserved
- Momentum is conserved
- Radius recalculated from volume conservation

当两个天体接触时：

- 质量守恒
- 动量守恒
- 半径按体积守恒重新计算

---

# 🎮 Controls | 操作方式

## Camera Movement | 摄像机控制

| Key | Function |
|------|----------|
| W / S | Move forward / backward |
| A / D | Move left / right |
| SPACE | Move up |
| LSHIFT | Move down |
| Mouse | Look around |
| TAB | Pause |
| F | Lock camera to selected body |
| LEFT / RIGHT | Adjust projection |

---

## Console Mode | 控制台模式

Press:

T


To enter console mode.

Press:

Shift + Enter


To execute command.

---

## Save State | 保存状态

Press:

0


Output file:


Running_output.txt


---

# ⚙ Build Instructions | 编译方式

## macOS / Linux

g++ main.cpp -o gravity -lSDL3 -fopenmp


## Windows (MinGW)

g++ main.cpp -o gravity.exe -lSDL3 -fopenmp


Make sure SDL3 development libraries are installed.

---

# 📁 Project Structure | 项目结构

main.cpp
README.md
Running_output.txt (generated)


---

# 🚀 Future Improvements | 未来改进方向

- Barnes-Hut optimization (O(n log n))
- GPU acceleration
- Better collision physics
- Real texture rendering
- UI overlay system
- Performance profiling tools

---

# 📜 License

This project is open for learning and personal research use.

---

# 👨‍💻 Author

Developed as a personal physics & rendering experiment.
