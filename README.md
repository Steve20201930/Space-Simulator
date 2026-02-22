# Gravity-System & Starship Simulator 🚀

An ultra-hardcore, high-performance N-Body gravity simulation engine built with **SDL3** and **OpenMP**. It features a realistic spacecraft flight model based on SpaceX's **Starship** specifications, allowing you to simulate interplanetary missions from Earth to Mars.

## 🌟 Key Features

* **N-Body Physics Engine**: Real-time gravitational interaction between all celestial bodies using multi-threaded OpenMP acceleration.
* **Starship Flight Model**:
    * Parameters modeled after **SpaceX Raptor** engines (Thrust, Mass Flow, Fuel Consumption).
    * Dynamic Mass calculation: Your ship gets lighter and accelerates faster as fuel is depleted.
* **Orbital Projection System**: Real-time trajectory prediction lines to help you visualize orbital paths and plan fuel-efficient maneuvers.
* **Astrophysical Phenomena**:
    * **Roche Limit Simulation**: Celestial bodies fragment and disintegrate when they cross the Roche limit of a larger mass.
    * **Elastic Merging**: Conservation of momentum and mass when planets collide.
* **Advanced Camera System**: 6-DOF movement with Target Locking and Synchronous Orbit tracking.

## 🛠 Controls

### General Controls
| Key | Action |
| :--- | :--- |
| `T` | Open Command Console |
| `P` / `O` | Increase / Decrease Camera Speed |
| `UP` / `DOWN` | Speed up / Slow down Simulation Time (dt) |
| `TAB` | Pause / Resume Physics |
| `ESC` | Release Mouse Cursor |

### SpaceShip Pilot Mode
| Key | Action |
| :--- | :--- |
| `Left Shift` | Throttle Up |
| `Left Ctrl` | Throttle Down |
| `W / S` | Pitch Down / Up |
| `A / D` | Yaw Left / Right |
| `Z` | MAX Throttle (100%) |
| `X` | Cutoff Thrust (0%) |
| `L` | Toggle Trajectory Prediction Line |

## 💻 Console Commands

Press `T` to enter the console and type:
* `spaceshipmode` : Toggle between Free Cam and Ship Pilot mode.
* `sel solar` : Instant load of the Solar System (Sun to Neptune + Moon).
* `target [Name]` : Set a celestial body as your navigation target.
* `goto [Name]` : Teleport camera to a specific planet.
* `newg [Value]` : Modify the Universal Gravitational Constant $G$.
* `roche` : Toggle the Roche Limit disintegration effect.
* `save` : Save the current universe state to `running.txt`.

## 🚀 Physics Specs (Starship Reference)

- **Dry Mass**: 120,000 kg
- **Single Raptor Thrust**: 2,745,000 N
- **Propellant Consumption**: 0.1 kg/s (per engine)(Real engine for 650 kg/s)
- **Engine Configuration**: 9x Raptor Engines
- **Theoretical $\Delta v$**: Calculated via the Tsiolkovsky Rocket Equation.



## 🔨 Build Instructions

### Dependencies
- **SDL3**: Latest development libraries.
- **Compiler**: GCC/Clang with OpenMP support.
- **C++ Standard**: C++17 or higher.

### Compile
```bash
g++ -O3 -std=c++17 main.cpp -o gravity_sim -lSDL3 -lm -fopenmp
```
# Gravity-System & SpaceShip Simulator 🚀

这是一个基于 **SDL3** 和 **OpenMP** 开发的高性能 N-Body 万有引力物理仿真系统。它不仅能模拟星系演化、罗氏极限（Roche Limit）下的天体解体，还内置了一个高度还原 SpaceX **Starship（星舰）** 参数的飞船驾驶模拟器。

## 🌟 核心特性

* **高性能 N-Body 模拟**：使用 OpenMP 多线程加速物理计算，支持数千个天体同时运行。
* **硬核飞船物理**：
    * 参考 SpaceX Raptor 发动机参数（推力、比冲、燃料消耗）。
    * 实时计算推重比（TWR）和质量变化。
* **轨道预测系统**：内置实时轨迹预测线，帮助你精准规划霍曼转移轨道（Hohmann Transfer）。
* **天体物理现象**：
    * **罗氏极限**：当卫星过近时会发生物理粉碎。
    * **天体合并**：支持完全弹性的质量与动量合并。
* **全 3D 摄像机**：自由视角、目标锁定以及同步轨道追踪模式。

## 🛠 控制说明

### 基础控制
| 按键 | 功能 |
| :--- | :--- |
| `T` | 开启控制台输入模式 |
| `P` / `O` | 加速 / 减速 摄像机移动速度 |
| `UP` / `DOWN` | 加速 / 减速 物理模拟步长 (dt) |
| `TAB` | 暂停 / 恢复 模拟 |
| `ESC` | 释放鼠标捕捉 |

### 飞船驾驶 (SpaceShip Mode)
| 按键 | 功能 |
| :--- | :--- |
| `Left Shift` | 增加油门 (Throttle Up) |
| `Left Ctrl` | 减少油门 (Throttle Down) |
| `W/A/S/D` | 调整飞船 俯仰 (Pitch) 和 偏航 (Yaw) |
| `Z` | 一键全开油门 (MAX Thrust) |
| `X` | 立即切断动力 (Cutoff) |
| `L` | 开启/关闭 轨迹预测线 (Trajectory) |

## 💻 控制台指令 (Console Commands)

在按下 `T` 后可以输入以下指令：
* `spaceshipmode` : 进入/退出飞船模式。
* `sel solar` : 快速加载太阳系预设（包含八大行星与月球）。
* `target [Name]` : 锁定特定天体为导航目标。
* `goto [Name]` : 瞬间移动摄像机至该天体。
* `newg [Value]` : 修改万有引力常数 $G$。
* `roche` : 开启/关闭罗氏极限粉碎效果。
* `save` : 保存当前系统状态到本地。

## 🚀 物理参数参考 (SpaceX Starship)

模拟器默认参考参数：
- **干重**: 120,000 kg
- **单台猛禽发动机推力**: 2,745,000 N
- **燃料消耗**: 0.1 kg/s 每台(实际为650kg/s)
- **配置**: 9 台真空/海平面引擎组合

## 🔨 编译要求

1.  **SDL3**: 确保系统中已安装最新的 SDL3 开发库。
2.  **OpenMP**: 编译器需支持 `-fopenmp`。
3.  **C++17** 或更高版本。

```bash
# 编译示例 (Linux/macOS)
g++ -O3 -std=c++17 main.cpp -o gravity_sim -lSDL3 -lm -fopenmp
