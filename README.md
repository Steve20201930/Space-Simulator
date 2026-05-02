# Gravity-System & SpaceShip Simulator 🚀

A high-performance N-Body gravitational physics simulation system built with **SDL3** and **OpenMP**. It simulates galactic evolution and celestial disintegration via the **Roche Limit**, and features a hardcore flight simulator based on SpaceX **Starship** specifications.

## 🌟 Core Features

* **High-Performance N-Body Simulation**: Physics calculations are accelerated using OpenMP multi-threading, supporting thousands of celestial bodies simultaneously.
* **Hardcore Spacecraft Physics**:
    * Modeled after **SpaceX Raptor** engine parameters (Thrust, Specific Impulse, Fuel Consumption).
    * Real-time calculation of Thrust-to-Weight Ratio (**TWR**) and dynamic mass depletion.
* **Orbital Prediction System**: Built-in trajectory projection to assist in precision **Hohmann Transfer** maneuvers.
* **Astrophysical Phenomena**:
    * **Roche Limit**: Satellites fragment and disintegrate when orbiting too close to a massive body.
    * **Celestial Merging**: Supports fully elastic merging of mass and momentum upon collision.
* **Full 3D Camera**: Supports Free Look, Target Locking, and Synchronous Orbit tracking modes.

## 🛠 Controls

### Basic Controls
| Key | Action |
| :--- | :--- |
| `T` | Open Command Console |
| `P` / `O` | Increase / Decrease Camera Movement Speed |
| `UP` / `DOWN` | Increase / Decrease Physics Time Step (dt) |
| `TAB` | Pause / Resume Simulation |
| `ESC` | Release Mouse Capture |

### SpaceShip Mode
| Key | Action |
| :--- | :--- |
| `I` | Throttle Up |
| `K` | Throttle Down |
| `W / S` | Pitch Down / Up |
| `A / D` | Yaw Left / Right |
| `C` | Automatically enters the Target planetary synchronous orbit along the tangent in the direction of the bow. |
| `Q` | Input Target Planet Name |
| `E` | Toggle Auto-Pointing to Target Planet |
| `Z` | Toggle Trajectory Prediction Line |
| `U` | Decrease trajectory lane |
| `O` | Increase trajectory lane |

## 💻 Console Commands

Press `T` to enter the console, then type:
* `spaceshipmode` : Toggle in/out of Starship pilot mode.
* `sel [Model]` : Instant load of the Model.
```bash
[Model]
solar : Include the basis solar system with 8 planet.
(<=v2.2.1)solarcompletely / (>=v2.2.2)solar2: Include the basis solar system and sapturnus ring. Only recommended for the powerful computer to load!!!
(>=v2.2.2)solar3 : Includes the moons of the primary planet, Saturn's rings, and the asteroid belt. Only recommended for the powerful computer to load!!!
(>=v2.2.2)J2000 : January 1, 2000, 11:58:55.816 Basic Star and Planetary Positions.
```
* `target [Name]` : Lock onto a body and move relative to its frame.
* `goto [Name]` : Teleport camera to a specific celestial body.
* `newg [Value]` : Modify the Universal Gravitational Constant $G$.
* `roche` : Toggle the Roche Limit fragmentation effect.
* `save` : Save the current system state locally.
* `newrocheK [Value]` : Modify Roche Limit coefficient (Default: 1.26).
* `newdnum [Value]` : Modify number of fragments upon disintegration (Default: 2).
* `newdsize [Value]` : Modify size of fragments (Default: 1.0).
* `debug` : Toggle Debug Mode.
* `clear` : Remove all planets.
* `clearmyship` : Reset all Starship parameters.
* `[Quantity]` : Input a planet with its full parameters. The standard format is: [Name] [x] [y] [z] [vx] [vy] [vz] [ax] [ay] [az] [Mass] [Radius] [R] [G] [B] [Transparency] [Is stellar?(true/false)]
* `shader` : Active/Inactive realtime shader
* `in/out [Method]` : Use different method to compute, default in CPU.(Enable/Disable)
```bash
[Method]
Metal : Use Metal to compute
```
* `autodt` : Enable/Disable time changing method
* `perform [Value]` :
```bash
[Value](default: 1)
1 : Reduce long distance shader computing.
2 : Normal.
3 : Overlay all stellar's light
```


## 🚀 Physics Specifications (SpaceX Starship)

Default simulation parameters:
- **Dry Mass**: 120,000 kg
- **Single Raptor Thrust**: 2,745,000 N
- **Fuel Consumption**: 0.1 kg/s per engine (Simulated; Real: 650 kg/s)
- **Configuration**: 9x Raptor Engines (Vacuum/Sea-level mix)

## 🔨 Build Requirements

1. **SDL3**: Ensure the latest SDL3 development libraries are installed.
2. **OpenMP**: Compiler must support the `-fopenmp` flag.
3. **C++17** or higher.

```bash
# Compilation Example (Linux/macOS)
g++ -O3 -std=c++17 main.cpp -o gravity_sim -lSDL3 -lm -fopenmp
```
# Gravity-System & SpaceShip Simulator 🚀

这是一个基于 **SDL3** 和 **OpenMP** 开发的高性能 N-Body 万有引力物理仿真系统。它不仅能模拟星系演化、洛希极限（Roche Limit）下的天体解体，还内置了一个高度还原 SpaceX **Starship（星舰）** 参数的飞船驾驶模拟器。

## 🌟 核心特性

* **高性能 N-Body 模拟**：使用 OpenMP 多线程加速物理计算，支持数千个天体同时运行。
* **硬核飞船物理**：
    * 参考 SpaceX Raptor 发动机参数（推力、比冲、燃料消耗）。
    * 实时计算推重比（TWR）和质量变化。
* **轨道预测系统**：内置实时轨迹预测线，帮助你精准规划霍曼转移轨道（Hohmann Transfer）。
* **天体物理现象**：
    * **洛希极限**：当卫星过近时会发生物理粉碎。
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
| `I` | 增加油门 (Throttle Up) |
| `K` | 减少油门 (Throttle Down) |
| `W/A/S/D` | 调整飞船 俯仰 (Pitch) 和 偏航 (Yaw) |
| `C` | 自动沿船头方向的切线进入Target行星同步轨道 |
| `Q` | 输入Target行星名称 |
| `E` | 锁定/解除 星舰自动指向行星 |
| `Z` | 开启/关闭 轨迹预测线 (Trajectory) |
| `U` | 缩短轨道预测线 |
| `O` | 延长轨道预测线 |

## 💻 控制台指令 (Console Commands)

在按下 `T` 后可以输入以下指令：
* `spaceshipmode` : 进入/退出飞船模式。
* `sel [Model]` : 快速加载预设。
```bash
[Model]
solar : 包含基础太阳系八大行星.
（<=v2.2.1）solarcompletely / （>=v2.2.2）solar2：包含基础太阳系和土星环。仅推荐高性能电脑加载！！！
（>=v2.2.2）solar3：包含主行星的卫星、土星环和小行星带。仅推荐高性能电脑加载！！！
（>=v2.2.2）J2000：2000年1月1日 11:58:55.816 基本恒星和行星位置。
```
* `target [Name]` : 锁定特定天体并进入其内部跟随其移动。
* `goto [Name]` : 瞬间移动摄像机至该天体。
* `newg [Value]` : 修改万有引力常数 $G$。
* `roche` : 开启/关闭洛希极限粉碎效果。
* `save` : 保存当前系统状态到本地。
* `newrocheK [Value]` : 修改洛希极限参数,默认为1.26
* `newdnum [Value]` : 修改行星解体数量,默认为2
* `newdsize [Value]` : 修改行星解体后碎片尺寸,默认为1.0
* `debug` : 开启/关闭Debug模式
* `clear` : 清空全部行星
* `clearmyship` : 清空星舰所有参数
* `[Quantity]` : 输入行星以及各种参数,标准格式为: [名称] [x] [y] [z] [vx] [vy] [vz] [ax] [ay] [az] [质量] [半径] [R] [G] [B] [透明度] [是否为恒星(true/false)]
* `shader` : 开启或关闭实时光影
* `in/out [method]` : 使用不同方法计算,默认在CPU计算(开启/关闭)
```bash
[method]
Metal : 使用Metal计算
```
* `autodt` : 开启/关闭 自动时间管理
* `perform [Value]` :
```bash
[Value](默认值：1)
1：减少长距离着色器计算。
2：正常。
3：叠加所有恒星的光线
```

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
```

##Realtime Shader
<img width="1624" height="997" alt="Screenshot 2026-02-26 at 13 40 12" src="https://github.com/user-attachments/assets/d958744d-a707-4fa5-b22c-8a0382825ed4" />
<img width="3248" height="1996" alt="Screenshot 2026-05-02 at 17 51 23" src="https://github.com/user-attachments/assets/74107401-bdd9-4030-908a-432c15c21165" />
<img width="3248" height="1994" alt="Screenshot 2026-03-02 at 22 09 57" src="https://github.com/user-attachments/assets/19f8d090-0de4-469e-be92-0266c44138db" />
<img width="3248" height="1996" alt="Screenshot 2026-05-02 at 17 27 51" src="https://github.com/user-attachments/assets/dca73026-8b28-4366-bddb-e01b9e7c8543" />
<img width="3248" height="1996" alt="Screenshot 2026-05-02 at 18 02 22" src="https://github.com/user-attachments/assets/859e0d08-67fe-47ae-bb55-e9dedac41f19" />

