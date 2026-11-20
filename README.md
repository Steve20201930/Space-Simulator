
## Realtime Shader
<img width="3248" height="1996" alt="Screenshot 2026-07-29 at 15 22 43" src="https://github.com/user-attachments/assets/de0d43c8-0a80-420f-9fc9-eaac44da7087" />
<img width="3248" height="1998" alt="Screenshot 2026-08-29 at 17 31 55" src="https://github.com/user-attachments/assets/d6a3206d-b8fc-42ef-91b9-22ffa85227b1" />
<img width="3248" height="1996" alt="Screenshot 2026-07-29 at 15 32 20" src="https://github.com/user-attachments/assets/1421d75a-f0a2-4850-85a4-f9e394ddeb49" />
<img width="3248" height="1994" alt="Screenshot 2026-03-02 at 22 09 57" src="https://github.com/user-attachments/assets/19f8d090-0de4-469e-be92-0266c44138db" />
<img width="3248" height="1998" alt="Screenshot 2026-08-29 at 22 36 34" src="https://github.com/user-attachments/assets/91f143ae-fd84-458d-9fe9-7a44817c9a09" />
<img width="3248" height="1996" alt="Screenshot 2026-05-02 at 18 02 22" src="https://github.com/user-attachments/assets/b3e1bdad-c0ce-42e6-8504-e0de381b0844" />
<img width="1624" height="998" alt="Screenshot 2026-11-20 at 22 12 45" src="https://github.com/user-attachments/assets/d7c8874f-174f-4a97-9c0e-ffecc288e1f6" />
<img width="1624" height="998" alt="Screenshot 2026-11-20 at 22 12 50" src="https://github.com/user-attachments/assets/8f162ac8-bc5e-44b9-be20-a583eb79c502" />




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
solar3 : Includes the moons of the primary planet, Saturn's rings, and the asteroid belt. Only recommended for the powerful computer to load!!!
J2000 : January 1, 2000, 11:58:55.816 Basic Star and Planetary Positions.
milkyway1 : Lite Milky Way simulation, only includes important stars. Includes 18,437 stars.
milkyway2 : Medium Milky Way simulation, almost includes all the stars. Includes 73,135 stars. (Recommended for all the users).
milkyway3 : Full Milky Way simulation, includes all stars. Includes 451,264 stars. (Only recommended for the super computer).
```
* `target [Name]` : Lock onto a body and move relative to its frame.
* `goto [Name]` : Teleport camera to a specific celestial body.
* `newg [Value]` : Modify the Universal Gravitational Constant $G$.
* `roche` : Toggle the Roche Limit fragmentation effect.
* `save` : Save the current system state locally.
* `newrochek [Value]` : Modify Roche Limit coefficient (Default: 1.26).
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
Metal : Use Metal which in M series chip to compute
CUDA : Use NVIDIA GPU which supported CUDA to compute
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
solar3：包含主行星的卫星、土星环和小行星带。仅推荐高性能电脑加载！！！
J2000：2000年1月1日 11:58:55.816 基本恒星和行星位置。
milkyway1 : 简单模拟银河系, 仅包含一些重要恒星。包含 18,437 颗恒星与行星。
milkyway2 : 中度模拟银河系，几乎包含所有恒星与行星。包含 73,135 颗恒星与行星. （推荐所有用户均可尝试）。
milkyway3 : 完全模拟银河系，包含所有恒星与行星。包含 451,264 颗恒星与行星. （仅推荐超级电脑尝试）。
```
* `target [Name]` : 锁定特定天体并进入其内部跟随其移动。
* `goto [Name]` : 瞬间移动摄像机至该天体。
* `newg [Value]` : 修改万有引力常数 $G$。
* `roche` : 开启/关闭洛希极限粉碎效果。
* `save` : 保存当前系统状态到本地。
* `newrochek [Value]` : 修改洛希极限参数,默认为1.26
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
Metal : 使用M芯片的Metal计算
CUDA : 使用支持CUDA的英伟达显卡计算
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

## These are some test data simple
```bash
StarA  -80   0    0   -0.9   0   0   500   6   255 200 50 255
StarB   80   0    0    0.9   0   0   500   6    50 200 255 255
StarC    0 120   -1.2  0     0   0   500   6   255  80  80 255
Planet   0 -220   1.8  0     0   0     1   3   200 200 200 255

10
Sun      0    0  0  0   0  1    0 0 0 10000 18 255 200 50 255

Mercury  40   0  0  0     15.8  0  0 0 0 1     3  200 200 200 255
Venus    70   0 0   0     11.95  0 0 0 0  2     4  255 180 120 255
Earth    100  0   0  0     10   0    0 0 0 2     4  100 150 255 255
Mars     140  0  0  0     8.45 0  0 0 0  1.5   4  255 100 100 255

Jupiter  200  0  0  0     7.07 0  0 0 0  50    9  255 200 150 255
Saturn   280  0  0  0     5.98  0 0 0 0  40    8  220 200 160 255
Uranus   360  0 0   0      5.27 0  0 0 0  20      7  150 220 255 255
Neptune  440  0  0  0     4.77 0  0 0 0  20    7  100 100 255 255
Moon 100.26   0  0  0     12.77 0   0  0  0  0.025  1.08  200 200 200 255
A        100  0    100    0   0    8.4   0     0   0   4    5  255 255 255 255

Star 1000 1000 -100      -100       0  0     1000    0.2  255 255 255 255

A  10    0  400  0   0  -10     0 0 0 20000 18 255 200 50 255
43

A  0    0  0  0   0  0     0 0 0 20000 18 255 200 50 255
D  100  100  -100  -4.38 -7.38 -4.38 0 0 0 100 9 255 255 255 255
B -400 300 200 0 0 -1 0 0 0 10000 16 100 0 100 255


P0   150 0 0   0 8.165 0   0 0 0  0.01 1 180 220 255 255
P1   147 26 0  -1.415 8.02 0   0 0 0  0.01 1 180 220 255 255
P2   140 51 0  -2.775 7.63 0   0 0 0  0.01 1 180 220 255 255
P3   128 75 0  -4.085 7.01 0   0 0 0  0.01 1 180 220 255 255
P4   112 96 0  -5.225 6.17 0   0 0 0  0.01 1 180 220 255 255
P5   93 115 0  -6.27 5.07 0   0 0 0  0.01 1 180 220 255 255
P6   71 132 0  -7.18 3.86 0   0 0 0  0.01 1 180 220 255 255
P7   47 143 0  -7.78 2.56 0   0 0 0  0.01 1 180 220 255 255
P8   22 148 0  -8.07 1.2 0   0 0 0  0.01 1 180 220 255 255
P9   -4 150 0  -8.17 -0.22 0   0 0 0  0.01 1 180 220 255 255
P100  160   0    0    0     7.906  0    0 0 0  0.01 1 180 220 255 255
P101  157   33   0   -1.63  7.75   0    0 0 0  0.01 1 180 220 255 255
P102  148   64   0   -3.12  7.30   0    0 0 0  0.01 1 180 220 255 255
P103  134   92   0   -4.48  6.62   0    0 0 0  0.01 1 180 220 255 255
P104  115  115   0   -5.68  5.68   0    0 0 0  0.01 1 180 220 255 255
P105   92  134   0   -6.62  4.48   0    0 0 0  0.01 1 180 220 255 255
P106   64  148   0   -7.30  3.12   0    0 0 0  0.01 1 180 220 255 255
P107   33  157   0   -7.75  1.63   0    0 0 0  0.01 1 180 220 255 255
P108    0  160   0   -7.91  0      0    0 0 0  0.01 1 180 220 255 255
P109  -33  157   0   -7.75 -1.63   0    0 0 0  0.01 1 180 220 255 255
P110  -64  148   0   -7.30 -3.12   0    0 0 0  0.01 1 180 220 255 255
P111  -92  134   0   -6.62 -4.48   0    0 0 0  0.01 1 180 220 255 255
P112 -115  115   0   -5.68 -5.68   0    0 0 0  0.01 1 180 220 255 255
P113 -134   92   0   -4.48 -6.62   0    0 0 0  0.01 1 180 220 255 255
P114 -148   64   0   -3.12 -7.30   0    0 0 0  0.01 1 180 220 255 255
P115 -157   33   0   -1.63 -7.75   0    0 0 0  0.01 1 180 220 255 255
P116 -160    0   0    0    -7.91   0    0 0 0  0.01 1 180 220 255 255
P117 -157  -33   0    1.63 -7.75   0    0 0 0  0.01 1 180 220 255 255
P118 -148  -64   0    3.12 -7.30   0    0 0 0  0.01 1 180 220 255 255
P119 -134  -92   0    4.48 -6.62   0    0 0 0  0.01 1 180 220 255 255
P120 -115 -115   0    5.68 -5.68   0    0 0 0  0.01 1 180 220 255 255
P121  -92 -134   0    6.62 -4.48   0    0 0 0  0.01 1 180 220 255 255
P122  -64 -148   0    7.30 -3.12   0    0 0 0  0.01 1 180 220 255 255
P123  -33 -157   0    7.75 -1.63   0    0 0 0  0.01 1 180 220 255 255
P124    0 -160   0    7.91  0      0    0 0 0  0.01 1 180 220 255 255
P125   33 -157   0    7.75  1.63   0    0 0 0  0.01 1 180 220 255 255
P126   64 -148   0    7.30  3.12   0    0 0 0  0.01 1 180 220 255 255
P127   92 -134   0    6.62  4.48   0    0 0 0  0.01 1 180 220 255 255
P128  115 -115   0    5.68  5.68   0    0 0 0  0.01 1 180 220 255 255
P129  134  -92   0    4.48  6.62   0    0 0 0  0.01 1 180 220 255 255

19
Sun 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 333000.00 18.00 255 200 50 255
Mercury 40.00 0.00 0.00 0.00 0.00 91.24 0.00 0.00 0.00 0.05 1.50 160 160 160 255
Venus 70.00 0.00 0.00 0.00 0.00 68.96 0.00 0.00 0.00 0.81 3.50 255 198 107 255
Earth 100.00 0.00 0.00 0.00 0.00 57.70 0.00 0.00 0.00 1.00 4.00 100 150 255 255
Moon 100.38 0.00 0.00 0.00 0.00 59.32 0.00 0.00 0.00 0.01 1.00 200 200 200 255
Mars 150.00 0.00 0.00 0.00 0.00 47.11 0.00 0.00 0.00 0.11 2.50 255 100 100 255
Phobos 150.15 0.00 0.00 0.00 0.00 47.96 0.00 0.00 0.00 0.001 0.50 150 150 150 255
Deimos 150.25 0.00 0.00 0.00 0.00 47.77 0.00 0.00 0.00 0.001 0.40 180 180 180 255
Jupiter 250.00 0.00 0.00 0.00 0.00 36.49 0.00 0.00 0.00 317.00 10.00 255 200 150 255
Io 252.00 0.00 0.00 0.00 0.00 49.08 0.00 0.00 0.00 0.01 1.20 255 255 100 255
Europa 253.50 0.00 0.00 0.00 0.00 45.42 0.00 0.00 0.00 0.01 1.10 200 200 255 255
Ganymede 255.50 0.00 0.00 0.00 0.00 44.08 0.00 0.00 0.00 0.02 1.50 150 150 150 255
Callisto 258.00 0.00 0.00 0.00 0.00 42.14 0.00 0.00 0.00 0.02 1.40 100 100 100 255
Saturn 400.00 0.00 0.00 0.00 0.00 28.85 0.00 0.00 0.00 95.00 8.50 220 200 160 255
Titan 405.00 0.00 0.00 0.00 0.00 33.21 0.00 0.00 0.00 0.02 1.80 255 180 0 255
Uranus 600.00 0.00 0.00 0.00 0.00 23.55 0.00 0.00 0.00 14.50 7.00 150 220 255 255
Titania 602.50 0.00 0.00 0.00 0.00 25.96 0.00 0.00 0.00 0.01 1.00 200 200 200 255
Neptune 800.00 0.00 0.00 0.00 0.00 20.40 0.00 0.00 0.00 17.00 7.00 100 100 255 255
Triton 803.00 0.00 0.00 0.00 0.00 22.34 0.00 0.00 0.00 0.01 1.20 220 220 255 255

2
S_A 0  0   0  -1  0  0  0  0   0  10000   30  255 200 50 255
S_B  -200 0  400  2 0 0 0  0  0  10000  26  100 150 255 255

10
Sun      0      0  0  0   0      0  0 0 0  10000    18      255 200 50 255
Mercury  39     0  0  0   16.03  0  0 0 0  0.00166  0.063   200 200 200 255
Venus    72     0  0  0   11.78  0  0 0 0  0.0245   0.156   255 180 120 255
Earth    100    0  0  0   10.00  0  0 0 0  0.0300   0.165   100 150 255 255
Mars     152    0  0  0   8.11   0  0 0 0  0.0032   0.088   255 100 100 255
Jupiter  520    0  0  0   4.38   0  0 0 0  9.54     1.808   255 200 150 255
Saturn   954    0  0  0   3.24   0  0 0 0  2.86     1.503   220 200 160 255
Uranus   1919   0  0  0   2.28   0  0 0 0  0.437    0.655   150 220 255 255
Neptune  3007   0  0  0   1.82   0  0 0 0  0.515    0.635   100 100 255 255
Moon     100.26 0  0  0   11.02  0  0 0 0  0.00037  0.045   200 200 200 255

-1517 607 -2867 0.18 5.5
2
BlackHole  0   0    0   0    0     0     0     0    0   100000  400 255 255 255  255 1
A  -2000   2000  0  5.5 0  0  0  0   0    100   200  255 200 50 255 0

Tes 0 0 0 0 0 0 0 0 0 100000 400 50 150 200 255 1
K 800 0 0 0 0 0 


Solar System Completely
10
Sun 0 0 0 0 0 0 0 0 0 1.9885e30 695700000 255 210 26 255 1
Mercury 6.98179e10 0 0 0 0 47870 0 0 0 3.3022e23 2440000 194 194 194 255 0
Venus 1.08939e11 0 0 0 0 35020 0 0 0 4.8676e24 6051000 255 236 59 255 0
Earth 1.521e11 0 0 0 0 29780 0 0 0 5.97237e24 6.371e6 28 115 232 255 0
Mars 2.49261e11 0 0 0 0 24080 0 0 0 6.4169e23 3.3895e6 245 91 22 255 0
Jupiter 8.165208e11 0 0 0 0 13070 0 0 0 1.8981e27 6.9886e7 255 213 125 255 0
Saturnus 1.353572956e12 0 0 0 0 9690 0 0 0 5.6846e26 5.8232e7 219 167 61 255 0
Uranus 3.004419704e12 0 0 0 0 6810 0 0 0 8.6810e25 2.5559e7 215 248 250 255 0
Neptune 4.55394649e12 0 0 0 0 5430 0 0 0 1.0243e26 2.4764e7 65 105 225 255 0

Moon 1.51736896e11 0 0 0 0 30802 0 0 0 7.3477e22 1.7371e6 194 194 194 255 0

MarsI  2.4925188e11 0 0 0 0 26229.2183 0 0 0 1.8e15 1.11e4  160 160 160 255 0
MarsII 2.4923754e11 0 0 0 0 25428.3773 0 0 0 1.48e15 6.2e3 163 163 163 255 0

J1 8.160991e11 0 0 0 0 30366.991 0 0 0 8.94e22 1.8213e6 164 164 164 255 0
J2 8.158499e11 0 0 0 0 26783.347 0 0 0 4.799844e22 1.5608e6 165 165 165 255 0
J3 8.154504e11 0 0 0 0 23926.737 0 0 0 1.4819e23 2.6431e6 166 166 166 255 0
J4 8.146381e11 0 0 0 0 19771.37 0 0 0 1.075938e23 2.4103e6 168 168 168 255 0
J5 8.1633944e11 0 0 0 0 39445.13 0 0 0 2.08e18 8.35e4   170 170 170 255 0
J6 8.034388e11 0 0 0 0 16175.526 0 0 0 4.2e18 1.414e5 172 172 172 255 0
J7 8.034388e11 0 0 0 0 16348.221 0 0 0 8.7e17 4.3e4 174 174 174 255 0
