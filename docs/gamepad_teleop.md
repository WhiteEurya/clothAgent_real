# 手柄连续遥操（默认 Arm6）

独立入口 `python -m cloth_agent.gamepad_teleop` 默认只打印输入，不连接机器人。
`--real` 默认控制 Arm6（left、6 轴、`192.168.2.232`）；只有显式指定
`--arm arm7 --real` 才控制 Arm7。采集入口 `gamepad_collect` 固定使用 Arm6。

```bash
conda activate cali
python -m cloth_agent.gamepad_teleop --list
python -m cloth_agent.gamepad_teleop                 # 输入预览
python -m cloth_agent.gamepad_teleop --real          # Arm6 独立遥操
python -m cloth_agent.gamepad_collect --real --iterations 10
```

多个设备时用 `--device /dev/input/eventN` 指定，优先使用稳定的
`/dev/input/by-id/*-event-joystick` 路径。`--real --connect-only` 只检查连接，
不切换模式、不恢复 STOP、不发送运动或夹爪指令。

## 当前 USB Gamepad 的定义

配置文件：`config/gamepads/0003_0810_0001_0110.json`。
左摇杆轴为 ABS_X / ABS_Y，两个轴 sign 都为 -1；右摇杆左右轴为代码 5
（ABS_RZ），sign 为 +1。实体键映射 A=290、B=289、X=291、Y=288、LB=292、START=297。

| 操作 | 当前已核定手柄的动作 |
|---|---|
| 按住 LB | 允许移动和夹爪操作 |
| 左摇杆前推 / 后拉 | World −X / +X |
| 左摇杆右推 / 左推 | World +Y / −Y |
| 右摇杆右推 / 左推 | 绕 World Z 负向 / 正向 yaw |
| Y / A | World +Z 上升 / −Z 下降 |
| 按住 X / B | 逐渐开爪 / 合爪，松开保持 |
| 右摇杆上下 | 未映射 |

这是固定 World 坐标系，不随夹爪朝向或相机画面旋转。Arm6 使用自己的
`world_from_base` 和六轴 FK/Jacobian 转换速度；换机械臂不自动翻转手柄轴。
无设备配置的标准 Xbox 默认方向是前推 World +X、右推 World −Y，
与本机已保存的 USB 手柄反向配置不同。

启动先松开全部按键、摇杆回中，再按 LB。松开 LB 立即请求零速度和夹爪保持，
重新按住可继续。Y/A 同按不升降，X/B 同按保持夹爪。先于 LB 按住的夹爪键
需要松开再按。新手柄可运行 `bash scripts/calibrate_gamepad.sh`，已有映射可运行
`bash scripts/check_gamepad.sh`；二者只读输入，不连接机械臂。

## START 与退出

- 独立 `gamepad_teleop`：START 停止并退出。
- `gamepad_collect`：松开全部控件后单独按 START，确认抓取抬升完成并交回自动搬运；Ctrl+C 中止。

Arm6 正常软件 STOP 后，重新启动实机控制可自动恢复位置模式和使能。
恢复要求控制器无错误/警告、无排队运动；不会自动清除实体急停或真实故障。
独立遥操不自动 Home、拍照或移动到 perception。

## 控制实现

默认平移 10 mm/s、yaw 5°/s、Arm6 夹爪 SDK 速度 2000，可以分别通过
`--speed-mm-s`、`--yaw-speed-deg-s`、`--gripper-speed` 调整。
`--gripper-rate` 保留为别名；Arm6 中它现在表示 SDK 速度，范围 (0,5000]。
Arm6 夹爪配置为开 850 pulse、合 5 pulse。按住 X/B 时只在按下或换向时发送一次开/合终点目标，持续按住不反复改写位置，避免 12 pulse 小步引起的反复启停。松开 X/B 或 LB 后发送实测开度保持；换向立即发送另一端目标。输入超时会请求夹爪保持，START 交接和退出也会保持。
若夹爪内部因受阻停止，不会持续重发闭合目标；需松键后重新操作。
完整夹爪位置目标没有机械臂速度指令的 150 ms 到期机制，主机或通信完全失效时无法保证撤回已下发目标。
控制使用同一连续速度实现、20 ms 目标周期、150 ms 命令有效期，以及各自的关节数和限位。
Arm6 松开移动控件（包括只操作夹爪）后先发零速度，等待命令有效期及通信余量，再切回位置保持模式；恢复移动时重新进入速度模式。位置保持期间持续比较实测关节和 TCP，偏移超过 0.1° 关节角、1 mm 位置或 0.2° 姿态即请求 STOP。零速度指令返回成功不再单独作为已停稳的依据。

按当前用户要求，Arm6 采集及独立 Arm6 遥操不运行路径/几何碰撞验证，也不连接 Arm7。
保留 IK、关节限位、速度限制、输入超时、反馈新鲜度、控制器故障和模式检查。
显式选择 Arm7 的原入口保持原有验证策略。

`continuous_config.json` 记录控制机械臂、IP、轴数、World 坐标系、速度和验证开关；
`controller/events.jsonl` 记录速度、夹爪与失败事件。现场运动方向与停止行为需实机验证。

Arm6 手柄下降增益为 1.5：默认下降目标速度 15 mm/s，上升仍为 10 mm/s。组合平移速度不超过 15 mm/s，实际运动仍受关节速度和加速度限制。

Arm6 当前默认手动平移/上升 20 mm/s、下降 30 mm/s（1.5 倍增益），合成线速度
封顶 30 mm/s；实际受关节速度/加速度约束。夹爪 SDK speed 2000。
Arm7 原有速度设置不变。
