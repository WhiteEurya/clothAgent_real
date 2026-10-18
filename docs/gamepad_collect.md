# Arm6 半自动示教轨迹采集

入口：`python -m cloth_agent.gamepad_collect`。使用 Camera A 和 Arm6；仅连接 Arm6，Arm7 无需在线，不读取其状态、不发送任何命令。兼容数据结构中的 Arm7 使用 `data/robot/xarm7_perception_pose.json` 保存的姿态作为静态几何，不代表实时位置。需要项目 `cali` 环境、可显示窗口的桌面（Tk）、相机及已完成映射核定的手柄。

```bash
conda activate cali
python -m cloth_agent.gamepad_collect --real --iterations 10
```

多个手柄时增加 `--device /dev/input/eventN`。默认输出到 `data/demonstrations/<会话时间>/traj_0000/`，每个 iter 先写入 `pending/iter_0000/` 临时目录。回到 perception 后弹出保存确认，选择“是”才移入正式 `traj_0000/`；选择“否”或关闭窗口则保留临时数据并标记 REJECTED，不计为正式 traj。不覆盖旧数据。未指定 `--real` 不连接硬件。

每次迭代：

1. 到已保存的 perception 位姿，用 Camera A 拍摄并保存 RGB 与 World XYZ。
2. 在照片窗口依次点击 **start** 和 **target**，Enter 确认；R 重选；Esc 中止。深度无效或局部深度跳变时拒绝选点，要求重新点击。
3. 对 start 附近的终点求 IK，再通过直接关节插值移动到 start 上方（不使用 RRT，不做路径碰撞验证），默认距离所选表面 50 mm。初始姿态默认采用保存的 Arm6 Home 姿态在 World 中的朝向。
4. 使用原有手柄映射：按住 LB，左摇杆平移、右摇杆左右 yaw、Y/A 升降、X/B 开合。手动完成最后接近、抓取和抬升。
5. **松开全部按键和摇杆，再单独按 START**，确认抓取成功并交回自动搬运。此采集入口中 START 表示阶段完成；原有 `gamepad_teleop` 的 START 停止语义保持不变。控件未松开时按 START 会中止。Ctrl+C 可中止采集。
6. 搬运高度由操作者决定，不再要求高于 start 表面固定距离，也不自动抬高。若交接高度低于 target 的放置高度，留在手柄阶段等待调整。交接时清零速度、保持夹爪、检查静止反馈，然后恢复位置模式。保持交接时的高度、朝向和夹爪开度，水平移动到 target 上方，下降到 target 表面上方 5 mm，先增加当前夹爪开度 180 pulse，到位后等待 0.5 秒让布松脱，再竖直抬起（默认 30 mm），保持小幅开度移动至 Home，到位后完全打开夹爪，再回 perception。
7. 弹窗确认保存，确认后才写入正式 traj，开始下一次迭代。IK、输入、控制器或记录异常会中止会话，保留失败轨迹，不自动放爪、回位或继续下一条。

可调整参数：

```bash
python -m cloth_agent.gamepad_collect --real --iterations 5 \
  --approach-height-mm 50 --clearance-mm 30 --place-offset-mm 5 \
  --speed-mm-s 10 --yaw-speed-deg-s 5 --gripper-speed 2000 \
  --output data/demonstrations
```

`--approach-rpy-deg R P Y` 显式指定接近时的 World 朝向（度）。速度参数只控制手柄阶段；自动段使用项目配置中的速度限制。所有高度均相对点选的 RGB-D 表面，单位 mm。搬运阶段严格使用交接时的高度；`--clearance-mm` 仅指定放下后的竖直撤离距离，不决定搬运高度；自动 TCP 段不搜索绕行路径。

每条 traj 包含：

| 文件 | 内容 |
|---|---|
| `observation/rgb.png` | 迭代开始的照片 |
| `observation/world_xyz_mm.npy`、`observation.json` | 像素对应 World 坐标、标定和拍摄信息 |
| `points.json` | start/target 的原图像素与 World XYZ |
| `states.jsonl` | 约 10 Hz 实测关节角、由 FK 得到的 World TCP、控制器 TCP、实测夹爪 pulse、采样时间与阶段 |
| `events.jsonl` | 阶段切换、手柄意图变化、START 交接位姿 |
| `approach/`、`transfer_place/`、`return_perception/` | 自动段源命令、实际计划和执行结果 |
| `manual/continuous_config.json` | 手柄连续控制配置 |
| `traj.json` | 参数、标定 ID、时间、采样数及 PENDING_CONFIRMATION / COMPLETE / REJECTED / FAILED / ABORTED 状态及人工确认结果 |

控制器事件保存在会话目录下的 `controller/`。轨迹记录覆盖拍照后选点、自动接近、手动调整、搬运放置和返回；首次拍照前的回 perception 准备单独保存在 `observation_perception/`。状态时间以各条轨迹的 monotonic 起点为基准；夹爪读取与关节报告不是硬件同步采样。轨迹是实际执行记录，尚未提供一键重放。

START 是操作者对抓取成功和抬升完成的确认，软件不判断是否实际抓住布料。当前新增流程通过离线测试，仍需现场验证点选定位、放置偏移和模式交接。

自动接近、Home 和 perception 关节运动均直接插值，不调用 RRT/RRTConnect。搬运、下降和撤离使用笛卡尔插值逐点 IK，也不启用 RRT 绕行。IK 或控制器执行失败时停止并报告，不搜索替代路线。

Arm6 采集与 Arm7 使用同一个 `SingleArmRuntime`：共享 IK、末端关节等价角选择、平行夹爪姿态策略、命令队列和执行检查（当前采集已关闭路径区间碰撞验证），不再继承旧的 `Arm6Runtime`。抓取高度偏移沿用 Arm7 的 0 mm，不沿用旧 Arm6 的额外下降 15 mm。手柄通过同一连续控制实现运行，只按 Arm6 的六轴模型计算。Arm6 的 URDF、关节限位、TCP、工作空间及 Camera A 标定保持各自实测配置；不会套用 Arm7 的七轴几何或 Camera B 外参。当前采集仍按用户要求禁用 RRT，并在手动交接高度执行搬运。

按用户要求，本采集入口的所有阶段均关闭路径/几何碰撞验证：初始准备、接近、手柄、搬运、放下与返回。不调用整段碰撞证明、碰撞包围区间细分或 RRT；也不再按保存的 Arm7 几何阻止动作。手柄只计算关节限位范围，不运行碰撞验证线程中的几何证明。IK 精度、关节限位、速度限制、控制器错误、反馈时效、输入超时、模式检查和到位检查仍保留。自动段 `planning_stats.json` 及手柄 `continuous_config.json` 记录此设置。该设置只作用于采集运行时，其他运行入口默认保持原检查。

正常中止仍立即 STOP 并结束本次会话，不自动继续运动。下次重新运行原采集命令时，无控制器错误/警告、无排队运动的软件 STOP 会自动恢复到位置模式并使能，然后开始采集；无需网页手动重启。即使已经处于 perception，也先完成恢复。实体急停或真实故障不会被自动清除，恢复失败会返回具体错误。

手柄按键及当前 USB 设备方向详见 [gamepad_teleop.md](gamepad_teleop.md)。独立手柄入口现也默认 Arm6；采集入口的 START 为交回自动搬运，独立入口的 START 为停止退出。

Arm6 手柄夹爪已改为连续终点目标：按住 B 闭合、按住 X 打开，按下/换向时只下发一次，松键请求保持实测开度。默认 `--gripper-speed 2000`，可设为 1000 降低速度、最高 5000；旧 `--gripper-rate` 为同一参数的别名，不再表示 12 pulse 小步推进。此修改消除主机反复改目标的来源，不代表夹爪本身的受阻提前停止已修复。

自动搬运的每个 move_tcp 使用一次控制器直线运动，不再逐个 5 mm IK 点停稳。主机仍提前计算中间 IK 点用于限位及运行反馈检查；水平搬运、下降、松爪和抬升之间保留明确到位边界。

Arm6 CollectionRuntime 自动夹爪（包括 replay）速度已与手柄默认保持一致：SDK 速度 2000。自动位置指令显式携带该速度，不再被原配置的 500 覆盖；松爪分段顺序及完成反馈检查保持不变。
