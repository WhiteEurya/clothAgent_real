# Arm7 手柄连续遥操

入口 `python -m cloth_agent.gamepad_teleop`。默认只读取并打印手柄输入；
`--real` 才连接机器人并执行。使用 Linux evdev 标准 Xbox 布局，适配暴露对应
按键和轴的北通阿修罗 USB PC 模式。阿修罗有多个型号，具体切换模式按对应
[北通官方说明书](https://community.betop-cn.com/)操作，不能把某一型号组合键通用到整个系列。

| 操作 | 功能 |
|---|---|
| 左摇杆上下 | World +X / -X 平移 |
| 左摇杆左右 | World +Y / -Y 平移（右推为 -Y） |
| 右摇杆左右 | 绕 World Z 旋转 yaw，右推为负方向 |
| Y / A | +Z 上升 / -Z 下降 |
| 按住 X / B | 逐渐开爪 / 合爪，松开请求保持当前开度 |
| 按住 LB | 允许上述操作 |
| START | 停止并退出 |

启动后先松开所有按键、摇杆回中，再按 LB。前后左右以上表 World 轴定义，
当前 USB Gamepad（0810:0001）按现场反馈在设备配置中反转了左摇杆的两个轴，
因此该手柄前推为 World -X、右推为 World +Y；右摇杆和按键映射不变。
不随夹爪朝向旋转，也不保证与操作者站位或相机画面的方向相同。
只调 yaw，roll/pitch 没有手柄控制。右摇杆上下未映射。
同时按 Y/A 不升降，同时按 X/B 保持当前开度。
松开 LB 停止移动与开合，再按住 LB 可继续，无需重启程序。
若 X/B 在 LB 按下前已经按住，需要先松开该键再按，防止重新使能时意外开合。

## 安装和预览

```bash
python -m pip install -r requirements-gamepad.txt
python -m cloth_agent.gamepad_teleop --list
python -m cloth_agent.gamepad_teleop --device /dev/input/eventN
```

预览输出 JSON，包含标准化摇杆、按键和映射结果，可由其他程序读取；也可以
直接导入 `Gamepad`、`Mapper` 和 `Intent`。输出的是输入意图，不能绕过检查
直接把它当作已经验证的运动命令。多个设备时必须明确 `--device`；优先使用
`/dev/input/by-id/*-event-joystick` 稳定路径。需要当前用户具备该设备的读取权限。
未检测到设备时先检查数据线和 USB PC 模式；没有设备不能验证映射。

默认右摇杆为 `ABS_RX`，仅当预览确认设备右摇杆使用 `ABS_Z` 时使用
`--yaw-axis ABS_Z`。必须先在预览中逐一核对实际 A/B/X/Y、LB 与三根轴。

## 一次性输入核定

新手柄使用不同的轴/按键编号时，先运行一次映射学习与核定：

```bash
bash scripts/calibrate_gamepad.sh
```

程序提示左右摇杆的六个方向，以及实体 A/B/X/Y、LB（或 L1）、START 键，
自动识别编号和轴方向。随后自动运行下面的 16 项功能核定。全部通过后才把
映射保存到 `config/gamepads/`，按 USB 总线/厂商/产品/版本匹配；预览、核定
和遥操会自动加载。失败或中止不会覆盖已有映射，整个流程不连接机器人。
同 USB 标识的多个型号无法仅凭编号区分；更换此类手柄或切换输入模式后应重新核定。
学习阶段要求双摇杆处于模拟模式并回中，不要同时触动其他控件。

在实机之前，可以一次性运行完整输入核定：

```bash
bash scripts/check_gamepad.sh
```

程序自动识别唯一的手柄，在终端连续引导 16 项检查；按提示拨动摇杆或按键，
保持约半秒，成功后自动进入下一项，无需回车或逐项运行命令。内容包括：
摇杆六个方向、ABXY、无 LB 时禁止输出、松开 LB 使输入归零、冲突按键和 START。
每项默认 45 秒超时，错误/未操作不会被记为通过。先回中再进入下一项；
回中超时、断连或 Ctrl+C 会保存未完成/错误报告。

核定过程只读手柄，不连接机器人。结果保存到 `reports/gamepad_check/<时间>/`
下的 `report.json` 和 `samples.jsonl`。PASS 仅代表手柄输入及软件映射核定，
不代表机械臂实际方向、运动和停止行为已经验证。`--probe-seconds 2` 只做
连接读取检查，结果标为 CONNECTION_ONLY，不会冒充完整核定。

## 实机执行

使用具有 xArm SDK、IK 和碰撞依赖的项目环境：

```bash
python -m cloth_agent.gamepad_teleop --device /dev/input/eventN --real
```

只向 Arm7 发送运动和夹爪命令。沿用 `config/dual_arm.local.json` 和原生
运行时，只连接 Arm7 控制器，不读取或要求 Arm6 控制器就绪。Arm6 使用现有
`data/robot/arm6_parked_for_arm7.json` 保存的静止姿态参与碰撞检查；此姿态
不是实时反馈，若移动了 Arm6，需更新记录。启动不会自动 Home、
拍照或恢复 STOP；控制器必须已经处于可用的位置模式。不经过模型规划。

默认平移速度 10 mm/s、yaw 5°/s，摇杆偏转按比例缩放速度。全体关节速度
同时受配置中的 5°/s 上限约束，所以部分姿态下实际 TCP 速度更低。
夹爪目标推进默认 100 pulse/s，可调整：

```bash
python -m cloth_agent.gamepad_teleop --real --speed-mm-s 10 --yaw-speed-deg-s 3 --gripper-rate 100
```

不再使用 `--step-mm` / `--yaw-step-deg`。新入口采用控制器模式 4 的关节速度
控制，目标周期 20 ms，以微分 IK 把 World 平移/yaw 转为关节速度，不逐点
等待停稳。每个速度命令带 150 ms 到期时间，发布中断后控制器将速度置零。
软件同时监测输入、反馈、工作线程和控制器错误；断连/超时/START 会停止会话。
松开摇杆或 LB 正常下发零速度，不把正常松键当作故障。启动保持只读，首次
有效移动才从已就绪的位置模式切到速度模式；退出恢复模式 0，但不自动使能
或解除 STOP。仅检查连接可运行 `python -m cloth_agent.gamepad_teleop --real --connect-only`。

碰撞验证在独立线程中对当前关节附近的整块范围做区间证明，包含保存的停止
余量。工作空间沿用 Arm7 pipeline 同一配置及 `certify_segment` 验证函数。
优先验证 ±2°，失败时依次尝试 ±1°、±0.5°、±0.25°、±0.125°，
同时根据验证范围降低关节速度，不修改工作空间数值。较早的高速指令在
有效期、通信延迟和反馈年龄覆盖时间内仍计入可达范围，不能因缩小范围而忽略。
每次速度下发前，检查反馈年龄、指令有效期和通信延迟覆盖的可达范围
是否仍包含在已验证范围中。没有有效证明时只发零速度，不绕过验证、不自动绕行。
接近工作空间边界或其他障碍时，可能停止等待可验证范围。
终端此时显示验证失败原因或“运动暂停”，不会仅凭手柄输入显示“连续控制中”。
最小范围仍验证失败时会暂停，实际边界处仍可能阻止回退方向；这不代表手柄断连。
`controller/events.jsonl` 保存 `teleop_bounds_blocked` 的原因和关节位置、
`teleop_bounds_restored` 恢复事件及 `teleop_failure` 停止原因。

夹爪通过独立 SDK 通道更新小幅位置目标，使用 `wait=False, wait_motion=False`，
不会等待完整开合，也不会等待机械臂运动结束。默认目标最多领先实测位置
12 pulse；松键发送当前实测开度以保持，实际停止存在采样、通信和机构延迟。
若输入或连接丢失，最后一个目标仍只是有限的小幅增量，不会继续走完整行程。
不通过关闭夹爪电机实现停止，以免释放持物。运动与夹爪可同时操作。

日志保存 `continuous_config.json` 与 `controller/events.jsonl`，包括速度、循环
耗时和夹爪目标；支持 `--output`。这是主机软件控制及条件性的几何检查，
记录的停止余量沿用现有配置，新速度模式的实机停止距离和手感尚待验证，
软件停止不替代实体急停。退出不会自动松爪或回 Home。

实现依据：[python-evdev 输入读取接口](https://python-evdev.readthedocs.io/en/latest/tutorial.html)。
新手柄映射已通过现场输入核定；连续控制仍需现场运动、松键及断连验证。
控制器接口依据：[UFACTORY SDK](https://github.com/xArm-Developer/xArm-Python-SDK)，
本机安装版本同时确认了速度命令的 `duration` 和夹爪的 `wait_motion` 参数。
