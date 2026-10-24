# Arm6 半自动示教轨迹采集

入口：`python -m cloth_agent.gamepad_collect`。使用 Camera A 和 Arm6；仅连接 Arm6，Arm7 无需在线，不读取其状态、不发送任何命令。兼容数据结构中的 Arm7 使用 `data/robot/xarm7_perception_pose.json` 保存的姿态作为静态几何，不代表实时位置。需要项目 `cali` 环境、可显示窗口的桌面（Tk）、相机及已完成映射核定的手柄。

```bash
conda activate cali
python -m cloth_agent.gamepad_collect --real --iterations 10
```

多个手柄时增加 `--device /dev/input/eventN`。默认输出到 `data/demonstrations/<会话时间>/traj_0000/`，每个 iter 先写入 `pending/iter_0000/` 临时目录。回到 perception 后弹出保存确认，点击“确认保存 traj”才移入正式 `traj_0000/`；点击“不保存”、Esc 或关闭窗口则保留临时数据并标记 REJECTED，不计为正式 traj。不覆盖旧数据。未指定 `--real` 不连接硬件。

确认窗口左侧显示 `observation/rgb.png`（操作前），右侧显示 `after/rgb.png`（操作后），以相同缩放比例完整展示，随窗口大小调整，便于人工判断抓取是否有效。自动采集结束回 perception 后已有的照片直接用于对比，不额外调用模型或移动机械臂。其他模式或旧记录若缺少照片，会明确显示“照片不可用”，不会借用下一条轨迹的照片。Enter 不触发保存。

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

## 独立手动轨迹采集

```bash
python -m cloth_agent.gamepad_record --real --iterations 10 --gripper-speed 2000
```

使用 Arm6 和 Camera A。每轮开始回 perception 拍照，然后直接进入手柄控制，
不选 start/target，不执行自动 approach 或搬运。LB 配合摇杆、Y/A、X/B 控制
移动、升降、开合。松开全部控件，再单独按 START 结束；保持当前位置并弹出保存确认。
若进入下一轮，会再次回 perception 拍照。默认夹爪 SDK 速度为 2000（上限 5000）。
采集、独立手柄与自动 replay 的默认速度一致。

数据仍保存为确认后的 traj，包含初始 RGB/depth、约 10 Hz 的 joints/TCP/夹爪反馈、
手柄输入事件和 final_state.json；manifest 的 collection_mode 为 manual_only。
没有生成自动动作模板，也没有末端固定视角后图。当前辅助示范图编译器不接受该模式，
不能把完整手动轨迹直接当成旧的两点搬运模板。

## 单点靠近后自由手动采集

```bash
python -m cloth_agent.gamepad_collect --real --single-point --iterations 10 \
  --speed-mm-s 20 --gripper-speed 2000
```

每轮在 Camera A 照片只选一个 start，按 Enter 确认；自动打开夹爪并用 IK
移动到该点上方 50 mm，然后自由手柄操作。无需放置 target，START 不检查搬运高度，
只停止手动控制并进入保存确认；不自动搬运、松爪或回 perception。
下一轮开始时会回 perception 拍照。轨迹标记为 single_point_manual，保存单点、
approach、完整手动反馈及 final_state，不兼容旧的两点自动搬运模板编译器。

Arm6 默认手柄平移/上升由 10 提高到 20 mm/s，下降增益 1.5，对应最高 30 mm/s；
合成平移速度封顶 30 mm/s。关节速度/加速度限值仍生效，实际速度可能较低。
此手动速度独立于自动 approach/replay 的 Cartesian 速度。夹爪默认仍为 SDK speed 2000。

### 两点全自动采集（无需手柄）

```bash
conda activate cali
python -m cloth_agent.gamepad_collect --real --auto-pick-place --iterations 10 \
  --gripper-speed 2000 --grasp-offset-mm -15 --lift-height-mm 30
```

相机 A 拍照后直接打开统一的选点与角度窗口：依次点 START（抓取位置）、TARGET（放置位置），在同一窗口调整角度，再点击“确认选点和角度 · 启动搬运”。底部按钮可重选起点或终点；Esc 取消。无需 Enter 或第二个预览窗口。
Arm6 在整段 IK 规划成功后，自动张开、接近、下降、闭合、抬升、搬运、放下、部分张开 180 pulse、撤离，回 home 后完全张开，再返回 perception。随后保存 `after/` 照片并弹窗确认是否发布为 traj。此模式不查找或连接手柄；Ctrl+C 中止，原有停止处理保留。

`--grasp-offset-mm` 是抓取 TCP 相对点击表面高度的偏移，默认 -15 mm；负值表示低于表面，不是相对桌面高度，需符合所用夹爪、物体与垫子的实际几何。`--lift-height-mm` 默认 30 mm，以两点中更高的表面为基准。姿态沿用现有 home 抓取姿态，也可使用 `--approach-rpy-deg`。该模式减少手动操作，抓取角度在本地手动调整，不调用模型；自动运动速度沿用当前 Arm6 配置，`--speed-mm-s` 仍仅用于手柄控制。

记录的 `collection_mode` 为 `automatic_pick_place`，自动命令和结果保存在同名子目录，仍保存 points、states、events 和确认结果。它没有手柄事件，不能当作旧的 assisted 手柄示范直接套用其基于手柄闭合/抬升事件的 graph 编译器。

### 展开采集后的随机重置

```bash
conda activate cali
python -m cloth_agent.gamepad_collect --real --auto-pick-place --iterations 100 \
  --random-reset --reset-lift-height-mm 200 --reset-radius-mm 30 \
  --gripper-speed 2000 --grasp-offset-mm -15
```

每条轨迹保存确认之后，另有三个按钮：**继续展开**（保持当前衣物状态采下一条）、**展开完成：随机抓起并落下**、**结束采集**。保存确认本身不触发重置；只有点击“展开完成”才执行。

重置用相机 A 新拍的照片调用 Codex，识别衣身中心附近完全位于可见布料内的圆形抓取区域，避开领口、边缘与遮挡。随后在圆内随机采样，要求真实深度有效、相对中心的 World XY 距离不超过 `--reset-radius-mm`（默认 80 mm，上限 80 mm）。低置信度（小于 0.8）、无有效深度或 IK 失败会中止，不使用猜测深度，不自动重试物理抓取。

夹爪默认在表面以下 15 mm 闭合，竖直抬升至表面以上随机 250–300 mm，再随机旋转 yaw ±45°，保持高度移动到 home World XY 周围半径 40 mm 内的随机位置，完全张开。等待 2 秒后回 perception 拍照。抓取、抬升、旋转、平移都在抓取前完成 IK 规划；不进行甩动。采样仍限定在视觉确认的可见衣身布料区域内，因此实际抓取范围可能小于 80 mm。

每次落下后调用 Codex 比较前后折叠形态，排除单纯平移/旋转带来的区别。只有明确判为 TOO_SIMILAR 且置信度 ≥0.8 才重新识别、重新采样并重置，默认最多额外 1 次；不确定、深度失败、IK 失败或执行故障不触发动作重试。到达重试上限仍相似时记录 TOO_SIMILAR，不声称已获得新状态。

参数：`--reset-radius-mm 80 --reset-lift-height-mm 300 --reset-height-jitter-mm 50 --reset-yaw-range-deg 45 --reset-drop-radius-mm 40 --reset-max-retries 1`。高度采样范围为 `[lift-height-jitter, lift-height]`；最大抬升 300 mm，旋转范围最大 ±90°，释放偏移半径最大 60 mm，最多额外重试 2 次。独立脚本对应参数去掉 `reset-` 前缀。旧命令显式传入 `--reset-radius-mm 30` 会保留 30 mm 范围，需改为 80 才扩大。

每次尝试保存随机种子、`grasp.json`、`release.json`、前后图和 `comparison.json`。重试记录在 `retry_01/` 等子目录；`result.json` 的 `final_after` 指向最终照片，`attempts` 保存全部尝试。重置记录仍与展开 traj 分开。


重置证据单独存放在 `session/resets/after_iter_XXXX/`，包含 `before/`、`after/`、`region.json`、`grasp.json`（随机种子、像素及实测世界坐标）、`grasp_preview.png`、模型调用日志、状态反馈、动作程序和 `result.json`。重置不是展开示范。下一条 traj 的 `episode_id` 增加并记录 `preceding_reset`；以后导入 graph 时不能把跨重置的观测误连成展开动作。

不启动采集界面、手柄或 replay，单独执行中心抓取抬升并丢下：

```bash
conda activate cali
python -m cloth_agent.center_lift --execute
```

此入口复用上述随机重置流程：相机 A 拍照、识别中心附近区域、抓取、抬升 250–300 mm、随机旋转、移至 home XY 周围释放、回 perception。夹爪速度沿用 Arm6 的 2000；记录独立保存在 `reports/random_reset_<UTC时间>/`。默认变化过小时最多额外重置一次；只做一次可加 `--max-retries 0`。固定抬高 300 mm 并在 home XY 释放可加 `--height-jitter-mm 0 --drop-radius-mm 0`。仅写入独立重置记录，不作为采集 traj。

原有独立入口也继续可用，例如指定参数：

```bash
python -m cloth_agent.random_fold_reset --execute \
  --output "reports/random_reset_$(date +%Y%m%d_%H%M%S)" \
  --lift-height-mm 200 --radius-mm 30 --grasp-offset-mm -15
```

只测试 Codex 看图，不连接硬件：

```bash
python -m cloth_agent.random_fold_reset --image /path/to/rgb.png \
  --output reports/random_reset_vision_test
```


### 手动夹爪角度预览（无模型等待）

`--auto-pick-place` 使用单一窗口完成 START、TARGET 选点及角度调整，不调用 Codex。初始角度由原抓取姿态投影到照片计算；若指定 `--approach-rpy-deg`，使用该姿态作为初始值。

左侧全局图、右侧抓取点放大图同步更新。黄色连线表示开合方向，两端短线为方向示意，不是精确碰撞包络。拖动 0–180° 滑块，点击 ±5°，或按 Q/E 微调 1°；“恢复初始”恢复原姿态的方向。只有点击“确认选点和角度 · 启动搬运”才进入动作规划与执行，Enter 不启动动作。取消、关闭窗口或 Esc 中止本次。

程序根据相机内参、当次外参和抓取高度，将确认的图像方向转换为 World yaw；保留原有俯仰，利用夹爪 180° 对称性选择较小旋转。标准 xArm 夹爪沿 flange Y 开合，同时考虑配置的 TCP 旋转。调整预览不发送运动命令；确认后仍执行深度、IK 和设备状态检查。

`grasp_angle/initial_angle.json` 与 `initial_preview.png` 保存初始方向，`preview.png` 保存最终确认方向。`decision.json` 保存人工确认、初始/最终角度、最终 World RPY，以及 `selection_method=manual`、`model_calls=0`。

此更改只去掉抓取角度的模型调用；点击“展开完成”后的随机重置仍调用 Codex 寻找衣服中心。

统一窗口中，点击显示图片之外的留白会被忽略，深度无效的点不会被接受。起点与终点水平距离至少 10 mm；点位未齐或不满足距离条件时启动按钮禁用。重选起点会恢复该点对应的原抓取方向，重选终点保留当前角度。

### 在同一窗口选择手柄微调抓取

统一窗口右侧增加“切换自动 / 手柄抓取”。默认自动；切到手柄后，启动只会先张开夹爪并到达 START 上方（默认 50 mm），不会自动下降闭合。手柄在接近动作之前连接，找不到唯一设备时不执行接近，可用 `--device` 指定。

按住 LB 操作：左摇杆平移、右摇杆 yaw、Y/A 升降、X/B 开合。逐渐调整高度，抓好并抬升后，松开所有控件，再单独按 START；系统恢复位置控制，从实际抬升姿态搬到 TARGET、放下并回 perception。若高度低于目标放置高度，会提示继续用手柄抬高，不直接开始搬运。

此处微调速度由 `--grasp-adjust-speed-mm-s` 控制，默认 5 mm/s；下降仍沿用手柄的 1.5 倍增益（默认 7.5 mm/s），夹爪 SDK 速度保持原值。该参数不改变自动搬运速度。临时手柄连接在本段结束或异常时关闭；纯自动模式不要求连接手柄。

选择手柄的轨迹保存为 `collection_mode=assisted`、`grasp_control=gamepad`，保留原有 approach、manual 事件、transfer_place 和反馈数据；角度文件夹中的 `grasp_control.json` 记录本次选择。GUI 角度决定初始接近朝向，最终搬运使用手柄交回时的实际姿态。

“轮廓展开”和“表面平整”是两个不同判断：存在折叠覆盖时继续展开；轮廓已经展开但局部有褶皱时可再采集局部轻拉铺平动作。当前没有加入自动压布抚平，也不能仅凭夹爪闭合保证只夹到一层。

### 将当前这一轮交给 Codex 展平

统一窗口右侧的 **交给 Codex 局部去皱** 按钮会直接授权并开始这一轮的闭环展平，不需要手动选 start/target 或角度，不弹出第二个确认窗口。普通手动选点/角度模式仍不调用模型。启动命令不变；`--smoothing-max-steps` 控制这一轮最多动作数，默认 3，范围 1–6。

`smooth_cloth()` 位于 `cloth_agent/cloth_smoothing.py`。它使用当前照片规划一步：只对已经展开的衣服做局部去皱，保持衣服轮廓和布层关系；卷袖、重叠和大范围展开交回操作者；模型选择原图抓取点、放置点和夹爪开合角度。主机从实测深度提取坐标，检查 10–25 mm 水平位移及两点表面高度差不超过 5 mm，将角度转换成 World yaw，并规划完整 IK 后执行。接近高度 50 mm，低抬高度为两点较高表面上方 15 mm，抓取偏移沿用 `--grasp-offset-mm`，放置偏移 5 mm。放下后部分张开、撤离，home 后完全张开并回 perception 拍照。

下一次模型调用同时看当前图和上一步动作前图，检查局部褶皱是否减轻及衣服其他部位是否保持稳定，而不是仅凭机械臂指令成功判定进展。低置信度（<0.8）、无进展、整件滑动、重叠加重、无可靠抓取点或达到动作数上限会停止；运动/IK 异常由采集器停止设备并记录失败，不盲目重复物理动作。达到动作数上限后仍进行最后一次看图评估。`MODEL_REPORTED_DONE` 是视觉模型判断，不是平整度的物理保证。

这一轮保存为 `collection_mode=codex_smoothing`，`smoothing/` 包含初始图、每步前后照片、提案、点位/姿态、预览、实际命令、耗时和 `result.json`，采集器继续记录完整反馈。结束后仍通过原来的确认框决定是否发布为 traj。该模式含多个自动动作，不可直接当作旧版单次手柄抓取轨迹导入 graph；原始证据保留。随机重置仍需另行点击“展开完成”。

局部去皱的模型输入包含实际 `--grasp-offset-mm`、深度有效比例、稀疏实测 World 坐标、近似像素尺度和动作限值。模型负责视觉选点；精确深度补丁、距离/高度差和 IK 由主机随后验证，不能把“模型只看到 RGB”误解为系统没有深度。衣物其他区域的卷袖不自动否决独立区域的局部去皱，但不得在本模式中处理卷袖或分层。结束时终端直接显示模型的停止/完成理由。


模型提示已简化为“把衣服做 smooth，让布面更平整、褶皱更少”，只保留当前控制接口、坐标格式和 ACT/DONE/STOP 的含义。去掉长篇禁止事项、85 mm 夹爪警告和逐条停止指令，让模型根据当前照片决定如何改善。此处只调整模型提示，主机已有的局部运动范围、深度/IK 验证、动作数上限和执行状态检查仍保留。

如果动作提案的像素点经实测转换后不满足几何范围，程序会在尚未执行动作时，把实际点位、错误和限制反馈给 Codex，允许修正一次提案；修正点仍须重新通过几何和 IK 检查。第二次仍不通过则停止并直接打印几何拒绝原因。该重试只针对执行前提案，绝不重试已开始执行的物理动作；原提案、拒绝原因及 `revised_proposal.json` 都保留。

### 界面直接启动“抓中间并丢下”

在 `--auto-pick-place` 的统一窗口右侧，点击 **抓中间并随机丢下** 即可直接启动一次随机重置，不需要选 start/target，也不需要先启用 `--random-reset`。按钮沿用上述全部随机重置参数和 `--grasp-offset-mm`。

流程为重新拍照、Codex 找中心附近可抓区域、随机选点、抓取、竖直抬高、保持高度移到 home 的 XY、张开、等待落稳、回 perception 拍照。完成后自动进入下一轮窗口，不额外弹出保存展开 traj 的确认。UI 重置占一个 iteration，记录保存在 `session/resets/ui_iter_XXXX/`，标记 `not_a_demonstration=true`，不会混入展开动作。下一轮增加 episode_id 并记录 preceding_reset。

### 多 run 轨迹合并为状态图（离线）

```bash
conda activate cali
python scripts/compile_arm6_archive_graph.py \
  --root data/demonstrations \
  --output reports/arm6_archive_graph_new
```

只导入 `COMPLETE` 且 `operator_accepted=true` 的 Arm6 / Camera A 示范，支持自动抓取搬运、手柄辅助和自由手动轨迹。随机重置、失败、取消和拒绝保存不作为示范边。优先使用本条 `after/`；旧数据只允许以同一 run、相邻编号、同一 reset episode 的下一轮初始观测作为带有连续性不确定性的后图。没有后图时保留未知结果节点，不跨 run 或重置补接。

编译时直接对比带 observation ID 的照片拼图，忽略单纯平移、旋转和轻微皱褶，合并可见布片覆盖、边界及可操作区域相同的状态；再逐条对比动作前后图及原始 start/target 标记，描述动作关系。每条原始示范保留一条独立有向边（包括自环），不会因为合并状态而丢掉动作。新图同时容纳折叠和展开示范，不擅自指定任务终点，也不启用实机 replay。

输出 `graph.json`、带哈希的 `trajectory.json`、`import_summary.json` / `summary.json`、每批模型请求记录，以及 `preview/overview.png`、`preview/states_*.jpg`、`preview/members/`（每个合并节点的全部照片）和 `preview/edges.csv`（边对应原始 traj）。PNG/JPG 可直接在 VS Code 查看。

可先加 `--import-only` 检查导入清单，再去掉该参数并加 `--resume` 继续视觉编译。`--resume` 使用已冻结的数据快照，不会把采集过程中新增的轨迹混进本次编译；已完成的视觉批次可以复用。新一批数据应使用新输出目录。

### 独立“辨认—探索—展开”实验

```bash
conda activate cali
python -m cloth_agent.experiments.arm6_recognize_pull \
  --execute --max-steps 12 --max-exploration-pulls 3
```

每轮先只看当前相机 A 照片，判断是否能辨认衣服的整体布局、主要部位及折叠遮挡关系。认不清时选择可见外露边角探索性拉开；认识但未展开时，按已辨认的衣物部位继续规划一步展开。每次执行后回 perception 拍照，重新判断。此脚本不加载 graph、参考状态图或历史示范，也不走 replay。

默认共最多 12 次动作，其中认不清状态下的探索最多 3 次。探索位移最多 80 mm；认识后的展开位移最多 150 mm（`--max-pull-mm` 可以调小）。抓取表面偏移 -15 mm、抬升 50 mm，复用 Arm6 普通 IK、实测深度、速度 2000 的夹爪以及小幅释放后回 Home 完全张开流程。识别/动作置信度不足、无可见抓取依据、深度或几何不满足、执行故障都会停止，不重复下发失败的动作。

“认识”不等于已展开：只有模型依据当前图片判断主要衣身、袖部和下摆已经展开，才以 `MODEL_REPORTED_UNFOLDED` 结束；允许普通皱褶，不要求熨平。此状态是视觉模型的判断，记录可供人工复核。每步保存识别结果、动作提案、点位预览、实测反馈、前后照片和耗时；结尾输出 `experiment_summary.mp4`。文件默认位于 `reports/arm6_recognize_pull_<UTC>/`，不会自动标成已确认采集 traj。

只验证某张已有照片的识别结果，不连接机器人：

```bash
python -m cloth_agent.experiments.arm6_recognize_pull --image path/to/rgb.png
```

## 从连续 traj 编译视觉 skill，再独立展开

这一实验不做 graph 节点匹配。编译器读取归档的原始轨迹快照，按 session、episode、
preceding_reset、连续迭代编号和显式 reset 目录切分；相邻动作的 after/before 再交给视觉模型
检查是否存在额外改变。模型用重叠窗口分析一到数步的子目标与后续利用关系，区分直接可见
效果与推测意图。例如，向内翻动可能是为了露出下方袖子；不会仅因面积变小判为失败。

编译先生成片段案例，再合并为技能。每个技能必须有可见子目标效果的支持案例；不确定案例
保留但不升级为支持证据。生成的技能仍是待测试经验，不是训练后的权重或成功率保证。
旧 `experience/unfold_experience.json` 不作为输入。

先验证全部归档并制作图示窗口，无模型调用、无硬件连接：

```bash
conda activate cali
python -m cloth_agent.experiments.cloth_skill_library \
  --trajectory reports/arm6_archive_graph_20261019/trajectory.json \
  --output reports/arm6_skill_all \
  --prepare-only
```

随后同一命令去掉 `--prepare-only` 开始视觉编译，可原命令重启并复用已完成请求缓存。
输入、分组参数和模型应保持相同；换数据或参数时使用新的 output。
默认读取所有 session；`--session RUN_ID` 可以重复以限定训练集，
`--holdout-session RUN_ID` 可以重复以留出整轮测试数据。窗口默认 6 步、重叠 2 步。
旧快照不包括后续新采集数据；新数据需先用 `python scripts/compile_arm6_archive_graph.py --import-only --output reports/新归档目录`
生成新的 trajectory 快照，编译 skill 不需要先生成 graph。

编译产物：

- `dataset.json`：训练/留出 session、分段窗口和原始记录索引。
- `evidence/*.jpg`：各步带 start/target 的前后图，可直接在 VSCode 打开。
- `analyses/*.json`：连续性判断、片段目的、可见证据、不确定之处。
- `skill/SKILL.md`、`skill/library.json`、`skill/examples/*.jpg`：可移植的技能包，保留证据哈希。
- `model_metrics.json`、`summary.json`：编译模型用时、调用数及技能数量。

先用已有照片检查选技能和点位，不连接机器人（纯 RGB 不验证深度或 IK）：

```bash
python -m cloth_agent.experiments.arm6_skill_policy \
  --library reports/arm6_skill_pilot_20261019/skill/library.json \
  --image reports/random_reset_20261019T073741853726Z/after/rgb.png
```

运行完整 Arm6 实验：

```bash
python -m cloth_agent.experiments.arm6_skill_policy \
  --library reports/arm6_skill_pilot_20261019/skill/library.json \
  --execute --max-steps 12
```

每步先看当前图与上一步前图，判断上一步子目标效果并选择技能，再加载该技能的少量完整
案例片段（最多 8 张案例对照图）定位当前抓点、落点和夹爪角度。不要求先认出整件衣服，
不复制历史像素，也不执行案例的整段关节轨迹。当前执行器支持单次抓取、抬升、搬移、放下；
技能中提到的多步目的通过重新观察逐步实现，尚不支持沿任意示范曲线拖动或抓持中变换姿态。

继续使用 Arm6/Camera A、普通 IK、连续笛卡尔运动和夹爪 speed 2000；放置时部分张开，
回到 home 后完全打开。默认抓取偏移 -15 mm、抬高 50 mm、单次平移 10–150 mm，参数沿用
独立实验；`--grasp-offset-mm` 和 `--max-pull-mm` 可调整。选技能的 ACT 是待核对候选，低置信度
仍可进入案例核对与点位规划，不会直接执行。最终 grounding 的置信度同时评估技能适用性和
具体抓取可靠性，仍需达到 `--confidence-threshold`（默认 0.8）。实测深度或 IK 拒绝、
最终动作低置信度、无适用技能时停下；完成判断也保留置信度要求。
连续两次子目标没有进展、变差或不确定时停止，避免反复试同一操作；
可用 `--max-stalled-steps` 调整。Ctrl+C 停止，不自动重发失败的机械臂动作。

输出在 `reports/arm6_skill_*/`：每步的 `selection.json`、`grounding.json`、
`proposal_preview.png`、真实执行与反馈记录、before/after、`timing.json` 和关键帧回顾视频。
视频是静态前后照片，不是连续录像。DONE 记录为 `MODEL_REPORTED_UNFOLDED`，属于模型判断。
这些实验记录不自动混入人工确认的采集 traj。

摸底测试可加 `--confidence-threshold 0`，关闭运行时的置信度数值门槛（含最终动作和完成判断），
同时保留原始分数便于事后分析。结果会记录 `confidence_gating_enabled: false`。
该模式使用尽力尝试的规划提示词：根据可见布边提出动作，不要求事先确定隐藏布层或保证展开成功，
以动作后的照片验证效果，记录 `planning_mode: best_effort_baseline`。普通模式保持原来的案例核对要求。
这不改变模型明确的 STOP 决定，也不改变实测深度、IK 或控制器故障处理。

```bash
python -m cloth_agent.experiments.arm6_skill_policy \
  --library reports/arm6_skill_pilot_20261019/skill/library.json \
  --execute --max-steps 12 --confidence-threshold 0
```

## 采集界面的 Graph 状态检查

选点与抓取方向窗口右上角的“检查当前照片是否在 Graph 中”使用左侧当前快照，
调用与 replay 相同的 DINO Top-10 + GPT 匹配。不会重新拍照、规划或执行动作。
识别窗口显示当前照片与参考照片，支持逐张浏览 Top-10；MATCH 时优先展示对应节点。
NO_MATCH 表示检索候选中未确认匹配，不证明整个 graph 都没有该状态。
识别期间可取消，关闭后返回原来的选点/角度；识别结果不自动启动搬运。
每次结果与日志保存到该 iter 的 `graph_checks/`，随轨迹确认一起保留。

默认使用 `reports/arm6_graph_incremental_20261020_v4/final/graph.json` 和同包的 `memory/`。
换图需同时指定 `--graph PATH --graph-memory DIR`，图库与索引不一致时会报错。
普通自动搬运的默认抬升已从 50 mm 改为 30 mm（两点较高表面上方）；
命令中若显式指定旧的 `--lift-height-mm 50`，请删除该参数或改成 30。
接近高度、随机抬升丢下高度及历史 replay 动作模板不受此默认值修改影响。
