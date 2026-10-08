# 双臂防碰撞与两种 Claude 引导模式

独立入口为 `python -m cloth_agent.dual_arm`，或 `python scripts/claude_dual_arm.py`。实现位于 `cloth_agent/dual_arm/`，原单臂入口不变。双臂执行器使用复制的 `single_arm_backend.py`、`kinematics.py`；源文件及复制时的 SHA-256 记录在 `copied_sources.json`。

本版增加连续运动包络、跟踪/停车余量、下发前检查和两种模式。**软件只在模型、标定、实测误差和控制器行为满足声明条件时验证几何净空，不提供无条件的实机安全保证。没有合格数据或无法证明净空时拒绝动作。**

## 两种模式

### `pin_pull`：一臂压住，另一臂抓住并向外拉开

Claude 选择压布手 `pin_arm`、受支撑的内部压布点和另一手的可抓边缘。压布点不执行额外的向下穿透偏移。

```text
两手张开 → 抬到通行高度 → 到达各自接近点
→ 压布手在空中闭合 → 压布手下降 → 检查压布接触
→ 另一手下降、闭合 → 检查压布和抓取
→ 另一手试抬、检查、抬升
→ 另一手沿远离压布点的方向分步拉开，每步检查
→ 撤回拉伸 → 放回、释放 → 操作手撤离
→ 压布手抬起、张开 → 两臂返回起始 TCP
```

压布期间保持压布手的**原关节向量**，不重复求 IK。执行器另外检查其漂移。视觉确认要求压布接触、压布处未滑动，以及操作手确实持布；不要求压布手“夹住布”。

**当前 `pin_pull` 支持规划、预检和仿真，拒绝实机运动。** 仓库没有已接入的受力限制接触控制器。仅以位置下降或看图确认接触，不能限制桌面接触力。CLI 和执行器均有独立拒绝逻辑，不能靠配置中的 verified 标志启用。后续接入带实测力阈值、接触检测和卸载策略的控制器后才能开放；本版没有假定设备配备力传感器。

### `center_pair`：先中心，再在附近选双抓点

这是默认模式。Claude 分两次调用：

1. 在原图上选中心点；宿主检查它的观测 ID、像素和深度。
2. 宿主固定中心，生成不缩放的辅助图，将实测中心邻域以外区域变暗。Claude 看原图及辅助图，选择中心两侧的两个抓点，不得改中心。

宿主要求两点分别位于中心的 `limits.center_radius_mm` 范围内，默认 **150 mm 三维半径**；两点在中心两侧，连线距离中心不超过 `center_line_tolerance_mm`，同时满足最小间距、工作区、IK 和完整路径检查。尺寸单位为 mm，不以图像像素半径代替。邻域太小、夹爪放不下时直接拒绝，不放松碰撞净空。

```text
中心选择 → 固定中心后的双抓点选择 → 完整预检
→ 两手张开、接近、下降、同时闭合 → 分别确认持布
→ 共同试抬、检查 → 共同抬升
→ 同时分步向外展开，每步检查 → 恢复间距
→ 放回、释放、撤离 → 返回起始 TCP
```

两种模式均允许 Claude 返回 `{"decision":"abstain","reason":"..."}`。抓点仍需来自可见衣物；邻域图只证明深度/距离关系，**不是衣物分割或可抓性认证**。视觉反馈的 confidence 是模型报告值，不是统计校准后的成功概率。

## 不连接硬件的演示

在安装了项目依赖的环境中运行，输出目录必须尚不存在：

```bash
python -m cloth_agent.dual_arm demo --mode center_pair --output /tmp/dual_center_demo
python -m cloth_agent.dual_arm demo --mode pin_pull --output /tmp/dual_pin_demo
```

演示使用实际六轴、七轴 URDF；底座、衣物、深度和持布反馈是合成数据，不连接机器人、相机或 Claude。演示的两底座相距 700 mm，为展示其 Home 附近运动，合成配置的中心半径为 500 mm；真实配置模板仍默认 150 mm。

每个目录包含 `config.json`、`observation/`、`proposal.json` 和 `run/`。其中：

- `run/preview.html`：顶视/正视和阶段滑块，显示加入跟踪/停车余量后的包络。
- `run/plan.json`：模式、阶段、配置摘要、连续碰撞检查范围和仿真标记。
- `run/trajectories/`：两臂共用时间轴的关节和 TCP 轨迹。
- `run/events.jsonl`、`run/execution.json`：执行反馈、失败原因和停机尝试结果。

离线 Claude 规划使用现有观测：

```bash
python -m cloth_agent.dual_arm plan --mode center_pair \
  --config /path/to/config.json --observation /path/to/observation \
  --backend remote --host company-planner --output /path/to/new_plan
```

换成 `--mode pin_pull` 即为压布模式。本机 Claude 使用 `--backend local`。中心模式日志保存 `claude/center/`、`claude/fixed_center.png`、`claude/points/`，可检查中心是否固定及选点依据。

## 防碰撞检查的具体范围

每个连杆、工具及附件都在共同世界坐标系中使用保守胶囊体。工具包络必须覆盖夹爪从完全闭合到完全张开的**所有状态**。不能把 TCP 间距当作夹爪间距。

从 URDF 链长推导每个胶囊体端点相对各上游关节的运动上界：

```text
端点运动上界 ≤ Σ(该关节到端点的最大距离 × 关节角变化，弧度)
每臂角度余量 = 允许跟踪误差 + 实测最大关节速度 × 反应时间 + 实测停车角位移
反应时间 = 反馈时效 + 控制器看门狗时间 + 停机命令延迟
           + 一个控制周期 + 允许调度延迟 + 允许下发时差
每个包络额外半径 = 基座误差 + 几何误差 + TCP 跟踪容差
                  + 角度余量对应的端点运动上界
```

在每个关节轨迹区间，继续按整段运动上界膨胀包络。如果膨胀后的形状保持净空，就覆盖了区间内部的连续运动，包括旋转扫过的空间；不能直接证明时递归细分。每次只细分一条臂的进度，保留两臂不同进度的组合，不能假定两台完美同步。超过计算预算仍无法证明时拒绝。

两臂的膨胀包络之间仍保留 `clearance_mm`，默认 10 mm。它现在是**误差和停车余量之外**的额外净空。同样检查包络与配置障碍物的关系。只允许固定底座与台架的安装接触排除项，不能排除工具或运动连杆。桌面接触例外永远不关闭双臂之间的碰撞检查。

预检覆盖整段程序。执行时，在下一组命令下发前检查实测关节到目标的区间，并拒绝过期反馈；之后继续检查跟踪、实际包络和故障。夹爪等待期间也监测机械臂，视觉等待期间检查保持状态。没有规划器自动绕障功能：不能证明所选路径安全就停止，不自行穿越障碍。

这些检查针对两臂之间及已配置的环境。未建模物体、错误标定、超出声明的控制器行为不会自动获得保证。断网停机必须有经实测的控制器/硬件看门狗支持，主机发送 `set_state(4)` 本身不能保证送达。

## 配置与实测数据

```bash
python -m cloth_agent.dual_arm init-config \
  --home data/robot/dual_arm_home.json --output config/dual_arm.local.json
```

`config/dual_arm.example.json` 故意保持 incomplete，误差及停车数据为空。旧配置缺少 `safety` 时拒绝，不回退为仅采样检查。新模型输出要求 `schema_version: 2`、`mode`、`center` 和 `pin_arm`；旧 proposal 同样拒绝。

模板把保存顺序中的六轴 `192.168.2.232`、七轴 `192.168.1.195` 命名为 left/right；必须核对实际安装，填写 `arm_layout_description`。连接时重新检查控制器序列号、轴数和 TCP。

必需几何包括：两基座到 world 的变换、各自基座下的 TCP 工作区、世界抓取朝向、每个连杆/工具/附件的胶囊体、桌面及障碍物。world 的 +Z 为抬升方向。所有变换平移用 mm；原手眼 YAML 的米需要转换。

基座拟合与包络工具仍可使用：

```bash
python -m cloth_agent.dual_arm fit-base --points /path/to/measured_points.json --output /path/to/base_fit.json
python -m cloth_agent.dual_arm envelopes \
  --urdf assets/robots/xarm6/xarm6_wo_ee.urdf --axis 6 \
  --tool-radius-mm 45 --tcp-offset-mm-deg 0 0 172 0 0 0 \
  --output /tmp/arm6_envelopes.json
```

拟合输入含 `base_points_mm/world_points_mm`（至少四对），以及 `validation` 下相同字段的至少两对独立验证点；拒绝退化/反射及超过 2 mm 的残差。示例工具半径不是已测量尺寸；自动工具包络远端结束于 TCP，需验证开爪、相机、线缆及附件，必要时增加多个包络。

`safety` 必须包括：

| 字段 | 实际含义 |
|---|---|
| `status`, `validation_id` | 实机必须为 measured，并能对应现场验证记录 |
| `open_tools_and_attachments_verified` | 所有开闭状态和附件都在声明包络内 |
| `controller_watchdog_verified`, `controller_watchdog_s` | 已验证通信/主机失效后的设备停机期限 |
| `joint_segment_tracking_verified` | 已验证伺服插值保持在命令区间与声明误差包络内 |
| `max_feedback_age_s`, `stop_command_latency_s` | 整次反馈采集时效与停止命令的延迟上界 |
| `arms.<arm>.base_error_mm` | 基座标定误差对整个包络的最大位移影响，含旋转误差 |
| `arms.<arm>.geometry_error_mm` | 工具/连杆模型及安装误差上界 |
| `arms.<arm>.max_joint_speed_deg_s` | 设备运动的实测速度上界，必须不低于计划速度 |
| `arms.<arm>.stop_excursion_deg` | 各轴在当前速度/负载下，从开始制动到静止的最大角位移 |

这些 verified 字段是现场验证声明，软件不会自动测量或证明其真实性。不能把合成演示的零误差复制到实机配置。实机要求非零误差与停车上界，且拒绝 synthetic 标记。改变安装、负载、夹爪、速度或固件后需要重新验证上界。

相机配置包含身份、分辨率和所属臂。腕部相机按 `world_from_base × base_from_mount(q) × mount_from_camera` 变换；固定相机设置 `mount_arm: null` 和 `world_from_camera_mm`。捕获期间两臂必须保持静止；观测保存原图、世界 XYZ、标定 ID、关节向量、时间及文件校验和。

## 实机入口的限制

实机除了 `--real --confirm-real`，还要求已测量几何、安全数据和完成空载伺服验证。正数的 `tool_contact_allowance_mm` 穿透例外在实机拒绝；只有仿真可以使用。`pin_pull` 无论 verified 字段如何都不能实机运动。`--preflight-only` 允许读取控制器、采集、调用模型和生成预览，不发送动作。

中心模式的预检入口：

```bash
python -m cloth_agent.dual_arm run --mode center_pair \
  --config config/dual_arm.local.json --output runs/dual_preview \
  --backend remote --host company-planner --real --confirm-real --preflight-only
```

压布模式预检可替换 `--mode pin_pull`；不能移除其预检标志来实际压布。中心模式在所有条件满足后，用新输出目录移除预检标志才执行。

空载 commissioning 为两臂从当前位置 world +Z 3 mm 抬升并返回，通过同一套连续碰撞检查：

```bash
python -m cloth_agent.dual_arm commission \
  --config config/dual_arm.local.json --output runs/dual_commission \
  --real --confirm-real --preflight-only
```

此命令允许 `servo_commissioned: false`，但不豁免安全测量要求，不自动张开夹爪、使能电机或清除故障。两臂必须已停稳、夹爪张开且空载。真实空载测试通过后才能设置 `servo_commissioned: true`；仿真不提供该证据。

异常、Ctrl+C、SIGTERM、过期反馈、跟踪误差、压布手漂移或视觉拒绝都会取消后续动作并分别尝试停止两台。送达未确认会如实记录。失败不会自动释放、回 Home 或继续。成功流程返回起始 TCP，七轴冗余关节不承诺返回完全相同的关节解。

## 验证

```bash
python -m pytest -q tests/test_dual_arm_runtime.py tests/test_dual_arm_safety_modes.py
```

测试覆盖：两端安全但中途相撞、旋转扫过碰撞、单臂滞后组合、停车误差膨胀、证明预算耗尽、禁止危险排除项、反馈过期、左右任一臂压布且关节固定、压布漂移停机、实机压布拒绝、冻结中心、防止超出邻域、模型拒答和旧协议拒绝。两种 URDF demo 验证离线完整流程。没有以自动化测试代替物理碰撞标定、受力验证或抓取成功率测量。

## 两臂回正（Home）

`scripts/gripper_home.py` 为双臂自动回位入口，启动即执行实机流程。
它不再使用单臂 `robot.example.json`，默认读取 `config/dual_arm.local.json`。
无需传入确认参数。完整预检先于任何运动或夹爪指令。
通用入口 `python -m cloth_agent.dual_arm home` 仍保持原有显式实机参数约定。

```bash
# 读取实机当前位置并预检，不执行回位/开爪
python scripts/gripper_home.py --preflight-only
# 合格配置下执行两臂回位，随后张开两个夹爪
python scripts/gripper_home.py
# 离线仿真
python scripts/gripper_home.py --simulate
```

也可直接调用普通阻塞函数；导入本身不连接或移动机械臂：

```python
from cloth_agent.dual_arm.homing import gripper_home

result = gripper_home()
assert result['status'] == 'COMPLETED', result
```

可选参数为 `config_path`、`output`、`simulated`、`preflight_only` 和 `project_root`。
返回值包含执行状态及 `output_directory`；预检失败抛出 `DualArmError`。

Home 使用配置里两台的 `home_joints_deg`，七轴也回到保存的关节解，不是只回到同一 TCP。
Home 默认使用 `controller_sequential`，另一臂在每段执行时保持原关节角。
对尚未到 Home 且两臂估算间隙扣除两臂配置位置误差后小于 30 mm 的姿态，优先规划分离动作：
根据全部跨臂连杆、夹爪和相机的最小表面间隙估计，尝试左右臂各关节正负 2°、5°、10° 的退让方向。
一次仅移动一个关节，要求参考路径上的最小间隙不减，并通过完整运动和停止范围检查。
方向评分也使用网格分离平面细化过大的连杆胶囊；附件仍使用保存的胶囊。
间隙估计用于方向选择，实际碰撞判定另行使用完整运动、已有误差及停止范围。
方向按间隙增益与部件位移上界的比值排序，优先尝试小幅有效退让；探索中的单个分离
候选最多使用 10 秒碰撞证明时间，耗尽时间即拒绝该候选。最终控制器段的完整证明不使用该探索时限。
最多累计 8 步，每步之后尝试两种回 Home 顺序；只有分离与回位的整条路径通过才执行。
分离后优先尝试末端直线参考路径，使用连续种子的 IK 让关节协调运动；随后校验并回到精确 Home
关节解。备选同步关节插值也必须满足同样的路径形状约束，不再逐关节大幅回位。
若一臂全部回位会被另一臂阻挡，可将两臂的协调路径分为 6、12 或 24 个阶段交替执行；
每段仍只移动一臂，两臂都逐步退出近距离工作区。整个交替方案预检通过后才下发。
七轴冗余解随路径进度连续引向保存的 Home 解。直接协调回位不通时，可尝试最多抬升 20 mm、
移动 50 mm 的短过渡点，再接协调路径；不通过大幅摆动单个关节制造高空回位路径。
同步关节回位时，距 Home 已在到位容差内的轴先保持，主体协调回位后再做微小修正，避免微小调整
在贴近桌面时也引入整段多轴停止范围。
`home_path` 默认限制额外高度为起点与 Home 较高处再加 20 mm、末端偏离起终点连线最多 100 mm、
参考路径长度最多比起终点直线距离多 100 mm；分离阶段距其起点最多 50 mm，额外抬升最多 20 mm。
对应字段为 `extra_height_mm`、`corridor_radius_mm`、`extra_path_length_mm`、`separation_distance_mm`。
参考路径和最终控制器关节段均检查高度及走廊边界，无法满足时继续找路径或拒绝该方案。
`plan.json` 中保存各臂的最高点、参考路径长度和偏离距离。任一阶段找不到后续路径时，不执行部分计划。
已经位于 Home 时不添加分离动作。分离搜索失败后仍尝试原有直接回位及基座方向回退策略。
整条路径及每个控制器指令的关节范围在运动前检查完毕。执行保持 mode 0，使用
`set_servo_angle(wait=True, radius=-1)` 下发非融合的位置运动，每段只发一次目标；
完成并确认实际关节/TCP 到位后才发下一段。Home 不再切换 mode 1 或按 20 Hz 发送伺服目标。
状态读取仅用于监督停机和到位检查，不再参与固定周期目标发送。
除了采样路径，还验证每段起终点之间的完整关节区间盒，避免假设控制器严格复现主机采样时序。
整段不能证明无碰撞时，运动前拆分为较短的控制器段（最多 256 段），不盲发未验证的大段。
保留几何/基座误差、跟踪误差和停止角位移；此模式不额外叠加主机伺服反馈延迟、看门狗等待、
发送周期、迟到和双臂发送时差位移，因为完整指令范围已预检，且没有同时运动或目标流。
非 Home 的伺服执行器仍使用原余量，拒绝混用顺序控制器模式的配置。
Home 的停止角位移不再转换为整圈半径：每段运动轴的关节范围向两侧扩展对应停止上界，
静止臂仅计入允许的静止漂移和跟踪误差。因为输入的停止上界不带方向，保留双向范围，
不假设制动一定不回弹。所有关节的独立取值组合通过自适应区间细分覆盖。
每个子区间以真实中点姿态计算外形，只为该小区间的剩余不确定性增加半径；粗包络相交
会继续细分，而不是直接认定实体碰撞。确切区间内姿态碰撞或证明预算耗尽仍拒绝。
子区间的胶囊端点位移采用中点空间关节轴到端点的垂直距离及旋转弦长求和；
按远端到近端复合旋转时，已累积的位移长度保持不变，因此该和覆盖独立关节变化，
同时避免把整臂最大伸展长度重复用于每个局部区间。
静态状态检查仅包含当前位置的几何和跟踪不确定性；完整停止范围由每段发送前的检查承担。
连杆胶囊相交时，进一步使用被该胶囊包含的 URDF 碰撞网格凸包做分离平面证明，
保留原外壳厚度、配置误差和整个关节子区间的运动界限；相机及夹爪附件仍使用配置胶囊。
路径验证先覆盖较大的关节区间盒，只有无法证明时才拆分，曲线路径包含全部中间极值。
右臂本地控制器模型为 `assets/robots/xarm7/xarm7_controller_fit.urdf`；
只读控制器 FK 查询、拟合及独立验证记录在 `config/calibration/right_controller_fk_20261008/`。
该校正不修改控制器参数，也不放宽实时 FK 一致性检查。
等向位移包络不能证明分离时，另外计算沿分离轴的投影区间：覆盖各关节旋转的正弦、余弦项及
旋转复合的交叉项。胶囊使用两端点加原半径，网格使用顶点加原外壳厚度，并保留配置位置误差。
这能减少横向运动对桌面方向的误膨胀；仍覆盖完整独立关节区间，不用有限采样代替碰撞证明。
关节速度/加速度保持配置上限，并根据 TCP 杠杆上界保守降速以满足笛卡尔上限。
直接回位的两种顺序都失败后，依次尝试回退 50、100、150 mm：TCP 在各自基座 XY 平面内
朝自身基座轴线移动，保持目标高度和朝向，不越过基座轴线。每个距离尝试左先、右先回退，
另一臂保持原关节角；两臂回退后再尝试两种回 Home 顺序。
回退的笛卡尔参考路径转换为预检的控制器关节段；实际段内不承诺严格笛卡尔直线。
完整回退和回位都通过才允许执行，不执行只有回退成功的部分计划。
若起点或 Home 终点的碰撞包络已冲突，直接报告对应端点并拒绝；回退不会绕过这一检查。
所有候选路径均不通过就拒绝，不调用不受检的 `set_servo_angle` 兜底。
这是有限路径选择，不是任意障碍场景的全局绕障搜索，也未增加同臂自碰撞模型。

回位不需要相机观测；基座标定、几何、停车边界和跟踪验证仍需要有效数据。
位置模式 Home 不再要求与它无关的 `servo_commissioned` 标记；启动时要求双臂静止且控制器指令队列为空。
回位路径不建模夹持衣物/负载；此入口用于空载回位，最终会打开夹爪。
普通单臂 API、Viewer 单臂 Home 按钮及历史手工测试脚本不自动获得此双臂保护；双臂现场使用上述入口。
## Viser 包络调整

`scripts/preview_dual_home_viser.py --directory results/viser_current_home --runtime-envelopes`
提供逐部件的半径分解、逐步叠加显示，以及外形半径、停止角位移、跟踪误差和延迟参数编辑。
表格中的数值均为半径增量，额外碰撞净空单独显示，不加到两臂的半径中。
点击“保存当前调整”直接写入 `config/dual_arm.envelope_draft.json`，已有草稿会自动备份。
草稿包含完整配置和半径分解，并清除适用性已改变的验证状态；不覆盖实机配置，不发送运动指令。
重启查看器仍从 `--directory` 的快照初始化；保存的调整以草稿文件为准。
默认按 Home 顺序位置模式显示余量，六项主机流式延迟增量为零；可切换旧伺服模型对比。
“停止空间算法”可切换新关节范围检查和旧统一半径。新视图显示当前位置外形及位置误差，
紫色虚影是部分制动姿态示意，不代替独立关节全范围的证明。点击“检查当前示例段”会
对选定机械臂朝 Home 的小段及其停止范围执行与运行端相同的区间检查，不发送运动指令。
旧半径分项表保留用于对照，不代表新算法将停止和关节跟踪误差删除。

## 工作空间的单侧限制

`arms.<arm>.workspace.min_mm` 和 `max_mm` 是各机械臂自身基座坐标系下的
`[x, y, z]`，单位毫米。数组中的单个 `null` 表示该方向没有工作空间限制；
整个数组缺失或为 `null` 仍属于配置不完整。例如
`min_mm: [null, -250.578537, -0.542602]`、
`max_mm: [null, 246.447205, null]` 表示 X 不限、Y 有上下限、Z 仅有下限。
这不影响关节限制及碰撞检查。
