# 当前现场：ZED Mini + 两台 xArm7

当前自动提案使用[俯视图主流程与原图对照](topdown_folding.md)：每个观测保留两路独立判断，
俯视图为主、原图仅作对照，继续保留机器人接入阶段。

## 2026-11-04 运动学与完整路径排查

执行配置现在分别使用 `robot_model/xarm7_left_factory.urdf` 和
`robot_model/xarm7_right_factory.urdf`。这些模型由两台控制器的只读出厂校准参数生成，
原始标称 URDF 与标定归档保持不变。参数来源与设备序列号保存在同目录的
`factory_kinematics.json`；网格、TCP、环境障碍物和安全余量未调整。

控制器 FK 已包含出厂校准，而此前共用的 URDF 使用标称参数。右臂 Home 的
模型/控制器差为 4.89 mm、1.98°。换用独立模型后，两臂 Home 与各关节 ±5° 的
虚拟 FK 查询最大位置差均小于 0.051 mm，最大角度差小于 0.008°；没有实机运动。
这证明运动学模型一致性，不是物理抓点精度验收。

原生路径规划和动作解析已支持当前双七轴，不再按旧六轴/七轴拆分关节。
`ZedSingleArmRuntime(..., planning_only=True)` 可在保持未完成门禁的情况下离线规划，
此实例明确禁止连接、采集实机观测和动作下发，不能用于真实执行。

完整提案离线预检：

```bash
python scripts/plan_zed_fold.py --proposals PATH_TO_TOPDOWN_PROPOSALS \
  --robot-state PATH_TO_READ_ONLY_REPORT --output NEW_OUTPUT_DIRECTORY
```

该检查保留原提案抓点，不应用旧流程的额外 10 mm 下压。校准模型的 3 mm 空载抬升/返回
离线预检通过；已保存折袖提案的完整路径在下降抓取阶段被桌面碰撞检查拒绝。
该提案抓点及标称工具最低点 Z=9.1546 mm，首次检查时配置位置误差余量为 10 mm，
仅加入该余量就达到 Z=-0.8454 mm，低于硬桌面 Z=0。没有修改余量或障碍物来放行动作。
实机门禁仍保持未完成，完整折叠执行闭环也尚未验证。

随后按用户明确指定，将两臂原有位置误差项从 10 mm 更新为 5 mm，记录在
`position_margin_update.json`。这个值当前存放在 `safety.arms.*.base_error_mm`，
其与零值的 `geometry_error_mm`、`tracking_error_mm` 合计为 5 mm。
抓点扣除位置误差后的静态净空为 4.1546 mm；停止过程仍需独立验证。
保留现有每关节 1.25° 停止范围时，在一个抓取 IK 解的独立停止盒 128 个角点中，
发现工具最低点可至 -3.3377 mm，计入 5 mm 位置余量后为 -8.3377 mm。
这只是配置允许的停止范围诊断，不是实测刹车轨迹，也不证明所有 IK 分支都失败。
没有因此修改停止范围、桌面或验收标志。

本轮证据：`outputs/onsite_acceptance_20261104_01/` 中的
`factory_fk_consistency.json`、`calibrated_preflight/` 和 `full_fold_preflight/report.json`。

已导入 2026-11-03 的 RobotCamCalib 标定、TCP、Home 和验证证据。
本阶段已生成当前现场的双臂执行配置；机器人身份的只读核验已完成。动作执行仍受碰撞包络、
控制器跟踪/看门狗门禁保护。当前 NativeRuntime 使用 `controller_sequential`，
不依赖流式伺服的 `servo_commissioned` 标志；不要把该标志误报为此入口的阻塞项。

后续在不改变 5 mm 位置余量和每关节 1.25° 停止范围的情况下，使用经证明的单关节
末段接近/退出，完整折袖提案已通过离线规划。其 155 段关节运动也全部通过真实控制器
检查模式，证据为 `full_fold_terminal_export/` 和
`full_fold_terminal_controller_checks.json`（均位于上述现场证据目录）。
控制器检查模式没有执行运动，不能替代动态跟踪或通信中断停止验证。

再次核对 RobotCamCalib：两臂 IP、轴数、TCP、Home 关节角与执行配置一致；
已存在实际到位和返回记录，但没有找到证明当前停止范围、通信中断停止与动态跟踪
误差的专项记录。详细索引见 `robotcamcalib_record_review.json`。当前正式入口仍缺
`collision_geometry_verified`、`open_tools_and_attachments_verified`、
`controller_watchdog_verified`、`joint_segment_tracking_verified` 四项。
正式采图现在从配置的 `base_transform_provenance.common_frame` 保留坐标系标签，
供俯视规划核对；未声明坐标系的旧配置不会被自动标注为右臂底座。

已增加 [Graph 双臂分工预览](partitioned_folding.md)：按当前中心线选择一臂完成整步动作，
输出当前七轴设备的笛卡尔位姿提案。当前执行模式是每步单臂顺序动作；未选中的 xArm7
不连接、不同步、不接受命令，只以保存的关节姿态和 URDF 胶囊作为静态碰撞模型。

| 项目 | 当前设置 |
|---|---|
| 相机 | ZED Mini 11483267，固定 thirdview，校正左目 1280×720 / 30 FPS |
| 左臂 | xArm7，192.168.1.237 |
| 右臂 | xArm7，192.168.1.230 |
| TCP | 两臂均为 `[0,0,172,0,0,0]` mm/度 |
| 公共坐标 | 右臂底座；控制器 world offset 均为零 |
| 工作区 | 每臂只保留基座坐标系最小 Z；X/Y 和最大 Z 不设人工边界，IK/关节限位/静态碰撞证明仍生效 |
| 深度 | ULTRA，米，置信度阈值 50，不填洞 |

## 入口

```bash
conda activate cloth-real
cd ~/agentAuto/exp/clothAgent_real

# 默认仅校验文件、来源哈希及坐标一致性；不连接设备
python -m cloth_agent.site_observation

# 读取已有标定项目的 RGB-D；相机、机器人均不连接
python -m cloth_agent.site_observation \
  --recording ~/agentAuto/RobotCamCalib/outputs/zed_rgbd_20261103_223524_907497

# 新采图：只打开 ZED，不连接机械臂
python -m cloth_agent.site_observation --live-camera
```

输出默认在独立的 `outputs/zed_observation_<时间>/`。
若当前登录会话尚未获得 `zed` 组权限，可使用
`sg zed -c "$CONDA_PREFIX/bin/python -m cloth_agent.site_observation --live-camera"`。
SDK 缺失、相机占用或深度模式不可用时明确报错，不切换到另一台相机。

输出包括 `rgb.png`、`depth_m.npy`、`world_xyz_mm.npy`、`observation.json`，
可由现有 `cloth_agent.dual_arm.observation.Observation.load()` 读取。
RGB 和深度来自同次 grab；内参使用当次 SDK 校正左目内参，BGR 转 RGB，
只在生成公共坐标点云时将米转换成毫米。无效深度保留 NaN，不用邻点补齐。
导入历史图像保留原采集时间，不伪装成刚拍摄的场景。
观测没有实时关节反馈，标记 `execution_allowed=false`；实时执行验证会拒绝该观测。

## 已接入的数据与程序接口

- `config/site.zed_dual_xarm7_20261103.json`：当前现场入口，含两臂外参路径、TCP、Home、
  基座变换、原局部目标范围和只读控制器身份证据。它是 observation-only 格式，不能作为 DualConfig 动作配置。
- `config/dual_arm.zed_dual_xarm7_20261103.json`：由当前 ZED 标定、Home、7 轴 URDF、
  只读控制器报告和局部目标范围生成的执行配置；仍保留未完成的物理安全门禁。
- `deployment-artifacts/zed_dual_execution_readiness.json`：记录已自动填写字段和最后待实测项目。
- `scripts/run_zed_dual_arm7.sh`：当前现场正式循环包装器，自动使用新执行配置和仓库内 portable
  运行目录；它不会绕过任何安全门禁。
- `scripts/create_zed_validation_record.py`：自动生成最后现场验证的空白记录，不连接或移动机器人。
- `scripts/apply_zed_validation_record.py`：校验证据路径和完整结果后自动应用现场验证，不接受部分结果。
- `deployment-artifacts/zed_robot_capabilities.json`：从当前执行配置生成的 xArm7/xArm7 能力报告。
- `config/perception.zed_dual_xarm7_20261103.json`：现有 `PerceptionConfig.load()` 和
  `capture_two_view_rgbd()` 可直接使用的 ZED 配置。仅启用相机 A；A 表示 ZED 校正左目，
  不表示机械臂左侧。台面 ROI 尚未按当前画面确认，衣物分割效果另行验证。
- `config/calibration/zed_dual_xarm7_20261103/`：两份标定会话、验证摘要、原 Home、
  标定使用的 xArm7 URDF 及其网格、对齐和运动记录。`manifest.json` 记录来源和 SHA-256。
- `cloth_agent.zed_rgbd.ZEDRGBD`：与原相机一致的 start/read/stop 接口，
  已接入通用感知和双臂 observation 相机工厂；未指定 backend 的历史配置仍使用 RealSense。
- `cloth_agent.site_observation.base_point_mm()`：公共坐标点转到所选机械臂底座。
  `convert_pose_mm_deg()`：在 left/right/common 间转换完整 XYZ/RPY，毫米/度。

运行不依赖原 RobotCamCalib 目录（读取历史采图时除外）。复制的原会话中的绝对路径
只作来源记录；有效数据和模型路径由现场配置指向本仓库。
若今后重新标定，使用新名字导入，不覆盖旧记录：

```bash
python scripts/import_robotcamcalib.py \
  --source ~/agentAuto/RobotCamCalib --name zed_dual_xarm7_NEW_DATE
```

新名字会生成对应 site/perception 文件，随后用 `--profile config/site.<新名字>.json` 选择。

## 当前验证范围

左右臂分别有单姿态 30 帧验证，平均平移差约 2.77 / 1.21 mm，不能解释为工具精度。
双臂瓶嘴上方目视粗检已完成，仍保留 `physical_alignment_validated=false`。
两臂 Home 已到达，保存了实测关节构型，但从新起点回 Home 需要重新规划检查。
标定文件仍保留原始局部目标矩形作为来源证据；执行配置只使用每臂基座坐标系的最小 Z，
不把该矩形当成整桌工作区。X/Y 和最大 Z 由当前选定臂的 IK、关节限制及静态碰撞证明约束。

本次接入验证：

- 新增 ZED/坐标/观测测试与相关回归共 100 项通过。
- 另有 3 项旧测试依赖 `dual_arm.local.json` 的已部署现场；本机该文件仍为
  `calibration_status=incomplete`，初次运行在该检查处失败，最终针对性回归排除了这 3 项。
  没有为测试而修改现场完成状态。
- 旧 Graph 的 8 条动作模板编译通过，机器人和模型调用均为 0。
- 已保存 RGB-D 导入成功，有效深度约 77.0%；真实 ZED 采集成功，约 77.3%。
  观测可由现有 Observation 加载，导入模型的 7 个关节、8 个可视网格均可加载。
- 本地证据位于 `deployment-artifacts/zed_site_recorded_observation/`、
  `deployment-artifacts/zed_site_live_observation/` 和 `deployment-artifacts/zed_site_graph_check/`。
  本次没有连接机器人，也没有运行模型识别或真实折叠。

## 之后接入机器人

Codex 模型调用统一使用 `codex -p rbs exec …`，保留现有模型、推理强度、图片和 JSON 参数。
宿主调用使用本机 `$CODEX_HOME/rbs.config.toml`；隔离的 Graph 模型调用只导入该 profile
所需的模型/服务商字段，并补齐基础 `config.toml` 中对应的服务商定义。
服务商要求的密钥只通过指定环境变量传递，不写入项目配置、命令参数或请求记录。
缺少 profile 或密钥变量时直接报错，不退回默认服务商。隔离环境不挂载基础用户配置，
只挂载单独的 `rbs.config.toml`。本机 CLI 的 `--ignore-user-config` 也会跳过命名配置，
因此隔离调用已去掉该参数，并通过真实模型调用验证。

原机器人连接、校验、规划、执行流程保留。当前两台 xArm7 的身份已按只读报告写入执行配置，
剩余步骤只涉及现场测量和空载验证：

只读控制器入口已增加：`python -m cloth_agent.site_robot_check --output NEW_REPORT.json`。
该命令读取身份、状态、关节、TCP、负载和夹爪反馈，不改变机器人设置或下发运动。
完整首步准备见[俯视图实验说明](topdown_folding.md)。正式折叠的 `move_tcp` 已改为连续 IK 采样后的显式关节路点，绕开已确认会选错逆解分支的控制器 TCP 插补；仍须在实机下发前完成现场路径预检和人工监护验证。

1. 只读核验左右臂身份、控制箱序列号、轴数、状态、TCP、负载与零世界偏移（已记录在
   `outputs/startup_robot_state_001.json`）。
2. 使用 `config/dual_arm.zed_dual_xarm7_20261103.json` 的 Home/模型/底座变换配置机器人；当前现场以右底座为公共系，
   不能套用旧部署的左底座公共系或旧六轴关节记录。
3. 确认桌面与工具几何，并将未选中的机械臂保持在配置记录的静态姿态；路径证明会把它作为障碍物。
4. 用新采图及 active arm 的实时关节快照生成可执行观测；每一步显式选择 `--active-arm left` 或 `--active-arm right`，不会同步移动两臂。

当前 readiness 清单只剩：张开夹爪/附件包络确认、碰撞几何实测、控制器看门狗实测、低速
关节段跟踪实测，以及空载 3 mm 抬升/返回 commissioning。不得手动把这些标志改为 true。
这些门禁针对当前会运动的 active arm；未选中的 arm 只以 `static_arm_models` 中的保存
Home 姿态参加静态碰撞证明。切换 active arm 前，必须把另一条机械臂实际放回对应保存姿态。

生成现场记录模板：

```bash
python scripts/create_zed_validation_record.py \
  --output deployment-artifacts/zed_validation_001.json
```

现场填写 `checks[*].operator_result` 和 `checks[*].evidence` 后，使用：

```bash
python scripts/apply_zed_validation_record.py \
  --record deployment-artifacts/zed_validation_001.json
```

该命令会先备份配置；任何一项缺失或证据文件不存在时不会修改配置。

安全门禁完成后，首轮可用：

```bash
bash scripts/run_zed_dual_arm7.sh --active-arm left --manual-execute --max-rounds 1 --viser-port 8088
# 下一步由右臂执行时重新启动，并先确认左臂仍在保存的静态姿态：
bash scripts/run_zed_dual_arm7.sh --active-arm right --manual-execute --max-rounds 1 --viser-port 8088
```

现有 `run_arm6_folding_graph.sh` 的 9 节点 / 8 边折叠图仍可离线编译检查。
其 CollectionRuntime 绑定六轴 Arm6、旧腕部 Camera A 和旧现场动作；
不能靠把 IP 改成当前 xArm7 来运行。图的视觉状态/动作语义可以保留，
七轴动作、当前固定相机及路径的适配属于上述机器人接入阶段。
本次不将旧的 `dual_arm.local.json` 标记为可执行，也不迁用旧现场安全参数；当前现场使用
`config/dual_arm.zed_dual_xarm7_20261103.json`。
