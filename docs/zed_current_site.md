# 当前 ZED 双 xArm7：RobotAPI 执行链

当前现场使用固定 ZED Mini 11483267、左 xArm7 192.168.1.237、右 xArm7 192.168.1.230。
两臂 TCP 为 `[0,0,172,0,0,0]`，公共坐标系为右臂底座。出厂校准 URDF、
基座变换、相机外参、工具胶囊及桌面障碍物保留。

## 单一执行链

`独立 Graph / 抓放提案 → ExperimentRunner → RobotAPI → XArmBackend`

Graph 适配器只做观测、按两 TCP 中点分工、坐标和动作格式转换；没有控制器派发实现。
每步只控制一台设备，另一臂必须保持静止；两臂状态均来自真实读取，不使用旧停车姿态代替反馈。
每轮记录独立，固定参考图不增长。旧 RobotAPI 的固定 roll/pitch 无法表示的动作明确拒绝。

`full_grasp_adapter` 是提案编译器，不是执行后端。`RobotCollision` 使用已有 PyRoki 求解关节路径并检查完整关节区间，
没有控制器连接、动作派发或认证标志。当前链不加载 NativeRuntime、stop_sweep 或 OMPL。
PyRoki 仅在规划 Cartesian 目标时加载，依赖沿用 `requirements-ik.txt`。

## 离线运行

```bash
conda activate cloth-real
bash scripts/run_arm7_partitioned_replay.sh --dry-run
python -m cloth_agent.experiments.full_grasp_adapter   --proposals deployment-artifacts/topdown_physical_trial_001/proposals_v2   --robot-state outputs/onsite_acceptance_20261104_01/robot_readonly_host.json   --output reports/robotapi_fold_NEW
```

给第二条命令添加 `--plan-joints` 可进一步离线求解并检查完整路径，结果写入 `joint_plan.json`；
若路径不通过，`integration_report.json` 记录具体动作及原因，返回非零退出码。
以上不连接机器人。底层正常保护保留：工作区与速度、轴数和关节限位、TCP 和设备身份、
夹爪完成反馈、双臂/工具/桌面几何检查、完整关节区间检查，以及执行异常停止。
执行故障后不再自动开爪或回 Home；停止异常单独记录，不掩盖原错误。

## 实际控制器路径边界

当前双臂 `move` 使用 Cartesian 插值点与 PyRoki 连续种子求解，随后直接下发关节目标，
不再调用 `set_position` 重新选取 IK 分支。Home 同样从实际关节状态检查整段路径。
`RobotCollision.validate_joint_path` 检查每段全部独立关节范围，包括原配置的 0.2° 容差，
并对工具/桌面和原工作区边界检查完整运动范围。PyRoki 在此用于 IK，不是自动绕障规划器；
插值路径碰撞或无解时拒绝该动作，不会自动改变抓取点或降低约束。

原 XArmBackend 使用 `set_servo_angle(radius=-1, wait=True)` 逐段下发，禁止混合和排队，
确认目标到达并静止后才发下一段。这沿用 SDK 的 MoveJoint 方式，
参见 [SDK 接口文档](https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/xarm/wrapper/xarm_api.py)。
move 的关节速度和加速度同时受原关节限制及保守 TCP 运动上界限制，运行会比连续 Cartesian 命令慢；
Home 沿用原关节速度和加速度。
执行期间读取两臂关节及故障状态；活动臂越出检查范围、对侧臂移动、反馈失效或命令失败时停止。
软件监测用于异常停止，不替代设备硬件保护，也不声称给出了停止包络证明。

对侧臂按步骤开始的实际关节姿态作为静态障碍，保留其工具包络。
按操作者确认，现有边界已排除夹爪与固定相机碰撞；相机支架测量记录不作为额外准入条件。
原几何来源和余量保留，没有把历史 verified=false 改成 true。

## 已移除与保留的范围

移除当前 ZED 的 ZedSingleArmRuntime、PyRoki Home 特殊分支、重复 cloth_execution 包装器、
验证记录生成/应用工具和 Native 专用 ZED 启动/规划脚本。当前配置移除了停止包络准入、
servo commissioning、静态对侧臂和流式时序参数。核心实验依赖不再自动安装 OMPL/JAX/PyRoki；
Viser 为旧可视化可选依赖。

旧 Arm6/Camera A 等仍有调用者的实验保留在仓库中；其 NativeRuntime 和检查不属于当前现场执行链。
它们仍需要各自的旧配置和可选依赖，不能用当前几何配置启动。
