# 当前 ZED 双 xArm7：RobotAPI 执行链

当前现场使用固定 ZED Mini 11483267、左 xArm7 192.168.1.237、右 xArm7 192.168.1.230。
两臂 TCP 为 `[0,0,172,0,0,0]`，公共坐标系为右臂底座。出厂校准 URDF、
基座变换、相机外参、工具胶囊及桌面障碍物保留。

## 单一执行链

`独立 Graph / 抓放提案 → ExperimentRunner → RobotAPI → XArmBackend`

Graph 适配器只做观测、按两 TCP 中点分工、坐标和动作格式转换；没有控制器派发实现。
每步只控制一台设备，另一臂必须保持静止；两臂状态均来自真实读取，不使用旧停车姿态代替反馈。
每轮记录独立，固定参考图不增长。旧 RobotAPI 的固定 roll/pitch 无法表示的动作明确拒绝。

`full_grasp_adapter` 是提案编译器，不是执行后端。`RobotCollision` 只检查几何和完整关节区间，
没有控制器连接、动作派发或认证标志。当前链不加载 NativeRuntime、stop_sweep、PyRoki 或 OMPL。

## 离线运行

```bash
conda activate cloth-real
bash scripts/run_arm7_partitioned_replay.sh --dry-run
python -m cloth_agent.experiments.full_grasp_adapter   --proposals deployment-artifacts/topdown_physical_trial_001/proposals_v2   --robot-state outputs/onsite_acceptance_20261104_01/robot_readonly_host.json   --output reports/robotapi_fold_NEW
```

以上不连接机器人。底层正常保护保留：工作区与速度、轴数和关节限位、TCP 和设备身份、
夹爪完成反馈、双臂/工具/桌面几何检查、完整关节区间检查，以及执行异常停止。
执行故障后不再自动开爪或回 Home；停止异常单独记录，不掩盖原错误。

## 实际控制器路径边界

旧 `validate_controller_trajectory` 在 Cartesian 路径上每隔约 10 mm 查询 IK，
且把 home 直接当成状态重置。它不返回 set_position 实际执行的关节分支，
不能证明七轴冗余分支的中间运动避开另一臂，也不能证明从实际起点到 Home 的整段运动。
返回记录明确标记 `controller_interpolation_covered=false`。

`RobotCollision.validate_joint_path` 对输入关节路径的完整独立关节区间做胶囊包络细分，
而非仅检查端点。它验证几何范围，但不把该范围冒充控制器实际路径。
当前 XArmBackend 不具备将 Cartesian 插补绑定到该范围的能力，因此真实动作返回
`CONTROLLER_PATH_UNCOVERED`，在打开动作连接前拒绝。未增加新的全局认证流程。

当前工具及固定相机支架的几何来源仍有未确认部分，原 provenance 保留；
几何余量沿用原值，没有修改 verified=false 为 true，没有伪造测量值。

## 已移除与保留的范围

移除当前 ZED 的 ZedSingleArmRuntime、PyRoki Home 特殊分支、重复 cloth_execution 包装器、
验证记录生成/应用工具和 Native 专用 ZED 启动/规划脚本。当前配置移除了停止包络准入、
servo commissioning、静态对侧臂和流式时序参数。核心实验依赖不再自动安装 OMPL/JAX/PyRoki；
Viser 为旧可视化可选依赖。

旧 Arm6/Camera A 等仍有调用者的实验保留在仓库中；其 NativeRuntime 和检查不属于当前现场执行链。
它们仍需要各自的旧配置和可选依赖，不能用当前几何配置启动。
