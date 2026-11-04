# 历史 Astra Native 循环

当前 ZED 双 xArm7 已迁移到 `ExperimentRunner → RobotAPI → XArmBackend`，
见[当前执行链](zed_current_site.md)及[独立 Graph replay](experiments/graph_policy_replay.md)。
旧 ZED 的 `--active-arm` Native 模式、静态对侧臂配置及专用包装脚本已删除。

`cloth_agent.codex_loop` 仍服务旧 Arm6/Camera A 与 Arm7/Camera B 实验。
这些旧设备入口仍使用其原有 NativeRuntime 检查，不能套用当前 ZED 配置。
保留这些有实际调用者的旧入口，不将其停止包络、流式控制参数或依赖带入当前执行链。
