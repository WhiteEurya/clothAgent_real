> **已撤回作为 baseline 部署方案。** 以下为早期自定义 Codex/手柄服务适配记录，
> 保留代码与日志供追溯，不代表官方 RAX 接入，不应按这里启动原版 baseline。
> 当前方案见 [BASELINE_CONFORMANCE.md](BASELINE_CONFORMANCE.md)。

# 受监督实机接入

代码已接入并通过离线故障注入测试；**本轮没有连接机器人、启动相机或执行物理动作**。
原生控制链路、实机急停时延和相机采集仍需现场验证，不能把离线通过当作实机认证。
默认仍为 replay。夜间入口拒绝 real_supervised。

## 两个终端启动

先接上已有配置的手柄，保持现场操作员在场、硬件急停可用。不要同时运行其他 Cloth Agent 或遥操作进程；
服务会复用现有机器人占用锁，不会抢占已有控制进程。日志目录必须是新的。

终端 1：操作员控制服务（这是显式实机连接命令，本次未执行）：

```bash
bash experiments/rax_cloth/scripts/run_control_service.sh \
  --config config/dual_arm.local.json \
  --directory /tmp/rax-cloth-control \
  --log-dir experiments/rax_cloth/runs/operator_real_001 \
  --gamepad auto --connect-real
```

服务要求交互式终端，并由现场操作员输入 `CONNECT`。`auto` 只接受恰好一个手柄；否则用
`--gamepad /dev/input/eventN` 指定已有手柄设备。使用现有 Gamepad 校准映射，不重新标定机器人。
连接时不 Home、不移动拍摄位、不自动恢复控制器 STOP。机器人必须处于现有系统要求的静止就绪状态。

终端 2：确认服务已输出 socket 地址后，启动独立 RAX session：

```bash
bash experiments/rax_cloth/scripts/run_agent_offline.sh \
  --profile chatgpt --mode real_supervised \
  --control-socket /tmp/rax-cloth-control/control.sock \
  --session experiments/rax_cloth/runs/real_supervised_001 \
  --max-model-calls 32 --max-seconds 1200 --progress-interval 5
```

脚本沿用原名，但现在明确支持 `--mode real_supervised`；未传时仍是 replay。
Agent 只拿到官方 MCP socket，拿不到操作员服务 socket、认证 capability、SDK、生产代码或机器人网络。

## 现场授权与停止

- 先松开全部手柄按键、摇杆回中。保持 LB 松开可先检查实时图像与状态，运动请求会拒绝。
- 操作员准备授权时持续按住 **LB**，摇杆保持中立；不使用 X/B/Y/A 遥操作衣物。
- 每个动作仍必须通过已有规划器的完整 SAFE 证明；持有 LB 不代表动作已经安全。
- 规划/执行中松开 LB、输入超过 250ms 未更新，或按 START，服务停止并锁定。松开再按不会继续旧动作。
- RAX 宿主每 200ms 发心跳；超过 1 秒没有心跳则停止。RPC 失败、状态过期、UNKNOWN/UNSAFE 或执行失败同样停止。
- Agent 正常结束、超时或 Ctrl-C 时，宿主先停止操作员服务，再保存结果。终端 1 Ctrl-C 也会停止。
- 停止后不自动清除控制器 STOP，不自动重新授权。由操作员通过既有流程恢复状态，再重新启动服务。
  正常退出会清理 socket/capability 目录；异常残留目录不会被自动覆盖。恢复 Agent 时仍使用同一预算与 socket 路径。
- 软件停止不是硬件急停的替代。这里的故障测试不能证明断电、内核故障、服务被 SIGKILL 等情况下的实机停止时延。

## 复用的生产接口

`supervised.NativePort` 只在操作员服务中创建现有 `NativeRuntime`、`DualArmCoordinator` 和连接对象。
`NativeRuntime.plan(..., preserve_command_boundaries=True)` 完成原有 IK、路径和停止范围碰撞证明，
再经原有 `_execute_operations()`、实时反馈及 `stop_all()` 执行；不重新实现规划器或安全检查器。
接入层只生成受限的基础 motion DSL，绝不执行 Agent 提交的 Python 字符串。

有意不调用 `NativeRuntime.observe()`，因为它会移动拍摄位；改用原有 `observation.capture()` 静止采集。
camA 使用原配置；camB 使用现有 `with_arm7_camera()` 的标定和采集配置。双臂仍共享同一控制器和世界坐标。
也不调用含可选衣物处理的 `NativeRuntime.execute()` 包装层，不使用 task-specific Runtime、抓取修复或高度重写。

世界坐标为 arm6/link_base，位置 mm、Euler xyz 度、关节度。转到现有 motion DSL 时仅做 m/rad 单位转换。
**桌面 Z=0 保持原值，不增加末端高度余量，不要求重做接触实验。**
路径验证被禁用、碰撞包络被禁用或某臂使用静态替身时，实机服务拒绝启动。

## 工具与观测绑定

- `observe(camera="camA" | "camB")`：静止采集，返回 RGB 与 observation_id。
- `robot_state()`、`coordinates()`：实时状态、坐标约定。
- `pixel_to_world(pixel=[x,y], observation_id=...)`：现有标定/实测深度查询。
- `move_tcp(arm=..., pose_mm_deg=[...], observation_id=...)`。
- `move_joints(arm=..., joints_deg=[...], observation_id=...)`。
- `gripper(arm=..., target="open" | "close", observation_id=...)`。
- `stop()`：停止并锁定服务。不存在 arm、修改安全配置或导入 SDK 的工具。

运动前后执行原有观察有效期、标定一致性与实时关节到位检查；执行后所有旧 observation_id 失效。
不能把很早的图像坐标直接用于新动作，也不能用另一张图的 ID 绕过关节一致性检查。
忙碌时不排队等待未来授权；只有 SAFE 才可能分发动作，UNKNOWN/UNSAFE 都不执行。

## 日志与限制

操作员日志目录保存 `service.jsonl`、原始 RGB/点云、motion DSL、原生产执行计划与结果。
RAX session 保存原始 RGB、MCP 审计、代码版本、模型请求、进度与 Memory。
物理动作次数和执行耗时来自服务；通信丢失无法确认时用 UNKNOWN/null，不伪造零动作或成功。
实机使用独立、初始为空的 `official-supervised-library`，不继承 replay 的颜色识别草稿。
官方 Workspace/Memory/通用 Skills 和库合并机制继续使用；没有人为植入衣物展开策略。

已经验证：现有真实配置与模型可加载、6/7 DoF 和双相机归属匹配、安全状态 measured、路径验证和碰撞包络启用；
该检查没有连接硬件。73 项项目测试覆盖协议认证、授权缺失、松开/重按、断联、过期观测、SAFE/UNSAFE/UNKNOWN、
动作数据转换和旧接口调用。还未验证：真实 socket 服务、手柄在场授权、双相机现场采集、实际动作和停止时延。
