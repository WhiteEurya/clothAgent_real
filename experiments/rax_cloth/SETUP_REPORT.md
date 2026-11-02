## 当前入口：先机械 reset，再探索；人工等待暂停模型（2026-10-30）

最新用户要求开头直接复用现有抓起/落下 reset，而非再次请求人工 reset。
`run_arm6_onsite.sh` 现在默认附加 `--initial-reset`。它通过当前唯一 backend 的受保护
reset 流程初始化场景，成功才启动 Agent；机器人操作仍需要完整安全证明，臂仅 arm6。
此动作是环境初始化，不是 RAX 自行发现的策略或任务成功。后续 reset 仍由现场操作者
处理。新任务说明明确初始场景已由 host reset，要求重新观察后开始任务。

初始 reset 记录 `initial-reset.json`，已有成功记录不在 resume 时重复；失败或中断
记录要求检查，不自动重放。运行中每五秒显示初始化状态，Ctrl+C/stop/预算到期均停止。
原有 reset 的视觉模型、采样、录制、参数保持复用；其模型开销在 reset 日志中单独记录，
不冒充主 Agent 的调用预算或学到的技能。

人工请求暂停时，模型 relay 在 reserve 和网络调用之前等待现场回复；不会像
arm6_onsite_20261030_184625_1374587 那样在 pending reset 时继续耗尽32次调用。
已经发送的在途请求不会被这道门控追溯取消。人工等待仍计入 session 时间预算。
回车后先恢复有限许可并写官方回复，再放行后续模型请求。不改官方探索/编译算法。

validation/pytest_initial_reset_wait.txt：126 passed；覆盖初始 reset 只执行一次、
失败不重试、预先取消不执行，以及实际 relay handler 在等待时不预占次数、不调用
provider，回车后只放行一次。未执行真实初始 reset 或该新版端到端物理实验。

使用新 session（脚本自动生成）：

```bash
cd /home/sja/clothAgent_real
bash experiments/rax_cloth/scripts/run_arm6_onsite.sh
```

该命令现在会在 Agent 启动之前尝试实际机械 reset，不再是纯观察入口。仅验证环境可
追加 --preflight-only，此时不会初始化场景或执行运动。旧目录仍保留原始日志。

## 当前推荐：单终端现场运行，人工 reset（2026-10-30）

按用户最新要求，使用以下一个命令：

```bash
bash experiments/rax_cloth/scripts/run_arm6_onsite.sh
```

命令创建新的 arm6/camA session，默认本机 ChatGPT 登录、32 次模型调用、1200 秒。
运行该前台命令即提供本轮运动许可，不需要另开终端或输入 ARM。Host 仍使用官方
arming 文件及原有安全判定；Agent 不可写该文件。此模式要求前台 TTY，不支持后台
或无人运行。时间上限仍是上限，不保证任务成功，也不强制 Agent 用满时间。

人工 reset 通过官方 request_operator 请求进行。在处理请求前 host 先撤销运动许可；
终端打印请求，你整理好衣服、手离开后按回车，host 恢复剩余本轮时间的许可并写入官方
reply。等待期限使用剩余 session 预算（单次至多一小时），不再沿用 Agent 的短等待。
自动 xarm_reset 在此模式不暴露且后端拒绝，避免人工 reset 时 Agent 又调用机器人 reset。
其他故障请求同样可以输入说明回复，但停止锁存后不能通过回车重新启用运动。

输入 stop/q、Ctrl+C、SIGHUP 或 stdin EOF 会结束本轮并进入既有停止清理流程。
终端连通不等于物理在场传感器，本模式依赖用户明确表示在场监看；不声称能检测人离开。
碰撞证明、UNKNOWN 拒绝、故障停止、arm6 限制和预算仍保留。未修改官方探索提示、
Skills 和生产代码。官方 operator 请求等待时长及现场 CLI 属于明确的适配层变更。

可追加 --max-model-calls / --max-seconds / --session；恢复已有现场 session 时必须传
原 session 路径、相同预算及 --resume。默认每次创建新目录，避免误读旧双臂/授权提示。
仅预检可追加 --preflight-only，不会启动模型、机器人或许可。

验证：validation/pytest_onsite_console.txt，122 passed，覆盖启动许可、人工 reset
暂停与回复、EOF/stop、故障不能重新授权、自动 reset 工具隐藏及后端拒绝。脚本语法通过。
没有执行现场启动、真实恢复或实机运动；不以这些离线测试宣称物理闭环已验证。

之前新增的自动 reset（普通非 onsite 模式 xarm_reset）复用了 random_fold_reset.run_reset，
以 host-only 适配保证 arm6、安全检查、单一控制入口，默认参数/重试/记录沿用现有函数；
模型开销独立记入 reset 目录，任务动作和 reset 动作分开计数。此次人工模式不用该功能。

## 启动时恢复无故障 STOP（用户要求，2026-10-30）

实机 RAX backend 创建 NativePort 时启用一次性的 STOP 恢复。连接仍通过现有 XArmConnection；仅对 arm6 的 state 4 调用现有 prepare_position()。现有实现检查无错误/警告、静止、空命令队列、未取消，再执行 motion_enable 与 set_state(0)，并检查恢复结果；该原有准备过程也设置/使能夹爪，不发开合或位置移动命令。必要时使用其已有 STOP 模式恢复逻辑。不会调用错误清除、Home 或 Agent 程序。

此功能不等于 arm 授权；现有每动作安全证明与授权门控保持不变。同一 backend 的 hold/错误/授权丢失仍保持停止锁存，不循环自动恢复。只读 check_real_setup.sh 的 NativePort 默认 recover_stop=False，不改变控制器状态。恢复状态保存到 robot/startup_stop_recovery.json。

这里根据控制器报告的状态、错误、队列等条件判定，不声称能独立识别过去 STOP 的物理来源。现存非零错误/警告不会被清除。

validation/pytest_startup_stop_recovery.txt：107 passed。新增测试使用原有 XArmConnection.prepare_position 验证正常 STOP 恢复、错误/队列/运动中/取消拒绝，以及 normal 时不使能。未执行真实机器人恢复或运动。此前文档中“所有 STOP 都需要人工恢复”的描述被此更新取代。

## Read-only session shutdown fix (2026-10-30)

`arm6_real_002` ended after 136 seconds / 8 model calls with zero physical attempts.
First state request failed with `left: controller fault/state 4, [0, 0]`; image request
was subsequently refused by the stopped backend, and the operator request expired.
This is different from the previous unarmed/reset wait. The codes alone do not prove
a new hardware fault or what caused state 4.

Inspection found that the previous backend unconditionally sent STOP even on normal
closure of a read-only session. Such a close can explain a subsequent STOP state;
the exact controller transition is not independently recorded. Normal close now only
sends STOP if a dispatch was attempted or an operation is active. It still disarms
and disconnects. Explicit hold, authorization loss and error paths retain stopping.
No existing controller STOP is reset automatically.

`validation/pytest_readonly_close.txt`: 101 passed, including read-only close without
STOP and close-after-dispatch with STOP. Actual controller recovery and movement are
not performed or verified. The existing state 4 needs onsite recovery before restart.

## 启动流程修复：共享库、状态与操作员通知（2026-10-30）

针对 `arm6_real_001` 已观测到的退出原因：

- 原始共享库只读绑定到 `/skill-library-origin`，在沙箱内修正 clone 的 remote，并在启动 probe 中实际 fetch。probe 还检查共享库不可写；不暴露宿主项目父目录，不在宿主执行 Agent clone 的 Git 配置。官方 merge 和校验保持不变。
- `safety=UNKNOWN` 保留，新增 `safety_scope=per_proposed_action`、解释字段和 `connection_state`。UNKNOWN 不再被描述为控制器故障；仍不能凭 UNKNOWN 执行动作。
- 终端轮询官方 operator requests，显示内容、请求编号、回复命令和超时/已回复状态。不会代答、自动 reset、自动 arm 或改写官方探索流程。
- 实机 session 建立时明确显示尚未授权运动。不会承诺启动命令等于机器人立即移动。

验证：`validation/pytest_operator_visibility.txt`，99 passed。共享库 fetch 的真实 sandbox 验收仍需在本机终端运行；本助手环境禁止 bwrap 网络命名空间，不能用单元测试冒充该验证。

仅做启动预检（不调用模型、不连接机器人），使用新的目录：

```bash
bash experiments/rax_cloth/scripts/run_rax_codex.sh \
  --profile chatgpt --mode real_supervised \
  --robot-config config/dual_arm.local.json \
  --session experiments/rax_cloth/runs/arm6_real_002 \
  --max-model-calls 32 --max-seconds 1200 --preflight-only
```

通过后同一命令移除 `--preflight-only` 启动模型。这里的预检不验证机械动作、现场停止时延或持续在场检测；这些仍未完成。较早的 arm6_live_read_001_001 已有主机端 live_reads_passed 记录，只能证明 arm6 状态/camA 获取链路。

## 当前任务范围更新：仅 arm6 / camA（2026-10-30）

用户将本任务改为单臂。当前工具只允许 arm6 动作和 camA 观测；不连接 arm7 控制器、不读取 arm7 实时状态、不初始化 camB。运行不要求 arm7 在线或准备就绪。

现有双臂几何规划接口仍保留 arm7 的静态障碍模型，使用项目已有 `data/robot/xarm7_perception_pose.json`，明确标为 assumed static obstacle，而非实时反馈。该几何假设要求闲置臂保持记录的停放姿态；实际改变其位置会使该假设失效。没有使用 CollectionRuntime 的关闭路径验证/碰撞包络设置；SAFE/UNSAFE/UNKNOWN、Z=0 和现有安全验证不变。

`robot_scope=arm6` 写入新 session 配置；请新建 session，不沿用之前双臂工作空间或其旧提示。已有实验记录未改写。只读检查入口现在仅连接 arm6、采集 camA：

```bash
bash experiments/rax_cloth/scripts/check_real_setup.sh --live-read \
  --output experiments/rax_cloth/runs/arm6_live_read_001
```

验证：`validation/pytest_arm6_scope.txt`，98 passed；包含生产 connect 函数仅创建 arm6 连接、arm7 静态对象拒绝控制命令、工具拒绝 arm7/camB。`validation/preflight_arm6_001/preflight.json` 配置检查通过。尚未现场验证或执行物理动作。下文双臂章节为此前阶段记录，以此更新为准。

> 最新：用户已允许 Codex 替换 Claude Code。当前为 RAX-Codex/xArm：官方工作流、registry、Frontend 与 arming；
> 不使用第二套操作员服务。详见 BASELINE_CONFORMANCE.md。新路径离线测试通过，端到端/实机未验证。

> 当前结论：原版 RAX baseline 未部署完成；此前 Codex/自定义实机服务测试属于适配实验。
> 详见 [BASELINE_CONFORMANCE.md](BASELINE_CONFORMANCE.md)。历史结果不改写、不冒充原版结果。

# RAX-Cloth Setting report — updated after official checkout

## Current status

Official `real/rainbow` source is now present at `vendor/rax/`, pinned to
`36c926ca3cf15921c6fc746a4bf309c26b0aa3c4`. The checkout is clean and unmodified.
The previous GitHub download blocker is resolved by the user's terminal download.

The adapter now runs the **official Configuration, Workspace builder, Session,
MCP Frontend and Skill Library checkout**, with the local ClothBackend.
Mock and real-image replay smoke tests and repeated-session resume have passed.
This is still an offline setting: no model/agent exploration, robot connection,
physical action or T-shirt flattening success has occurred.

## Actual upstream reuse and adaptation boundary

- `rax_cloth/upstream.py` verifies the Git commit and clean checkout before loading
  official `services.backends`; cloth is registered in process only.
- `rax_cloth/official.py` registers robot facts in the official workspace generator,
  renders its original universal control/memory/continual-learning skills, creates
  an empty Git skill library using the official template, and invokes official checkout.
- The official Session owns its lock, state and audit. The official Frontend validates,
  serializes and records calls to ClothBackend. Model/Gemini support is not enabled.
- Generated workspace changes are explicit: offline mode preamble, xArm interface
  documentation and no offline skill promotion. Each session saves
  `workspace-build.json` and `workspace-overrides.diff`. No garment strategies,
  existing Cloth Agent prompts, fixed trajectories or folding skills are supplied.
- Skill library merge is deliberately disabled in offline sessions. Initial learned
  skills are empty. Mock/replay cannot produce verified physical skills.
- Historical web snapshots in `rax_cloth/upstream_reference/` are retained for provenance;
  they are no longer the active backend registry when the full checkout is available.

The real/rainbow branch supports Claude Code (`CLIS=('claude',)`). Codex is **not**
silently substituted. The agent launcher, credentials/model relay and isolated code
execution are still not connected/validated for this adaptation.

Sources: [official branch](https://github.com/robot-agent-exploration/rax/tree/real/rainbow),
[adaptation guide](https://github.com/robot-agent-exploration/rax/blob/main/docs/REAL_ROBOT_ADAPTATION.md).
All implementation references above were now checked against the local pinned source.

## Local API re-audit — correction to the earlier gap assessment

The project already contains the model, protected execution, observation, planning,
IK, cancellation and operator-watchdog implementations needed as building blocks.
Earlier statements that these interfaces were missing were too broad: **the gap is
RAX integration and validation, not absence of local implementations**. The source
inventory, exact symbols, line numbers and hashes are recorded in
`validation/existing_interfaces.json`.

| Capability | Existing reusable implementation |
|---|---|
| Codex multimodal model | `experiments/discovery_model.py:CodexMultimodal.invoke` (under cloth_agent); prompt/images, returned text, usage, elapsed time, timeout and failure handling |
| Model CLI/logging | `codex_astra_pipeline.py:build_command/run`; existing JSON events, token/time accounting |
| Claude transport | `planner_backend.py:LocalClaudeBackend/RemoteClaudeBackend`; `harness/model.py:RuntimeClaude` |
| Coding-agent namespace | `experiments/trajectory_memory_agent_probe.py:isolated_command`; mounts Codex package and tools without the host project; task-specific prompts/memory are excluded from RAX reuse |
| Protected motion | `codex_native_runtime.py:NativeRuntime.plan/execute`; existing certified joint/TCP paths, ordered grippers, pre-dispatch rechecks and failure stops |
| State/observation | `NativeRuntime.capture_current`, `DualArmCoordinator.snapshots`, `Observation.sample`; camB mapping in `codex_camera_b.py` |
| Operator input/stop | `gamepad_teleop.py:InputGuard`, `gamepad_arm7.py:Arm7Jogger._watch`, `gamepad_continuous.py:ContinuousArm7._watch_loop` |
| Cancellation/fresh feedback | `sequential_home.py:wait_operation/check_feedback`, `DualArmCoordinator.validate_fresh/stop_all` |

Important integration boundaries found in the actual source:

- `NativeRuntime.observe()` may connect and move to the perception pose. It must not
  be exposed as a supposedly read-only observe tool. `capture_current()` is the
  existing stationary-capture method after operator-controlled initialization.
- Use the generic NativeRuntime/basic command layer, not the task subclasses:
  DualArmRuntime adds folding context and 10mm grasp lowering; Arm6Runtime adds
  15mm lowering and disables selected uncertainty envelopes; Arm7Runtime also
  adjusts active-arm envelope policy. Those are existing experiment choices, not
  acceptable defaults for this RAX baseline. Do not edit those production classes.
- Base execute is not policy-free under every config: a shared_arm7_target_z mode
  rewrites heights; auto_grasp_debug enables recovery routines. The existing loaded
  config and optional flags must be checked before service wiring. Preserve Z=0
  and no additional height policy for RAX.
- NativeRuntime.execute uses `check_age=False` for its historical observation while
  checking live stationary joints; the RAX service must explicitly enforce the
  required freshness of authorization/state. The file lock named `_lease` provides
  ownership, not proof that an onsite operator remains present.
- Existing InputGuard/watchdogs provide reusable authorization primitives, but they
  are attached to gamepad workflows. RAX must connect their lifecycle to its single
  owner, cancellation and MCP disconnect without copying teleoperation/task policy.
- Existing model wrappers do not by themselves equal the original RAX coding-agent
  Harness. discovery_model disables shell tools; the fuller coding-agent isolation
  mounts credentials and is not an independently network-isolated robot firewall.
  Reuse generic transports where appropriate while preserving the official workflow
  and separately identifying any Codex launcher adaptation.

Actual validation in this audit: **12 existing motion/parser/fault-injection tests
passed**, plus **4 existing input guard/watchdog/model-error tests passed**. Tests
used synthetic models, fake connections and mocked model processes; no robot,
camera, remote model or paid inference was contacted. No production source changed.

## Environment compatibility findings

1. System Git is 2.25.1; upstream `git init -b main` fails. The adaptation initializes
   the same empty library with `git init` + `git symbolic-ref HEAD refs/heads/main`,
   then uses upstream checkout. No global Git configuration or upstream source changed.
2. This execution sandbox refuses Unix socket `bind` with EPERM. Official smoke tries
   the normal Unix socket path first; here it falls back to OS pipes. The official
   server `serve()` and client `request/call/value` methods are retained; only the
   client transport initializer/close are adapted. **The user subsequently ran the same
   smoke command in the local terminal and the original Unix socket path PASSED.**
   The saved result was independently read from the workspace: transport is
   `real Unix socket MCP`, elapsed time 0.11782944947481155 s, zero physical actions
   and zero LLM calls. See `validation/official_mock_native.json`. This resolves the
   socket transport blocker on the host, not the assistant sandbox restriction or
   the separate Agent isolation requirements.
3. Bubblewrap cannot create the required network namespace. `--driver agent` remains
   fail-closed. A Python venv or AST check is not treated as an Agent security boundary.
4. The isolated host venv is `.venv-host/`, sharing the existing cali scientific
   dependencies. Missing Jinja2 3.1.6 / MarkupSafe 3.0.3 were copied from the locally
   installed base distributions into this venv only; MarkupSafe uses its pure Python
   fallback. Source/version/file provenance is in `validation/local_host_dependencies.json`.
   No base or cali package was modified. This host venv is **not** an Agent environment.

## Local robot API audit

| Capability | Existing implementation and adapter status |
|---|---|
| Mapping | left=arm6 (6DoF), camA; right=arm7 (7DoF), camB. Names are robot identifiers, not image sides. |
| Coordinates | `dual_arm/config.py:ArmConfig.base_pose`, `geometry.py:pose_matrix/matrix_pose`; world=arm6 link_base, mm and Euler xyz degrees. Actual read-only transform test passed. |
| Table contact | Z=0 mm retained. No extra gripper-height allowance or repeat contact experiment. |
| Observation | `dual_arm/observation.py:Observation.load/sample` actually used in replay; original RGB/cloud/hash/provenance preserved. capture/create_camera inspected, not invoked. |
| Safety | `stop_sweep.py:verify_joint_box/require_safe` delegated unchanged; SAFE passes, UNSAFE/UNKNOWN reject. |
| Planning / IK | Host-only delegation to existing `DualArmPlanner.paths.joint_path/cartesian/solve_grasp_pose`; this subsystem uses m/rad. No new planner and no numerical planning claim. |
| Motion / gripper | `XArmConnection` and `DualArmCoordinator` inspected but not instantiated. Existing protected service bridge with operator lease/watchdog is still pending. |

The current config permits null grasp orientation for read-only geometry via its
existing `homing_only=True` loader option. This grants no execution authorization.
The independent `motion/preflight.py` explicitly does not authorize physical execution;
its restriction is preserved. This does not imply the existing Cloth Agent lacks its
own working online execution path.

## Modes and logging

- mock: synthetic RGB interface fixture, no cloth physics; actions NOT_EXECUTED.
- replay: real historical RGB/calibrated cloud, no counterfactual action feedback;
  example manifest contains one observation and no demonstrations/action strategy.
- sim: rejected; no action-conditioned cloth simulator has been validated. Existing
  simulated robot classes do not count as a cloth simulator.
- real_supervised: hard disabled; no arm tool or direct SDK access. Future execution
  must use the existing protected service and onsite operator authorization/watchdog.

The previous `run_overnight.sh` remains a bounded offline diagnostic runner with
per-episode observations/actions/results, code hashes, budget accounting, SIGTERM and
resume. It is not an autonomous RAX exploration loop. Default maximum duration is 8h;
no 8-hour run was performed. The diagnostic runner does not call providers; actual LLM calls are 0. The new experimental Codex entrypoint below is separate.

The new `run_official_smoke.sh` is a finite official-component validation entrypoint,
with original `mcp.jsonl`, workspace/skills/library, `official-result.json`, and summary.
It safely reopens the same compatible session and appends observations/audit. It does
not replace the bounded overnight runner or claim full launcher resume validation.

## Executed tests

- `validation/pytest.txt`: earlier **28 passed** (local adapter suite), including official
  workspace/Frontend/client, mock/replay and repeated-session resume.
- `validation/upstream_frontend_tests.txt`: **8 passed, 2 failed** in the unmodified
  official frontend suite. Failures are Unix socket bind EPERM and unsupported
  `git init -b`; do not report the full upstream suite as passing.
- `validation/official_mock.json`, `validation/official_replay.json`: actual official
  component smoke outcomes from the assistant sandbox, using pipe adaptation.
  `validation/official_mock_native.json` additionally records the user-terminal
  successful original Unix socket run; these are distinct execution contexts.
- Current sessions: `runs/official_mock/`, `runs/official_replay/`.
- `validation/SETUP_REPORT.initial.md` and earlier logs preserve the initial stage,
  including its then-unresolved download status. They are historical, not current.

No model reasoning, physical task success, unseen-cloth evaluation, real watchdog,
full original launcher, or generated-program sandbox execution was tested.

## Tested command

From the repository root:

```bash
bash experiments/rax_cloth/scripts/run_official_smoke.sh --session experiments/rax_cloth/runs/official_mock
```

Replay:

```bash
bash experiments/rax_cloth/scripts/run_official_smoke.sh --session experiments/rax_cloth/runs/official_replay --mode replay --replay-manifest experiments/rax_cloth/configs/replay.example.json
```

Both commands ran successfully, including subsequent reuse of the same sessions.
After the provenance upgrade, old diagnostic sessions containing the earlier null
commit intentionally fail immutable-config resume; use a new diagnostic session.
Do not rewrite those historical manifests to make them appear to use this checkout.

## Remaining work

- Original Unix socket transport is now validated on the host. Still validate Agent
  OS/file/network isolation in a suitable execution environment. Keep SDK, production repository, operator files,
  controller networks and credentials outside Agent access.
- Validate the newly implemented, explicitly labeled Codex adaptation below on the host,
  including live Responses compatibility, full CLI/program execution and thread resume.
  The original Claude launcher remains unchanged and has not been run.
- Wire generic primitive RPC to the existing NativeRuntime/Coordinator and reuse the
  discovered InputGuard/watchdog building blocks for onsite authorization. Validate
  the combined execution-time authorization, state freshness and disconnect stops.
  Maintain SAFE/UNSAFE/UNKNOWN and all existing collision constraints.
- Only then run agent-driven Explore/Compile/Evaluate/Improve; real feedback requires
  onsite supervised interaction unless a credible cloth simulator is later verified.

All production Cloth Agent code and global robot settings remain untouched.

## Continued implementation: Codex offline launcher

Added `scripts/run_agent_offline.sh`, `rax_cloth/codex_agent.py`,
`rax_cloth/responses_relay.py` and isolated `agent_assets/`.
This is an **experimental new CLI/transport adaptation**, not an execution of the
original Claude Code harness. Official sources remain unmodified at the pinned commit.
Official workspace, universal workflow skills, session ownership, Frontend, Context,
MCP bridge and bounded HTTP server are reused. Existing project model selection and
Codex event completion classification are reused; no task-specific garment prompt,
controller, demonstrations or strategy is imported into the agent workspace.

The host runs in the separate venv. The agent runs in an empty-root bubblewrap mount
and network namespace with system Python/NumPy/OpenCV and the standalone Codex package.
Only its workspace and private Codex state are writable host mounts. RGB frames,
operator CLI config and original Python API are read-only mounts. Host login credentials
stay outside the sandbox; a restricted relay injects them for one fixed HTTPS endpoint.
Both preflight and real launch use a clean subprocess environment. Old bubblewrap's
lack of `--clearenv` is handled without inheriting the host environment.

Codex custom provider and MCP configuration follow the official
[configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
and [MCP documentation](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).
The ChatGPT-login HTTPS route is a separate experimental compatibility path, not a
claim of documented public API stability or a remotely validated provider integration.
No credentials were printed or copied into the workspace; tests use fake credentials.

Each invocation runs one bounded integration episode, saves raw CLI JSON events,
model requests/stream responses, elapsed request time/usage when supplied, original
MCP audit and observations, and source hashes before model requests and at completion.
SIGINT/SIGTERM stop the process group. A crash reservation still consumes a model call;
resume retains usage, recovers a saved thread from events, and conservatively charges
elapsed time since an interrupted attempt. Source snapshots are read as data without
following symlinks or executing code on the host. Test evidence remains in agent/tool
transcripts; host snapshotting itself is not a program correctness test.

Current limitations:
- ChatGPT login has no reliable dollar cost accounting here: cost is `null`, not zero;
  `--budget-usd` is rejected. This path enforces call/time caps, not a USD cap.
- Provider acceptance/auth freshness, real agent code execution, nested Codex sandbox
  compatibility and end-to-end CLI resume still require a successful host run.
- No automatic multi-episode model-driven overnight runner is claimed; the existing
  overnight script remains diagnostic only. No trustworthy cloth simulator was found.
- No generated controller has yet run, and no physical success was evaluated.

Executed validation:
- `validation/pytest_codex.txt`: **41 passed**, including 13 new cases covering request
  restrictions, mocked streaming/usage, persisted caps, safe archives, crash recovery,
  configuration/mount policy and preflight failure before model construction.
- `validation/codex_config_parse.json`: installed Codex CLI accepts the generated
  provider/MCP config (`codex mcp list --json`); no model request or MCP launch.
- `validation/official_replay_latest.json`: official replay component smoke passed,
  historical RGB, zero physical actions and zero LLM calls; pipe adaptation in this sandbox.
- `runs/codex_preflight/codex-adaptation/isolation.json`: real attempted Agent preflight
  **failed** with `bwrap: loopback: Failed to create NETLINK_ROUTE socket: Operation not permitted`.
  The launcher stopped before creating a model relay or connecting hardware.
- Upstream `git status --porcelain` remains empty.

Next host-terminal validation command (prepared, **not yet proven end-to-end**):

```bash
bash experiments/rax_cloth/scripts/run_agent_offline.sh --session experiments/rax_cloth/runs/codex_replay_trial --mode replay --max-model-calls 8 --max-seconds 600
```

The same command with `--preflight-only` checks isolation without starting a model.
Use the same parameters plus `--resume` only after a recorded model attempt with a
valid thread id. The official smoke commands above remain the tested successful commands.

## Host trial diagnosis and socket/budget correction

The user's `runs/codex_replay_trial` actually passed the outer isolation probe and
made eight successful HTTP 200 model requests. Direct MCP observations and agent-written
perception code execution worked. The ninth request was denied by the configured local
8-call cap (not a provider rate limit), and the final turn failed. The agent also reported
that its Python Context connection was denied by the inner Codex network policy.
See `validation/host_codex_trial_diagnosis.json` for compact evidence and source hashes.
Historical logs were not rewritten. Full controller completion remains unverified.

The generated inner `workspace-write` config now permits socket networking inside the
mandatory outer `--unshare-all` namespace; host/robot network and filesystem access
remain excluded. Source: [OpenAI Docs security](https://learn.chatgpt.com/docs/agent-approvals-security).
Local cap denials are persisted and reported separately from upstream HTTP failures;
unsuccessful attempts return structured results and exit code 2 rather than an opaque traceback.
No automatic cap increase or reset was added. New host validation: use session
`codex_replay_trial_v2`, `--max-model-calls 32 --max-seconds 1200`.
This fix still needs a host run; the assistant environment cannot create the outer namespace.

## Second host trial: Python MCP and agent code iteration verified

The user ran `codex_replay_trial_v2` with 32 calls but **300 seconds**. The timeout
stopped the session after 300.40 seconds with nine requests reserved. Within the
isolated CLI shell, the generated controller successfully read historical observations
and queried calibrated pixel coordinates using the official Context. The Agent's
first test run had 3 failures out of 6; it fixed the code and added a test, then the
recorded rerun passed all 7 tests. Controller output still says physical success
NOT_EVALUATED and execution NOT_REQUESTED. Memory was empty at timeout and no final
turn completed. Evidence: `validation/host_codex_trial_v2_diagnosis.json` and original
session events. These are agent-produced offline tests, not physical validation.

The trial also exposed upstream status reporting the unused Claude budget (zero calls,
$10 limit). The Codex frontend now substitutes its actual relay call/time budget and
null dollar accounting; this does not modify historical logs or upstream source.
Validation: `validation/pytest_codex_status_fix.txt` (47 passed).
A new session with `--max-model-calls 32 --max-seconds 1200` allows more time for memory
and final reporting; success is not guaranteed. The exhausted 300-second session cannot
be extended by `--resume`; original budgets remain immutable.

## Operator-requested local API profile

Default agent selection now follows local `codex -p api`. The installed CLI uses separate
`$CODEX_HOME/api.config.toml` profile files, per the
[official profile documentation](https://learn.chatgpt.com/docs/config-file/config-advanced).
The current host profile selects gpt-6-astra/high and DMXAPI Responses. The agent's CLI
actually receives `--profile api` with a sanitized read-only profile pointing to the host
budget relay. The host retains original HTTPS routing and command-based authentication;
no credential or original personal config is mounted into the agent. Existing mount and
network isolation remain enforced. Unrelated personal prompts, tools and permission
settings are not inherited. A missing profile does not fall back to ChatGPT login.

Added `rax_cloth/codex_profile.py`; extended the relay and launcher/profile mount.
`--profile api` is now default; `--profile chatgpt` explicitly selects the old login path.
Profile/provider identity is immutable per session. No old run or global config was changed.
Dollar accounting remains unavailable without configured provider pricing; null costs
and rejected USD caps are retained rather than treating API calls as free.

Validation: `validation/pytest_codex_api_profile.txt` (51 passed), including fake-key
profile authentication, fixed endpoint forwarding, no credential logs, missing-profile
failure and no ChatGPT fallback. `validation/codex_api_profile_preflight.json` records
successful actual local config parsing and credential-helper validation without exposing
the credential. **Zero real provider requests were made for this new profile path.**
Use a new session, e.g. `--profile api --session experiments/rax_cloth/runs/codex_api_trial
--mode replay --max-model-calls 32 --max-seconds 1200` for host validation.

## API host run and timeout correction

`codex_api_trial` made five successful HTTP 200 API requests; Python Context observation
also worked. The sixth request failed with TimeoutError before the relay sent response
headers to the CLI. The old logs do not distinguish connection/send from response-header
waiting, so the exact upstream phase cannot be asserted retrospectively. No controller
file was saved in this attempt. This validates API transport connectivity, not completion.

The relay's previous 45-second socket timeout is now 300 seconds, capped by remaining
session time. The CLI idle timeout is 360 seconds so the relay can report its timeout
first. Failures record connect/send, response_headers or response_stream phase. There
are no hidden retries, and reservations are not refunded. Tests cover short remaining
budgets, long reasoning waits and phase reporting: `validation/pytest_codex_timeout_fix.txt`
(53 passed). This timeout fix has not yet been tested against the real provider.

Resume the same API session with the original profile/mode/caps plus `--resume`; this
retains its thread, elapsed time and six reserved requests. Do not change the immutable
1200-second / 32-call budgets. See `validation/api_timeout_diagnosis.json`.

## Live debugging/progress output

Added `rax_cloth/progress.py` and launcher polling of incremental Codex JSON events.
Stage transitions print immediately; heartbeat defaults to 10 seconds, configurable
with `--progress-interval` (1..300). Reports tool/command activity, provider request
phase and wait duration, cumulative time used across resumes, remaining time budget,
and calls used/limit. Remaining time is a deadline countdown, never an estimated task
completion time. Output goes to stderr; final result JSON remains on stdout.

Progress is saved to `codex-adaptation/progress.jsonl` and latest `progress.json`.
Raw image/tool payloads are not printed; command summaries are truncated and terminal
control sequences removed. Existing full audit/events remain unchanged. This monitor
makes no extra model calls. Running old processes need a subsequent launch/resume to
load it. Tests: `validation/pytest_progress.txt` (56 passed), covering split JSON events,
resume countdown, provider-wait heartbeat, output trimming and exclusion of image data.
`validation/progress_demo.txt` is explicitly a synthetic UI test, not an agent run.

## Direct local Codex diagnostic requested by operator

Attempted `codex exec -p api --ephemeral --sandbox read-only` directly, with a
minimal exact-answer prompt, no RAX relay, and an empty working directory.
The original home failed before model work with a read-only in-process app-server
initialization error (`validation/direct_codex_api/`). Retrying with temporary writable
Codex state and the same local base/profile files reached thread startup but repeatedly
reported `Connection failed: error sending request` in this assistant execution context
(`validation/direct_codex_api_writable/`). This is not a valid provider latency measurement
and does not establish a provider outage on the user's host. No successful answer was seen.

Added `scripts/test_local_codex_api.sh` for the identical direct host-terminal check:
180-second bound, ephemeral read-only CLI, no RAX/robot service, separate logs, elapsed
time and exit code. Shell syntax checked. The full script has not succeeded here.
The original RAX logs include successful requests lasting 107, 173 and 236 seconds;
those durations include the entire streamed response, not just network latency. They
cannot by themselves distinguish provider delay, model generation time or prompt overhead.

## Current selection: local Codex ChatGPT login

At the user's request, CLI default is restored to `--profile chatgpt`. This uses the
installed local Codex binary and existing host ChatGPT login through the budget relay;
`--profile api` remains opt-in. No global login/config or historical session was modified.
Local auth_mode is chatgpt and required credential fields are present; no values were
printed. No fresh remote-login request was made in this environment. Prior host trials
already validated this transport, but that does not prove current token validity.
Use a fresh `codex_chatgpt_trial` session; do not resume an API session with a different provider.

## Supervised robot integration (current implementation)

Added `supervised.py`, `control_service.py`, `real_backend.py` and
`scripts/run_control_service.sh`; added explicit `real_supervised` support to the
Codex launcher and official workspace adaptation. Default remains replay; overnight
runner still rejects real. Operational commands and exact boundaries are in
[REAL_SUPERVISED.md](REAL_SUPERVISED.md).

An operator-started process reuses production NativeRuntime plan/primitive execution,
DualArmCoordinator, fresh feedback, safety certification and stop_all. An authenticated
Unix RPC boundary exposes only typed primitives. Agent SDK imports, production files,
operator capability and controller networks remain outside its sandbox. No global robot
configuration or production code was changed. No extra Z margin or cloth strategy added.
Stationary capture reuses existing capture + Camera B config helper; it never calls the
observation method that moves to perception. The existing InputGuard supplies continuous
LB authorization, stale-input rejection and release/repress generation invalidation.
Independent host heartbeat and watchdog stop/latch on loss. No automatic re-arm or STOP
recovery. Every motion is planned/certified by the existing implementation, observations
are age/joint/calibration checked before dispatch and invalidated after execution.

Validation: `validation/pytest_supervised.txt` (73 passed; fake actuator/fault injection,
not physical trials). `validation/supervised_config_audit.json` loads existing config and
models only, confirming 6/7 DoF, camera ownership, measured safety, enabled path validation
and collision envelopes. evdev is installed; no gamepad/hardware connection was attempted.
Control-service help and shell syntax checked. Upstream checkout remains unmodified.
No actual real socket/service, physical camera capture, motion, stop latency, hardware
watchdog behavior or real cloth task outcome was verified. Those remain onsite checks.

## Codex substitution with original RAX workflow retained

Added baseline.py, xarm_backend.py, xarm_driver.py and operator.py. Native robot-only
wrappers were extracted from the old supervised service; that service and capability
protocol are not instantiated on this path. Codex launcher now defaults to `upstream`
workflow, using generated START_PROMPT and upstream RESUME_PROMPT, original universal
skills and registry.make_backend. Operator authorization and replies use upstream
services.arming and services.controller.operator. Previous short integration prompt is
available only as explicitly labeled offline_smoke. No Claude install is required for
the user-authorized Codex variant. No SDK code, motion planner or garment policy rewritten.

Validation: 82 passed (pytest_baseline_codex.txt), including byte-identical workflow/skills,
registry construction, unchanged tri-state checks and upstream arm-file fault injection.
Original Frontend mock invocation passed (baseline_registry_mock.json); no model/hardware.
Continuous recording and dashboard telemetry remain missing, and shell network/library
availability differs from upstream. New full model/real run not yet tested. These are
reported differences, not a claim of full original Claude/Rainbow equivalence.

## Skill library compatibility regression (2026-10-30)

`skill_library_compat.py` supplies a commit ref in the upstream merge function's
sanitized scratch repository before fetch. The pinned implementation copied objects
but advertised no refs, causing local Git to reject the SHA fetch. This explicit
runtime compatibility patch preserves upstream validation and merge; vendor remains
unmodified. It adds no skill or exploration policy.

Validation: `validation/pytest_skill_library_fix.txt`: 83 passed, including real Git
merge, a new session reading the draft, and rejection of verified promotion without
evidence. These are isolated fixtures, not physical experiments.

An isolated copy of the actual `rax_codex_replay_001` library/branch now passes fetch
but is correctly rejected by the official validator: its evidence says `outcome:
blocked`, whereas upstream accepts success/failure. See
`validation/skill_library_real_draft_regression.json`. Historical artifacts were not
modified or promoted; the agent must correct its evidence format without asserting
physical success. No hardware was connected during this work.

## Read-only real setup diagnostic (2026-10-30)

Added `scripts/check_real_setup.sh` / `rax_cloth/preflight.py`. Default loads the
existing NativeRuntime models/configuration without connecting hardware. Explicit
`--live-read` uses NativePort for state, stationary camA/camB captures, existing
observation/arrival validation, then disconnects. No arm, Home, plan, movement,
gripper, controller recovery or STOP command is invoked by this diagnostic.
It is a host diagnostic, not an agent tool or an exploration policy.

Verified command (configuration only):

```bash
bash experiments/rax_cloth/scripts/check_real_setup.sh \
  --output experiments/rax_cloth/validation/preflight_config_003
```

That output directory already exists; use a new path when repeating. Result:
configuration_passed; both axes, camera ownership, calibration/model loading and
required executable paths checked. Earlier development attempts 001/002 failed
and are retained as such. Tests: `validation/pytest_preflight.txt`, 86 passed;
new tests ensure default never connects and live-read only invokes read/close calls.

Attempted live reads: `validation/preflight_live_001/preflight.json` failed with
`connect socket failed`. An independent IPv4 socket creation check returned
`PermissionError: [Errno 1] Operation not permitted` in this assistant environment.
No actual state/image or physical execution was verified. Run on the host terminal,
with other robot owners stopped and both arms stationary in their existing ready
state (the script will not enable motors or recover faults):

```bash
bash experiments/rax_cloth/scripts/check_real_setup.sh --live-read \
  --output experiments/rax_cloth/runs/live_read_check_001
```

Use a fresh output directory. A passing live-read does not establish emergency-stop
latency, operator presence enforcement, motion safety for future actions, or cloth
success. Those onsite validations remain outstanding.

## Durable execution accounting (2026-10-30)

The real xArm backend now writes a host-owned `actions/<id>/dispatch.json` before
calling the existing executor and saves returned results or execution exceptions in
`result.json`. Authorization is checked again after writing the dispatch record.
Fresh backend instances reconstruct completed primitive count, dispatch attempts,
recorded execution duration and unresolved attempts from those files. Restart still
clears authorization and never replays an interrupted request.

A dispatch record without a result is counted conservatively as unresolved; it does
not prove a physical action occurred. An execution exception is likewise UNKNOWN,
not a task failure/success judgment. Timings exclude unobserved crash intervals and
must not be reported as complete wall-clock physical duration. Observations are
invalidated on both returned executions and exceptions. Controller-level returned
failure detail remains preserved. This recording changes no planning or safety rule.

Validation: `validation/pytest_execution_resume.txt`, 89 passed. New fake-driver
fault-injection tests cover successful execution then restart, exception then restart,
and a crash marker without a result. No physical action or stop latency is validated
by these tests. Physical connectivity and onsite authorization/stop checks are still
blocked on host-terminal validation; the assistant cannot create IPv4 sockets.

## Current completion audit / external blockers (2026-10-30)

- Current official-Codex launcher preflight was actually attempted in mock mode:
  `validation/current_isolation_preflight_001/codex-adaptation/isolation.json`.
  Isolation failed before model/hardware startup; no sandbox bypass was attempted.
- `run_overnight.sh --mode mock --session experiments/rax_cloth/validation/overnight_audit_001 --episodes 2 --interval-seconds 0`
  completed two interface diagnostics, with explicit no-exploration summary. This
  entry is not an automatic official-Codex episode runner; its old agent driver is
  disabled. A full model overnight runner remains outstanding and must not be
  described as verified by these diagnostic episodes.
- The live-read diagnostic remains blocked by socket permissions in this executor.
  Host-terminal live-read results and onsite authorization/stop/motion checks are
  required before claiming real exploration works.
- Original finite arm grants cannot detect an operator walking away. The current
  baseline documents this limitation; continuous-presence enforcement remains
  unresolved. No second service or custom garment policy was reintroduced.
- Continuous video/telemetry and final independent physical evaluation remain
  unverified/incomplete. There is no validated action-conditioned cloth simulator.

The overall real-ready objective is NOT complete. Offline test results do not prove
real camera availability, stopping behavior, physical interaction or task success.

## Official-Codex offline overnight batches (2026-10-30)

Added `scripts/run_rax_overnight.sh` and `rax_cloth/overnight.py`. This is distinct
from the legacy `run_overnight.sh` interface fixture. Each episode calls the existing
`codex_agent.launch(workflow="upstream")` with the unchanged upstream prompts and
creates a separate official session, sharing the offline skill library. It does not
provide a garment strategy, simulator, automatic retry or real-mode transition.

```bash
bash experiments/rax_cloth/scripts/run_rax_overnight.sh \
  --profile chatgpt --mode replay \
  --session experiments/rax_cloth/runs/rax_overnight_001 \
  --max-hours 8 --episodes 24 --max-model-calls 256 \
  --episode-calls 32 --episode-seconds 1200
```

The command above is prepared for the host terminal, NOT a verified completed model
batch in this restricted executor. Defaults are eight hours, 24 episodes, 256 total
model-call reservations, 32 calls/1200 seconds per episode. Episode limits are capped
by remaining batch budgets. Mock/replay only; sim and real are rejected. Dollar caps
are rejected because this transport cannot enforce them; costs remain null.

Resume with the identical command plus `--resume`. A recorded interrupted thread is
resumed with its immutable caps; a threadless attempted session stops for inspection.
Errors stop the batch immediately; no automatic retries. A batch lock prevents two
owners. Missing usage evidence is conservatively charged to the allocated budget;
`model_call_budget_charged` differs from measured `model_calls_reserved` (null when
unknown). `batch.json`, `results.json`, `summary.md` and episode directories preserve
state and results. SIGINT/SIGTERM use existing launcher shutdown behavior. Time is
cumulative launcher accounting, not an estimate of task completion; cleanup may add
a small amount beyond the configured execution cap.

Validation: `validation/pytest_official_batch.txt`, 96 passed. New tests cover shared
call budgets, explicit resume/caps/thread preservation, mode and dollar rejection,
immutable configuration, and conservative accounting after a crash. Actual command
entry was exercised in `validation/official_batch_entry_002`: stopped before model
startup at bwrap isolation with Operation not permitted. No model batch or physical
learning success is claimed. This supersedes the earlier report's missing batch-runner
implementation, but end-to-end host validation remains outstanding.
