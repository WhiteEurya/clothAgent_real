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

# 当前 baseline：RAX-Codex / dual xArm

用户允许把 Claude Code 换成 Codex；实验工作流仍取自固定官方
`real/rainbow@36c926ca3cf15921c6fc746a4bf309c26b0aa3c4`。
这是明确标注模型/CLI 变化的 RAX 移植，不称为未经修改的 Claude/Rainbow 实验。

## 保留与改动

保留官方 Workspace recipe、四个通用 Skills、START_PROMPT、RESUME_PROMPT、Memory、
Skill Library、Configuration、Session、MCP Frontend、Context、MCP bridge、arming 与 operator requests。
测试逐字节比较 START_PROMPT 和四个通用 Skills，确认没有改成短编程任务或人工衣物策略。

只有两类适配：

1. **Codex CLI**：现有登录/Responses relay、Codex 文件和网络隔离配置、CLI 事件及 thread resume、预算和进度输出。
   模型默认 gpt-6-astra；这是实验变量。原始 Claude subscription 等待/恢复机制未宣称等价复现。
   Gemini 未启用，subagents=0，均属明确的实验配置。Agent 目前无任意互联网/在线安装能力；这是与原分支的能力差异。
2. **xArm Backend**：通过官方 registry 创建 `XArmBackend`；调用现有双臂规划/碰撞验证/原生执行和静止相机采集。
   `xarm_*` 工具前缀、坐标和身份转换属于机器人适配。SAFE/UNSAFE/UNKNOWN、Z=0 与现有约束保留。

**不再使用**第二套 control_service、capability RPC 或手柄 LB；它们仅保留为旧实验源码，baseline 无依赖。
仅有官方 Frontend 这个机器人服务入口；Agent 与其 Python 程序使用同一个 backend/dispatcher。
模型 relay 仍存在，这是替换 Anthropic 为 Responses 时的 CLI 适配，凭证不会交给 Agent。

## 启动

先验证新路径的 replay（不需要操作员服务，也不连接硬件）：

```bash
bash experiments/rax_cloth/scripts/run_rax_codex.sh \
  --profile chatgpt --mode replay \
  --session experiments/rax_cloth/runs/rax_codex_replay_001 \
  --max-model-calls 32 --max-seconds 1200 --progress-interval 5
```

该路径默认 `--workflow upstream`，使用官方 START_PROMPT；不再默认要求完成一个短 offline integration episode。
原有短测试显式选择 `--workflow offline_smoke`，不能作为实机或完整探索 baseline。

### 实机入口（已接线，尚未现场验证）

不运行 `run_control_service.sh`，不需要 `/tmp/rax-cloth-control/capability`。
现场操作员在场、硬件急停可用、其他控制进程停止后，主终端启动：

```bash
bash experiments/rax_cloth/scripts/run_rax_codex.sh \
  --profile chatgpt --mode real_supervised \
  --robot-config config/dual_arm.local.json \
  --session experiments/rax_cloth/runs/rax_codex_real_001 \
  --max-model-calls 32 --max-seconds 1200 --progress-interval 5
```

后端初始未授权；只读请求可连接并采集，不自动 Home 或移动拍摄位。
待 session 建立后，现场另一终端使用薄 CLI 写**官方** arming 文件：

```bash
bash experiments/rax_cloth/scripts/rax_operator.sh status --session experiments/rax_cloth/runs/rax_codex_real_001
bash experiments/rax_cloth/scripts/rax_operator.sh arm --session experiments/rax_cloth/runs/rax_codex_real_001 --minutes 1
bash experiments/rax_cloth/scripts/rax_operator.sh disarm --session experiments/rax_cloth/runs/rax_codex_real_001
```

arm 要求现场输入 ARM；默认有限授权 1 分钟，最多 30 分钟，不支持无限期授权。后端在规划/执行期间每 20ms
重新读官方 arm 文件，disarm、过期或授权代次改变会停止。该机制**不检测人是否物理离开**，现场操作员必须
离开前 disarm；它不是已经验证过的自动在场检测。新 backend 启动会撤销旧授权，不自动恢复 STOP。
原有机器人安全检查、状态过期和停止机制不放宽。断电/进程 SIGKILL/停止时延仍须依赖并验证实际控制器安全层。

官方 operator 请求/回复也直接复用：

```bash
bash experiments/rax_cloth/scripts/rax_operator.sh requests --session experiments/rax_cloth/runs/rax_codex_real_001
bash experiments/rax_cloth/scripts/rax_operator.sh reply --session experiments/rax_cloth/runs/rax_codex_real_001 --request-id REQUEST_ID --text done
```

## 验证与未完成项

- `validation/pytest_baseline_codex.txt`：82 项通过，包括官方工作流字节一致性、registry、arm/disarm、UNKNOWN/UNSAFE、停止和离线恢复。
- `validation/baseline_registry_mock.json`：实际调用原始 Frontend + registry 创建后端，mock 观测通过，运动 NOT_EXECUTED，模型/物理调用为 0。
- 官方 checkout 未修改。修改与新增代码全部在独立实验目录；生产代码及全局配置未修改。
- 新路径 `rax_codex_replay_001` 已完成模型端到端 replay：236.6 秒、11 次调用，物理动作 0、task_success=null。实机仍未验证。
- 连续高频录像/stream 和 xArm dashboard telemetry 尚未接入；当前仅保存请求观测及现有执行日志，不声称与官方分支功能完全相同。
- 未校准 Codex 和原版 Claude 的模型费用/能力等价性；美元费用未知，不能把 token 或调用数当成美元费用。

代码入口：`baseline.py`（官方注册/工作空间）、`xarm_backend.py`（Backend Contract）、
`xarm_driver.py`（现有接口接线）、`codex_agent.py`（CLI 适配）、`operator.py`（官方授权/请求薄 CLI）。

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
