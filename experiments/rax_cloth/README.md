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

## 当前任务范围更新：仅 arm6 / camA（2026-10-30）

用户将本任务改为单臂。当前工具只允许 arm6 动作和 camA 观测；不连接 arm7 控制器、不读取 arm7 实时状态、不初始化 camB。运行不要求 arm7 在线或准备就绪。

现有双臂几何规划接口仍保留 arm7 的静态障碍模型，使用项目已有 `data/robot/xarm7_perception_pose.json`，明确标为 assumed static obstacle，而非实时反馈。该几何假设要求闲置臂保持记录的停放姿态；实际改变其位置会使该假设失效。没有使用 CollectionRuntime 的关闭路径验证/碰撞包络设置；SAFE/UNSAFE/UNKNOWN、Z=0 和现有安全验证不变。

`robot_scope=arm6` 写入新 session 配置；请新建 session，不沿用之前双臂工作空间或其旧提示。已有实验记录未改写。只读检查入口现在仅连接 arm6、采集 camA：

```bash
bash experiments/rax_cloth/scripts/check_real_setup.sh --live-read \
  --output experiments/rax_cloth/runs/arm6_live_read_001
```

验证：`validation/pytest_arm6_scope.txt`，98 passed；包含生产 connect 函数仅创建 arm6 连接、arm7 静态对象拒绝控制命令、工具拒绝 arm7/camB。`validation/preflight_arm6_001/preflight.json` 配置检查通过。尚未现场验证或执行物理动作。下文双臂章节为此前阶段记录，以此更新为准。

# RAX-Codex / dual xArm baseline

当前采用用户允许的 Codex CLI 替换，保留官方 RAX 工作流；不再使用自定义手柄/RPC 操作员服务。
当前唯一启动说明见 [BASELINE_CONFORMANCE.md](BASELINE_CONFORMANCE.md)。
新入口为 `scripts/run_rax_codex.sh`，实机授权为 `scripts/rax_operator.sh`，均不需要 capability 文件。
82 项离线测试和官方 Frontend mock 调用通过；新路径的模型整轮与实机尚未验证。

下文保留旧阶段记录，若有冲突以以上链接为准。

---

# RAX-Cloth：以官方部署为 baseline

**当前状态更正：原版 baseline 尚未完成实机部署。** 已有成功运行属于 Codex 适配实验，
不能标记成官方原版 baseline。后文记录保留用于追溯，当前边界见 [BASELINE_CONFORMANCE.md](BASELINE_CONFORMANCE.md)。

官方入口（只转发到固定 checkout，不替换 launcher）：

```bash
bash experiments/rax_cloth/scripts/run_official_rax.sh --help
```

该帮助命令已测试。原版实际运行需要 Claude Code/相应认证及 xArm 后端注册，目前还未具备。

---

# RAX-Cloth（离线 Agent 已跑通，受监督实机接入待现场验证）

当前实机接入说明：[REAL_SUPERVISED.md](REAL_SUPERVISED.md)。本轮仅完成代码与离线验证，未执行机器人动作。

独立目录，不改动生产 Cloth Agent、机器人配置或衣物策略。详细状态见 [SETUP_REPORT.md](SETUP_REPORT.md)。
已取得完整官方 real/rainbow checkout，并接入其 Workspace、Session、Frontend 与 Skill Library checkout。
新增官方组件测试命令：

```bash
bash experiments/rax_cloth/scripts/run_official_smoke.sh --session experiments/rax_cloth/runs/official_mock
```

用户本机终端已通过原生 Unix socket MCP smoke test；我的执行沙箱仍只能使用管道传输适配。
两种结果分别保存在 validation 中。没有运行探索 Agent，没有真实衣物闭环结果。

## 已测试的安装

在项目根目录执行（已有 cali 环境提供离线依赖，仅作为 **host 服务环境**，不可交给探索 Agent）：

```bash
/home/sja/miniconda3/envs/cali/bin/python -m venv --system-site-packages experiments/rax_cloth/.venv-host
experiments/rax_cloth/.venv-host/bin/python -m pip install --no-deps --no-build-isolation -e experiments/rax_cloth
```

脚本默认使用现有 cali Python，允许 `RAX_CLOTH_PYTHON` 指向上述独立 host 环境。
官方 workspace 渲染额外依赖 Jinja2（本地 host venv 已补齐）；新环境可安装 `jinja2` 和 Python 3.10 所需的 `tomli`。
旧诊断入口的 `--driver agent` 仍会失败关闭；新增独立 Codex 入口见下方，必须通过 OS 隔离预检。

## 运行

```bash
bash experiments/rax_cloth/scripts/run_overnight.sh --mode mock --session experiments/rax_cloth/runs/my_mock --episodes 2 --interval-seconds 0
bash experiments/rax_cloth/scripts/run_overnight.sh --mode replay --session experiments/rax_cloth/runs/my_replay --replay-manifest experiments/rax_cloth/configs/replay.example.json --episodes 2 --interval-seconds 0
```

旧诊断 session 记录的上游 commit 为 null，升级后不要修改历史记录来恢复它们；请新建诊断 session。
对当前版本的诊断 session，在 **相同命令** 后追加 `--resume`。官方组件 smoke 自身允许重新打开同一兼容 session。Ctrl-C / SIGTERM 会停止并输出 summary。
已有运行不会被覆盖；恢复时模式、随机种子、输入 manifest 内容、预算必须完全一致。中断 episode 保留，下一次创建新目录。

`--max-hours` 默认 8；`--episodes` 默认 100；`--max-llm-calls` 和 `--budget-usd` 默认 0。
离线诊断不调用模型，设更大模型预算也不会启动 LLM。`--interval-seconds` 控制诊断间隔。
若需长时间检查记录与恢复，可增加 episodes；这不是离线学习，不应将重复静态帧计为探索样本。
`--driver mcp` 提供离线 stdio MCP；它不是官方完整 frontend 的替代品。

## Codex 离线 Agent 入口（新增，尚未完成模型端到端验证）

```bash
bash experiments/rax_cloth/scripts/run_agent_offline.sh \
  --session experiments/rax_cloth/runs/codex_replay_trial_v2 \
  --mode replay --max-model-calls 32 --max-seconds 1200
```

这是**待在本机终端验证**的命令，不是已通过的模型实验。默认使用本机 Codex 的 ChatGPT 登录及 `gpt-6-astra`；只开放 mock/replay。
可先追加 `--preflight-only`，通过后去掉该参数启动。若已有模型执行记录，使用完全相同
参数追加 `--resume`；时间和调用预算不重置。没有有效 thread id 的失败需新建 session。
隔离预检失败不调用模型；不允许降级到宿主直接执行生成代码。

保留官方 Workspace、通用技能、Memory、Frontend、Context 和 MCP stdio bridge；
Claude Code launcher/Anthropic relay 被明确标记的 Codex CLI/Responses **适配层**替代。
这轮入口只做一个短离线代码迭代 episode，不等于完整衣物探索或夜间学习。
系统 Python 3.8、NumPy/OpenCV 用于 Agent，`.venv-host` 仅供宿主服务使用。
Agent 看不到宿主登录文件、生产仓库、机器人 SDK 或机器人网络。

`codex-adaptation/` 保存配置与隔离结果、episode 的 prompt/events/stderr、模型请求/响应、
调用账本、代码版本、thread id；原始图像及 MCP 审计由官方 session 保存。
模型 relay 只允许固定供应商 endpoint，不开放任意 URL 或托管网络工具。
当前传输没有配置可靠的供应商美元计费规则，所以费用为 `null`，指定 `--budget-usd`
会明确拒绝；调用次数和时间限制可配置。该传输还未实测远端接受，也不刷新过期登录。

本轮预检被当前助手沙箱拒绝：`NETLINK_ROUTE socket: Operation not permitted`。
本机原生 Unix socket smoke 成功不能证明 network namespace 或模型 relay 已通过。

## 文件

- `rax_cloth/backend.py`：官方接口形状的 mock/replay backend，RGB 与受限基础工具。
- `rax_cloth/existing.py`：仅 host 使用的现有坐标、安全与规划器委托；实机 transport 硬禁用。
- `rax_cloth/upstream_reference/`：网页取得的官方 registry 与 frontend 参考快照（Apache-2.0）；历史快照；当前实际使用 vendor/rax 的完整官方组件。
- `rax_cloth/upstream.json`：来源、源码哈希及已核对 commit `36c926ca3cf15921c6fc746a4bf309c26b0aa3c4`。
- `scripts/bootstrap_upstream.sh`：完整 real/rainbow checkout 与 commit 固定入口，用户终端已执行成功；重复执行不会覆盖源码。
- `rax_cloth/session.py`、`budget.py`：独立日志、版本、恢复、模型费用预留账本。
- `configs/replay.example.json`：一份真实历史观测及文件哈希，不含现有衣物策略或示范动作。
- `tests/`、`validation/`：测试代码及实际结果。

## 输出

每个 session 包含 `agent-workspace/{TASK.md,AGENTS.md,code,memory,skills,skill-library}`、
`episodes/<id>/{observations,events.jsonl,result.json}`、`programs/<sha256>/`、
`state.json`、`events.jsonl`、`results.json`、`summary.md`。
初始衣物 skills 为空。测试 fixture 显式标注 test_fixture，不被提升成技能。
返回给 Agent 的 RGB 不包含现有 Cloth Agent overlay/prompt；云数据只在 host 上用于已有标定坐标查询。

## 安全边界

mock 不连接硬件；replay 不执行动作，不提供反事实结果。所有请求仍留下日志。
sim 未验证，拒绝启动。real_supervised 已新增独立操作员服务，必须显式选择并由现场操作员授权；见 [REAL_SUPERVISED.md](REAL_SUPERVISED.md)。
没有 arm / SDK / 任意导入 / 修改配置工具。Z=0 保持原定义，不添加夹爪高度余量。
当前 bubblewrap 命名空间被环境拒绝，不能启动探索 Agent 或不可信 Python。
不要用宿主 subprocess、AST 检查、Python import 黑名单或普通 venv 替代 OS 隔离。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/rax_cloth:. experiments/rax_cloth/.venv-host/bin/python -m pytest experiments/rax_cloth/tests -q -o cache_dir=.pytest_cache
```

## 本机首次模型运行后的修正

`codex_replay_trial` 已通过外层隔离，并成功完成 8 次模型请求（全部 HTTP 200）。
Agent 已读取历史 RGB、编写感知代码并运行一次图像分析；随后触发本地 8 次调用上限，
没有完成整个 controller 测试。原始记录保留，见 `validation/host_codex_trial_diagnosis.json`。

修正了内层 Codex `workspace-write` 的 socket 网络配置，允许 Python Context 访问 MCP；
外层 `bwrap --unshare-all` 和全部文件挂载限制保留，不能访问宿主或机器人网络。
该修正依据 [OpenAI Docs](https://learn.chatgpt.com/docs/agent-approvals-security) 的沙箱配置说明。
修正后的实际 Python-to-MCP 联通仍需主机重跑验证。预算耗尽现在输出明确原因和事件日志路径，
以退出码 2 结束，不再只抛出指向空 stderr 的异常。
使用上面的新 session、32 次/1200 秒命令重新验证；旧 session 的预算不会自动重置或增大。

## 使用本机 `codex -p api`（可选）

```bash
bash experiments/rax_cloth/scripts/run_agent_offline.sh \
  --profile api --session experiments/rax_cloth/runs/codex_api_trial \
  --mode replay --max-model-calls 32 --max-seconds 1200
```

读取 `$CODEX_HOME/api.config.toml`（未设置 CODEX_HOME 时为 `~/.codex/api.config.toml`），
按 base config + profile 加载模型/provider；本机解析结果为 `gpt-6-astra`、`high`、DMXAPI。
隔离内的本地 Codex CLI 使用 `--profile api`，profile 经安全适配后只指向预算 relay。
宿主 relay 使用原 profile 的固定 HTTPS Responses 地址和认证命令；密钥与宿主原配置不挂载给 Agent。
仅迁移模型、推理强度及支持的 provider transport/auth；不继承个人 prompts、MCP 和放宽权限设置。
provider 缺失或认证失败直接报错，不自动回退。`--profile chatgpt` 为当前默认登录模式。
换 provider 必须新建 session；API 和 ChatGPT 的历史记录不混合。

已验证：profile 配置解析、宿主认证 helper、模拟 API 转发和 51 项测试；
新 API profile 的真实远端请求及整轮 Agent 运行尚未验证。历史成功调用来自旧 ChatGPT transport。

## 实时进度 / Debug

启动阶段、模型请求阶段（连接发送/等待响应头/接收输出）、MCP 工具、命令开始/完成、
Agent 文本和退出状态现在默认输出到终端 stderr；最终 JSON 仍单独输出到 stdout。
每 10 秒输出心跳，包括 session 累计已用时间、距时间上限、已用/最大模型调用数。
`--progress-interval 5` 可改为每 5 秒；该参数不是不可变实验预算，可在恢复时调整。
倒计时表示运行时间预算，不预测任务成功时间；恢复后仍计入此前已用时间与调用次数。
完整命令/结果留在 events.jsonl，终端只显示短摘要，不输出原始图像载荷或认证内容。

日志位于 `<session>/codex-adaptation/progress.jsonl`，最新状态为 `progress.json`。
可在另一终端执行：

```bash
tail -f experiments/rax_cloth/runs/codex_api_trial/codex-adaptation/progress.jsonl
```

已启动的旧进程不会自动加载新代码；下次启动或 `--resume` 时生效。

## 当前默认：本机 Codex ChatGPT 登录

按用户要求恢复 ChatGPT 登录为默认。使用本机 Codex 可执行文件，宿主读取已有登录，
模型请求仍经预算 relay；凭证不挂载到 Agent，隔离策略保留。`chatgpt` 是本项目的
transport 选择，不要求本机存在同名 `.config.toml` profile。

```bash
bash experiments/rax_cloth/scripts/run_agent_offline.sh \
  --profile chatgpt --session experiments/rax_cloth/runs/codex_chatgpt_trial \
  --mode replay --max-model-calls 32 --max-seconds 1200 --progress-interval 5
```

API session 的来源与预算不可更换；新建上述 session，不加 `--resume`。
已有登录字段检查通过不代表远端登录有效性已重新验证。
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
