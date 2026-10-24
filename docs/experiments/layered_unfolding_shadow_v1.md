# 分层 unfolding planner：Fast + shadow gate v1

后续用户授权的错误恢复：新增 `CodexRecoveryPlanner`，仅在显式启用 recovery 的
runtime 中处理允许列表内的执行前错误。连续采集默认开启，单 episode 默认关闭。
Gate 不触发恢复，Host reject 不作为 policy-quality 标签；替代动作继续走原 Host。
每次错误/每 episode 分别限制 2/4 次，恢复成功后返回 FastPlanner；硬件故障、
dispatch 后错误及中止仍停止。Recovery round 使用 `DELIBERATIVE_ACTION`，
`plan.json` 属于本轮 Codex，不伪造 `fast_plan.json` 或 gate 标签。Fast-only 人工
review 不把后续 Codex 的执行结果配给原 Fast gate；Codex 计划/图像见 episode report
及对应 round 文件。下文记录原始 shadow-only 默认模式；完整扩展见
[连续采集说明](one_shot_goal_continuous_v1.md)。

本版是框架和 shadow capability probe。执行动作仍来自原 one-shot policy，
没有自动 routing、模型训练或慢 Agent。以前已保存的实验不会自动补 gate 标签。

```text
Observe RGB-D
  → FastPlanner(current RGB, fixed goal RGB)
  → ACTION: PreActionGate(current RGB, goal RGB, Pa→Pb overlay) [SHADOW]
  → 原 Host checks
  → 至多执行一次原 lift_pull_place
  → Observe

Fast DONE / ABSTAIN → 原终止流程，不发 gate 请求
Host reject → SAFETY_STOP，不请求慢规划，不修改 gate 标签
Gate TRUST / DELIBERATE / UNCERTAIN / request error → 同一 Fast action 继续原 Host checks
用户中止 → 原停止流程，即使正在等待 gate 也不继续执行
```

## 接口与不变量

- `cloth_agent/unfolding_planners.py`
  - `FastPlanner.plan(...)` 包装原 `plan_round`，原 prompt、payload、schema、
    `gpt-6-astra/high`、tools=[]、tool_choice=none 与无 retry/repair 保持不变。
  - `ShadowPlannerRouter` 保存一份 Fast action，给 gate 的是独立副本。执行端只能取 Fast action；
    gate 不返回坐标，也没有替换动作或调 slow planner 的路径。
  - `DeliberativePlanner.plan(DeliberativeContext)` 与
    `MemoryPlanner.plan(MemoryContext)` 目前明确抛 `PlannerDisabledError`，默认 disabled。
    输入上下文的类型预留了 RGB-D、proposal、compact history、Host uncertainty、
    frozen memory / Graph 引用；本版不读取这些资源或导入旧 Agent wrapper。
    未来返回值约定为与 FastPlanner 相同 action schema 的 replacement ACTION 或 ABSTAIN。
- `cloth_agent/pre_action_gate.py`
  - `PreActionGate.evaluate(current_rgb, goal_rgb, proposed_action)`。
  - 独立 direct `gpt-6-astra/high` 请求，strict schema 只有
    `{"decision":"TRUST|DELIBERATE|UNCERTAIN"}`，无工具、无 retry。
  - 输入严格限定为 current、goal、当前 action overlay 三张图，以及 Pa/Pb/motion 字段。
    不传 FastPlanner rationale、after、execution、history、Graph、previous rejection、
    深度或前一次 response ID。Host 几何检查不交给 gate。
  - 请求/解析错误写 `gate_status.status=ERROR`，`gate_result` 用合成 UNCERTAIN 占位；
    与模型真正返回 UNCERTAIN 分开统计。用户 KeyboardInterrupt/SIGTERM 不被吞掉。
  - Gate 是顺序等待的 shadow 实验，会增加本轮真实等待；不会改变执行选择。
    `--gate-timeout-s` 默认120秒，是既有 HTTP transport 的读取超时参数，
    不是包含持续流式事件在内的严格总 wall deadline。
- 框架首版的 Host `prepare_action`、`Observation.sample`、IK/workspace、安全阈值、
  collection runtime、初始化 primitive 和底层接口均未修改。
  Gate 等待后仍运行原 live arrival 检查。不新增 snap、slope-aware 放行或额外动作。

后续用户授权的桌面缺深度修正：one-shot 入口启用 `BARE_TABLE_TARGET_PLANE_V1`。
仅 Pb 缺深度时允许通过裸桌 RGB/实测支撑面校验后求射线与平面的交点；
Pa、衣服区域和已测得的深度断层仍走原检查。不修改原始 RGB-D，不把估计标成实测。
共享 `prepare_action` 的默认行为不变；one-shot 显式传入 target sampler。
每轮新增 `target_depth.json`，并将来源写入 points/host_validation。
完整条件见 [one-shot 实验说明](one_shot_goal_full_episode_v1.md#bare-table-pb-depth-fallback)。

后续用户修正：本入口的初始化 release 参数采用 `HOME_Y_HOME_YAW_V1`。
放下 world Y 固定为 Home Y，yaw_delta=0（输入姿态本来就是 Home 姿态）；
抓取点继续在衣服中心附近采样，X 偏移与抬升高度沿用原采样。
仅修改本入口的初始化参数，不改共享 `reset_program`、采集脚本默认值、
后续 FastPlanner Pa/Pb 或 lift_pull_place。旧实验记录不回写。

Router events：
`FAST_ACTION / FAST_DONE / FAST_ABSTAIN / HOST_REJECT /
GATE_TRUST / GATE_DELIBERATE / GATE_UNCERTAIN /
DELIBERATIVE_ACTION / MEMORY_ACTION`。
最后两项为未来接口保留，本版不会产生。HOST_REJECT 与 policy uncertainty 独立，
不会把 depth/IK 拒绝写成 NEED_DELIBERATION。

## 启动与日志

运行方式不变；此命令会真实初始化和执行，本次代码重构没有自动运行它：

```bash
cd /home/sja/clothAgent_real
bash scripts/run_one_shot_goal_episode.sh --execute
# 可选 --max-rounds 12 --gate-timeout-s 120 --goal-image /path/to/goal.png
```

每轮新文件：

```text
round_XXX/
  before/
  fast_plan.json
  fast_action_overlay.png
  fast_status.json
  gate_input_overlay.png
  gate_input_manifest.json
  gate_result.json
  gate_status.json
  gate_request/                 # 独立 attempt、payload、raw response、request metrics
  planner_states.json
  host_validation.json
  execution.json
  after/
  timing.json
```

保留旧 `plan.json / proposed_action_overlay.png / request_* / raw_response.*`，
它们仍只属于 FastPlanner，方便兼容之前的分析脚本。新 fast 文件是同一计划的副本。

DONE/ABSTAIN 的 gate_status 为 NOT_APPLICABLE、无模型调用，gate_result 的 UNCERTAIN
仅是占位，不计入 gate 标签。FastPlanner 未成功产生有效计划时，缺失 action/overlay
不会伪造；fast_status 标明无效，gate 为 NOT_RUN。

`host_validation.json` 分别记录 NOT_RUN / NOT_APPLICABLE / CHECKING / PASS /
REJECT / INTERRUPTED；只把真正进入 Host 检查后抛出的拒绝记为 HOST_REJECT。
保留既有运行异常/终止日志，不用 gate 吞掉 Host 失败。

计时：

- `fast_planner_wall_s`：原请求 wall latency，不含 gate；旧 `planning_s` 是同值。
- `gate_wall_s`：独立 gate evaluate 的实际 wall time（准备、请求及记录开销）；请求 wall
  另见 `gate_status.request_wall_s` / `gate_request/model_metrics.json`。
- `host_validation_s`：原 Host preparation、重复动作检查及执行前 live arrival 检查。
- `robot_execution_s`：原动作执行与返回观测位时间。
- `total_round_s`：整轮 wall time，包括 gate；旧 `round_wall_s` 同值。

Summary 保留旧统计，并增加 `planner_layers`：Fast 决策、有效模型 gate 决策、
gate error/interrupt 数、Host reject、physical execution、分别的 latency 分布、
ACTION-only Fast latency、逐轮 gate/human 对照及五个诊断问题。
`total_planning_s` 仍仅为 Fast 请求，gate 单列 `total_gate_wall_s`。
`total_model_planning_s_including_gate_and_initialization` 提供额外等待总口径。
成功 Fast latency 不混入无效输出；所有尝试耗时仍在逐轮 timing 中保留。

## 人工评审及计算口径

结束后打开新目录 `review.html`。每个有效 ACTION 展示 current、goal、Fast overlay、
gate 结果、Host/执行状态及 after（缺失时明确显示），人工填写：

- TRUST_FAST / NEED_DELIBERATION / UNCERTAIN；
- 可选 reasonable、progress、note；
- episode 最终 FOLD_READY / NOT_FOLD_READY / UNCERTAIN。

本地浏览器保存草稿；导出 `shadow_gate_human_review.json` 后，离线汇总：

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.planner_shadow_review \
  --episode /absolute/path/to/episode \
  --ratings /absolute/path/to/shadow_gate_human_review.json \
  --output /absolute/path/to/new_review_analysis
```

只写新的 analysis 目录，不修改原 plan、gate、execution 或 episode summary。
review_manifest 绑定 episode、goal、protocol、每轮计划/图像/gate/Host/执行记录哈希，
错 episode、重复 round 或变动证据会拒绝汇总。原 summary 里的人工标签继续是运行时的
PENDING；analysis/report.md 与 review_summary.json 提供后续标签和对应统计。

有效 gate 与人工标签形成 3×3 表：

- false-TRUST：gate TRUST 且人工 NEED_DELIBERATION。
  同时报其占人工 NEED_DELIBERATION 的比例，以及占人工明确评审的 gate TRUST 的比例。
- false-DELIBERATE：gate DELIBERATE 且人工 TRUST_FAST；分母为人工 TRUST_FAST。
- bad action 检出率：gate DELIBERATE / 人工 NEED_DELIBERATION。
- 未评、human UNCERTAIN、gate UNCERTAIN 和 gate 请求错误明确单列。
  无配对数据的 false-TRUST 记 null，不伪装成0；分母为0的比例记 null。
  Host reject 不生成任何人工标签，也不作为 bad action 的代理标签。

首次真实运行及人工评审前：Fast 是否仍约20–25秒、gate 检出率、误报率、false-TRUST
均未实测。报告只显示观测值及分母，不声称统计显著性。蒸馏价值保持 UNRESOLVED，
不会根据本轮结果训练、改 prompt 或启用自动路由。

## 离线验证

```bash
/home/sja/miniconda3/envs/cali/bin/python -m pytest -q \
  tests/test_unfolding_planners.py \
  tests/test_one_shot_goal_full_episode_v1.py \
  tests/test_one_shot_goal_planner_probe_v1.py
```

测试覆盖：输入白名单、严格 schema、错误不重试、gate 三种标签不影响执行、
原 Host 拒绝不升级、gate 无权变更动作、DONE/ABSTAIN 不调用 gate、人工中止、
人工标签与原证据绑定、错误不混入模型标签及原 prompt/config 的一致性。
全部请求与机器人在测试中使用替身，不运行真实 API 或硬件。

结构化输出接口依据：
[OpenAI Docs Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)。
