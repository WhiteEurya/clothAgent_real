# Arm7：Seed → Compile Graph → Graph Replay

独立的单次策略复用实验，默认 `full_fold`：完整复用当前 arm7 的五张 demonstration（左袖、右袖、左侧衣身、右侧衣身、下摆）。仍可显式选择 `--subgoal right_sleeve` 做单袖实验。不训练模型、不挖掘历史批次、不合并或扩展 graph、不修改 folding policy、不写 approved skill library。节点条件与动作关系由运行时 GPT-6 从本次真实 seed 提出，代码没有预定义衣物状态。

本次开发只执行离线测试，未运行机器人、未调用 GPT-6，也尚无真实实验成功率。

## 启动

在仓库根目录、原 arm7 的 `cali` 环境下执行。`seed` 和 `replay` 会真实移动 arm7，沿用原有校准、Camera B、点云修正、15 mm 抓取偏移、IK、预览及执行验证；不连接 arm6。`compile` 不构造机器人 runtime。

```bash
python -m cloth_agent.experiments.graph_policy seed \
  --subgoal full_fold \
  --output runs/graph_policy_full01 \
  --max-rounds 30

python -m cloth_agent.experiments.graph_policy compile \
  --run runs/graph_policy_full01

# 用户重新粗略摆放衣服、手离开工作区后：
python -m cloth_agent.experiments.graph_policy replay \
  --graph runs/graph_policy_full01/graph_v1.json
```

也可使用绝对解释器 `/home/sja/miniconda3/envs/cali/bin/python`。默认配置是 `config/dual_arm.local.json`，预览端口 `8088`，与当前 arm7 相同；两个物理命令支持 `--config`、`--viser-port`。不要与其他机器人进程同时运行。

Seed 直接调用现有 `codex_loop.loop`，同一个 Codex 会话继续多轮，默认使用全部五张参考图，以整件衣服最终折叠状态为完成条件。只完成袖子或任一中间阶段不能结束任务。没有 graph 输入，不向自由规划器灌入 graph 模板；不修改它的抓取或规划策略。模型仍为原 pipeline 的 GPT-6 配置。

- Seed 的 full_fold 默认最多 30 轮，right_sleeve 默认最多 8 轮，提前 `motion.complete()` 即停止。达到上限的 `ROUND_LIMIT` **不是成功**；仍可编译已记录的实际尝试。
- Compile/replay 默认 `--model gpt-6-astra --timeout-s 900 --max-calls 256`。每次响应最多一次格式修复，预算包括修复；不无限重试。
- Replay 默认最多执行 graph 的 edge 数量，可用 `--max-steps N` 指定有限上限；`--match-threshold` 默认 0.8。
- Compile 同一 graph 不重新生成或覆盖。要开展下一轮实验，使用新 seed 输出目录。
- Replay 默认创建 `replay_<UTC timestamp>/`；可用 `--output` 指定**尚不存在**的目录。物理阶段不从旧目录恢复、不自动重复 dispatch。
- `--no-cache` 适用于 compile/replay 的模型请求；不会覆盖已有 graph。
- 不提供自动 fallback。GRAPH_MISS 或 AMBIGUOUS 会停止，不能把另一次自由规划的结果混入本轮统计。

完整 graph 的含义是覆盖本次实际观察到的整条折叠轨迹，不是预先为每个 demonstration 步骤硬编码一个节点。每个阶段可以有多个动作节点；到达轮数上限或中断时记录 `PARTIAL_TRAJECTORY`，不得伪造缺失阶段。`SEED_PLANNER_REPORTED_COMPLETE` 只表示模型宣布整个目标完成，不代表独立物理验证。

旧 `runs/graph_policy_exp01` 只包含右袖动作，保留原数据与原目标；不能凭空扩展成完整折叠图。请使用新目录采集完整 Seed。五张目标参考图及哈希会随 trajectory/graph 保存，并作为多模态图片传给 Builder 和 Replay，防止后续审核把单袖完成误当整件完成。`graph_v1.md` 同时输出 Mermaid 连接图与 edge/seed-step 对照。

## Seed 记录

```text
runs/graph_policy_full01/
  experiment.json
  seed/                       原 arm7 planner、motion、native_plan、夹爪反馈、录像、timing
    round_001/...
    round_002/...
    terminal_after/            仅轮数上限恰好结束于成功执行时补拍
  seed_trajectory.json
  seed_trajectory.json.sha256.json
  seed_trajectory.md
```

每步分开保存 GPT 原始输出、宿主最终 commanded action（优先 `effective_motion.py`）、controller outcome、执行期间 controller event 切片、夹爪反馈、before/after 和各证据哈希。下一轮正常拍照作为上一轮 after，不在规划步骤中插入额外反思或改策略。

异常后不会为了获得 after 再移动机器人；缺失 after 明确标记。完成/blocked 是无动作决策，也保留在 trajectory，但不是物理 edge。图的每条 edge 必须精确引用一个真实发生的 seed attempt，不能用未执行的 IK 失败方案冒充执行轨迹。

已有 execution 视频作为审计证据保留，但此最小版不自动解析视频。Builder 实际收到所有已记录 before/after RGB 与执行反馈；不能声称已从视频证明夹持。`EXECUTED`、爪闭合或 planner 自报完成均不是物理 cloth outcome 的 ground truth。

## Compile

`compile` 使用已有 `CodexMultimodal` 真正传入图片，独立会话，不继承 seed planner 的对话。保存请求、图片哈希、响应、token usage（可用时）、耗时、重试与缓存信息。

```text
  compile/requests.jsonl
  compile/cache/*.json
  graph_v1.json
  graph_v1.json.sha256.json
  graph_v1.md
```

Graph 除要求的 node/edge 字段外，还包含 `seed_step_id`、`seed_outcome`、`outcome_evidence`、seed hash、goal、宿主下降偏移及 compile metrics。

- 所有 physical attempt 必须被覆盖，edge 的 exact command 必须与冻结 seed 逐字段一致。
- node 的代表图、edge 的起止节点必须与实际 before/after 对应；未知不能变成观察事实。连续物理步骤共享同一中间观测时，前 edge 的 target_node 必须等于后 edge 的 source_node。
- `ONE_SHOT_HYPOTHESIS` 永远不等于 validated rule。
- controller 失败，或 Builder 根据证据判定 seed 失败，都必须标 `OBSERVED_FAILURE`，禁止作为 replay 可选 edge。
- seed outcome 为 UNKNOWN 的非失败 edge 仍是待检验的一次性假设，是否可重放还须通过新的匹配与模板审核。

## 抓取朝向

Seed 与 Replay 都复用 arm7 与 unfold 相同的平行夹爪 180° 等价处理：按当前世界 TCP 朝向选择较近的一支，对 grasp 及全部持布目标施加同一个工具局部 Z 轴半圈变换。XYZ 与相对世界旋转保持，原有下降偏移只加一次；不改变 action family。`grasp_orientation_policy.json` 记录是否应用，`effective_motion.py` 保存实际用于规划的姿态和高度。大角度 wrist 转动原有逆时针偏好继续生效，完整路径仍走原 IK/验证。这只能消除不必要的等价半圈，不能保证所有 IK 路径都无绕行。

## Replay 的约束

1. **NODE MATCHING**：第一请求只有当前 RGB、goal 和冻结 graph；响应 schema 不允许动作字段。低置信度、违反条件、未解决的条件或多个候选均不执行。
2. **Existing edge instantiation**：只选择匹配节点的非失败 outgoing edge。模型输出每个现有操作的像素 anchor、相对实测表面的高度、world RPY 与简短关系证据。宿主以当前 RGB-D `Observation.sample` 取数，不让模型编造 depth。
3. **Host validation**：保持 seed `grasp → move_tcp / move_transfer ...` 指令类型、数量和顺序，只允许 right 臂。禁止新 edge、新 action family、任意 Python、增减 grasp cycle。第一 grasp 使用原始表面 Z，宿主原有额外下降只施加一次。其他 seed 指令格式不能表达时明确停止，不自由替代。
4. **Semantic conformance**：另一个独立 GPT 请求审核当前定位、位姿与 edge 的 grasp/target/Z/yaw/其他约束是否一致。UNKNOWN 或违反即停止；这仍是模型判断，不能形式化证明语义一致。
5. **Dispatch**：先落盘 `dispatch_intent.json`，再交原 Arm7Runtime 做 IK、原有路径验证和执行。无自动 free-plan 或 IK 代码重写。
6. **Outcome audit**：用 before/after 和反馈判断预期 target node / 整个任务目标是否到达。只在真实执行、预期 target 高置信确认、目标 COMPLETE 时计为 `graph_only_success`。初始就处于 terminal node 不算 edge reuse success。
7. 多步仍依次重做 node matching；与上一步 target 判定矛盾时停止并记 `NODE_MISMATCH`。不更新图，不自动扩图。

相对表面高度本版仅支持 0–300 mm；模板需要其他表达时停止。当前 RGB 或深度不足、seed 只包含不适配的低层动作、以及模板对应不确定都允许失败，实验优先可审计性。

## Replay 产物与计量

```text
  replay_<timestamp>/
    graph_used.json            固定版本快照与 hash
    initial_observation/      arm7 perception RGB-D
    step_000/
      node_match.json
      instantiation.json
      surface_measurements.json
      conformance.json
      dispatch_intent.json    GRAPH_REUSE，existing edge ID、command、before hash
      motion.py
      effective_motion.py     原 runtime 输出（抓取偏移后）
      execution_events.json
      outcome.json
      after/
      assessment.json
    model/requests.jsonl
    model/cache/
    metrics.json
    final_metrics.json
    final_metrics.json.sha256.json
    report.md
```

未走到的阶段不伪造文件。`report.md` 对比 seed/replay 调用量、视觉工具调用、可观察规划耗时、action family 和最终结果，同时给出具体 transition 与实例化证据链接。

`model_calls` 是实际 Codex 进程调用数，含有限格式修复，cache hit 不算新调用；`completed_model_turns` 与 `tool_calls` 来自事件日志，内部底层 API 调用数不可得。直接作为多模态附件发送的 RGB 不算一次 visual tool call。`reasoning_time_s` 是模型请求墙钟耗时，不是私有思维过程长度，也不含机器人执行。编译成本单独记录，不混入 replay；此单次对比不能证明统计意义上的加速。

失败归因包括 `NODE_MISMATCH`、`EDGE_GENERALIZATION_FAILURE`、`ACTION_EXECUTION_FAILURE`、`GRAPH_MISS`、`UNKNOWN`。未证实的原因保留 UNKNOWN；不会因匹配失败强行选节点，也不会因一次失败断言模板普遍无效。

## 离线验证

```bash
python -m pytest -q tests/test_graph_policy.py
```

测试使用假 runtime、合成 RGB-D 和假模型响应，不访问机器人或外部模型。

## 建图请求的体积控制

Builder 接收全部步骤、精确 commanded actions 和全部 before/after/目标 RGB；高频夹爪轮询、录像时间戳和底层轨迹数列使用 `dense-telemetry-v1` 摘要。摘要保留数量、首尾、数值范围、机械反馈/错误统计和原数列哈希，明确说明内部细节未传给模型。controller 事件顺序与所有 native-plan 操作保留。原始 seed 和证据文件不改动，不能据摘要断言全过程持布成功。

`compile/input_projection.json` 记录原始/投影字符数和来源 hash。Graph 请求文本超过 800,000 字符会在调用前明确拒绝，不静默截断、跳过步骤或消耗一次注定超限的模型请求。超限失败后可直接重跑同一 `compile --run ...`；不会重新执行机器人。

### 未观测终点

中断尝试没有 after RGB 时，允许 GPT 创建 `representative_observations: []`、`confidence: 0`、有明确不确定性且无 outgoing edge 的未知 sink。它只保存证据缺口，不是可匹配的观察状态。只有实际缺少 after 的尝试才能指向这种 sink；匹配和 edge 资格检查都会排除它及指向它的边。

`UNAVAILABLE` 等缺失结果不能自动等同已确认的 physical failure，也不能宣称 SUCCESS；明确的 `EXECUTION_FAILED` 仍必须标为 `OBSERVED_FAILURE`。未知和负面证据不会为了满足图结构而改写成成功。

## 显式合并状态并生成可重放的有向图

```bash
python -m cloth_agent.experiments.graph_merge \
  --graph runs/graph_policy_full02/graph_v1.json \
  --output runs/graph_policy_full02/graph_v2.json
```

此独立命令只调用 GPT 看已有 RGB，不连接机器人。模型输出视觉状态 partition 与合并/不合并证据；宿主保留每条原动作及其方向、精确命令、真实 outcome 和来源，仅按映射重定向端点。允许多个 outgoing edge、平行边、自环；不为了制造分支而虚构动作。未知终点保持独立。

产物：`graph_v2.json`、`graph_v2_merge_review.json`、`graph_v2.md`、带状态图片的 `graph_v2.svg` / `graph_v2.png`，以及每个状态的 representative montage。绿色实线是可选的一次性假设动作；红色虚线是失败或未知终点边，仅作审计。Reachability 报告检查可选边能否到达最后一个已观察状态，不能把它等同任务成功。

```bash
# 重摆衣服后，显式指定合并图：
python -m cloth_agent.experiments.graph_policy replay \
  --graph runs/graph_policy_full02/graph_v2.json
```

合并图 Replay 的 node matching 附带各状态代表图，在当前匹配节点的已有可选出边中实例化动作。执行后验证目标节点并重新观察，再选择下一条边。同一条边每次 Replay 最多尝试一次，避免合并产生的自环重复；不重写图、不转入 free planner。未合并的旧 graph_v1 保持原有行为。

当前用户审阅版本为 `runs/graph_policy_full02/graph_v3.json`：`S1 → S2` 标为 `PROGRESS`，删除未知终点 `S11` 和边 `e013`，指定 `S10` 为终点。原评估和被排除的尝试保留在 `user_review`，原 v2 与 seed 不变；该标签调整不代表新增物理验证。

```bash
python -m cloth_agent.experiments.graph_policy replay \
  --graph runs/graph_policy_full02/graph_v3.json
```

审阅图通过 `terminal_node_ids` 声明终点；可靠匹配终点后停止，不再要求进一步调整。初始观察已经处于终点时不执行动作，也不统计为成功复用；执行已有边后可靠到达终点才计为 graph-only success。含排除步骤或指定终点的图必须保留 `user_review` 与冻结源图的哈希来源；controller 执行失败仍不能被 `PROGRESS` 覆盖。
