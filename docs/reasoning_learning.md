# 固定证据上的 Harness Learning

独立入口：`python -m cloth_agent.harness.reasoning_learning`。在同一个选点前 evidence package 上重复 planning，由运行时 Claude 反思并生成新的 JSON harness，再按新 harness 实际重跑。支持串行修炼和从同一基线产生多个分支。

本版只做 inner loop，不接入主循环，不执行机器人，不修改 Z、grounding、preflight、IK、evaluator、failure analysis 或 experience。真实动作后的 outer learning loop 保持原有实现。产物只是离线候选，不会自动部署。

## 实际调用链

```text
固定并校验 Z（图片、goal、当前 registry）
  → 冻结 H0
  → RuntimeClaude：按 H0 的阶段重新规划
  → RuntimeClaude：读取 H0 + 公开阶段输出 + 成本，提出 H1
  → Host 校验、保存 H1 和独立反思证据
  → RuntimeClaude：按 H1 重新规划同一个 Z
  → … 有界重复
  → action consensus
  → 稳定簇内按成本选择 harness 和一个实际产生过的 action
```

复用 `RuntimeClaude.invoke → LocalClaudeBackend.invoke / RemoteClaudeBackend.invoke`。图片通过实际 base64 多模态输入发送。规划与反思均禁用外部工具、文件读取和会话延续；不同 rollout 没有共享 Claude 会话。模型只输出公开的中间概念、证据摘要、缺失信息和决策，不要求私有思维链。

默认 H0 是明确标注的开发者基线：一次模型调用完成固定证据上的规划。它不冒充 learned policy，也不冒充原来的自由工具 planner。可以用 `--initial-harness` 指定本工具之前保存的冻结 harness；新 observation 上仍重新计算状态。

## 动作合同

原视觉 planner 的 `motion_intent` 是自然语言，无法在 Host 上可靠计算目标距离。本实验使用独立的视觉动作合同：

- `selected_reference = {camera, reference_id, reason}`：沿用当前候选选择的结构；Host 从当前 registry 取得抓点像素。
- `target.pixel_xy`：当前完整 clean 图中的目标像素。
- `target.relation`：`toward / onto / across / away_from / hold`。
- `target.anchor_pixel_xy`：该关系所指的可见锚点。
- `target.reason`：简短可见证据。

Host 校验当前 observation 身份、候选存在性、有限数值和像素边界，并用现有 affine 约定映射原图坐标。`selected_action.json` 明确标注 `robot_executable=false`。这里没有三维目标、抓取深度或物理可达性结论。target relation 的正确含义仍由 Claude 判断，Host 只做结构检查和比较。

## 输入：只允许选点前证据

有两种入口：

1. `--manifest`：使用 collector 的 manifest，多个 decision 时必须指定 `--decision-id`。只从 `pre_decision` 读取当前 clean、overlay、reference、hint、goal 和 registry；不读取历史答案、评价、request prompt、旧 planner ROI 或派生 crop。
2. `--evidence`：显式提供合法 evidence JSON，或复用本工具输出的 `evidence/evidence.json`。

显式输入字段为：

```text
schema_version: 1
observation_id: 当前 observation ID
fold_goal: 当前折叠目标字符串
candidate_registry: 现有 collector 的、与 observation 绑定的 registry
images: [{path, role, status: AVAILABLE, size: [宽, 高], rgb_sha256}]
```

相对路径相对于输入 JSON 所在目录。要求恰好一张当前 clean 和 overlay，支持 reference、hint；总计二至十六张图片。reference 可用 `reference_kind=source / target / unspecified` 标明来源/目标参考；从 manifest 读取时保留原文件名中的明确角色。显式准备的 `clean_crop` / `overlay_crop` 必须附上 `lineage={availability: PRE_DECISION, parent_rgb_sha256, to_original}`，Host 校验图片 hash、父图身份、有限且可逆的 affine 与边界。此标记是输入来源声明，不是对语义必要性或历史可用性的独立证明；请使用真实的选点前准备结果。

首次运行复制这些图片并生成 `evidence_hash`。同一实验中的所有 planning 和 reflection 调用都传入完整的相同 Z；每次调用前后校验 hash。不会在实验中动态裁剪、换图或重新运行 perception。新阶段可改变信息关注顺序，但不能改变可用证据。后续才扩展 `O → Z` 学习。

## Harness 与状态

每个 harness 是小型 JSON：`schema_version / name / applicability / stages / on_exhaustion`。阶段包含 `id / instruction / context / allow_ready`。Host 按顺序执行，每阶段最多一次 Claude 调用；`context` 只能引用同一 rollout 中的早先阶段。

模型可新增中间概念、重排/合并/拆分推理阶段、修改指令、变量绑定和提前停止条件。每次输出 `CONTINUE / READY / NEEDS_LEARNING`；最后还未 READY 就返回 NEEDS_LEARNING。没有任意 Python、无界循环或自动慢速 fallback。

缓存仅限同一 rollout 的显式阶段状态。后一轮不会拿前一轮的 action、ROI、理由或反思正文作 planning 提示。反思可读取上一轮公开输出和成本，但它的解释与证据保存于 sidecar，只有通过校验的 harness 进入重跑。可执行 prose 禁止数字、路径、历史 ID 和固定坐标；这能阻止明显答案常量，不能证明任意自然语言指令可泛化。Host 不把“删除某步可能更快”的假设记为已验证消融结论。

最多八个新版本，每版最多六个阶段；默认一次反思机会产生一个 patch，非法 patch 被拒绝且不执行，没有无限格式修复。文件只新增，已有输出目录不会删除或覆盖。

## 单独运行

先运行软件测试；所有 fixture 和 mock 都明确是合成测试数据：

```bash
python -m pytest -q tests/test_reasoning_learning.py
python -m cloth_agent.harness.reasoning_learning --help
```

如果已有选点前 evidence 文件，可以先验证输入，不调用模型：

```bash
python -m cloth_agent.harness.reasoning_learning \
  --evidence /path/to/pre_decision_evidence.json \
  --output results/reasoning_prepare \
  --prepare-only
```

然后直接复用刚冻结的证据，进行串行搜索：

```bash
python -m cloth_agent.harness.reasoning_learning \
  --evidence results/reasoning_prepare/evidence/evidence.json \
  --output results/reasoning_serial \
  --search serial --variants 3 --repeats 2 \
  --backend local --max-calls 40 --max-seconds 1800
```

同一 Z，从同一个 H0 出发生成分支：

```bash
python -m cloth_agent.harness.reasoning_learning \
  --evidence results/reasoning_prepare/evidence/evidence.json \
  --output results/reasoning_branch \
  --search branch --variants 4 --repeats 2 \
  --grasp-mode candidate --epsilon-target 20 --epsilon-anchor 20
```

使用 collector manifest 时，将 `--evidence ...` 换成 `--manifest /path/to/manifest.json --decision-id ACTUAL_DECISION_ID`。远端使用 `--backend remote --ssh-host company-planner`；可加 `--model`，省略时保留现有 Claude 配置。所有输出目录必须互不相同。

复用选出的 harness：`--initial-harness results/reasoning_serial/selected_harness.json`。这只复用程序，不复用已选 action；可以搭配新的 `--evidence`。`--variants 0` 仅测基线，没有足够独立 harness 时预期不会选择 winner。

返回码：`0` 为 PREPARED 或 SELECTED；`2` 为没有形成可选共识、成本缺失、预算退出或错误；具体区别见 `report.json`。输入文件缺失时写 `blocked.json`，不会发生 Claude 调用。

## 共识与成本

默认按相同候选 ID 聚类，同时要求目标点、关系锚点距离在阈值内，关系枚举相同。`--grasp-mode distance --epsilon-grasp 12` 可改为抓点像素距离。所有距离阈值单位都是当前完整 clean 图像像素，分辨率变化时应重新配置。

使用完整相容簇：簇内任意两个 action 都要相容，禁止 single-link 的长链效应。要求唯一最大簇、默认至少三个不同 harness、占已冻结不同 harness 的比例至少 `0.6`。重复同一个 harness 只算一票；改名但不改实际程序不算新版本。一个 harness 的所有重复 rollout 必须都 READY 且互相相容，否则不参与投票，但仍计入版本总数。

在 dominant cluster 中，以各版重复结果的**中位成本**选择最低成本 harness。最终 action 取自该 harness 的一次真实 rollout，不平均合成坐标。最快的离群版本不会被选中。不同自然语言程序可能仍高度相关，票数不表示独立证据量。

成本：`--cost-weights TIME CALLS TOOLS THINKING`，默认 `1 0 0 0`，即规划墙钟耗时。若非零权重对应的指标不可用，整个成本选择返回 COST_UNAVAILABLE，不将 missing 当作零。

每轮记录耗时、后端调用数、可见工具调用数、输入/输出 token、cache read/write token、显式 thinking token 和供应商报告费用。模型未明确提供 thinking tokens 时为 `null`，不拿 output tokens 或公开文字长度推算。StructuredOutput 往返也计入可见工具数；不把它称为视觉探索。

`rollout_seconds` 和 `reflection_seconds` 分开；总 inner-loop 时间、token 和费用包括所有候选、失败及反思。仅对实际模型测量报告 baseline 与所选版的 planning 中位耗时比。修炼的总时间可能更长，不能只报最优 rollout 的耗时而隐藏学习成本。固定顺序和供应商 prompt cache 也可能影响结果；当前是探索性比较，不是随机对照或无偏泛化测试。

## Debug 产物

- `events.jsonl`：实时输出 session、call、stage、rollout、reflection 事件、时间、hash、状态和错误。
- `evidence/`：冻结 Z、所有实际输入图片、尺寸、像素 hash 与 lineage。
- `harnesses/`：H0 和每个通过校验的版本，记录 parent hash、来源和时间。
- `rollouts/<id>/harness.json`：本轮实际使用的程序快照。
- `rollouts/<id>/calls/<stage>_input.json`：即使 CLI 启动失败也保存完整请求、schema 和 evidence hash。
- `rollouts/<id>/calls/<stage>/`：RuntimeClaude 原始命令、多模态 input、stdout/stderr、解析返回、调用审计；远端还有 transport debug。
- `state.json / result.json / call_audits.json / exception.txt`：每阶段公开状态、校验、分阶段耗时、最终 action 或拒绝原因、异常栈。
- `reflections/patch_*/`：反思输入、原始提案、校验与拒绝原因、成本、`harness.diff`。不存在预写 learned harness。
- `report.json / report.md / index.html`：逐轮比较、两两相容关系、所有最大相容簇、未入选原因、加权成本及总开销。HTML 可展开查看每次公开决策和反思。
- `selected_harness.json / selected_action.json`：只在共识与成本条件成立后产生；仍标为不可直接执行的离线结果。

终端在每个边界即时打印日志，报告在每轮/反思后更新。意外退出可根据最后一个 call_start 和对应调用目录定位卡在哪一步；程序不自动恢复半次模型调用或把缓存答案计为重跑。

## 本机验证记录（2026-10-03）

新功能测试 41 passed，相关回归 146 passed。全量测试 1294 passed、21 failed、2 skipped；独立归档的未修改基线 c6a5e91 为 1253 passed、相同的 21 failed、2 skipped，没有新增失败用例。语法检查与 git diff --check 通过。

当前 /mnt/newssd 不存在，已有目标 run manifest 没有 pre-decision traces。实跑预检输出 `results/reasoning_learning_20261003_verified_preflight/blocked.json`，模型调用为零；该目录的 `verification.json`、`current_pytest.log` 和 `baseline_pytest.log` 保存软件验证记录。本机尚未获得真实 Claude 修炼结果、规划加速或物理验证；合成测试的选择结果不作为实验结论。
