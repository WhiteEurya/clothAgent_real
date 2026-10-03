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

缓存仅限同一 rollout 的显式阶段状态。后一轮不会拿前一轮的 action、ROI、理由或反思正文作 planning 提示。反思可读取上一轮公开输出和成本，但它的解释与证据保存于 sidecar，只有通过校验的 harness 进入重跑。可执行 prose 要求尽可能参数化：抓点、目标、锚点、区域和几何量从当前观察、目标和候选表推导，避免复制历史答案。数字、步骤编号、阈值和技术示例不触发词法拒绝；是否真正可泛化仍需后续实验判断。Host 不把“删除某步可能更快”的假设记为已验证消融结论。

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

## 本机 k=3 测试入口

`--k` 是 `--variants` 的别名，表示初始 H0 之外最多三次新 harness 提案机会；无效、重复提案或提前 STOP 可能使实际版本数少于四个。`--repeats` 是每个版本独立规划的次数，不是 k。

已校验的左袖选点前证据：`results/reasoning_k3_prepared_20261003/evidence/evidence.json`（prepare-only，未调用模型）。从项目根目录运行：

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.harness.reasoning_learning \
  --evidence results/reasoning_k3_prepared_20261003/evidence/evidence.json \
  --output "results/reasoning_k3_$(date +%Y%m%d_%H%M%S)" \
  --search serial --k 3 --repeats 2 \
  --backend remote --ssh-host company-planner \
  --max-calls 60 --max-seconds 3600 \
  --call-timeout 300 --rollout-timeout 900
```

最多四个不同版本，每版两次规划，另有最多三次反思；多阶段版本的一次规划会包含多次模型调用。默认以规划耗时选优，仍要求至少三个不同版本形成唯一稳定共识；没有共识时输出 NO_SELECTION，不强行选择最快版本。此次修复补充读取 `usage.output_tokens_details.thinking_tokens`，保留顶层 thinking_tokens 和嵌套 reasoning_tokens 的兼容，以及未知值不计作零的规则。

## Token 总预算

使用 `--max-tokens 200000` 可设置整个 inner loop 的累计预算，不指定则不限制 token（仍记录用量）。规划、反思以及有用量回执的失败调用共用预算。计数口径为供应商返回的 input_tokens + output_tokens + cache_read_input_tokens + cache_creation_input_tokens；thinking 已包含在 output 内，不重复加。该口径不是美元费用，缓存 token 与普通 token 价格不同。

每次实际调用后写 `token_usage` 事件和 `token_budget.json`；最终 `report.json` 的 `totals.token_budget` 保留限额、已知累计值、剩余额度、未知回执数量和停止原因。达到限额时停止后续调用（TOKEN_BUDGET_EXHAUSTED）；用量不完整时，启用预算的实验停止后续调用（TOKEN_USAGE_UNKNOWN），不把缺失用量当零。

预算是调用间检查，不能在 Claude 调用中精确截断：单次调用可能超出剩余预算，超出部分会如实记录。现有结果仍可参与共识选择，但报告必须同时查看 token_budget；达到预算不一定意味着没有可选结果。控制 token 不改变 --k、--repeats 和共识要求。

## 视觉判断的缺口与不确定性

阶段输出新增必填 `residual_uncertainty`：保存不阻止当前视觉提案的残余误差、限制和后续物理验证事项。`missing_information` 仅保存会阻止当前视觉决策的信息缺口。READY 必须允许提前完成、包含完整 action 且 missing_information 为空；CONTINUE/NEEDS_LEARNING 的 action 必须为 null，后者还必须说明具体缺口。这些约束同时写入模型 schema 和提示，Host 不自动清空缺口或把阻塞问题改成残余不确定性。历史实验保留原样，不重新标成成功。

校验记录 READY_BLOCKING_GAP、READY_NOT_ALLOWED、READY_ACTION_MISSING、NON_READY_ACTION 等具体原因，并在阶段记录里保存 validation_error。反思被明确告知字段冲突不是抓点错误或必须拆分阶段的证据。缺少深度/IK 不自动阻止 RGB 视觉提案，也不代表已经满足真实执行条件。

当前 reasoning learning 已移除 harness 说明文字的数字、路径、URL、历史 ID 和 base64 关键词拦截，名称与阶段 ID 允许 `refined_v2`、`stage_1`。反思提示要求尽可能参数化、解释固定启发式的适用条件，并优先使用场景相对量。提到路径或示例不授予文件访问权限，也不使历史答案成为当前证据。仍校验 schema、阶段依赖、当前图片及候选身份、坐标和输出状态；通过校验不代表参数化质量或视觉正确性已获验证。此调整针对 reasoning learning，旧 compiler 的 policy 校验未改动。

输出通道兼容：RuntimeClaude 不再因为仅用于查找 StructuredOutput 的 ToolSearch 请求拒绝整次返回。接受精确查询 `StructuredOutput` 或 `select:StructuredOutput`，并在 `output_tool_lookup_calls` 中单独计数；总工具调用、token 和耗时仍保留。不因此开放 Read、网络搜索或其他证据工具；宽泛/混合查询仍不符合固定输入实验合同。此修改调整 Host 校验，不保证 CLI 环境实际提供 ToolSearch；查找失败时可以直接返回符合 schema 的 JSON，仍须通过原有结果校验。

## 轻量格式预检与自动修复

实验开始先运行本地合成合同预检，输出 `format_preflight.json`，不调用 Claude。每次判断或反思返回后保存 `format_repair.json`（原始返回、规范化结果、字段差异、规则、模型调用数），再执行原有校验。原始 proposal.json 和模型 returned.json 不覆盖。

当前确定性修复范围：旧判断缺少 residual_uncertainty 时补空字符串。阶段指令中的 `(1) … (2) …` 编号原样保留，不再改成项目符号，以保留后文对步骤编号的引用。数字坐标、历史图像编号、抓点、目标、状态、阻塞缺口不改。其他结构或一致性问题仍由原有校验报告，不靠模型猜测字段值。

若某版本所有规划均为合同拒绝，停止对该版本做视觉反思并报告 FORMAT_BLOCKED；其他 ERROR 行不把被拒绝的阶段判断送作视觉失败证据。修复仅意味着格式可接受，不表示策略正确或加速有效。旧实验结果不回写、不追认；离线回放检查仅输出新诊断目录。

## 以降低耗时为目标：只优化一次

反思的首要目标改为降低完整视觉规划的墙钟耗时，保持判断质量和稳定性；token 和调用次数作为次要成本统计。`optimization_objective` 随实际反思请求保存。Claude 自由提出删除、合并、重排、按需深入、提前结束等可执行假设，不预定操作流程；新增阶段必须解释它取代了哪些更昂贵的工作。当前后端仍只支持固定图片与阶段/提示编排；需要新增 Host 算法或工具的提案必须说明未实现，不能当作已经执行。

反思可同时输出 `operations`，描述信息目标、方法、输出、成功条件、信息不足处理、被替代工作及预计省时机制。真正参与下一轮规划的是返回的 harness；操作说明不会触发额外调用。该字段可省略以兼容旧输出，缺失时导出空候选列表，不凭空补写经验。

从项目根目录运行以下命令：

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.harness.reasoning_learning \
  --evidence results/reasoning_k3_prepared_20261003/evidence/evidence.json \
  --output "results/reasoning_opt1_$(date +%Y%m%d_%H%M%S)" \
  --search branch --k 1 --repeats 2 \
  --backend remote --ssh-host company-planner \
  --min-harnesses 2 \
  --max-calls 30 --max-seconds 3600 \
  --call-timeout 300 --rollout-timeout 900
```

执行顺序：H0 基线规划两次 → 一次 Claude 优化提案 → H1 规划两次。`k=1` 是一次提案机会，不是总共只调用一次模型；STOP、无效或重复提案不会产生 H1。单阶段 H1 通常共五次模型调用，多阶段会增加调用次数。没有 token budget，仍完整统计用量。只读取已保存图片，不执行机器人。

`optimization_comparison.json`、`report.md`、`index.html` 会展示各版本中位耗时、调用数、token、重复稳定性及基线/候选耗时比，即使没有形成共识也保留比较结果。只有真实测量且两版所有重复均 READY 才给耗时比；比值大于一只表示本次观察到更短耗时，不说明判断正确或稳定。`min-harnesses=2` 使两个版本有机会形成共识，并不保证能入选；没有共识仍返回 NO_SELECTION（CLI 退出码 2），不等于实验崩溃。

`operation_candidates.json` 单独保存抽象操作候选，初始 UNVERIFIED，不写入 approved skills。evidence 的 supported_by/failed_in/unresolved 只表示观察方法是否得到所需信息：本轮整体 READY 和耗时不能证明单个方法成功，因此试跑引用先归 unresolved，等待方法层面的核对。总学习耗时和 token 仍在 totals 单独记录。固定顺序、单场景实验不能证明普遍加速或正确性。

## 可执行补丁与 Host 数值操作

每个通过校验且非重复的新提案在测试前写入 `reflections/patch_*/executable_patch.json`，包含父版本、新版本哈希、实际 harness 及 Host 操作数量，测试后追加逐轮执行结果。这是由解释器执行的声明式补丁，不是任意生成的 Python 源码。提示优化仍可单独作为补丁；只有显式绑定的 Host 操作才会执行，候选文字不会自动变成工具调用。

新增已实现操作：

- `reflect_point`：点关于直线的镜像。
- `affine_point`：六参数仿射坐标变换。
- `rank_candidates`：仅对当前候选表中指定子集按给定轴投影排序。

Claude 在先前阶段的 `measurements` 输出当前观察的测量值；后续阶段通过 `host_operations` 绑定这些值，Host 在下一次模型调用前执行计算，结果以 `host_results` 提供给模型。示例阶段片段：

```json
{
  "id": "decide",
  "instruction": "Use host_results.mirror to propose the target only after verifying the mapped point against the current fabric and reference end-state. UNKNOWN requires resolving its missing evidence or stopping.",
  "context": ["measure"],
  "allow_ready": true,
  "host_operations": [{
    "id": "mirror",
    "op": "reflect_point",
    "source_stage": "measure",
    "bindings": {"point": "grasp", "line_start": "hinge_a", "line_end": "hinge_b"}
  }]
}
```

`host_results` 实际是按操作 ID 标记的记录数组。前序 `measure` 阶段应输出 `measurements.grasp`、`measurements.hinge_a`、`measurements.hinge_b`（二维当前 clean 像素坐标）。未知值可以为 null，不从历史补坐标。每轮 `host_execution.json` 保存实际输入、输出、耗时与失败缺口；`metrics.host_operation_count` 和 `host_operation_seconds` 单独统计，且仍包含在规划总耗时中。缺失测量、零长度轴等返回 UNKNOWN，不能当作成功几何。

Host 数值计算不识别袖子或缝线，不证明点在布料上，也不自动裁剪越界目标。模型仍需视觉验证，最终动作仍经过当前候选/坐标检查。操作说明与模型视觉判断是否一致尚不能由 schema 保证。额外阶段可能抵消计算省时，反思必须评估总成本；新增目录之外没有机器人动作或正式 skill 更新。

仍可使用上面的命令，改为 `--k 1 --repeats 1`，执行基线一次、提案一次、新方案一次。新方案如采用两阶段“测量→Host 计算→判断”，共四次模型调用，而不是三次；Host 计算本身不调用模型。旧实验不会自动获得这些能力，需启动新的实验。
