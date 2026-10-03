# Candidate patch evolution（独立离线实验）

入口：`python -m cloth_agent.harness.patch_evolution`。这条流程把 patch 当成有版本、有测试、有回放和晋升证据的优化动作：

```text
Plan baseline → Diagnose → PatchProposal → Implement in candidate directory
→ mandatory gates → fresh replay → consensus/cost comparison → working or reject
```

它不修改正式源码、不接机器人、不修改原主循环、grasp depth/Z、grounding、preflight、IK、evaluator 或 experience。现有 `information_flow.ObservationHost` 已改成 registry 分发，默认内置行为保持兼容；旧 reasoning-learning 入口继续存在。此入口单独验证新的候选进化机制。

## 开放范围与真实运行链

- `skills/registry.py`：`ObservationSkill.applicable / prepare / execute / validate_output` 与 `SkillRegistry`。每个 skill 用 `id@vN` 和内容 hash 标识；同版本不可覆盖，返回副本，执行期间检查 registry 未改变。
- `executors/restricted.py`：解释运行候选的受限 Python 源码。
- `executors/observation.py`：唯一持有图片 IO、变换、缓存、预算和 provenance 权限的 Host。
- `candidate_patch.py`：候选目录、类型边界、源码 diff、内容封存及强制测试。
- `patch_evolution.py`：运行时 Claude 诊断/实现、独立回放、比较和 working 晋升。
- `patch_lifecycle.py`：人工或外部系统审核过的真实物理结果归档及后续离线 frozen 导出。

Claude 调用复用 `RuntimeClaude → LocalClaudeBackend / RemoteClaudeBackend`，模型真正接收 base64 图片。Claude 无文件修改工具；它以结构化结果返回源码和测试数据，Host 只写当前 candidate 目录。正式仓库不是模型的修改目标。

默认 baseline 是本入口的固定观测请求阶段 + 现有 reasoning harness；不是原自由工具 planner 的历史耗时。比较各方使用相同的 baseline 执行边界。

## PatchProposal 与修改层级

诊断输出包含 `level / target / problem / proposed_change / expected_effect / must_preserve / evidence_rollouts / unverified`，必须引用真实发生的 baseline rollout。提案不立即生效。

- `PROMPT`：只能改 observation 或 reasoning 文字，不能改 stage topology、变量绑定或能力集。
- `HARNESS`：可改受限 configuration、阶段和上下文绑定；不能包含源码。
- `SKILL_CODE`：修改现有 observation skill，version 必须精确加一；保留其他能力、baseline prompt 和 reasoning。
- `NEW_TOOL`：新增 skill ID，初始 version 为一；只允许新增对应 capability，不夹带 reasoning 改动。

每个候选独立从同一 baseline 分支产生。无效候选立即拒绝，不无限修复；下一次提案是新的、预算内的候选。只改名称、空白或源码注释不会增加共识票数。语义等价的一切程序无法由 Host 完全识别，因此不同程序仍可能高度相关。

## 候选实现究竟能做什么

`implementation.py` 是实际被解释执行的 Python 子集，唯一入口：

```python
def prepare(request, source, available):
    roi = request["roi"]
    return {
        "roles": ["clean", "overlay"],
        "operations": [{"op": "crop", "roi": roi}],
        "reuse_existing": True,
    }
```

上例只是接口说明，产品没有把它作为 Claude 的编译结果或预置获胜策略。

允许局部赋值、if/else、return、数值运算、比较、容器/索引和 `min/max/abs/round/int/float/len/floor/ceil`。Host 在解释每一步时执行数量、深度、数值和容器大小限制。没有 imports、属性访问、循环、递归、eval/exec、任意函数调用、文件或网络权限。源码通过 Python syntax compile 检查，但不会用 Python `exec` 或普通 import 执行未经隔离的模块初始化。

`request` 使用现有 gap、skill ID、source image、ROI、rotation、enlarge 合同。`source` 只含本轮图片 ID、role、size，不传图片 fingerprint、历史答案或 candidate registry。`available` 含缓存视图数与剩余操作预算。

返回 recipe 可动态计算 ROI、组合最多四个 crop/rotate/resize、选择 clean 或 clean+overlay、决定复用已有变换前缀。新增 `paired_roi_reuse` 这类实现可以真正加入 registry；clean 已准备而 overlay 缺失时，Host 能复用 clean 并只生成缺失部分，不重复附图。缓存严格限于单次 replay，跨 rollout 不继承。

第一版扩大的是**有边界的 observation 实现空间**。可信像素原语仍为 crop/right-angle rotate/resize；不允许自动安装库、任意像素处理 Python 或扩张机器人权限。新的底层像素原语需要开发者扩展 Host 合同及对应 invariants。不是通用自改整个程序的 agent。

## 测试 gate

每个候选目录包含：

```text
candidates/patch_00/
  base_version.json
  proposal.json
  implementation.json
  config.json
  bundle.json
  patch.diff
  seal.json
  cloth_agent/harness/skills/<id>/
    skill.json
    implementation.py
    tests/cases.json
    tests/test_skill.py
    evidence.json
  tests/case_*/
  gates.json
  evaluation.json
```

配置 patch 没有 skill 源码目录；仍执行配置合同与内置技能回归 gate。code/tool patch 必须包含至少两个 Claude 生成的合成测试数据及 expected recipe；`test_skill.py` 是 Host 生成的可重复 pytest 包装，不执行任意模型生成的测试代码。可在项目根目录执行 `python -m pytest /absolute/candidate/path/.../tests/test_skill.py`。

不可绕过的测试包括：

1. 源码语法、AST capability allowlist、有限执行和输出 schema。
2. 配置字段、stage 绑定、目标 skill ID/version、patch 层级边界。
3. 候选测试的 expected recipe；所有既有 capability 的合法请求回归。
4. 独立 Pillow 像素 oracle 与像素中心 affine oracle；clean/overlay 尺寸和变换一致。
5. 零预算不得产生任何图片操作；整批请求非法时不得部分执行；重复观察必须停止。
6. 原始图片不可覆盖，图片/hash 检查，禁止 robot/module/IO 权限。
7. 测试后的 source/spec/tests/config/bundle 内容 hash 必须与 seal 一致，回放前后再核对。

gate 不调用模型。schema 或代码错误不能解释成视觉能力不足；失败候选不进入昂贵的 planning replay。测试覆盖的是执行合同，不证明该新方法视觉上充分或物理上正确。

## Replay 与晋升

所有版本从相同冻结选点前根图、goal 和 registry 重新开始。不传 baseline action、选点理由、后验图片、evaluation 或旧 planner crop。新的观测请求由当前候选配置在本轮产生；派生图由通过 gate 的当前 implementation 生成，保留完整 affine lineage。

因此 `root_evidence_hash` 在各版本一致，`derived_evidence_hash` 可以不同。模型在选点阶段必须解释本轮 UNKNOWN gap；READY 仍不允许 blocking missing information，Host 不能用“已经 crop”代替语义解释。

输出 A 仍是现有离线 reasoning 合同：当前 registry 抓点与原图视觉目标/anchor/relation。**没有三维动作 grounding**；candidate_legal 只表示当前 registry 与像素坐标检查。所有 working/frozen 产物均 `robot_executable=false`。

复用 complete-link action consensus：抓点/target/anchor 相容、关系相同，簇内所有 pair 相容；不同 harness 一票，重复测量不增加票。默认至少三个版本、唯一 dominant cluster；失败或未完成的已准入版本不会消失成虚高覆盖率。

working 条件全部满足才晋升：

- 精确版本通过 mandatory gates；所有重复 replay READY 且视觉候选合法。
- baseline 与 candidate 根证据一致，有真实模型响应；mock 不可晋升。
- 是 dominant cluster 中成本最低的版本，且 baseline 也在这个簇内。
- 候选中位 planning wall time 相比 baseline 至少降低 `--min-latency-reduction`（默认 0.1）。
- 无额外 unresolved blocking information。

baseline 耗时包括本轮 observation Claude、Host 图像处理、reasoning、传输和校验；候选同样。generation、tests、reflection 和所有失败的总开销另列，不能拿单个最快 rollout 代表总体加速。缓存 token、输入/输出 token、可见 tool 往返和 thinking token（未报告则 null）沿用现有审计。

获胜者只复制到本次实验的 `working/<bundle_hash>/`，不改全局 active registry，不自动永久冻结。无共识、无收益、基线失败、预算或 gate 不足均明确拒绝。

## 单独运行

软件验证：

```bash
python -m pytest -q tests/test_patch_evolution.py tests/test_information_flow.py
python -m cloth_agent.harness.patch_evolution --help
```

使用之前的合法 pre-decision evidence 或 reasoning-learning 冻结的 `evidence/evidence.json`：

```bash
python -m cloth_agent.harness.patch_evolution \
  --evidence /path/to/evidence.json \
  --output results/patch_evolution_prepare \
  --prepare-only
```

有真实 Claude 配置和数据时：

```bash
python -m cloth_agent.harness.patch_evolution \
  --evidence results/patch_evolution_prepare/evidence/evidence.json \
  --output results/patch_evolution_trial \
  --patches 3 --repeats 2 \
  --max-calls 60 --max-seconds 1800 \
  --max-host-ops 12 --min-latency-reduction 0.1 \
  --backend local
```

也可使用 `--manifest /path/to/manifest.json --decision-id ACTUAL_ID`；远端为 `--backend remote --ssh-host company-planner`，模型可用 `--model` 指定。输出目录必须是新目录。测试已有 working 时，可指定 `--baseline-bundle /path/to/working/<hash>/bundle.json`；它会验证该版本的测试和 working 证据，不接受未通过测试的任意 baseline 文件。

返回码 0 表示 PREPARED 或 WORKING；2 表示未晋升或失败，具体状态见报告。

## 真实执行之后：独立审核，不自动归因

`patch_lifecycle` 不执行机器人，也不修改现有 evaluator。它接收外部物理记录与明确 reviewer 的审核结果。记录必须绑定 `bundle_hash / observation_id / root_evidence_hash / action_hash / outcome`，并提供可读的物理记录文件和 replay report 内容 hash。

审核 JSON 按 `OUTCOME_SCHEMA`，包括：

```text
schema_version, bundle_hash, observation_id, root_evidence_hash, action_hash,
physical_record_path, physical_record_hash, outcome（SUCCESS/FAILURE/UNKNOWN）,
reviewed_by,
attribution: {task_decision, grasp_depth, target, observation_patch, rationale},
replay_report_path, replay_report_hash
```

四种归因均为 CAUSE / CLEARED / UNKNOWN。`action_hash` 必须对应这一版本在该状态的一次 READY replay 的完整 action；物理记录也必须绑定这个 action。旧的未绑定 run outcome 不能直接拿来认证新 patch。

```bash
python -m cloth_agent.harness.patch_lifecycle review \
  --working /path/to/working/BUNDLE_HASH \
  --review /path/to/externally_reviewed_outcome.json \
  --output /path/to/review_ledger/episode_001.json

python -m cloth_agent.harness.patch_lifecycle freeze \
  --working /path/to/working/BUNDLE_HASH \
  --reviews /path/to/review_ledger \
  --output /path/to/offline_frozen/BUNDLE_HASH
```

未知 outcome、patch 归因未清楚或 patch 导致失败均 HOLD。失败要认定 patch 无责，还必须明确至少一种 task/depth/target 原因。保守地要求至少三份不同 clean RGB 内容的已审核状态，且所给 ledger 中没有未解决记录；仅换 observation ID 不算新 state。冻结时重新核对所有引用文件与 tested bundle。

这是一条**外部证据审核接口**，不是系统自动证明物理因果。物理 record 与 reviewed_by 的真实性依赖调用者提供的可信外部记录；不要排除不利 episode 来制造完整 ledger。导出产物为 `FROZEN_OFFLINE_LIBRARY`，仍不自动激活到真实机器人。

## Debug

- `events.jsonl` 和终端：逐 candidate、call、gate、replay 的事件、预算、状态、错误。
- `diagnose/implement/`：实际多模态请求、原始 stdout/stderr、模型配置、token 和调用审计。
- `patch.diff / seal.json / gates.json / gate_exception.txt`：执行变更、精确版本与各 gate 结果。
- `replays/*/observe/`：当前根图上的请求产生过程。
- `replays/*/observations/execution.json`：skill ID/version/hash、参数、复用视图、实际操作和完整 lineage。
- `replays/*/reasoning/`：逐阶段请求、公开概念、格式校验、最终 action 或原因。
- `report.json / report.md / index.html`：所有候选的拒绝/晋升原因、两两相容性、成本、总学习开销；HTML 链接原始 artifacts。
- `working/`：只有通过所有晋升条件时才存在。

软件测试与 gate 使用合成图片、mock policy 的结果只用于验证机制，不计为真实效率或物理成功实验。

## 本次验证（2026-10-03）

新增 43 个合成/mock 测试，相关回归合计 178 passed；生成到候选目录的独立 pytest 包装另测 2 passed。完整测试为 1376 passed、21 failed、2 skipped；未修改 HEAD `94701ff` 为 1333 passed、21 failed、2 skipped，失败 test ID 集合完全一致。

实际重新收集 `fold_20260924T014204289928777Z`，仍因 `/mnt/newssd` 未挂载而发现 0 条决策。新入口在调用 Claude 前明确返回 BLOCKED；没有真实生成 patch、速度收益、共识或 working/frozen 实验结论。审计文件位于 `results/patch_evolution_20261003_preflight/verification.json`，本次新 manifest 为该目录下 `collection/manifest.json`，入口结果为 `fresh_preflight/blocked.json`。
