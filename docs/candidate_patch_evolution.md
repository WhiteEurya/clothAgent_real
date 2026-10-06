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

## 快速单补丁实验：耗时优先

诊断和实现阶段明确以降低完整 replay 墙钟耗时为目标，涵盖观察调用、图片传输、Host 处理和全部规划阶段；token 与调用次数作为次要指标记录。诊断请求包含 `optimization_objective`、`evaluation_policy`（提速门槛、共识要求、重复次数），报告也保存目标。要求解释省掉或替代哪些工作，不因答案分歧自动增加阶段，也不能为赶时间隐藏真正的信息缺口。具体补丁方向仍由 Claude 选择，不强制生成源码。

`--patches 1` 默认要求两个版本形成共识；其他设置仍默认三个。可用 `--min-harnesses` 显式覆盖，该值不会被自动降低。单次试跑只用于探索性比较，不能衡量重复稳定性。候选仍须测试通过、与基线满足共识条件、真实测量且达到默认 10% 提速，才可能标记 WORKING；正确性与物理效果未因此得到证明。

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.harness.patch_evolution \
  --evidence results/reasoning_k3_prepared_20261003/evidence/evidence.json \
  --output "results/patch_opt1_$(date +%Y%m%d_%H%M%S)" \
  --patches 1 --repeats 1 \
  --backend remote --ssh-host company-planner \
  --max-calls 15 --max-seconds 1800 \
  --call-timeout 300 --replay-timeout 900
```

顺序是基线一次、诊断提案一次、实现一次、测试、新方案一次。基线/新方案各包含观察和规划调用，单阶段正常路径共六次模型调用；没有自动增加重复次数。基线失败或测试拒绝会提前停止相应路径。仅使用保存图片，不执行机器人，不设置 token 上限。

## 简化实验前置拦截（2026-10-04）

当前补丁实验调整如下（以本节为准）：

- 不再要求每个 UNKNOWN 配一个观察请求。所有信息缺口原样交给规划阶段，由模型判断哪些真正阻止当前 RGB 视觉决策；物理执行验证可作为 residual_uncertainty。真正阻塞的缺口仍必须输出 NEEDS_LEARNING，不能自动清空或标成已知。
- 不再用最终 concept 名称是否完全等于缺口 ID 判断任务完成。仍检查输出结构、图片引用、候选身份、坐标和 READY/missing_information 一致性。语义正确性需另行核对，不由字段名证明。
- 观察请求数量上限从 3 调整为 12，与信息条目上限一致；实际图像操作仍受 max-host-ops 约束，一个视图可以供多个信息任务使用。
- 补丁 replay 启用重复观察复用：相同操作链只生成一次图像，后续请求复用同一 image ID 并记录 reused_image_ids，不增加图像操作数，不把复用计成新的独立证据。旧 information_flow 的补充观察停止策略不变。
- SKILL_CODE/NEW_TOOL 可配套更新观察和规划提示；所有变化写入 config/diff，评价的是整个补丁，不再声称隔离了代码贡献。既有代码执行范围、图像坐标检查和版本约束不变。

保留最终筛选：测试通过、相同决策前输入、有效规划、基线与候选共识、最低成本及默认至少 10% 总耗时改善。基线真正失败仍不进入炼化。图片身份、坐标对齐、输入不可覆写、执行资源上限与受限代码解释器继续保留。

### KNOWN 附带缺失信息说明：非阻塞提示

`KNOWN` 的 `missing_information` 非空不再直接拒绝。观察原文保留在 `observation.json`；`information_validation.json`、replay 结果和事件日志记录 `KNOWN_WITH_MISSING_INFORMATION`，并将原文与提示一起交给后续规划。提示要求区分“无缺失”的解释、对其他信息项的引用和真实阻塞缺口，不自动清空字段或把未知变为已知。真正阻塞的视觉问题仍应由规划返回 NEEDS_LEARNING。

没有图片依据、引用不存在图片、空 finding、结构错误等仍拒绝；最终规划的 READY/action/missing_information 一致性与补丁筛选条件未放宽。

### 提案引用的轻量规范化

诊断 schema 将 evidence_rollouts 限定为本轮实际 rollout ID；可选 evidence_explanation 保存解释。Host 兼容带说明的旧引用：仅在词边界明确且唯一对应一个已知 ID 时提取并去重，原文及无法解析的条目记录在 citation_audit.json，并作为补充说明传给实现阶段。图片说明不冒充 rollout 引用。proposal.json 保留模型原始返回，proposal_normalized.json 保存真正送入实现的提案。

至少一个有效引用时继续；全部无法解析时最多进行一次 patch_citation_repair，只修正引用字段，不重跑基线、诊断或改变其他提案字段。修正仍无支持记录则停止该候选；调用计入原有时间、调用数与 token 统计。引用身份有效不代表引用中的因果解释已经验证。

### 已知信息也可以请求辅助图像操作

KNOWN/UNKNOWN 只描述信息状态，不再作为 Host 图像操作的权限条件。请求的 gap_id 必须对应已有信息项，但可以针对 KNOWN 执行转正、放大等辅助操作；expected_information_gain 说明预期用途。观察提示同步说明这一点，不要求模型为操作而把 KNOWN 改为 UNKNOWN。

execution.json 每条历史增加 information_status_at_request，保留发起时状态，便于之后评估该操作是否多余。原始信息不改写。未知信息 ID、无效图像/参数、坐标对齐、操作预算和最终补丁筛选继续保留。

### Observation source lineage

Observation requests may reference any available catalog image, including overlays,
references, and intermediate crop/rotate/resize outputs. ROI coordinates are local
normalized coordinates of the selected image, not coordinates of the original.
Every executed intermediate is published in the catalog with `parent_image_id`,
`original_image_id`, operation/arguments, `to_parent`, `to_original`, and the full
lineage. Derived reference images retain their reference identity; processing does
not turn reference geometry into current-scene geometry.

Single-view recipes process the chosen image. Paired clean/overlay recipes replay
the selected view's operation chain on the aligned counterpart before applying the
new recipe. Unknown sources, changed pixels, unaligned pairs, and resource limits
remain errors. Only existing catalog IDs can be requested; dependent requests use
the IDs returned by a previous host execution, not guessed future IDs. Planning
evidence includes the lineage as well as the final affine mapping.

### Combined orientation and measured implementation scope

The built-in orientation recipe accepts optional ROI and enlargement: crop in the
selected source coordinates, rotate clockwise, then enlarge if useful. Null ROI
preserves full-view rotation. Each operation retains its own affine lineage. An
unsupported generated recipe returning null now reports the skill and parameters
instead of a generic JSON object error.

Replay metrics include `exclusive_phases_s`: observation call, Host dispatch,
planning, and other work, whose sum equals `elapsed_s`. Planning submetrics and
Host compute are nested measurements, not additional elapsed time. The diagnosis
prompt explicitly explains this and distinguishes aggregate model calls from
observation rounds. Failed/incomplete replays have no latency-reduction score.

Each candidate now records `implementation_effects.json` computed from actual
config/source differences. This reports prompt changes, topology, enabled skills,
and generated skill code. The existing isolated contract cannot edit Host dispatch;
writing a merging/dispatch policy into a prompt does not implement a deterministic
Host gate. This is disclosed to both proposal and implementation calls and in the
report without adding a prose-based rejection rule. Proposed speedups remain
unverified until a complete candidate replay.

### Complete baseline first; reuse completed measurements

Patch replays no longer abandon the whole observation batch when its estimated
operation count exceeds the Host limit. In request order, the Host executes whole
requests that fit (including exact-cache reuse); overflow is recorded as DEFERRED
with no delivered evidence. Planning receives executed views and the explicit
unexecuted requests, and decides whether information is sufficient. There is no
forced READY or fabricated observation. This behavior applies equally to baseline
and candidate. Legacy information-flow batch behavior is unchanged.

The CLI now defaults to a verified baseline cache at
`<output-parent>/.patch_baseline_cache`. The first complete READY baseline is saved
before optimization begins, so a later candidate failure does not lose it. Reuse
requires matching evidence/task, baseline bundle, runtime code, model configuration,
repeat count and execution limits; copied artifacts are checked by file hashes.
The cache key uses the configured model selection, so use an explicit model or a
fresh run if the provider's implicit default changes. Failed baselines are not
cached as successful measurements. Candidate observation/planning still receives
only the same pre-decision evidence, not the cached baseline's answer.

Use `--baseline-cache-dir PATH` to select a cache or `--no-baseline-cache` for fresh
measurements. A hit preserves original timing for comparison and is marked
historical, never a new independent sample. Current-run call/token totals and
replay/Host totals exclude the copied historical artifacts; cached baseline time
is reported separately. Cache write failure does not cancel a valid experiment.

### Fixed-observation comparison for planning-only changes

When observation instruction, enabled skills and registry hashes are unchanged,
candidate replay copies the corresponding baseline `prepared/` package verbatim
and invokes only the candidate reasoning harness. Its original/derived image IDs,
lineage, information gaps and Host history are identical. The baseline's final
answer and reasoning are never passed to candidate planning. Observation-changing
patches still run a fresh observation pipeline.

`observation_reuse` identifies the reused baseline and evidence hash. Current
`elapsed_s`, calls and tokens measure only work actually performed. Promotion also
reports baseline and candidate planning seconds. Consensus/promotion total-time
comparison uses `comparison_elapsed_s`: shared baseline non-planning cost plus new
planning cost. This is explicitly a reconstructed total, not an independently
measured end-to-end speedup. Failed candidates still have no latency-reduction score.

For an already completed historical run, including across runtime updates, use
`--reuse-baseline-run results/patch_opt1_20261004_164702` with matching original
model/time-limit settings (`--call-timeout 600 --replay-timeout 900 --max-host-ops 12
--repeats 1`). Evidence, baseline bundle, settings, result and prepared image hashes
are checked before reuse. A mismatch fails without silently rerunning a baseline.
Historical measurements remain marked as such. This option skips baseline calls;
it still generates a new proposal/implementation, and planning-only candidates
then need just one planning call for the default single-stage harness.

### Live remote diagnostics and timeout recovery

Remote CLI nonterminal JSONL events now stream immediately, with timestamps from
the remote monotonic clock; raw output is still spooled for audit. Terminal results
are released only after complete-stream and process-exit validation. Partial events
on timeout do not count as a completed planning result. Retry events (including
status and attempt) and token/text/tool event types are preserved in the local
`transport/claude_events.jsonl` and `stdout.log` as they arrive.

`stderr.log` also contains `__CLOTH_PROGRESS__` diagnostics: CLI start, a five-second
heartbeat with event count, last event type and idle seconds, and CLI finish with
exit code and terminal availability. These describe observable CLI activity, not
internal model reasoning. A silent interval alone cannot distinguish upstream
queueing from other provider waits. Invalid output bytes are retained as base64
diagnostic records, never parsed as successful model output.

Remote work has a deadline shorter than the local SSH wait (30 seconds reserved
for a 600-second call, scaled down for short calls). Download/setup time is deducted
before launching CLI; GNU timeout terminates the CLI before the local waiter,
allowing partial diagnostics and status to return. If local SSH still times out,
all already received streams are flushed and saved as `claude_stdout.txt` and
`claude_stderr.txt` as well as the streaming logs. A dropped connection can still
prevent delivery; no fabricated completion or inference-time attribution is made.

### Capability-aware generation and one local implementation repair

Diagnosis and implementation now receive the explicit editable runtime contract.
Stage context refers only to earlier reasoning stages; observation evidence is
already supplied directly. Session continuation, provider cache controls, and
transport/dispatcher edits are not implemented by writing stage names or prompts.
`stage_costs` separates whole-replay counters from planning counters and calculates
observation counters as their difference; output tokens already include thinking.

Candidate structural validation runs without creating bundle/source files. On a
structural error there is one bounded `patch_implementation_repair` call with the
original proposal, initial implementation, exact error, schemas and capabilities.
It does not repeat baseline or diagnosis. Both initial and repaired artifacts are
kept. If preserving the proposed mechanism requires an unavailable runtime API,
the repair returns NEEDS_RUNTIME_SUPPORT instead of silently deleting references
and presenting an unrelated prompt as an implementation. A repaired candidate
still passes the same deterministic gates and actual replay before evaluation.

### Proposal transport and retrying a saved candidate

The proposal transport represents `must_preserve` as one quoted string, avoiding
provider/tool-call failures from unquoted repeated long prose array entries. Host
normalization preserves its contents in the canonical list representation; both
transport and canonical proposal artifacts remain available. This changes output
representation, not policy constraints or the allowed optimization mechanisms.

Use `--retry-candidate <old-run>/candidates/patch_00` together with
`--reuse-baseline-run <baseline-run>` and `--patches 1` to re-evaluate an already
materialized, tested candidate. The seal, baseline bundle and input evidence must
match. Diagnosis and implementation calls are skipped; deterministic checks are
rerun. For a planning-only candidate this makes exactly one new planning call.
The old run is preserved and the new output directory contains its own complete
result. Do not re-generate candidates merely to recover from provider timeouts.

### Sequential feedback for unattended offline runs

Each candidate now writes `feedback.json` with its actual implementation, replay actions,
measured phase costs, failure details, and provisional promotion checks. Before generating
another candidate, the runner writes `diagnosis_context.json` containing all completed
rollouts and previous feedback. Diagnosis can cite prior candidate rollout IDs; implementation
also receives the feedback so it can revise earlier code/configuration. Provisional consensus
is explicitly provisional and is recomputed at the end. Failed execution is never a speedup.

The baseline remains fixed for comparison. Each implementation is a complete replacement
relative to that baseline, not a diff blindly applied to the previous candidate. Earlier
answers are supplied only to optimization calls, never to the evaluated planner.

`--max-consecutive-failures` defaults to 3. Consecutive generation/validation failures or
ERROR/BUDGET_EXHAUSTED replays stop further candidates. A completed valid replay resets the
streak even if it is slow or not promoted. Existing finite call/time/operation limits remain;
tokens are counted without adding a token budget. Reports and candidate files stay in the
experiment directory; there is no automatic production activation or robot execution.

Example: reuse the historical baseline and explore up to five candidates, one replay each:

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.harness.patch_evolution \
  --evidence results/reasoning_k3_prepared_20261003/evidence/evidence.json \
  --reuse-baseline-run results/patch_opt1_20261004_164702 \
  --output "results/patch_iter5_$(date +%Y%m%d_%H%M%S)" \
  --patches 5 --repeats 1 \
  --backend remote --ssh-host company-planner \
  --max-calls 25 --max-seconds 7200 \
  --call-timeout 600 --replay-timeout 900 \
  --max-host-ops 12 --max-consecutive-failures 3
```

This is an exploratory search, not a repeatability/physical accuracy test. CLI exit code 2
can mean a completed search with NO_PROMOTION; inspect report status, stop_reason and rollout
statuses before treating it as a crash.

Transport stderr embedded in failure reasons is summarized before entering later prompts;
full raw errors remain in the report and transport files. This prevents repeated heartbeat
logs from exhausting the prompt limit. Semantic findings and implementations are not truncated.
The capability description explicitly limits each code candidate to one concrete skill ID.

After a run has terminated, `--resume-run <previous-output>` continues its completed candidate
history in a fresh output directory. Keep the evidence, baseline, model, repeats and replay
settings identical; `--patches` is the desired total candidate count, including imported ones.
A final prompt-limit stop before any proposal is retried at that index. Candidate seals are
verified, histories copied for audit, and imported calls/timing are excluded from new totals.
The consecutive-failure streak starts fresh on an explicit resume. A source report still in
progress is refused. This option is separate from `--retry-candidate`, which only evaluates
one previously generated candidate again.

### Experience reflection without optimization

Use `python -m cloth_agent.harness.experience_review --run <experiment-directory> --output <new-directory>`
to summarize saved experience rather than generate/evaluate faster candidates. Default backend is remote;
`--ssh-host`, `--model`, `--call-timeout` and `--prepare-only` are supported. One text-only Claude call
receives operation traces, reported judgments, actual implementations, and failure summaries. No image
upload, baseline/replay invocation, latency gate, candidate promotion or skill activation occurs.

Outputs: `context.json`, exact `prompt.txt`, `schema.json`, raw call artifacts, `experience.json`,
`experience.md`, and `report.json` with elapsed time and token accounting. Each lesson declares its
information need, method, success_check, on_insufficient, evidence.supported_by/failed_in/unresolved,
and limitations. Citation IDs must exist in the input. Evidence describes information acquisition,
not physical grasp outcomes; a saved READY result does not certify visual truth. This is a draft text
review, not an independent visual assessment or an automatically approved skill. The older optimization
entry point remains available explicitly via patch_evolution.
# Reasoning 经验实验模式

`--learning-mode experience --reasoning-record <reasoning_experience目录>` 在原有
diagnose → implement → materialize → gates → replay → feedback 链路中运行一次经验实验。
配合 `--reuse-baseline-run`、`--patches 1 --repeats 1` 跳过重复 baseline，并直接复用其预决策图片包。
新的 reasoning 记录与审核意见只进入诊断；历史 action 不进入候选执行输入。

该模式要求 reasoning-only HARNESS 补丁，保持图像处理配置不变，并实际绑定 Host Python 算子。
产物是可执行 harness 程序及参数绑定，复用已有 `reflect_point` / `affine_point` / `rank_candidates`
实现；不是新生成 Python 算子源文件。缺少运行时支持的计算不能仅靠文字声称已实现。
补丁经过原有本地检查后重跑；结果记录具体 Host 输入、输出及状态，时间和 token 仅用于统计。
不执行 latency promotion、不自动修改正式 skill。`TESTED` 表示至少一个 Host 算子实际计算，
不是视觉正确性认证；planning_statuses 仍单独记录 READY / NEEDS_LEARNING / ERROR。
