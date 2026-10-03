# 离线 Harness 提炼

## 当前默认：只总结图像处理

`compile` 和 `experiment` 默认使用 `--scope image-processing`。按 iteration 时间顺序扫描，只向 Claude 提供规划阶段的图像工具调用、当前输入视图及其派生图，以及上一轮总结和累计证据。选点答案、候选坐标表、执行评价、抓取结果、experience 和 skill 库不进入此模式的模型输入。没有新视觉调用的 iteration 留下 `NO_NEW_IMAGE_TRACE` 记录，沿用草稿，不制造新经验。

输出 `image_processing_summary.json` 与中文 `image_processing_summary.md`。每条规则包含：操作、适用条件、输入视图、处理方式、预期获得的视觉信息、停止条件、限制及证据引用；另列待验证问题。只允许 list_images / image_info / view_image / crop_image / rotate_image / resize_image / map_point / stop，不要求 Rxxx、return_decision 或模型复制固定 output_schema。

这是结构化总结，标记 `executable=false`，不能交给选点 executor 执行，也不会写回 approved.json。`experiment` 在此模式下总结完即结束，不执行候选点回放。历史工具使用记录不等于必要性证据，不能把裁剪/旋转或重复查看直接归为无用操作。

```bash
conda activate cali
python -m cloth_agent.harness experiment \
  --scope image-processing \
  --run-id fold_20260924T014204289928777Z \
  --output results/harness_image_processing_20261002 \
  --backend remote --ssh-host company-planner --timeout-s 300
```

每轮 `compiler_iterations/NNN/draft.json` 保存当前累计总结；全部完成后状态为 `SUMMARIZED`。中途失败不生成最终总结。图像预算仍按轮重置。CLI 结构化输出往返上限调整为四轮，允许格式修正，但探索工具继续禁用。

以下章节描述保留的 `--scope candidate-selection` 模式；需要完整选点策略和回放时才显式启用。

候选选点兼容模式研究「当前 observation + 当前折叠目标 → 当前 registry 中的 Rxxx」。它不接机器人，也不改原视觉规划循环、Z、动作规划、grounding、IK、preflight、执行、evaluator 或 experience 更新。

## 候选选点兼容模式的运行链路

1. `collector.collect_run` 复用 `run_storage.storage_roots`，查找 run metadata、跨段 summary、iteration record、process-review 副本；按真实时间排序、去重。没有时间时报告 chronology unavailable，不把文件名排序当作时间证据。
2. 同一 iteration 的 visual_planning 调用分别整理；height_retry 且没有新视觉调用是复用。相同 Rxxx 不等于复用。未知类型明确报告。
3. 为每个 decision 分别保存 pre_decision、post_decision 和完整 trace。编译器可看到二者，执行器 API 只接受前者。
4. `compiler.compile_policy` 按 iteration 的真实时间逐轮调用 `RuntimeClaude.invoke`。每次只发送当前 iteration 的 trace、前后图和反馈，加上上一轮策略草稿及累计证据摘要；图片预算每轮重新计数。同一轮的多次视觉调用一起处理。高度重试和中断轮也参与扫描，并明确标记它们不代表新选点或成功。缺失或相同时间戳导致顺序不明确时停止。适配器复用 `LocalClaudeBackend.invoke` / `RemoteClaudeBackend.invoke`、原结果解析和 token 统计。图片通过 `remote_output.multimodal_message` 编成真实 base64 image blocks，以 stream-json 送入 Claude；不以本地路径字符串替代图片。
5. Claude 生成 JSON policy 和分离的证据引用。Host 做 JSON Schema、变量引用、操作、预算与终止条件校验，每轮最多一次格式修复。每轮保存草稿，只有全部轮次成功后才冻结最后一份 policy；中途失败保留已有草稿和错误，不发布部分扫描结果。
6. `freeze_policy` 新建版本目录，保存规范化内容 SHA-256、policy、provenance、validation。CLI 重启时会先删除已有实验输出目录；一次运行内仍新建版本目录。冻结只表示满足执行合同，不表示可靠。
7. `execute_policy` 加载内存快照，逐步执行冻结 policy。Host 做图像变换，Claude 只做规定的视觉判断。无任意 Python/eval，无循环，无自动慢速 planner。
8. 回放结束后 `report.comparison` 才读取历史答案进行比较。

在 `--scope candidate-selection` 模式下，`compile` / `experiment` 还会读取当前项目的 `data/fold_experience/rules.json` 和有效技能库（包含内置技能及已批准补丁），保存为 `knowledge_snapshot.json` 后提供给编译器。这些是带条件、可能冲突的历史先验，不是新的成功证据；编译提示要求保留不确定性，只提炼视觉选点流程。来源快照的 hash 写入 compilation 和 policy provenance。输出仍是原有 `policy + evidence` 合同，不修改 `approved.json` 或经验库。回放不会读取这些来源文件。使用 `--trace-only` 可恢复只从 run trace 编译的模式。

## 数据要求与发现

默认扫描配置中的 run 根目录、本地 runs 和 results。SSD 未挂载时记录问题，同时继续查本地。可重复使用 `--search-root` 加入复制的原始 run 或整理目录。不会根据「两段、十五轮、九次选点」这些人工线索补记录或设定计数。

支持现有 `results/fold_exploration/<segment>/iteration_*/record.json`、planning_diagnostics.visual_plan_result、`claude_image_tools/visual_planning_*/`，以及含原 run 路径的 `*_record.json`。建议保留整个原始 run；单独 CSV、PPT、汇总图片无法证明选点时可用输入，因此不用于 replay。

旧日志中的 NaN / Infinity 在导入时转为 null，manifest issues 中保留源文件、字段路径和原值；不会改写原始日志。候选坐标中的缺失值仍会阻止回放，policy/schema 的严格有限数值校验保持启用。同一目录存在同一 iteration 的最终 `record.json` 时，`partial_record.json` 只记录为被最终记录替代，不重复计算；只有 partial 的中断记录仍会保留。

图像身份来自原调用的 image debug 快照及 hash，或 record 中的选点前输入。只允许明确识别的当前 clean/overlay、预先可用 reference/hint 进入 replay。历史派生 crop 仅是 compiler 学习证据。其他输入保留在 manifest，并不默认转发。post-action 和 previous-attempt 图片不会作为当前 clean/overlay。

Registry 必须由 iteration 的 camera_A_upright_mapping.json 明确关联到原 coordinate guide 和 raw image，raw 旋转后的像素 hash 必须与当前 clean 一致；记录的旋转约定和原图、显示图尺寸也必须一致。同一 observation 对应多个不同候选表时退出。原 host 会把不可变感知批次存放在 iteration 之外，因此允许该显式链接；不把可覆盖的 workspace/perception_views 当作历史证据。移动完整 run 或包含 summary 的 segment 时按相对后缀重新定位，不按 basename 猜测。缺图、缺表、表冲突、错误 observation ID 都明确退出。

Image debug 的 parent、to_parent、to_original 和公开工具结果组成 lineage。模型图像输入证据来自已验证的返回图片内容，包括编辑直接附图，不只看 view_image；只有路径、Read 已完成或缺少后续引用都不能证明已看过或没用过。未确认时为 UNKNOWN。只在 source 是原图、identity transform、同一 original index 且坐标未变时标记 identity-map no-op；显示变换不等于无用操作。

## Policy 语言

允许四种操作，自由排列为有限顺序和条件分支：

- inspect：指定 views、早先判断 context、固定语义 instruction、固定 output_schema；Claude 输出适用性、movable/preserved region、动态 ROI、是否需要 reference，以及 READY/CONTINUE/NEEDS_LEARNING。
- prepare_views：source 绑定当前 clean/overlay 对，roi_from 引用在这一对上产生的判断；Host 同时 crop、旋转、等比例 resize，保存像素中心 affine 链。ROI 为归一化边界，旋转为正交角度，scale 有限。
- return_decision：引用 READY 判断，必须属于当前 observation/registry，输出现有 selected_reference={camera, reference_id, reason} 结构及 registry 原图位置。
- needs_learning：明确停止原因，末尾必须有无条件 fallback。

内置图像变量为 observation（clean + overlay）、reference、hint。条件只能引用先前 inspection 的 status 或 needs_reference。可选参考缺失时退出，不自行搜索新图；条件不满足则跳过。

Host 不会通过「识别袖口」字符串实现确定性语义识别；这些判断由 Claude 完成。它只校验 schema、身份、引用、坐标与预算。policy 不能存在未来动作的历史 Rxxx、固定框或坐标。第一版对可执行 prose 更保守：禁用数字、路径和编码字段，几何只能来自当前判断；来源证据放独立 sidecar。自由语义 instruction 的泛化和可靠性仍未经证明，不能因此自动用于机器人。

调用使用 safe-mode，禁用工具、MCP、hooks、skills、Chrome 和 session persistence，避免自动读项目记录或 CLAUDE.md。保留用户 auth/model/provider 配置。CLI 不支持所需选项时失败并留档，不降级为自由工具调用。若 CLI 暴露 StructuredOutput 工具往返，单独计数；其他工具调用使该次调用失败。

## 命令

安装仓库依赖（新增 jsonschema>=4.18），在项目根目录运行。`--output` 指向专用实验目录；如果目录已存在，CLI 会先删除其中全部旧产物再开始，不再因目录存在而退出。需要保留旧结果时请使用不同目录。输入 policy、项目根目录及原始数据目录不能作为清空目标。

```bash
python -m cloth_agent.harness collect --run-id fold_20260924T014204289928777Z --output results/harness_collect_20260924

python -m cloth_agent.harness compile --scope candidate-selection --run-id fold_20260924T014204289928777Z --output results/harness_compile_20260924 --backend local --timeout-s 300

# 自动读取实际编译产物；编译失败时停止。
POLICY=$(python -c 'import json; x=json.load(open("results/harness_compile_20260924/compilation.json")); assert x["status"]=="FROZEN", x; print(x["policy_path"])')
python -m cloth_agent.harness replay --policy "$POLICY" --run-id fold_20260924T014204289928777Z --output results/harness_replay_20260924 --backend local --timeout-s 300

python -m cloth_agent.harness experiment --scope candidate-selection --run-id fold_20260924T014204289928777Z --output results/harness_experiment_20260924 --backend local --timeout-s 300
```

远端模式改为 `--backend remote --ssh-host company-planner`，仍使用现有 SSH/R2 配置。两种模式均可用 `--model` 显式指定模型。`--search-root /path/to/archive` 可在每条命令加入；路径应包含完整日志和原图，不能只放最终报告。

编译默认每个 iteration 最多 48 张去重图片，可通过 `--max-compile-images` 提到每轮 64 张；每次调用默认文本预算 400,000 字符，包含已有策略和累计证据。无需把全 run 图片一次输入模型。若单轮或累计文本超限，会明确停止，不默默截断。每次 policy 的调用数、图像数、Host 操作数和总时间预算由冻结内容确定，Host 再限制最大值。

可从 collect 的 `manifest.json` 中选择 decision ID，用可重复的 `--decision-id` 明确限定编译样本；所选和排除的 ID 都写入 `compilation.json`，policy provenance 保存训练 ID。已有经验库仍完整输入；`experiment` 的回放仍覆盖全部可回放决策，因此不能将未选入编译的 trace 自动视为完全独立的测试集。

现在默认扫描全部 iteration，无需指定三个样本。启动完整顺序提炼并回放（已有输出目录会清空重建）：

```bash
conda activate cali
python -m cloth_agent.harness experiment --scope candidate-selection \
  --run-id fold_20260924T014204289928777Z \
  --output results/harness_experiment_20261002 \
  --backend remote --ssh-host company-planner --timeout-s 300
```

只想提炼、不回放时，把 `experiment` 换成 `compile`。终端逐轮显示 `[harness compile n/total]`。`--timeout-s` 是每次模型调用的上限，不是整个扫描的上限。`--decision-id` 仍可显式选择子集，届时只扫描包含这些决策的 iteration。

## 产物与统计

- manifest.json、traces/<decision>/pre_decision.json、post_decision.json、trace.json：事实及缺失情况。
- compiler_input.json：扫描顺序和每轮输入文件索引。
- compiler_iterations/NNN/input.json、draft.json、attempt_*/：当前轮输入、累计策略草稿、实际图片和模型调用审计。草稿尚未冻结，不直接用于回放。
- compilation.json：是否调用后端、是否收到响应、状态、耗时、模型配置、policy 路径/hash。预检失败时 compilation_seconds=null，预检耗时单独保留。
- policies/<version>/policy.json、provenance.json、validation.json：只新增版本。
- replays/<decision>/calls/、views/、result.json：实际模型调用、Host 图像 lineage、READY/NEEDS_LEARNING/ERROR。
- report.json / report.md：计数、漏项、跳过原因、候选合法性、agreement、原图位置距离、调用数、Host 图像操作数、工具往返、视觉阶段时间与 coverage/fallback/skip。

旧视觉耗时只采用 visual-plan 调用时间或 image-debug 的 call boundary，不采用完整机器人 iteration。新耗时为整段视觉 policy（Host 变换 + Claude 调用与传输），无物理执行；二者是视觉阶段比较，具体边界均记录。旧边界无法确定为 unavailable/null。不将旧工具数与新 Claude 调用数相除充当加速。未回放 decision、异常和未知输入单独列出。NEEDS_LEARNING 可能是缺图、缺步骤、非法返回或预算，不自动等于 OOD。

在同一 run 上编译并回放只是 trace compression / 同源回放。agreement 不等于抓取成功，不同 candidate 不自动等于失败。没有真实模型重跑，不会报告实际模型加速。

## 验证

```bash
python -m pytest -q tests/test_harness.py
```

Fixture、mock 编译结果与 mock replay 都在测试文件明确标为合成测试，没有随产品分发预制 policy。

2026-10-02 最终代码验证：Harness 专项 41 passed；完整测试 1179 passed、21 failed、2 skipped。对未修改基线 ec0118b 的独立归档执行同一完整测试，1138 passed、21 failed、2 skipped，失败用例集合完全相同，没有新增失败。语法编译和 git diff --check 通过。测试日志与差异清单保存于下述实验目录的 current_pytest.log、baseline_pytest.log、verification.json。

2026-10-02 本机实跑指定 run：配置中的 /mnt/newssd 未挂载，本地无原 run 和 process-review 目录。最终复查输出 results/harness_experiments/fold_20260924_offline_20261002_verified/ 的阻塞报告，Claude 编译调用数为零、compilation_seconds=null；未获得 policy、agreement、真实 replay 耗时或加速结论。仓库已有汇报 CSV/PPT 不作为原始数据补造输入。inspection_attempts、实际后端调用数与收到模型响应数分别计数；找不到 CLI 不会被记为真实模型回放。

### 图像经验的累计更新

图像处理 compiler 每轮读取上一轮累计状态，只返回 `updates` 和当前轮 `evidence`，由宿主校验并应用增量，不允许整份覆盖已有经验。

- `SKIP`：内容相同，保留原规则，仅补充证据引用。
- `ADD`：新增独立经验；完全相同的规则即使换了 ID 也会去重。
- `MERGE`：兼容的新条件或细节合并到原规则，保留稳定 ID 和已有证据。不同适用条件应写成条件分支。
- `CONFLICT`：保留原规则和替代规则，记录原因与证据，标记 `PENDING`，不自动覆盖。
- `RESOLVE`：后续轮出现能区分双方的证据时才能提交裁决；宿主拒绝用冲突当轮证据裁决，也拒绝普通 MERGE 绕过未决冲突。证据是否足以裁决仍由 Claude 判断，尚无独立语义验证器。

`image_processing_state.json` 是每轮更新的累计状态；扫描结束输出 `image_processing_summary.json` 和 `.md`。`compiler_iterations/*/updates.json` 和 `draft.json` 仅用于审计、追溯，不是每轮新建的独立经验。扫描结束仍可保留未决冲突，不以多数票或最后一轮自动裁决。现有历史输出不会被自动改写，需要重新运行 compiler 才会采用此更新协议。

### 单轮信息获取速度测试（独立入口）

实验经验位于 `data/skills/experimental/visual_information.json`，从既有图像总结人工整理为 8 类“信息需求 → 有顺序的方法”。每项包含 `success_check`、`on_insufficient`、局限及三类方法证据。能肯定或否定地判断都算取得信息；看不清应为 UNKNOWN。原记录只有工具调用、不能证明获得视觉信息的来源保留在 `unresolved`，不以抓取成功作视觉方法证据。此文件不写入正式 approved skill 库。

```bash
python -m cloth_agent.harness.information_probe \
  --output results/visual_information_probe_20261002_v1
```

入口默认使用既有 manifest 最后一条 decision 的 clean、overlay、reference 根图，排除历史工具轨迹、选点答案、动作结果和语义 hint。可用 `--manifest` 和 `--decision-id` 指定另一 obs。`--prepare-only` 仅做本地输入检查，不调用远端。

同一个 obs 分别运行带 skill 与不带 skill 两组，默认 skill 先运行；`--order baseline-first` 可交换顺序。两组模型（`--model`）、信息任务、图像、系统提示、工具和预算一致。默认每组 360 秒、4 次编辑、12 个模型 turn；Claude 可实际使用查看、裁剪、旋转、缩放和映射工具。每组独立调用现有 remote backend；需使用该环境已配置的 R2 图像传输和 company-planner Claude 服务。

传给 Claude 的 skill 只含方法及适用条件，不包含历史证据引用或过去的场景结论。完整证据保存在本地 `skill_snapshot.json`。输出：

- `obs/observation.json` 与 PNG：两组相同输入。
- `with_skill/`、`without_skill/`：各组输入、实际工具 trace、JSON 信息判断及计时。
- `report.json`：总耗时、上传/远端阶段耗时、工具及编辑次数、模型 token 用量、SATISFIED/UNKNOWN 数量和原始耗时比。

这是信息获取试验，不测总结 compiler 速度，也不执行机器人。信息是否完成由模型自报，需人工核对图像；UNKNOWN 并非错误答案。单次配对不证明稳定加速，且第二组可能受缓存影响；后续应交换顺序重复，并使用未参与经验提炼的 obs 测试泛化。已有目录拒绝覆盖，使用新目录保留原始测量。

### 固定 skill 流程测试：一次绑定 → 连续执行 → 一次验收

上一入口仍允许 Claude 每次看完工具结果后再选择下一操作。要检验减少这种来回决策，使用：

```bash
python -m cloth_agent.harness.direct_information_probe \
  --output results/visual_information_direct_20261002_v1
```

此入口只支持实验 skill 中明确声明的 `execution_recipe`，不生成新流程。第一次 Claude 调用直接接收四张根图，只输出当前 clean 的归一化 ROI、是否需要成对局部比较、是否需要放大。宿主按固定配方连续执行条件裁剪和放大，保存各步实际图片与坐标血缘；中途没有模型调用。放大尺度由视图尺寸与配方显示上限决定，裁剪坐标由当前 obs 绑定，不复制历史坐标。第二次调用接收原图及全部执行结果，按相同四项信息任务验收；不足则 UNKNOWN，不启动自动重规划或重复放大。

记录 `model_invocations`（上限两次）、`host_image_ops`、`host_execution_s`、`exploratory_tool_round_trips`（零）和总耗时。这里两次调用分别用于视觉参数绑定和结果判断；不能称为模型没有内部思考，StructuredOutput 格式处理也可能发生额外 CLI turn。它与自由工具调用测试的交互架构不同，耗时差不能全部归因于 skill 内容。

输出 `binding.json`、`execution/execution.json`、执行图片、`result.json`、`report.json`，保留输入及 skill 快照。固定配方是人工整理的实验流程，不是已验证的通用图像策略。

### 接入原视觉推理的 A/B 测试（无独立验收）

以下描述的是首批实验的旧接法，现需显式指定 `--skill-flow legacy-preprocess`。默认入口已经切换到下文的信息驱动流程，旧实验记录不改写。

```bash
python -m cloth_agent.harness.planning_probe \
  --skill-flow legacy-preprocess \
  --output results/planning_skill_ab_20261002_v1
```

该入口在保存的 `RemoteFoldClient` 视觉规划接口边界做离线对照，复用原请求的 prompt、system prompt、输出 schema 以及当时的 context 文件。两组共享全部原始任务/历史上下文，不读取本轮执行结果或历史最终选点答案。当前支持只有 clean/overlay/reference 根图的保存请求，其他根图角色会明确拒绝，避免默默改变“原流程”的输入。默认原组先运行，可用 `--order skill-first` 交换。

- `original`：原视觉推理，保留查看、裁剪、旋转、放大等图像工具，6 次编辑预算。
- `skill_then_reasoning`：一次任务相关的 ROI/条件绑定，宿主按固定 skill 连续执行图像处理，然后立即进入同一个原视觉推理接口。没有独立的信息验收调用；推理阶段保留查看图片、读取原上下文的工具，编辑预算为零。原图始终可用，派生图随附坐标血缘。

预处理通过 `prepare_skill_views` 接口复用；其参数绑定接收当前折叠目标和参考状态，不能固定选择前次的左端/标签/缝线。实际的 Rxxx 选择由最后的推理阶段完成。

输出与原视觉推理相同：衣物观察、所选 Camera/Rxxx 及理由、策略、运动意图、预期观察与约束。宿主使用现有 `validate_visual_plan_payload`，并检查所选点是否属于本 obs 的绑定 registry、是否已被拒绝。这里只比较视觉推理，不进入 metric grounding、IK 或物理执行；点合法不代表语义正确或抓取成功。

`report.json` 包含预处理、原推理、端到端耗时、模型调用次数、工具统计和最终选点；`index.html` 展示两组结果与实际图片。两组 context 和 obs 的 hash 可用于检查共同输入；单次结果不证明统计显著的加速。没有对“读取上下文/观看图片”这些原有模型轮次做额外优化，测试的变化仅为图像预处理流程与编辑工具的关闭。

要检验连续图像处理的收益，应在调用前选定历史确实使用多步图像工具的 obs，并保留本次原组的自然行为。`operation_sequence.json` 按 `tool_use_id` 去重，保存实际工具顺序及其所属模型响应组；不要把事件日志中的 started/completed 等重复记录当作多次调用，也不要把响应组数称为内部思考次数。

`processing_coverage.qualified_multi_step_comparison` 只有在两组均完成、原组本次至少进行两次编辑且分属至少两个响应组、实验组至少执行两次宿主编辑时才为真。历史使用过多步操作不保证本次也会触发。未触发或失败的试验同样保留，不作为完整多步加速的成功证据。当前宿主配方只有裁剪和放大；若原组使用旋转，应注明两种策略的操作能力不同，不能将耗时差全部归因于取消工具间决策。

完整对照的首批两例保存在 `results/planning_full_ab_20261002_v1/`：`left_side` 原组先运行、`right_side` 实验组先运行；每组总预算 600 秒、规划预算 16 turns。`suite.json` 记录预先冻结的选例及结果，`index.html` 汇总实际指标并链接各例图片与推理。跨例交换顺序只能部分减轻顺序影响；这两例参与过经验来源，不是独立泛化集。离线重建展示可运行该目录的 `build_report.py`，不会重复调用模型。

### 信息驱动推理：全图理解 → 观察 skill → 选点（当前默认）

```bash
python -m cloth_agent.harness.planning_probe \
  --skill-flow information \
  --max-supplements 1 \
  --output results/planning_information_flow_v1
```

这是离线视觉推理对照入口的新版实验组；原组保持原请求。`information_flow.py` 替换旧实验组的“额外绑定调用 + 原完整规划器”，不在末尾再次调用旧规划器。当前任务、原规划约束、候选 registry 和决策前上下文保留并内联传入；不额外 Read 文件。实验组改变了阶段划分、图片交付及观察策略，因此是整条流程的对照，不是仅改变 skill 文本的单变量实验。

1. **全图理解一次完成四件事**：理解当前目标和所有根图、记录选点所需的已知信息与具体缺口、按缺口选择已有观察 skill、绑定当前图参数。不单独调用参数绑定模型，也不在此阶段提前选点。
2. **Host 执行选中的方法**：`orientation` 同角度旋转 clean/overlay；`local_boundary` 裁剪 clean 并按需放大；`overlay_occlusion` 对齐裁剪 clean/overlay 并按需放大。全图布局、图片清单与参考比较在第一阶段利用现有输入完成；坐标血缘由宿主管理。ROI 始终指向当前 clean 根图，旋转不会改变 ROI 参数的坐标系。每批操作预先检查，最多 12 次编辑；不允许调用任意代码。没有缺口就不加工图像。
3. **选点直接接收信息包**：原图、每个观察方法的最终图片（不重复发送其内部裁剪/放大中间图）、当前目标、信息状态、原图映射、执行记录和对应 `success_check/on_insufficient` 一起作为实际多模态内容输入。模型无需逐张调用 view_image。选点阶段在使用结果时更新信息状态，没有独立的验收调用。
4. **信息不足有明确出口**：`CANDIDATE` 只允许所有选点所需项为 KNOWN、附带合法的原视觉规划结果；`NEED_MORE` 必须列出具体 UNKNOWN 和预期增益，绑定能补足该缺口的新观察；`UNKNOWN` 不输出候选。默认至多补充一轮（全流程最多三次模型调用），`--max-supplements 0` 禁止补充。重复相同观察、操作预算耗尽或无法通过现有方法解决时明确退出 UNKNOWN。工具执行失败、模型请求失败或输出违反协议记录为 FAILED。

信息项格式：`id / need / status / finding / missing_information / source_image_ids`。后续阶段不能悄悄删掉前一轮的信息需求；KNOWN 必须有图像引用和结论，UNKNOWN 必须说明缺什么。肯定和否定判断都能完成信息任务；工具执行成功本身不代表缺口已经解决。已有结论是带来源的模型观察，遇到冲突可以修正，不当作独立验证的真值。观察请求必须针对当前 UNKNOWN，重复检测根据实际像素区域、来源、变换和尺度，避免换个 gap 名字重复做相同操作。

新输出包括 `global_bundle.json`、`global_result.json`、`observations/execution.json`、`selection_bundle_*.json`、`decision_*.json`、`information_state.json`。仅输出合法候选时生成 `result.json`。report 分开记录全图理解、宿主操作、选点、补充轮数、总耗时及模型响应组；历史表格中的 preprocessing 列在此模式代表“全图理解 + 宿主执行”，并非一个额外绑定阶段。两组都结束但有信息不足时，整组状态为 `COMPLETED_WITH_UNKNOWN`，不把主动退出冒充成功选点。

新流程取消了说明字段的 300 字符限制，提示模型简短输出；这去掉了上一实验的具体超长重试触发条件，但并不保证服务永远没有其他格式重试，也不保证内部推理量下降。是否加速须重新调用真实 Claude 测量。本地协议测试不等于远端速度测试或机器人成功率测试。

### 仅比较观察方法对后续推理的影响（精度对照入口）

```bash
python -m cloth_agent.harness.observation_ab \
  --decision-id fd15207c944ff34b5e17eb41 \
  --output results/observation_method_ab_20261002_v1
```

本入口按用户澄清后的实验边界设计，不使用 `run_information_flow()` 的新选点状态机。A 组由 Claude 自主调用观察工具并选择有用的最终派生视图；B 组根据现有视觉 skill 绑定观察请求，由 Host 连续执行。两组观察阶段输入相同的原图、任务和决策前上下文，均有 6 次编辑预算、360 秒观察预算。

观察后都进入同一个 `reason_common()`：相同提示模板、原始规划任务及上下文、模型、原输出 schema、校验器、360 秒推理预算，均直接附带图片，均关闭后续工具。该阶段只允许图片及其必要几何元数据不同。skill 文本、观察阶段的场景结论、候选建议、缺口文字或预期增益不传给后续推理。两组始终包含原始根图；派生视图统一编号并携带 root ID、尺寸、角色和 `to_original`。原组并非生产中“观察与规划混在一个调用”的完整复刻，而是隔离出的自主观察对照组。

`reasoning_contract.json` 保存不含图片目录的共同推理合同；两组 hash 必须一致。`reasoning_bundle.json` 的共同上下文也一致，差异只在图片目录。原始模型输出先保存在 `raw_reasoning_result.json`，只有 schema/当前候选绑定验证通过才生成 `result.json`。失败不自动重跑，不把无候选的提前退出当作加速。

质量检查标准在调用前写入 `evaluation_rubric.json`，复核结构/朝向、任务侧与方向、选点的可见支持、坐标来源及不确定性，不以另一组答案作为真值。单例未独立标注的图像复核不能称为统计准确率或抓取成功率。默认 A 后 B，每组各一次；先比较推理结果，再辅助报告分阶段耗时。该实验同时比较了自主操作与 skill/Host 操作方法，不能进一步把差异全部归因于某一条 skill 文本。

### 原始 planner 单次计时

```bash
python -m cloth_agent.harness.planner_profile \
  --decision-id fd15207c944ff34b5e17eb41 \
  --output results/original_planner_profile_20261003_new
```

此入口重放保存的原始视觉规划请求，复用 prompt、system prompt、schema、按需读取的 context、图片和原始工具预算；不使用 A/B 的观察拆分或 skill 预处理，不启动机器人。输出目录必须不存在。测试需要访问项目配置的 R2 和远端 Claude。

可选 `RemoteClaudeBackend(record_event_timing=True)` 在远端 Claude 输出每一行时用单调时钟记录时间，完成后仍按原有方式校验并回传完整日志。默认运行行为不变，终止结果保持原样。`_cloth_timing.elapsed_s` 是远端 CLI 事件读取时间；旧 `received_elapsed_s` 是本地收到回传的时间，两者不能混用。

`profile.json` 和 `profile.md` 记录加载输入、后端调用、结果校验、传输阶段、每轮响应等待/输出窗口、工具提交与返回，以及结构化输出错误。输出时间只说明事件边界，不能精确划分模型内部“判断朝向”“比较候选”等隐藏语义过程，也不能把首事件前等待全部当成模型思考。SSH、Claude、工具区间有包含或重叠关系，不可直接求和。
