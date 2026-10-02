# 离线 Harness Policy 实验

本模块只研究「当前 observation + 当前折叠目标 → 当前 registry 中的 Rxxx」。它不接机器人，也不改原视觉规划循环、Z、动作规划、grounding、IK、preflight、执行、evaluator 或 experience 更新。

## 运行链路

1. `collector.collect_run` 复用 `run_storage.storage_roots`，查找 run metadata、跨段 summary、iteration record、process-review 副本；按真实时间排序、去重。没有时间时报告 chronology unavailable，不把文件名排序当作时间证据。
2. 同一 iteration 的 visual_planning 调用分别整理；height_retry 且没有新视觉调用是复用。相同 Rxxx 不等于复用。未知类型明确报告。
3. 为每个 decision 分别保存 pre_decision、post_decision 和完整 trace。编译器可看到二者，执行器 API 只接受前者。
4. `compiler.compile_policy` 调用 `RuntimeClaude.invoke`。适配器复用 `LocalClaudeBackend.invoke` / `RemoteClaudeBackend.invoke`、原结果解析和 token 统计。图片通过 `remote_output.multimodal_message` 编成真实 base64 image blocks，以 stream-json 送入 Claude；不以本地路径字符串替代图片。
5. Claude 生成 JSON policy 和分离的证据引用。Host 做 JSON Schema、变量引用、操作、预算与终止条件校验，最多一次格式修复；失败不写入成功 policy。
6. `freeze_policy` 新建版本目录，保存规范化内容 SHA-256、policy、provenance、validation。已有实验目录不可覆盖。冻结只表示满足执行合同，不表示可靠。
7. `execute_policy` 加载内存快照，逐步执行冻结 policy。Host 做图像变换，Claude 只做规定的视觉判断。无任意 Python/eval，无循环，无自动慢速 planner。
8. 回放结束后 `report.comparison` 才读取历史答案进行比较。

## 数据要求与发现

默认扫描配置中的 run 根目录、本地 runs 和 results。SSD 未挂载时记录问题，同时继续查本地。可重复使用 `--search-root` 加入复制的原始 run 或整理目录。不会根据「两段、十五轮、九次选点」这些人工线索补记录或设定计数。

支持现有 `results/fold_exploration/<segment>/iteration_*/record.json`、planning_diagnostics.visual_plan_result、`claude_image_tools/visual_planning_*/`，以及含原 run 路径的 `*_record.json`。建议保留整个原始 run；单独 CSV、PPT、汇总图片无法证明选点时可用输入，因此不用于 replay。

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

安装仓库依赖（新增 jsonschema>=4.18），在项目根目录运行。每条命令使用新的输出目录。

```bash
python -m cloth_agent.harness collect --run-id fold_20260924T014204289928777Z --output results/harness_collect_20260924

python -m cloth_agent.harness compile --run-id fold_20260924T014204289928777Z --output results/harness_compile_20260924 --backend local --timeout-s 300

# 自动读取实际编译产物；编译失败时停止。
POLICY=$(python -c 'import json; x=json.load(open("results/harness_compile_20260924/compilation.json")); assert x["status"]=="FROZEN", x; print(x["policy_path"])')
python -m cloth_agent.harness replay --policy "$POLICY" --run-id fold_20260924T014204289928777Z --output results/harness_replay_20260924 --backend local --timeout-s 300

python -m cloth_agent.harness experiment --run-id fold_20260924T014204289928777Z --output results/harness_experiment_20260924 --backend local --timeout-s 300
```

远端模式改为 `--backend remote --ssh-host company-planner`，仍使用现有 SSH/R2 配置。两种模式均可用 `--model` 显式指定模型。`--search-root /path/to/archive` 可在每条命令加入；路径应包含完整日志和原图，不能只放最终报告。

编译默认最多 48 张去重图片，可通过 `--max-compile-images` 提到 64；默认文本预算 400,000 字符。超过预算在调用前报告阻塞，不默默删除成功或失败 trace。更大规模数据需要明确划分实验数据集，当前不实现递归摘要编译。每次 policy 的调用数、图像数、Host 操作数和总时间预算由冻结内容确定，Host 再限制最大值。

## 产物与统计

- manifest.json、traces/<decision>/pre_decision.json、post_decision.json、trace.json：事实及缺失情况。
- compiler_input.json、compiler_calls/attempt_*/：图片、实际 input stream、命令、stdout/stderr、调用审计、模型返回与校验结果。
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
