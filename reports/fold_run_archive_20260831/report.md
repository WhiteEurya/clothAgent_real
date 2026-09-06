# 折叠衣服实验归档报告

统计窗口：2026-08-24 00:00（Asia/Shanghai）至 2026-08-31。

## 总览

- 顶层 run：**84**
- summary 记录：**288**
- 有 evaluation 的迭代：**97**
- 无人值守重复尝试不是独立样本。

Summary 状态分布：

- `BAD`: 1
- `BLOCKED`: 2
- `BLOCKED_SELECTION_OR_MASK_DISAGREEMENT`: 1
- `COMPLETED`: 10
- `FAILED`: 251
- `INTERRUPTED`: 6
- `MAX_FOLDS_REACHED`: 2
- `PHYSICAL_PROBE_COMPLETED`: 1
- `RUNNING`: 14

Evaluation 状态分布：

- `grasp_acquisition=FAILURE`: 83
- `grasp_acquisition=SUCCESS`: 5
- `grasp_acquisition=UNKNOWN`: 7
- `laydown=FAILURE`: 4
- `laydown=NOT_REACHED`: 87
- `laydown=SUCCESS`: 1
- `laydown=UNKNOWN`: 3
- `target_structure_acquired=CONTRADICTED`: 88
- `target_structure_acquired=SUPPORTED`: 1
- `target_structure_acquired=UNKNOWN`: 6
- `task_progress=NEUTRAL`: 92
- `task_progress=REGRESSED`: 3
- `transport=INSUFFICIENT`: 3
- `transport=UNKNOWN`: 92

## 结构性问题

结构性问题是跨 run 重复出现、需要修改 perception、pipeline、控制约束或观测架构的问题。

### S1 · 袖子 acquisition 反复空抓

97 个有 evaluation 的迭代中有 83 个 acquisition failure。代表性 Cam-C hold-check 显示夹爪抬起后没有悬挂布料，遥测 position_pulse=0、mechanical_grasp_detected=false；这说明抓取接触/边缘进入方式仍是首要瓶颈。

证据：fold_night iteration 014 的 Camera-A before/after、Camera-C hold-check、Camera-C/AB 视频、evaluation、gripper telemetry 和 debug.log。

视频证据：已归档 2 个 32 倍速视频。

归档目录：`structural/S1_acquisition_empty_grasp`

### S2 · 抓到的是整团布/滚卷而非目标层

存在少数机械上确实抓到布料的迭代，但目标层未被分离：整件衣服被拖动，释放后变成更窄、更高的 sausage/roll。对应 evaluation 中 acquisition=SUCCESS、target_structure=CONTRADICTED、laydown=FAILURE 或 task_progress=REGRESSED。

证据：fold_inherit iteration 001 的 A/B before/after、AB depth 视频、evaluation 和 trajectory。

视频证据：已归档 2 个 32 倍速视频。

归档目录：`structural/S2_whole_bundle_drag_roll`

### S3 · RGB→Rxxx→XYZ grounding / mask 不一致

出现过 selected Rxxx 与实际 grasp XY 相差约 320 mm、选点落在 production mask 外、以及目标虽叫 image-left sleeve 却解析到肩部/衣身的情况。另有 bbox bottom=718/720 被统一判为 PARTIAL。错误发生在视觉点到执行坐标的链路，不能归咎于 Claude 的折叠策略。

证据：single_sleeve 的 selected_point_vs_production_mask.png；fold_holdcheck 的 320 mm 日志；fold_night iteration 009 的 Camera-A/C 图像、evaluation 和视频。

视频证据：已归档 2 个 32 倍速视频。

归档目录：`structural/S3_grounding_mask_mismatch`

### S4 · Claude planning / grounding 上下文导致极端延迟

76 次 planning 返回记录的 visual planning 平均约 193 s、最大 1304 s；final grounding 平均约 189 s、最大 392 s。持久 session 还发生多次 context rollover，随后出现 Prompt is too long / Argument list too long。瓶颈在提示词、结构化输出重试和上下文管理，而非相机或机械臂移动。

证据：fold_persistent debug.log、summary、persistent_claude_session.json、timing_summary.json，以及同一迭代的 32x AB 视频。

视频证据：已归档 1 个 32 倍速视频。

归档目录：`structural/S4_latency_context`

### S5 · 安全高度 / 工作空间 / controller IK 约束互相冲突

多次出现 no legal engaged grasp Z（surface_z 低于 table/robot 下限）、x 低于 safe lower bound、controller IK code=10。说明桌面/海绵/衣物高度模型和最终 controller 可达域没有形成稳定、可执行的统一约束。

证据：collar_high_lift_05 的高度失败 summary 与 collar overlay；fold_holdcheck_004839 的 IK debug/summary；single_sleeve 的安全边界 summary。

视频证据：无（执行前失败，未产生 rollout 视频）。

归档目录：`structural/S5_safety_ik_height`

### S6 · 抓取成功判据受夹爪遮挡且 telemetry 仍是弱信号

Camera-A 在夹爪靠近时遮挡了大部分目标；Camera-C 只能作为未标定 RGB observer。虽然加入了 hold-check，但多次 telemetry 样本标记 simulated=true、时间戳重复、mechanical_grasp_detected=false，因此只能作为弱机械证据，不能独立证明目标层被抓住。

证据：fold_camC iteration 001 与 fold_night iteration 014 的 Camera-C 图像/视频、hold-check、gripper_telemetry.json。

视频证据：已归档 2 个 32 倍速视频。

归档目录：`structural/S6_observability_telemetry`

### S7 · 无人值守状态机在 bunched/left_sleeve 上反复横跳

fold_night run 曾累计 194 次 unattended attempt；在衣服已 BUNCHED 时仍反复停留在 left_sleeve，repair 或 planning failure 后状态没有真正推进到可折叠状态。该问题会消耗整夜时间，却不产生新的有效物理证据。

证据：fold_night_20260830T100630 的 summary、debug.log、unattended restart 记录和 iteration 009 视频；另有长 run 的 summary 记录。

视频证据：已归档 1 个 32 倍速视频。

归档目录：`structural/S7_unattended_state_loop`

### S8 · 相机/光照质量门槛使 perception 整批阻塞

night run 大量出现 table luma 约 2–6（阈值 90）导致 before perception 连续失败；另有 mask silhouette coverage 仅 0.056。此类失败发生在 Claude/机器人动作之前，是采集质量和验证门槛的问题。

证据：fold_night 多个 failed summary/debug.log；neat_fold_overnight 的 perception failure 图像与视频作为同环境对照。

视频证据：已归档 1 个 32 倍速视频。

归档目录：`structural/S8_capture_quality`

## Claude planning 能力不足（原“非结构性问题”）

这里的非结构性问题专指 Claude 规划本身暴露出的算法缺口，需要通过候选排序、因果归因、探索策略和输出自检提高成功率；网络、SDK 和人工中断不计入此类。

### P1 · 目标点语义选择不稳定

Claude 有时把 sleeve 目标选到肩部、衣身或 Rxxx 内部点，即使 RGB 中袖子可见。需要部件拓扑、左右语义、外轮廓距离和反事实检查。

证据：S3 的 mask 对照、320 mm grounding 错位日志和 iteration 009 视频。

视频证据：已归档 2 个 32 倍速视频。

归档目录：`structural/S3_grounding_mask_mismatch`

### P2 · 没有形成边缘优先抓取策略

连续空抓后，Claude 仍主要改变高度、yaw 或小范围位移，没有稳定切换到 sleeve hem/free-boundary、edge-straddle 或 opposition。

证据：S1 的 Camera-C hold-check、telemetry 和 32 倍速 rollout。

视频证据：已归档 2 个 32 倍速视频。

归档目录：`structural/S1_acquisition_empty_grasp`

### P3 · 运输与落放规划不足

有些迭代确实抓到布，但计划是零横向位移、回到原点，或把整团布拖成更窄更高的 roll。需要显式验证 source→destination 位移和落放后的轮廓扩展。

证据：S2 的 before/after、trajectory 和 AB 视频。

视频证据：已归档 2 个 32 倍速视频。

归档目录：`structural/S2_whole_bundle_drag_roll`

### P4 · 失败归因和探索多样性不足

多次失败后仍重复相近假设，并在 BUNCHED/left_sleeve 状态循环。需要维护已证伪假设集合，强制下一轮改变因果维度。

证据：S7 summary/debug.log、iteration 009 视频和保留的 experience ledger。

视频证据：已归档 1 个 32 倍速视频。

归档目录：`structural/S7_unattended_state_loop`

### P5 · 计划输出契约不稳定

Claude 偶发缺少 safety_notes、close_gripper、move target 或返回非法 skill invocation。需要更短的 schema 提示、规划前自检和局部重试。

证据：P5 目录中的 schema/contract rejection 日志。

视频证据：无（执行前失败，未产生 rollout 视频）。

归档目录：`planning/P5_plan_contract`

## 运行性附录

### O1 · 网络、相机和 SDK 瞬时故障

connect socket failed、API ENOTFOUND、SDK TypeError 属于运行基础设施问题，不计入 Claude planning 能力评估。

证据：O1 目录中的 socket、DNS 和 TypeError 日志。

视频证据：无（执行前失败，未产生 rollout 视频）。

归档目录：`operational/O1_infra_transient`

### O2 · 人工中断和旧版运行上限

INTERRUPTED、MAX_FOLDS_REACHED 和 KeyboardInterrupt 主要反映旧运行方式或人工停止。

证据：O2 目录中的 interrupted/max-fold summary。

视频证据：无（执行前失败，未产生 rollout 视频）。

归档目录：`operational/O2_interruptions_limits`

## 时间开销

- `visual_planning_s`: n=76, mean=193.4 s, median=159.0 s, max=1303.9 s
- `final_grounding_s`: n=76, mean=189.1 s, median=187.2 s, max=391.8 s
- `total_planning_s`: n=76, mean=382.6 s, median=353.2 s, max=1418.1 s
- `supervisor_s`: n=72, mean=105.2 s, median=86.2 s, max=345.5 s
- `perception_s`: n=183, mean=33.2 s, median=36.9 s, max=41.4 s
- `molmo_s`: n=54, mean=24.8 s, median=17.1 s, max=46.6 s

视觉 planning 和 final grounding 的耗时主要来自上下文增长、reselection/retry 和结构化输出重试，不是机械臂运动耗时。

## 归档说明

- 已复制并校验代表性图像、JSON、日志和视频证据。
- 视频证据为 32 倍速版本。
- 选中的源证据已删除；experience ledger 和未选中的原始数据保留。
