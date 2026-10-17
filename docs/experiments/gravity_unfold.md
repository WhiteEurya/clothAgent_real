# Gravity-aware unfolding seed v1

独立实验入口：`cloth_agent.experiments.gravity_unfold`。原 unfolding/folding loop、抓空重试、IK、安全和 collision 实现均不改动。不训练、不建立最终 policy graph、不定义 unfolding node taxonomy。

## 保留的 v0

已冻结 `/mnt/newssd/sja/clothAgent_real/runs/codex_astra/unfold_arm7_20261011T084945269513Z` 的 12 次 physical action：

`/mnt/newssd/sja/clothAgent_real/runs/gravity_unfold_experience_v0/experience_graph_v0.json`

该运行有完整回顾报告，末帧显著展开，但原记录没有最终完成判定；报告也明确标为 `NO_FINAL_REPORT`，不能当成自动认证的 FOLD_READY。更晚的 09:44、09:57 两次运行均为 MODEL_FAILED，末图仍明显堆叠。检索没有找到记录为 MODEL_REPORTED_COMPLETE/FOLD_READY 的旧 unfold。

v0 是已有 experience 的冻结档案，不是新推导的 policy graph。日志、RGB、点云等复制到独立 evidence 目录，所有原始文件保留 SHA256；视频留在原处并校验哈希。原目录不改、不删除。不会用 EXECUTED 代替视觉 progress。未评估指标用 null，不能拿缺失值当零。

## 四个 primitive

- LiftAndReveal：观测原始 grasp → 严格竖直 lift → 保留夹持，等待 1.5 秒 → 原位 RGB-D。lift 不旋转；当前仍完整验证宿主 pick/place/retreat 计划，但只派发到最终 release 之前的已验证前缀。
- LiftDrapeRelease：根据悬垂新图确认 extremity/structure → 保持 grasp → 竖直 lift → 附近可见空闲区域上方 → 下降 → release。横向总路径 <=300 mm。
- PeelOverlap：独立 overlap edge → 小幅竖直 lift（<=60 mm）→ 横向分离（总路径<=150 mm）→ 下降、release。
- ShortDragReveal：没有可靠可抬边缘、也没有可 peel 的 overlap 时 fallback；单段长度 <=50 mm，高度变化 <=15 mm。

所有 primitive 使用原 Arm7Runtime 的验证及执行器；实验 adapter 按用户已验证 130% 的说明，在构造规划器前将 arm7 的运行时速度上限设为 6.5°/s，并在原生分段合并后仅为 arm7 重新计算命令速度及超时；双臂共用默认速度保持 5°/s，避免对静止 arm6 提出更高的速度上限要求；原加速度 10°/s² 保持不变。Home/perception 也使用此运行时速度。只更新本实验 arm7 的速度上限，arm6 与全局配置文件不改动。停止余量由旧值按速度比平方外推到 2.1125°，该值是估算而非新实测；原停止边界检查逻辑保留。每步 speed_policy.json 记录用户确认来源及外推来源。ShortDragReveal 仍保持原基准的 35%（1.75°/s），按比例延长超时。持布 move_tcp 保持 Cartesian 路径，不使用关节路径绕行；现有安全边界不变。额外抓取偏移为 0 mm，直接使用桌面深度补丁修正后的观测高度，不调用加深重试。

Camera B 在手腕。LiftAndReveal 后拍照不执行常规开爪/Home/perception 准备，保留抓取。悬垂 RGB-D 使用独立的 `_hanging_unpatched` 标定身份，不应用仅限 perception 位姿的桌面高度补丁；保留原补丁的位姿校验。该观测只允许已有抓取的 drape continuation，不允许新抓取；放置位置与桌面高度读取上一桌面观测的已修正点云。视角可能变化或不够完整，评估必须允许 UNKNOWN。持布时如果无法确认 hanging structure，返回 BLOCKED 跳过本轮动作，并原位重新观测；不自动释放或盲目搬动。与原 unfold 一样，BLOCKED 不表示任务结束；连续受阻仍受轮数上限约束。异常/动作上限时可能仍持布，日志写 holding_at_stop，不擅自执行释放恢复动作。

## GPT hierarchy 和 progress

每个决策返回 primitive、reason、grasp、targets、五项物理 evidence 和四项 fold_ready 条件。宿主校验 hierarchy 和几何；有 exposed extremity + safe lift 优先 LiftAndReveal；已持布且新图确认 hanging structure 优先 LiftDrapeRelease；否则 overlap 用 PeelOverlap；最后才 ShortDragReveal。

沿用原 unfold 的观测 → 同一 GPT 会话决策 → 执行节奏，每轮只调用一次 GPT，不再单独调用动作后 evaluator。下一轮决策同时返回 previous_progress，顺带记录上一动作的六个 progress 字段、视觉 reason 和四项 fold_ready 条件：

- 新 landmark/boundary：STRUCTURE_REVEALED。
- 可见面积/重叠/结构改善：PROGRESS。
- 仅整体平移或轻微 wrinkle：NO_PROGRESS；宿主将 translation-only 的成功标签降为 NO_PROGRESS。
- 明显变糟：REGRESSION。
- 视角不可比、遮挡、缺证据：UNKNOWN。

NO_PROGRESS 保存到 failed_evidence，绑定实际 observation 路径、动作 family、命令及前后 RGB；下一决策携带全部 negative evidence。直接重复上一无效 family 在执行前被拒绝。没有自动“加深/拖远”重试。

torso_open、most_landmarks_locatable、no_severe_self_occlusion、folding_graph_compatible 全为 true，且已 release，下一轮决策返回 FOLD_READY 时停止。无需 perfect flattening，不进入 folding 执行。

动作上限时仍保存末帧，但不追加 GPT 调用，最后一次效果暂记 UNKNOWN。控制器失败后只尝试不运动的 stationary capture；不追加模型分析，未评估则 UNKNOWN 并保留原因。规划失败不计 physical action。partial execution 计一次 physical primitive attempt；Home、观测准备和底层 waypoint 不另计。LiftAndReveal 与后续 LiftDrapeRelease 分别计一次。

## 运行命令

在仓库根目录使用原 `cali` 环境。以下是未来运行命令，本次实现没有启动：

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.gravity_unfold run \
  --baseline /mnt/newssd/sja/clothAgent_real/runs/gravity_unfold_experience_v0/experience_graph_v0.json \
  --output /mnt/newssd/sja/clothAgent_real/runs/gravity_unfold_v1 \
  --max-actions 30 \
  --execute
```

运行先校验 v0 哈希，然后直接沿用原 unfold 的开爪 → Home → perception → RGB-D 观测流程。仅在实机 rollout 达到 FOLD_READY、已释放衣物且机器人连接关闭后，若 v0 尚未视觉评估，再用同一 evaluator 离线评估，写独立 `gravity_unfold_v1_v0_review` 目录，保留原档案。未完成、失败、中断或仍持布时不启动对比，记录 baseline_review_status=DEFERRED。离线评估失败只记录 baseline_review_status=PENDING，不改变新 rollout 的结果、不丢失记录。离线 review 中断后重跑会核对已有证据的哈希，并复用相同 prompt 的有效完成响应；失败请求另建 retry 目录，不删除旧记录。旧动作重力辅助属性由命令和图像证据另行标注；不确定为 null。缺失 after 或 UNKNOWN 导致无法准确比较的指标继续为 null，并报告 unknown 数量。到达 FOLD_READY 所需动作数取旧图首次满足条件的位置；未确认到达则 null。

与原 unfold 一致，默认在生成 Viser 预览后自动执行，不需要点击确认；`--auto-execute` 可显式选择自动模式，只有指定 `--manual-execute` 才等待 Viser 点击。原有自动 IK、碰撞和执行前检查保持不变。没有 `--execute` 会在构造 runtime 前退出。输出目录不能存在，重复实验必须换目录。

如以后找到另一条已确认的旧 rollout，可仅离线冻结：

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.gravity_unfold freeze-v0 \
  --baseline /absolute/path/to/old_unfold_run \
  --output /absolute/path/to/new_v0_archive \
  --evaluate
```

不提供 `--baseline` 时仅自动选择有明确完成记录的 unfold；不存在就报错，不会擅自把最近一次运行标为完成。

## 产物与验证

v1 运行逐步更新 `unfolding_rollout_v1.json`、`unfolding_rollout_v1.md`；包含 before/after RGB 路径、primitive、grasp/targets、命令、原执行结果、progress、outcome、negative experience、v0/v1 指标。每步保存 planner 原始请求响应与执行器证据；progress 并入下一轮 planner，不生成逐步 evaluator 请求。失败和中断也输出报告；未实机运行前不会伪造 v1 rollout。

```bash
/home/sja/miniconda3/envs/cali/bin/python -m pytest -q \
  tests/test_gravity_unfold.py tests/test_codex_unfold_memory.py \
  tests/test_codex_arm7_runtime.py tests/test_unfold_grasp_retry.py
```
