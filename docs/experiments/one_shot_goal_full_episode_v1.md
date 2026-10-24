# one_shot_goal_full_episode_v1

2026-10-22 人工排名入口 `scripts/run_one_shot_preference_episode.sh` 已改为
`S = Ps + 固定 alpha × Pt + Rθ(x)`。只采perception，使用当前RGB＋固定目标；
先展示修正后Top10，人工调整完整排名并单独选择执行动作，先保存训练前指标，再训练Rθ。
本轮执行选择不变，更新从下一轮影响排名。冻结DINO提取特征、零初始化线性修正层，
不增加GPT请求。后续每个状态保存各历史checkpoint排名，与当轮人工判断比较；
不把同轮训练后拟合当成泛化效果。详见[当前操作说明](../../reports/one_shot_residual_instructions_20261022.txt)。
下文双视角与单次模型请求适用于原始自动one-shot入口，不适用于人工排名入口。

夹爪方向更新：正常 one-shot 和可选 Codex 恢复都可输出 `grasp_angle_deg`。
它是 perception 图中两指连线的开合轴方向：向右 0°、向下 90°、图像中顺时针增加，
范围 0–180°（0° 与 180° 等价）。`null` 表示保留 Home 方向；DONE/ABSTAIN 必须为 null。
只在抓取前调整方向，夹住之后保持相同姿态完成抬升、平移、放下，不增加持布旋转动作。
Host 使用原手动角度编辑器的 `image_angle_to_rpy`、实测 Pa 深度、相机内外参和 TCP 偏移，
转换为 World 姿态；选择相对 Home 最近的半圈等价方向，保留倾角。
图像角度不是机器人 yaw 或 joint6 角度。角度无效或标定/IK 检查失败时不执行。
请求仍是一次 GPT 调用，回程仍与请求并行。旧离线 probe 的 prompt/schema 保持可读。
动作 overlay 的绿色标记显示开合轴；`grasp_orientation.json` 记录原方向、模型角度、
转换后的 RPY 和 `rotation_while_holding=false`，`motion.py` 保存实际生成的动作。
Shadow gate 同样接收角度；重复动作检查同时比较 Pa/Pb 和开合方向（模 180°，容差 5°）。
启动命令不变；旧 prompt/schema 的 episode 不能用当前版本直接 continuation。
此更新已经离线验证，尚未实机验证旋转抓取效果。

2026-10-21 双视角更新：每轮先在 perception 保存 RGB-D，再让 Camera A / Arm6
移动到已记录的 `data/robot/views/right_view.json`，静止后保存第二张图。
FastPlanner 按 perception、right view、固定 goal 的顺序接收三张 RGB，优先抓清楚可见、
能带动展开的边缘或角点；Pa/Pb 始终使用第一张 perception 原图坐标。
两段观测移动都限制关节速度不超过 10°/s、加速度不超过 20°/s²，且不超过原配置。
采用现有 CollectionRuntime 关节命令及检查，无夹爪开合；回程返回第一张照片记录的关节位置。
先规划去回两段，再开始移动。拍完 right view 后启动一次 GPT 请求，同时慢速返回 perception。
只有回程核验和模型输出都成功，才进入原 Host 检查和抓取。回程失败或中断不会执行模型动作。
HTTP 工作线程不持有机器人；停止不会等待仍在网络中等待的请求，迟到结果只留作审计、不可执行。

逐轮新增 `right_view/rgb.png`、`to_right_view/`、`view_return_perception/`、
`view_planning.json`。每场冻结 `right_view_pose.json`；相机/机器人身份和 TCP offset 必须匹配。
`timing.json` 和 `latency.csv` 新增：

- `view_observation_s`：去 right view、静止、拍摄及观测路径准备。
- `view_return_s`：回程及到位核验。
- `planner_return_overlap_s`：模型工作任务与回程的重叠区间（包括请求准备，不能等同服务器推理时长）。
- `model_wait_after_return_s`：回到 perception 后还需等待结果的时间。

不要把 planning 与回程相加当作墙钟总耗时；整轮看 `total_round_s`。
接触表已包含 right view。独立 shadow gate 继续使用 perception、goal 和动作 overlay。
可选 Codex 错误恢复同样采集、接收双视角，但其工具调用阶段仍在回程完成后同步启动。
初始化的区域识别仍只使用初始化当前图；离线历史 probe 默认保留原两图接口。
本次只有代码与离线测试，尚未验证真实观测路径、双视角收益或真实时间节省。

2026-10-21 用户授权 prompt 修正：在原任务说明后追加固定夹爪边界，恢复 Agent
也接收相同段落。沿用 `data/robot/xarm_boundaries_new.json` 和当前配置的原值：
Arm6 基座毫米坐标 Y ∈ [-250.578537, 246.447205]，Z ≥ -0.542602，X 未设限，
Z 未设上限。不增加内缩距离、图像边界或新的 Host gate。双视角版本为三张 RGB 和单次
Fast 请求；毫米边界不是图像像素，模型没有因此获得边界的图像投影或深度。
新 episode 的 prompt.txt / protocol.json / request_payload.json 保存此变更；
旧实验记录不改写，旧 prompt 的停止段不能通过原 continuation 一致性检查。

新增可选错误恢复：单 episode 默认仍关闭；加 `--codex-recovery` 可对已知的执行前
深度、动作几何、计划 IK/workspace 拒绝和 Fast 输出错误启动独立 Codex 重新规划。
连续采集入口默认启用。每次错误最多 2 次、每 episode 最多 4 次，计入 round 上限。
新 round 重新采图，Codex 读取当前 RGB-D、固定 goal、原拒绝记录，不访问 memory/Graph。
替代动作仍须通过原 Host 检查；成功执行后回到 FastPlanner。初始化/硬件故障、
已经 dispatch 后的错误、未知错误、Codex 本身失败或预算耗尽均停止。
Gate 仍只做 shadow；Fast 使用上述双视角 prompt，仍是单次调用。下面的无 Agent、首错即停说明指
默认未启用恢复的单 episode 模式。详细日志与限制见
[连续采集说明](one_shot_goal_continuous_v1.md)。

```bash
bash scripts/run_one_shot_goal_episode.sh --execute --codex-recovery
```

## 独立终端启动：中心附近抓起丢下 + one-shot 闭环

在另一个 VS Code 终端运行：

```bash
cd /home/sja/clothAgent_real
bash scripts/run_one_shot_goal_episode.sh --execute
```

脚本使用现有 `cali` Python 环境，每次生成新的
`reports/one_shot_goal_full_episode_v1_<timestamp>/`，不会覆盖之前的实验。
不传参数只显示用法，`--help` 只显示参数；这两种操作不连接机器人。

流程沿用本实验的原始实现：采集 RGB-D → 单次直接请求识别衣服区域 →
在识别中心周围 80 mm 内选择有效抓点 → 既有抓起丢下初始化一次 →
perception 采集 → right view 采集 → 三张 RGB 的单次 `gpt-6-astra/high` 请求与回程并行 →
独立 PreActionGate（SHADOW ONLY）→ Host 检查 → 至多执行一个 `lift_pull_place` →
重新采集并进入下一轮。Gate 仅对 ACTION 请求，不拦截、替换或修复动作；
DeliberativePlanner / MemoryPlanner 只定义接口并保持禁用。
初始化抬升 250–300 mm，放下时 world Y 和 yaw 固定为 Home 值；
X 保留原有不超过 ±40 mm 的偏移采样，不再随机改变 Y 或 yaw。
它不是强制抓取精确中心像素。初始化不自动重试，默认不等待额外人工确认文件，
执行完成也不证明抓取成功；可查看 `initialization/before/`、`after/` 和执行记录。

默认最多 12 轮；每轮不提供 memory 或 episode history，prompt、primitive 和线上
检查均沿用当前版本。DONE、ABSTAIN、检查失败或达到轮数上限时结束，不自动续跑。
前台按 `Ctrl+C` 走现有停止与记录保存流程，运行时保持现有人工监督和急停流程。

```bash
# 调整轮数或目标图；省略时使用此前实验的固定目标图。
bash scripts/run_one_shot_goal_episode.sh --execute --max-rounds 6
bash scripts/run_one_shot_goal_episode.sh --execute --goal-image /absolute/path/to/goal.png
```

初始化、逐轮 RGB-D、计划、动作图、执行记录及耗时均保存在新结果目录；
结束后查看 `report.md`、`episode_summary.json`、`episode_contact_sheet.png`，
通过 `review.html` 导出人工展开质量判定；后续分析写到独立目录。

当前版本新增 [分层 planner 与 shadow gate 文档](layered_unfolding_shadow_v1.md)。
每轮 FastPlanner 使用上述三张 RGB 和双视角 prompt；ACTION 额外产生一个独立 gate 请求，
其输入仅 current、goal、Pa→Pb overlay。Gate 标签和 gate 耗时单列。
`review.html` 用于逐轮人工评审；导出的标签通过离线汇总命令写到独立目录，
不自动改动原 episode 执行记录。

Real closed-loop counterpart of `one_shot_goal_planner_probe_v1`. The strict schema is retained; the multiview prompt prefers visible edges/corners. Each round makes one FastPlanner direct gpt-6-astra/high request with perception RGB, right-view RGB and the frozen goal RGB, overlapping the verified return to perception; tools=[], tool_choice=none, no HTTP retries, no repair, no previous-response ID, no trajectory or episode history. ACTION proposals additionally receive one independent shadow-gate request. RGB-D remains available only to Host checks. No Agent CLI is launched.

Run exactly one new episode (refuses existing output):

```bash
/home/sja/miniconda3/envs/cali/bin/python -u -m cloth_agent.experiments.one_shot_goal_full_episode_v1 --execute
```

Defaults: max_rounds=12, goal copied from `reports/goal_only_full_episode_v1_restart02/goal/target.png`, output `reports/one_shot_goal_full_episode_v1`. CLI supports --goal-image, --output, --config, --max-rounds, --timeout-s. SIGINT/SIGTERM stops the runtime and records USER_STOP. Keep existing physical emergency stop and operator supervision.

Initialization uses existing reset_program and sample_grasp with the HOME_Y_HOME_YAW_V1 release policy: random central visible fabric grasp, -15 mm offset, 250–300 mm lift, Home Y and Home yaw at release, existing X offset sampling bounded by ±40 mm. No extra yaw rotation. One direct tool-free region localization request replaces the old Agent localization; no comparison model, no repeated reset. This request and initialization physical action are counted separately. Older saved episodes used randomized Y/yaw and are not rewritten.

Execution uses EpisodeCollectionRuntime (recording subclass of CollectionRuntime) and the existing lift_pull_place. Following the user-authorized bare-table depth change, only this one-shot entrypoint enables `BARE_TABLE_TARGET_PLANE_V1`: Pa still requires measured depth; a Pb missing depth may use the current measured support-plane intersection after bare-table checks. Other callers of shared `prepare_action` retain the original behavior. Metric displacement/workspace/IK/joint/live-feedback checks and the Home/perception path remain in place. CollectionRuntime already disables swept path validation and left additional collision envelopes; these are not newly disabled or claimed as passed. Any remaining check rejection stops without alternate pixels or retry. A Host-only repetition guard stops an unchanged observation with the same proposed action; history never enters the model input.

### Bare-table Pb depth fallback

`cloth_agent/table_target_depth.py` tries the original depth sampler first. Only
`DepthUnavailableError` for Pb invokes the fallback; a measured depth discontinuity
is still rejected. Pa and initialization grasps never use this fallback.

The fallback reuses the existing `perception.free_exploration.json` bare-background
ROI, RGB foreground exclusion with 24 px clearance, complete 15×15 reference
patches, and robust measured support-plane fit (6 mm inlier tolerance). It does
not assume the hard-table Z=0: the work surface includes the raised mat.
The camera-A factory RGB intrinsics are checked against the captured XYZ cloud.
Additional local checks require nearby measured support within 25 px, at least
100 complete-patch reference centres within 50 px, coverage surrounding Pb,
local plane consistency, and no contradictory valid depths in Pb's 5×5 patch.
The clicked Pa must have clear appearance contrast from the table. Matching/dark
garments and ambiguous RGB evidence are unsupported and rejected.

The result is a ray/plane intersection at the unchanged Pb pixel. Only its target
XYZ is substituted: the RGB-D artifact is never filled or rewritten. This is
an appearance heuristic plus geometry evidence, not proof that an occluded
object cannot exist. No physical success is inferred from an offline pass.

Each round records `target_depth.json` with `MEASURED_DEPTH` or
`TABLE_PLANE_ESTIMATE`, plane coefficients, actual reference patches, residuals,
configuration hashes and rejection reasons. `points.json` records the same
provenance, and `host_validation.json` embeds it. The episode freezes the fallback
policy/configuration/calibration hashes; changes before Host preparation stop the
episode. FastPlanner prompt, shadow gate and manipulation primitive are unchanged.
No robot motion or automatic episode restart accompanies this code change.

Each round records full before RGB-D, request_manifest/request_payload, raw SSE/JSON, exact plan, proposed overlay, execution, after, timing, execution images and joint/gripper feedback. Rejected or interrupted rounds use after/unavailable.json rather than fabricate an after or move after stop. Physical action counts include dispatched partial executions. Acquisition stays UNKNOWN_PENDING_HUMAN_REVIEW.

Report separates initialization from episode planning/execution, includes per-round and median/mean/p95 latencies, total times, dispatches, safety rejections and termination. MODEL_ERROR_NO_RETRY fails closed under SAFETY_STOP with failure_phase=MODEL (not counted as geometric safety rejection). Human choice remains null until FOLD_READY / NOT_FOLD_READY / UNCERTAIN is entered. DONE is never automatically success.

API reference: [OpenAI Docs structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

User-authorized continuation is supported with `--continue-from <stopped-output> --max-rounds <remaining> --output <new-output>`. It requires a SAFETY_STOP source with the same model/prompt/schema, preserves the goal, starts with fresh RGB-D at the existing observation pose, and does not reset the garment. The original 12-round budget includes rejected rounds. Executed prefixes remain Host-only repetition-guard evidence; they never enter model input. Every enabled-check rejection stops that segment; an authorized later segment begins a new round, not an HTTP retry or same-round repair. The consolidated run with authorized retries must be reported separately from the first-stop protocol.

`--review-initialization` holds the robot stationary after reset until an image-hash-bound ACCEPT/REJECT record is written to `initialization/review_decision.json`; it does not make another planner call or infer a human FOLD_READY label. Timeout or rejection fails closed.
