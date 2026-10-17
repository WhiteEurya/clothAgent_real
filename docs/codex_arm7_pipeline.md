# Arm7 单臂折衣 pipeline

当前动作接口已更新为 **GPT 声明抓点＋持布动作列表，宿主规划前后接近、抓取、释放与回 Home**，详见 [抓点与持布动作接口](codex_grasp_task.md)。下文旧 `move_tcp`/`gripper` 流程仍兼容历史代码；新模型输出遵守上述分工。

2026-10-17 起，arm7 的宿主放布顺序为：最终放置点 → 小开爪 → 竖直抬升 → 完全开爪 → Home。
小开爪依据释放前实测夹持开度，最多增加 120 pulse；接近全开时最多使用剩余行程的一半，
确保仍有独立的全开阶段。若剩余行程不足以可靠确认部分打开，则停止，不进行抬升。
小开爪沿用原位置反馈与静止确认，不能以“指令已发送”替代完成。抬升依次尝试 60/40/20 mm，
保持 XY 与末端姿态，不在全开前改成横向撤离；整段仍在抓取前完成 IK、路径和碰撞检查。
抬升到位后开至原配置的全开位置，再回 Home。此流程只改变宿主 `grasp` 任务的释放阶段，
不增加 GPT motion 字段，也不改 Graph；直接的旧式 `gripper(open)` 仍表示全开。
`native_plan.json`/计划统计记录顺序与抬升高度，`gripper_*.json` 记录部分开爪的实测初值、目标和完成反馈。
120 pulse 是控制器开度增量，不作为毫米间距或布料已脱离的证明；最终放置姿态仍需让布料受到接收区支撑。

在项目根目录启动：

```bash
/home/sja/miniconda3/envs/cali/bin/python scripts/codex_arm7_pipeline.py
```

独立折衣入口使用本机 Codex（gpt-6-astra/low）、五张顺序人类示范图、当前 RGB-D 点云和动作代码接口；每轮只有一次模型规划，不调用额外反思，不注入历史经验。低推理强度使用参数 `low`，包括同一会话的后续轮次；Graph replay 的模型设置独立保留。

IK 失败且未执行时，下一次规划尝试复用同一 obs，将具体失败步骤和完整报告交回 GPT，只修正失败代码，保留既定抓点和折叠顺序，不重新看图规划任务；不重复 Home/拍照。与双臂共用这一失败反馈逻辑，执行故障不自动重试。

流程：仅打开 arm7 夹爪、arm7 回 Home、arm7 到自己的已保存 perception（整段先验证、显示 Viser 后自动执行）→ 到位后采集 obs → GPT 输出仅含 right 的动作 → IK/RRTConnect 与严格碰撞验证 → Viser 预览后自动执行 → 下一轮。模型从当前 obs 判断完成时输出 `motion.complete()` 结束。

预览默认 `http://127.0.0.1:8088`，可用 `--viser-port` 修改。其他参数和双臂入口一致，例如 `--max-rounds 1`、`--task`、`--reference`、`--output`。默认记录目录为 `/mnt/newssd/sja/clothAgent_real/runs/codex_astra/loop_arm7_时间戳/`。

拍照准备阶段与 GPT 生成的折衣代码都只允许 arm7/right。模型动作解析阶段拒绝任何 left 命令；下发前再次验证折衣操作属于 right，且 arm6 关节保持不变。与双臂入口共用硬件进程锁。

单臂入口使用 arm7 的 `data/robot/xarm7_perception_pose.json`，Camera B 序列号 `233622079809`，外参取自 `config/calibration/dual_arm_working_20261008/camB_extrinsics.yaml`（平移从米换成毫米，挂载于 right/link_eef）。内参由实际 RealSense 流读取。配置只在单臂进程中派生，不修改双臂配置文件；不继承 arm6 的 1° 倾角修正。

Camera B 深度曝光固定为 **4000 µs（4 ms）**、增益 **16**，折叠和展开入口共用；RGB 曝光不随之修改。该设置来自布料 IR 过曝排查和实机验证，避免重新使用自动深度曝光时产生已观察到的深度变化。观测文件记录曝光策略。当前相机外参按实测 **32.5 mm** 黑色标签边长重新拟合，左右两处标签中心的目测对位及约 50 mm 指尖间隙已通过验证；不叠加排查期间的 16 mm／9 mm 手动 XY 偏移。单次褶皱抓取成功不等于所有平铺布边都已验证，抓取下降策略仍按下文执行。

arm6 在准备、采集和折衣期间均保持原位，不使能其运动、不操作其夹爪。arm7 专用入口完全不连接 arm6；碰撞检查使用 `data/robot/arm6_parked_for_arm7.json` 保存的固定关节位置。每次采集前（包括首次启动），若 arm7 已在保存的 perception 位置允许的关节误差内，则复核到位后直接采集，不回 Home、不操作夹爪；未到位时仍按上述准备流程执行。发生故障时停止已连接的 arm7，arm6 固定障碍适配器不发送停止命令。

原双臂启动入口不变。离线测试及实机测试结果分别记录在测试套件和 `results/arm7_run_check/`。

抓取高度：arm7 闭合前最后一条显式 `move_tcp` 的 World Z 由宿主额外减去 15 mm。接近、抬升、折叠运输、松开目标保持原值；重复闭合不重复修改同一目标。提示词要求 GPT 输出原始观测高度，避免模型与宿主重复偏移。实际动作保存在 `effective_motion.py`，偏移记录在 `grasp_height_policy.json`；IK、碰撞验证和 Viser 均使用修改后的目标，若降低后碰撞仍拒绝执行。

## 自动执行与计时

arm7 单臂入口现在默认在严格规划验证通过、生成 Viser 预览、确认起始关节状态后自动执行，不再等待按钮。失败或 UNKNOWN 仍拒绝下发。需要人工预览暂停时加 `--manual-execute`；`--auto-execute` 可显式启用自动模式。双臂入口现也默认自动执行，可用 `--manual-execute` 恢复等待点击。

每轮打印耗时并写入 `round_NNN/timing.json`：`observation_s` 包含 Home/perception 准备及 RGB-D 采集；`gpt_planning_s` 为模型调用；`trajectory_planning_s` 为 IK/RRT/严格验证；`preview_s` 为预览生成；`preview_wait_s` 为人工等待（自动模式接近零）；`pre_dispatch_check_s` 为执行前状态复核；`execution_s` 为动作执行；`round_total_s` 为全轮总时间。`motion_pipeline_s` 是运动阶段总计，包含上述规划/预览/执行子项，不应重复相加。会话总时间存于 `result.json.elapsed_s`。失败和中断也记录已花费时间；未开始的阶段不伪造为零耗时。

GPT 附加指令：尽可能和human demonstration相似。此要求与 arm6 同步。

## 腕部旋转与相机线缆

观察方向按“从机械臂朝夹爪看”：当前模型 joint7 的负向旋转为逆时针。arm7 抓取接近阶段会比较 joint7 相差 360° 的等价解，在关节限位内先选择较近解；若较近解仍需顺时针转动超过 90°，则优先采用逆时针等价解。例：从 0.3° 到 185.9° 改为到 −174.1°，TCP 位置和姿态均保持不变。这条 joint7 规则只更改关节角圈数；另有以下独立的平行夹爪 180° 等价朝向修正。

下降与持布的连续 IK 选择最接近前一关节角的等价解，避免 ±180° 表示切换造成额外绕圈。回 Home/perception 保留原关节目标，允许反向退回。所有新路径仍须经过现有碰撞验证；此规则是抓取目标转向偏好，不是线缆几何模型，也不保证避碰绕行中每个小段都逆时针。若关节限位不允许逆时针等价解，则保留限位内较近解。

## 队列下发

arm7 折衣阶段使用与 arm6 相同的队列下发：相邻同臂移动最多 32 段一批，中间指令 `wait=False`，队尾等待完成，减少逐段 SDK 等待和到位复核产生的软件间隔。夹爪动作以及准备阶段的 Home/perception 边界保留。下发前逐段检查规划中的 arm6 固定关节不变；执行中监控 arm7 实时反馈，arm6 不读取实时反馈。

原路点、速度、加速度和 `radius=-1` 保持不变；队列减少软件等待，但控制器仍可能在路点处减速或停止，并非连续速度插补。arm7 专用入口关闭 right 的额外碰撞包络（停止距离、跟踪角误差、静止角容差及位置误差膨胀），与 arm6 专用入口的策略对应；仍保留名义几何和完整指令运动范围的碰撞检查、执行反馈容差及停止处理。此设置仅作用于当前运行时，不修改共享标定文件或双臂入口。`native_plan.json` 中的 `queued_segments` 记录原始路点。


## 展开复原与持续会话

新增入口将揉乱的衣服恢复到 human demonstration 的最初展开状态：

```bash
/home/sja/miniconda3/envs/cali/bin/python scripts/codex_arm7_unfold_pipeline.py
```

默认目标图为 `data/reference/fold_states/active/state_00_unfolded.png`，只使用这一张目标图。可通过 `--reference /absolute/path/initial.png` 替换；此模式只接受一张参考图。默认 arm7 单臂自动执行，可加 `--manual-execute` 或 `--max-rounds N`。结果目录前缀为 `/mnt/newssd/sja/clothAgent_real/runs/codex_astra/unfold_arm7_`。展开复原模式从额外下降 17 mm 开始；闭合后继续搬运，不单凭机械信号判空。下一轮图像与可靠闭合反馈共同支持抓空时增加 3 mm，最多额外下降 30 mm。普通折叠模式仍为 15 mm。腕部转向偏好、队列和名义碰撞检查继续生效。

同一次 run 的所有 iteration（包括 IK 修正）复用同一个 Codex 会话：首轮 `codex exec`，从事件读取 `thread_id`，后续使用明确 ID 的 `codex exec resume`。不使用 `--last`，不同 run 新建独立会话。续接失败、会话 ID 缺失或改变时停止，不静默创建新聊天。GPT 仍只读分析，运动只由宿主验证后执行。

不再要求输出或传递 `memory` 摘要。展开模式只返回 `code` 与 `grasp_assessment`；后者是宿主谨慎判空需要的当轮结果，不是交接记忆。每轮提供当前观测、目标参考和上一轮执行结果；展开时同时附上一轮动作前图用于直接比较。IK 修正轮沿用原观测，不虚构新动作。旧坐标不能直接用于新一轮动作。

- `session.json`：本 run 的会话 ID 和最近一次模型调用轮次。
- `round_NNN/session.json`：当轮会话 ID 及是否续接。
- `round_NNN/grasp_assessment.json`：当前对上一动作的视觉判断。
- `iteration_history.jsonl`：动作、执行结果、评估和观测目录的审计记录，不作为每轮记忆摘要注入。

`--remember-iterations` 和 `--memory` 只兼容旧启动命令，不再启用摘要转交。普通单臂、双臂折叠循环也复用各自 run 的会话。新启动的 run 不自动继承旧 run；已有进程需重启才使用该实现。持续会话仍受模型上下文长度和 Codex 自动压缩机制约束，不代表无限保留全部历史原文。

展开流程仅在当前图显示衣物已接近目标展开形状时返回 `motion.complete()`；执行成功不代表任务完成。

### unfold 抓取反馈

arm7 初始额外下降 17 mm（在原 15 mm 基础上再下降 2 mm）。夹爪闭合完成后继续搬运，到原定释放点才开爪，不单凭位置或机械信号中止。下一轮 GPT 在 `grasp_assessment` 中用前后图像评估上一轮：仅 `result=missed`、`cloth_motion=unchanged` 且有具体依据，同时可靠闭合反馈支持空闭合，才在本轮增加 3 mm。证据不足、衣物已移动或反馈异常均保持深度；成功后保留当前偏移。优先重新定位同一抓点附近，坐标须从新观测读取。额外下降上限为 30 mm；在该深度仍确认失败时以 `GRASP_RETRY_LIMIT` 停止，等待人工检查。

判断记录保存为 `grasp_retry_policy.json`，实际偏移为 `grasp_height_policy.json`；判断使用上一轮的执行反馈，不把本轮尚未执行的预期当成事实。IK 修正轮不重复加深。调整后的实际目标仍经原有 IK 和路径检查；此改动不触发立即开爪。

默认输出遵循 `config/run_storage.json` 的 `runs_root`，所有轮次图像、深度、视频和 planner 日志都保存在该输出目录下。配置要求的数据盘未挂载时会报错，不回退到系统盘。显式 `--output` 仍覆盖默认位置，请勿指定到已满的系统盘。修改仅影响新启动的 run，已有输出不自动迁移。

arm7 的 grasp 任务（普通折叠、unfold、Graph Seed 和 Graph Replay）在规划前按当前世界姿态选择平行夹爪的 180° 等价朝向，并对所有后续持布目标施加同一个工具 Z 轴半圈变换，保留 XYZ 和相对旋转。选择记录保存为 grasp_orientation_policy.json；有效姿态及下降偏移保存在 effective_motion.py。相机和附件并非对称，变换后的完整路径仍经过 IK 与碰撞检查。此策略由 Arm7Runtime.parallel_jaw_grasp_symmetry 默认开启，不再仅依赖 unfold 标志；不改动 Home/perception 关节目标，也不改变当前抓取深度策略。


### GPT 每轮规划事件记录

新启动的循环默认开启可观察规划事件记录。GPT 在实际读取、搜索、形成假设、修改候选代码、验证、修正和结束本轮规划时，用中间消息发送带 `PLANNING_TRACE` 标记的简短 JSON；宿主旁路读取事件流，实时追加，不等待最终回答再回忆补写。最终动作格式、只读权限和机器人执行流程保持不变。

- `<run>/planning_trace.jsonl`：跨轮按实际接收顺序追加，`step` 全局递增。
- `<run>/round_NNN/planning_trace.jsonl`：该轮同一组事件。
- `<run>/round_NNN/planner/result.json` 的 `planning_trace`：接收数量、拒收数量、阶段和记录错误；初始化失败另有 `planning_trace_error`。
- `iteration`、`session_id`、`recorded_at`、`model_step`、`source_event_line`、`source_item_id` 用于对应轮次及 `planner/events.jsonl` 中的真实事件。`known` 和 `hypothesis` 是模型显式声明，事实是否正确仍须核对引用的工具/图像证据。

只收集模型主动输出的公共摘要，不从 reasoning 事件、工具输出或最终动作反推出思维过程。`complete` 仅表示收到合法 ORIENT/DONE 边界且无记录错误，不保证每个该记录的事件都被模型报告；`coverage_verified=false` 明确表示未验证完整覆盖。缺失记录不会补造，trace 错误不替代正常任务结果。DONE 只表示本轮规划结束，不表示物理任务完成。额外摘要会增加少量 token 与时间开销，不保证零开销。
# 海绵桌面 Z 拉平（2026-10-14 当前策略）

按用户要求替换旧 Z 补偿，再从未补偿的 world 点云拉平，禁止叠加旧补偿。
共享配置 `config/arm7_support_height_patch.json` 的
`z_policy=raw_plane_to_reference_height` 使用最新裸露海绵拟合：
`z_plane=-0.03825796146786573*x+0.020079931433019782*y+83.55115145136632`。
`z'=z_raw-z_plane(x_raw,y_raw)+52.5038183117912`（mm）。
52.5038 mm 使用用户确认的海绵接触 TCP 高度，来源为 `diagnostics/constant_height_cross_20261014T131934Z/plan_summary.json`。
这是经验抓取高度基准，不宣称等于真实光学海绵表面高度；保留当时接触姿态记录，未假设已完成不同夹爪朝向的指尖换算。
相对此前 55.0429 mm 基准，所有相同观测抓点整体降低 2.5390 mm，衣物高度差仍保留。
旧锚点 XY 仅作为历史参考保留，当前 Z 计算采用显式 `reference_height_mm`。
拟合海绵面的高度统一；衣物相对高度与测量残差保留，不强行把所有像素 Z 设成常数。
该算法允许上调和下调，不采用仅下调裁剪；绝对高度仍为经验基准，并非独立实测标定。

相机采集始终先生成未经补丁的 world XYZ，再在 save_observation 中应用一次合并补偿。
XY 补偿仍独立使用原始 XY。原点云和刚体外参保留；仿射映射记录在观测元数据中。
Camera B 单臂/双臂/Graph/fallback 共用配置，姿态检查和 calibration_id 版本隔离保留。
旧补偿备份及新方案离线验证见 `diagnostics/table_z_relevel_20261014/`。
重启实例重新采图后生效；当前基础抓取下探和右侧额外下探均为 0 mm，本次不修改。

### 2026-10-14 XY 试运行补偿

共享补丁中的 `xy_patch` 默认开启，单位为 world mm：
`X'=X-12`；`Y'=Y-8*(Y+73.1014556885)/261.3136444092`。
左右端点取当前标定下两次候选悬停的 world Y：-73.1014556885 和 188.2121887207 mm，
记录路径见配置 `xy_patch.anchor_records`。偏差幅度来自用户本次口述，
不是从旧记录中的 16/9 mm 偏移重新拟合；作为经验试运行补丁，尚无新实机验证。
中间位置 Y 减 4 mm，左右端点分别减 0/8 mm。端点之外继续线性外推，
因此左端之外可增加 Y、右端之外可减少超过 8 mm；该范围尚未验证。
XY 与 Z 均从原始点云计算一次，合并仿射矩阵保存在观测元数据，原始点云保留。
补丁参数改变会改变 calibration_id；重启后重新采图生效。将 `xy_patch.enabled`
设为 false 可单独关闭 XY，保留 Z。当前基础抓取下探为 0 mm，右侧额外 3 mm 仍关闭。
