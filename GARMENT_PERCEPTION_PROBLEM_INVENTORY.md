# Garment / Deformable Manipulation 问题与证据清单

> 审计日期：2026-08-16
> 范围：当前 `cloth_agent` 代码、保存的 garment 真实运行、RGB/RGB-D/height-map/coordinate artifacts、失败记录、诊断记录与 Git 历史。
> 限制：本清单没有启动机器人、没有采集新数据，也没有把 Claude 的文字判断直接当作物理真值。

证据标签：

- **FACT (saved record)**：直接来自已保存的 JSON、日志、代码或 Git 记录。
- **FACT (manual image audit)**：本次逐张核对保存的 before/after PNG 后可以直接看到的内容。
- **INFERENCE**：由事实支持但尚无传感器直接证明的因果解释。
- **UNKNOWN / INSUFFICIENT EVIDENCE**：现有文件不能区分候选解释，或根本没有相应实验输出。

统计口径：下文“32 次”只统计 `runs/**/results/auto_exploration/**/iteration_NNN.json` 形式的 32 个顶层、已合并真实迭代记录，不把 `iteration_NNN_failed.json`、只有 proposal/preflight 的未执行迭代、或未生成合并记录的执行混入分母。对 `observed_change` 的 12/8/9 条归类是本次对保存文本的人工编码，不是新的视觉 ground truth。

# 1. Executive Summary

## 1.1 核心结论

1. **当前 state representation 能描述可见几何，不能恢复材料拓扑。** 6 mm A/B voxel fusion、table-relative height、garment mask、90 分位 height-gradient edge 都是有用的几何线索，但没有任何一个量直接给出层数、上下层关系、材料连通性或“抬起后哪一侧会自由下垂”。
2. **raised ridge / steep gradient 不是 good lifting anchor 的充分条件。** 最近对相似“raised ridge/lip”的真实动作分别得到 no-op（`111708`、`164015`）、局部 drag/flatten（`193710` iteration 1）、双侧 tent/redistribution（`193710` iteration 2）、局部 flap reveal（`200329` iteration 1）以及轻微位移但无评估（`201533` iteration 1）。
3. **“机器人动作成功”与“抓持成功/研究目标成功”必须分开。** 32/32 合并迭代都记录 `physical_execution=true`、`execution_completed=true`、所有 `actual_robot_actions[].success=true` 且 `robot_errors=[]`；但 Claude 标签只有 11/32 useful。`111708` 和 `164015` 的轨迹完整成功，before/after 却近乎不变。
4. **whole-garment / multi-layer motion 是反复出现的主要失败。** 至少 8 条保存的 `observed_change` 明确描述整件/整体平移或近似刚体运动；diagnostic lift 的原始 Claude 返回还明确给出 `MULTI_LAYER_GRASP`，同时可见面积下降 6.6%、mound height 未下降。
5. **抓到并移动真实 cloth 仍不等于得到 usable hanging anchor。** `200329` iteration 1 确实揭示了印字和 doubled stitched edge，但 post-release 图中没有 hanging column，整体 footprint 变化小；这是“局部信息增益”为正而“anchor quality”未证实的清楚实例。
6. **before/after evaluator 当前存在决定性证据缺口和实际时序错误。** planner 在 before 阶段能看到 A/B RGB、height maps、gradient/boundary/coordinate overlays 和 fused artifacts；after 阶段虽然保存 RGB 与 raw depth，`_save_frame_images()` 只把两张 RGB 路径交给 evaluator。`192145` 和 `192822` 又各出现一次肉眼可证的 before/after 颠倒描述。
7. **多视角和 depth 已改善，但不是稳定的材料对应。** 双相机第二次静态测试的 shared-view symmetric p95 为 15.09 mm、97.16% 在 20 mm 内，明显好于第一次的 106.70 mm/74.17%；但 Camera A 始终比 B 更抖，窄 lip 仍出现 A/B 高度冲突，table-plane 漂移也会改变 `height_above_table` 而不改变同一点绝对 z。
8. **坐标命令保存可以很准，但绝对视觉语义仍可能错。** diagnostic audit 中 Claude 的抓点 `(600,30,42)` 与 xArm close pose 几乎一致，同一标定下回投像素误差为 `0.000011947 px`；这排除了该次“命令被改写/控制器没有到位”，却不能证明外参绝对正确、抓点属于所想的材料结构，或 TCP 真正夹住了 cloth。
9. **semantic landmark 已显示不稳定；DINO/canonical surface 尚无运行证据。** 三次平铺单视图 landmark 输出分别只有 6/10、2/10、4/10 visible，且至少一个点落到 garment 外/图像边缘附近。当前 dense path 已不调用 Molmo。`runs/` 与 `results/` 中没有找到 DINOv3 correspondence、review、match 或 canonical graph 的实验输出，因此“DINO 因低纹理/对称失败”目前是 **UNKNOWN / INSUFFICIENT EVIDENCE**，不能写成已观察事实。
10. **Laydown 尚未被可信地验证。** `192822` 是唯一生成合并 judgement 且显式记录 Laydown invocation 的迭代，但 evaluator 把 B 相机的前后遮挡关系说反；`200329` iteration 2 只到 proposal/preflight/IK，未执行；`201533` 的 Laydown 动作执行了却没有 Claude evaluation 或合并记录。现有证据不足以给出成功率或适用条件。

## 1.2 合并真实迭代统计

| 保存记录统计 | 数量 | 解释边界 |
|---|---:|---|
| 合并真实迭代 | 32 | **FACT (saved record)**；不含 failed/未合并 artifacts |
| physical execution / execution completed | 32 / 32 | 只证明执行流程完成 |
| 所有实际 action success，且无 robot error | 32 / 32 | 不证明接触、夹持、单层抓取或 anchor 成功 |
| Claude useful=true / false | 11 / 21 | 日志标签，不是 ground truth |
| `observed_change` 明述无明显/近同/无意义变化 | 12 | 本次对保存文本的人工归类 |
| `observed_change` 明述整体/近刚体平移 | 8 | 本次对保存文本的人工归类 |
| 11 条 useful 中提到新 print/logo/lettering | 9 | 说明文字/图案出现经常驱动 positive label |

**可靠性警告：** `192145` 与 `192822` 的人工图像核对证明，至少两条近期 judgement 对 before/after 的基本时序事实错误。因此 11/21、12、8、9 只能描述“系统保存了什么判断”，不能直接估计真实成功率。

## 1.3 最近真实运行人工复核

| Run / iteration | 执行状态 | FACT (manual image audit) | 保存 judgement 与可信度 |
|---|---|---|---|
| `auto_20260815_111708 / 001` | 完整执行，无 robot error | A/B before/after 实质近同，未见目标 flap peel | useful=false 与图像一致；miss/slip/空夹/错误 z 不能区分 |
| `auto_20260815_164015 / 001` | 完整执行，无 robot error | A/B fold pattern 与 footprint 近同 | useful=false 与图像一致；close position 0 不能证明夹到布 |
| `auto_20260815_192145 / 001` | 完整执行，无 robot error | before A/B 无手；after A/B 明显有人手接触 garment | judgement 写成“before 有手、after 无手”，时序完全颠倒；因果归因无效 |
| `auto_20260815_192822 / 001` | 完整执行；显式 Laydown | before A 已是可读 shirt，before B 清楚；after B 被 gripper/fixture 大幅遮挡 | judgement 写成 before B 被遮挡、after B 清楚，并判 reduced-scale Laydown 成功；不可信 |
| `auto_20260815_193710 / 001` | 完整执行，无 robot error | 左侧 cloth 有移动/变顺，但 collar label 被覆盖；没有确认悬挂 | “partial drag/flatten, not confirmed hang”基本受图像支持；B-after 自遮挡严重 |
| `auto_20260815_193710 / 002` | 完整执行，无 robot error | cloth 重新分布并出现 tent-like 结构；未见悬挂列 | “tent/redistribute”有支持；“collar/shoulder”身份仍是推断 |
| `auto_20260815_200329 / 001` | 完整执行，无 robot error | A 新露出“en the robots!”；B 新露出 print 与 doubled stitched edge；无 hanging column | useful=true 的“有新表面”部分可信；usable anchor 仍未证实 |
| `auto_20260815_200329 / 002` | 未执行 | 只有 seamed-edge Laydown proposal、preflight、controller IK artifacts | 不能作为 Laydown outcome |
| `auto_20260815_201533 / 001` | 完整执行，无 robot error | before/after 有轻微局部/整体位移，未保存 evaluator 结论 | 没有 `claude_evaluation.json` 或合并 `iteration_001.json`；outcome class 为 UNKNOWN |

## 1.4 当前衣服的 Molmo 与 agent 抓取点图

### 2026-08-17 实时重新读取 Camera A/B

本次按用户要求重新读取了实时 Camera A/B，保存到
`results/current_garment_point_figures_20260817/recapture_camAB_20260817/`。采集清单为
`capture_manifest.json`；Camera A/B 深度有效率分别为 `0.9998` 与 `0.8794`。本次只采图、离线识别和作图，`robot_motion=false`，没有启动机器人或执行抓取。

Molmo 对新 A/B 图分别重新查询全部 10 个 garment semantic landmarks，并允许 abstain。按用户要求取消衣服状态、遮挡、场景排除和输出格式等先验，只使用统一 zero-shot 模板：
`Point to this garment's <part>. If unknown, answer UNKNOWN.`

- **FACT (new live recapture, zero-shot)：** Camera A 返回 `10/10` 个可解析 Molmo pointing token；Camera B 返回 `8/10`，其中 neckline 与 left bottom hem 为 `UNKNOWN`。没有人工补点。
- **FACT (visual audit)：** 由于没有加入互斥或纠错约束，输出中有多组重复点。A 的 `right_shoulder/right_sleeve_tip`、`left_bottom_hem/lower_left_half_center`、`right_bottom_hem/lower_right_half_center` 分别重合；B 的 `left_shoulder/left_sleeve_tip` 与 `right_shoulder/right_sleeve_tip` 分别重合。这些点是 zero-shot Molmo 原始返回，不是 verified landmark ground truth。
- 没有 Molmo pointing token 的文本输出仍按 `UNKNOWN` 处理，未转换为像素点。
- 原始 prompt、generated text、raw points 和状态保存在 `molmo_all_parts_unknown_allowed_current_AB.json`。

![Live recapture Camera A Molmo garment-part points](results/current_garment_point_figures_20260817/recapture_camAB_20260817/molmo_all_parts_unknown_allowed_camera_A.png)

![Live recapture Camera B Molmo garment-part points](results/current_garment_point_figures_20260817/recapture_camAB_20260817/molmo_all_parts_unknown_allowed_camera_B.png)

![Live recapture Camera A/B Molmo garment-part overview](results/current_garment_point_figures_20260817/recapture_camAB_20260817/molmo_all_parts_unknown_allowed_AB.png)

Agent 对新图另行提出一个**未执行**的抓取表面候选，未复用旧状态的 `R088`：

- Camera A 直接选择可见衣服内部点 `(321.0, 262.0)`；其测量 surface base XYZ 约为 `(587.6, 24.2, 12.0) mm`。
- 同一个 base-frame surface point 投影到 Camera B 为 `(303.7, 167.8)`；B 局部深度回投与 A 点相差约 `7.5 mm`。
- 该点只是离线 perception proposal。抓取是否成功、是否单层均为 `UNKNOWN`，没有生成或执行 TCP 运动命令。
- 完整记录保存在 `agent_grasp_point_current_AB.json`。

![Live recapture Camera A agent grasp proposal](results/current_garment_point_figures_20260817/recapture_camAB_20260817/agent_grasp_point_camera_A.png)

![Live recapture Camera B agent grasp proposal](results/current_garment_point_figures_20260817/recapture_camAB_20260817/agent_grasp_point_camera_B.png)

![Live recapture Camera A/B agent grasp proposal](results/current_garment_point_figures_20260817/recapture_camAB_20260817/agent_grasp_point_AB.png)

Molmo 与 agent 的新 A/B 四图对比：

![Live recapture Molmo versus agent Camera A/B](results/current_garment_point_figures_20260817/recapture_camAB_20260817/molmo_vs_agent_current_AB.png)

### 较早保存状态（保留作历史对照）

以下旧图使用同一次保存状态：`runs/auto_20260815_201533/results/perception/center_20260815T121539610518Z/`。它们不是上述实时重采图；当时也没有重新采集相机或启动机器人。

Molmo 说明：

- 对 A/B 分别查询 10 个部位：`garment_center`、`neckline`、左右 shoulder、左右 sleeve tip、左右 bottom hem、lower-left/right half center。
- 当前主结果**不强制每项必须有点**：能从可见证据识别时返回一个 Molmo point；不能识别时返回 `UNKNOWN`，不生成坐标，不人工补点。
- 完整 prompt、generated text、raw points 与 `UNKNOWN` 状态保存在 `results/current_garment_point_figures_20260817/molmo_all_parts_unknown_allowed_current_AB.json`。
- **FACT (saved result)：** Camera A 只有 `garment_center` 返回可解析 point `(335.8, 228.0)`，其余 9 项为 UNKNOWN/无点；Camera B 10 项全部为 UNKNOWN/无点。
- 部分 generated text 声称看到 collar/seam，却没有输出 Molmo pointing token；本清单按“没有可解析 point 即 UNKNOWN”处理，没有从自然语言或文本坐标中人工造点。
- 早先的 forced-best-estimate 版本保存在 `molmo_all_parts_current_AB.json`，仅作为“为什么不应强迫输出点”的对照，不再作为主结果。它曾把多个互斥部位放到同一像素。

### Camera A：Molmo 部位点（允许 UNKNOWN）

![Camera A Molmo garment-part points with UNKNOWN allowed](results/current_garment_point_figures_20260817/molmo_all_parts_unknown_allowed_camera_A.png)

### Camera B：Molmo 部位点（允许 UNKNOWN）

![Camera B Molmo garment-part points with UNKNOWN allowed](results/current_garment_point_figures_20260817/molmo_all_parts_unknown_allowed_camera_B.png)

A/B Molmo UNKNOWN-allowed 总览：

![Camera A and B Molmo garment-part points with UNKNOWN allowed](results/current_garment_point_figures_20260817/molmo_all_parts_unknown_allowed_AB.png)

Agent 说明：

- agent 在 Camera B 选择测量参考 `R088`，surface base XYZ 为约 `(573.1, -249.6, 25.0) mm`；实际规划的 TCP close 为 `(573, -250, 13) mm`、yaw `60°`。
- 红圈表示所选 cloth surface point，黄色十字表示 TCP close point 的投影。Camera B 是直接测量点；Camera A 是同一个 base-frame surface/close point 用已保存 A 标定做的跨视图投影。

### Camera A：agent 抓取点

![Camera A agent grasp point](results/current_garment_point_figures_20260817/agent_grasp_point_camera_A.png)

### Camera B：agent 抓取点

![Camera B agent grasp point](results/current_garment_point_figures_20260817/agent_grasp_point_camera_B.png)

Molmo UNKNOWN-allowed 部位点与 agent 抓取点的 A/B 综合对比：

![Molmo UNKNOWN-allowed garment-part points versus agent grasp target for Camera A and B](results/current_garment_point_figures_20260817/molmo_unknown_allowed_vs_agent_AB.png)

原单点 Molmo / agent 四图总览仍保留，便于比较 garment-center query 与 agent target：

![Current garment Molmo and agent points for Camera A and B](results/current_garment_point_figures_20260817/current_garment_points_AB_2x2.png)

# 2. Perception Failures

## 2.1 Height/ridge/gradient 只能说明可见几何突变，不能说明层数或 anchor quality

- **Problem：** planner 多次把高 ridge、窄 gradient band 或 underside shadow 当作“可能只含一两层、易形成悬挂”的依据；当前图像量本身不含层数与材料连通性。
- **Concrete evidence：**
  - **FACT (code)：** `cloth_agent/perception.py:1088-1145` 先平滑 height surface，再计算二维梯度，以 garment 内梯度 90 分位且至少 1.25 的阈值画 edge。它是 surface-height gradient，不是 layer classifier。
  - **FACT (saved record)：** `111708`、`164015` 都选了 raised fold/ridge 并完整执行，after 近同；`193710` iteration 1 得到局部 drag，iteration 2 得到 tent；`200329` iteration 1 得到局部 flap displacement 与 print reveal。
  - **FACT (manual image audit)：** 上述 outcome 在“ridge-like target”这一外观类别内明显分叉。
- **对应 run / 文件 / 日志路径：**
  `runs/auto_20260815_111708/results/auto_exploration/20260815T031708538205Z/iteration_001.json`；
  `runs/auto_20260815_164015/results/auto_exploration/20260815T084015856468Z/iteration_001.json`；
  `runs/auto_20260815_193710/results/auto_exploration/20260815T113711203001Z/iteration_001.json`、`iteration_002.json`；
  `runs/auto_20260815_200329/results/auto_exploration/20260815T120330204215Z/iteration_001.json`。
- **Claude 当时认为发生了什么：** 计划文本反复认为窄、陡、抬高的 crest 比宽 plateau 更可能只夹少量层，并把“tent vs hang”作为预期判别。
- **实际 before/after 显示了什么：** 同一类外观先验既可对应 no-op，也可对应局部滑移、双侧 tent 或真实局部揭示；没有一种稳定映射到 one-sided hang。
- **当前最可能的问题来源：** `perception` + `partial observability`；具体 no-op 的物理原因仍为 `unknown`。
- **已经尝试过的解决办法：** 从旧的中心/最高点启发式转向 full RGB-D、height/gradient/boundary artifacts，并在 prompt 中明确“不要假设 highest point/fold convergence 是好 anchor”；Claude 也开始使用“possible boundary/flap”而非强制 semantic label。
- **是否真正解决：** **否。** 表达方式更谨慎，但没有新增 layer-count、material connectivity 或 hang-retention 观测。
- **对后续 research 的启示：** 把 ridge 当作“可执行主动探测的几何候选”尚可；不能把它直接当作“少层”或“好 anchor”的 posterior。真正需要预测的是抬升后的单侧自由度、负载分布与材料连通性。

## 2.2 Appearance mask 与 table zero 对曝光/场景参考敏感

- **Problem：** 当前 garment mask 依赖与估计 table RGB 的颜色距离、宽松 height envelope 和最大 XY connected component；table plane 又依赖低且亮的点及 corner/edge reference。曝光改变会同时影响 RGB 饱和、mask membership 和 table reference selection。
- **Concrete evidence：**
  - **FACT (code)：** `cloth_agent/perception.py:503-569` 拟合 table plane；`2090-2233` 以 `color_distance >= threshold`、height envelope 与 largest XY component 选 garment。
  - **FACT (saved record)：** Camera A exposure sweep 中 saturation 从 exposure 100 的 0.2756% 增到 exposure 800 的 63.9148%；当前配置固定 A 为 800（`config/perception.free_exploration.json:2-8`）。
  - **FACT (saved record)：** 同一静止、无机器人 heightmap exposure sweep 中 garment p95 height 从 exposure 150 的 21.63 mm 跳到 200 的 53.41 mm，800 为 67.05 mm；table-plane residual p95 仍约 4.1–6.5 mm。
- **对应 run / 文件 / 日志路径：**
  `results/exposure_sweep/camera_A_100_to_800_step_50_v3/manifest.json`、`exposure_grid.png`；
  `results/heightmap_exposure_sweep/camera_A_100_to_800_step_50/manifest.json`、`camera_A_heightmap_grid.png`；
  `results/heightmap_test/20260815T122550763527Z/` 至 `results/heightmap_test/20260816T_live_A_exposure_900_v2/`。
- **Claude 当时认为发生了什么：** 多个 judgement 把 brightness/height ridge 的变化用于 fold collapse、layer relief 或 flattening 推断，同时 caveat 也承认 overexposure 会掩盖深色 cloth 细节。
- **实际 before/after 显示了什么：** exposure sweep 没有 cloth motion，却产生大幅 garment p95 height 变化；这证明该 pipeline 的“garment height distribution”包含 mask/reference selection 变化，不能全部解释为物理高度。
- **当前最可能的问题来源：** `perception`。
- **已经尝试过的解决办法：** absolute shared 0–40 mm scale、occlusion-aware mask、corner/edge table interpolation、live exposure 800/900 与独立 exposure sweep。
- **是否真正解决：** **否。** 可视化和参考拟合经历了多轮修正，但残差和相机间 range 差仍存在；固定 exposure 800 还处于高饱和区。
- **对后续 research 的启示：** 所有由 mask 面积、p95 height 或亮色 ridge 推出的进展，应同时报告曝光、table-plane 参数、mask 支持集合与不确定度；静态曝光 sweep 应作为 perception stability baseline，而不是 garment geometry ground truth。

## 2.3 Semantic keypoint 不稳定，语义可见性与返回坐标脱节

- **Problem：** 即使在平铺 shirt 上，Molmo landmark 也经常 `not_found`、落点偏到其他结构或图像/桌面边缘；模型文字可自信描述 landmark，却不一定返回有效点。
- **Concrete evidence：**
  - **FACT (saved record + manual image audit)**：三次单视图 10-landmark baseline 的 visible/not_found 分别为 6/4、2/8、4/6；第一组 `lower_left_half_center` 为像素约 `(3.9,316.9)`，人工查看 annotated PNG 可见其在画面/garment 外缘附近，shoulder 点也常靠近 neckline/upper edge 而非稳定肩点。
  - **FACT (offline replay on saved A/B images, 2026-08-17)**：对最新 `201533` A/B 图允许 abstain/UNKNOWN 后，Camera A 只返回 1/10 个可解析点，Camera B 返回 0/10；说明当前折叠状态下大多数 semantic part 没有足够可见证据。
  - **FACT (forced-output control)**：强制 best estimate 时模型会为 A/B 各返回 10 个标签，但多个互斥部位落到完全同一像素。Camera A 的 left/right shoulder 与 right sleeve tip 同点，左右 bottom hem 与 lower-right center 同点；Camera B 的 right sleeve tip 与左右 bottom hem 同点。这证明强制“返回齐 10 点”会制造伪确定性。
- **对应 run / 文件 / 日志路径：**
  `runs/landmarks_20260812T135443619133Z/landmarks_baseline.json`、`camera_0_A_landmarks.png`；
  `runs/landmarks_20260812T135604487303Z/landmarks_baseline.json`；
  `runs/landmarks_20260812T135701223972Z/landmarks_baseline.json`；
  `results/current_garment_point_figures_20260817/molmo_all_parts_unknown_allowed_current_AB.json`、`molmo_all_parts_unknown_allowed_camera_A.png`、`molmo_all_parts_unknown_allowed_camera_B.png`；
  forced-output 对照见 `molmo_all_parts_current_AB.json`。
- **Claude 当时认为发生了什么：** 早期流程尝试用 neckline、shoulder、sleeve tip、bottom hem 等名字建立 garment structure；后续 judgement 仍会从新开口/轮廓推断“collar/shoulder/hood”。
- **实际 before/after 显示了什么：** landmark detector 连平铺状态都不能稳定覆盖 10 个点；严重 crumpled 状态下更没有证据支持稳定 semantic correspondence。
- **当前最可能的问题来源：** `perception` + `partial observability`。
- **已经尝试过的解决办法：** Git commit `6d28061` 加入 semantic structure；`1972b6f` 解决 Molmo decoder 资源问题；当前 dense path（`cloth_agent/perception.py:2090-2112`）不再启动 Molmo，prompt 要求弱证据时 abstain。
- **是否真正解决：** **部分规避，不是解决。** 当前动作不再依赖 unstable semantic keypoint，但 garment identity/topology 仍未被另一种稳定表示替代。
- **对后续 research 的启示：** semantic part 可以作为后验解释或成功证据，不能在缺乏 visibility/uncertainty 校验时作为主 grounding。应把“模型描述了一个部件”和“返回了 garment 上可复现的物理点”分成两个事件。

## 2.4 Before perception 丰富、after perception 贫乏

- **Problem：** evaluator 的 before evidence 与 after evidence 不对称，导致无法比较高度、梯度、mask、坐标或 3-D，并容易把遮挡/视角变化误当 cloth change。
- **Concrete evidence：**
  - **FACT (code)：** `perception_image_paths()`（`cloth_agent/free_exploration.py:642-688`）会收集 before RGB、per-camera height/global/boundary/gradient/coordinate artifacts 和 fused preview/map。
  - **FACT (code)：** `_save_frame_images()`（`cloth_agent/auto_exploration.py:555-568`）确实保存 after RGB 和 raw depth `.npy`，但返回列表只含 RGB；`1379-1384` 将这两个 RGB 路径作为 `after_images` 送入 evaluator。
  - **FACT (saved record)：** 最近 caveat 反复写“after 只有 RGB、没有 height map/boundary/gradient”。
- **对应 run / 文件 / 日志路径：** `111708`、`164015`、`193710` 两次、`200329` 的 `iteration_001/claude_evaluation.json`；上述代码位置。
- **Claude 当时认为发生了什么：** 常从 after RGB shading/轮廓推断 ridge collapse、layer-count reduction 或 flattening，同时在 caveat 中承认不能量化。
- **实际 before/after 显示了什么：** post-release RGB 可以确认大幅 silhouette、print、hand 或 hardware 遮挡变化，但不能确认 peak lift 是否夹持、何时 slip、table-relative height 是否下降或 layer count 是否减少。
- **当前最可能的问题来源：** `perception` + `evaluation` + `partial observability`。
- **已经尝试过的解决办法：** after raw depth 已保存；一些更早 diagnostic run 还生成 after RGB-D structure/point-cloud artifacts。
- **是否真正解决：** **否。** 当前自动 evaluator 没有消费保存的 after depth，也没有 peak-lift observation。
- **对后续 research 的启示：** 评估协议必须与成功谓词对齐：要判定 hanging anchor，至少需要“闭合后/峰值 lift/释放后”之一的直接保留证据，而不仅是 post-release RGB。

# 3. Manipulation / Anchor-Search Failures

## 3.1 完整执行后的 no-op：miss、slip、错误接触或 taut-layer 无法区分

- **Problem：** 命令和机器人状态全部成功，但 garment before/after 无可见变化；系统不能说明是空夹、滑脱、抓得太浅/太高、TCP-to-cloth 关联错误，还是只碰到一层绷紧面。
- **Concrete evidence：**
  - **FACT (saved record)：** `111708` 与 `164015` 都为 `physical_execution=true`、`execution_completed=true`、`robot_errors=[]`，所有实际 action success。
  - **FACT (manual image audit)：** 两次 A/B before/after 均实质近同。
  - **FACT (saved record)：** 两次 close 的 gripper `position_result` 均到 0；这个结果只说明执行器到闭合端，不能区分空夹与压缩的薄 cloth。
- **对应 run / 文件 / 日志路径：**
  `runs/auto_20260815_111708/results/auto_exploration/20260815T031708538205Z/iteration_001/execution.json`、`iteration_001_after/camera_0_A.png`、`camera_1_B.png`；
  `runs/auto_20260815_164015/results/auto_exploration/20260815T084015856468Z/iteration_001/execution.json` 及相邻 before/after artifacts。
- **Claude 当时认为发生了什么：** 两个 planner 都预期 pinching fold/ridge 后 peel/reposition；evaluator 正确判断“无明显变化”，并列出 miss 或 slip。
- **实际 before/after 显示了什么：** 没有 flap 翻开、没有新表面、没有整体平移；但视觉终态不能提供失败发生在哪个时刻。
- **当前最可能的问题来源：** `unknown`；候选包括 `perception`、`coordinate grounding`、`grasp geometry`。
- **已经尝试过的解决办法：** staged descent、surface_z 下方小 bite、runtime bounds/IK checks、fresh perception 后改抓 crisp/free edge 的建议。
- **是否真正解决：** **否。**
- **对后续 research 的启示：** “actual pose 到位 + gripper command 0”不应被编码为 successful grasp。需要独立的 contact/retention state，才能把 perception error 与 finger/cloth mechanics 分开。

## 3.2 Multi-layer grasp / whole-item translation

- **Problem：** 抓取穿过多层或作用在内部连通结构上，导致整个 garment/堆叠近似刚体平移，而非剥离一层或形成悬挂。
- **Concrete evidence：**
  - **FACT (saved record)：** 32 条合并记录中有 8 条 `observed_change` 明确描述 whole-item/rigid-body translation。
  - 强证据包括 `runs/claude_auto_01/results/auto_exploration/20260813T050343475098Z/iteration_{001,002}.json`；`runs/claude_auto_20260812_195401/results/auto_exploration/20260812T115402167618Z/iteration_002.json`；`runs/claude_auto_20260812_202645/results/auto_exploration/20260812T122645171957Z/iteration_{002,003}.json`；`runs/claude_auto_20260813_153538/results/auto_exploration/20260813T073539560657Z/iteration_001.json`；`runs/claude_auto_layers_01/results/auto_exploration/20260813T101854374025Z/iteration_002.json`；`runs/semantic_structure_test_01/results/auto_exploration/20260813T115716901842Z/iteration_001.json`（其中一些条目兼有旋转、曝光或人手污染，不能精确测量位移）。
  - **FACT (saved raw response)：** `runs/diagnostic_lift_01/results/claude_auto/20260814T043648832897Z_evaluation_failed.json` 中 Claude 返回 `outcome_class=MULTI_LAYER_GRASP`，称 garment 作为 coherent mass 移动、visible area -6.6%、mound height 未下降。外层失败原因是代码 `NameError: OUTCOME_CLASSES`，不是模型没有返回判断。
- **对应 run / 文件 / 日志路径：** 上述合并 `iteration_NNN.json`；`runs/diagnostic_lift_01/results/claude_auto/20260814T043648832897Z_evaluation_failed.json`。
- **Claude 当时认为发生了什么：** action plan 通常预期 pinch top ply、短 drag 到 open table、扩大 visible area；终态 evaluator 多次改判为整体 translation。
- **实际 before/after 显示了什么：** silhouette/fold/print 随整体共同移动，未出现预期的新扁平 tongue、under-layer 或持续 footprint 扩张。
- **当前最可能的问题来源：** `partial observability` + `grasp geometry` + `action planning`。
- **已经尝试过的解决办法：** 从 interior ridge 转向 outer/free perimeter；缩小 bite、沿 edge 法向拉；使用 height/layer overlays；在 prompt 中明确 whole-translation failure mode。
- **是否真正解决：** **否。** 后续确有 positive free-edge peel，但 whole-mass motion 仍在不同 run 重现。
- **对后续 research 的启示：** anchor score 应显式惩罚“目标点与 garment bulk 双侧材料连通/高多层概率”，并用 centroid motion 与 newly visible material 分开评价，而不是只看是否发生 motion。

## 3.3 真实 cloth motion / local flap reveal，但没有 usable hanging configuration

- **Problem：** action 能移动一块真实 cloth，甚至揭示新 print/seam，却没有证据说明峰值 lift 时保持住单侧悬挂，或能支持后续 Laydown。
- **Concrete evidence：** **FACT (manual image audit)**：`200329` iteration 1 的 before A 胸前均匀深色；after A 清楚出现“en the robots!”；after B 同时出现 print 与新的 doubled stitched edge。两张 after 均为 post-release，cloth 已回到桌面，无 hanging column，整体 footprint 基本不变。
- **对应 run / 文件 / 日志路径：**
  `runs/auto_20260815_200329/results/perception/center_20260815T120335523061Z/camera_0_A.png`、`camera_1_B.png`；
  `runs/auto_20260815_200329/results/auto_exploration/20260815T120330204215Z/iteration_001_after/camera_0_A.png`、`camera_1_B.png`、`iteration_001.json`。
- **Claude 当时认为发生了什么：** planner 把 raised R026 ridge 当作 anchor test；evaluator 正确识别 real cloth/local flap displacement，并推荐抓新露出的 seamed doubled edge。
- **实际 before/after 显示了什么：** 新表面与 seam 是强事实；“ridge 保持到 205 mm”“形成过 hanging column”均没有图像证据。
- **当前最可能的问题来源：** `partial observability` + `grasp geometry`；该动作作为 information-gain action 是正例，不应全部归为失败。
- **已经尝试过的解决办法：** iteration 2 基于新 seam 生成 committed Laydown proposal，并通过 preflight/controller IK。
- **是否真正解决：** **否。** iteration 2 未执行，不能知道新 seam 是否真是 usable anchor。
- **对后续 research 的启示：** 必须把两个 reward 分开：`information gain / surface reveal` 与 `anchor retention / task progress`。print reveal 可以证明 material exposure，不证明 grasp quality。

## 3.4 同类 ridge 外观产生不同 physical outcome

- **Problem：** 当前视觉描述把多个结构都归为 raised ridge/lip/steep flank，但其物理 response 不同，说明 observation aliasing 严重。
- **Concrete evidence：** **FACT (saved records + manual image audit)**：
  - `111708`：near no-op；
  - `164015`：near no-op；
  - `193710` iteration 1：partial drag/smoothing；
  - `193710` iteration 2：tent/redistribution，无 hang；
  - `200329` iteration 1：局部 flap displacement，揭示 print/seam；
  - `201533` iteration 1：轻微局部/整体 shift，未评估。
- **对应 run / 文件 / 日志路径：** `runs/auto_20260815_111708/results/auto_exploration/20260815T031708538205Z/`；`runs/auto_20260815_164015/results/auto_exploration/20260815T084015856468Z/`；`runs/auto_20260815_193710/results/auto_exploration/20260815T113711203001Z/`；`runs/auto_20260815_200329/results/auto_exploration/20260815T120330204215Z/`；`runs/auto_20260815_201533/results/auto_exploration/20260815T121534353225Z/` 中对应 proposal、execution、before/after PNG。
- **Claude 当时认为发生了什么：** 不同计划都以 ridge 的 height、窄度、gradient flank 或 shadow 推断可夹层数与 hang 概率。
- **实际 before/after 显示了什么：** 一个可见几何类别映射到至少四种终态，且现有终态图还不能区分 peak-lift retention。
- **当前最可能的问题来源：** `perception` + `partial observability`。
- **已经尝试过的解决办法：** 让动作本身变成“tent vs hang”的主动实验，记住 previous outcomes，避免重复已失败的 R 点。
- **是否真正解决：** **否，但主动探测方向有信息价值。**
- **对后续 research 的启示：** 研究目标不应只是更精细地检测 ridge，而应学习/估计 ridge 的 material role，并把 action-conditioned response 纳入 state。

## 3.5 Laydown 的有效性尚未成立

- **Problem：** 当前 `LAYDOWN_SKILL` 只提供 quasi-static 流程文字，不提供固定坐标，也没有可信的、带 peak-lift 与 post-laydown geometry 的闭环验证集。
- **Concrete evidence：**
  - **FACT (code)：** `cloth_agent/skills.py:23-42` 只说明 hang、retreat、gradual descent、low release；坐标全部由 Claude 选择。
  - **FACT (saved record + manual image audit)：** `192822` 的合并 judgement 宣称 reduced-scale Laydown success，但把 before-B/after-B 遮挡关系说反。
  - **FACT (saved artifacts)：** `200329` iteration 2 计划抓 seamed doubled edge 后 Laydown，未生成 execution；`201533` proposal 显式调用 Laydown且 execution completed，但没有 evaluator/合并 record。
- **对应 run / 文件 / 日志路径：** `runs/auto_20260815_192822/results/auto_exploration/20260815T112822886014Z/iteration_001.json`；`runs/auto_20260815_200329/results/auto_exploration/20260815T120330204215Z/iteration_002/`；`runs/auto_20260815_201533/results/auto_exploration/20260815T121534353225Z/iteration_001/{proposal,execution}.json`。
- **Claude 当时认为发生了什么：** `192822` 认为 garment 从被 fixture 遮挡变为清楚、平铺；`201533` 预期 partial laydown，但没有保存 judgement。
- **实际 before/after 显示了什么：** `192822` 的人工核对与该叙述相反；`201533` 只可见轻微 shift，缺少 peak-lift 和 after height artifacts。
- **当前最可能的问题来源：** `evaluation` + `partial observability` + `action planning`。
- **已经尝试过的解决办法：** procedural skill、low controlled release、逐步下降、prompt 中仅在 believed useful anchor 时调用；当前 evaluator prompt 又进一步禁止仅因“flat/easy”停止。
- **是否真正解决：** **UNKNOWN / INSUFFICIENT EVIDENCE。**
- **对后续 research 的启示：** 先定义可观察的 Laydown success：确认 grasp retention、单侧 hang、deposition 后 material coverage/height/topology 改善；否则“执行了 Laydown waypoint”与“Laydown 成功”不能同义。

## 3.6 已实际出现的基础设施失败，但不能解释多数已完成物理失败

- **Problem：** perception disagreement、controller IK、physical motion 和 evaluator/schema 都曾使 loop 在得到合并 judgement 前失败。
- **Concrete evidence：**
  - **FACT (saved record)：** `runs/claude_auto_01/results/auto_exploration/20260812T142754733751Z/iteration_005_failed.json` 因同一 Molmo 点 A/B depth 相差 63.3 mm，超过 50 mm gate 而停止。
  - **FACT (saved record)：** `claude_auto_20260812_191028` iteration 2 与 `claude_auto_20260812_192847` iteration 1 均为 controller IK `code=10`。
  - **FACT (saved record)：** `auto_20260815_162746` iteration 1 出现 physical `set_position failed, code=-9`。
  - **FACT (saved record)：** diagnostic evaluation 返回了合法-looking `MULTI_LAYER_GRASP` 内容，但外层因 `NameError: OUTCOME_CLASSES` 失败；另有 schema/evaluation diagnostic failures。
- **对应 run / 文件 / 日志路径：**
  `runs/claude_auto_01/results/auto_exploration/20260812T142754733751Z/iteration_005_failed.json`；
  `runs/claude_auto_20260812_191028/results/auto_exploration/20260812T111028946809Z/iteration_002_failed.json`；
  `runs/claude_auto_20260812_192847/results/auto_exploration/20260812T112847815069Z/iteration_001_failed.json`；
  `runs/auto_20260815_162746/results/auto_exploration/20260815T082747158986Z/iteration_001_failed.json`；
  `runs/diagnostic_lift_01/results/claude_auto/20260814T043648832897Z_evaluation_failed.json`。
- **Claude 当时认为发生了什么：** 其中一些 run 在 proposal/evaluation 已形成物理解释，但 infrastructure gate/代码异常阻止合并。
- **实际 before/after 显示了什么：** 每个 failed artifact 的证据完整度不同，不能统一当作物理 outcome；diagnostic lift 的 RGB-D 对 multi-layer motion 有支持，但其 outer loop 仍是 failed。
- **当前最可能的问题来源：** `perception` / `coordinate grounding` / `action planning` / `evaluation`，依具体 failure 而定。
- **已经尝试过的解决办法：** view-disagreement threshold、controller-side IK、physical error stop、schema validation。
- **是否真正解决：** **部分。** Gate 能阻止某些危险/不一致 action；代码/schema failures 说明实验记录链仍可能中断。
- **对后续 research 的启示：** infrastructure failure 应单独计数；不能用它解释 `111708`、`164015` 等 32 个“完整执行且无 robot error”的 perception/manipulation failures。

# 4. Partial Observation / Hidden-State Problems

## 4.1 单张/双张 RGB-D 都没有给出上下层材料连通关系

- **Problem：** 可见的 top surface、外轮廓和 depth discontinuity 无法回答“这条 RGB edge 是材料自由边还是遮挡边”“ridge 两侧是否属于同一连续层”“下面还有几层”“哪一块会随抓点一起移动”。
- **Concrete evidence：**
  - **FACT (manual image audit)：** `193710` iteration 2 的目标 ridge 被抬后出现双侧仍接触桌面的 tent-like state；`200329` iteration 1 则使局部 flap/print 露出。before 的 height/gradient 表达并未提供能事先区分这两者的材料连接标签。
  - **FACT (code)：** 当前 garment mask 是最大 appearance-different XY component；height-gradient 是平滑 surface gradient。没有 mesh/material ID、layer index、visible-to-hidden edge state 或 underside observation。
  - **FACT (engineering note, not experiment)：** 当前 `DINOV3_CORRESPONDENCE.md:250-263` 明确说 sparse points 不是 verified intrinsic partition，visible adjacency 只是 conservative proxy；相似的 touching layers 仍可能看似连续。
- **对应 run / 文件 / 日志路径：** `runs/auto_20260815_193710/results/auto_exploration/20260815T113711203001Z/iteration_002.json`；`runs/auto_20260815_200329/results/auto_exploration/20260815T120330204215Z/iteration_001.json`；`cloth_agent/perception.py:1088-1145,2090-2233`；`DINOV3_CORRESPONDENCE.md:250-263`。
- **Claude 当时认为发生了什么：** 常以“possible flap/free edge/interior ridge”列候选，但动作几何仍必须押注其中一种 hidden topology。
- **实际 before/after 显示了什么：** 某些 ridge 是双侧牵连的内部结构，某些能翻出局部表面；终态不能补回动作前隐藏的完整 connectivity。
- **当前最可能的问题来源：** `partial observability` + `perception`。
- **已经尝试过的解决办法：** 双视图 RGB-D、boundary/gradient overlays、语义 abstention、用主动 pinch-lift 的 tent-vs-hang response 更新下一步。
- **是否真正解决：** **否。**
- **对后续 research 的启示：** state 应允许“多个拓扑假设 + unknown edge”，而不是强制一张单一 layer map。动作可用于消除假设，但 evaluation 必须观察动作中间态。

## 4.2 峰值 lift、接触和 slip 时刻不可见

- **Problem：** 当前自动记录只在动作前与全部 release 后拍 perception；关键的 close 后、peak lift 和 transport 中间态没有同步 visual/force evidence。
- **Concrete evidence：**
  - **FACT (code)：** `cloth_agent/auto_exploration.py:1359-1384` 在动作完成并 settle 后才 capture；after evaluator 输入只有 post-release RGB。
  - **FACT (saved records)：** `111708`、`164015` 的终态无变化，但 execution 不能说明 close 是否夹到布；`200329` 终态有 local reveal，却不能说明 cloth 是否保持到 205 mm。
  - **FACT (saved record)：** gripper `position_result` 在 0、1、7、24 等低值出现过，但当前记录没有经验证的“该值对应夹持几层 cloth”的映射。
- **对应 run / 文件 / 日志路径：** `111708`、`164015`、`200329`、`201533` 的 `execution.json`；`cloth_agent/auto_exploration.py:555-568,1359-1384`。
- **Claude 当时认为发生了什么：** expected observation 把 “at z=170/205 mm hangs”作为判别点；evaluator 却只能看释放后的两帧，因而只能写“cannot verify retention”。
- **实际 before/after 显示了什么：** 只能看到 action 的净终态，不能定位 miss、initial catch、mid-lift slip 或 release-settle。
- **当前最可能的问题来源：** `partial observability`。
- **已经尝试过的解决办法：** actual pose、gripper position、robot error 全量落盘；动作使用慢速 staged lift；但没有把这些量标定成 cloth contact/retention。
- **是否真正解决：** **否。**
- **对后续 research 的启示：** good anchor 的最低可观测定义需要 action-phase observation。只保存终态会把本质不同的 miss、slip、tent、local lift 压成同一个“变化小”类别。

## 4.3 Robot/fixture 自遮挡会破坏跨视角比较

- **Problem：** arm、gripper 和顶部 fixture 在 before/after 的位置不同，直接改变 garment 可见面积、边界和局部 fold visibility。
- **Concrete evidence：**
  - **FACT (manual image audit)：** `192822` before-B 几乎整幅是清楚 cloth；after-B 左/中部被 QR-labelled fixture 和 jaws 大幅遮挡。保存 evaluator 把顺序说反。
  - **FACT (saved record + manual image audit)：** `193710` iteration 1 evaluator caveat 记录 after-B 左半被 robot body 遮挡；iteration 2 的 gripper 占据 B 中心，正好遮住要判断 retention 的区域。
- **对应 run / 文件 / 日志路径：**
  `runs/auto_20260815_192822/results/perception/center_20260815T112828154774Z/camera_1_B.png`；
  `runs/auto_20260815_192822/results/auto_exploration/20260815T112822886014Z/iteration_001_after/camera_1_B.png`；
  `runs/auto_20260815_193710/results/auto_exploration/20260815T113711203001Z/iteration_{001,002}.json`。
- **Claude 当时认为发生了什么：** `192822` 认为 garment 从 gripper 遮挡中移出；`193710` 对被遮区域给出 flatten/tent 解释但同时列 caveat。
- **实际 before/after 显示了什么：** 可见区域变化的一大部分是 robot pose 改变；在 `192822` 中甚至足以触发完全相反的叙述。
- **当前最可能的问题来源：** `partial observability` + `evaluation`。
- **已经尝试过的解决办法：** 某些动作最后 home 或 mandatory return home；evaluator caveat 被要求报告遮挡。
- **是否真正解决：** **否。** `193710` 未总是先清空两相机视野再拍 after，且 caveat 不能修复缺失像素。
- **对后续 research 的启示：** before/after 比较应使用相同 robot-clear observation pose，或显式 robot mask/visibility mask；不可用“可见 cloth 增多”直接当 surface reveal。

## 4.4 人手进入场景会污染因果归因，并暴露安全/同步风险

- **Problem：** 保存帧出现人手接触 garment，系统却可能把手造成的遮挡/平整变化归因给机器人，甚至颠倒人手出现时序。
- **Concrete evidence：**
  - **FACT (manual image audit)：** `192145` before-A/B 无人手；after-A/B 清楚有人手压在 garment 左下/中部。
  - **FACT (saved record)：** 该次 evaluator 写成“before 有手、after 无手”，并据此担忧计划抓点靠近人。
  - **FACT (saved record)：** 更早 `claude_auto_20260812_202645` iteration 3 的 `observed_change` 正确记录 after Camera A 出现 hand/forearm，因此人手污染不是单次孤例。
- **对应 run / 文件 / 日志路径：**
  `runs/auto_20260815_192145/results/perception/center_20260815T112151223615Z/camera_0_A.png`、`camera_1_B.png`；
  `runs/auto_20260815_192145/results/auto_exploration/20260815T112145901976Z/iteration_001_after/camera_0_A.png`、`camera_1_B.png`、`iteration_001.json`；
  `runs/claude_auto_20260812_202645/results/auto_exploration/20260812T122645171957Z/iteration_003.json`。
- **Claude 当时认为发生了什么：** `192145` 认为 plan 基于含手 before frame，动作结果被“手移走”混淆。
- **实际 before/after 显示了什么：** 恰好相反：手只在 after 出现。无论动作前后，scene contamination 都使机器人因果效果不可判。
- **当前最可能的问题来源：** `evaluation` + `partial observability`；人手为何进入、capture bundle 是否有同步问题为 `unknown`。
- **已经尝试过的解决办法：** evaluator 建议重新采集 clean scene；当前 prompt 强调不可靠观察应停止。
- **是否真正解决：** **否。** 没有找到自动 human-presence gate 或同一时刻 RGB/depth bundle 的已验证防污染记录。
- **对后续 research 的启示：** human/foreign-object detection 应是评估有效性与执行安全的前置条件；检测到污染时，该 transition 应标为 invalid，而不是 useful/not useful。

# 5. Coordinate / Multi-view / Depth Problems

## 5.1 旧流程围绕单一验证点操作，图像方向到 base 方向未验证

- **Problem：** 早期 run 没有 dense pixel-to-base grounding；Claude 只能围绕 Molmo/center point 加 offset，并凭图像方向猜 base XY 拉动方向与 yaw。
- **Concrete evidence：**
  - **FACT (saved proposal)：** `111708` 明写“Only one validated 3D anchor exists”且“cannot map any image pixel to base coordinates”；`164015` 也只信任 center surface point。
  - **FACT (saved evaluation/history)：** `claude_auto_20260812_210824` iteration 3 承认 image-to-base direction unverified，实际新 coverage 出现在不同 flank，semantic “hood/collar”也只是推断。
- **对应 run / 文件 / 日志路径：** `runs/auto_20260815_111708/results/auto_exploration/20260815T031708538205Z/iteration_001.json`；`runs/auto_20260815_164015/results/auto_exploration/20260815T084015856468Z/iteration_001.json`；`runs/claude_auto_20260812_210824/results/auto_exploration/20260812T130824848032Z/iteration_003.json`。
- **Claude 当时认为发生了什么：** 以 center/单点为 metric anchor，再按 ±x/±y 设计 peel。
- **实际 before/after 显示了什么：** no-op 与方向不符都出现过；无法判定是抓点错、方向错、z 错还是 cloth mechanics。
- **当前最可能的问题来源：** `coordinate grounding`。
- **已经尝试过的解决办法：** 当前保存 full-resolution per-camera base-XYZ map，并以均匀 Rxxx overlay 供 Claude 自选视觉结构后落到测量参考。
- **是否真正解决：** **旧问题显著缓解，但未完全解决。**
- **对后续 research 的启示：** 应把“commanded base coordinate 有来源”与“该 coordinate 对应预期 material feature”分开审计。

## 5.2 当前 48 px Rxxx guide 稀疏、每次重编号，窄边仍会跨视图不一致

- **Problem：** overlay 每 48 px 只放一个 garment-valid reference；窄 lip/occlusion edge 可能落在采样之间。R ID 只是当帧索引，历史中同名 R 并无语义稳定性。
- **Concrete evidence：**
  - **FACT (code)：** `_save_camera_coordinate_guide` 的默认 `sample_stride_px=48`，位于 `cloth_agent/perception.py:1346-1453`；full-resolution `.npy` 存在，但 planner 主要文本引用最近 R marker。
  - **FACT (saved proposal)：** `201533` 的 B R088 报告 lip 约 20.1 mm above table，A 最近稀疏点只有约 5.3 与 -1.1 mm；Claude 给 crest 约 ±10 mm uncertainty。
  - **FACT (saved proposal)：** `193710` iteration 2 明写 previous R015/R016 不是 current R015/R016。
- **对应 run / 文件 / 日志路径：** `runs/auto_20260815_201533/results/perception/center_20260815T121539610518Z/camera_{A,B}_coordinate_guide.json`；`runs/auto_20260815_201533/results/auto_exploration/20260815T121534353225Z/iteration_001/proposal.json`；`runs/auto_20260815_193710/results/auto_exploration/20260815T113711203001Z/iteration_002/proposal.json`。
- **Claude 当时认为发生了什么：** 用 nearest measured R point grounding，并在点间插值推断 lip/edge。
- **实际 before/after 显示了什么：** 当前保存资料能证明 command 有明确测量来源，但不能保证最近 R 点处于同一窄材料结构；A/B 也可能测到结构不同侧或受遮挡。
- **当前最可能的问题来源：** `coordinate grounding` + `perception`。
- **已经尝试过的解决办法：** full XYZ map、R marker 明确标为 unranked reference、prompt 要求报告空间不确定度并避免复用旧 R ID。
- **是否真正解决：** **部分。**
- **对后续 research 的启示：** 对 narrow edge 应保存 target pixel、局部邻域 surface/visibility 和跨视图 association，而非只记一个会重编号的最近 R ID。

## 5.3 A/B depth、table plane 与 shared-view alignment 仍不稳定

- **Problem：** Camera A depth 时序噪声较大；不同 capture 的 table plane 可移动数毫米；跨相机 shared surface 对齐有时有长尾，且“shared-view agreement”不等于 absolute robot-base validation。
- **Concrete evidence：**
  - **FACT (saved record)：** `runs/dual_depth_test_20260813T090706615329Z/report.json`：A simultaneous jitter median/p95 = 2.21/16.17 mm，B = 0.73/2.75 mm；A garment p50/p95 height = 79.27/134.32 mm，B = 25.11/55.20 mm；symmetric alignment median/p95 = 5.38/106.70 mm，74.17% within 20 mm。
  - **FACT (saved record)：** `runs/dual_depth_test_20260813T091243909107Z/report.json` 改善到 symmetric median/p95 = 4.55/15.09 mm、97.16% within 20 mm；A jitter仍为 3.59/7.34 mm，B 为 0.73/2.82 mm。
  - **FACT (saved proposal)：** `193710` iteration 2 记录同一位置 absolute z 仍为约 32.38 mm，但 `height_above_table` 下降约 4.7 mm，主要因为 fitted plane 移动；该 capture table residual p95 约 4.76 mm。
- **对应 run / 文件 / 日志路径：** `runs/dual_depth_test_20260813T090706615329Z/report.json`；`runs/dual_depth_test_20260813T091243909107Z/report.json`；`runs/auto_20260815_193710/results/auto_exploration/20260815T113711203001Z/iteration_002/proposal.json` 及该 iteration 的 perception result。
- **Claude 当时认为发生了什么：** 计划用 mm-level ridge height、table clearance 和 A/B confirmation 选择 grasp z。
- **实际 before/after 显示了什么：** mm 数值足以提供安全近似与粗几何，但数毫米 plane drift、窄结构 sampling 和偶发长尾会改变“是否高于桌面/抓多深”的判断。
- **当前最可能的问题来源：** `perception` + `coordinate grounding`。
- **已经尝试过的解决办法：** temporal median、A/B voxel fusion、corner/edge table references、shared absolute table-zero scale、cross-view disagreement gate（50 mm）。
- **是否真正解决：** **部分。** 第二次静态 test 明显改善，但真实 run 的 local disagreement 和 plane drift 仍在。
- **对后续 research 的启示：** target 应带 per-view/local uncertainty；“A/B 都有点”不等于两点属于同一 material patch。抓取 z 应对 table-plane uncertainty 与 cloth-surface uncertainty分别建模。

## 5.4 可达边界与材料自由边不重合

- **Problem：** 最有希望的 visible outer lip 有时接近/越过 workspace limit，迫使 planner 改抓内部 ridge，改变了实验问题本身。
- **Concrete evidence：** **FACT (saved proposal)**：`193710` iteration 2 的 outer-left references 约为 y=-285 到 -320 mm，而 `y_min=-303.571`；R017=-298.0、R024=-311.3、R033=-319.9，planner 明确拒绝更外侧 lip，转向内部 R026 ridge。
- **对应 run / 文件 / 日志路径：** `runs/auto_20260815_193710/results/auto_exploration/20260815T113711203001Z/iteration_002/proposal.json`。
- **Claude 当时认为发生了什么：** 认为 outer lip 更像 boundary，但重复旧点或越界都不安全，于是执行预先约定的 internal-ridge fallback。
- **实际 before/after 显示了什么：** fallback 得到 tent/redistribution，无单侧 hang。
- **当前最可能的问题来源：** `action planning` + `coordinate grounding`；这是 workspace/garment placement 的耦合，不是纯 perception error。
- **已经尝试过的解决办法：** bounds margin、controller IK gate、拒绝 unsafe interpolation。
- **是否真正解决：** **安全上解决了越界执行，研究目标上没有解决 anchor 可达性。**
- **对后续 research 的启示：** 候选 anchor 的评价必须包含可达性；不能把“安全 fallback 的结果”当作对最佳 visible boundary 的直接反证。

## 5.5 正面证据：命令保存与 controller arrival 可以非常准确

- **Problem：** 需要判断失败是否由 Claude 坐标在 proposal→script→controller→actual pose 链路中被改写。
- **Concrete evidence：** **FACT (saved diagnostic)**：Claude 要求 close at `(600,30,42)`；actual xArm pose 为约 `(600.000,29.999994,42.000076)`。同一 CamA 标定下两者回投误差 `0.000011947 px`；preflight/execution action audit 全部 match；actual-pose audit 在 2 mm / 1° tolerance 内通过。
- **对应 run / 文件 / 日志路径：** `runs/diagnostic_lift_01/results/auto_exploration/20260814T043002999078Z/iteration_001/{grasp_point_comparison,command_projection,coordinate_audit_preflight,coordinate_audit_execution,coordinate_audit_actual_pose,execution}.json`。
- **Claude 当时认为发生了什么：** 计划 pinching raised upper layer 后短距离 peel/place。
- **实际 before/after 显示了什么：** 原始 evaluator 返回将结果判为 multi-layer coherent-mass translation；控制命令精确到位没有带来正确物理 outcome。
- **当前最可能的问题来源：** 对该次失败，`command corruption/control arrival` 的解释不受证据支持；更可能是 `perception` / `grasp geometry` / `partial observability`。绝对外参误差仍为 `UNKNOWN`。
- **已经尝试过的解决办法：** restricted action script、preflight、controller IK、command/action/actual pose audits。
- **是否真正解决：** **命令保存链路在该诊断上已验证；绝对视觉-物理对应和 cloth contact 未验证。**
- **对后续 research 的启示：** 后续分析可优先研究“选中了什么材料结构、以何种 bite 接触”，但不能把单次同标定回投误差外推成所有相机外参绝对正确。

# 6. Claude Reasoning Failure Patterns

## 6.1 两次可直接证伪的 before/after 时序颠倒

### 6.1.1 `auto_20260815_192145`：人手时序反转

- **Problem：** evaluator 把 after 才出现的人手写成 before 有、after 无，并围绕错误事实构建安全与因果解释。
- **Concrete evidence：** **FACT (manual image audit)**：before A/B 无手，after A/B 有手；**FACT (saved record)**：`observed_change` 和 `reason` 写反。
- **对应 run / 文件 / 日志路径：** `runs/auto_20260815_192145/results/perception/center_20260815T112151223615Z/camera_{0_A,1_B}.png`；`runs/auto_20260815_192145/results/auto_exploration/20260815T112145901976Z/iteration_001_after/camera_{0_A,1_B}.png`；`iteration_001.json`。
- **Claude 当时认为发生了什么：** hand removal/manual smoothing 混淆 robot result，且计划可能对人危险。
- **实际 before/after 显示了什么：** hand addition/接触发生在 after capture；动作效果仍被人手污染，但危险时序和因果叙述完全不同。
- **当前最可能的问题来源：** `evaluation`。
- **已经尝试过的解决办法：** evaluator 输出 caveat、建议 clean re-capture；当前 prompt 对 unreliable evidence 允许 stop。
- **是否真正解决：** **否。**
- **对后续 research 的启示：** evaluator 首先应做机械的 image-role/scene-occupancy consistency check，再做 causal reasoning；高层 caveat 不能补救底层时序读取错误。

### 6.1.2 `auto_20260815_192822`：robot occlusion 时序反转

- **Problem：** evaluator 把清楚的 before-B 写成几乎被 gripper 遮挡，把实际被硬件遮挡的 after-B 写成清楚 cloth，并以此宣布 Laydown success。
- **Concrete evidence：** **FACT (manual image audit)**：before-B 为大面积 garment；after-B 的 QR fixture/jaws 覆盖左/中部。before-A 本身已呈可读 shirt silhouette。
- **对应 run / 文件 / 日志路径：** `runs/auto_20260815_192822/results/perception/center_20260815T112828154774Z/camera_{0_A,1_B}.png`；`runs/auto_20260815_192822/results/auto_exploration/20260815T112822886014Z/iteration_001_after/camera_{0_A,1_B}.png`；`iteration_001.json`。
- **Claude 当时认为发生了什么：** garment 从 fixture occlusion 中被移出，变成 single dominant layer，successful anchor test + reduced-scale Laydown，因此 `useful=true, stop=true`。
- **实际 before/after 显示了什么：** 主要遮挡变化方向相反；post-release 图没有 hanging panel，layer-count reduction 无 after depth 支持。
- **当前最可能的问题来源：** `evaluation`。
- **已经尝试过的解决办法：** 该保存 prompt 要求 caveat；当前代码 `cloth_agent/auto_exploration.py:300-315` 后来进一步禁止只因 flat/recognizable/easy 就 stop。
- **是否真正解决：** **该 judgement 无效；当前 prompt 修订是否能防止同类视觉颠倒为 UNKNOWN。**
- **对后续 research 的启示：** 这次不能作为 Laydown 正例。需要把 camera identity、before/after label 与 robot visibility 作为结构化 evaluator 输入/自动检查。

## 6.2 无物理证据的 semantic part inference

- **Problem：** Claude 从轮廓/开口推断 collar、shoulder、hood、sleeve/hem，即使相应结构在 severe folds 下可能只是 overlap 或另一开口。
- **Concrete evidence：**
  - **FACT (saved record)：** `193710` iteration 2 一方面称新露出“collar/shoulder boundary”，另一方面 caveat 明写可能是 sleeve opening 或 overlapped hem。
  - **FACT (saved record)：** `claude_auto_20260812_210824` iteration 3 使用 “hood/collar roll”，但没有 semantic ground truth，且 outcome flank 与预测不一致。
  - semantic landmark baseline 本身只有 2–6/10 visible。
- **对应 run / 文件 / 日志路径：** `runs/auto_20260815_193710/results/auto_exploration/20260815T113711203001Z/iteration_002.json`；`runs/claude_auto_20260812_210824/results/auto_exploration/20260812T130824848032Z/iteration_003.json`；三个 `runs/landmarks_*/landmarks_baseline.json`。
- **Claude 当时认为发生了什么：** 通过 garment-like silhouette 给新边界命名，并据此推荐下一 anchor。
- **实际 before/after 显示了什么：** “有一个新可见 boundary/opening”是可支持事实；具体 garment part identity 为 **INFERENCE**。
- **当前最可能的问题来源：** `perception` + `evaluation`。
- **已经尝试过的解决办法：** 当前 planner system prompt 禁止 mandatory semantic labels，要求使用 possible boundary/flap/uncertain structure。
- **是否真正解决：** **部分。** 规划文本更常 abstain，但 evaluator 仍会自然补全部件身份。
- **对后续 research 的启示：** downstream action 若只需要“可见 doubled/free-looking edge”，不应让未经确认的 part name 提高 confidence。

## 6.3 从“窄且高”推断“层少”，再推断“好 anchor”

- **Problem：** Claude 将 geometric narrowness/height/steep flank 解释为少层并可悬挂，跳过了材料 topology 与夹爪接触模型。
- **Concrete evidence：** **FACT (saved proposal)**：`193710` iteration 2 明写 narrow 20 mm ridge “should capture a small number of layers”，宽 plateau 像 piled roll；实际 after 是 shallow tent/redistribution。`200329` iteration 1 同样认为 steep flanks 更可能容许手指抓单层，实际只有 local reveal。
- **对应 run / 文件 / 日志路径：** `runs/auto_20260815_193710/results/auto_exploration/20260815T113711203001Z/iteration_002/{proposal,claude_evaluation}.json`；`runs/auto_20260815_200329/results/auto_exploration/20260815T120330204215Z/iteration_001/{proposal,claude_evaluation}.json`。
- **Claude 当时认为发生了什么：** ridge apex close 可捕获 roughly two layers，并以 hang vs tent 判别。
- **实际 before/after 显示了什么：** 一次 tent，一次 local flap reveal；均无 confirmed hang。
- **当前最可能的问题来源：** `perception` + `grasp geometry` + `action planning`。
- **已经尝试过的解决办法：** prompt 已明确 height/ridge 不是默认好 anchor；Claude 自己把动作降级为 cautious test。
- **是否真正解决：** **否。**
- **对后续 research 的启示：** narrow/high 可以作为“finger clearance”特征，但 layer count 与 one-sided connectivity 必须是独立不确定变量。

## 6.4 从“cloth 确实动了/可抓”推断“anchor 有用”

- **Problem：** graspability、material motion、information gain 与 useful hanging anchor 被混合。
- **Concrete evidence：**
  - `193710` iteration 1：cloth 确实随 gripper 移动，但 evaluator 只能判 partial drag、not confirmed hang。
  - `200329` iteration 1：真实表面被翻出，print/seam 出现，但无 hanging column。
  - `192822`：仅凭错误的可见性叙述就宣布 anchor test 与 Laydown 均成功。
- **对应 run / 文件 / 日志路径：** 上述三次合并 `iteration_001.json`。
- **Claude 当时认为发生了什么：** 将“engaged real cloth”视作 anchor evidence，再在部分 run 中推进/宣告 Laydown。
- **实际 before/after 显示了什么：** real cloth motion 至多证明 contact/action effect；只有单侧悬挂与可控 deposition 才直接对应 lifting anchor。
- **当前最可能的问题来源：** `evaluation` + `action planning`。
- **已经尝试过的解决办法：** 新 prompt 将 usable anchor 定义成 lift 后形成 useful hanging configuration，并要求不因 promising anchor 自动 stop。
- **是否真正解决：** **定义上改善，实证未解决。**
- **对后续 research 的启示：** outcome taxonomy 至少应拆成 contact、retention、material extent moved、one/two-sided hang、surface reveal 与 deposition quality。

## 6.5 Expected observation 与实际 outcome 不一致，且 evaluator 可能受预期锚定

- **Problem：** proposal 中的 expected observation 很详细，evaluator prompt 又原样提供 strategy/skill/expected observation；模型可能在模糊 after RGB 中“看见”预期的 collapse、part 或 Laydown。
- **Concrete evidence：**
  - **FACT (code)：** `cloth_agent/auto_exploration.py:285-320` 把 previous strategy、invoked skills、expected observation 与 before/after 路径放进同一 evaluation prompt。
  - **FACT (saved record)：** `192822` 预期 free edge→hang→reduced Laydown，evaluation 最终声明该流程成功，尽管它对 B 遮挡时序的描述可直接证伪。
  - **FACT (saved records)：** `111708`、`164015` 的 evaluator 没有强行判成功，说明 expected outcome 并非必然导致正标签。
- **对应 run / 文件 / 日志路径：** 上述代码；`runs/auto_20260815_192822/results/auto_exploration/20260815T112822886014Z/iteration_001/{proposal,claude_evaluation}.json`；`runs/auto_20260815_111708/results/auto_exploration/20260815T031708538205Z/iteration_001.json`；`runs/auto_20260815_164015/results/auto_exploration/20260815T084015856468Z/iteration_001.json`。
- **Claude 当时认为发生了什么：** 在不清楚的 post-release evidence 上完成了从 target ridge 到 successful Laydown 的因果链。
- **实际 before/after 显示了什么：** 该链条至少在 `192822` 不受图像支持；其他 run 中 evaluator 也能拒绝 expected outcome。
- **当前最可能的问题来源：** `evaluation`。
- **已经尝试过的解决办法：** caveats、confidence、当前更严格 stop rule。
- **是否真正解决：** **UNKNOWN / INSUFFICIENT EVIDENCE。** “prompt anchoring 导致错误”是 **INFERENCE**，现有数据没有 blinded evaluator 对照实验，不能断言因果。
- **对后续 research 的启示：** 可以对同一 transition 做 blind observation-first judgement，再单独读取 proposal 检查 prediction error；这样能区分“看到了什么”与“是否符合预期”。

# 7. Useful Empirical Findings

## 7.1 可见 free perimeter、outside-in peel、低位释放有最强的正向实证

- **Finding：** 从可见外围边界向外 peel，比重复 interior ridge 更有可能揭示新表面并扩大/变平 footprint。
- **Concrete evidence：**
  - **FACT (saved record)：** `claude_auto_20260812_210824` iteration 1 的 interior ridge pinch 只得到 near no-op/local fold relaxation。
  - **FACT (saved record + manual audit of saved sequence)：** iteration 2 改为 outside-in free-boundary peel 后，新 printed logo 与 care label 出现，silhouette 更宽、更平，深 fold 减少；iteration 3 又部分 unroll 一处 raised region 并增宽 footprint。
- **路径：** `runs/claude_auto_20260812_210824/results/auto_exploration/20260812T130824848032Z/iteration_{001,002,003}.json` 及相邻 before/after RGB、point-cloud/area artifacts。
- **解释边界：** 这是目前最强的 action-pattern 正证据，不是“所有可见 edge 都是 material free edge”的证明。iteration 3 的语义“hood/collar”未验证，新增 coverage 出现在与预测不同的 flank，也没有 verified full lay-open。
- **Research implication：** perimeter evidence、outward direction 与 low deposition 值得保留为先验；应以真实 surface reveal 和非刚体 footprint change 验证，而非部件名称。

## 7.2 Print/logo/label/seam 是强 observation evidence

- **Finding：** 原来不可见的 print/logo/care label/seam 在 after 中出现，能直接证明某块 material surface 的可见性发生变化。
- **Concrete evidence：**
  - `200329` iteration 1：新露出“en the robots!”与 doubled stitched edge。
  - `210824` iteration 2：新 logo 与 care label。
  - 32 条合并记录中，11 条 `useful=true` 有 9 条 `observed_change` 提到新 print/logo/lettering。
- **解释边界：** 文字/缝线出现是 surface reveal 的强证据；它不自动证明 visible area 变大、层数减少、夹持保持到 peak lift 或 anchor 可用于 Laydown。
- **Research implication：** 可把这些高辨识度 cue 用作 material reappearance/observation gain 的稀疏 ground truth，但 anchor reward 需另算。

## 7.3 Height map / gradient 对“可见几何”有用

- **Finding：** 即使不能判拓扑，height-above-table、gradient edge 与 boundary 仍能定位 raised band、低平区、steep transition 和 table clearance。
- **Concrete evidence：** `193710` iteration 2 用当前绝对 z 与移动的 fitted plane 区分“物理点是否下降”；`200329` proposal 能避开全局 fixture maximum，选择局部 R026；`201533` 明确报告 A/B disagreement 和 ±10 mm uncertainty。
- **解释边界：** 这些是 geometry/uncertainty evidence，不是 layer label。
- **Research implication：** 后续表示不必丢掉 height/gradient；应把它们作为材料拓扑假设和主动探测的观测项，而非最终 anchor classifier。

## 7.4 坐标/执行链具有可审计的强正证据

- **Finding：** 在 diagnostic run 中，Claude action 未被 template 替换，preflight、execution、actual pose 与回投都一致。
- **Concrete evidence：** `coordinate_audit_preflight.json`、`coordinate_audit_execution.json` 无 differences；`coordinate_audit_actual_pose.json` 在 2 mm/1° 内 match；close-point same-calibration reprojection error `1.1947e-5 px`。
- **路径：** `runs/diagnostic_lift_01/results/auto_exploration/20260814T043002999078Z/iteration_001/`。
- **解释边界：** 证明 command preservation/controller arrival，不证明 absolute extrinsics、cloth contact 或 semantic target correctness。
- **Research implication：** 这套 audit 可以继续作为每个失败的排除项，让 perception/state hypothesis 不与 command corruption 混在一起。

## 7.5 Camera B 在保存的静态 depth tests 中更稳定

- **Finding：** 两次 dual-depth test 中 B 的 jitter 约 0.73 mm median、2.75–2.82 mm p95，并且 garment p50/p95 height 从 25.11/55.20 到 25.04/55.30 mm 基本稳定；A 同期更噪。
- **路径：** `runs/dual_depth_test_20260813T090706615329Z/report.json`、`runs/dual_depth_test_20260813T091243909107Z/report.json`。
- **解释边界：** 这是这两次静态场景的 temporal stability，不证明 B 在所有 robot pose/occlusion/cloth geometry 下更准确，也不提供 absolute base-frame ground truth。
- **Research implication：** 可以用 B 作为 local temporal consistency 参考，但跨视图 material association 仍需独立验证。

## 7.6 当前 prompt 已吸收部分真实失败经验

- **Finding：** 现有 planner prompt 明确禁止把 center、highest point、fold convergence 或 most occluded region 默认当 good anchor；要求 semantic abstention、nearest measured reference、uncertainty、previous outcomes 和 explicit waypoints。
- **路径：** `cloth_agent/free_exploration.py:409-444,564-639`。
- **解释边界：** 这是工程修正的事实，不是问题已解决的实验证据；`192145`/`192822` 说明语言约束本身不能保证 evaluator 读图正确。
- **Research implication：** 继续保存 prompt version 与 transition evidence，才能区分 heuristic 更新与真实成功率变化。

# 8. Current Perception Assumptions That May Be Wrong

| 当前假设 | 来源 | 与之冲突或尚缺的证据 | 状态 |
|---|---|---|---|
| 与 table RGB 差异最大的最大 XY component 就是完整 garment | `perception.py:2090-2233` | fixture、clips、手、阴影/饱和可能改变 component；exposure sweep 改变 mask/height distribution | 未验证为普适 |
| corner/edge depth references 能稳定定义 table zero | `perception.py:572-760` | plane residual p95 常约 4–6 mm；同一绝对 z 的 relative height 可因 plane 移动约 4.7 mm | 部分有效，有漂移 |
| garment 内 90 分位 height-gradient 是 fold/occlusion edge | `perception.py:1088-1145` | 代码只检测 surface gradient；不辨识材料自由边、内部折痕、sensor discontinuity | 几何意义成立，拓扑意义未成立 |
| 窄而高的 ridge 比宽 plateau 层数更少 | Claude proposals | `193710` iteration 2 narrow ridge 产生 tent；`200329` 只 local reveal | 不受现有证据支持 |
| RGB 中“free-looking edge/lip”就是 material perimeter | planner reasoning | overlap edge、occlusion boundary、folded interior 都可能有同样外观 | UNKNOWN |
| 每 48 px 最近 R reference 足以给 grasp XY/z/yaw | `perception.py:1346-1453` | `201533` narrow lip 跨 A/B 高度冲突；局部峰可落在网格之间 | 部分，不足以覆盖窄结构 |
| A/B 最近点指向同一材料结构 | multi-view reasoning | first dual-depth test p95 106.7 mm；真实 narrow lip A/B 可能看结构不同侧 | 未验证 |
| surface z 减 heuristic bite 就会形成正确 finger-cloth contact | 多个 proposal | `111708`、`164015` 完整执行仍 no-op；table/crest uncertainty 与 jaw geometry 未直接观测 | 未验证 |
| gripper close position 接近 0 表示夹到 cloth | execution logs | 0 同时出现在 near no-op 与 cloth motion run；无 layer/contact calibration | 不成立为充分条件 |
| post-release 两张 RGB 足以判 hang/retention/layer count | evaluator input | after 没有 peak-lift view；最近 caveat 反复承认不可判 | 已被证据否定 |
| 新 print/logo 表示 good anchor | useful labels | `200329` 有 print reveal 但无 hanging column | 只能证明 surface reveal |
| Rxxx ID 在 history 中具有稳定语义 | planner history | 每次 capture 重新采样/编号；`193710` 明确警告旧 R≠新 R | 已被证据否定 |
| 写“uncertain/possible”能防止 semantic hallucination | prompt | `193710` evaluator 仍补成 collar/shoulder；`192822` 仍颠倒时序 | 不充分 |
| shared-view point-cloud agreement 证明 absolute robot-base calibration | dual-depth report 的潜在误读 | report 明写只测 calibrated surface agreement，不是 absolute validation | 不成立 |
| current visible adjacency 可代理 material connectivity | canonical graph engineering assumption | touching similar layers 仍可看似连续；没有 saved canonical experiment | **UNKNOWN / INSUFFICIENT EVIDENCE** |
| DINOv3 在同材质/低纹理/对称 garment 上已经失败 | 用户关注的假设 | `runs/`、`results/` 没有任何 DINO match/review output | **UNKNOWN / INSUFFICIENT EVIDENCE** |

# 9. Open Research Questions

1. **Visible material state 如何表示？** 怎样同时表示可见 surface patches、确定 adjacency、ambiguous/unknown adjacency、遮挡边与可能的 hidden continuation，而不强迫恢复唯一完整拓扑？
2. **什么 cue 真正预测 one-sided hang？** 在 RGB-D ridge、visible perimeter、underside shadow、seam、local curvature 和 active response 中，哪些能区分 free boundary、interior tent、multi-layer bundle 与 shallow wrinkle？
3. **如何在 peak lift 验证 contact、retention 与 layer extent？** 需要什么最小观测才能区分 miss、初始抓住后 slip、单/多层保持、双侧 tent 和 global lift？
4. **如何做 uncertainty-aware narrow-edge grounding？** 当 48 px sparse reference、full XYZ、A/B view 在 lip 两侧给出不同高度时，如何产生 target distribution 而非单一伪精确点？
5. **跨视图 material correspondence 如何验证？** 在 dark/low-texture、对称或同材质接触层上，DINO/其他 dense feature 与 geometry/topology 约束分别贡献什么？现阶段没有 DINO 实验结果，首先需要 ground-truth correspondence 与 ambiguity 指标。
6. **如何让 evaluation 对遮挡和污染不变？** 怎样自动检测 robot self-occlusion、viewpoint change、exposure drift、human hand contamination 和 before/after role inversion，并将 transition 标为 invalid？
7. **Laydown 的触发条件与成功谓词是什么？** 应在何种 retention/hang evidence 下调用；成功是 coverage、height、材料可见性、拓扑可辨识度还是后续 graspability 改善？
8. **如何分开信息增益与任务进展？** `200329` 说明 local print/seam reveal 很有信息，但未证明 anchor；reward/日志如何同时保存两种结果？
9. **如何把 workspace reachability 纳入 anchor search？** 当最好 outer lip 越界、fallback 只能抓 interior ridge 时，怎样避免把可达性约束混成 perception failure？
10. **如何量化 exposure/table-plane 对 state 的影响？** 同一静态 garment 的 mask、height distribution 与 boundary 在 exposure/参考选择变化下应有多稳定，什么阈值才足以支持 mm-level bite？
11. **历史 transition 如何跨 capture 对齐？** R ID 每次重排，怎样在不假设 dense correspondence 已解决的情况下保存“这里试过且 tent/no-op”的材料区域证据？
12. **控制链与绝对外参如何独立校准？** command preservation 已有强证据；还缺什么 independent fiducial/contact experiment 才能证明 camera-to-base 的绝对目标误差？

# 10. Evidence Index

## 10.1 当前代码与配置

| 主题 | 路径 / 行号 | 本清单使用的事实 |
|---|---|---|
| Claude system prompt / coordinate contract | `cloth_agent/free_exploration.py:409-444` | Rxxx 为均匀、unranked references；Claude 自选区域和 waypoint |
| Anchor-search prompt | `cloth_agent/free_exploration.py:564-639` | usable anchor 定义、禁用 center/highest/fold-convergence 默认、semantic abstention |
| Before image/artifact 收集 | `cloth_agent/free_exploration.py:642-688` | planner/evaluator before 输入含丰富 height/gradient/coordinate/fused artifacts |
| Before/after evaluator prompt | `cloth_agent/auto_exploration.py:285-320` | expected observation/skill 与图像一起输入 evaluator |
| After RGB/depth 保存函数 | `cloth_agent/auto_exploration.py:555-568` | 保存 RGB 与 depth NPY，但返回 evaluator 的路径只有 RGB |
| Before 与 after evaluation call | `cloth_agent/auto_exploration.py:1104-1118,1379-1409` | before/after evidence 不对称 |
| A/B 6 mm fusion | `cloth_agent/perception.py:426-500` | current dense point fusion |
| Table plane / reference interpolation | `cloth_agent/perception.py:503-760` | low/bright cloud fit 与 corner/edge references |
| Height-gradient edge | `cloth_agent/perception.py:1088-1145` | smoothed surface gradient、90 percentile threshold |
| Coordinate guide | `cloth_agent/perception.py:1346-1453` | full XYZ map + 48 px uniform R markers |
| Current dense mask/state | `cloth_agent/perception.py:2090-2233` | no Molmo launch；appearance + height envelope + largest component |
| Laydown skill | `cloth_agent/skills.py:23-42` | procedural guidance only，无固定坐标 |
| Static generated-script gate | `cloth_agent/experiment.py:40-194` | action whitelist、AST validation、restricted runtime |
| Controller IK/TCP gate | `cloth_agent/robot_api.py:31-146` | live TCP check 与每个 Cartesian target 的 controller IK |
| Camera A exposure | `config/perception.free_exploration.json:2-24` | Camera A fixed exposure 800、A/B active、25-frame median、50 mm disagreement limit |

## 10.2 最近真实运行：主证据

- 本节中的 `{a,b}` 是同一目录下多个实际文件的紧凑写法，不代表名为花括号的文件。
- `runs/auto_20260815_111708/results/auto_exploration/20260815T031708538205Z/iteration_001.json` 与 `runs/auto_20260815_111708/results/auto_exploration/20260815T031708538205Z/iteration_001/{proposal,execution,claude_evaluation}.json`、`runs/auto_20260815_111708/results/auto_exploration/20260815T031708538205Z/iteration_001_after/camera_{0_A,1_B}.png`。
- `runs/auto_20260815_164015/results/auto_exploration/20260815T084015856468Z/iteration_001.json` 及对应 artifacts。
- `runs/auto_20260815_192145/results/perception/center_20260815T112151223615Z/` 与 `runs/auto_20260815_192145/results/auto_exploration/20260815T112145901976Z/iteration_001.json`、`runs/auto_20260815_192145/results/auto_exploration/20260815T112145901976Z/iteration_001_after/`。
- `runs/auto_20260815_192822/results/perception/center_20260815T112828154774Z/` 与 `runs/auto_20260815_192822/results/auto_exploration/20260815T112822886014Z/iteration_001.json`、`runs/auto_20260815_192822/results/auto_exploration/20260815T112822886014Z/iteration_001_after/`。
- `runs/auto_20260815_193710/results/auto_exploration/20260815T113711203001Z/iteration_{001,002}.json` 及每次 proposal/execution/evaluation/after PNG；iteration 3 只有 perception/replan feedback。
- `runs/auto_20260815_200329/results/perception/center_20260815T120335523061Z/`；`runs/auto_20260815_200329/results/auto_exploration/20260815T120330204215Z/iteration_001.json`、`runs/auto_20260815_200329/results/auto_exploration/20260815T120330204215Z/iteration_001_after/`；`runs/auto_20260815_200329/results/auto_exploration/20260815T120330204215Z/iteration_002/` 只有 plan/preflight/IK。
- `runs/auto_20260815_201533/results/perception/center_20260815T121539610518Z/`；`runs/auto_20260815_201533/results/auto_exploration/20260815T121534353225Z/iteration_001/{proposal,execution,after_capture}.json` 与 `runs/auto_20260815_201533/results/auto_exploration/20260815T121534353225Z/iteration_001_after/`；没有合并 judgement。
- 当前 A/B Molmo 与 agent 点图：`results/current_garment_point_figures_20260817/`。主结果采用 UNKNOWN-allowed 规则，见 `molmo_all_parts_unknown_allowed_current_AB.json` 与 `molmo_all_parts_unknown_allowed_camera_{A,B}.png`；forced-output 对照见 `molmo_all_parts_current_AB.json`；严格 garment-center abstention 见 `molmo_center_current_AB.json`。

## 10.3 历史 action/outcome 序列

- 最强 free-boundary positive sequence：`runs/claude_auto_20260812_210824/results/auto_exploration/20260812T130824848032Z/iteration_{001,002,003}.json`。
- Recurrent whole-item/multi-layer examples：
  `runs/claude_auto_01/results/auto_exploration/20260813T050343475098Z/iteration_{001,002}.json`；
  `runs/claude_auto_20260812_195401/results/auto_exploration/20260812T115402167618Z/iteration_002.json`；
  `runs/claude_auto_20260812_202645/results/auto_exploration/20260812T122645171957Z/iteration_{002,003}.json`；
  `runs/claude_auto_20260813_153538/results/auto_exploration/20260813T073539560657Z/iteration_001.json`；
  `runs/claude_auto_layers_01/results/auto_exploration/20260813T101854374025Z/iteration_002.json`；
  `runs/semantic_structure_test_01/results/auto_exploration/20260813T115716901842Z/iteration_001.json`。
- Early positive logo/footprint examples：`runs/claude_auto_01/results/auto_exploration/20260812T142754733751Z/iteration_{002,003}.json`、`runs/claude_auto_20260812_191028/results/auto_exploration/20260812T111028946809Z/iteration_001.json`、`runs/claude_auto_20260812_194508/results/auto_exploration/20260812T114508888528Z/iteration_001.json`。这些仍受旧 area/point-cloud/evaluator limitations 约束。

## 10.4 Diagnostic、depth、exposure 与 landmark 证据

- Command/pose audit 与 diagnostic images：`runs/diagnostic_lift_01/results/auto_exploration/20260814T043002999078Z/iteration_001/`。
- Raw `MULTI_LAYER_GRASP` evaluation 与 outer-loop NameError：`runs/diagnostic_lift_01/results/claude_auto/20260814T043648832897Z_evaluation_failed.json`。
- Dual-depth reports：`runs/dual_depth_test_20260813T090706615329Z/report.json`、`runs/dual_depth_test_20260813T091243909107Z/report.json`。
- Exposure sweep：`results/exposure_sweep/camera_A_100_to_800_step_50_v3/manifest.json`、`results/exposure_sweep/camera_A_100_to_800_step_50_v3/exposure_grid.png`。
- Heightmap exposure sweep：`results/heightmap_exposure_sweep/camera_A_100_to_800_step_50/manifest.json`、`results/heightmap_exposure_sweep/camera_A_100_to_800_step_50/camera_A_heightmap_grid.png`。
- Heightmap pipeline iterations：
  `results/heightmap_test/20260815T122550763527Z/`；`results/heightmap_test/20260815T122853802759Z/`；
  `results/heightmap_test/20260816T_retry_table_zero/`、`results/heightmap_test/20260816T_retry_table_zero_v2/`；
  `results/heightmap_test/20260816T_corner_interpolation/`；
  `results/heightmap_test/20260816T_live_A_exposure_800/`、`results/heightmap_test/20260816T_live_A_exposure_900_v2/`。
  Live capture infrastructure failure：`results/heightmap_test/20260816T_live_A_exposure_900/failure.json`。
- Molmo landmark baselines：`runs/landmarks_20260812T135443619133Z/`、`runs/landmarks_20260812T135604487303Z/`、`runs/landmarks_20260812T135701223972Z/`。

## 10.5 Failed-loop artifacts

- Cross-view Molmo-point disagreement 63.3 mm：`runs/claude_auto_01/results/auto_exploration/20260812T142754733751Z/iteration_005_failed.json`。
- Controller IK code 10：`runs/claude_auto_20260812_191028/results/auto_exploration/20260812T111028946809Z/iteration_002_failed.json`、`runs/claude_auto_20260812_192847/results/auto_exploration/20260812T112847815069Z/iteration_001_failed.json`。
- Physical `set_position code=-9`：`runs/auto_20260815_162746/results/auto_exploration/20260815T082747158986Z/iteration_001_failed.json`。
- 这些证明 infrastructure failure 实际存在；它们不解释 32 条合并完成记录中的 no-op、multi-layer motion 或 judgement 错误。

## 10.6 Git history：只说明尝试过什么

| Commit | 记录的尝试 | 不能据此声称 |
|---|---|---|
| `72faca3` | autonomous garment exploration before semantic landmarks | 初始 exploration 已解决 anchor search |
| `6d28061` | semantic garment structure observations | semantic landmarks 稳定 |
| `1972b6f` | Molmo decoder CPU offload | 资源问题修复等于感知正确 |
| `8f46984` | garment boundary、diagnostic、exposure 相关工作 | boundary/height 已成为 topology ground truth |
| `0c22fe7` | dense A/B height-map、coordinate guide、Laydown、DINO scripts/direction | Laydown 或 DINO 已有成功实验 |

## 10.7 明确的证据空白

- **DINO / canonical correspondence：UNKNOWN / INSUFFICIENT EVIDENCE。** `runs/` 与 `results/` 中未找到 DINOv3 match JSON、review JSON、similarity output 或 canonical graph experiment result。当前只有 standalone scripts、`DINOV3_CORRESPONDENCE.md` 的方法说明/engineering issues，以及未提交的 canonical-area prototype；不能声称低纹理/对称已使 DINO 失败，也不能声称 topology refinement 有效。
- **Research history：INSUFFICIENT EVIDENCE。** 项目文件检索没有发现独立的 `research_history` 文件；本清单使用的是 runs、结果目录、代码、README/方法文档与 Git history，不能引用不存在的统一研究日志。
- **Contact/force/layer count：UNKNOWN。** 没有找到经标定的 cloth contact force、tactile、finger gap→layer count 映射或 material load signal。
- **Peak-lift retention：UNKNOWN。** 当前自动 run 没有保存 close 后/峰值 lift 的同步 camera evidence；post-release RGB 不能重建 slip 时刻。
- **Absolute camera-to-base accuracy：INSUFFICIENT EVIDENCE。** diagnostic same-calibration reprojection 强力验证 command preservation，但不是独立 absolute extrinsic validation。
- **Laydown success distribution：INSUFFICIENT EVIDENCE。** 一个合并 label 被人工审计否定，另一个未执行，一个执行后无 evaluation；无法估计成功率。
- **Material topology ground truth：UNKNOWN。** 没有 verified intrinsic surface partition、hidden adjacency 或 per-pixel layer labels；现有 boundary/gradient/semantic cues 都不能替代。

---

**总括：** 保存证据支持的主要 bottleneck 是“从局部可见几何到 action-conditioned material state 的缺口”：系统已经能较可靠地保存图像、给出 base-frame waypoint、通过 IK 并精确执行；但仍不知道夹爪面对的是哪一层、该点与哪部分材料相连、提升时是否保持、以及释放后变化是否真由机器人造成。最近最有价值的正向信号是 visible free-perimeter peel、print/label/seam 的新出现和可审计的 command preservation；最需要避免的是把 ridge、motion、semantic name 或 post-release flat appearance 直接等同于 useful anchor。
