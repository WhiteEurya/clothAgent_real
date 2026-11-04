# 俯视图判断 + 同帧原图动作对照

当前 ZED Mini / 双 xArm7 的自动提案入口是 `bash scripts/run_topdown_folding.sh`。
俯视图用于主流程的 Graph 状态判断、出边选择、中心线和抓放点定位。
同一帧的原始 RGB 独立生成一份对照提案；不把原图分支的答案传给主流程。
每步仍只有一只夹爪负责，左右归属沿用[双臂分工规则](partitioned_folding.md)。
机器人接入继续暂缓：本入口只有观测到动作提案，没有执行参数或 SDK 调用。

## 研究开发与真机执行的边界

停止性能的完整认证不是算法开发的统一前置条件。复用已有控制、IK、碰撞和路径验证，
不因状态识别、锚点定位或选边算法修改重新要求整套物理实验。
没有实测的最大停止位移、动态关节跟踪误差和通信中断响应保持 `UNKNOWN`。
配置中的 1.25° 停止范围和 0° 跟踪误差只是现有计算输入，不能据此声称已测得这些值。
官方 Category 1 表格仅作为参考，不能补齐当前设备的未知测量。

可立即进行：历史 RGB-D 回放、俯视重建、Graph 状态/出边判断、抓放点定位、
左右臂分工、同帧原图对照、模型调用评估、模拟测试及给定参数下的离线路径检查。
这些入口不需要应用现场验收记录，不触碰真机执行开关。
`scripts/plan_zed_fold.py` 的结果明确列出计算假设；路径通过不自动升级为实机许可，
历史通过报告同样只在其配置假设下成立。

门禁来源审计（以当前仓库 HEAD 为基线，不推断更早现场流程）：

- `cloth_agent/dual_arm/safety.py::validate_safety(real=True)` 中的工具包络、看门狗、
  跟踪验证标志已存在于提交历史（该文件最近提交 `84aa9af`）；保留原检查。
- `cloth_agent/dual_arm/config.py::require_real` 的几何检查和伺服模式区分也已存在。
  `controller_sequential` 的当前 NativeRuntime 入口不要求流式伺服 commissioning。
- 本轮新增的 `create_zed_validation_record.py` / `apply_zed_validation_record.py`
  六项全通过记录流程不是所有研究入口的必要条件；其中指定 +3 mm 空载动作是新增
  验收方案，不是既有算法修改必须重复执行的流程。不要为了运行离线研究调用它们。
- 当前生成的 ZED 执行配置未获准，不代表历史 RobotCamCalib 已执行功能从未存在。
  历史成功运动、到位反馈与动态误差上限是不同证据，分别保留，不相互替代。

具体风险边界：`cloth_agent/dual_arm/stop_sweep.py::native_joint_ranges` 使用配置中的
停止/跟踪范围扩张路径包络；若实际偏移超出输入范围，特别是贴近桌面的抓取阶段，
模型无碰撞不能保证实际停止过程无碰撞。因此依赖该范围的新动作仍受原实机门禁限制，
但此限制不阻断上列研究任务，也不意味着已观察到实际碰撞。

```text
同一份 ZED RGB-D（带标定与哈希）
  ├─ 桌面俯视图 → Graph 节点识别 → 出边与锚点 → 回投原图 → 实测深度 → 选臂 → 主提案
  └─ 原始 RGB  → Graph 节点识别 → 出边与锚点 → 实测深度 → 选臂 → 对照提案
                                              ↓
                              状态、动作边、机械臂、抓放点毫米距离对比
```

## 运行一轮

```bash
conda activate cloth-real
cd ~/agentAuto/exp/clothAgent_real

# 已保存照片：先检查俯视图，不调用模型
bash scripts/run_topdown_folding.sh \
  --observation deployment-artifacts/zed_site_live_observation \
  --output outputs/topdown_inputs_001 --prepare-only

# 同帧的两路独立模型提案；输出目录必须是新目录
bash scripts/run_topdown_folding.sh \
  --observation deployment-artifacts/zed_site_live_observation \
  --output outputs/topdown_pair_001
```

以上输入是历史采图。当前场景先单独采一帧，再将输出目录交给提案入口：

```bash
python -m cloth_agent.site_observation --live-camera --output outputs/current_capture_001
bash scripts/run_topdown_folding.sh \
  --observation outputs/current_capture_001 --output outputs/current_pair_001
```

如果当前终端尚未取得 ZED 设备组权限，沿用此前 `sg zed -c '…'` 的采图方式。
相机采集命令不连接机器人。重复实验时使用新输出目录；要比较同一帧的模型波动，复用
同一个 `--observation`。更换实际衣物状态后必须重新采图，不能沿用旧锚点。

默认每路最多 4 次 `codex -p rbs` 调用：正常为识别和定位各一次，额外预算仅用于格式/
模型失败重试。两路相同模型、相同设置、相同历史参考图、独立会话、独立日志且禁用缓存。
识别阶段不提供动作边；定位阶段只提供已识别节点的受支持出边。
不确定时保留 `STATE_UNCERTAIN` / `ANCHORS_UNCERTAIN` / `BLOCKED` 等结果。
俯视图失败不会用原图提案替代；仍可保存原图分支用于分析。

## 查看结果

| 文件 | 内容 |
| --- | --- |
| `comparison.md` / `comparison.json` | 两路节点、动作边、选臂及抓放点距离；失败项不伪造差值 |
| `comparison.png` | 两路提案回投到同一张原图后的并排比较，左主流程、右对照 |
| `primary_plan.json` | 仅俯视图主提案，始终 `execution_allowed=false` |
| `observation/` | 两路共用的原始 RGB、实测点云和观测元数据副本 |
| `topdown/rgb.png` | 模型使用的俯视图 |
| `topdown/view.json` | 桌面拟合、裁切标志、坐标系、变换约定、输入输出哈希 |
| `topdown/source_xy.npy` | 每个俯视像素对应的原图像素，形状 H×W×2，顺序 x/y |
| `topdown/source_height_mm.npy` | 同一映射下的源深度高度；空洞为 NaN，非正交高度栅格 |
| `topdown/proposal/decision_preview.png` | 主流程在俯视图选择的中心线与抓放点 |
| `topdown/proposal/source_preview.png` | 回投原图后的中心线、抓放点和选臂 |
| `raw/proposal/` | 原图独立提案及原图标注 |
| `*/proposal/model/requests.jsonl` | 完整输入、响应、校验/失败记录和调用统计 |

`plan_observation(observation, output, ...)` 是之后机器人主循环可调用的连接点。
每个新观测调用一次，读取 `primary_plan.json`，再经过尚待接入的 IK、路径、双臂占用空间、
当前关节状态与现场边界检查。执行后的下一轮重新观察和识别，不把预期目标节点当作事实。
模型的自报置信度不等于已完成独立语义验证；现有验证标记继续保持 false。

## 几何与实验范围

俯视图复用 `RobotCamCalib/reconstruct_tabletop.py` 的桌面拟合与逆向 RGB 映射，
统一到当前的右臂底座公共系。外部标定代码未修改。
配置在 `config/topdown.zed_red.json`：当前红色衣服、浅色低饱和桌面、ZED 固定机位和 ROI。
`plane_fit_polygons_px` 单独指定四个印刷标记附近的桌面区域，避免大块纯白桌面的错误双目深度
主导平面拟合；衣物分割仍使用完整桌面 ROI。拟合阈值未放宽，候选区域预览、筛选前后数量
和实际内点率保存在 `table_fit_candidates.png` / `view.json`。
换颜色、机位或桌面范围后需相应修改分割/ROI，不能宣称当前版本适配任意衣服。
环境需要 OpenCV (`cv2`)；当前 `cloth-real` 已安装。

RGB 俯视外观假设衣物贴近桌面，抬高的褶皱会有视差；不会补全裁切或遮挡部位。
锚点通过 `source_xy.npy` 回到原图，再经原有 5×5 实测深度与离散度检查。
无有效深度即拒绝，不用桌面平面补出抓点高度。1 mm 栅格采样间距不代表 1 mm 标定精度。
两路选臂和高度使用同一桌面法向，原图模型不接收俯视图、分割结果或主分支答案。

这是“俯视重建预处理与原图”的小型对照：预处理同时包含透视变换、衣物分割、裁切和
背景去除，因此不能将差异全部归因于视角。历史 Graph 参考图保持原始透视，两路完全相同。
一次成对提案也不能排除模型随机性，更不能说明实际折叠成功率；后续可用同帧重复运行和
多种真实衣物状态扩充证据。现阶段记录动作差异，不执行两套动作。

## 本机首轮结果

使用历史观测 `deployment-artifacts/zed_site_live_observation` 完成真实模型对照，结果位于
`deployment-artifacts/topdown_ab_003/`。两路均为 `PREVIEW_READY`：

| 项目 | 俯视图 | 原图 |
| --- | --- | --- |
| 当前节点 | n_initial | n_initial |
| 动作边 | e_initial_left_sleeve | e_initial_left_sleeve |
| 选定臂 | 物理右臂 | 物理右臂 |
| 抓点公共系 mm | [201.523, -92.828, 9.829] | [198.272, -74.801, 10.401] |
| 落点公共系 mm | [417.730, -71.810, 9.495] | [410.990, -23.924, 8.744] |

抓点三维距离 18.33 mm，落点 48.36 mm。共 4 次 `codex -p rbs` 调用、0 次模型工具调用、
0 次机器人调用。两路状态与动作类型一致，但具体位置不同；不能据此认定哪路物理效果更好。
该帧有裁切标志，桌面拟合残差 RMS 为 2.62 mm，红色区域有效深度约 99.52%。
相关几何、分工和 ZED 测试 51 项通过，另已通过模型调用和 Graph 回归检查。

隔离模型调用只挂载筛选后的 `rbs.config.toml`，不挂载用户基础配置。
本机 CLI 实测 `--ignore-user-config` 也跳过命名配置，因此该隔离调用不再加此参数。
配置探针及真实成对调用已验证修正；其他隔离 Graph 调用入口同步修正。

## 首次实机准备（尚未执行）

### RobotCamCalib 最小适配核查

`cloth_agent.experiments.robotcamcalib_adapter` 将已有主提案校验、公共系到所选臂
基座的位姿核对、当前 7+7 轴配置与身份核对接到 `set_position` 检查接口。
复用 `prepare_partitioned_trial.controller_checks` 的 TCP 限制、IK/FK、关节限制和
非零 `only_check_type` 路径查询，并保留旧后端的模式、故障、TCP、世界偏移、
开爪及起点核对。输入还按归档的 `source_target_bounds_mm` 校验原始抓放目标，
不会用当前较宽的工作区配置偷偷替代旧脚本目标边界。

```bash
python -m cloth_agent.experiments.robotcamcalib_adapter \
  --proposals deployment-artifacts/topdown_physical_trial_001/proposals_v2 \
  --robot-state outputs/onsite_acceptance_20261104_01/robot_readonly_host.json \
  --output outputs/robotcamcalib_adapter_NEW
```

此命令只生成不可执行的 `adapter_request.json`，没有机器人连接或执行选项。
端到端测试使用真实提案/标定/历史状态文件和假的 SDK 接口；接口成功不代表真实控制器
已批准路径。另一项测试回放该提案的真实历史 TCP 查询，在 `approach` 保留拒绝结果。

这不是 `execute_point_plan.py` 完整执行接入完成：原脚本不操作夹爪，始终要求开爪，
只支持单个目标加 5–150 mm 净空。独立的 `descend_selected_point.py` 虽支持闭爪，
也只允许在目标表面上方 5 mm 闭爪；不能不改语义就执行当前表面抓取提案。
旧脚本读取原始 depth/confidence 单像素选择，而新提案是经哈希和再生成核验的
中值深度锚点；没有伪造旧 selection 或 `controller_checks_passed` 文件。
此外当前 lifted/across/retreat 的 20 mm/s 超过旧脚本第五段起的 10 mm/s 上限，
适配器保留原提案并列出冲突，不静默降速或删除夹爪事件。

原脚本执行后的到位检查、异常 `set_state(4)` 及退出逻辑完全未改动；本适配器
没有执行分支，因此没有声称已测试这些实机逻辑。原脚本本身不验证桌面/另一臂等
外部碰撞；不能将已通过的显式关节路径证明套用于控制器自行插补的笛卡尔路径。
完整执行仍需要动作语义适配和同一笛卡尔路径的原有碰撞检查接入，不需要为此
先满足 NativeRuntime 的四项验收标志。本轮未修改这些门禁或执行配置。

按用户选择，准备完整的一次折袖，包含高处转向/横移、接近、抓取、抬升、搬运、放置和退离。
新增 `cloth_agent.experiments.prepare_partitioned_trial` 重新校验主提案与源观测，读取两臂身份和
姿态，然后查询选定臂的 TCP 限制、IK/FK、关节限制和控制器 check-only 路径。
本入口没有 `--execute`，所有路径查询的 `only_check_type` 均非零；不会控制夹爪或修改使能/模式。
控制器通过也只会标为 `CONTROLLER_CHECKS_PASS_EXTERNAL_REVIEW_REQUIRED`，不等于外部碰撞已验证。

```bash
python -m cloth_agent.site_robot_check --output outputs/trial_robot_state.json
python -m cloth_agent.experiments.prepare_partitioned_trial \
  --proposals outputs/topdown_pair_001 \
  --baseline outputs/trial_robot_state.json \
  --output outputs/trial_preparation_001
```

输出目录必须为新目录。默认使用原工具姿态；`--pitch-offsets-deg` 可明确要求仅检查几个
倾角候选（每个 ±45° 以内），保留同一组抓放 XYZ，记录姿态偏离，绝不自动执行。
候选仅用于诊断，倾斜后的抓布和工具净空仍需单独验证。

本机结果记录在 `deployment-artifacts/topdown_physical_trial_001/trial.md` 和 `trial.json`：
两臂在线且仍在 Home，TCP/世界偏移匹配，检查前后关节未变化。新采图生成了右臂折左袖方案。
**首个低处接近点触发 TCP 限制，原姿态和已试倾角均未通过；完整路径因此尚未获准。**
额外只读查询可返回 IK，但 TCP 检查仍返回限制，原因尚待确认。原图对照继续保存。
本轮运动命令与夹爪命令均为 0，没有为放行动作而调整限位、抓点或现场完成状态。
