# Online Pilot 01 — 可运行入口与实验记录

## 人工复位更新

每个 run 开始前（包括第一个）暂停，状态写入 `WAITING_FOR_MANUAL_RESET`。
操作员把衣服完全展开并放回台面中央，手离开后在原终端输入 `y`；
随后才运行原有自动抓起放下初始化。每个 run 结束后不能自动开始下一次 reset。
确认前尚未创建下一 run 的 runtime；确认记录保存在 `manual_reset_XX.json`。
提示为 `[y/N]`，输入 `n`、直接回车或遇到 EOF 会退出，不执行下一 run；不支持恢复已退出的批次。
执行/初始化/请求异常仍终止批次，不会凭复位确认继续错误运行。
第六次结束后仅提示人工整理，不执行额外 reset。

这是按用户最新要求改成的人工协助复位实验，不能再称为全自动六次探索；
原有自动初始化动作、70/30 随机选择及两个 Skill 算法不变。

## 小深度空洞的局部关联

融合俯视图中的补色空洞可关联到附近实测点：半径最多2像素且不超过5 mm，
邻点RGB各通道差不超过30，至少3个候选，至少75%候选位于局部中位Z的±5 mm内。
随后仍检查原始点来源一致性和既有同表面实测支持。优先选最近的有效实测点，
不平均Z、不填补XYZ、不改变原始unknown mask，不扩展模型请求次数。
`topdown/depth_resolution.json`记录原点、实际点、偏移、来源支持或失败原因；
动作geometry中的source_references也保留原始/关联后的俯视坐标。
关联规则只处理原本缺深度的补色点，不对已有深度点新增高度变化检查。

离线回放 `skill_gate_online_20261106_094405_899654/run_01/round_004` 的
`[340,259]`仍拒绝：5 mm邻域6个点中只有4个在主高度层，另有85/96 mm的点，
无法确认同一布面；不是通过填平均深度放行。56项相关离线测试通过，未驱动硬件。

## 直接运行

从仓库目录执行：

```bash
# 仅检查协议及黑衣目标路径/哈希；不连接硬件或模型
bash scripts/run_skill_gate_online.sh check

# 真机启动一个新的、最多6次的批次；不续跑或覆盖旧批次
bash scripts/run_skill_gate_online.sh start

# 查看最近批次的阶段、Skill、执行次数和具体失败原因
bash scripts/run_skill_gate_online.sh status

# 指定旧批次；不连接硬件，也不要求当前配置仍可运行
bash scripts/run_skill_gate_online.sh status reports/skill_gate_online_20261106_074120_448897
```

`start` 在前台运行；Ctrl+C中断后沿用原执行链的异常处理并保存状态。
控制台完整输出自动保存到 `reports/skill_gate_console_*.log`。
每步打印分配Skill、选择概率、单次规划结果及ABSTAIN原因；原控制器执行日志保持可见。
批次最终生成 `batch_summary.json`、`diagnosis.json`，异常堆栈保存到 `batch_exception.txt`。
每个run仍保留初始化、原始观察、规划请求/响应、实际动作和32倍速录像。
正常完成6个run返回0；批次异常停止返回1；用户中断返回130；参数错误返回2。
6个run结束不等于6次衣物均完成展开，应查看各run的结果及原始图像。

单独Python入口：`python -m cloth_agent.experiments.skill_gate_online_batch`，
默认只检查；`--execute`才驱动设备，`--status latest --json`可读取结构化诊断。
可用 `--output` 指定新批次目录或 `--protocol` 指定同一实验协议文件；既有目录拒绝覆盖。

这次只补齐运行和诊断入口，没有修复或放宽上批暴露的180.7/156.7mm距离超限及reset IK失败。
相关44项离线测试、shell语法检查和旧批次诊断回放通过；未重启真机实验。

## 历史准备与运行记录

**已获授权启动：**批量入口 `cloth_agent.experiments.skill_gate_online_batch` 已接通，
复用现有Runtime、自动reset及录像，最多6次；初始化/执行/请求异常停止整个批次。
相关44项离线测试通过，最终批次与随机化17项复核通过，dry-run未访问硬件。
本次批次：`reports/skill_gate_online_20261106_074120_448897/`。
启动清单：同名 `_launch.json`；控制台日志：同名 `.log`。
以批次中的 `batch_summary.json` 为当前状态依据，下文准备阶段描述为历史记录。
入口不带 `--execute` 只核对协议和目标；显式 `--execute` 才启动。

**目标已补齐（2026-11-06，最新状态）：**用户摆好当前Mario黑衣后，已只用相机采集
`data/reference/black_mario_goal_20261106_153449_487709/goal.png`，并将路径及SHA256绑定到
`config/skill_gate_online_pilot01.json`。原始双相机RGB-D、topdown、mask和manifest均保存。
`robot_calls=0`，未启动初始化或展开。现剩6次批量启动入口待接通；下文缺目标描述为历史记录。

**当前状态（2026-11-06，后续更新优先于下文历史检查）：**用户确认采用
`docs/vision_baseline_20261106.md` 中的双相机融合与黑衣mask。视觉准备已通过本轮离线检查。
剩余准备是正确的黑衣展开目标（配置仍为null）及6次批量启动入口；单次随机化库适配器已存在。
以下“mask不可用”等描述保留为早期诊断记录，不再代表当前版本。

本轮复核记录：`reports/online_pilot_01_recheck_20261106/`。
重新检查88个已知目标候选（10张不同图），仍未找到当前Mario黑衣的正确展开目标。
两条强制分支均用测试确认发送本run的 `goal/target.png`，不会内部替换成默认红衣图；
真实黑衣目标尚未绑定，因此不能声称Simple已经使用了正确黑衣目标。
8个保存状态重新调用当前Tangent分割入口，mask逐像素与用户接受的视觉基线一致。
补充日志：规划返回时立即写 `planning_success`；汇总时保存 `action_geometry`
和执行器返回的 `actual_robot_actions`，未观测的另一技能仍不评估、不标为更优。
本轮不启动硬件、模型请求、人工复位或Gate训练；没有新增批量执行器。

替代上一轮人工匹配 A/B 实验。本轮配置见 `config/skill_gate_online_pilot01.json`。
固定 6 个 run、seed=2026110601，沿用原默认每 run 最多 12 步及自动抓起/中线放下初始化。
不训练或部署 Gate，不改 Simple/Tangent 的抓点、切线法线、最远交点或物理动作实现。

## 规划前随机化

新增 `skill_gate_online.RandomizedRequest`，在每次现有正常观察之后、任何规划请求之前：

1. 使用固定 seed、run ID 和 iteration 计算可复现随机数；不读取图像内容或事后效果决定概率。
2. `u < 0.3` 选 Tangent，否则选 Simple。先写 `exploration.json`，记录 observation ID、RGB 哈希、随机数、概率和 forced_skill。
3. 调用原 `partitioned_unfold.plan_round(..., forced_skill=...)`。Simple 复用原 pure planner；Tangent 复用原局部几何计算。
4. 新入口 `max_attempts=1`；外层 `episode(max_plan_attempts=1)`。没有候选重新选择、距离反馈后的第二次模型决策或另一 Skill fallback。
5. 无效输出、几何/深度候选失败记 ABSTAIN。后续运动预规划拒绝也在探索日志中记为 ABSTAIN，原执行器错误另行完整保留。控制器异常保留其真实失败状态并停止。

没有执行的另一 Skill 必须记 `NOT_EVALUATED`，不能猜测其可用性或优劣。
初始化本身继续复用原来的单次抓取区域请求，不属于 unfold iteration。
原常规运行入口的默认重选次数保持不变，只有本探索入口限制为一次。

`run_episode` 是库适配器，调用者需传入原 runtime、明确的黑衣目标和初始化 seed。
它不构造新控制器、不提供默认红衣目标、不在导入时启动设备；本轮没有开放自动实机启动命令。
每个 run 的目标清单需包含 `garment_id=black_mario`、来源和图像 SHA256，避免沿用默认红衣目标；这不是运动认证。
现有普通 `run_arm7_unfold.sh` 仍默认红衣目标，不能把它当作新探索入口运行。

## 记录

- 每步 `exploration.json`：随机分配、观测 ID/图像、所选候选状态、请求次数、ABSTAIN 原因。
- 原始 `request_payload.json` / 模型响应、选点意图、plan、geometry、motion、RobotAPI preview 和 execution 全保留。
- `iterations.jsonl`：正常 After RGB、执行结果和原始文件引用；衣物变化等待离线评价，不用电机完成冒充抓取成功。
- `exploration_summary.json`：初始化结果、实际 Skill 次数、最终图像、模型完成判断。
- 四个语义端点、平整度、残余褶皱/堆叠暂记 UNKNOWN，后续离线分析，不反向干预在线策略。

## 黑衣视觉检查

在 `data/reference` 和 `reports` 中检查了 88 个目标/展开参考文件，按哈希归并成 10 张不同图像。
现有正面黑衣展开参考是旧 Minecraft 图案衣服，不是当前 Mario 衣服；相机诊断目录中命名为 `goal.png` 的当前黑衣图仍是折叠状态，不是目标。
未找到当前 Mario 黑衣的正确展开目标及配置。新配置的 `goal_image` / `goal_sha256` 明确保留 null；未指向红衣。

使用已保存 G01 黑衣观测离线运行现有 Tangent 分割：

- 融合图 mask 有 83,989 像素；其中 78.4% 在人工检查的保守衣物包围框之外。这是诊断下界，不是精确分割准确率。
- 不做颜色修正仍有 77.9% 在框外。
- 分割种子要求 HSV 饱和度≤65、亮度≥65。黑衣在原始配色中太暗，在校色后又呈高饱和棕色；桌垫更容易满足种子条件。
- GrabCut/最大连通区域在错误种子基础上保留桌垫，是颜色先验导致 mask 错误，尚未进入 Tangent 几何选点即可看到失败。
- 103,262 个有效 topdown 格与其所属相机源像素的保存 world XYZ 差值为 0；未见存储索引错位。这不证明物理标定绝对正确。

最小视觉适配建议（本轮未实施）：只在 mask 入口增加黑衣配色配置，从未校色 RGB 和有效桌面范围提取暗色前景，再复用现有 GrabCut/连通域/轮廓流程；mask 必须保持当前 topdown 尺寸及像素索引。印花应作为同一衣物内部，不填补真实深度。切线、内法线、最远交点、抓取高度及执行算法全部不变。

## 当前结论

随机策略和单次请求适配已完成，视觉准备未通过：缺少正确目标，现有黑衣 mask 不可用。
没有启动任何真实 run、自动初始化、相机采集或 Gate 学习。数据报告见 `reports/online_pilot_01_preparation`。

### 黑衣分割适配（离线）
Online forced Tangent 的视图显式使用 `segmentation_profile=black`，读取同网格的
`unmatched_rgb.png`，以有效投影内的暗色为种子，再复用 GrabCut。默认其他实验仍为
原 light profile；普通入口可用 `CLOTH_SEGMENTATION_PROFILE=black` 显式选择，取消该变量即可回退。
没有改变几何选点、物理动作、深度采样或增加模型请求。

四张保存的黑衣图离线结果见 `reports/online_pilot_01_preparation/dark_mask/comparison.jpg`。
G01 mask 面积 20046 像素，在人工保守衣物包围框外的比例从旧结果约78%降为0%；
这不是人工真值 IoU。其余三张不再选择整桌，但靠近外边界的彩色印花存在漏分，
可能影响局部切线，尚不能声明所有黑衣状态的轮廓都准确。
本次不连接硬件。正确黑衣展开目标仍缺失，因此不启动六次探索。

### 印花漏分修正
上一版仅用暗色训练前景，GrabCut 会排除连到外边界的彩色印花，外轮廓填洞不能补回这种缺口。
现在从种子外侧局部环带提取桌面 Lab 颜色集合，将靠近黑衣、与桌面颜色不同且连到衣物的区域加入前景种子。
不使用凸包填满衣物凹口，不改变坐标或实测深度。四张离线对比已更新；原先印花处的大缺口消除，
轮廓仍有局部毛刺和阴影误差，颜色接近桌面的印花不保证都能区分。新增贴外边缘印花回归测试。

### 3×3 轮廓平滑
黑衣分割输出现在默认一次 3×3 闭运算后接一次开运算；函数参数 `smooth=False` 可回退。
四张图分别改变102、88、174、129个像素，约为原mask面积的0.5%–1.8%。
对比图：`reports/online_pilot_01_preparation/dark_mask/smoothing_comparison.png`。
小毛刺减少，主要凹口保留；原有阴影/背景误分仍存在，形态学平滑不能纠正其语义。
仅修改mask，不修改深度、切线拟合、动作或物理检查。相关24项离线测试通过，未连接真机。
