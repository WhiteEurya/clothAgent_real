# CV Feedback Harness：V0–V2

实现位置：`cloth_agent/cloth_harness/`。这是离线最小系统，没有接入机器人执行，也没有实现 V3 自动改代码或 V4 抓点修正。原来的真机启动脚本、Pa→Pb 输出格式和底层控制器保持原样。

## 当前能力

- 从既有 `episode_*/round_*` 读取真实 before/after RGB-D、dispatch_intent、execution。核对图像来源与派发记录，排除缺后图、没有实际动作、重规划图像不一致等样本；保留排除清单。部分执行单独记录，不解释为完整操作。
- 沿用紫色衣服/深色桌面分割先验，检测触碰图像/裁剪ROI边界。不是跨衣服通用分割。分割质量只是工程启发式，未经独立标注验证。
- 同一观测视角下报告像素投影面积变化。尚无静态重复采集标定，所以默认 `UNCALIBRATED`，不直接标 INCREASED/UNCHANGED，也不判展开收益。
- 相邻衣物纹理 ORB 双向匹配、RANSAC、无缩放刚体拟合。检查匹配数、内点比例、纹理覆盖范围、尺度、残差、图像越界。报告刚体对齐 IoU、轮廓变化和中心运动像素量。
- **像素运动不伪装成毫米**。当前 translation_mm=null。暂未实现桌面平面投影后的全局毫米运动估计、局部材料点跟踪或 RIGID_DOMINANT 分类。现有值只能作为未验证的数值证据。
- 由确定性函数生成上一动作的 compact feedback。电机执行完成永远不会自动升级为抓住布料。失败/部分执行保留原始状态。
- 每条件只调用一次 GPT：A 当前图；B 当前图+上一动作前图和真实动作；C 当前图+自动反馈；D 当前图+人工确认的结构化结果。各组都有同一展平目标，完全相同的短prompt和输出schema。没有额外的锚点识别调用。
- 固定枚举人工标注与评价器：Accuracy、Coverage、False Confident Rate 分开，UNKNOWN 不冒充正确。缺标签时输出 null，不输出虚构准确率。

## 使用

```bash
PY=/home/sja/miniconda3/envs/cali/bin/python

# V0/V1：只进行本地CV，绝不访问模型或机器人
$PY -m cloth_agent.cloth_harness.experiment prepare \
  --source reports/ggo_rollout_20261025_142045_387801620 \
  --output reports/cv_feedback_new

# V2：按样本时间顺序等间隔选6个状态，A/B/C各一次请求；无人工标签的D明确跳过
$PY -u -m cloth_agent.cloth_harness.experiment run \
  --output reports/cv_feedback_new --limit 6

# 填完 human_review.html 对应的 oracle_labels.json 后，只补D
$PY -u -m cloth_agent.cloth_harness.experiment run \
  --output reports/cv_feedback_new --limit 6 --conditions D

# 统计CV与独立结果标签的一致性；未标注时指标为空
$PY -m cloth_agent.cloth_harness.experiment evaluate \
  --output reports/cv_feedback_new

$PY -m pytest tests/test_cv_feedback_harness.py -q
```

`--limit 0` 使用全部可用、实际完成动作对。第一次试验默认6例不是统计验证。请求顺序用固定种子打乱；2路并发，每条件每状态1次，不重试失败模型请求。重新运行时保留现有结果；中断且已经发送的请求不会悄悄补发。要改变输入或重复试验应使用新输出目录。

输出的 `index.html` 展示前后mask、实际动作和反馈；`abcd/index.html` 展示新策略动作；`human_review.html` 只展示真实前后图和实际动作，不展示CV预测。`oracle_labels.json` 的 confirmed_by、evidence 及五项固定标签须由独立人工填写，不允许把下一动作建议写进Oracle。Unknown合法。没有有信息量的确认标签时D不运行。

## 实验解释与边界

本轮来源是我们已经看过的14:20 run，仅用于开发演示。没有将相邻帧随机拆成验证集，也没有把已看过数据宣称为held-out。

每个V2样本的当前图是真实动作后的图，B的前图与实际动作来自同一个真实动作对。离线新动作永远不接入下一条历史状态。当前目标图在各条件中一致，是固定任务说明，不是引入参考配准。

当前无下一动作独立标注，无法报告抓点准确率、决策改善率或真机成功率。A/B/C动作有差异只是差异，不能把历史后续结果赋给新动作。每组一次采样也不能排除模型随机性。

当前 feedback 约100英文token量级，固定最多一个动作，不随轨迹增长。后续应实测token和决策延迟；不能假设短prompt必然更快。

## 后续独立验证与RSI边界

1. 采集真实静态重复观测，冻结面积噪声阈值；建立分割与配准独立人工标签。当前工程门限不是已校准门限。
2. 先完成D组Oracle对照，获得允许多个合理动作的独立决策标注。否则没有依据判断应优先改CV还是反馈格式。
3. 按整个episode分离development/validation/heldout。当前演示数据全部development；新独立数据应由外部评价进程保管。
4. V3创建受限候选环境时，只授权修改 `cv_harness/`；冻结 `runtime/`、`evaluation/`、标签和验收阈值，并由外部进程校验哈希。当前source_snapshot是可复现记录，**不是访问隔离或防篡改安全机制**。
5. 不能在缺少人工标签、固定验收门限和独立新数据时自动接受“改进版本”。本实现未进行任何自动接受、部署或机器人重启。

## 更新：Geometry Check → 真机执行 → Observed Outcome

人工“最佳/适合抓取区域”不再作为当前流程的前置条件。8例区域标注实验已标为被替代，尚未运行48次请求。

新增 `--cv-feedback`，用现有One-shot Host进行完整执行：

1. 当前图+固定任务目标+上一实际动作的compact feedback，一次GPT调用，输出原有Pa/Pb。
2. 新鲜执行前观测检查布料mask支持、抓点局部有效深度（5×5至少18点且中心有效）。边缘距离、局部高度范围只记录，不作为最佳区域排名，也不恢复高度差拒绝门限。
3. 原有Host进行世界坐标、夹爪姿态、工作空间、行程/IK和到位检查。CV的ACCEPTED仅表示局部检查通过，不等于机器人可执行或语义部位正确。
4. 执行后用真实前后观测和执行日志记录 `cv_outcome.json`、`cv_feedback.txt`、`cv_outcome_sources.json`，下一轮只携带这一轮反馈。观察器失败输出UNKNOWN，不臆造成功。
5. CV证据不足停止并留档，不自动重规划，保证正常动作每轮一次GPT请求。无额外模型shadow gate。

局部自动修正暂为 `DISABLED_UNCALIBRATED`：没有可靠标定前不挪动Pa。mask仍是紫色衣物先验，浅色印花可能导致UNVERIFIED，不能把它解释成不属于衣服的真值。边界裁剪/视角差异仍可能造成UNKNOWN。硬件中途故障沿用原Host停止流程；没有安全采集的后图时不生成正常完成的Outcome。

手动摆好衣服后，运行一次（不会先reset）：

```bash
bash scripts/run_cv_feedback_rollout.sh --execute --episodes 1 --no-reset
```

使用现有随机reset再开始一次：

```bash
bash scripts/run_cv_feedback_rollout.sh --execute --episodes 1
```

新入口默认一次episode、最多12个动作，沿用20–40mm reset半径配置。只写入代码，尚未在真机执行验证。原run_ggo_rollout.sh仍指向原GGO实验，用上面的新入口才会启用新Harness。RSI自动改代码/验证接纳仍未启用；需要固定独立结果标签与验收标准后再开展，不把CV自评当作进步。
