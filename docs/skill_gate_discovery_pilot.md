# Skill Gate Discovery Pilot

本轮仅做 4 组真实初态的 A/B 首步对照，验证 Harness 能否归纳适用条件。
不训练模型权重、不修改现有两个 Skill 的内部算法、不部署或热更新 Gate。
当前只准备代码和协议；尚未采集本轮结果，也没有已学习的 Gate。

## 预先固定的协议

|组|初态摆放参考|用途|执行顺序|
|---|---|---|---|
|G01|单侧折入|学习|A → 匹配复位 → B|
|G02|中央堆叠|学习|B → 匹配复位 → A|
|G03|双侧折入|留出测试|A → 匹配复位 → B|
|G04|改变朝向的堆叠|留出测试|B → 匹配复位 → A|

这些是采样状态，不指定哪个 Skill 应胜出。全部使用当前黑衣以及它自己的展开目标。
每组保存一张 canonical Before 和两条轨迹的实际 Before，评估时检查复位是否可比。
匹配失败、信息不足、首步无效或指标冲突必须记为不可比，不能用另一条轨迹的成功补齐。
不要通过重复尝试后挑选最好一条来替换失败轨迹。

- A：Simple、Simple、Simple、Simple，最多四个真实抓放动作。
- B：Tangent、Simple、Simple、Simple，最多四个真实抓放动作。
- Simple 复用 `plan_round_pure`；Tangent 复用当前 `run`、`measured_candidate`、轮廓分割和最远交点计算。
- 实验指定的是调用分支，不是预设适用条件。Prompt 不包含旧的“何时启用 Tangent”判别句。
- 无有效输出时 ABSTAIN/NOT_APPLICABLE，不切换另一个 Skill 补做，不强迫运动。
- 复用 `episode` / `prepare_action` / `PartitionedReplayRuntime` / `RobotAPI`，正常关节、路径、碰撞、深度与异常停止检查保留。
- 每步只使用正常当前 RGB＋固定目标做规划，深度继续由既有执行链使用。保留既有无效候选的重新选点；所有次数写入原记录。这不是额外语义复核。
- 不随机 reset，不追加按压，不读取上一轮图像作规划上下文。
- 轨迹入口仍有既有张开夹爪和 Home；每次准备好衣物并离开工作区后再启动。

已知限制：当前分割器的浅色衣物先验在黑衣/融合图上可能失效。本轮不修该算法，也不能把错误分割算作有效 Tangent。不能因此把 Simple 标成语义上更优；应记录不适用及证据。

## 命令

默认目录 `reports/skill_gate_pilot_20261106`，可用 `CLOTH_GATE_PILOT_ROOT` 指定一个新的独立目录。
以下步骤由操作者逐项执行，不会自动串行启动机器人。

```bash
# 只初始化协议；已有目录不可覆盖
bash scripts/run_skill_gate_pilot.sh init

# 黑衣展开、方向确认、手离开桌面后：只拍摄，不连接机器人
bash scripts/run_skill_gate_pilot.sh capture-goal --note '黑衣展开目标'

# 摆好 G01，保存不含后果的初态；只拍摄
bash scripts/run_skill_gate_pilot.sh capture-before --group G01 --note '单侧折入'

# 不带 --execute 只检查协议，不连接机器人或模型
bash scripts/run_skill_gate_pilot.sh trial --group G01 --arm A

# 明确启动一条真实轨迹；最多4步，结束后不会自动启动下一条
bash scripts/run_skill_gate_pilot.sh trial --group G01 --arm A --execute --note '初态已摆好'
# 人工按 canonical Before 匹配复位、手离开工作区后
bash scripts/run_skill_gate_pilot.sh trial --group G01 --arm B --execute --note '按G01初态匹配复位'
```

每组结束后离线检查正常采集的 Before/After（不调用每步额外 VLM）：

```bash
bash scripts/run_skill_gate_pilot.sh review --group G01
# 打开 G01/review.html，填写 G01/feedback_template.json 的真实观察结果
bash scripts/run_skill_gate_pilot.sh feedback --group G01 \
  --feedback reports/skill_gate_pilot_20261106/G01/feedback_template.json
```

反馈必须包含：初态可比性、各观察的领口/袖口/下摆角可辨认状态、首步效果、最终平整度、任务完成、副作用、有效性以及实际文件证据。
不明确的部位记 `UNKNOWN`；没有可信的衣物变化证据不能把控制器 `EXECUTED` 当作成功。
平整度使用协议中的 0–3 序数量表，仅作离线人工图像评分，不声称为毫米级物理平整度。
整体完成、部位恢复和平整度一致支持一方时才能给 A/B 偏好；冲突时记 INCOMPARABLE，效果相当记 TIE。
该 Pilot 不能用一个不确定的加权分数掩盖副作用。

G02 按 B → A 顺序同样执行。前两组反馈齐全后：

```bash
bash scripts/run_skill_gate_pilot.sh discover
```

`discover` 创建独立、无工具、无历史的一次模型请求。请求白名单只有 G01/G02 初态、两条轨迹实际初态和真实反馈，以及固定目标。
它不导入现场对话、手写切换规则、G03/G04 图片或结果。模型自主归纳规则，或输出 `INCONCLUSIVE`。
保存 `gate.json`、`gate_seal.json`、训练证据摘要哈希和完整请求。已有 Gate 不允许覆写。

随后摆放 G03/G04 的初态，分别 `capture-before`；每个初态必须在第一次动作前预测：

```bash
bash scripts/run_skill_gate_pilot.sh predict --group G03
# 然后按顺序开展 G03 的 A/B，暂不反馈或打开比较报告
# 摆好 G04 初态、capture-before 后
bash scripts/run_skill_gate_pilot.sh predict --group G04
# 然后开展 G04 的 B/A
```

每次 `predict` 仅一张 Before RGB＋固定目标、一次独立决策；不读取该组动作或结果。
两组预测均保存后才开放留出结果的 `review` / `feedback`。物理操作者不可避免会见到场景，但这些场景与反馈不会回传已冻结的学习请求。
Gate 仅为独立冻结规则文件；将来可拼入现有单次 Agent 决策请求，本轮不修改生产入口。

```bash
bash scripts/run_skill_gate_pilot.sh report
```

输出 `report.json` 和简短 `report.md`，明确区分：

- 数据尚未完成；
- 学习组偏好不明确；
- 有偏好但 Harness 无法归纳 Gate；
- 留出选择错误；
- 留出不可比较或 Gate 不确定；
- 仅这两个留出状态选择得到支持。

两组试验不能用于宣称泛化。软件准备阶段不生成伪 Gate 或伪真实结果。
