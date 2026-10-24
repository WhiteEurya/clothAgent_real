# One-shot goal planner probe v1

本实验使用已保存的真实 before observations 做离线规划比较，模型请求需要联网，
但不连接机器人、不执行动作，也不重新运行原 Agent。

结果：[summary.md](../../reports/one_shot_goal_planner_probe_v1/summary.md)；
人工评审：[review.html](../../reports/one_shot_goal_planner_probe_v1/review.html)。

## 样本

采用 `goal_only_full_episode_v1_restart02` 的 3 轮与
`goal_only_full_episode_v1_continue01` 的 7 轮，共 10 组配对。
包含原先被深度检查拒绝的 ACTION，以及最后的 DONE；
不纳入人工摆平后直接 DONE 的零动作试跑。
所有 query 使用相同冻结 goal RGB。原 plan 仅用于对照/绘图，不作为真值或模型输入。

## 调用边界

- Host 提前准备 current RGB、goal RGB、简化 prompt、strict schema 和完整 HTTP body。
- 直接使用现有 Codex 账号的 Responses HTTPS 传输；没有 CLI/Agent 循环。
- `gpt-6-astra`，`reasoning.effort=high`，`tools=[]`，`tool_choice=none`。
- 每 query 最多一次 HTTP 请求；持久化 `attempt.json` 防止重复。
- `max_retries=0`，不跟随 redirect，不 repair，不接续模型调用。
- 不提供 shell、文件检索、trajectory memory、crop/view 或 depth。
- 只有双方均 ACTION 时报告 Pa/Pb 原图欧氏像素距离，绝不当作误差或动作质量分数。
- 速度配对使用原 Agent 保存的 wall latency；Host 准备时间单独记录。

## 命令

```bash
/home/sja/miniconda3/envs/cali/bin/python \
  -m cloth_agent.experiments.one_shot_goal_planner_probe_v1 prepare

/home/sja/miniconda3/envs/cali/bin/python -u \
  -m cloth_agent.experiments.one_shot_goal_planner_probe_v1 run
```

`prepare` 要求新目录；`run` 永远跳过已有 attempt 的 query，包括失败和未完成项。
重新汇总不调用模型：

```bash
/home/sja/miniconda3/envs/cali/bin/python \
  -m cloth_agent.experiments.one_shot_goal_planner_probe_v1 report
```

用浏览器打开 `review.html`，逐条选择
`ONE_SHOT_BETTER / ORIGINAL_BETTER / EQUIVALENT / BOTH_BAD`。
选项默认空白。浏览器 localStorage 自动保存，可导出 JSON；导出后运行：

```bash
/home/sja/miniconda3/envs/cali/bin/python \
  -m cloth_agent.experiments.one_shot_goal_planner_probe_v1 review \
  --ratings /path/to/one_shot_goal_planner_review.json
```

人工尚未提交时，质量结果必须保持 PENDING，不能自动推断 EQUIVALENT。
评审页不展示延迟，减少速度数字对质量评分的影响。

## 解释边界

该实验删除的不仅是工具执行时间：原 Agent 可访问 RGB-D 和本 episode 历史，
本次按照请求只提供两张 RGB，prompt 也简化。因此不能严格把质量/速度变化
单独归因于 orchestration。原延迟包含 CLI 启动，而 one-shot 是准备完输入后的
直接请求 wall time。每 query 只调用一次，服务负载、缓存、调用时间不同。

“接近 20 秒”的描述性口径预先固定为 median≤25 s，同时报告 ≤20 s 的数量、
mean、median、p95 和所有逐轮原始数据；不以此替代质量判断。
