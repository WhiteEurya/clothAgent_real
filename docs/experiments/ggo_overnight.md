# 连续夜间展开实验

只运行当前新版：上下文 one-shot＋自动锚点＋修复后GGO。不是新旧随机对照。不会使用TopN或学习参数。

## 启动

```bash
cd /mnt/newssd/sja/clothAgent_real
bash scripts/run_ggo_overnight.sh --execute
```

默认8小时，最多1000次实验。`--max-hours 6`可修改时长，`--wait-before-first`可先启动控制页、确认后再运动。控制页 http://127.0.0.1:8769 。不要叠加 `--no-reset`。

每次随机选择中心抓取半径上限U[0,40] mm；不是释放高度，也不是保证抓点距中心20–40 mm。实际采样偏移、半径和种子另行保存。释放高度沿用原策略。零半径抓已验证中心；小半径200次拒绝后退回已验证中心并记录，不填补无效深度。

每次先reset，再连续执行最多12个动作。最后一个动作之后仍观察并判断DONE，但不执行第13个动作。DONE或ROUND_LIMIT且清理完成后立即进入下一次reset。ABSTAIN、执行异常、硬件故障或人工停止终止整批；不自动清故障重试。到8小时不再开启新实验，允许当前次完成；剩余磁盘少于20 GiB也不开始新实验，不删除旧证据。时间从首次就绪后开始计算。

## 记录

- `protocol.json`：配置、模型、源文件hash、时长、半径范围。
- `reset_schedule.json`：预生成半径/种子序列；`episode_NNNN.reset_radius.json`实际派发参数。
- `events.jsonl`：实验开始/结束、进程、人工控制、时长/磁盘停止事件。
- `episode_NNNN.run.log`：完整子进程输出。
- 每个episode中的初始化RGB-D、抓放位置、轨迹、模型请求/响应、执行采集以及各轮before/after、Pa/Pb图、理由、几何、恢复和耗时。
- `analysis/index.html`：初态/终态和逐轮动作证据。
- `analysis/episodes.csv`：每次结束原因、模型DONE、动作数、时间、半径、实际偏移、释放高度、reset是否退回中心等。
- `analysis/rounds.csv`：每步Pa/Pb、方向、理由、几何状态、执行状态、恢复和分阶段耗时。
- `analysis/human_review.csv`：独立人工复核标签，重新分析时保留；填写DONE/NOT_DONE/UNCERTAIN。

种子不能复现物理布料状态，以实际观测为准。EXECUTED只代表运动命令完成，不证明夹取成功或中途未掉落。UNKNOWN不得解释为展开。当前简化prompt没有上一步效果输出，因此没有启用连续3次无改善终止。

## 历史比较

结束后运行（将NEW_RUN换成本次终端输出目录）：

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.overnight_analysis \
  --root NEW_RUN \
  --historical reports/reset_radius_sweep_20261024_161453_855154284 reports/ggo_rollout_20261024_201047_587656891
```

指标：模型DONE/正常完成次数、Wilson区间、全部结束原因与排除样本、动作数/耗时、半径分组、配准可用率、重规划/深度恢复及左右动作明细；人工复核结果独立统计。停止/故障不悄悄合并到成功样本中。

今晚0–40 mm与历史固定20 mm初态分布不同，历史还可能有不同prompt及软硬件状态。历史比例差异只作观察性比较；若要证明修改提高成功率，需要相同条件下的新旧交错实机试验和独立人工结果标签。当前运行不做这一因果承诺。
