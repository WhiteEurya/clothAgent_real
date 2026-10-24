# goal_only_full_episode_v1

独立的真实目标图自由探索实验。默认目标为上一轮 memory episode 的
`round_006/after/rgb.png`，只冻结这张 RGB，不提供其动作、路径、深度或其他过程。
默认最多 12 rounds，`gpt-6-astra / high`，本轮不修改 prompt。

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.goal_only_full_episode_v1 --execute
```

先运行一次原有 `run_reset(max_retries=0)`。每轮独立 Agent 会话只挂载：

- 当前 RGB-D observation；
- 只读 `/goal/target.png`；
- 本次探索已执行步骤的 before/action/after。

不创建、不挂载人工 trajectory memory；不提供上一 episode 的历史或未来观察。
`supporting_experiences` 必须为空；目标图和当前图均作为图像附件传入。
目标是接近参考图的衣片布局与展开程度，允许整体平移/旋转及普通小褶皱，
不要求复制背景或桌面绝对位置。

用户已经摆好衣物时，用 `--skip-initialization` 从当前状态开始，
只按现有流程回到观测位姿并采图，不再抓起放下。重新开始必须指定新的 `--output`，不覆盖旧记录。

复用上一实验的单步 Host 执行、重复拦截、图像记录和停机流程。
与用户现用 `CollectionRuntime` 对齐，不改共享 runtime、primitive 或机器人接口。
ACTION / DONE / ABSTAIN；安全拒绝、12-round 上限或用户停止也会结束本次运行。
本轮不训练、不使用 Graph、不在结果不理想时自动修改方法。

输出 target、initial、final 对照，逐步 before/action/after、执行图像、Agent 日志、
时长及人工目标达成/FOLD_READY 记录。两轮的初始随机状态不同，
只能作能力观察，不能作为严格的 memory 因果对照。
