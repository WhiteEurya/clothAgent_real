# trajectory_memory_full_episode_v1

单次真实 Arm6 闭环 unfolding episode；`gpt-6-astra / high`，默认最多 12 rounds。
冻结用户确认范围内所有 accepted、真实执行的 10/19–20 unfolding 人工轨迹。
保留真实执行但可能未成功展开的历史证据，不把 accepted 当作成功。
本 episode 单独存储，不更新历史、不训练、不建立 Graph、不运行 no-memory。

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.trajectory_memory_full_episode_v1 --execute
```

输出目录必须不存在；不自动恢复或重复已执行动作。`--max-rounds` 可显式配置。
使用现有 `run_reset(max_retries=0)` 初始化一次；初始化不计入展开动作数。
每个 round 独立 Agent 会话，挂载当前 RGB-D、只读冻结 memory、之前已执行 round 的 before/plan/execution/after。
Agent 无机器人访问权。历史与本 episode 经验目录明确分开；没有未来观察。

动作严格为 ACTION/DONE/ABSTAIN JSON。唯一支持的 ACTION 为已有 `lift_pull_place`，
像素经实际 RGB-D 采样后调用 `automatic_program`，沿用 collection Home 夹爪姿态、
抓取偏移 -15 mm、抬升 50 mm、放置偏移 5 mm、撤离 30 mm 和分阶段开爪。
沿用已有 10–150 mm 位移上限。Agent 不生成代码或特殊动作。

用户要求与现用采集脚本对齐后，实验直接继承 `CollectionRuntime` 的配置、
规划和 Home/perception 行为：`path_validation_enabled=false`，left 附加碰撞余量关闭；
只增加执行图像记录，不修改共享 runtime、primitive、配置文件。
原采集路径已启用的深度、距离、IK、关节及实时反馈检查保留；
几何 workspace/collision 验证沿用采集脚本关闭状态，不记录为通过。
任何拒绝直接 SAFETY_STOP，不重试动作。相同观测下重复相同 Pa/Pb
被 Host repetition guard 拒绝，终止为 ABSTAIN 并保留原 Agent plan。

现有夹爪反馈不能可靠确认 acquisition；保存约 5 Hz RGB 图像序列和原有关节/夹爪反馈，
标记 UNKNOWN_PENDING_HUMAN_REVIEW。零个“已确认抓空”不等于全部抓取成功。
物理动作数记录 manipulation dispatch attempts，包含潜在部分执行；执行状态单独保存。
每步完成后重新采图，DONE/ABSTAIN 也保存终止观察。安全停后不再移动机器人取景。

每轮目录含 before、selected_experiences、plan.json、execution.json、after、
agent_events、timing.json。根目录含冻结清单、固定协议、episode_summary.json、
report.md、全过程缩略图及 human_review.json。只有人工填写才产生人工 FOLD_READY 结论。
