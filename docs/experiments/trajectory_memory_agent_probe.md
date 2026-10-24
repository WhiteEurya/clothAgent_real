# trajectory_memory_agent_probe

独立离线 Agent 经验迁移诊断，不训练新模型，不修改 Graph/planner/robot runtime。
两种条件均为 `gpt-6-astra` / `high`；每个 query/condition 创建独立 ephemeral Codex 会话。

本轮用户单独授权 4 次 Arm6 抓起放下初始化，使用原有
`scripts/test_arm6_random_fold.py --execute --max-retries 0`。
初始化与实验代码分开；实验模块没有硬件入口，不执行计划。

本轮输出目录：`reports/trajectory_memory_agent_probe/`。
每个 query 有 `query.png`，每个 condition 有严格 schema 的 `plan.json`、
`proposed_action_overlay.png`、`selected_experiences_contact_sheet.png` 和访问统计。
根目录和各 query 有 `report.md`，根目录 `human_review.csv` 留给人工比较。

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.trajectory_memory_agent_probe prepare --spec reports/trajectory_memory_agent_probe/query_spec.json
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.trajectory_memory_agent_probe run
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.trajectory_memory_agent_probe report
```

`run` 需要模型 API 网络，但 Agent 的 shell 使用禁网 workspace sandbox。
bubblewrap 只挂载系统工具、当前 query 和 memory 条件的只读历史库。
不挂载宿主项目、Graph、初始化日志、其他 query、其他条件输出或物理设备。
不可绕过失败的 namespace 回退到宿主工作目录。

memory 可自主读全部历史、看图、编写 Python/Pillow/NumPy 脚本做索引和拼图，
可以组合多条 manipulation pattern；不要求 exact state matching。
no-memory 使用同一模型配置和单步任务描述，但没有历史挂载，引用必须为空。

历史范围沿用用户确认的 2026-10-19/20 accepted unfolding collection sessions。
保留辅助手柄轨迹的实际日志，不把选点冒充实际手动抓点。
只采纳真实执行且有专门 before/after 的 accepted 记录；reset/pending 不在历史库中。
query 若来自历史采集，整个来源 session（包括过去和未来）都从历史库屏蔽；
与 query 完全相同像素的历史图所在 session 也屏蔽。

输入以 hash 冻结；所有方法选择在调用前固定。
模型输出不合 schema、超时或传输失败会保存原始输出并报错，不伪造 ABSTAIN。
`run` 跳过已有有效 plan；失败后重新运行会保存新的 attempt，不覆盖失败证据。

计数使用 Agent 不可见的宿主侧 IN_ACCESS 内核监控：
读取索引不算读了所有 trajectory；批量脚本读原图计入真实文件访问，
但不意味着模型逐张认真看过。保留 CLI 可观察的工具事件、`/work` 中生成的拼图/脚本、
所有唯一被读历史源图的访问清单和每条引用的 before/action/after 访问核验。
CLI JSONL 不完整记录看图调用，因此总看图调用数记为 unknown；
`/tmp` 临时文件随隔离环境销毁。报告另外生成最终引用的完整 contact sheet。

输出的动作合理性、Pa/Pb 合理性、引用相关性和 memory 相对收益最终由人工判断。
本轮不因结果自动调整算法、训练或执行动作。
