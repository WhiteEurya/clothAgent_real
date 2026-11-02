# 离线失败挖掘

报告：`reports/offline_failure_mining_20261029/review.html`。

只读取历史记录、生成分析产物，没有机器人入口，不导入或修改动作策略。
固定时间窗为 2026-10-22 03:18 至 2026-10-29 03:18（Asia/Singapore），重试不移动窗口。

复用 `cloth_harness.runtime.dataset.load_observation` 和既有 `outcome_observer.observe`。
检查真实观测、时间、图像哈希和 dispatch 绑定。适配旧 `observation_sha256`、新
`before_sha256`、`pre_execution_observation`；缺少可核验动作或计划哈希不符则弃权。
执行完成不是夹取成功。新重算评价和原始评价文件分开保留。

- `index.py`：扫描包括 ignored 文件，按捕获时间纳入；去重；索引完整序列和终止信息。
- `prepare_analysis.py`：所有六段初筛候选都进入数据包；只含匿名 run、真实图像、动作与固定评价。对两个平台检索跨 run 面积增长案例，以归一化轮廓 IoU 排序，不能当作因果对照。
- `report.py`：要求独立分析结果存在才生成 HTML，包含全部 run 和各步原图／命令／评价。
- `validate.py`：核对筛选窗口、来源对应、独立结果结构和 HTML 本地链接。

上述脚本使用 `/home/sja/miniconda3/envs/cali/bin/python`。重算索引会使用已保存评价缓存。
2% 是未标定的筛查阈值，报告同时列出 1% 与 3% 敏感性。负向大变化与小幅平台分开。
初始化、只有采图的记录和无法确定日期／绑定的旧执行日志分别列出，不装成定量实验。

独立分析实际使用 Codex 子会话 `independent_failure_analysis`，`fork_turns=none`。
仅给 `analysis_packet.json` 和 `analysis_images.json` 列出的八张拼图；未传父历史、人工技能、旧策略理由或上一轮 RSI 结论。
子会话自行观察图片并写 `independent_analysis.json`，未生成策略代码。
先前准备的 `native_analysis.py` CLI 路径没有启动（审批等待期间中断），不是本报告的分析来源。

不自动运行任何建议实验。下一次真实执行仍需原有工作空间、碰撞、深度、路径及确认机制。
