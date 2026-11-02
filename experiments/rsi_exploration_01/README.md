# RSI Exploration 01

连续训练探索：正常reset → reset后RGB和经验记录 → 2–3候选及预测 → 保存冻结预测 → 现有Host预检 → 独立选择 → 现有检查通过 → 仅执行一个动作 → 独立观察 → 逐预测核验及假设更新 → 下一轮最新RGB。

```bash
bash scripts/run_rsi_exploration_01.sh --execute
```

开场正常reset一次，之后每轮最多一个探索动作；reset单独计数（每轮单独计数，另含必要的准备和返回运动）。仅连接Arm6。未传`--execute`时不连接。不会调用生产Planner或改写其Prompt；不修改生产配置文件、不部署Skill。复用现有运行器和动作准备函数，10–150 mm及原检查保持。本实验进程启用现有完整路径验证，并让reset准备/返回路径也走同一原生验证入口；不调整碰撞模型、工作空间、关节或速度限制。有效配置保存在`runtime_policy.json`。任何检查拒绝或证明不完整都停止，不绕过。

按最新用户授权，reset及其准备/返回运动、所选探索动作均在检查通过后自动执行，不等待输入、不需要批准口令。`execution_authorization.json`记录会话授权、具体观测/预测/候选/轨迹绑定，并明确`per_action_human_review=false`，不伪造人工审批。`review.html`仍保留供回看。

场景变化、姿态变化、轨迹/输入变化、检查拒绝及执行异常仍停止。开场reset一次，明确DONE后下一轮再reset；每轮探索动作最多下发一次（包含释放和返回观察位）。成功完成分析后自动继续，默认不限轮数；Ctrl+C停止。NO_VALID_EXPLORATION、检查拒绝、执行/分析异常或关闭异常均停止，不自动重试。

候选少于2种、坐标微调冒充机制、不能确认不同机制、无可用候选，返回NO_VALID_EXPLORATION。选择比较Task Progress、Information Gain、Exploration Diversity，不使用成功率评分。所有候选预测先保存，其他候选始终标为未测试，执行后模型只能评估实际被选候选的预测。

默认经验来源是22:01、22:15、22:54三次运行中11条有完整反馈的真实动作。只读取动作和实际反馈，不读人工Skill、历史Expert Prompt或旧规则结论。历史模型标签明确无人核验，不把UNKNOWN当失败。没有历史冻结预测的记录会明确标注，不伪造预测。

可用`--memory /path/to/memory_after.json`指定下一场的假设记录。保留最近12条初始观察和8条探索假设更新，下一轮生成和选择都必须参考。每轮仅执行一个动作；下一轮自动读取上一轮memory_after.json。单次结果不产生Skill，也不永久排除某种机制。

保存实际模型请求、附件及原始输出、冻结预测、各候选动作图与Host检查、选择理由、审批、连续过程帧、Before/After、独立观察、逐预测SUPPORTED/REFUTED/UNKNOWN、假设记录和中文report.md。常规探索每轮5次GPT请求（当前图DONE检测、生成、选择、观察、更新），开场reset另有一次区域定位请求；模型配置gpt-6-astra/high，无重试。任何执行异常均单独记录，不算物理操作无改善。


## Reset抓点要求

直接复用`one_shot_goal_full_episode_v1.initialize`原有reset定位Prompt、REGION输出格式及`random_fold_reset.sample_grasp`，不另设衣身中心/左右小区域或二次距离筛选。模型定位一个靠近衣身中心的可靠布面圆形区域；以该区域中心为采样中心，在其可见圆内采样，并按实测世界XY距离限制半径。

每次reset随机选择20–40 mm的半径上限R，允许距离≤R（圆盘，不是圆环）。保留原采样器的有效深度检查及200次拒绝后使用已验证区域中心的行为。实际半径、采样中心、抓点与种子保存于`initialization/result.json`、`region.json`、`grasp.json`。请求原样转发并保存审计，不修改探索Prompt或原10–150 mm牵拉范围。

新脚本仅由用户在现场启动；默认无参数只打印说明，不连接机器人。

## 连续运行与续跑

默认`--execute`开场reset一次后无限循环；可用`--max-rounds N`限制轮数。每轮保存到`round_000/`等子目录，顶层status.json记录累计动作数与停止原因。

`--resume-from /path/to/completed_cycle`从完整轮次或暂停的连续run目录继续，继承最近完整假设记录及末轮已有动作/观察，跳过开场reset；现场应保持暂停后的衣物状态。续跑也会重新采集当前RGB并通过原检查，不重放旧动作。

## DONE及暂停续跑

每轮拍摄最新RGB后单独判断DONE/NOT_DONE/UNKNOWN。整件衣物主要区域可见且无明确大面积折叠、自覆盖或卷折才允许DONE；出画、遮挡等无法判定时UNKNOWN。DONE不下发展开动作，下一轮按原reset流程重置后重新观察，保留经验；UNKNOWN不触发reset。DONE是模型视觉判定，不宣称实机成功率。

`--resume-from`可指向暂停的整个run。新建续跑目录并保存resume_memory.json，原目录保持不变。末轮已执行但分析中断时，保留已有动作和物理观察，未知结果不补成失败；新轮重新拍照规划，不重放旧动作。末轮有下发记录却不能确认EXECUTED时拒绝自动续跑。
