# Arm6 原版 One-shot 流程 + 原因分析

当前入口直接调用 `cloth_agent.experiments.one_shot_goal_full_episode_v1.main()`；不再运行独立 Skill 学习循环。原实现保留在 `legacy_skill_loop.py` 供追溯，当前入口仅复用其中请求审计、JSON存储和距离检查函数，不调用它的运行循环、Teacher 或 Planner。

```bash
bash scripts/run_live_skill_discovery_01.sh --execute
```

离线查看配置：

```bash
bash scripts/run_live_skill_discovery_01.sh --dry-run
```

## 实际流程

原版正常 reset → 最新 perception RGB + 静态目标图 → 原版 One-shot Planner → 原版 Host 检查与 shadow gate → 执行动作 → 原版返回观察位 → After → 独立原因分析 → 下一轮。

保留用户此前要求：默认6轮、单视角（无侧视相机运动）、开场reset中心采样半径20–40 mm、short_reason不限制句数、不等待人工确认。此前误加的Pa→Pb 20–40 mm约束已删除，原版单视角Prompt及牵拉10–150 mm检查恢复。每次新场reset在20–40 mm均匀抽取一个半径上限，再按原采样器在中心圆盘内选点；不是距中心至少20 mm的环带。半径记录在协议及初始化日志中。

启用原版 `--same-prompt-retry --retry-plan-feedback`：允许列表内的执行前拒绝消耗当前轮，下一轮重新观察并携带被拒计划与原因。不是所有错误都重试；执行异常仍按原版逻辑停止。DONE、ABSTAIN和轮数结束均沿用原版行为；6轮是上限，不保证6次动作。

不启用 sleeve/hem 模式、Top5、Skill 生成或 Skill 注入。原因分析不回传 Planner，不改变下一轮动作，也不影响终止判断；分析 UNKNOWN 或普通请求异常只记录，Ctrl+C / SIGTERM仍可终止。

## 两层独立反馈

每次动作完整执行后，顺序调用两次独立 GPT 请求，原始 Before/After 都保留。

第一层只看 Before/After，不提供 Planner 理由、目标或期望效果。按 Before 方位记录左上区域、中央衣身、右侧布片、下摆、新形变的变化、对应关系、图像证据与未知项；另记整体位姿变化和局部相对变化。不判断成功、失败、收益或机制。

第二层读取已保存的第一层记录、Before/After、原计划及执行状态，分别判断：

- 原定目标是否完成：YES / NO / UNKNOWN。
- 是否有可确认局部进展：YES / NO / UNKNOWN。
- 是否有可确认不利变化：YES / NO / UNKNOWN。
- 尚不可判断的结果，以及第一层与图像的冲突。

三个维度独立，没有整体成功/失败标签。目标未完成可以同时有局部进展；局部进展与不利变化也可能同时存在。证据不足必须UNKNOWN，不能转成失败、零收益或无进展。原目标不明确时目标完成项为UNKNOWN。没有训练标签数值化或自动填零逻辑，当前结果不用于学习。

每轮记录：

- `physical_observation/`、`physical_observation.json`、`physical_observation.md`：第一层实际请求、响应与区域变化表。
- `reason_analysis/`、`reason_analysis.json`、`reason_analysis.md`：第二层实际请求、响应与独立判断；引用第一层文件哈希。
- `reason_analysis_error.json`：失败发生在哪一层。第二层失败仍保留第一层，不补造标签。

两层结果均标记无人核验，不反馈给 Planner，不生成 Skill；UNKNOWN或普通请求失败不会阻止原版rollout，操作中断仍生效。历史记录不改写。

原版 `episode_summary.json`、`report.md`、`review.html`、实际轨迹等仍由原版生成；原版 Planner 的 DONE 终止逻辑未改变，反馈层不另给笼统成功结论。根目录另存 `reason_analysis_protocol.json` 与入口快照。每次已完成动作额外两次反馈调用及相应等待时间（6轮最多12次）。

可指定 `--max-rounds`、`--goal-image`、`--config`、`--output`。未传 `--execute` 不连接机器人。没有恢复或改写既有记录。

## 接着上一场，不设轮数上限

```bash
bash scripts/run_live_skill_discovery_01.sh --execute \
  --resume-from reports/live_skill_discovery_01_20261028_213305_911554 \
  --max-rounds 0
```

`0` 表示无轮数上限；仍保留原版 DONE、ABSTAIN、不可重试异常及 Ctrl+C 等终止条件，不自动开启下一场。续跑不reset，使用现场当前衣物状态和来源目标图，并核对原Prompt；仅允许从旧Prompt移除误加的20–40 mm牵拉语句，并在新记录中标记修正。新建独立记录目录、记录来源关联；不改写旧run，不把旧分析注入Planner。轮号在新目录从000开始。两层反馈和单视角不变；牵拉恢复10–150 mm。普通启动未指定 `--max-rounds` 时仍默认6轮。
