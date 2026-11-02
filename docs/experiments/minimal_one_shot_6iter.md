# 最简 Prompt One-shot：每场最多 12 次迭代

当前上限已改为 12 轮；为兼容原命令，脚本名仍保留 `6iter`。

启动一场，先执行现有正常 reset：

```bash
bash scripts/run_minimal_one_shot_6iter.sh --execute
```

达到 12 次仍未输出 DONE 时，终端提示并响铃，退出码为 2，不自动 reset、不启动下一场。你手动 reset 后，重新启动时跳过自动 reset：

```bash
bash scripts/run_minimal_one_shot_6iter.sh --execute --no-reset
```

仅查看 Prompt 和运行设置，不连接机器人、不请求模型：

```bash
bash scripts/run_minimal_one_shot_6iter.sh --dry-run
```

每场是一个独立新目录 `reports/minimal_one_shot_6iter_<时间>/`。使用 `gpt-6-astra/high` 和现有 ChatGPT 账户的直接 Responses 调用，不启动 Agent CLI，不注入 Skill。Prompt 原文在 `--dry-run` 和每场 `prompt.txt` 中，只有展开目标、动作接口、坐标系和 ACTION/DONE/ABSTAIN 格式；没有部位策略或历史经验；重试时附带上一轮被拒绝的 plan 和拒绝原因。

每轮只输入主视角 perception 当前 RGB 和静态 goal 参考图（共两张图片，只有一个实时拍摄视角）；不读取侧视位姿、不做侧视移动、不拍侧视图，输出一次 Pa/Pb，夹爪沿用 Home 方向。每轮最多一次规划请求，不调用额外模型评审、结构识别、候选排序或 Codex 恢复。reset 保持现有 `initialize()`，其定位调用和物理动作单独计数，不占十二次规划迭代。默认沿用现有初始化半径 80 mm；可显式使用 `--initialize-center-radius-mm 20`。既有可选 reset 按压通过 `--press-after-drop` 启用，默认不启用。

12 次指最多 12 次规划/判定，不保证执行 6 个动作。DONE、ABSTAIN、非白名单错误或执行阶段错误会提前结束；白名单中的执行前拒绝占用下一轮重规划。第十二次若仍为 ACTION，执行后保存观测即停止，不额外发第十三次 GPT 判断 DONE。每次命令只运行一场，即使 DONE 也不自动开始下一场。

保留现有 Host、深度、IK、运动及执行检查，使用原 one-shot 的 reset 和运动实现。唯一关闭的 gate 是不参与执行决策的额外模型质量评审，记录中标为 DISABLED。所有覆盖仅在本入口进程内生效，不改写共享生产 Prompt 或安全配置。

输出包含 `episode_summary.json`、`operator_notification.json`、逐轮请求/原始输出、Before/After、动作图和执行记录。`operator_notification.json` 记录是否 DONE、轮数及是否等待手动 reset。模型 DONE 不自动认定为物理成功。

检查范围：8 项离线测试通过，涵盖十二轮参数、停止通知、手动 reset 状态保留、额外模型调用禁用、原 reset/Host 函数保持不变，以及模拟完整episode验证侧视路径不会被调用。尚未进行本入口的联网请求或实机验证。

```bash
/home/sja/miniconda3/envs/cali/bin/python -m unittest discover -s tests -p test_minimal_one_shot_6iter.py -v
```

## Prompt / Skill 审计

展开请求的instructions必须逐字等于本入口PROMPT，正常轮文字输入只允许图像尺寸和current/goal顺序；重试轮额外允许与该轮 `retry_feedback.json` 逐字一致的上一条被拒 plan、原因和未执行标记，schema也必须一致；若超出上述输入合同则在发送前拒绝。每轮保存request_content_audit.json。未注入人工Cloth Skill、C Skill、袖口/下摆策略、结构分析或历史操作。

正常reset保留原有独立的中心衣物区域定位提示；它用于抓起丢下初始化，不是展开Skill，不传入随后每轮的展开请求。

回归检查：相关套件64项通过、1项历史快照检查失败。失败项为`test_fast_prompt_only_adds_authorized_boundary_and_keeps_host_snapshot`：现有`gamepad_collect_runtime.py`哈希与旧实验快照不同；本次没有修改该运行时文件，也没有为通过旧快照测试覆盖它。新入口的8项离线检查全部通过。

## 执行前错误补丁（2026-10-28）

最简入口启用 `--same-prompt-retry`，复用原 `codex_recovery.recovery_category` 错误白名单：距离超限、抓点缺深度、目标深度拒绝、计划阶段 IK/工作区拒绝，以及白名单中的模型响应错误。下一轮先检查机器人故障与关节位置，再采集新的主视图，基础 instructions 和接口保持不变；按用户最新要求，重试输入附带上一条被拒绝的 plan（包括 Pa/Pb）、拒绝原因及 physical_execution=false，明确坐标属于上一轮观测。仅传递最近一次拒绝，不累计历史；不调用 Codex Recovery，不移动到其他视角。

所有重试计入 12 次总迭代，不增加同轮请求。第十二轮仍被拒绝则退出并提示手动 reset。硬件故障、取消、未知错误和已经开始执行的动作错误仍立即停止，不自动清故障或恢复运动。每轮记录 `same_prompt_retry.json`，后续轮次记录静止检查 `same_prompt_preflight.json`。

保留原有 10–150 mm 范围与全部 Host 检查；已有目标桌面深度回退继续按原条件工作。本补丁不会裁剪或缩短 Pa/Pb，也不保证模型下一次能避开同一个错误。旧版 Codex/Skill 或改变观察位姿的恢复流程没有启用，以保持此次实验约束。

本次错误补丁离线验证：`test_minimal_one_shot_6iter.py`、`test_codex_recovery.py`、`test_one_shot_goal_full_episode_v1.py` 共 55 项测试及 5 个分项检查通过。与出错场次 `minimal_one_shot_6iter_20261028_161102_776945/prompt.txt` 逐字比对一致；未启动机器人或联网调用。
