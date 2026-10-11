# Astra 单轮代码规划接口

默认入口现已升级为[完整实机循环](codex_astra_loop.md)。本文描述通过 `--single` 保留的单轮离线接口。

本机 `codex exec --model gpt-6-astra -c 'model_reasoning_effort="high"'` 接收简短衣服任务，自主查看项目和观测、确定抓点，最终输出 Python 动作代码。宿主解析代码后交给昨天的 `motion.PathPlanner`、时间参数化和连续验证。不会 `exec` 模型代码。

```bash
/home/sja/miniconda3/envs/cali/bin/python scripts/codex_astra_pipeline.py --single \
  --scene /path/to/collision_scene.json --state /path/to/current_joints.json
```

`--scene` 使用原运动规划器的 CollisionScene JSON；不能用 `dual_arm.local.json` 替代，也不会默认加载演示场景。
`--state` 是现场读取的起始关节角：`{"units":"rad","left":[六轴弧度],"right":[七轴弧度]}`。
不提供这两个文件时仍可生成代码，结果为 `CODE_READY`，不会宣称已规划。

支持的代码形式：

```python
# 此处仅展示语法，数字不是可执行的现场目标。
motion.move_tcp(left=[0.3, 0.1, 0.2, 3.14, 0, 0])
motion.gripper(arm="left", action="close")
motion.move_joints(left=[0, 0, 0, 0, 0, 0], right=[0, 0, 0, 0, 0, 0, 0])
```

移动可省略不动的一侧。TCP 使用场景 World 坐标的米和弧度；关节为弧度。夹爪 action 为 open/close。
缺少证据时返回 `motion.blocked(reason="...")`。不接受 import、SDK 调用、循环或任意表达式。
这里的代码是声明动作意图的 Python 子集，找抓点和动作顺序仍由模型决定，不套用旧 Claude 抓点 JSON。

`move_joints` 使用顺序关节路径，`move_tcp` 使用原规划器的笛卡尔路径。
时间参数化使用 5°/s、10°/s²，速度同时不超过 URDF 上限；连续验证通过后才返回完整规划。
夹爪事件按轨迹节点保存。当前通用规划器仍为离线接口，**不发送真实动作**；昨天已接通真机的是 Home，不能据此认为任意折叠轨迹也已接通。

```bash
python scripts/codex_astra_pipeline.py --single --dry-run
python scripts/codex_astra_pipeline.py --single --task '用双臂展开衣服，再折叠整齐。'
python scripts/codex_astra_pipeline.py --single --read-only --task '分析当前项目的双臂能力。'
```

输出 `results/codex_astra/<UTC>/`：`request.json`、`prompt.txt`、`events.jsonl`、`stderr.txt`、`answer.txt`、`motion.py`、`motion_plan.json`（提供规划输入时）、`result.json`。
记录耗时和已完成 turn 的 token 用量；中断时用量可能不完整。
`PLANNED` 表示离线规划通过，`PLANNING_FAILED` 表示未生成可用轨迹，`BLOCKED` 表示模型缺少任务依据。
`task_success: null` 不代表物理任务成功。既有本机 Codex 配置和项目指令仍生效，完整模型上下文不只有任务提示词。
