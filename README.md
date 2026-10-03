# ClothAgent

ClothAgent 是一个面向 xArm7 和双 RealSense 相机的布料操作实验框架。它把视觉、动作规划和机器人执行串成一个可审查的流程：

```text
相机 A/B RGB-D → 点云融合 → Claude 生成动作 → 静态检查与 IK → 模拟或一次真实执行
```

项目适合做布料抓取、展开和折叠相关实验。每次运行都会保存相机数据、生成的动作程序、检查结果和执行日志，方便复现与排查问题。

## 主要功能

- 使用两台已标定的 RealSense 相机进行 RGB-D 融合，估计布料区域和表面高度。
- 让 Claude 根据当前画面提出动作计划。
- 只允许生成受限的 Robot API：`move`、`open_gripper`、`close_gripper`、`home`。
- 在动作执行前检查 Python AST、工作空间、速度限制、轨迹和 xArm 控制器 IK。
- 支持模拟运行、Viser 可视化预览，以及需要明确确认的真实机器人运行。
- 保存完整的运行证据，便于人工标注成功/失败并继续下一轮实验。

## 环境要求

- Python 3.10 或更高版本
- 本项目依赖：NumPy、Pillow、PyYAML、SciPy、Viser、yourdfpy
- 使用真实相机时：Intel RealSense SDK（`pyrealsense2`）和两台已标定的相机
- 使用真实 xArm 时：xArm Python SDK（`xarm`）和可连接的 xArm7
- 使用 Claude 规划时：系统中可调用 `claude` 命令
- `semantic_local` 兼容模式还需要 PyTorch、Molmo 和可用 GPU；推荐模式是 `claude_global`

安装 Python 包：

```bash
python -m pip install -e .
```

如果运行环境已经提供了项目所需的 RealSense、xArm 或 Claude 工具，直接使用该环境的 Python 即可。

## 配置

第一次使用前，请检查以下文件：

| 文件 | 用途 |
| --- | --- |
| `config/robot.example.json` | xArm IP、速度、夹爪和安全参数 |
| `config/perception.example.json` | 相机序列号、分辨率和外参路径 |
| `config/perception.free_exploration.json` | 自动展开流程使用的视觉配置 |
| `config/experiment.example.json` | 手工实验参数模板；使用感知时可保留为 `null` |
| `xarm_boundaries.json` | 已测量的机器人工作空间边界 |
| `data/robot/xarm_init_pose.json` | Home 位姿 |
| `data/robot/xarm_perception_pose.json` | 独立观察点 / 感知位姿 |

如果机器人、相机或夹爪移动过，请重新检查边界、位姿、相机外参和 TCP 偏移。

默认用左右两点设定边界，无需判断底座 X/Y 方向：

```bash
/home/sja/miniconda3/envs/cali/bin/python scripts/record_xarm_boundaries.py \
  --ip 192.168.2.232 \
  --output data/robot/xarm_boundaries_new.json
```

1. 用 UFactory 手动操作机械臂，保持夹爪朝向，沿关心的左右方向将 TCP 移到第一侧安全极限，停稳后按回车。
2. 移到另一侧安全极限，停稳后按回车。两点顺序不限，两点应沿需要限制的方向选取，不要斜着跨到不同前后位置。
3. 程序显示两侧间距，输入 `SAVE` 保存；输入 `q` 取消选点不会修改文件。覆盖已有输出时自动备份。
4. 将 `config/robot.example.json` 的 `boundaries_file` 改为新文件路径，重新加载配置。

此模式只读机器人位置，不使能电机、不切换模式。两点的 XY 连线定义固定在桌面上的限制方向，运行时用点在该方向上的投影检查是否越界。夹爪转动不会旋转或放宽这两侧限制。记录的是 **TCP 安全极限**，选点时应给夹爪本体留出空间。

新文件以 `boundary_mm.lateral_points_mm` 保存两点，替代旧 X/Y 轴限制；与连线垂直的水平移动不受这两侧限制。两侧程序不询问、不修改已有的高度下限。Home 和观察点不变。这是软件 TCP 边界，不是控制器碰撞保护，也不定义前后方向的操作策略。

Z 最低高度用另一个程序独立设置（不设上限）：

```bash
/home/sja/miniconda3/envs/cali/bin/python scripts/record_xarm_z_bounds.py
```

手动将 TCP 移到最低允许高度，按回车，再输入 `SAVE`。这个程序更新 `z_min` 并清除已有的 `z_max`，保留左右两侧边界，也不会使能或移动机械臂。

两个程序默认更新同一份 `data/robot/xarm_boundaries_new.json`，运行顺序不限。首次运行不自动沿用旧环境的边界；如果输出文件已存在，则保留另一项已经设置的值和采样记录。左右和高度都完成后，边界才可用于真实运行。使用自定义文件时，两个程序的 `--output` 必须一致。

双相机感知要求 A、B 两台相机都能提供有效深度；当前版本不支持单相机模式。请确认 `config/perception*.json` 中的序列号和外参路径与实际安装一致。

注意：示例配置中的 Camera B 外参默认指向上级目录的 `RobotCamCalib` 项目。如果本机没有该目录，请把 `extrinsics_file` 改成实际的标定文件路径。

## 快速开始

下面的步骤从不移动真实机器人开始。

### 1. 只运行感知

创建一个运行目录，并采集双相机数据：

```bash
python -m cloth_agent create \
  --run-id preview_01 \
  --goal "locate the garment center" \
  --robot-config config/robot.example.json

python -m cloth_agent perceive \
  --run-dir runs/preview_01 \
  --perception-config config/perception.example.json
```

结果保存在 `runs/preview_01/results/perception/`，包括 RGB、深度、融合点云和高度图。该步骤会连接相机，但不会连接或移动 xArm。

### 2. 运行一轮完整的模拟流程

该命令会采集感知、调用 Claude 生成实验程序、打印动作计划，并在模拟器中执行：

```bash
python -m cloth_agent session \
  --goal "grasp the garment, lift it, release it, and return home" \
  --intent "Inspect the A/B views and create a cautious grasp-lift-release plan with explicit move coordinates." \
  --detect-center \
  --robot-config config/robot.example.json
```

运行前会显示生成的 Python 源码和每个 `x/y/z/yaw` 动作。确认计划是否符合预期后，再进行下一步。

### 3. 打开 Viser 预览

```bash
python -m cloth_agent viewer --run-id viewer_01
```

然后打开 <http://127.0.0.1:8080>。Viser 可以查看相机画面、点云、动作路径、IK 和 URDF 动画。默认不会开放真实执行按钮。

### 4. 运行真实机器人（谨慎）

只有在确认相机、边界、TCP 和急停均已准备好后，才使用 `--real`：

```bash
python -m cloth_agent session \
  --goal "grasp the garment, lift it, release it, and return home" \
  --intent "Inspect the A/B views and create a cautious grasp-lift-release plan with explicit move coordinates." \
  --detect-center \
  --robot-config config/robot.example.json \
  --real
```

程序会先打印完整计划，不会立即发送动作。只有在终端中输入 `EXECUTE` 后，才会执行这一轮。每轮真实运行结束后，程序都会尽力调用 `home()` 返回观察位。

## 常用命令

RealSense 实时画面和滑块调参（曝光、增益、白平衡等）：

```bash
conda activate cali
python scripts/tune_realsense.py
```

手动打开 <http://127.0.0.1:8086>，默认选择 Cam A。支持保存参数、截图，以及将曝光 / 白平衡写回折叠配置。[详细使用说明](docs/realsense_tuner.md)

当你希望把流程拆开执行时，可以使用以下子命令：

```bash
# 在已有运行中调用 Claude 生成实验程序
python -m cloth_agent generate \
  --run-dir runs/preview_01 \
  --prompt "Create a cautious grasp, lift, release, and home sequence."

# 执行前检查，不移动机器人
python -m cloth_agent preflight \
  --run-dir runs/preview_01 \
  --experiment experiment_001_grasp_lift_drop.py

# 在模拟器中运行
python -m cloth_agent run \
  --run-dir runs/preview_01 \
  --experiment experiment_001_grasp_lift_drop.py

# 查看结果或源文件
python -m cloth_agent inspect \
  --run-dir runs/preview_01 \
  --experiment experiment_001_grasp_lift_drop.py
python -m cloth_agent inspect \
  --run-dir runs/preview_01 \
  --file memory.md
```

真实执行的拆分式命令必须同时使用 `--real --confirm-real`，并且应先单独运行 `preflight` 检查。

### 连续自动探索（可选）

如果需要让 Claude 连续提出多轮“展开布料”动作，可先运行一轮不连接真实 xArm 的 dry run：

```bash
python -m cloth_agent.molmo_keypoint_cli \
  --run-id explore_01 \
  --planning-policy claude_global \
  --perception-config config/perception.free_exploration.json \
  --robot-config config/robot.example.json \
  --max-iterations 1
```

确认结果后，再额外加入 `--enable-real` 开启真实执行；连续运行可将 `--max-iterations` 设为 `0`。详细参数见 [FREE_EXPLORATION.md](FREE_EXPLORATION.md)。

## 安全机制

真实机器人是可选功能，默认关闭。执行前会进行多重检查：

- 生成的实验文件只能包含一个 `run()` 函数和四个受限动作，不能导入模块、访问文件或运行 Shell。
- 所有坐标必须是有限数值，并通过已测量的工作空间和 Z 高度边界。
- 速度、加速度和夹爪参数有硬上限。
- 对每个 Cartesian 目标执行只读的 xArm 控制器 IK 检查。
- 检查控制器 TCP 偏移是否仍与配置一致。
- 真实运行需要操作者明确确认；检查失败时不会发送机器人命令。

这些检查不能替代现场安全规范。连接真实设备时，请始终准备急停，并在低速下观察第一轮动作。

## 运行结果

每次运行的主要文件位于 `runs/<run-id>/`：

```text
runs/<run-id>/
├── run_metadata.json
├── workspace/
│   ├── ROBOT_API.md
│   ├── robot_config.json
│   ├── experiment_config.json
│   ├── experiment_*.py          # Claude 生成的受限实验程序
│   ├── memory.md                # 人工结果与下一轮假设
│   └── results/claude/           # Claude 调用记录
└── results/
    ├── perception/              # 相机、深度、点云和高度图
    ├── experiment_*.json        # 执行结果
    ├── experiment_*.stdout.txt  # 标准输出
    └── experiment_*.trace.json  # 动作与错误轨迹
```

运行结束后，可以用 `label` 记录人工结果（`SUCCESS`、`FAILED_GRASP`、`FAILED_LIFT` 或 `OTHER_FAILURE`），再用 `memory` 保存下一轮实验的假设。

### Token 消耗统计

本地和远程 Claude 调用会自动追加到 `runs/<run-id>/results/token_usage.jsonl`。
包括代码生成、视觉规划、动作生成、评估、折叠监督、抓取检查和经验更新；每次重试独立记录，重启运行后保留累计记录。

```bash
python -m cloth_agent token-usage --run-dir runs/preview_01
python -m cloth_agent token-usage --run-dir runs/preview_01 --json
```

统计包含输入、输出、缓存读取、缓存写入 token 和 Claude 返回的美元费用；JSON 还提供按阶段、模型的汇总。
`total_tokens` 为四类 token 之和（含缓存），费用直接采用返回值，不按单价估算。
只统计 CLI 最终返回的用量，避免流式事件、模型明细与总量重复相加。
失败调用若返回用量也会计入；缺失字段、超时未返回用量的调用标为未知，汇总中的 `missing_fields` 显示缺失次数，此时数值仅为已知部分。
该功能从启用后开始记录 Claude 用量，不回填历史日志，也不估算本地 Molmo 的 token。

## 自动探索与专题文档

需要连续展开、折叠或 Molmo 兼容流程时，再查看：

- [FREE_EXPLORATION.md](FREE_EXPLORATION.md)：Claude 驱动的布料展开流程
- [LANGUAGE_SKILL_PIPELINE.md](LANGUAGE_SKILL_PIPELINE.md)：语言/技能管线
- [GARMENT_PERCEPTION_PROBLEM_INVENTORY.md](GARMENT_PERCEPTION_PROBLEM_INVENTORY.md)：感知问题记录
- [data/reference/flat_garment_reference/README.md](data/reference/flat_garment_reference/README.md)：平铺布料参考图说明

推荐先用 `claude_global` 做一轮预览，再考虑启用自动或真实执行。

## 离线 Harness Policy 实验

独立入口 `python -m cloth_agent.harness` 支持 `collect`、`compile`、`replay` 和 `experiment`。系统自动整理 learning run，调用运行时 Claude 编译受限视觉选点 policy，校验冻结后做隔离历史答案的离线回放。不会接入主循环或执行机器人动作。命令、数据要求、统计口径和局限见 [离线实验文档](docs/offline_harness.md)。

## 测试

```bash
python -m pytest
```

测试通常不需要连接真实机器人；涉及相机、xArm 或 GPU 的脚本应在对应硬件/环境中单独运行。

## 固定证据上的脑内 Harness Learning

新增独立入口 `python -m cloth_agent.harness.reasoning_learning`：冻结一份选点前证据，由运行时 Claude 反思并生成多版 reasoning harness，重新规划同一 observation，通过抓点与视觉目标共识筛选簇内成本最低版本。支持串行/分支搜索、重复测量、预算、完整调试产物及 HTML 报告；不执行机器人。见 [运行与调试说明](docs/reasoning_learning.md)。单独测试：`python -m pytest -q tests/test_reasoning_learning.py`。

## Candidate patch evolution

独立入口 `python -m cloth_agent.harness.patch_evolution` 支持运行时 Claude 提出 PROMPT/HARNESS/SKILL_CODE/NEW_TOOL patch，写入隔离候选目录，通过强制测试后重新规划并比较；达标才进入离线 working。ObservationHost 已改用版本化 SkillRegistry，候选代码只在受限 observation 接口内运行。源码、测试、diff、坐标不变量、预算、共识与晋升均留 debug；不执行机器人。见 [运行与审核文档](docs/candidate_patch_evolution.md)。
