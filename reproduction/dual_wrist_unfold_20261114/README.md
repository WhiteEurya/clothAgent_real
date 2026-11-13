# 双腕相机 Unfold 复现快照（2026-11-14）

本目录对应分支 `repro/dual-wrist-unfold-20261114`，复用当前正式实验链。
不是 graph replay，不会训练或部署新的 Gate。历史实验脚本也保留在仓库中；请以本页入口为准。

## 本快照包含什么

- 双 xArm7 的 RobotAPI / ExperimentRunner 执行链、完整路径检查和连续运动实现。
- 两台腕部 RealSense + 固定 ZED 的 RGB-first topdown 拼接、来源像素/实测深度映射和 Grounded SAM2 分割。
- 当前 Simple、割线、切线、腾空间规划流程。几何 Skill 优先割线；不能可靠选两个端点或割线候选失败时才用切线。
- 当前现场标定、关节/工作范围/碰撞模型、perception 位姿、目标图、测试。
- 批次、人工复位确认、续跑、日志、32倍速录像。

这是软件与当前场地配置的复现快照，不保证每次视觉模型返回相同动作。标定矩阵、相机序列号、机械臂 IP 和位姿属于当前实验台；更换设备或安装位置需要使用对应标定，不能原样套用。本快照没有更改运动算法或运动保护。

## 环境

已使用的版本见 `environment.json`。运动环境为 Python 3.10，分割器使用独立环境，避免 PyTorch 与 JAX 依赖互相影响。

```bash
python3.10 -m venv .venv-unfold
.venv-unfold/bin/python -m pip install -e .
.venv-unfold/bin/python -m pip install -r reproduction/dual_wrist_unfold_20261114/requirements-runtime.txt
export CLOTH_PYTHON="$PWD/.venv-unfold/bin/python"
```

分割环境使用 Python 3.11 或更高（实际版本见 environment.json）：先安装适合 GPU/驱动的 PyTorch 与 torchvision，再安装：

```bash
/path/to/segmentation/python -m pip install -r reproduction/dual_wrist_unfold_20261114/requirements-segmentation.txt
export CLOTH_SEGMENTATION_PYTHON=/path/to/segmentation/python
"$CLOTH_SEGMENTATION_PYTHON" scripts/prepare_unfold_models.py
```

下载器使用 `models.json` 中固定的 Hugging Face revision，并校验与本次实验模型文件的 SHA-256。权重不放进 Git。再次离线检查可使用 `--verify-only`；正常分割运行只读本地权重。

另需：系统 FFmpeg、ZED SDK 及其匹配的 pyzed（装入运动环境）、RealSense USB 3 连接、Codex CLI。规划器调用已登录的 CLI `openai` profile，模型/推理强度记录在 environment.json；账号必须有模型访问权限。认证文件、API key 和 `.env.local` 不在仓库中。RBS 是可选传输，本复现入口不依赖它。

## 当前完整入口

以下命令都在仓库根目录执行。先检查配置加载（不会连接机器人、打开相机或请求模型）：

```bash
bash scripts/run_reproduced_unfold.sh --dry-run
```

`DRY_RUN_OK` 仅表示本地配置加载成功，不是实机可执行保证。正式启动会使用既有链路连接设备、张开夹爪、回 Home、到 perception 位姿采图，并执行规划动作：

```bash
bash scripts/run_reproduced_unfold.sh --runs 1
```

当前默认目标是白色黑猫衣服，路径：
`data/reference/white_black_cat_20261113_160203/goal.png`。
本入口明确指定三相机 stitched topdown 配置，避免误用旧的默认 topdown。

默认是**人工复位**：每个 run 开始前按提示确认 y/N，确认后跳过自动抓起丢下 reset。连续跑六轮：

```bash
bash scripts/run_reproduced_unfold.sh --runs 6
```

换衣服必须换相应展开目标，保持目标图与当前俯视坐标体系一致：

```bash
bash scripts/run_reproduced_unfold.sh --runs 1 --goal-image /absolute/path/to/goal.png
```

续跑中断的 run，不重新 reset；`--runs 5` 表示本次续跑加后续4个新 run。引用的是**具体 run 目录**，不是 batch 目录；目录参数保持一整行或加引号：

```bash
bash scripts/run_reproduced_unfold.sh --continue-from /absolute/path/to/run_directory --runs 5
```

达到20轮上限的 run 不能用 continue 再延长。一个 round 中腾空间后可能重拍并规划第二次抓放，因此 round 数不等于物理抓放次数。上限后另有一次只判定 DONE 的请求，不执行额外动作。

## 观测、模型与执行

正常请求使用当前 topdown、目标参考图，并在有历史时加入上一轮 topdown 与已执行动作记录；原相机 RGB 用于重建/深度/分割并保存。运行时的 `request_payload.json`、`planner_prompt.txt` 与附图是实际请求的最终依据，部分历史 protocol 描述可能仍提到 ZED 直接输入。

模型先选技能/点；几何技能由 CV 计算对侧交点与抓放点。点映射到实测深度后，执行链验证机械臂分区、工作范围、IK及碰撞路径。失败且未执行的候选可将错误交回规划器重规划；发生执行异常或设备故障仍会停止，不保证无人值守永不中断。

输出：`reports/formal_experiments/<batch>/<run>/`；录像：`videos/formal_experiments/`。规划、执行、动作后图、模型事件和异常均保留。原始实验 reports、视频、环境、凭据未上传。

## 无设备验证

```bash
"$CLOTH_PYTHON" -m pytest -q tests/test_unfold_skill_planner.py tests/test_unfold_batch.py tests/test_partitioned_unfold.py tests/test_dual_wrist_observation.py tests/test_dual_rgb_mosaic.py tests/test_grounded_sam2.py tests/test_openai_cli_planner.py
"$CLOTH_PYTHON" scripts/check_reproduction_geometry.py
```

`sample/` 是保存的真实 round002 图、mask、模型边界选择和原几何结果；检查脚本复用几何程序重算抓点，不访问设备、不调用模型。它仅验证像素几何复现，不替代实测深度和完整运动检查。

当前发布验证结果见 `validation.json`。其他历史分析脚本可能引用未分发的 reports 或旧环境；它们不属于上述正式复现入口。
