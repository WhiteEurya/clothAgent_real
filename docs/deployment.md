# 新机器部署与多衣服实验

本说明对应当前源码迁移。历史 Graph 数据复现仍见 `reproduction/README.md`。
新实验不需要下载全部历史 `runs/`、`reports/`、模型缓存或复现导出副本。
仓库提供实验代码，不包含账户凭据、机器现场配置、全部历史训练/检索样本或模型权重。

## 方法范围

当前方法尚未验证任意衣物泛化。`geometry_report.garment_mask` 和 CV feedback
使用紫色阈值、固定横向 ROI；GGO 的七个部位标签和多个提示词假设 T 恤结构。
换颜色时优先测 RGB-only 基线；换成裤子、裙子等非 T 恤需另外设计任务/部位定义，
不要把旧提示和几何规则的结果解释成通用衣物能力。RGB-only 仍用实测深度、
运动规划和现有 Host 检查执行动作，并不绕过机器人安全检查。

## 安装

推荐 Linux x86_64、Python 3.10（与既有运动规划环境一致）。在仓库根目录执行：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r requirements-experiments.txt
export CLOTH_PYTHON="$(command -v python)"
export CLOTH_RUN_STORAGE_CONFIG="$PWD/config/run_storage.portable.json"
python scripts/check_deployment.py
```

`requirements-experiments.txt` 是功能依赖清单，不是逐包环境锁。
`reproduction/requirements-cali.txt` 是旧 Graph 快照的环境记录，不代表当前所有实验。
RealSense 还需要宿主机 SDK/USB 权限；xArm 需要正确网段与设备连通。
手柄另装 `requirements-gamepad.txt`。DINO/Molmo/Matcha/RoMa 等可选 GPU 方法仍按对应
`docs/experiments/` 安装模型，不是上述 RGB-only 基线的必要条件。
模型服务使用新机器自己的登录和访问权限；部分实验读取当前用户的模型 CLI 登录状态，
并非任意 API key 都兼容。部署检查不验证模型权限，也不发起模型请求。

`CLOTH_RUN_STORAGE_CONFIG` 为显式存储配置覆盖。示例将中央运行目录设为仓库下
`runs/`，不要求旧机器的 `/mnt/newssd`。未设置时保留旧配置及其挂载检查。
脚本的 `--output` 仍决定各实验自己的输出位置；不是所有实验都走中央存储函数。

## 现场配置

```bash
cp -n config/dual_arm.deployment.example.json config/dual_arm.local.json
```

模板保留原现场完整字段供迁移，但标定状态设为 `incomplete`，碰撞验证和 servo
commissioning 标志未启用。核对机器人的型号/IP、相机序列号及所属臂、TCP、外参、
世界坐标、工作空间、碰撞模型、Home/观察位/闲置臂位姿。换电脑但同一套固定硬件可以
复核原测量；换现场、移动相机或底座须重新测量。不能只改验证标志就认为完成标定。
`config/CALIBRATION.md`、`docs/codex_arm7_pipeline.md` 及相关机器人流程提供背景。
`data/robot/` 和 `config/calibration/` 中的记录均来自原现场。

换相机/桌面时，放置点桌面深度回退还需配置 `CLOTH_CAMERA_INTRINSICS`（新相机的
内参 YAML）和 `CLOTH_TABLE_APPEARANCE_CONFIG`（含桌面 ROI/平面参数的感知 JSON）。
未设置时仍使用原现场文件；序列号、图像尺寸和实测点云重投影检查保持生效。

## 每件衣服的实验输入

在新现场为每件衣服采集同视角的展平目标图，保存到例如
`data/garments/shirt_001/goal.png`；测试中固定相机、目标图与方法参数。
不要复用仓库中的紫色衣服目标图作为另一件衣服的任务目标。
`data/reference/deployment_example/` 只是可校验的旧衣服示例，不是新现场标定。

下面的 RGB-only 单次试验使用人工整理的初始状态。**带 `--execute` 的命令会运行实机**：

```bash
python -m cloth_agent.experiments.automatic_top5_batch \
  --execute --visible-sleeve-hem --stateful-context --rgb-only \
  --no-reset --episodes 1 --max-rounds 13 --max-actions 12 \
  --config config/dual_arm.local.json \
  --goal-image data/garments/shirt_001/goal.png \
  --output reports/generalization/shirt_001/rgb_only_trial_001
```

换衣服、试次、方法时使用新的输出目录。`--no-reset --episodes 1` 便于固定人工初态；
删除 `--no-reset` 会使用已有机械抓起/落下 reset。不要把不同初始化方式混为同一组。
每件衣服多次重复，记录材质、颜色、尺寸、衣型、初态、目标图 SHA256、提交版本、
模型版本、成功判据、动作数、停止原因和人工干预；区分模型 DONE 与人工评定成功。

简短提示基线的离线协议检查（不连接硬件、不调用模型）：

```bash
bash scripts/run_minimal_one_shot_6iter.sh --dry-run
```

该历史文件名含 `6iter`，当前代码实际上限是 **12 轮**。运行时可传 `--execute`、
`--no-reset`、`--goal-image`、`--config`、`--output`，但其提示词仍指定 T 恤。

## CV / GGO 对比

CV feedback 入口为 `scripts/run_cv_feedback_rollout.sh`，传入当前衣服的
`--goal-image`、`--config`、`--output`，并显式 `--execute`。
它仍有紫色分割先验，不能作为不同颜色上的通用分割方法。

GGO 除目标图外还需要当前衣服在同一标定下的展平 RGB-D 观测目录，包含
`rgb.png`、`world_xyz_mm.npy`、`observation.json`，以及七点参考标注。
观测应由现有采集流程生成，保留 SHA256 和 calibration_id，不手写伪造深度或标定。

```bash
export CLOTH_GEOMETRY_REFERENCE="$PWD/data/garments/shirt_001/reference"
python -m cloth_agent.ggo.cli annotate \
  --reference "$CLOTH_GEOMETRY_REFERENCE" \
  --current "$CLOTH_GEOMETRY_REFERENCE" \
  --out data/garments/shirt_001/annotation_ui
```

具体 CLI 参数以 `python -m cloth_agent.ggo.cli --help` 为准。按生成的页面/模板标注参考点，
使用对应图像哈希；自动锚点模式会在现场图中重新识别三个上半身锚点。

```bash
bash scripts/run_ggo_rollout.sh --execute --no-reset --episodes 1 \
  --config config/dual_arm.local.json \
  --goal-image "$CLOTH_GEOMETRY_REFERENCE/rgb.png" \
  --ggo-reference-annotations data/garments/shirt_001/annotations.json \
  --output reports/generalization/shirt_001/ggo_trial_001
```

参考目录必须在进程启动前设置。图像哈希和标定不匹配会被拒绝；自动锚点不意味着
可免除参考标注，或已支持非 T 恤衣型。

## 其他实验代码

- `cloth_agent/experiments/`：在线展开、恢复、图策略、轮廓抓取及离线分析。
- `cloth_agent/cloth_harness/`、`cloth_agent/ggo/`：反馈评估和几何观测模块。
- `experiments/live_skill_discovery_01/`、`rsi_exploration_01/` 等：技能发现入口。
- `experiments/rax_cloth/`：单独安装 `python -m pip install -e experiments/rax_cloth`，
  设置 `RAX_CLOTH_PYTHON="$CLOTH_PYTHON"`，执行其 `scripts/bootstrap_upstream.sh`
  下载固定提交的上游代码；按该目录说明准备系统依赖与现场配置。
- `experiments/` 中以日期命名的离线诊断/挖掘脚本保留历史路径和证据需求。
  源码已保留不代表历史数据已打包；它们不是新现场的统一启动入口。

RAX 的 onsite 启动脚本会尝试机械 reset，不用于部署自检。
可先运行以下离线测试；部署检查和测试均不代表新机器的物理闭环已验证：

```bash
python -m pytest -q tests/test_run_storage.py tests/test_ggo.py tests/test_minimal_one_shot_6iter.py
```
