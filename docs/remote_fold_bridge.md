# Fold 远程 Claude 桥接

`scripts/claude_fold_exploration.py` 默认使用 `--planner-backend remote`，SSH 主机默认是 `company-planner`。规划、运动提案、执行后评估和 fold supervisor 都通过 HTTPS 图片中转 + SSH 调用公司电脑的 Claude。Alienware 不需要本机 Claude。显式指定 `--planner-backend local` 可以使用旧调用链。

公司端需要免密 SSH、`curl`、`sha256sum`、GNU `timeout`、`date` 和已登录的 `claude`。非交互 SSH 环境的 PATH 必须能找到这些命令。桥接不指定模型，使用公司端 Claude CLI 配置的默认模型。远端调用是独立会话，不续用 Alienware 的 Claude session。

## 快速测试

Alienware 安装本地 Molmo 袖部定位依赖，可在项目目录运行：

```bash
bash scripts/setup_molmo.sh
```

脚本创建或复用 Python 3.11 的 `molmo` Conda 环境，安装指定版本的依赖、下载 `allenai/MolmoPoint-8B` 并检查 CUDA、模型配置和处理器缓存。它不启动机器人，完成后会打印带有正确 `--molmo-python` 路径的折叠命令；完整模型推理仍需另行验证。

仅有一张 RGB PNG 时，先检查传输和公司 Claude 的图像读取：

```bash
python scripts/remote_planner_test.py test.png
```

`--mock` 可在不联网时测试传输命令和 JSON parser；它不验证公司 Claude。

已有成功保存的 perception run 时，运行实际 fold planner 链路：

```bash
python scripts/remote_fold_smoke.py \
  --run-dir runs/<已有的run目录名> \
  --host company-planner \
  --step left_sleeve \
  --mode ACQUISITION_PROBE
```

该测试需要该 run 的 `workspace/perception_views/` 中有匹配的 RGB、Rxxx guide、garment mask、XYZ 和本地表面测量文件；单张 RGB 不足以验证本地 grounding。它调用生产代码中的 supervisor 和 fold planner，用本地测量生成并检查 Proposal，写入 `results/remote_fold_smoke/<时间>/proposal.json`。测试不采图、不连接机器人、不做控制器 IK 或物理执行。只验证传输时使用第一个命令。

正常 fold 启动时保留原参数，或显式附加：

```text
--planner-backend remote --remote-planner-host company-planner
```

启动日志会显示 `planner_backend='remote'` 和 `remote_planner_host='company-planner'`。相机或 perception 阶段失败时仍会在 Claude 调用之前停止。

单腕部相机模式下，fold 会在 `--real --confirm-real` 已设置、配置只有一台活动相机且当前观测相机与配置一致时，向 session 传入 `single_view_confirmed=True`，无需额外命令行参数。相机模式在规划前检查，并在启动录像/执行前重新核对 run 元数据；双相机配置意外只得到单视角时仍会阻止执行。日志显示实际 `active_cameras` 和 `single_view_confirmed`。

## 数据与动作边界

远端仅收到白名单中的 RGB（包括 RGB 上的 Rxxx 标注、RGB 视频接触图）和筛选后的语义任务/历史。不会发送深度图、热图、XYZ、标定矩阵、工作区数值、完整机器人状态或本地文件清单。上传 PNG 原字节，远端下载后先验证 SHA256。

远端先选择现有 Rxxx，再显式提出完整动作序列。move 使用选定 grasp 或当前 upright RGB 的 pixel，加相对抓取高度与相对 Home yaw；这些是运动提议，不是远端测量的坐标。本地将 upright pixel 逆旋转回原图，查询保存的深度/XYZ；抓取高度仍由原 `resolve_grasp_height` 决定。不会补默认动作、猜缺失深度或把完整 fold 自动改成 probe。该远程接口要求 closure 精确对应选定 Rxxx，不使用旧本地生成器可选的偏移抓取。

原有 Rxxx 可执行性、任务模式、工作区、IK 和执行校验继续生效。网络/Claude/schema/grounding 失败不返回 Proposal；重试耗尽后远程 supervisor/evaluator 不使用旧的默认状态 fallback。每个远端 job 使用 UUID 临时目录、远端 EXIT trap 和本地 finally 的有界尽力清理。公网中转服务的文件按一小时有效期处理，删除远端目录不会删除公网副本。

## 离线回归

## 越界点离线诊断

每次远程规划尝试会在 `iteration_*/planning_attempt_*/` 下保存 `workspace_diagnostics.json`、`workspace_targets_raw.png`、`workspace_targets_upright.png` 和 `workspace_base_xy.png`。红色点表示本地工作区检查拒绝；JSON 同时给出动作编号、当前 upright 像素、标定后的机器人 Base XYZ（毫米）和左右边界有符号距离。图像只用于 Alienware 本地调试，不会上传给 Claude。基坐标图使用保存的标定结果，不能代替 IK 或标定精度验证。

也可以对历史失败调用完全离线重绘：

```bash
python scripts/debug_workspace_targets.py \
  --run-dir runs/<run> \
  --motion-json runs/<run>/results/fold_exploration/<stamp>/iteration_001/planning_attempt_fold_01/pixel_motion_invocation.json \
  --visual-json runs/<run>/results/fold_exploration/<stamp>/iteration_001/planning_attempt_fold_01/visual_planning_invocation.json
```

Viser 会按“Before / Workspace targets / Actual Claude RGB inputs / Rollout”分组显示图片；顶部 timing 面板显示每个阶段的细分耗时。日志里的 `+Xs` 是从运行开始的累计时间，`+Ys` 是相邻事件间隔，嵌套阶段不要相加。

```bash
python -m pytest -q tests/test_remote_fold.py tests/test_fold_exploration_pipeline.py tests/test_auto_exploration.py
```

这些测试使用模拟传输和保存的合成 RGB/几何数据，验证生产 planner 路由、本地 grounding、坐标逆旋转、失败拦截和清理；不会调用机器人或公司 Claude。
