# 腕部相机标定诊断

`scripts/check_camera_calibration.py` 使用保存的感知数据进行离线检查，不连接机器人或相机、不改标定、不执行动作。需要 numpy、Pillow；棋盘格检查还需要 OpenCV（项目 molmo/cali 环境提供）。

它分别报告：

| 检查 | 能回答什么 | 不能证明什么 |
|---|---|---|
| 内参及变换矩阵合法性 | 尺寸、焦距、主点、旋转矩阵是否有效 | 标定的物理准确性 |
| 深度＋内外参复算 XYZ | 保存的像素映射是否与计算一致 | 错误外参也可自洽 |
| 棋盘格保留角点重投影误差 | 当前针孔内参模型是否与标记图像相容 | 单视角拟合位姿可吸收部分内参误差 |
| 棋盘格深度边长误差 | 内参和深度组合是否恢复已知尺寸 | 单独区分内参与深度误差 |
| 固定点多姿态坐标漂移 | 腕部变换链是否随姿态产生不一致 | 所有姿态共同的绝对偏移 |
| 对照独立实测基座坐标 | 测试点处视觉落点的绝对误差 | 未测试区域的精度、夹爪 TCP 精度 |

没有独立实测点时，绝对准确性始终为 `NOT_VERIFIED`，不会因为“复算通过”就报告外参准确。

## 先检查 R081

```bash
python scripts/check_camera_calibration.py \
  --iteration /实际路径/iteration_002 \
  --reference R081
```

脚本用 `before_raw/camera_A_upright_mapping.json` 精确找到本轮动作前的 perception 目录，不选择最新 workspace。若 run 被搬迁，映射中的绝对路径失效，明确指定原始感知目录：

```bash
python scripts/check_camera_calibration.py \
  --perception /实际路径/results/perception/center_时间戳/result.json \
  --reference R081 --pixel 359 551
```

像素必须是**未旋转的原始 RGB 坐标**。当前 1280×720 原图中，upright `(168,359)` 对应原图 `(359,551)`。`--reference` 自动查原始坐标。查询中心像素缺深度时不会用邻域补出一个“有效”落点。

输出在 `results/calibration_checks/<时间戳>/`，可用 `--output-dir` 更改父目录：

- `report.json`：矩阵、地图复算误差、点深度、邻域深度 10/90 分位、原始及 Z 修正后基坐标、固定点漂移及绝对误差。
- `*_raw_rgb_points.png`：在原始 RGB 上画出待查点，避免误读旋转或缩放坐标。
- `summary.txt`：简要结果。

`base_xyz_mm` 是直接使用相机变换计算的坐标；`pipeline_xyz_mm` 加上感知保存的 `base_z_offset_mm`。与保存的 XYZ 图/Rxxx 比较时使用后者，不把高度修正误判成转换错误。地图默认误差容限 0.05 mm 仅用于数值一致性，不是实机精度要求。

## 检查内参及深度尺度

打印或使用平整棋盘格，实际测量格长，不依赖打印设置声称的尺寸。在多个距离、倾角和画面位置拍摄并保存感知数据，板子角点应清楚可见。尽量用未参与标定的数据。

```bash
python scripts/check_camera_calibration.py \
  --perception /capture_1/result.json --perception /capture_2/result.json \
  --checkerboard 7 5 --square-mm 20 --tolerance-px 1
```

`7 5` 是内角点数，不是黑白方格数。用交替角点估计板子位姿，用其余角点报告最大/RMS 重投影误差，同时比较深度恢复的相邻角点距离与实测格长。当前生产转换仅使用 fx/fy/cx/cy，没有镜头畸变修正，本脚本按同一针孔模型检查；不能凭单张正对棋盘格给内参背书。棋盘格角点自动排序不适合直接跨姿态认定同一个物理角点，固定点测试请明确指定有唯一标识的点。

## 检查腕部外参链和真实落点

在桌上固定一个明确的物理标记 P1，在多个安全的相机姿态保存 perception 数据。标记不能移动，相机姿态应有位置及转角变化；脚本只读取数据，姿态采集由现有受控流程完成。每次标出同一物理点在原始 RGB 中的像素。

建立 `observations.json`，路径相对该文件，或使用绝对路径：

```json
[
  {"point_id": "P1", "result_json": "capture_1/result.json", "pixel_xy": [359, 551]},
  {"point_id": "P1", "result_json": "capture_2/result.json", "pixel_xy": [402, 480]}
]
```

```bash
python scripts/check_camera_calibration.py \
  --observations observations.json --tolerance-mm 5
```

至少需要两个不同 capture，且保存的相机变换间有 ≥10 mm 平移或 ≥5° 转角才报告姿态漂移检查；这些是最低数据条件，建议采集 5–10 个覆盖实际工作区域及多轴转角的姿态。重复拍同一位姿会报告 `INSUFFICIENT_POSE_VARIATION`。漂移是各次预测基坐标之间的最大距离。

若已用**独立校准探针或其他可靠方法**测得 P1 基座坐标，在每条记录加入：

```json
"known_base_xyz_mm": [400.0, 100.0, 30.0]
```

此数值是格式示例，必须替换为真实测量；不能复制相机算出的 XYZ，否则只是自我验证。报告将给出每次 XYZ 误差向量以及总体最大/RMS 误差。5 mm 默认限值只是示例，应按夹持余量及参考测量不确定度调整。测试点结果在容限内也不代表所有区域合格。

如果预测随腕部转角漂移，优先检查手眼外参、机械臂运动学、采图与关节角时序；如果一致但相对实测点有共同偏移，检查绝对参考和坐标变换；如果视觉结果准确而夹持中心仍偏，独立核查 TCP。旧 run 没保存求变换时的确切关节角，只能检查当时保存的最终 `X_base_camera`，不能从记录拆解全部误差来源。

退出码：0 表示诊断完成（不等于标定认证通过）；1 表示输入/处理错误；2 表示发现地图不一致、棋盘格重投影/深度边长超限、固定点漂移或绝对误差超限。缺少参照、未检测到棋盘格和无效深度分别在报告里明确标记，不当作通过。

只有 RGB 照片不足以运行全部检查。外部采集数据也可组织成目录，放 RGB PNG、与 RGB 对齐的米制深度 `.npy`，再写 `result.json`，其 `views` 中至少包含 `label`、`image`、`depth_m`、3×3 `intrinsics`、4×4 `X_base_camera`（平移单位米）。必须填写该次静止采集实际使用的参数，不能把所有腕部姿态都套用一个外参矩阵。没有 XYZ 图或 Rxxx 表时仍可做棋盘格和固定点检查，但不要传 `--reference`。
