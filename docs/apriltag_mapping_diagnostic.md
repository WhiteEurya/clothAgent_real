# AprilTag 多姿态坐标诊断

使用你现有的 `RobotCamCalib/assets/apriltag_grid/compact_apriltag_grid_4x4_tag48mm_a3.pdf`，已确认缩印到 A4。无需重新制作标记，也无需用夹爪逐点对准。

仓库内 `config/apriltag_board_4x4_tag48mm.yaml` 是原目录板定义的副本：tag36h11、4×4、ID 0–15，原始黑色边长 48 mm。检测严格沿用 `pupil_apriltags.Detection.corners` 顺序及原 YAML 的板坐标，**没有重新猜测角点方向或 ID 排布**。

默认使用原 `extr_calib.py` 的 A3→A4 比例 `0.7071067811865476`，黑色码边长约 **33.941 mm**。README 中原尺寸 A3 的说明与该程序默认值不同；这里按你确认的 A4 使用。打印机“适合可打印区域”可能再次缩放，默认值不是对实物尺寸的独立认证。桌面另一份名为 `apriltag_25h9...pdf` 的文件不适用本默认板。

## 运行

进入仓库，在具备相机依赖的 `cali` 环境执行：

```bash
python scripts/diagnose_apriltag_mapping.py
```

需要 OpenCV、pupil-apriltags、Matplotlib、PyYAML、RealSense SDK、xArm SDK 和 yourdfpy。本地 `cali` 环境已验证具备检测和绘图依赖。

1. 将 A4 标定纸贴平并固定在桌面，整个采样过程不能移动板子。
2. 用示教器/厂商界面把腕部相机移到能清楚看到整板的位置。**脚本从不发送运动、使能、模式或夹爪指令**，不自动回观察位。
3. 保持静止，终端按 Enter，连续采集三张。脚本自动检测板子、保存 RGB/深度/关节角、计算误差。
4. 每组完成后，再手动换观察位置或腕部角度。建议 5–8 组，既有左右/远近位置变化，也有不同轴的倾斜，始终看得到至少四个清晰 Tag。不要只重复同一个姿态；略带倾角有助于避免正对平面时的 PnP 二解歧义。
5. 输入 `q` 结束。每组结束都会更新报告和图像，Ctrl+C 也保留已经完成的样本。

终端会打印输出目录。默认保存在 run storage 的 `tools/apriltag_diagnostics/<日期>/<时间戳>/`。可使用 `--output-dir /your/path`、`--repeats 3`、`--camera A`、`--robot-config PATH`、`--perception-config PATH`。

## 看什么结果

打开输出目录的 `diagnostic.png`：

| 图 | 含义 |
|---|---|
| 固定板中心 XY 散点 | 相同标定板被不同观察位算到了哪里；理想情况聚在一起 |
| dX/dY/dZ 曲线 | 每次相对第一张有效观测的基座坐标变化，单位 mm |
| RGB 重投影误差 | 全板拟合误差，以及用一半 Tag 拟合、另一半 Tag 检验的误差 |
| 深度减 RGB 板估计深度 | RGB＋已知板几何与 RGB-D 深度是否一致，单位 mm |

`report.json` 另有同组重复采样散布、相机姿态变化范围、最大两两板中心漂移及板姿态角漂移、深度比例、候选排查项。

每张 `sample_XXX/detections.png` 标出绿色检测角点、红色拟合角点及误差连线。`capture.json` 保存采样前后 TCP、关节角（弧度）、读取耗时和实际采用的 `X_base_camera`。`session.json` 保存内参及畸变系数、外参原文、URDF 哈希、板定义哈希和缩放比例。每个接受的 capture 还有兼容先前离线校验工具的 `result.json`。

采图使用新鲜对齐 RGB-D 单帧，前后夹着机器人状态读取，并连续读取四帧清除旧帧；用前读关节角通过与生产管线相同的 URDF 链计算相机外参。前后关节变化超过 0.2°、TCP 平移超过 1 mm、TCP 角度变化超过 0.5°或反馈读取超过 0.5 秒，样本标记 `REJECTED_MOTION`，保留照片但不计入几何诊断。这不是硬件同步，拍照中移动后又回原位仍可能漏检。

## 结果不能被误读成什么

- 少于四个有效 Tag、检测 ID 重复或 PnP 失败：不计算有效几何结果。
- RGB 拟合/保留 Tag 检验 RMS 大于 2 px，或平面二解接近且方向差异明显：标记 `RGB_POSE_UNRELIABLE`，不把它的基座漂移当成可靠手眼证据。
- 至少三个有效姿态组，且相机范围至少 10 mm 平移或 5°转角，才报告 `CONSISTENCY_MEASURED`；否则明确显示样本或姿态变化不足。它不等于认证“通过”。
- 大于 5 mm 的漂移/深度残差、2°的板姿态漂移只作为排查提示，不直接判定哪一个参数坏了，也不触发自动修正。
- 板的打印比例错了可能保持很低的 RGB 重投影误差，却产生错误毫米尺度；深度比例异常也可能来自深度误差，不能二选一直接归因。
- 本脚本使用与生产管线相同的无畸变针孔模型；保存运行时畸变信息但不直接混用不同 RealSense 畸变模型。较大/分布性 RGB 残差需要进一步检查内参、畸变、纸张平整度与板布局。
- 多姿态漂移可能来自手眼外参、URDF/FK、时序、板移动或 RGB 位姿误差。没有独立基座真值，绝对准确性始终 `NOT_VERIFIED`；实体夹指/TCP 精度始终 `NOT_TESTED`。

## 同一批数据离线重算

```bash
python scripts/diagnose_apriltag_mapping.py --session /path/to/session
```

离线模式不导入相机/机器人 SDK 来连接硬件，默认使用该会话保存的板定义和比例，输出到会话内新的 `reanalyzed/<时间戳>/`。可显式调整假设进行比较，例如：

```bash
python scripts/diagnose_apriltag_mapping.py \
  --session /path/to/session --board-scale 1.0
```

`1.0` 代表假设按 A3 原尺寸，**不是你已确认的 A4 实际设置**。重算不修改原始照片、现有外参、TCP 或生产配置。

测试：`python -m unittest tests.test_apriltag_diagnostic -v`。覆盖已知几何、打印尺度错误、深度偏置、手眼偏移引起的跨姿态漂移、无效深度、采图运动拒绝、原始板图实际检测以及离线无硬件重算。
