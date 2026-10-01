# 手动 Y 坐标补偿

感知配置顶层 `manual_base_y_offset_mm` 控制主 `ClothCenterPerception` pipeline 的基坐标 Y 平移，缺省为 `0.0`（关闭）。当前正式配置 `config/perception.free_exploration.json` 和 candidate 测试配置均为 `0.0`（已关闭）。此前的 `-20.0` 表示 `Y_corrected = Y_calibrated - 20 mm`；2026-09-23 按用户要求在采用 candidate 外参后关闭此经验补偿。

这是经验坐标修正，不是移动 RGB 像素，也不是重新标定。X、Z、原始 RGB、深度和磁盘上的外参文件不变。主 pipeline 在处理帧入口复制外参并修正其基坐标 Y 平移，使点云、坐标图、融合和衣物中心使用一致坐标；重复应用同一值不会累加。

结果中的 `X_base_camera` 是已含修正的有效变换，`manual_base_y_offset_mm` 记录已应用值。读取这些结果时不要再减一次 20 mm。配置影响后续新感知结果，不追溯修改旧结果。直接相机采集、AprilTag 诊断和 TCP 投影工具不应用此主 pipeline 修正。

此经验修正并不证明真实误差在所有位置或姿态下恒定。
