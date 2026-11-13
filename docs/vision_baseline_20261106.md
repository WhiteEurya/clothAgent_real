# 2026-11-06 黑衣双相机视觉基线

用户已查看融合图及 mask，并确认采用此版本继续实验。

## 生效配置与执行链

`config/topdown.unfold_zed.json`：`appearance_render=surface_rgb`、
`appearance_seam=local_gain`、`segmentation_profile=black`。
`unfold_topdown.primary` 读取该配置，双相机进入 `unfold_dual_topdown.reconstruct`。
原生深度表面重采样、原始 RGB 取色、局部拼接及亮度增益生成规划主图。
黑衣 CV 分割读取同一 `rendered_rgb.png`，保留印花并使用 3×3 轮廓平滑。
没有改变 Simple/Tangent 内部抓点、方向、最远交点、运动或安全检查。

拼接选择可能改变像素所属相机，但实测 XYZ 仍逐点取自该相机的原始测量。
显示插值和颜色补全不是抓取深度。来源、颜色补全、颜色校正分别保留 sidecar。

## 已验证证据

- `reports/local_seam_validation_20261106/validation.json`：8 个保存状态、代码哈希、耗时与限制。
- `entrypoint_check.json`：默认规划入口确实加载新配置，输出与离线验证逐像素一致。
- `round_007_mask_detail.png`、`all_masks.png`：RGB、mask 及叠加图。
- `shadow_trace.json`、`shadow_mask_check.json`：9 个原图阴影抽样均未进入衣物 mask。
- `seam_metric.json`：round007 固定旧接缝区域149对相邻像素，平均RGB跳变10.76降至2.63。
- 84项相关离线测试通过，输出实测XYZ与原始所属点最大差为0。

单次重建约5.7–7.3秒。尚无人工像素真值，不报告mask IoU或宣称通用分割。
当前基线针对黑色印花衣物；相机位置、衣物配色改变时需重新检查。
保留原图中的真实阴影，不通过抹掉阴影或补造深度来美化结果。

## 回退与保存

只关闭本次接缝修正：删除 `appearance_seam` 或设为 null。
恢复旧点投影：将 `appearance_render` 设为 `point_rgb`。
源码、配置、文档快照及SHA256清单位于
`reports/vision_baseline_20261106/`。快照是当前工作区文件，不是独立可运行的完整仓库；
不覆盖或清理工作区中其他历史修改。恢复单个文件前先保留其后续改动。

## 下一阶段：Online Pilot 01

后续更新：已完成相机专用目标采集，当前目标是
`data/reference/black_mario_goal_20261106_153449_487709/goal.png`。
路径及哈希已绑定Online Pilot配置，状态为`BLOCKED_BATCH_ENTRYPOINT`。
下文目标缺失说明为基线冻结时的历史状态；源码快照保留原样，不覆盖。

视觉准备通过本轮离线检查。固定6次、Simple 70% / Tangent 30%、规划前随机化、
每步一次正常规划、无同一步fallback的协议保持不变；不训练或部署Gate。

尚未启动真机：`config/skill_gate_online_pilot01.json` 的黑衣目标路径和哈希仍为null。
普通unfold脚本默认目标仍是旧参考，不能作为黑衣探索目标。需要先保存正确的黑衣展开目标。
`skill_gate_online.run_episode` 已有单次库适配器，但6次批量启动入口尚未接好；
须复用既有自动初始化及执行链接入，不能用人工复位代替。
