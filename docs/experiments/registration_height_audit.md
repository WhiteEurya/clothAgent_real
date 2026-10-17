# Camera B 多高度对照

独立入口，不修改旧registration_field的测量方法、正式标定或机器人代码。
默认只读基线为已完成的 `runs/registration_field/20261016T080154_133094Z`。

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.registration_height_audit run
```

先用稳固平整的垫块将整张板抬高约50 mm，仍平放、在桌面内斜置约30°。不要手持，不移动相机或机械臂。
窗口依次提示L-M/C-M/R-M；每次板中心进入绿色框、整板可见、手移开后按Enter。
每处10次有效repeat；其间不要移动。换位置时保持相同高度和朝向。Esc停止并恢复设置。
名义抬高量仅作标签，程序按native depth记录实际Z，不将50 mm强加为测量值。

输出到 `runs/registration_height_audit/<UTC>/`：
- height_session.json：基线、名义高度、采集路径、完成/失败状态。
- capture/<UTC>/：旧采集器生成的完整RGB/IR/native depth、角点、拒绝记录与恢复核验。
- layer_samples.csv：跨层全部角点，包括拒绝记录，附高度层和源session。
- layer_statistics.csv、comparison.json、comparison.md、height_comparison.png：分层统计。
- common_rgb_tiles.csv：固定64px RGB网格的共同覆盖比较，含实际坐标差、Z差和残差差。

不拟合/部署校正。整个placement的中位数不是精确同像素对照；共同网格也要检查格内坐标差。
若相机序列号、流内参/内部外参、depth scale或采集曝光配置不同，禁止自动比较。
当前采集失败仍尝试生成部分报告，但不会标记COMPLETE。

离线重算（不接相机）：
```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.registration_height_audit compare --baseline BASELINE_SESSION --current NEW_CAPTURE_SESSION --output REPORT_DIRECTORY --raise-mm 50
```

`--current`应指向包含session.json和all_samples.csv的capture子目录。
同一高度层连续帧不能当独立数据拆分训练/验证；后续拟合应按整位置/高度留出。本入口未进行任何拟合。
