# Host对齐离线验证

独立代码：`cloth_agent/experiments/host_depth_alignment_audit.py`。
不连接相机、机器人，不导出可被线上Observation直接加载的world_xyz文件，不修改任何正式标定。

运行（输出目录必须不存在，防止覆盖）：

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.host_depth_alignment_audit --output reports/host_depth_alignment_new
```

默认使用已冻结的 `rgb_T_only` 候选，读取两层L/C/R的第一条有效记录，共六组。
同一份最后一帧emitter-on native depth分别生成factory/candidate对齐结果，附候选文件SHA256。

几何语义：
- 像素footprint投影到RGB，冲突取最小native Z，沿用SDK通用实现的映射语义；不补洞。
- 保存原生depth来源索引，显式换算RGB optical Z，沿RGB射线得到RGB-frame XYZ。
- 保留旧流程把native Z视为RGB Z的对照查询值；不能把两种Z混用。
- RGB坐标查询采用相同整数像素和5×5有效性检查；仅记录Z跨度，不调用grasp算法。
- 原生像素取源误差用每个tag的观测RGB/IR单应关系作参考，包含量化/平面假设误差，不等于XYZ绝对真值。

SDK对齐一致性复核：已有同一frameset保存的SDK aligned depth与Host factory结果，全图99.775%完全相同；共同有效像素99.834%相同。优化版SDK在footprint量化等边界可能不同，因此本模式不是逐位相同的线上替代品。

本次成功输出：`reports/host_depth_alignment_audit/complete_run/`。
初次批处理发生native detector内存错误，保留在根目录的单case结果不是完整运行；代码已改为整批复用同一个Detector，后续两次六case批处理正常退出。

没有衣物/真实深度边界的独立真值，不能据此直接部署到抓取。下一步是静止衣物的相机-only对照；部署前还需明确RGB/world坐标系语义并保留现有安全检查。
