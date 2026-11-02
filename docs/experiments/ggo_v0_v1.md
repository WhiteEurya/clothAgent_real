# GGO V0/V1：上半身刚体对齐与结构区域偏差

实现范围遵循设计说明最后的优先级：先验证 V0/V1，暂不实现自动角点身份验证或自动上半身匹配。此模块不调用 GPT、不连接机器人、不修改当前在线 one-shot。原有在线流程中的双模型调用仍存在；本模块验证后才可用它替换该结构识别调用。不能把此版本称为已完成的 V3 单调用集成。

## 输入与标注

使用现有 `Observation` 文件夹（`rgb.png`、`world_xyz_mm.npy`、`observation.json`），加载时校验哈希、尺寸、世界毫米坐标系；两图要求相同 calibration ID。XYZ 已由现有 RGB-D 捕获流程使用内外参反投影，不把原始相机深度当桌面高度。

参考固定为同一件衣服的展开 RGB-D。一次标注 N 领口中心、AL/AR 左右腋下、SL/SR 左右袖口外端、HL/HR 左右下摆角；每张当前图只标 N/AL/AR。左右沿参考衣服自身坐标保持一致。当前下摆点不会进入配准或区域缺失计算。

生成本地标注网页：

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.ggo.cli annotate \
  --reference reports/trajectory_memory_full_episode_v1_collection_aligned/round_006/after \
  --current reports/offline_geometry_ab_20261024/cases/S02/before \
  --out reports/ggo_annotations_new
```

打开生成的 `index.html`，切换参考/当前图及点名，点击标注，下载 `annotations.json`。看不准的当前点设为未知；参考不完整应重新建立参考。标注应在相应材料部位布面内侧，不为了得到有效深度跳到别的材料点。程序不自动吸附或填补深度。JSON 包含图像哈希，防止跨图误用。

运行：

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.ggo.cli run \
  --reference reports/trajectory_memory_full_episode_v1_collection_aligned/round_006/after \
  --current reports/offline_geometry_ab_20261024/cases/S02/before \
  --annotations /absolute/path/annotations.json \
  --out reports/ggo_result_new
```

输出目录必须不存在，避免覆盖证据。可用 `--settings settings.json` 覆盖 `Settings` 同名字段。

## 算法与可解释输出

1. 复用紫色衣服/深色桌面的颜色分割，填充外轮廓内部的 RGB 印花空洞（不填补深度），记录分割规模、裁切、有效深度比例。不是通用服装分割器。
2. 从参考图的非衣物背景拟合桌面 `z=ax+by+c`：固定种子 RANSAC、3 mm 内点带，至少 60% 一致性；建立正交桌面基。两图点云进入同一个桌面坐标系，得到相对桌面的高度网格。
3. 仅对 N/AL/AR 的有效实测点做 SE(2) RANSAC + 加权 Kabsch/Huber 优化，无缩放、无反射。至少 3 个内点，检查残差、锚点空间跨度、非共线覆盖、多解歧义、上半身轮廓一致性。默认 12 mm 残差界、80 mm 跨度、12 mm 次轴标准差；这些是待标定参数。
4. 参考轮廓和路径保存为物理坐标；左右下衣身按固定腋下线及身体中轴分区，袖子为上半身外侧参考区域（腋下间距 0.38 倍之外）。它是透明的几何分区，不是当前图自动语义分割。参考的解剖点标错会直接影响分区。
5. 以 2 mm 栅格比较参考和当前投影。参考分区腐蚀 6 mm、当前 Mask 膨胀 6 mm；同时报告缺失比例、缺失 mm²、最大连通缺失块 mm²。不存在有效深度的位置不是背景；区域或路径可观测覆盖低于 95% 时返回 UNKNOWN。
6. 参考侧边到当前衣物区域的截断距离按路径长度加权（上限 80 mm），衣物内部距离为 0；不需要当前下摆角坐标。参考曲线三维采样存在跨层跳跃时，三维长度为 null。
7. 初始异常规则要求比例≥12%、缺失面积≥800 mm²、最大块≥600 mm²、轮廓偏差≥10 mm，同时成立。主要区域按绝对最大连续缺失块选择，不仅比较百分比。这些阈值未由真实多角度平铺样本校准，不能声称准确率。
8. 用 ±3 mm 平移和绕上半身中心 ±2° 旋转重算。局部判断变化则该区域 UNCERTAIN，主要区域翻转则不输出主要区域。V2 尚未实现，衣角身份始终 UNVERIFIED，不能把对齐参考的隐藏角坐标当抓点。

## 产物

- `report.json`：输入哈希、手工标注来源、平面、配准、区域指标、不确定性、参数、计算耗时。
- `geometry.npz`：参考/当前 Mask、毫米网格、实测覆盖、相对桌面高度、missing/excess。
- `reference_anchors.png` / `current_anchors.png`：可审查的标注。
- `reference_regions.png`：固定参考中的四个结构分区，便于检查人工锚点是否合理。
- `metric_overlay.png`：黄虚线参考、绿线当前、红色缺失、蓝色超出、灰色未观测。
- `rgb_overlay.png`：桌面平面投影的辅助叠加图；凸起布料有视差，定量结果看毫米网格。
- `index.html`：汇总上述内容。

## 验证与边界

```bash
/home/sja/miniconda3/envs/cali/bin/python -m pytest tests/test_ggo.py -q
```

测试覆盖旋转/平移/异常锚点、拒绝缩放、锚点覆盖不足、平铺旋转不误报、左右衣身/袖子局部缺失、无效深度不能算缺失、桌面高度、端到端右下内收但右下角仍未知、缺失锚点抑制输出。

`reports/ggo_v0_v1_20261024` 是本地真实图预览。示例标注由助手看图提供，不是用户人工确认的真值；修订历史保留。自配准和合成测试验证实现的一致性，不证明真实衣服识别准确率。正常误差阈值、人工锚点质量、不同旋转姿态下真实误报率仍需真实标注数据验证。当前指定的“展开参考”本身存在褶皱，可能造成偏差；不要为了获得期望结果调参。

## V1 新增：可验证的 Geometry Assessment

每次 `run` 额外输出 `geometry_assessment.json`、`geometry_assessment.txt`（可直接作为后续 GPT 输入）、`body_cross_sections.csv`、`body_cross_sections.png`。仍是离线工具，没有新增模型调用或接入机器人。

- **缺失面积**：左右分别报告容差后的比例、面积 mm²、27 次姿态扰动下的范围。`projected_deficit` 的 LOW/MODERATE/HIGH 仅按面积比例范围判定，不与轮廓形变标签混为一谈；范围跨等级时 UNCERTAIN，观测不足时 UNKNOWN。默认 LOW <5%，HIGH ≥12%，为待标定阈值。
- **截面曲线**：参考衣服腋下线中心为原点，u 沿衣服左到右，v 沿腋下到下摆；默认每隔 10 mm。左右内收分别为 `current_left-reference_left` 和 `reference_right-current_right`，正值代表内收，负值代表外扩。每个截面在 ±2 mm 邻域复查，并要求边界内外至少 6 mm 连续实测布料/背景。多段交点、无效深度、出画面、缺少边界支持时为 null，并保存原因。没有最近点替代、填洞或跨缺测连线；测得的是可见外轮廓，不证明其是材料侧边。连续内收长度仅为连续可测采样端点间距离，未知截面中断计数。
- **配准不确定性**：在 dx、dy、角度的负边界/零/正边界取 27 个组合。平移半径为 `max(配置下限3mm, 锚点RMSE)`；旋转半径为 `max(配置下限2°, atan(RMSE/锚点跨度))`。这是有记录的敏感性测试范围，不是统计置信区间，不保证覆盖所有真实标注误差。所有变换及每区指标写入完整报告。当前曲线对应名义配准，曲线自身不声称具有姿态置信区间。
- **左右不对称**：单独检查每个扰动的 `D_R-D_L`，默认超过 5 个百分点才认为明显。全部 27 次都超过同一方向阈值才 STABLE；翻转或跨阈值则 UNCERTAIN；任一扰动观测不足则 UNKNOWN。仅稳定不对称且对应区域持续高缺失时给出“该几何差异不能由本次采样的全局刚体变换解释”。这不证明某个具体衣角身份；语义角仍为 UNVERIFIED。

本次新增实测报告在 `reports/ggo_v1_assessment_20261024`。沿用之前手工标注和固定参数，未使用新模型调用或人工填补截面。


## 自动实机单步实验

启动：`bash scripts/run_ggo_auto_single_steps.sh --execute`。先退出旧实验进程；此命令立即开始随机初始化。默认 3 次，每次 reset 半径 20 mm、不下压、最多执行一个展开动作。正常完成自动进入下一次；异常、ABSTAIN、人工停止结束批次，不自动重试。可选 `--wait-before-first` 仅在首次启动前等待控制页确认。

自动识别当前 N/AL/AR（三点，不可见为 null），复用固定参考七点，计算 GGO，然后一次动作模型选择 Pa/Pb。不再启动人工标注服务器。每次观测有一次锚点模型调用和一次动作模型调用；随机初始化自身的调用另计。自动锚点不是人工真值，配准通过不代表语义识别正确；未知几何保持 UNKNOWN，由动作模型结合 RGB 决策。

每轮 `ggo_annotation/` 保存识别请求、响应、指标和带 RGB 哈希的坐标；`ggo/` 保存配准、缺失面积、截面曲线和可视化；`request_manifest.json`、`model_metrics.json` 分开记录锚点、几何和动作耗时。原始 RGB-D、动作图、Host 检查及执行前后证据仍由原有流程保存。离线检查这些日志，无需在线人工提交。


## 连续 rollout（当前默认）

`bash scripts/run_ggo_rollout.sh --execute`：3 次独立展开，每次只在开始时 reset（20 mm），连续规划执行最多 12 个动作，每轮自动识别当前锚点并重新计算几何，同时提供上一动作前图和实际动作。第 12 个动作后额外观察并判断 DONE，不执行第 13 个动作。沿用简化的 sleeve/hem one-shot prompt，不恢复上一步效果分析，因此不启用连续无改善计数。正常 DONE 或达到上限后才进入下一次 reset；异常/ABSTAIN/人工停止终止批次。保留原有缺失深度换视角和目标拒绝的一次重规划；新观测重新计算几何。旧 `run_ggo_auto_single_steps.sh` 已转发到此连续入口，避免沿用旧命令后每步重置。手工单步研究入口仍单独保留。

当前 rollout 默认 reset 半径上限改为每个 episode 从 [20,40] mm 均匀采样，实际半径记录在批次目录 `episode_NNNN.reset_radius.json`。这是采样圆盘的半径上限，实际抓点仍可落在中心附近，不是只采 20–40 mm 的圆环。可用 `--initialize-center-radius-range-mm 20 40` 配置。

手动初态：`bash scripts/run_ggo_rollout.sh --execute --no-reset --episodes 1`。默认仍开启随机reset；`--no-reset`跳过每个episode的随机抓放，先移动到观测位获取当前RGB-D，再正常连续展开。不会从旧run加载历史。建议单次实验以保留手动摆放的状态；不传此参数则按原默认运行。
