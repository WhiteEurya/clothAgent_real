# 冻结 DINO 双图点对应工具 v1

加载干净参考 RGB A 和当前 RGB B，在任意一侧点击位置，显示另一侧的余弦热力图、
Top-1 和最多三个空间分离候选、独立峰分差，以及目标图→源图的全图反向匹配误差。
这是独立的离线诊断工具，不连接相机/机器人，不调用 GPT，不修改 Graph。

## 启动

在本机仓库根目录执行：

```bash
bash scripts/start_dino_correspondence.sh
```

默认加载已准备的 `reports/dino_correspondence_cases_v1/cases.json`，
结果保存到 `reports/dino_correspondence_v1/`。窗口支持上一对、下一对、下拉选择、
文件选择器加载新 A/B，不需要改代码。依赖 NumPy、Pillow、Torch、Transformers、Tk。
脚本优先使用本机已有的 `visual_retrieval_env`，也可设置 `CLOTH_DINO_PYTHON`。

指定图片和独立结果目录：

```bash
bash scripts/start_dino_correspondence.sh \
  --a /absolute/reference.png --b /absolute/current.png \
  --output reports/my_dino_points
```

模型默认本机已下载的 `facebook/dinov2-small`，通过
`--model-path /directory/with/config.json/and/model.safetensors` 更换位置。
只允许本地加载，不自动下载。`--device auto` 默认选择可用 GPU，否则 CPU；
也可显式 `--device cpu` 或 `--device cuda`。

## 交互与标签

- **点击探索**：点击 A 查询 B，点击 B 查询 A，立即显示并保存结果。A/B 图片位置不变，
  热力图及候选显示在目标图，黄色源点和青色反向位置显示在点击图。切换方向不重新编码。
  JSON/CSV 的 `direction` 标记 `A_to_B` / `B_to_A`；结果 `images` 始终按源图、目标图排序，
  保存的 `source_cycle.png` / `target_*.png` 同样按查询角色命名。
- **盲标评估**：先在 A 选点；在 B 依次点击多边形顶点并封闭区域，可画多个合理区域；
  或选择“不可见 / 不存在”“人工不确定”。输入标注者，再点“保存标签并揭晓”。
- 能确定同一物理位置时，可 **Shift+点击 B** 标记精确对应点 `GT`；可与区域并用，
  也可只标点。“清除 GT 点”可重新标记。无法确定精确身份时不要强行标点。
- 标注前不显示模型结果。此输出目录中已经展示过结果的相同图片对，会保守地标记为
  非盲标；结果仍保存，但不计入盲标命中率。此检查只能覆盖本工具保存的曝光记录，
  不能证明操作者没有从其他地方看过结果。
- 曝光检查统一在实验层执行，UI、batch 和直接调用 `query` 均生效；按 RGB 内容识别，
  重命名 pair ID 不能绕过。预测返回前先保存 `exposures/` 记录，即使后续图片保存失败，
  重启也不会把这对图当成未曝光。该检查范围仍限于当前输出目录。
- 分组 `A` 为同衣服小构型变化，`B` 为目标可见的局部折叠变化，`C` 为跨衣服，
  `D` 为目标不可见，`SELF` 为同图自检；均由人工确认。`UNASSIGNED` 不冒充任何组。
- 探索模式下 B 的点击发起反向查询，不修改评估标签。盲标界面仍按 A→B：B 点击画区域，
  Shift+点击 B 标 GT。任一方向展示预测后，两个方向均视为已曝光，重启或交换图片不能绕过。
  已揭晓标签不原地修改；重新选点可另存一条记录。
- B 的历史执行抓点不用于 GT。纯色布面且人工不能确定对应身份时，选择“不确定”。
- 第一版不使用衣服前景 mask；人工对应区域完全不参与搜索或特征提取。

若需同一图片对多个点均为预先盲标，可先独立标好所有点/区域，在批处理 manifest
的 `points` 中一次提供，再运行 batch；不要先看第一点结果再声称整对图片从未曝光。
batch 会在该对图的第一条预测前冻结并保存全部标签；重复运行同一输出目录会降为非盲标。

## 固定模型与几何契约

`cloth_agent/dino_correspondence.py` 为纯匹配与编码实现；
`dino_correspondence_tool.py` 管理案例、记录和 CLI；`dino_correspondence_ui.py` 为 Tk UI。

- 冻结 DINOv2 ViT-S/14，12 层、384 维；`eval()`、`requires_grad_(False)`、`inference_mode()`。
- 复用本地 Hugging Face 格式的官方预训练权重；读取最终 LayerNorm 后的
  `last_hidden_state[:, 1:]`，即去掉 CLS 的 patch tokens，对应官方
  `forward_features()['x_norm_patchtokens']`。取后仍逐向量做 FP32 L2 归一化。
  不使用已有检索模块的 CLS 特征或默认中心裁剪预处理。
- RGB → PIL bicubic 等比例缩放，长边 448、短边四舍五入 → 居中补至 448×448。
  均值 `(0.485, 0.456, 0.406)`、标准差 `(0.229, 0.224, 0.225)`。
  补边在标准化后为 0（等价于归一化前填充精确通道均值）。
- 保存每张图片实际 resize 尺寸、独立 `scale_xy`、padding 与 offset。
  坐标统一为连续左上边界坐标，整数像素索引须先加 0.5；UI 点击直接返回连续坐标。
  两个变换函数是 `original_to_input` 与 `input_to_original`，B 预测使用 B 的反变换。
- 每图输出 `1024×384`，点击只取 A 所在单 patch，不平均邻域。
- 搜索范围为所有 patch 中心位于原图内容内的位置。纯补边和中心落在图外的边缘 patch
  排除；中心在图内但包含部分 padding 的 patch 保留。极端细长图片可能没有有效中心，
  返回明确的 `NO_VALID_*_PATCH`，不产生假候选。
- 取 patch 中心为预测点；NMS 默认欧氏半径 2 patch（距离 ≤2 被抑制），最多三个候选。
  分差为第一峰减第二个**独立峰**。不足三个时按实际数量保存，缺失分差为 null。
- 反向搜索 A 的所有有效 patch，记录网格欧氏误差和原图反向位置。
- 热力图固定余弦范围 `[-1,1]`，插值只用于显示。不做每图归一化或概率转换。
  `heatmap.npz` 保留原始 32×32 cosine 与 valid mask，无效位置为 `-inf`。
- 特征缓存键包括 RGB 像素、尺寸、权重 SHA256、预处理契约和软件/设备信息；
  相同干净图只编码一次。内存和磁盘均可复用；新图片、权重或预处理变化会换缓存。
  箭头、点、文字和标注只绘制到副本，缓存输入始终是干净 RGB。

1280×720 图片在该设置下每个 patch 约为原图 40×40 像素；高分或反向一致均不能
保证语义正确。当前没有训练或校准过的拒绝规则，也没有自动接受/拒绝机械臂动作。

## 结果与时间

每次点击自动写入 `results.jsonl` 和 `results.csv`，包括 pair ID、点名称、分组、
源点、人工可见性、三候选位置/分数、Top-1/Top-3 区域命中、Top-1 到对应区域的
最短原图像素距离、独立峰分差、反向误差和耗时。落在合理区域中距离为 0；
这是区域误差，不是已知物理点的精确误差。不可见/不确定/未标注不产生定位命中率。
有 `target_xy` 时另存 Top-1 和各候选到该点的欧氏像素误差；仅标点时区域命中率为空，
不会自动设定一个命中阈值。点误差反映相对人工标签的图像定位误差，不是机械臂毫米误差。

每条 `queries/<id>/` 保存：

```text
annotation.json          # 匹配前保存的独立标签
result.json              # 输入出处/hash、变换、完整候选、标签、评估、时间、模型契约
heatmap.npz              # 原始 cosine 与有效掩码
source_cycle.png         # A 源点和反向位置
target_heatmap.png       # 固定尺度 B 热力图与候选
target_candidates.png    # 干净 B 上的候选和评估区域
```

`images/` 保存干净输入副本，`features/` 保存按内容寻址的 patch 特征。
模型首次加载、预处理、前向/传输、两图总加载、单点匹配分别记录。
GPU 编码前后同步；匹配在 CPU 上使用缓存的 FP32 NumPy 特征。
UI `click_to_draw_s` 计到 Tk 完成 idle 绘制，包含匹配及显示处理，不含随后图片/表格保存，
也不是显示器实际扫描延迟。batch 无 GUI，因此该字段为空。
图片及索引由单独的后台线程顺序保存，读取当次查询的固定输入，不读取当前界面的图片对。
保存队列状态单独显示；失败时保留待保存结果，可点击“重试保存”；正常关闭窗口会等待
队列完成，保存失败则保持窗口以便重试。强制结束进程不保证待保存图片完整，但曝光记录保留。
`artifact_save_s` 记录图片、热力图的生成与保存耗时（不含随后结果索引更新）。

按图片对汇总，保留多点相关性：

```bash
/home/sja/.cache/cloth_agent/visual_retrieval_env/bin/python \
  -m cloth_agent.dino_correspondence_tool summarize --output reports/dino_correspondence_v1
```

`summary.json` 包含每对图的命中率、按图片对平均的 Top-1 命中率。区域命中率的分母
只含盲标且有对应区域的可见样本；没有预测的区域样本计为未命中。
`diagnostic_distributions` 只含盲标样本；非盲标和探索样本放在独立的
`exploratory_diagnostic_distributions`。两者分别提供可见命中、可见未命中、仅标点、
无预测、不可见等样本的数量与原始分数/分差/反向误差分布。
`point_error` 汇总盲标点误差的样本数、均值、中位数、P90；
`point_error_when_region_hit` 只统计同时圈区域且 Top-1 位于区域内的点误差，
用于观察“找对部位后还有多大偏差”。无预测时点误差为空，另计 `visible_without_prediction`。
以上像素误差受原图分辨率影响，跨不同分辨率比较时需注意；当前没有毫米换算。
反复点击仍逐条保存；不要把这些记录当成独立样本或统计置信区间。

## 第一批案例与验证

### 完全展开参考与随机抓起落下状态（2026-10-22）

```bash
bash scripts/start_dino_flat_vs_drop.sh
```

`reports/dino_flat_vs_drop_v1/cases.json` 固定 A 为同一件紫色 T 恤的展开图，B 为 5 张
抓起、释放落下后的照片。用“上一对 / 下一对”切换，探索模式直接点 A 或 B 查询另一侧。
结果保存到 `reports/dino_flat_vs_drop_queries_v1/`；真实模型功能检查另存，不混入交互记录。
`contact_sheet.jpg` 提供六张干净图总览，`evidence/` 保存所选重置的执行与释放参数记录。

样本来自 `data/demonstrations` 中 12 次具有落下后 RGB 和执行记录的随机重置，目视挑选
团皱、叠压明显且形态不同的 5 次，不按 DINO 分数筛选。“最随机”没有量化真值；展开图
两袖、领口和下摆均展开，仍有轻微表面褶皱。manifest 保留原路径、文件哈希和选择依据。
没有预填对应点真值或可见性；这些图片用于探索大形变匹配能力，不直接产生准确率。

双向查询改动通过 26 项离线回归测试；5 对图片的 10 次真实 DINOv2 CPU 双向查询均成功，
切换方向不重新编码。功能检查保存在 `reports/dino_flat_vs_drop_validation_v1/`，使用无标签
中心点，验证运行与保存流程，不构成对应点准确率评估。

针对“形态不同、折叠程度相近”的新筛选标准，优先使用
`reports/dino_correspondence_shape_matched_v3/`：从干净 RGB 目视筛选并复核
10 对（校准 5 对、测试 5 对），每对来自不同采集批次，校准和测试的批次不重叠。
匹配轮廓、折叠位置和朝向存在差异，但衣身展开/叠压程度大致接近的状态，
不再直接使用同一次操作前后帧。折叠程度只是目视粗估，不是定量真值。
排除了筛选时已有 DINO 记录中的 33 张输入；筛选期间没有运行 DINO 匹配。
仍须独立标注目标部位、可见性及 GT，分组暂为 `UNASSIGNED`。

```bash
bash scripts/start_dino_correspondence.sh \
  --cases reports/dino_correspondence_shape_matched_v3/calibration_cases.json \
  --output reports/dino_shape_calibration_v3
```

测试时使用同目录 `test_cases.json`，结果输出至 `reports/dino_shape_test_v3`。
两个 `*_contact_sheet.jpg` 可查看实际配对；v2 已完成的标注保持原样，不混入这批新实验。

可直接加载的补充候选池：`reports/dino_correspondence_candidates_v2/`。
从 `data/demonstrations/<session>/traj_*/observation/rgb.png`（A，操作前）及同一
`traj_*/after/rgb.png`（B，操作后）选出 30 对、60 张不重复图片；按采集批次分为
15 对校准候选和 15 对测试候选，同一批次不跨两组。按 RGB 内容排除已有 DINO
结果记录中的 29 张输入，选择只依据采集顺序，未运行 DINO 匹配或依据分数筛选。
这不保证操作者没有在其他地方看过相关结果，也不代表已经满足 A/B/D 各组数量。
每个案例暂为 `UNASSIGNED`；可见性、对应区域、精确点和分组由人工选定目标后判断。
A 也可能处于折叠状态。这批仍是同一件紫色衣服，不提供跨衣服覆盖。

```bash
bash scripts/start_dino_correspondence.sh \
  --cases reports/dino_correspondence_candidates_v2/calibration_cases.json \
  --output reports/dino_calibration_v2
```

测试集将 `calibration_cases.json` 换为 `test_cases.json`，输出换为
`reports/dino_test_v2`。目录中的 `calibration_contact_sheet.jpg` 和
`test_contact_sheet.jpg` 为干净图片总览；manifest 保存每对图的原始路径、采集批次与哈希。
开始点击前切换到“盲标评估”，每对图先做一个点；用“下一对”切换，无需逐个选择文件。

2026-10-21 从现有 Arm6 archive graph 和后续真实观测选出 **16 对跨图 + 1 对同图**，
29 张不同 RGB，位于 `reports/dino_correspondence_cases_v1/`。
`contact_sheet.jpg` 是干净图片总览；`cases.json` 保存原始出处和哈希。
选择按图中多观测节点、历史动作前后和后续 episode 分层进行，没有根据 DINO 分数挑选。
这批主要是同一件紫色衣服，**尚无人工对应区域，也未满足跨衣服 C 组覆盖**。
有遮挡候选，但只有给定源点后经人工判定，才可归入 D 组。界面可自行加载其他衣服。

已有检查：

- 13 项纯离线测试：几何往返、奇数尺寸、横竖图、padding、全局反向搜索、NMS、
  区域距离、固定色阶、缓存与评估隔离。
- 本地真实模型 17 对 smoke 全部输出结果，全部为未标注，命中率保留 null。
  初次 CPU 模型加载约 1.82 s，两个新图编码约 0.25 s，匹配中位数约 0.60 ms。
- 实际桌面验证同图点击回到原 patch；第二次点击不编码；独立标注后揭晓、
  既有曝光标记、不可见标签和三候选绘制通过。一次点击到绘制约 63 ms。
  UI 测试中人为构造的标签只验证软件流程，不是人工语义评估。
- 检查材料：`reports/dino_correspondence_v1_smoke/`、
  `reports/dino_correspondence_v1_ui_validation/ui_validation.json` 和 `ui.png`。

后续修正增加了曝光持久化、批量标签冻结、后台保存快照与失败重试、盲标分布隔离、
精确对应点及条件误差统计的回归测试。上述历史 smoke/UI 记录仍不构成人工语义评估；
修复不会把旧数据中已经声明的盲标标签重新认证为真实盲标。

2026-10-21 修正版验证：23 项离线回归测试通过；真实桌面同图/跨图、连续点击、
曝光降级流程通过，点击至绘制约 63 ms。真实模型的同图点误差检查与批量盲标检查
保存在 `reports/dino_correspondence_v1_fix_validation/`，仅验证软件和坐标，不计入语义评估。

重新准备候选池（输出目录必须不存在）：

```bash
python -m cloth_agent.dino_correspondence_tool prepare \
  --graph reports/arm6_archive_graph_20261019/merged/graph.json \
  --current-root reports/one_shot_goal_continuous_v1_20261021_154005_752162810 \
  --output reports/my_dino_cases --count 16
```

批处理：

```bash
/home/sja/.cache/cloth_agent/visual_retrieval_env/bin/python \
  -m cloth_agent.dino_correspondence_tool batch \
  --cases reports/dino_correspondence_cases_v1/cases.json \
  --output reports/my_dino_batch --device cpu
```

自备 JSON：`{"cases": [...]}`，每项含 `pair_id`、相对 manifest 的 `a`、`b` 路径、
可选 `group` 和 `points`。例：

```json
{
  "cases": [{
    "pair_id": "sleeve_pair_01", "a": "a.png", "b": "b.png", "group": "B",
    "points": [{
      "name": "left_cuff", "xy": [384.5, 163.5],
      "label": {
        "visibility": "visible", "before_prediction": true,
        "annotator": "your_name", "provenance": "human independent annotation",
        "target_xy": [380.5, 160.5],
        "regions": [[[340, 130], [420, 130], [420, 210], [340, 210]]]
      }
    }]
  }]
}
```

示例区域仅说明格式，不是真值。`before_prediction` 仅在确实预先标注时填写 true。
没有 `points` 的 batch 仅在建议源点或图像中心运行一个未标注 smoke，不推断合理对应区域。

测试命令：

```bash
/home/sja/.cache/cloth_agent/visual_retrieval_env/bin/python \
  -m pytest tests/test_dino_correspondence.py tests/test_dino_correspondence_ui.py -q
```
