# 随机初态 → 冻结 DINO 选择参考动作

本实验只完成随机初始化和参考动作检索。选出的动作边是参考记录，还没有迁移抓点、规划或执行。

## 扩大到原始示范目录

扩大搜索的入口会扫描 `data/demonstrations/**/rgb.png`，包含动作前图、动作后图、pending、随机重置和失败记录图片。图像排名覆盖全部扫描图片，完整排名保留原路径和角色；动作排名仅使用经已有档案导入器验证、操作者确认 COMPLETE 且具有明确 Pa/Pb 的示范前图。动作后图不会借用前图的点对。

```bash
/home/sja/.cache/cloth_agent/visual_retrieval_env/bin/python -m cloth_agent.experiments.dino_corpus_search \
  --data-root data/demonstrations \
  --previous-cache reports/dino_reference_baseline_v1/cache \
  --previous-query reports/dino_reference_baseline_v1/round_002/retrieval \
  --output reports/dino_reference_expanded_v2
```

本次输出目录已存在，不要覆盖重跑。更新示范目录后使用新的输出目录。旧参考编码按内容哈希复用，当前图复用第二轮已有特征，不重新编码；模型及预处理保持不变。

- `all_images_top9.jpg` / `all_images_ranking.json`：所有 RGB 的排名与来源。
- `action_images_top9.jpg` / `action_images_ranking.json`：有明确原始点对的参考动作排名。
- `archive_snapshot.json`：完整示范和未成为动作候选的排除原因。
- `summary.json`：搜索覆盖范围、新旧第一名与编码计数。
- `retrieval/`：扩大动作库后的标准检索证据，可继续传给 grounding 程序。

**后续新轮次的动作检索使用 `reports/dino_reference_expanded_v2/action_cache`。** 下面旧基线示例中的 `--cache reports/dino_reference_baseline_v1/cache` 应替换为该路径；保留旧缓存用于复现前两轮。全图检索包含没有可用动作的照片，不能把全图第一名直接交给机器人执行。

## 固定的实验定义

- 固定当前紫色 T 恤、Camera A、桌面及拍照位，使用已有 Arm6 随机抓起放下流程。
- 每轮只随机初始化一次：`--max-retries 0`。抓取范围上限 80 mm；抬升 250–300 mm；随机偏航 ±45°；在 Home XY 周围 40 mm 内放下；抓取高度偏移 -15 mm。实际抽样参数及种子以 `reset/result.json`、`grasp.json`、`release.json` 为准。
- 放布后等待、回拍照位，保存 RGB、深度对应世界坐标和机器人状态。实际 RGB 是初态记录，种子不能重建真实布料状态。
- 参考库：`reports/arm6_graph_incremental_20261020_v4/final/graph.json`。每条边使用其 `seed_step_id` 对应的实际前图，并验证该观测属于源节点。
- DINOv2 ViT-S/14 冻结权重，FP32，CPU；长边缩放到 448，居中填充，保持宽高比，无中心裁剪。一次前向同时取最终 LayerNorm 后的 CLS 和 32×32 patch tokens，分别 L2 归一化。此预处理沿用点对应工具，不能与旧中心裁剪的检索缓存混用。
- 精确全局 CLS 余弦相似度；一个节点有多个动作前图时取最大值，并保留产生该最大值的原动作边。并列时按 node_id、edge_id 排序。Top-3 为不同源节点。
- 当前库包含折叠和展开动作，没有目标筛选。分数表示视觉相似度，不是动作成功概率，也不是不可见检测或可靠性判定。

## 离线准备一次

在仓库根目录运行。缓存目录必须不存在；参考图、图结构、模型或预处理改变后，使用新目录重新准备。

```bash
/home/sja/.cache/cloth_agent/visual_retrieval_env/bin/python -m cloth_agent.experiments.dino_reference_baseline prepare \
  --graph reports/arm6_graph_incremental_20261020_v4/final/graph.json \
  --output reports/dino_reference_baseline_v1/cache
```

缓存保存原始 RGB 副本、CLS、patch、有效 patch 掩码、原图到模型输入的缩放/偏移/填充参数，以及模型、输入、特征文件哈希。当前这份缓存已经准备好，可直接复用。

## 再运行一轮

下面命令第一条会真实移动 Arm6。保持现场正常实验条件，使用新的轮次目录。第一轮已使用 `round_001`。

```bash
/home/sja/miniconda3/envs/cali/bin/python scripts/test_arm6_random_fold.py \
  --execute --max-retries 0 \
  --output reports/dino_reference_baseline_v1/round_002/reset
```

仅在初始化成功后运行检索。因关闭重试，当前图固定为 `reset/after/rgb.png`；程序会验证它与成功初始化记录的最终观测一致。

```bash
/home/sja/.cache/cloth_agent/visual_retrieval_env/bin/python -m cloth_agent.experiments.dino_reference_baseline query \
  --cache reports/dino_reference_baseline_v1/cache \
  --current reports/dino_reference_baseline_v1/round_002/reset/after/rgb.png \
  --reset reports/dino_reference_baseline_v1/round_002/reset \
  --output reports/dino_reference_baseline_v1/round_002/retrieval
```

每轮会生成：

- `reset/`：初始拍照、随机参数和种子、运动记录、释放后拍照及状态。
- `retrieval/I_current.png`：该轮实际初态的 RGB 副本。
- `retrieval/top3.jpg`：当前图与三个参考前图的对比。
- `retrieval/evidence.json`：Top-3 分数、节点、关联动作、特征缓存、坐标变换、初始化出处，以及在线编码计数。
- `retrieval/current_features.npz`：当前图一次编码得到的 CLS 和 patch 特征。
- `retrieval/selected_action_edge.json`：选中的原始动作边，保留原图的重放限制。
- `retrieval/ranking.json`：所有源节点的完整排序。

先查看实际 RGB，确认布料已经释放、目标没有被手臂遮挡。相机画面边缘可能保留夹爪或固定设备；如果遮挡了目标，应将该轮标记无效。检索不自动判断画面质量。

## 验证

```bash
/home/sja/miniconda3/envs/cali/bin/python -m pytest tests/test_dino_reference_baseline.py tests/test_random_fold_reset.py -q
```

覆盖节点去重及动作绑定、并列排序、缓存复用、仅编码当前图一次、损坏缓存拒绝，以及原有随机初始化的采样和执行顺序测试。

## 把参考 Pa → Pb 定位到当前图，查看 A → B

检索结束后，可以用 GPT 对 DINO 已选定的 Top-1 做视觉 grounding。GPT 只得到这一张参考图、原始点对标记和当前图，没有其他候选，也不能换图。若选中动作缺少原始 Pa/Pb，程序报错，不会偷偷改选其他动作。

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.dino_action_grounding \
  --retrieval reports/dino_reference_baseline_v1/round_002/retrieval \
  --output reports/dino_reference_baseline_v1/round_002/grounding
```

这一步需要现有 GPT 接口联网。程序不连接机械臂，输出：

- `current_action.png`：当前图上 A（红，抓取）→ B（蓝，放置）的黄色箭头。
- `action_comparison.png`：参考 Pa→Pb 与当前 A→B 并排对比。
- `review.html`：无需服务器的独立页面，可显示/隐藏预测、点选 GT A 和 GT B、导出像素误差 JSON。
- `grounding.json`：原图像素坐标、定位依据和不确定性。
- `result.json`、`model_metrics.json`、`model/requests.jsonl`：输入哈希、固定参考身份、模型调用和耗时。

GPT 可以报告 `UNCERTAIN` 或 `NOT_APPLICABLE`；不强求每张图都给出点对。B 是释放目标，可以在桌面，不能把它一律当作衣服上的匹配特征。箭头仅表示二维动作意图，不表示已完成深度、可达性或执行验证。

误差分别计算 `||A_pred − A_GT||₂` 与 `||B_pred − B_GT||₂`（原图像素）。没有人工 GT 时不报告准确率；模型自报 confidence 不是准确率。页面默认展示预测，因此其标注用于辅助校对，不能算作盲审。

第一轮的 DINO 原始记录只有前向耗时，grounding 程序另外测量缓存上的精确排序回放以及 GPT 请求 wall time。这些分项不等于一次完整在线总耗时。失败的联网请求保留日志，单独计入失败开销。

从第二轮开始，另行记录 DINO 模型加载、解码/预处理/前向、实际排序和查询存图总耗时。`grounding/model_timing/invocation_001.json` 按 CLI 事件记录请求前的宿主准备、进入请求前的启动时间、请求窗口和退出时间；它不记录推理正文。请求窗口仍含网络、服务端排队、推理和生成，不能当作服务端纯推理时间。

当 GPT 返回空点对时，页面明确显示“未产生可定位的动作”，该轮记为未出结果；不换参考图、不强制猜点。此时不能计算定位误差，也不能把缺失误差算成零。
