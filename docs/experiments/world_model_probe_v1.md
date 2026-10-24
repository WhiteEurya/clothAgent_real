# world_model_probe_v1

独立离线诊断 `(frozen DINOv2 CLS_t, executed action_t) -> CLS_t+1`。
不导入机器人或 GPT runtime，不预测 RGB，不规划，不 fine-tune vision。

本次范围由用户确认为 2026-10-19/20 unfolding collection sessions。
只用 accepted COMPLETE、真实执行、专门采集 before/after 的记录。
手柄辅助记录不把点击 Pa 冒充实际手动抓点；不填补缺失 action。
原始演示只读，结果保存在 `reports/world_model_probe_v1/`。

```bash
/home/sja/miniconda3/envs/molmo/bin/python -m cloth_agent.experiments.world_model_probe_v1 audit
/home/sja/miniconda3/envs/molmo/bin/python -m cloth_agent.experiments.world_model_probe_v1 encode
/home/sja/miniconda3/envs/molmo/bin/python -m cloth_agent.experiments.world_model_probe_v1 train
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.world_model_probe_v1 report
```

`molmo` 有 torch/transformers，`cali` 有 matplotlib；全部使用本地依赖与权重。
编码用 CPU，默认 4 threads；不要求设备连接或下载。
审计会冻结 protocol、源文件 hash 和 split，不覆盖已有审计。编码按模型、
processor 和图像像素 hash 缓存，每张唯一图片只计算一次；训练不加载 encoder。
重复编码可复用缓存。重跑 train 会覆盖同路径实验产物，复现实验建议先保存原结果。

按完整 session 分组；既有归档记录的跨 session 连续采集及完全重复 RGB
进一步合并为同组。固定 seed 划分 train/validation/test；后两者不参与动作标准化。
5 个 training seeds；20/40/60/80/100% 按整 group 嵌套抽样，比例向上取整，
实际 transition 数量写入 CSV。early stopping 仅看固定 validation groups。

Identity、State-only 和 Action-conditioned 使用同一批 held-out transitions。
两个 MLP 隐藏层均为 256/256，最后一层零初始化以 Identity 开始。
C 多出的参数仅是 action 输入权重。AdamW/early stopping 参数在训练前冻结。
报告同时给出 transition 均值和 group-macro 均值；稳定性判断使用后者，
置信区间以 group 而非 step 或 seed 为抽样单位。

Action shuffle 只从其他 held-out transitions 选 action，100 次固定随机抽取，
不改变 state/target。它只检测 action 使用，不是 counterfactual dynamics。
`protocol.json` 记录预先固定的判定门槛。无改善或曲线平台不足以独立证明
REPRESENTATION_LIMITED；本轮不会自动扩大模型、改变表示或追加数据。

结果入口：[summary.md](../../reports/world_model_probe_v1/summary.md)。
