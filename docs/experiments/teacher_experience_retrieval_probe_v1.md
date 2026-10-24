# Teacher experience retrieval probe v1

本实验只读取 `trajectory_memory_full_episode_v1_collection_aligned` 的已保存
`before/rgb.png`、`plan.json` 引用与启动前冻结的 memory。它衡量 cheap visual
ranking 是否找回 teacher 最终引用的具体 experience，不使用 Graph 标签。

实现：[teacher_experience_retrieval_probe_v1.py](../../cloth_agent/experiments/teacher_experience_retrieval_probe_v1.py)。
结果：[summary.md](../../reports/teacher_experience_retrieval_probe_v1/summary.md)。

## 固定协议

- 7 query，139 个 reference，reference 只编码 BEFORE RGB。
- 直接从原 `unfold_visual_retrieval_benchmark.py` 提取 `decode`、
  `DinoEncoder`、`build_index` 三个定义；不导入该旧模块的 GPT/Graph 路径。
- 模型权重哈希和 processor 配置必须与原 benchmark 一致：
  frozen DINOv2-small、CLS 384d、L2 normalize、短边 256、原 center crop 224。
- Primary 为精确余弦排序，完整保留 139 项；K=1/3/5/10/20。
- HNSW 使用原 M=32、efConstruction=200、efSearch=64，与 exact Top-K 比较；
  不因不一致改变参数。
- Diversity secondary 按 historical `trajectory_id`（session）留最高分 step。
  Recall 分母与具体引用 ID 不变，未保留的 cited step 视为 miss。
- 139 个 reference 各编码一次。准确率 query 各编码一次；延迟测量独立，
  每 query warmup 2 次、计时 20 次，共 140 个 batch-size-1 样本。
- 分类阈值、线程设置、K 和处理器在排名之前写入 `protocol.json`。
  分类只描述这一个 episode，不宣称跨 episode 泛化或统计显著性。

## 运行

使用已有本地环境，无需启动模型服务或机器人。

```bash
/tmp/unfold_visual_retrieval_benchmark_env/bin/python \
  -m cloth_agent.experiments.teacher_experience_retrieval_probe_v1 audit

/tmp/unfold_visual_retrieval_benchmark_env/bin/python -u \
  -m cloth_agent.experiments.teacher_experience_retrieval_probe_v1 run
```

`audit` 要求新输出目录；`run` 拒绝覆盖已生成的 embedding 或 summary。
GPU 需要正常设备权限；不会静默改用 CPU。所有 Hugging Face 加载均为
`local_files_only=True`，并启用 offline 环境变量。

独立合成测试：

```bash
/tmp/unfold_visual_retrieval_benchmark_env/bin/python -m pytest -q \
  tests/test_teacher_experience_retrieval_probe_v1.py
```

本次两次启动错误均发生在第一次 encoder forward 之前：缺失可选线程库，
随后是复用函数的 PIL 名称未绑定。已保存失败日志及原协议/源文件快照；
修复仅涉及依赖和名称绑定，没有基于结果修改 descriptor、K 或 crop。
