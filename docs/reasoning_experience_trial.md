# 保存 baseline 并总结完整视觉 reasoning

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.harness.reasoning_experience_trial \
  --baseline results/patch_opt1_20261004_164702/replays/baseline_r00 \
  --output "results/reasoning_experience_$(date +%Y%m%d_%H%M%S)" \
  --ssh-host company-planner --timeout 900
```

复制指定 baseline 的完整目录并校验文件哈希，再用其 `prepared/evidence.json` 图片包和
`reasoning_version.json` 运行原有 `run_rollout`。不调用 observation_code_trial，不重新处理图片，
不把历史 result.json 或后续答案提供给新推理。原 baseline 的预决策观察信息属于相同输入包。

该 baseline 的接口是视觉抓取候选和目标点，不生成轨迹、不执行机器人。固定图片推理中不开放
新的图像工具；这是保存的 baseline reasoning 边界，不等同于完整机器人主程序。

推理后进行一次文字总结，覆盖判断规则、信息需求、可代码化计算、成功条件、未知情况和证据。
依据显式判断和记录，不推断模型内部思考过程；不以比 baseline 快为通过条件。

产物：baseline/ 完整副本、baseline_manifest.json 文件校验、reasoning/result.json 新规划、
reflection_input.json 精确总结输入、experience.json / experience.md 草案、report.json 耗时和 token。
统计排除 baseline 中历史调用费用。经验保存后不会自动激活，也未证明跨场景可复用；下一次使用
需要明确加载并验证。每次输出目录必须新建，防止覆盖历史实验。
