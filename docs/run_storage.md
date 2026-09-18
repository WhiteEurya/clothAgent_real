# 实验结果存储

配置：`config/run_storage.json`。新实验统一保存在 SSD，日期使用 UTC：

```text
/mnt/newssd/sja/clothAgent_real/runs/
  YYYY-MM-DD/
    fold_<timestamp>/
      run_metadata.json       # run ID、项目路径、创建时间、配置
      workspace/              # 配置快照、生成代码、经验记录
      results/                # 各阶段、各 iteration 的图片、视频和日志
```

一个 run 是独立的保存和删除单位。保留 run 内部的结果层级，以兼容现有日志、可视化和恢复流程。
`--run-id` 自动跨日期查找；也可用 `--run-dir` 指定完整路径。
SSD 未挂载时启动失败，不会回退到系统盘。启动脚本在相机操作前检查存储。

```bash
python scripts/manage_runs.py check
python scripts/manage_runs.py list --size
python scripts/manage_runs.py delete fold_<timestamp>  # 仅预览
python scripts/manage_runs.py delete fold_<timestamp> --confirm fold_<timestamp>
```

删除前停止对应实验和查看器。确认删除会移除该 run 的全部配置、经验和结果。
旧项目 `runs/` 数据已按用户要求删除，没有迁移。相机持久配置在 `config/` 中。
独立工具若显式指定 `--output-dir` 或 `--run-dir`，仍使用指定路径。

## 独立工具

相机调参、单独拍照、手动相机标定的默认输出：
`/mnt/newssd/sja/clothAgent_real/tools/<类别>/YYYY-MM-DD/`。
图像诊断和关键点标注使用与实验相同的日期/run ID 目录。
历史报告归档脚本中的固定样本路径只用于旧报告，不是新实验入口。
