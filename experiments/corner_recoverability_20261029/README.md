# Corner Recoverability 离线验证

产物：`reports/corner_recoverability_20261029/review.html`、`report.md`。
无硬件调用，不导入策略模块；原始历史数据只读。

- `prepare.py` 复用失败挖掘索引，按预先固定的终止/平台分层选择5段、14动作、19状态；保存9组连续性桥接。DONE不作可见性标签。
- `annotations.py` 保存Codex逐图观察提出的标签。不是预测器、不是人工真值、不是角点检测算法。未见或不确定端点无坐标。
- `metric.py` 实现三值G、N区间、H=2折扣首次到达回报。先前可能到达但未知时不赋精确回报。短后缀不赋0。
- `test_metric.py` 覆盖用户要求的逻辑例子与未知/截尾边界。
- `report.py` 输出模型提案条件下的结果和单独的人工复核层；提供无需服务器的图片点选、三值编辑、核验人确认和JSON导出页面。导出不会自动改变报告。

运行Python使用 `/home/sja/miniconda3/envs/cali/bin/python`。测试：

```sh
/home/sja/miniconda3/envs/cali/bin/python -m pytest -q experiments/corner_recoverability_20261029/test_metric.py
```

评价对象是项目既有的袖口中点和真实下摆交角，不是四个任意轮廓顶点。
材料身份采用穿着者左右。不能根据旋转后的屏幕位置重新命名；不可靠时标UNKNOWN。
唯一精确匹配的历史人工标签S16存在左右约定和下摆判断冲突，原样保留等待复核。

结论：逻辑性质成立，但本批小样本没有建立交换恢复的实证；没有确认的真实交换成功/循环。
新标签均未人工确认，不能报告“已证实10次成功”；10只是条件于模型提案的可计算成功后缀数量，其中大部分当前已经四端可见。
R是行为策略的实际后缀回报，不是V*。未来图像只用于离线标签计算，不给决策策略输入，不替代未执行动作的反事实。
