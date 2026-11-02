# 下摆／衣袖抓取规则开关

在已有启动命令末尾添加 `--hem-sleeve-pull-rule`，默认关闭。支持原有 sleeve/hem one-shot、GGO 和 CV feedback 模式；直接调用 Python runner 时同时需要 `--visible-sleeve-hem`。

启用后的规划规则（v2）：

- 目的是展开下摆：抓取可见下摆并拉开下摆。
- 衣袖仅保留用户原句：`如果目前的袖子可以被摩擦力拉开,则抓对向`。

已移除原先的“叠在衣身上层抓本侧／否则抓对向／不可见时抓类似衣袖代理点”分支，以及基础 prompt 的不可见或临近边界时拉对侧的触发规则。原有可见衣袖／下摆限制、动作目的、Pa/Pb 输出与执行检查保留。不增加模型调用。

手动摆放初态、跳过 reset、运行一轮 episode 的示例（执行真机）：

```bash
bash scripts/run_cv_feedback_rollout.sh --execute --episodes 1 --no-reset --hem-sleeve-pull-rule
```

每个 episode 保存 `hem_sleeve_pull_rule.json`、最终 `prompt.txt` 和源代码快照，protocol 记录开关。下一次去掉参数即可恢复默认指令。

## 仅增加深度图

`--rgb-only --depth-image` 表示不启用几何报告、结构识别或 CV feedback，只在原有 RGB 请求后追加 world Z 图。`rgb-only` 在此用于关闭旧的几何分析，不表示禁止这张可选深度图。

每轮首次规划前，在已保存的 perception 位、夹爪已张开的情况下，使用首次 RGB-D，闭合夹爪再拍一帧，然后恢复张开。相机位姿或标定不一致时不合并；RGB局部明显变化的区域不补值。仅补第一帧无效、第二帧有实测有效深度的像素，原有有效测量保持不变。该方法不能保证消除全部空洞。

合并图仅供 GPT 观察，不替换机器人执行使用的实测深度。换视角或拒绝后重规划使用新观测的单帧图，不复用旧融合图。每轮保存两次观测、补值掩码、融合统计、夹爪反馈和耗时；每轮仍只增加一张输入图，不增加规划调用。

```bash
bash scripts/run_one_shot_sleeve_hem.sh --execute --episodes 1000 \
  --max-rounds 13 --max-actions 12 --initialize-center-radius-range-mm 20 40 \
  --auto-next --rgb-only --hem-sleeve-pull-rule --depth-image
```
