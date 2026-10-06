# 原始 reasoning 中的参数化代码提取实验

目标是提取可复用计算，不修改 planner 的阶段、顺序、视觉判断或最终任务。使用现有 `patch_evolution` 入口的 `code-extraction` 模式，复用已保存的 baseline 图片和完整规划指令。支持 1–2 个迭代。

每轮由 Claude 从公开的历史判断与调用记录中提出至多两个参数化 Python 函数，生成参数/返回值 JSON Schema、使用说明、信息不足处理和测试。函数以 `def run(arguments)` 接受当前参数，由有步数和数据规模限制的 AST 解释器执行；支持有界数组循环，不允许文件、网络或机器人访问。旧图像 recipe 的执行权限不变。

通过本地测试的函数进入本次实验的临时库。原 planner 可以直接返回普通规划结果，也可以输出 `REQUEST_CODE` 请求计算，收到实际结果后继续同一规划任务。没有强制函数使用次数、镜像方法或固定测量/验证阶段。当前后端每次请求后仍会重新发起模型调用并提供本轮显式上下文，不是服务端会话恢复；会产生额外调用和图片传输成本。

第二轮沿用第一轮函数库及实际调用反馈，可修改有问题的函数或提取其他操作。提炼器能看到历史结果；新规划只接收当前图片、原指令、函数接口以及自己本轮的请求/结果，不接收历史答案和测试坐标。适用说明仍是模型生成文字，其语义泛化需要审查。

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.harness.patch_evolution \
  --evidence results/reasoning_k3_prepared_20261003/evidence/evidence.json \
  --reuse-baseline-run results/patch_opt1_20261004_164702 \
  --reasoning-record results/reasoning_patch_more_20261005_122702 \
  --learning-mode code-extraction \
  --output "results/reasoning_code_$(date +%Y%m%d_%H%M%S)" \
  --patches 2 --repeats 1 \
  --backend remote --ssh-host company-planner \
  --max-calls 10 --max-seconds 3600 \
  --call-timeout 600 --replay-timeout 1200 \
  --max-code-request-rounds 2
```

输出：

- `iter_XX/extraction.json`：提取内容、参数、证据和留给模型的判断。
- `iter_XX/functions/ID/function.py`：实际生成的 Python 函数。
- 同目录 `specification.json` 和 `tests.json`：契约、例子与测试结果。
- `iter_XX/reasoning/host_execution.json`：planner 实际请求的参数与函数输出；未调用时不存在。
- `iter_XX/reasoning/result.json`：完整视觉规划结果及调用耗时。
- `library.json`：本次临时函数库，不自动写入生产 skills。
- `report.json`：每轮函数测试、实际使用、规划结果、token 和时间统计。

`COMPLETED` 仅指实验轮数执行完毕；`TESTED` 不等同于 READY、正确或加速。函数自带测试不是独立正确性证明。记录回放仅检查预期值确实存在于引用记录，输入绑定解释仍需复核。未使用的函数应明确标成 NOT_USED，不强迫 planner 调用。没有耗时淘汰，也没有 token budget，只有计数；失败调用缺失计量不记作零。

本实验覆盖保存图片上的完整 grasp/target reasoning，图像预处理复用 baseline，不重新运行真实机器人或完整采图流程。

## 公开过程输入

提炼器现在优先读取每条记录的 `operation_trace`，包含已保存的信息需求、观察请求、Host 图像操作、父子图片与坐标变换、工具请求/返回、公开观察说明、候选决策记录以及最终选点。不会根据最后的答案反推未记录的候选比较或内部思考。

新规划的判断 schema 支持可选 `decision_log`：信息需求、实际操作、证据引用、简短结果、候选的 considered/kept/rejected/selected/unknown 更新及显式依赖。原始远程规划调用也会请求简短的 `[[process]]` 公开记录，并在 debug 目录输出 `public_process.json`。这些改变用于可观测性，会增加少量输出成本，不重排规划步骤、不强制额外模型阶段。

每次 reasoning 输出 `trace/trajectory.json` 和可阅读的 `trajectory.md`。提炼时传入全部公开操作链，并要求优先寻找有证据支持的多操作片段；需要视觉模型完成的判断仍保留给模型，不声称纯函数替代了视觉识别。提炼输入字符上限为 400000，仍显式报错而非静默裁剪；token 继续实际计数。

旧记录可以离线恢复，不调用模型：

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.public_process \
  --replay results/patch_opt1_20261004_164702/replays/baseline_r00 \
  --output results/public_process_baseline_review
```

覆盖度始终明确标记：图片交付不证明已经查看，请求目的不等于实际获得信息；候选比较历史缺失则保留缺失。只有工具 ID、显式图片引用/依赖和接收顺序建立连线，不把条目排列或单次调用时间解释为某个判断的思考时间。只导出公开操作/结果，不将 thinking 或 redacted_thinking 块送入提炼器。

如果 Host 的格式处理需要修正，可加 `--reuse-code-extraction-run <原输出目录>` 重测同一批已生成函数，不再支付提取模型调用。报告会标注来源和哈希，并说明缓存的第二轮生成没有看到补测的第一轮规划反馈；不可把它描述为基于该反馈重新学习。引用字段允许唯一的 `record_id: 说明` 写法，保留原文并记录解析。浮点测试使用 `rel_tol=1e-7, abs_tol=1e-6`，整数精确比较，避免低于像素分辨率的舍入差异误伤。

## 选点过程计时（可选诊断）

在现有远程实验命令后增加 `--semantic-timing`。输出保存在实验目录旁的
`<output>_timing/`，两个目录都必须是新目录。原始实机 planner 继续使用
`--semantic-timing --timing-output <新目录>`；这些开关不会替代停止执行机器人所需的选项。
本次实现未自动启动远程实验。

公开阶段可区分：任务读取、衣服方向、图像结构识别、信息缺口与观察操作选择、跨图对应、
抓取候选评估、目标点构造、选点核验、运动方案、工具恢复、结果提交。阶段可重复或省略，
并不规定执行顺序。看图与选点无法分离时使用 mixed_visual_selection。

- `summary.json/md`：Host 函数层级计时，含模型调用、证据准备/验证、生成函数执行、选点输出校验。
  父阶段含子阶段，不能直接相加；自身时间扣除子阶段覆盖区间。
- `claude_calls.json`：逐轮响应和工具事件原始时间分析。
- `semantic_timing.json/md/csv`：任务阶段 × 响应等待、thinking 块窗口、文字、工具参数、工具等待、其他。
- `semantic_timeline.csv`：每次调用内部按时间顺序的不重叠区间；可直接用 Excel 打开。
  start_s/end_s 为该调用的远端 CLI 时钟，不能跨调用相减；trace 字段定位原始记录。
- 原有 `trace/trajectory.json/md`：操作目的、输入输出、证据引用和选点依据，与上述计时结合查看。

程序使用远端流式事件时间戳，不接受模型自报秒数。公开 start/end 标记只说明模型声明的任务窗口，
不是内部计算剖析。首响应等待无法进一步拆成图像编码、服务排队和网络；thinking 块时间也不是
纯计算时间。没有实时边界、缺少闭合或同一时刻批量补发时，不推断阶段，计入 UNKNOWN。
即使阶段记录齐全，也不能证明视觉理解与选点在模型内部严格串行。

诊断不会为了计时增加模型调用或强制阶段，但状态标记会增加少量输出并可能影响行为和延迟。
历史 baseline 不补造新标记；复用 baseline 时不会把其历史时间计为本次实际耗时。

### 只计时，不提炼

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.harness.reasoning_timing \
  --reuse-baseline-run results/patch_opt1_20261004_164702 \
  --output "results/reasoning_timing_only_$(date +%Y%m%d_%H%M%S)" \
  --ssh-host company-planner --timeout 900
```

该入口复制并验证已准备图片与原始单阶段 harness，重新调用一次选点规划，不复用历史答案，
不提炼、不加载生成函数、不重试规划、不运行机器人。一次 Host 模型调用仍可能包含 CLI 的
结构化输出提交轮次，这些轮次照常记录。计时默认开启，位于输出目录内的 `timing/`；
选点结果和公开证据位于 `reasoning/result.json` 与 `reasoning/trace/`。
本模式仅测准备好图片后的规划，历史裁剪/旋转不重跑、不计入本次时间。
