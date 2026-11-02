# Intention–Outcome Hypothesis Mining

只建立逐案例 Hypothesis Bank。默认读取现有 532 条历史索引，每条独立使用 `codex exec -p api --ephemeral`；不读取其他案例的分析结果，不合并假设，不生成 Skill/策略，不连接机器人。

## 启动

```bash
cd /mnt/newssd/sja/clothAgent_real
PYTHONPATH=. /home/sja/miniconda3/envs/cali/bin/python -u \
  experiments/intention_outcome_hypothesis_mining/run.py \
  --workers 2 --debug
```

默认使用 `gpt-6-astra / high`、`api` profile；输出至 `reports/intention_outcome_hypothesis_bank_v1/`。运行前确保本机 `codex -p api` 已可用；在普通终端启动即可，不需要机器人权限。全部有效数据均会处理，不会在首批完成后停止。

可选：先检查数据和报告结构，**零模型调用**：

```bash
PYTHONPATH=. /home/sja/miniconda3/envs/cali/bin/python -u \
  experiments/intention_outcome_hypothesis_mining/run.py --dry-run
```

检查后正式运行，或 Ctrl+C 后继续：

```bash
PYTHONPATH=. /home/sja/miniconda3/envs/cali/bin/python -u \
  experiments/intention_outcome_hypothesis_mining/run.py \
  --resume --workers 2 --debug
```

同一输出目录有文件锁，禁止两个 runner 同时调用。已有完整结果直接读取；处理中断会终止当前API子进程并保留原始尝试。SIGKILL/断电后如果仍检测到旧Codex进程，不重复派发该记录。需处理旧进程后再续跑。

## 进度与调试

- `tqdm` 显示数据核验、已返回的记录数、有效完成数、失败数、活动请求数。
- 默认每15秒输出总体进度和粗略剩余时间；`--debug` 同时显示单请求等待时间。ETA包含当前失败和并发变化，只是估算。
- `run.log` 为可读日志，`events.jsonl` 为结构化事件，`progress.json` 为当前快照。
- `review.html` 每30秒及结束时刷新文件；在浏览器刷新即可查看新结果。页面有经验索引、全部假设索引、搜索及角点显露/意外遮挡/UNKNOWN过滤。
- `--no-progress` 关闭tqdm，适合重定向日志；结构化进度仍保留。

```bash
tail -f reports/intention_outcome_hypothesis_bank_v1/run.log
```

## 超时与重试

默认每次请求超时900秒；每条经验最多3次尝试（首次+最多2次重试），退避5秒、10秒。超时、进程错误、输出Schema/引用校验失败均保留完整记录并有限重试；模型违反工具隔离则不重试。重试使用完全相同的输入，不传入失败回答或其他经验，不挑选“最好”的输出。只保留第一个完整合法输出用于汇总，所有尝试仍可审计。

`--timeout`、`--max-attempts`、模型、推理强度、分析代码与Prompt在首次准备时冻结；恢复时不能更改。耗尽次数的经验不会因为再次 `--resume` 获得无限重试。若以后修改实验配置，应新建 `--output`，保留原实验。

## 输出

- `protocol.json`、`prompt.txt`、`schema.json`、`code_snapshot/`：冻结版本。
- `experience_index.json`、`data_audit.json`：所有有效记录、来源、校验、排除/重复索引原因。
- `inputs/Xxxxx/`：原始Before/After快照、原始意图及动作、执行信息。保留索引动作的原始抬升参数及原生指令序列，不统一改成其他物理参数。
- `calls/Xxxxx/attempt_XX/`：精确请求、图像上传哈希、CLI输出、stderr、耗时、Token、错误。
- `experience_observations.json`：每条经验状态及完整意图—结果比较；未完成项的observation为null，不能当作无发现。
- `hypothesis_bank.json`：所有成功分析产生的假设，逐条对应证据，不去重、不合并。
- `unprocessed.json`：失败、中断、未启动、等待重试等未完成条目。
- `summary.json`：覆盖率、不同Mismatch类型、假设数、证据等级、调用及成本计数。
- `review.html`、`cases/`：全部可浏览证据。可直接从文件系统打开，无需Web服务。

只重建报告，不发模型请求（运行中的目录被锁定，先等任务结束或暂停）：

```bash
PYTHONPATH=. /home/sja/miniconda3/envs/cali/bin/python \
  experiments/intention_outcome_hypothesis_mining/run.py --render-only
```

退出码：0=完整完成/离线准备或渲染成功，2=分析结束或中断但有未处理项，1=配置或程序错误。不能只看tqdm到达100%判断分析覆盖，必须看 `processed_experiences / valid_experiences`。

## 数据与证据边界

有效性依据：物理执行标记、原生抓取指令、可读取且与索引哈希匹配的Before/After。不会依据DONE、面积增加、原目标成功或旧发现名单筛选。缺少原始意图时保留UNKNOWN，不从After补写。索引里的完全重复操作条目只保留一次；不同操作产生相同假设仍全部保留。

一次请求只含一条经验。意图提取和结果观察在同次调用里完成，因此不是两次盲法独立判断；逐字引用检查能减少意图改写，却不能消除模型受After影响的语义解读。`CONFIRMED`仅为模型视觉证据等级，不是独立材料真值，更不是机制已获物理验证。执行完成也不等于抓取成功。

## 测试（不使用真实模型/机器人）

```bash
PYTHONPATH=. /home/sja/miniconda3/envs/cali/bin/python -m unittest discover \
  -s experiments/intention_outcome_hypothesis_mining/tests -v
```

测试只创建临时合成图片、模拟CLI输出，覆盖零调用预检、完整汇总、重复假设保留、断点恢复、重试上限、取消、引用与四角校验、模型工具隔离及冻结配置保护。
