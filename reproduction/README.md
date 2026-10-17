# Graph 实验跨设备复现（2026-10-17 快照）

现成结果与真实动作对照见 [Graph Demo](../examples/graph_demo/README.md)：
包含可打开的图、节点原图、一次真实状态变化和一次真实无进展案例，以及自动比对脚本。
只想查看结果时下载 [Demo ZIP](https://github.com/WhiteEurya/clothAgent_real/releases/download/graph-demo-20261017/graph-demo-20261017.zip)
并打开 `graph_demo/index.html`。需要以下自动对比脚本时，使用 `graph-demo-20261017`
标签的代码；409 MiB 原始数据包仍由 `manifest.json` 指向 `graph-repro-20261017` Release。

本目录对应 GitHub Release **graph-repro-20261017**。代码、实验 Markdown、测试和
依赖版本位于 Git；真实观测、图快照、轨迹、模型请求/响应和诊断产出位于 Release
分卷数据包。仅 `git clone` 不包含这些实验数据。

## 1. 本次包含什么

- 当前 `cloth_agent/experiments/` 的全部实验实现、相关 runtime 修改及测试。
- `docs/experiments/` 的全部流程说明，以及更新后的 pipeline 文档。
- 当前 graph 的 5 个 run：空图起点、采集过程和三次恢复尝试，具体清单见 manifest。
- 本次实机的 `config/dual_arm.local.json`；标定、参考图、机器人记录在 Git。
- 原始 RGB/深度/点云、图快照、动作代码、模型请求与响应、匹配及执行反馈。
- 不包含无关实验、完整录像、生成的 SVG/DOT/HTML 和 live_graph 展示副本。
  `manifest.json` 的 `omitted_files` 列出明确省略的录像/可视化；历史 JSON 仍保留录像
  路径作为审计信息，但数据包不包含这些可选录像。RGB 等核心证据的哈希不变。
- `manifest.json` 列出分卷大小和 SHA256；Release 的 `files.jsonl` 给出逐文件路径、
  大小和 SHA256。数据未改写，冻结图及核心证据哈希保持原样。

共 **1,235 个文件，548,122,673 字节原始数据，428,714,320 字节下载包**，
约 409 MiB 下载量。预留 2 GB 用于缓存与解压即可（容器镜像另需空间）。
其他 graph 历史和其他实验仍保留代码、测试、流程文档，但未包含它们的数据；本包
只承诺恢复下列当前 graph 复现链，不是全部历史实验备份。不包含账号凭据或缓存。

## 2. 下载、校验与恢复

在新设备克隆并切换到发布标签，在仓库根目录执行（Linux x86_64）：

```bash
git clone https://github.com/WhiteEurya/clothAgent_real.git
cd clothAgent_real
git checkout graph-demo-20261017
python3 -m venv .venv-restore
.venv-restore/bin/pip install zstandard==0.25.0
.venv-restore/bin/python scripts/restore_reproduction.py \
  --cache /your/data/cloth-release-cache --ssd-root /your/data/cloth-ssd
```

把 `/your/data` 换成有足够空间的目录。下载从 `manifest.json` 指定的 GitHub Release
获取；已有且校验通过的分卷会复用。中断产生的 `.download` 文件下次会重新下载。
脚本逐分卷校验，再解压到本仓库和 `--ssd-root`。已有同名文件内容不同时会拒绝覆盖，
因此推荐使用干净克隆。`--verify-only` 只下载并校验分卷，不解压。

## 3. 保留历史绝对路径运行

历史 `latest.json`、冻结图和观测记录包含原机器的绝对路径。直接批量替换字符串会
破坏哈希及证据引用，因此使用容器挂载保留原路径；宿主机的仓库目录可以任意命名。
另外，`config/run_storage.json` 要求 `/mnt/newssd` 为挂载点，下面的 bind mount 满足它。

```bash
docker build -f reproduction/Dockerfile -t cloth-repro:20261017 .
docker run --rm --network none \
  -v "$PWD:/home/sja/clothAgent_real" \
  -v /your/data/cloth-ssd:/mnt/newssd \
  cloth-repro:20261017 \
  python -m cloth_agent.experiments.graph_policy_replay \
    --graph runs/policy_replay_20261017T022854003479Z/empty_policy.json \
    --growth-state runs/graph_recover_20261017_134332/episode_0001/latest.json \
    --online-growth --recover-no-progress --grounding-mode full --dry-run \
    --output /mnt/newssd/reproduction/check_001
```

每次使用新的输出目录。预期状态是 `DRY_RUN_OK`，`robot_calls=0`、`model_calls=0`。
此检查加载图、校验冻结快照并生成静态图；不是重演物理轨迹，也不调用模型重新推理。
接着在同一容器路径映射下比较结果：

```bash
docker run --rm --network none \
  -v "$PWD:/home/sja/clothAgent_real" \
  -v /your/data/cloth-ssd:/mnt/newssd \
  cloth-repro:20261017 \
  python scripts/compare_graph_demo.py --snapshot /mnt/newssd/reproduction/check_001/latest.json
```

预期 `PASS`，7 个状态、14 个动作、11 条已观测转移。还可在宿主仓库根目录执行
`python scripts/verify_demo_evidence.py --demo-only` 核对 43 个小样例证据副本；
恢复数据后执行 `python scripts/verify_demo_evidence.py` 核对全部 51 项原始证据。
这两个核对脚本只使用 Python 标准库。完整比对范围和真实预期值见 Demo README。

本机无 Docker，Dockerfile 尚未实测构建；本机相同 Python 环境的上述离线命令已通过。
依赖锁定在 `requirements-cali.txt`；模型服务、SDK、驱动和硬件差异不属于位级确定性复现。

## 4. 实验版本与流程

| 阶段 | 入口/说明 | 本次保留的证据 |
|---|---|---|
| 原始 seed → 编译 graph → 合并/策略化 | [graph_policy.md](../docs/experiments/graph_policy.md) | 历史路径 `runs/graph_policy_full02/`；本精简包不含该旧实验数据 |
| 从空图开始 | [graph_policy_replay.md](../docs/experiments/graph_policy_replay.md) 的 `--from-scratch --online-growth` | `runs/policy_replay_20261017T022854003479Z/empty_policy.json` |
| 采集实际动作与转移 | 同文档的 `--collect-only --grounding-mode full` | `runs/graph_collect_20261017_112855/episode_0001/latest.json` → `growth_000084.json` |
| 已采集图重放及无进展恢复 | `--online-growth --recover-no-progress --grounding-mode full` | `runs/graph_recover_20261017_130922/`、`133136/`、`134332/` |
| 最新保存快照 | 以上最新目录的 episode 指针 | `runs/graph_recover_20261017_134332/episode_0001/latest.json` → `growth_000096.json` |

“最新保存”不表示该次物理运行完整结束或整件衣服折叠成功；最新目录没有顶层
`latest.json`，因此明确使用 episode 内的指针。离线读出的累计统计为 7 个节点、
11 个转移、13 次 physical actions，graph-policy actions 为 0，不能宣称 graph-only success。
此前完成退出的恢复运行也保留，不能仅按 growth 文件序号跨不同 run 判断新旧。

本包证据包括 graph/snapshot/sha256、before/after 观测、RGB/深度与点云、模型输入
及响应、动作程序、执行反馈、匹配结果、recovery audit、统计与画布。以每个 run 中
实际存在的文件为准；中断或失败实验不会补造缺失的结果。

## 5. 其他实验文档入口

- [决策状态发现](../docs/experiments/decision_state_discovery.md)
- [最小动作模板](../docs/experiments/seed_minimal_templates.md)
- [动作鲁棒性](../docs/experiments/action_robustness.md)
- [边修复恢复](../docs/experiments/edge_repair_recovery.md)
- [注册场](../docs/experiments/registration_field.md)、[高度审计](../docs/experiments/registration_height_audit.md)
- [深度对齐审计](../docs/experiments/host_depth_alignment_audit.md)
- [重力展开](../docs/experiments/gravity_unfold.md)
- [局部 Matcha](../docs/experiments/local_node_matcha_e000.md)、[局部 RoMa](../docs/experiments/local_node_roma_e000.md)
- [Arm7 pipeline](../docs/codex_arm7_pipeline.md)、[手柄控制](../docs/gamepad_teleop.md)

历史文档中的具体 run 是当时的实验输入，不能全部替换为“最新图”。本页给出本次发布
选用的最新快照与入口，原文保留实验背景。硬编码的原路径在上述容器挂载内成立。

## 6. 模型与实机复现

默认离线容器不联网、不映射设备。需要重新调用模型时，在自己的环境安装并认证
项目使用的 Codex CLI，配置相应模型权限；本发布不携带账号凭据或 CLI 登录状态。
原始 prompt/response 在归档中，可先离线审阅。模型后端变化可能导致动作不同。

需要实际执行时，先完成离线检查，再根据原文选择有限 episode/step 的命令并显式加
`--execute`。使用现场机器人与 Camera B 时还需要 xArm 网络、RealSense USB/驱动、
Codex CLI 和对应相机/机器人权限。提供的配置只对应本次 Arm7、相机及安装几何；
换机器人或移动相机/底座必须重新校准，不能把历史标定当作另一套硬件的测量值。
Docker 示例只提供离线复现，不把 `--execute` 偷加到启动命令。

## 7. 验证范围

核心 graph 测试 151 项通过，手柄相关测试 47 项通过，最新快照离线加载通过。
独立代码目录恢复数据后，graph、手柄及归档工具合计 203 项测试通过；
精简包 1,235 个文件已在独立目录解压并通过 SHA256 校验，最新图引用的
62 条带哈希的核心证据全部包含；详情见 `validation.json`。
完整测试套件另有历史失败/错误，不能据此声称全库测试通过；详见 `validation.json`。
容器构建及另一台设备的实际运动尚未验证。
