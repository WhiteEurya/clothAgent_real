# 独立 Frozen Graph-as-Policy Replay

2026-10-17 发布快照的跨设备安装、数据下载、路径挂载及最新保存图入口，见
[复现说明](../../reproduction/README.md)。本页保留各阶段命令及行为演进；下文
“本次最新采集图”是该段实验的起始图，发布时最新保存快照是
`runs/graph_recover_20261017_134332/episode_0001/latest.json`。

## 全屏实时 Graph 画布（2026-10-16）

Replay CLI 默认提供只读页面 `http://127.0.0.1:8090`；端口可用 `--graph-port` 调整，
实际地址见终端的 `LIVE GRAPH:`。页面仅包含全屏网格和节点连线，无工具栏、表格或侧栏。
拖动空白平移，滚轮或双指缩放，拖动节点调整布局；双击空白或按 F 适配全图。
每秒读取遥测，新节点自动加入，刷新保留当前视角和手动布局。
点击节点可查看参考图片和最近匹配观测，点击 recovery 可查看恢复前后图片；
图片在浮层中展示，Esc、右上角关闭按钮或点击外部空白关闭。拖动节点不会打开图片。
尚未采集或文件缺失的图片会明确显示暂无图片，不借用其他节点的观测。

绿色标出当前确认位置，蓝色虚线表示尚未确认到达的预计目标，紫色表示本次新增节点。
重新观测/匹配期间，上一位置标作“上次确认”；停止后标作“最后位置”。
robustness gate 触发的 recovery 用金色虚线分支显示：原节点 → recovery，只有可靠重匹配后
才显示 recovery → 实际匹配节点；没有可靠匹配时不绘制返回目标。失败显示“未解决”。
recovery 记录只写入可视化遥测的 `recoveries`，不添加 policy 节点、边，也不参与 state matching。
页面显示的是规划/验证执行阶段以及匹配结果，不将 controller 成功当作 cloth 成功。

每次运行保存 `live_graph/state.json`、`live_graph/index.html` 及 `live_graph/images/` 下的图片副本。
Replay 结束后本轮 HTTP 服务关闭，可直接在浏览器打开 HTML 查看最终快照及图片；
移动或分享离线快照时需保留同目录的 `images/`。HTTP 仅允许读取已导出的图片，不提供任意文件路径接口。
本次离线交互检查与 recovery 展示截图保存于 `reports/replay_canvas_20261016/`；其中的新节点、
当前位置及 recovery 事件是 UI 测试数据，不是实机执行结果。

入口：`cloth_agent.experiments.graph_policy_replay`。默认只做离线静态检查，**只有 `--execute` 会连接相机/机器人并移动 arm7**。已有节点继续按图选边；匹配失败时默认最多调用原 arm7 free planner 恢复 3 步，每一步后重新匹配。原 policy 不变，新观察和实测转移保存到独立扩展图。`--max-fallbacks 0` 可恢复严格 graph-only 模式。

当前冻结 policy：`runs/graph_policy_full02/policy_graph_v1.json`，有配套 SHA256 文件。来源是保留完整 14 条经验边的 `graph_v2.json`；原 graph_v1/v2/v3 不变。它沿用已完成的视觉合并，不重新生成 taxonomy。

| 条目 | 本轮 policy |
|---|---|
| e001 | PROGRESS，PREFERRED，replayable=true；curled 为中间进展 |
| n006/n007/n008 | 沿用合并状态 S6 |
| e006/e007 | NO_CHANGE / FAILED_EVIDENCE，replayable=false |
| e008 | S6 唯一 PREFERRED、replayable=true |
| n009/n010 | 沿用合并状态 S7 |
| e009 | NO_CHANGE / FAILED_EVIDENCE，replayable=false |
| e010 | S7 唯一 PREFERRED、replayable=true |
| e012 | UNVERIFIED_OUTCOME，replayable=false；S9 停止，REQUIRES_FRESH_VALIDATION |
| e013 | MISSING_AFTER，replayable=false；只作审计 |

共 12 个节点，其中 11 个有图可匹配，S11 为无观测审计节点。9 条可重放边、3 条失败证据边、2 条缺少验证的禁用边。从 S0 出发最多执行到 S9，本轮不执行 e012，也不声称完成整件衣服折叠。此策略独立于旧 graph_v3 的 S10 终点设置。

## Dry-run

```bash
python -m cloth_agent.experiments.graph_policy_replay \
  --graph runs/graph_policy_full02/policy_graph_v1.json \
  --dry-run
```

省略 `--execute` 也是 dry-run。它验证冻结来源、seed/图/代表 RGB 哈希、固定 primitive 结构、每个节点唯一首选边和停止位置；输出 `dry_run.json`、matcher 的实际字段投影和图片清单。**不采新图，不调用 GPT，不跑 IK，不代表实机验证通过**。测试使用注入的模型和 runtime 验证完整控制流，不能代替新衣服上的实际匹配。

## 真实 Replay

重新摆好衣服、移开手后，显式执行：

```bash
python -m cloth_agent.experiments.graph_policy_replay \
  --graph runs/graph_policy_full02/policy_graph_v1.json \
  --execute
```

默认使用 `gpt-6-astra`、arm7 Camera B 和 `config/dual_arm.local.json`，沿用当前 arm7 抓取偏移、夹爪对称旋转、IK、workspace/collision 配置和 dispatch 校验，不改变现有 collision 配置中的豁免。可配置 `--config`、`--model`、`--max-steps`（默认20）、`--max-calls`（默认64，包括有限格式修复）、`--match-threshold`（默认0.8）、`--timeout-s`、`--viser-port`、`--no-cache`。输出默认新建带 UTC 时间戳目录；指定 `--output` 时必须是不存在的路径，禁止恢复旧物理 dispatch。

## 强制控制流与信息边界

1. `observe` 获取新的 RGB-D（实机时包含现有 perception 准备动作）。
2. 新隔离 GPT 请求仅含当前 RGB、已有节点视觉条件、代表 RGB 和任务/左右约定。没有 edges、action、历史结果、previous match 或 expected target。S6 条件中原来提示 outgoing action 的一句话从 matcher 投影移除；原节点文字与删除记录保留在 policy 文件中。
3. Host 验证唯一可靠 MATCH，并选择唯一 `PREFERRED && replayable` 出边。多个首选边使图校验失败，不交给模型选择。
4. 独立 GPT 请求仅 grounding 该边的 relation/template 与固定 primitive slots。它看不到候选边集合；Host 校验 edge ID、action family、slot 顺序和数量，用当前 RGB-D 实测生成命令，不接受模型 Python。抓取下降由现有 host 应用一次。
5. 原 `Arm7Runtime.execute` 在 dispatch 前完成观测有效性、IK、workspace/collision/path 和到位检查。拒绝后停止，不改参数重试、不运行 free planner。
6. 每次仅 dispatch 一条边。执行成功后重新 observe，并以完全相同的盲 matcher payload 再做节点匹配。Host 才比较实际节点与 source/target；绝不把 controller EXECUTED 当成到达 target。即使达到 max-steps，也必须完成最后一次新图匹配。

每阶段最多一次格式修复，使用现有 isolated multimodal 接口与真实 image attachments。每次请求保存 prompt/hash、图片hash、响应、usage、耗时与重试信息。每次 dispatch 前和下一轮匹配前都检查 graph 未改变；运行快照单独冻结。

Full grounding 的 `reference_orientation` 提供 seed 动作各点的原始 World RPY。
参考观测可用时同时附上动作前 RGB（`action_reference`）及其相机到 World 的几何，
用于解释模板中的“保持原夹爪相对衣服角度”。兼容 seed observation ID 和累计图中的
观测图片路径；有保存校验和的参考图会先校验。缺少参考图时显式标记，不编造参考方向。
模型仍在当前图定位抓点和目标；历史 XYZ 不作为新目标，参考动作也不自动视为成功。
此补充不改变 action schema、Graph、安全检查或 minimal grounding 路径。

停止状态：`NO_MATCH`、`AMBIGUOUS`、`NO_REPLAYABLE_EDGE`、`EDGE_INSTANTIATION_FAILURE`、`SAFETY_REJECT`、`SOURCE_NODE_UNCHANGED`、`UNEXPECTED_NODE`、`REQUIRES_FRESH_VALIDATION`；另有执行故障 `ACTION_EXECUTION_FAILURE`、预算 `STEP_LIMIT`、中断/异常。所有情况均不 fallback。

输出包含 `policy_used.json`、每轮匹配及当前图片来源、Host选边、grounding、实测坐标、固定 motion、dispatch intent、实际执行记录，以及 `final_metrics.json` / `report.md`。真实复用仅在执行成功后记录；匹配结果未知不会标记 target 已到达。本轮禁止 e012，因此不输出整任务成功声明。

重建同一审阅规则可通过离线函数 `consolidate(source_path, new_output_path)`，只适用于保留 e000–e013 和已核验 S6/S7 合并的经验图；它不会在 replay 中调用，也不会自动推广到别的历史 run。

## Constraint ownership（v2 grounding 契约）

冻结 graph、动作 family/primitive 序列和原 Host validator 不变。旧模板中的约束按责任归属解释：

- GPT：验证当前 RGB 可见的 edge 语义关系，定位抓点、目标点和姿态参数。`visual_violations` 必须指出具体约束和当前可见违规证据；只有存在这类证据才能返回 `BLOCKED`。
- Host：继续承担 RGB-D 测量、IK、workspace、robot/camera collision、trajectory clearance 与 dispatch 校验。GPT 无须用 RGB 证明这些检查通过，也无权宣称可以跳过它们。Host 拒绝仍立即返回 `SAFETY_REJECT`。
- 不可预先观测的布料耦合、是否带动衣身、隐藏层关系：写入 `execution_risk`，不能仅因为无法证明这些效果就 `BLOCKED`。`READY` 只表示视觉语义候选，不是安全批准或成功保证。
- 如果确实缺少可定位的抓点/目标点等必要视觉参数，返回 `NEED_MORE_INFORMATION` 并填写 `missing_visual_parameters`；runner 停止，不猜坐标。Host 待验证项目及未知布料效果不得放入此字段。

新 schema 只作用于独立 runner，不改变旧 graph experiment。风险保存在 grounding、dispatch intent 和已执行 transition 中，责任分工保存为 `constraint_ownership.json`。模型的无证据 `BLOCKED` 会触发有界格式/契约修复，仍不合规则停止；Host 不自动把旧 `BLOCKED` 转成 `READY`。Prompt/schema hash 已变化，因此旧 grounding cache 不会被复用。无需重建 policy graph，启动命令不变。

Grounding 同步接收图中原始 `side_convention`，以“领口朝上、不镜像”的 viewer 左右定义袖侧，不能使用原始 image-left/right 替代。每次记录 `garment_frame`（当前领口中心、下摆中心与视觉依据）。对于左右袖 action family，Host 在派发前检查抓点位于该轴的指定侧，并保存 `garment_side_check.json`；领口朝下时，图里的左袖应位于画面右侧。轴不确定时停止获取信息。该检查依赖模型定位的领口/下摆，不等于独立证明定位正确；不改变抓取下降量或安全校验。


## NO_MATCH 恢复与扩展观察图

用户开启的恢复模式默认启用（CLI `--max-fallbacks 3`）；真实执行依然必须 `--execute`。

```bash
python -m cloth_agent.experiments.graph_policy_replay \
  --graph runs/graph_policy_full02/policy_graph_v1.json \
  --execute --max-fallbacks 3
```

发生 NO_MATCH 后，运行时 GPT 先依据当前 RGB 描述新的候选观察节点，存下图像引用及 hash；随后复用原 `codex_loop.loop(max_rounds=1)` 和 arm7 runtime，使用同一当前观测、原任务参考图及正常 motion contract 自由规划一步。结束后重新拍照、盲匹配：回到已有策略节点便恢复 Host 选边和 grounding。一次恢复最多一步，不连续无观察地执行多步。总动作仍受 `--max-steps` 限制；恢复规划及图描述共享 `--max-calls` 预算。恢复使用原 arm7 planner 的模型配置（目前 gpt-6-astra），`--model` 控制独立匹配/grounding/节点描述调用。

每次新增节点/实测转移生成不可覆盖的 `policy_extended_001.json`、`policy_extended_002.json` 等快照及对应 Markdown 有向图。原 policy 文件不动，扩展图可用同一 `--graph` 参数重载。扩展里的观察节点参与下一次匹配；恢复转移有 exact command、before/after、实际执行结果，并标 `OBSERVED_TRANSITION_ONLY`、`replayable=false`。它们是经验边，不从一次恢复自动生成可靠动作模板；再次遇到这些无策略的新节点时继续使用有界恢复。

匹配到既有节点后按其原规则走图，e012 的 `REQUIRES_FRESH_VALIDATION` 屏障不变。AMBIGUOUS、Host 安全拒绝、执行失败、执行后同一节点等不触发自由规划。达到恢复次数上限返回 `FALLBACK_LIMIT`。自由规划返回 complete 仅记为 `FALLBACK_REPORTED_COMPLETE`，不是图策略成功。

输出明确记录 `FALLBACK_FREE_PLAN`、`fallback_used`、`fallback_steps`、`fallback_transitions` 和 `extended_graph`。图执行与自由规划分开统计；一旦使用 fallback，不能宣称 graph-only success。原文中“所有情况均不 fallback”仅适用于 `--max-fallbacks 0` 的严格模式；新模式也不会因安全拒绝而转自由规划。

## 抓取下降量

2026-10-13 按用户要求取消右袖和右侧衣身额外 3 mm 下探。
所有图动作使用图记录的基础下降量（当前 15 mm），`right_side_extra_mm` 和每步额外偏移均为 0。
共享 Camera B 海绵桌面高度补偿继续生效，fallback 保持原有基础下降策略。
GPT 输出未扣减的表面高度，Host 施加一次基础偏移；每步仍保存 `grasp_offset_override.json`
和实际高度记录。原冻结图、禁用边及启动命令不变，新启动进程生效。


## Topology-local matching

默认首帧做 global matching；明确从标准初态摆放时可加 `--initial-node S0` 先验证S0。S0验证失败仍逐级兜底，不强迫接受初态。

执行已有 `S → T` 边后，首先只比较 S/T；可靠MATCH立即结束匹配。失败则扩展到 S/T、T 的一跳相邻节点（出入边都考虑）及明确的 recovery node，最多4个；有 recovery node 时预留1个位置，优先较新恢复节点。扩展仍失败才全局匹配。相同候选集合不会重复请求；free-planner恢复没有已知target时使用global。

每层默认只有当前RGB与候选节点的短文字signature/已有视觉条件，不含edges、历史动作、期望target标签、代表图片。只有该层返回AMBIGUOUS，才追加这层候选节点对应的代表RGB重新判别一次；仍不能可靠匹配才进入下一层。低置信度MATCH不强行接受，也不会仅因此加载代表图。节点描述来自原图，不重生成taxonomy。

每步 `matching_attempts.json` 保存层级、候选IDs、图片IDs、响应和耗时。`final_metrics.json` 新增以下累计指标（时间单位秒，包含该层代表图重试、有限格式修复及缓存处理的Host墙钟时间）：

- `local_match_time`
- `expanded_match_time`
- `global_match_time`
- `representative_image_fallback_count`

实机命令：

```bash
python -m cloth_agent.experiments.graph_policy_replay \
  --graph runs/graph_policy_full02/policy_graph_v1.json \
  --execute --max-fallbacks 3
```

明确标准初态时追加 `--initial-node S0`。原Host安全校验、右侧额外下降和恢复预算均不变。dry-run的代表图清单是可用证据库存，并不意味着每次模型请求都附上这些图。

## 状态不变也触发恢复

启用恢复时，图动作或一次fallback执行后仍匹配source（例如S2→S2），现在也进入有界fallback，而非直接结束。模型收到当前新观测、上一动作与匹配反馈，规划一个修正步骤；不自动原样重放旧edge，不把状态不变解释为抓空，也不自动增加下降量。再次观察后回到可用图节点则继续图策略；连续无进展消耗同一 `--max-fallbacks` 预算，上限后返回 `FALLBACK_LIMIT`。

`--max-fallbacks 0` 仍返回 `SOURCE_NODE_UNCHANGED`。安全拒绝、执行故障、AMBIGUOUS及非预期已知节点仍按原规则停止。新经验快照可只包含已有节点间的实测自环，无需捏造新节点。`recovery_context.json` 与 `fallback_intent.json` 记录触发原因。

## Online graph-growth（多 outcome，2026-10-13）

直接在现有 `graph_policy_replay.py` 增加 `--online-growth` 分支。未指定此参数时，原 Replay/Recovery 行为不变。继续使用 GPT-6 matcher 和原 arm7 runtime；没有接入 DINO、classifier，也没有修改 IK、collision、workspace、安全检查。此模式使用现有 folding 的速度/抓取下降配置，不调用 gravity-unfold 的 runtime。

### 启动与持续采集

以下命令会真实执行 folding；实现阶段只做离线测试，没有启动它：

```bash
python -m cloth_agent.experiments.graph_policy_replay \
  --graph runs/graph_policy_full02/policy_graph_v2.json \
  --online-growth --execute \
  --episodes 0 --max-steps 100 --max-calls 1024 \
  --output "/mnt/newssd/sja/clothAgent_real/runs/online_graph_growth_$(date +%Y%m%d_%H%M%S)"
```

`--episodes 0` 持续采集；每个 episode 完成或到达步数上限后，先关闭机器人连接，再提示重新摆放衣服，按 Enter 开始下一 episode，输入 q 结束。不会自动揉乱衣服或无观察地重复物理动作。默认 `--episodes 1` 只做一个 episode。每个 episode 分别受 `--max-steps` 物理尝试轮数和 `--max-calls` 模型调用预算约束；没有旧恢复模式的 3 次 frontier 限制。安全拒绝、执行故障、异常、中断会退出持续采集，不自动继续 dispatch。CLI `--max-fallbacks` 仅用于旧模式。

`policy_graph_v2.json` 是已有 S10 终点授权版本；也可以指定原 `policy_graph_v1.json`，此时会保留它的终点/禁用边约束，不自动替它放行 e012。旧状态没有可用 policy action 时返回 `NO_REPLAYABLE_ACTION`，不会为了持续运行而偷偷自由规划。

离线检查（不连接机器人、不调用模型）：

```bash
python -m cloth_agent.experiments.graph_policy_replay \
  --graph runs/graph_policy_full02/policy_graph_v2.json \
  --online-growth --dry-run \
  --output /tmp/folding_growth_preflight
```

跨进程续跑：使用上一采集目录的 `latest.json`，也可直接指定某个冻结的 `growth_XXXXXX.json`。新输出目录必须不存在：

```bash
python -m cloth_agent.experiments.graph_policy_replay \
  --graph runs/graph_policy_full02/policy_graph_v2.json \
  --online-growth --execute --episodes 0 \
  --growth-state /absolute/path/to/previous_growth_run/latest.json \
  --max-steps 100 --max-calls 1024 \
  --output "/mnt/newssd/sja/clothAgent_real/runs/online_graph_growth_$(date +%Y%m%d_%H%M%S)"
```

续跑核对 base policy、snapshot、代表 RGB 和动作来源。每个 episode 都从新观测开始，不恢复上次未完成的物理 dispatch。旧图和旧 episode 不覆盖。

### 多 outcome schema

新快照使用 `protocol = online-multi-outcome-v1`，把状态、动作及实际转移分开保存：

```json
{
  "nodes": {
    "S1": {
      "status": "EXISTING",
      "node": {"node_id": "S1", "observation_condition": "原有 decision conditions"},
      "observation_count": 8,
      "incoming_transitions": ["T0000001"],
      "outgoing_actions": ["e001"],
      "outcome_history": ["T0000002", "T0000003"]
    }
  },
  "actions": {
    "e001": {
      "source_node": "S1",
      "preferred": true,
      "replayable": true,
      "nominal_target": "S2",
      "edge": "原有 action template 与固定 primitive sequence",
      "outcomes": {
        "S2": {"seed_evidence": true, "observed_count": 4, "history": []},
        "S1": {"seed_evidence": false, "observed_count": 2, "history": []},
        "P000012": {"seed_evidence": false, "observed_count": 1, "history": []}
      }
    }
  },
  "attempts": [],
  "transitions": [],
  "observations": {},
  "merge_candidates": [],
  "episodes": [],
  "metrics_history": []
}
```

上例只展示结构，省略了证据列表实际内容。每个 `(source_node, action_id)` 的 action 身份固定；每条实际转移有独立 transition ID。nominal target 是旧 policy 的预期结果，不是唯一允许结果，不因新 outcome 被改写。seed evidence 不算本实验新采集次数。

节点 `status` 只有来源元数据 EXISTING / PROVISIONAL；不预定义衣物状态 taxonomy。新节点保存当前 RGB/hash、触发 source/action、task progress、GPT decision condition 和 representative observation。一次观察不会把新节点晋升为永久节点。

### 三层识别与行动选择

1. 每个 episode 首帧做全图 GPT decision-condition 匹配。未观察过的 audit-only sink 不参与匹配，但保留在图及可视化中。
2. 已知 S 优先选择唯一 preferred/replayable action；沿用原 grounding、RGB-D sampling、hem/side 检查和 `execute_policy_edge`。同一 action 可以在本 episode 重复执行，统计自然出现的多 outcome。
3. 执行成功后必须重新拍 RGB，即使刚好达到动作上限，也先完成这次 outcome 匹配。local 候选严格是 source S 加 `Outcomes(S,a)` 中所有可观察节点（包括旧 nominal outcome）。没有额外邻居层。
4. 仅明确 LOCAL_NO_MATCH 时查询整个已有 graph（包含 provisional）。可靠命中已有节点则保存 GLOBAL_REENTRY，不创建节点。匹配 payload 只有视觉签名/decision conditions，不泄漏动作、期望目标、执行历史。
5. 只有明确 GLOBAL_NO_MATCH 才调用 GPT 描述 PROVISIONAL_NEW_NODE。低置信 MATCH、未解决的 AMBIGUOUS 不等价于 NO_MATCH，不扩图。AMBIGUOUS 会按原机制附候选代表图再判一次；仍无法判断则停止本 episode。
6. 只有当前节点为 PROVISIONAL 且无 preferred/replayable action，才通过原 `free_plan_one` 执行一次 frontier folding action。已有节点没有 policy action 时不会触发 frontier。
7. frontier 执行之后同样 local → global → new。独立视觉 effect 判断用于决定该动作是否已有成功证据；controller EXECUTED、单纯换了 node、UNKNOWN 或 self-loop 都不能使它变为可复用动作。
8. frontier 到达不同节点且视觉为 PROGRESS/COMPLETE，并成功提取该**已执行命令**的相对模板后，记录 `replayable=true`、preferred。以后到该 provisional 节点优先用此固定 primitive/template 做新 RGB-D grounding，不复制旧 XYZ，不重新 free plan。该动作仍是一条有实测成功证据的一次性假设，不声称已验证普适性。模板描述失败时保留转移但不开放复用。

新模板允许复用不等于永久合并/晋升节点。涉及 provisional 的未解决多节点匹配可以形成 MERGE_CANDIDATE，注明 decision-equivalence 尚未确认；只留审计，不修改节点身份，也不永久 merge。

### 日志、统计与图

每次物理尝试先写 intent，执行后立即写结果。未匹配或部分执行的动作也保留在 attempts；只有有已识别终点的执行才写入 transitions。每条 transition 包括 source_node、selected_action、local_match（候选/结果/confidence/完整证据）、global_reentry、new_node_created、result_node、transition_type、planner_source、before/after 与原执行结果。

transition_type：EXPECTED、ALTERNATIVE_OUTCOME、NO_PROGRESS、GLOBAL_REENTRY、NEW_NODE。无法获得终点的 attempt 明确记 UNKNOWN/NOT_OBSERVED，不伪造转移。self-loop 优先判为 NO_PROGRESS，即使视觉 effect 错误宣称进展也不会提升 frontier action。

每次已识别转移打印 `ONLINE_GROWTH`，并持续保存：

- total_physical_actions（含部分物理执行，不含纯 IK 拒绝）
- graph_policy_actions / gpt_frontier_actions
- local_outcome_matches / global_reentries / new_provisional_nodes
- no_progress_self_loops
- multi_outcome_actions（具有至少两个实测 target 的 action 数）
- additional_observed_outcomes（各 action 在第一个实测 target 之外新增 target 的数量）
- current_nodes / current_transitions（后者为累计已识别实测转移事件数）
- graph_action_ratio = graph-policy physical actions / total physical actions
- new_node_rate = 动作后新建节点数 / total physical actions（首帧新节点不计入分子；零动作时 null）
- observed_outcome_counts：每个 action 各实测 target 的次数
- unmatched_physical_attempts：有物理执行但尚无可靠终点的尝试数

统计跨 episode 累积，`metrics_history.csv` 和 snapshot 的 metrics_history 保留比率变化轨迹；不把未知动作从分母删除。

目录结构：

```text
online_graph_growth_.../
  latest.json                       最新冻结 snapshot 的路径/hash
  episodes.json
  episode_0001/
    initial_observation/
    step_0000/
      dispatch_intent.json
      grounding.json / frontier/
      execution.json
      after_observation/
      outcome_match.json
      transition.json
    model/                          原 Requests 的 prompt/response/image-hash 审计
    growth_000001.json               更新即冻结；逐版本保存
    growth_000001.json.sha256.json
    latest.json
    growth_metrics.json
    metrics_history.csv
    growth_<episode末版>.dot
    growth_<episode末版>.svg
    growth_<episode末版>.md
    episode.json
```

图为完整已有 graph 加累计新经验，画作 State → Action → Outcome State。一个 action 方框向多个 target 分叉；绿色为 preferred nominal，橙色为 alternative，红色为 no-progress 返回 source，紫色为 global re-entry，灰色为原图禁用证据，provisional state 用橙色底。可视化不会把一次 trajectory 冒充整张 policy graph。使用 Graphviz `dot` 输出 SVG；没有 dot 时仍有 DOT 与 Mermaid Markdown，渲染故障不会丢失冻结 JSON。

### 测试

```bash
python -m pytest -q tests/test_online_graph_growth.py \
  tests/test_graph_policy_replay.py tests/test_graph_policy_matching.py
```

新增测试覆盖同 action 多 outcome、source self-loop 后继续执行、严格 local 候选、global re-entry、双重 miss 后建 provisional、歧义不扩图、successful frontier 跨 episode 复用、UNKNOWN/无进展不提升动作、模板失败保留转移、部分执行计数、安全拒绝不自由规划、重载防篡改、MERGE_CANDIDATE 不合并、持续 episode 人工 reset、dry-run 无硬件/模型。测试和真实图静态加载均不代表实机成功。

### 仅更新经验图，每步都由完整 GPT 规划

`--online-growth --collect-only --grounding-mode full` 载入已有图进行状态匹配和经验记录，
每一步重新调用完整 GPT planner。它不调用 Graph 选边或 edge grounding，不向动作 planner
提供节点、edge、旧动作模板或目标节点；仅提供当前 observation、任务目标和任务参考图。
已匹配节点没有出边时仍可规划，不依据 Graph 的终态节点提前结束，由 planner 根据当前
观测报告完成。状态匹配的可靠性检查、robustness prompt 和 Host 安全检查保持原有规则。

执行后的新动作、前后观测、匹配结果、实测转移和成功次数继续写入累计图。采集的新动作
保留为经验，`preferred=false / replayable=false`，不自动替换已有优先动作；原历史和基础图
不覆盖。`episode.json` 的 `runner_config.action_selection=FULL_GPT_EVERY_STEP` 标明此模式。
为兼容原图结构，独立 GPT 动作仍记录为 `planner_source=GPT_FRONTIER`，本模式不会出现新的
`planner_source=GRAPH` 执行。带入旧图的累计指标仍包含历史执行。

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.graph_policy_replay \
  --graph runs/policy_replay_20261017T022854003479Z/empty_policy.json \
  --growth-state runs/policy_replay_20261017T022854003479Z/latest.json \
  --online-growth --collect-only --grounding-mode full \
  --episodes 1 --max-steps 20 --max-calls 128 --max-recovery-attempts 0 \
  --output "runs/graph_collect_$(date +%Y%m%d_%H%M%S)" --execute
```

更新的累计图见新输出目录的 `latest.json`；实时画布默认 `http://127.0.0.1:8090`。
去掉 `--collect-only` 才恢复原有 Graph 优先、无可用边时 frontier 的行为。

### 从空图开始（不继承旧 folding graph）

`--from-scratch --online-growth` 与 `--graph` 互斥，也不能同时传 `--growth-state`。
初始化只保存 folding 任务目标、原 runtime 抓取偏移；节点、动作、转移与统计均为空。
不读取旧 seed trajectory 或旧图，不预定义节点分类。首张实测观测建立 provisional 节点，
首步使用 GPT frontier；之后沿用成功动作复用、local/global matching 和 provisional 扩图。
原有 IK、碰撞、workspace 与执行检查保持不变。任务目标不是已有 graph 节点。

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.graph_policy_replay \
  --from-scratch --online-growth --episodes 0 --max-steps 100 --max-calls 1024 \
  --output "/mnt/newssd/sja/clothAgent_real/runs/graph_from_scratch_$(date +%Y%m%d_%H%M%S)" \
  --execute
```

输出根目录的 `empty_policy.json` 是这张图的空白起点，`latest.json` 指向累计经验。
未来续建这张图时，用 `--graph <该目录>/empty_policy.json --growth-state <该目录>/latest.json`
替换 `--from-scratch`，并指定新的输出目录；保留原目录和证据。再次传 `--from-scratch`
会开始另一张独立空图。

不带 `--execute`（或带 `--dry-run`）仅生成空图 JSON/SVG，不启动机器人或模型。
空图没有预置终态节点；GPT frontier 的 `motion.complete()` 按现有逻辑记录为
`FRONTIER_REPORTED_COMPLETE`，不冒充已验证的永久终态节点。

### 根据采集图 replay，并修正无进展的 edge

`--online-growth --recover-no-progress --grounding-mode full` 从已有累计图选动作，
不与 `--collect-only` 同用。优先使用已验证的修正版、已有成功记录的动作、原有优先边；
如果该节点只有未验证的采集记录，则允许尝试最近一个带完整模板的候选，但不会伪造
其 `preferred/replayable` 或成功次数。完整全图匹配明确返回 `NO_MATCH` 时，使用当前
真实观测创建 provisional 节点并保存参考图；适用于初始观测、普通动作后和 recovery 后。
新节点无出边时调用完整 planner 规划一步，继续记录动作和实测转移。局部无匹配先做
全图匹配，匹配有歧义时停止，不用新节点掩盖不确定性。已有非 provisional 节点无动作时仍停止。

若动作确实执行且重新匹配仍是原节点，调用一次原完整 planner，提供所选 edge、
原动作、执行结果和重新采集的 observation。修正动作仍使用原机器人 primitive 和
Host 安全检查，不原样重复失败动作，也不将 self-loop 自动判为空抓并固定加深。
每个 episode 最多执行一个 recovery action；之后必须重拍、重匹配。

仅当 recovery 实际执行成功、视觉确认进展、可靠匹配到预期已有目标节点后，才替换
同一个 edge ID 的模板和完整动作（包括 Z）。若原候选只有自环、没有已知目标，要求
视觉进展且匹配到不同的已有节点。Recovery 后全图 `NO_MATCH` 则建新节点、记录
`NEW_NODE` 转移并从新节点继续；保留原 edge，不把新状态冒充原目标或已验证修复。
无法确认已有目标上的效果、无法编译模板、匹配有歧义或安全拒绝时
返回 `RECOVERY_UNRESOLVED`，原 edge 保留。控制器故障仍停止并报告 `CONTROLLER_FAULT`。

修正版保存在新 run 的 growth snapshot 中，`action.revisions` 保留旧 edge、修正版、
触发转移和验证转移；原输入图文件保持不变。再次加载更新图后，Graph 仍选择这个
edge，完整 planner 按当前 observation 实例化其已修正策略，允许自主选择 TCP 高度，
不送回要求 grasp 表面相对高度为零的旧 grounder。尚未修正的 edge 沿用原 grounding。

`episode_0001/step_*/recovery/audit.json` 记录恢复前后图、动作、执行结果、匹配和提交状态。
实时画布显示 recovery 分支，点击可查看前后图片。以下命令使用本次最新采集图：

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.graph_policy_replay \
  --graph runs/policy_replay_20261017T022854003479Z/empty_policy.json \
  --growth-state runs/graph_collect_20261017_112855/episode_0001/latest.json \
  --online-growth --recover-no-progress --grounding-mode full \
  --episodes 1 --max-steps 20 --max-calls 128 --max-recovery-attempts 0 \
  --graph-port 8090 --output "runs/graph_recover_$(date +%Y%m%d_%H%M%S)" --execute
```

这里 `--max-recovery-attempts 0` 关闭的是旧的 frozen replay grasp-handle fallback；
新模式的 self-loop recovery 由 `--recover-no-progress` 控制。下一次应将
`--growth-state` 指向本次新输出目录的 `latest.json`，以继续使用已提交的修正版。
没有 `--execute` 时只验证快照与生成静态图，不调用机器人或模型。
