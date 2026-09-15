# Fold 远程 Claude 桥接

`scripts/claude_fold_exploration.py` 默认使用 `--planner-backend remote`，SSH 主机默认是 `company-planner`。规划、运动提案、执行后评估和 fold supervisor 都通过 HTTPS 图片中转 + SSH 调用公司电脑的 Claude。Alienware 不需要本机 Claude。显式指定 `--planner-backend local` 可以使用旧调用链。

公司端需要免密 SSH、`curl`、`sha256sum`、GNU `timeout`、`date`、`sed`、Python 3.10+、Pillow 和已登录的 `claude`。非交互 SSH 环境的 PATH 必须能找到这些命令。桥接不指定模型，使用公司端 Claude CLI 配置的默认模型。远端调用是独立会话，不续用 Alienware 的 Claude session。

## Claude 自选图像工具

远程 backend 默认启用 `cloth_image` MCP 工具，内置工具仍只开放 `Read`。工具清单不是仅供阅读的 Markdown：桥接通过 `--mcp-config` 注册真实可调用工具，并用 `--allowedTools` 逐项授权。`--strict-mcp-config` 将本次 MCP 集合限定为图像工具。

每次请求自动在公司端 `/tmp/cloth_remote_<uuid>/` 写入：

- `image_tools.py`：独立工具服务，只依赖 Python + Pillow。
- `tool_list.json`：工具说明、参数和本次原始图像 ID。
- `image_tools.mcp.json`：本次 Claude CLI 使用的注册配置。
- `image_tools.settings.json`：仅对本次 CLI 生效的 Read 开始、成功、失败记录 hook。
- `view_<id>.png`：Claude 自主调用工具后生成的观察图。
- `image_tool_calls.jsonl`：参数、结果、耗时及错误记录。

可调用工具是 `image_info`、`rotate_image`、`crop_image`、`resize_image`、`map_point`。旋转支持任意角度，正值顺时针；缩放保持宽高比例（整数尺寸有舍入）。Claude 需要用 `Read` 看处理结果，不能仅凭工具返回的路径声称看过图像。工具不生成新的衣服内容、不镜像、不改变原图，不接触相机、深度和机器人。

原始 ID `image_0`、`image_1` 等对应同次请求的图像清单。每张处理图记录到原始 RGB 的像素中心仿射变换，可以连续裁剪、旋转、放大；`map_point` 返回原始图像编号及像素，并拒绝旋转空白区域。正式运动提案的运输点必须返回 `image_id` 和该来源图上的 `pixel_xy`，由本机映射、取整并查深度；不要把原图坐标与处理图 ID 混用。静态 reference、标注图、旧请求或未知 view ID、未通过哈希核对的处理图都不能作为运输坐标来源。Rxxx 身份不随看图旋转改变。每次最多生成 24 张图、调用 64 次图像工具，每张最多 16MP/单边 8192px。

折叠链路现在由 Claude 最终判断衣服方向、袖子和折叠目标。Molmo 仍提供当前 RGB 上的轴线/袖子点提示，但不以低置信度、反侧点或固定袖子比例带阻断 Claude。前置语义失败会记录为提示不可用，并提供当前 RGB；Claude 可以纠正或忽略 Molmo。抓点必须仍在当前衣物 mask 上，深度、工作区、夹爪高度、轨迹和 IK 检查保留。提供工具不代表 Claude 每次都会使用，也不保证语义判断正确。

公司端默认使用 `python3`。在公司端一次性安装依赖：

```bash
python3 -m pip install 'Pillow>=9.1'
```

若公司端已有 Conda 环境，把 `CLOTH_REMOTE_IMAGE_PYTHON` 设置为该环境 Python 的绝对路径，并确保非交互 SSH 能读到该环境变量。无需安装整套 clothAgent、Molmo 或机器人 SDK。工具脚本由 Alienware 自动同步；不会修改公司端全局 Claude 配置。缺少 Python/Pillow 时在调用 Claude 前报错，不默默关闭工具继续规划。

Alienware 使用已有 RGB 验证（无机器人连接）：

```bash
# 仅验证本地图像变换和坐标映射，不联网、不调用 Claude
python scripts/remote_image_tools_test.py test.png --offline

# 验证 HTTPS → 公司 Claude → 实际 MCP 调用 → 原图坐标返回
python scripts/remote_image_tools_test.py test.png --host company-planner

# 在公司电脑直接验证本机 Claude，省去 HTTPS 和 SSH；省略图片时生成测试色块图
python scripts/remote_image_tools_test.py --local-claude
```

在线测试检查真实工具审计中存在旋转、裁剪、缩放、坐标映射调用，并核对返回坐标；不会只相信模型自报成功。结果保存在 `results/image_tools_smoke/<时间>/`。`replayed_views/` 是本地根据审计重建的处理图，不是从远端下载的截图。

正常 fold 命令不变。规划阶段的 `<stage>_invocation.json` 新增 `image_tool_events` 和 `image_debug_directory`。监督、规划、运动提案和评估的每次调用，都在当前 iteration 下自动建立 `claude_image_tools/<阶段>_<ID>/`。没有 iteration 的独立调用使用对应诊断目录。

## 在 Viser 查看完整图像调试

正常折叠启动时继续使用 `--viser`，浏览器打开 `http://127.0.0.1:8765`。新增的 **Claude image operations** 文件夹按 iteration 和每次 Claude 调用分开显示：

界面最多展示最近两个 iteration。第三个出现时移除第一个的图片、面板、工具详情和轨迹，并清理对应缓存；历史文件不会删除，仍可单独打开历史 iteration 目录查看。

- 本次原始输入，以及每次旋转、裁剪、缩放后的图片，按事件顺序显示，不在通用图片组里重复展示。
- 原始图像编号、父图 ID、操作参数、输出尺寸、操作耗时、Read 耗时。
- `map_point` 的源图选点与原图映射点标记图。标记图明确标为 `DEBUG ONLY / NOT SENT`，不是 Claude 实际读取的图片。
- 完整展开的 request（实际 prompt、system prompt、schema、命令）、工具结果/坐标矩阵/哈希、Claude stdout、stderr、异常堆栈。文件内容不截断。

远端连续回传小型 JSON 事件，本地持续重建图片，Claude 尚未完成整轮推理时就能显示。正常退出时补传完整审计并按事件 ID 去重，然后清理远端 job；不会通过 SSH 回传 PNG。照片和诊断保存在被 Git 忽略的 `runs/` 或 `results/` 内。

图片标记含义：

| 标记 | 含义 |
| --- | --- |
| `VERIFIED` | 本地重建图的尺寸及 RGB 像素哈希与远端一致，来源链也已核对 |
| `HASH_MISMATCH` / `UNVERIFIED_REPLAY` | 原图或重建图未核对通过；不能当作远端一致图。完整记录提供双方哈希和 Pillow 版本 |
| `READ_STARTED` | 记录到 Read 开始，尚未确认成功 |
| `READ_COMPLETED` | Claude 的 Read 工具成功返回；不代表模型已经正确理解图片 |
| `READ_FAILED` | 记录到读取失败 |
| `UNKNOWN` | 记录尚未齐全，不能判断是否读取 |
| `NO_READ_RECORDED` | 已收完审计，但没有这张图的 Read 记录；可能没读，也可能公司端禁止了 hook |

Ctrl-C、超时或网络中断后，保留已收到的图片和事件，并标明 `INTERRUPTED`/`FAILED`。尚未收到的数据不会补造为成功，未收到结束标记时 `audit_complete=false`。若公司端 Claude 版本不支持/禁用了这些 hooks，不能证实读取成功，在线测试会报告缺少 Read 确认。

保存结构：

```text
iteration_001/claude_image_tools/<阶段>_<ID>/
    request.json
    image_debug.json
    events.jsonl
    stdout.log
    stderr.log
    exception.log       # 异常时写入
    images/             # 原图快照及处理图
    points/             # 本地选点调试标记
```

也可以在完全不连接硬件时查看在线工具测试结果：

```bash
python scripts/remote_image_tools_test.py test.png
python -m cloth_agent.fold_exploration_viser results/image_tools_smoke/<测试输出的时间目录>
```

第二条命令打开已保存的调试界面；测试进行中也能用同一输出目录启动查看。仅 `--offline` 的变换测试不包含 Claude 的 Read 记录。

`--local-claude` 使用同一份生产工具注册、Read hooks、远端工作目录清理和本地调试流，只将图片传输换成本机文件读取。它调用真正的 Claude，仍需 Claude 登录及模型网络访问，结果中明确标记未测试 HTTPS/SSH。不指定图片时生成带方向文字的四色测试图，不上传工作场景照片。

2026-09-15 本机 Claude CLI 2.1.228 的真实合成图测试通过：MCP 初始化/工具清单、image_info、顺时针旋转 90°、裁剪、放大 2 倍、map_point，以及原图/旋转图/放大图的成功 Read hook 均有记录。四张原始/处理图的像素哈希全部一致，最终坐标 `[128.25, 286.75]` 与工具结果一致；整次约 176 秒。120 秒时限的前一次测试超时，因此建议工具测试保留默认 300 秒。这不验证衣服判断准确率、HTTPS/SSH 链路或机器人执行。中间裁剪图没有 Read 记录，即使模型文字声称全部读过，也以审计为准。

## 快速测试

Alienware 安装本地 Molmo 袖部定位依赖，可在项目目录运行：

```bash
bash scripts/setup_molmo.sh
```

脚本创建或复用 Python 3.11 的 `molmo` Conda 环境，安装指定版本的依赖、下载 `allenai/MolmoPoint-8B` 并检查 CUDA、模型配置和处理器缓存。它不启动机器人，完成后会打印带有正确 `--molmo-python` 路径的折叠命令；完整模型推理仍需另行验证。

仅有一张 RGB PNG 时，先检查传输和公司 Claude 的图像读取：

```bash
python scripts/remote_planner_test.py test.png
```

`--mock` 可在不联网时测试传输命令和 JSON parser；它不验证公司 Claude。

已有成功保存的 perception run 时，运行实际 fold planner 链路：

```bash
python scripts/remote_fold_smoke.py \
  --run-dir runs/<已有的run目录名> \
  --host company-planner \
  --step left_sleeve \
  --mode ACQUISITION_PROBE
```

该测试需要该 run 的 `workspace/perception_views/` 中有匹配的 RGB、Rxxx guide、garment mask、XYZ 和本地表面测量文件；单张 RGB 不足以验证本地 grounding。它调用生产代码中的 supervisor 和 fold planner，用本地测量生成并检查 Proposal，写入 `results/remote_fold_smoke/<时间>/proposal.json`。测试不采图、不连接机器人、不做控制器 IK 或物理执行。只验证传输时使用第一个命令。

正常 fold 启动时保留原参数，或显式附加：

```text
--planner-backend remote --remote-planner-host company-planner
```

启动日志会显示 `planner_backend='remote'` 和 `remote_planner_host='company-planner'`。相机或 perception 阶段失败时仍会在 Claude 调用之前停止。

单腕部相机模式下，fold 会在 `--real --confirm-real` 已设置、配置只有一台活动相机且当前观测相机与配置一致时，向 session 传入 `single_view_confirmed=True`，无需额外命令行参数。相机模式在规划前检查，并在启动录像/执行前重新核对 run 元数据；双相机配置意外只得到单视角时仍会阻止执行。日志显示实际 `active_cameras` 和 `single_view_confirmed`。

## 数据与动作边界

左右袖现在统一为“衣领朝上、下摆朝下时的图像左／右”。默认 remote 折叠链路在袖子定位前新增一次 Claude 图像准备调用：Claude 自己选择旋转、裁剪或缩放，并 Read 最终选定图；本地验证像素、来源及方向声明后，把同一张 RGB 交给 Molmo。Molmo 只提供这张图上的区域提示，本地映射回原始 Cam A，再把标注图交给 Claude 最终判断。固定相机显示旋转不再被当成衣服已经摆正。原图已摆正时允许直接选原图，无需强制重复旋转。

Viser 的 `Claude image operations` 显示操作过程；iteration 摘要显示 `Claude → Molmo → Claude` 交接状态、实际输入路径／哈希和三个坐标系的点。`claude_molmo_orientation/molmo_input/camera_0_A.png` 是 Molmo 实际输入，`molmo_handoff.json` 保存提示词，`molmo_sleeve_locator/pixel_mapping.json` 保存映射。方向不明确、未 Read、哈希／来源不合法或调用失败会停止交接，不会退回旧图。新增调用会增加每次袖子步骤的耗时；其他本地安全检查继续执行。完整文件说明见 [fold_garment_frame.md](fold_garment_frame.md)。

远端仅收到白名单中的 RGB（包括 RGB 上的 Rxxx 标注、RGB 视频接触图）和筛选后的语义任务/历史。不会发送深度图、热图、XYZ、标定矩阵、工作区数值、完整机器人状态或本地文件清单。上传 PNG 原字节，远端下载后先验证 SHA256。

远端先选择现有 Rxxx，再显式提出完整动作序列。move 使用选定 grasp 或当前 upright RGB 的 pixel，加相对抓取高度与相对 Home yaw；这些是运动提议，不是远端测量的坐标。本地将 upright pixel 逆旋转回原图，查询保存的深度/XYZ；抓取高度仍由原 `resolve_grasp_height` 决定。不会补默认动作、猜缺失深度或把完整 fold 自动改成 probe。该远程接口要求 closure 精确对应选定 Rxxx，不使用旧本地生成器可选的偏移抓取。

原有 Rxxx 可执行性、任务模式、工作区、IK 和执行校验继续生效。网络/Claude/schema/grounding 失败不返回 Proposal；重试耗尽后远程 supervisor/evaluator 不使用旧的默认状态 fallback。每个远端 job 使用 UUID 临时目录、远端 EXIT trap 和本地 finally 的有界尽力清理。公网中转服务的文件按一小时有效期处理，删除远端目录不会删除公网副本。

## 离线回归

## 越界点离线诊断

每次远程规划尝试会在 `iteration_*/planning_attempt_*/` 下保存 `workspace_diagnostics.json`、`workspace_targets_raw.png`、`workspace_targets_upright.png` 和 `workspace_base_xy.png`。红色点表示本地工作区检查拒绝；JSON 同时给出动作编号、当前 upright 像素、标定后的机器人 Base XYZ（毫米）和左右边界有符号距离。图像只用于 Alienware 本地调试，不会上传给 Claude。基坐标图使用保存的标定结果，不能代替 IK 或标定精度验证。

也可以对历史失败调用完全离线重绘：

```bash
python scripts/debug_workspace_targets.py \
  --run-dir runs/<run> \
  --motion-json runs/<run>/results/fold_exploration/<stamp>/iteration_001/planning_attempt_fold_01/pixel_motion_invocation.json \
  --visual-json runs/<run>/results/fold_exploration/<stamp>/iteration_001/planning_attempt_fold_01/visual_planning_invocation.json
```

Viser 会按“Before / Workspace targets / Actual Claude RGB inputs / Rollout”分组显示图片；顶部 timing 面板显示每个阶段的细分耗时。日志里的 `+Xs` 是从运行开始的累计时间，`+Ys` 是相邻事件间隔，嵌套阶段不要相加。

```bash
python -m pytest -q tests/test_remote_fold.py tests/test_fold_exploration_pipeline.py tests/test_auto_exploration.py
```

这些测试使用模拟传输和保存的合成 RGB/几何数据，验证生产 planner 路由、本地 grounding、坐标逆旋转、失败拦截和清理；不会调用机器人或公司 Claude。
