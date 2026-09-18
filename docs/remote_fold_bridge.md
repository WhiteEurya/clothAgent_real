# Fold 远程 Claude 桥接

`scripts/claude_fold_exploration.py` 默认使用 `--planner-backend remote`，SSH 主机默认是 `company-planner`。规划、运动提案、执行后评估和 fold supervisor 都通过 HTTPS 图片中转 + SSH 调用公司电脑的 Claude。Alienware 不需要本机 Claude。显式指定 `--planner-backend local` 可以使用旧调用链。

公司端需要免密 SSH、`curl`、`sha256sum`、GNU `timeout`、`date`、`sed`、Python 3.10+、Pillow 和已登录的 `claude`。非交互 SSH 环境的 PATH 必须能找到这些命令。桥接不指定模型，使用公司端 Claude CLI 配置的默认模型。远端调用是独立会话，不续用 Alienware 的 Claude session。

## Claude 自选图像工具

远程 backend 默认启用 `cloth_image` MCP 工具，内置工具仍只开放 `Read`。工具清单不是仅供阅读的 Markdown：桥接通过 `--mcp-config` 注册真实可调用工具，并用 `--allowedTools` 逐项授权。`--strict-mcp-config` 将本次 MCP 集合限定为图像工具。

每次请求自动在公司端 `/tmp/cloth_remote_<uuid>/` 写入：

- `image_tools.py`：独立工具服务，只依赖 Python + Pillow。
- `tool_list.json`：工具说明、参数和本次原始图像 ID。
- `image_tools.mcp.json`：本次 Claude CLI 使用的注册配置。
- `image_tools.settings.json`：仅对本次 CLI 生效的 Read/MCP 生命周期与图片内容检查 hook。
- `view_<id>.png`：Claude 自主调用工具后生成的观察图。
- `image_tool_calls.jsonl`：参数、结果、耗时及错误记录。

可调用工具是 `list_images`、`image_info`、`view_image`、`rotate_image`、`crop_image`、`resize_image`、`map_point`。`view_image` 直接返回原图或已保存视图；旋转、裁剪、缩放也直接返回 MCP image block 和元数据，不再要求额外 `Read` 才能看见结果。旋转支持任意角度，正值顺时针；缩放保持宽高比例（整数尺寸有舍入）。路径或成功状态不能代替图片内容。工具不生成新的衣服内容、不镜像、不改变原图，不接触相机、深度和机器人。

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

在线测试检查真实工具审计中存在旋转、裁剪、缩放、坐标映射调用，并核对返回坐标及 CLI 中的原图／旋转图／放大图像素；不会只相信模型自报成功。结果保存在 `results/image_tools_smoke/<时间>/`。`replayed_views/` 是本地根据审计重建的处理图，不是从远端下载的截图。CLI 自动将 PNG 转为同尺寸 JPEG 时，使用下述转码校验；CLI 隐式缩放、错误图片或超出容差的内容变化仍会阻止交接。

正常 fold 命令不变。规划阶段的 `<stage>_invocation.json` 新增 `image_tool_events` 和 `image_debug_directory`。监督、规划、运动提案和评估的每次调用，都在当前 iteration 下自动建立 `claude_image_tools/<阶段>_<ID>/`。没有 iteration 的独立调用使用对应诊断目录。

## 在 Viser 查看完整图像调试

正常折叠启动时继续使用 `--viser`，浏览器打开 `http://127.0.0.1:8765`。新增的 **Claude image operations** 文件夹按 iteration 和每次 Claude 调用分开显示：

界面最多展示最近两个 iteration。第三个出现时移除第一个的图片、面板、工具详情和轨迹，并清理对应缓存；历史文件不会删除，仍可单独打开历史 iteration 目录查看。

- 本次原始输入，以及每次旋转、裁剪、缩放后的图片，按事件顺序显示，不在通用图片组里重复展示。
- 原始图像编号、父图 ID、操作参数、输出尺寸、操作耗时、Read 耗时。
- `map_point` 的源图选点与原图映射点标记图。标记图明确标为 `DEBUG ONLY / NOT SENT`，不是 Claude 实际读取的图片。
- 完整展开的 request（实际 prompt、system prompt、schema、命令）、工具结果/坐标矩阵/哈希、Claude stdout、stderr、异常堆栈。文件内容不截断。

远端连续回传工具审计，本地持续重建图片，Claude 尚未完成整轮推理时就能显示。CLI 原始消息流还包含实际返回的图片块；本地解码并核对尺寸和像素哈希。正常退出时补传完整审计并按事件 ID 去重，然后清理远端 job。照片和诊断保存在被 Git 忽略的 `runs/` 或 `results/` 内。

图片标记含义：

| 标记 | 含义 |
| --- | --- |
| `VERIFIED` | 本地重建图的尺寸及 RGB 像素哈希与远端一致，来源链也已核对 |
| `HASH_MISMATCH` / `UNVERIFIED_REPLAY` | 原图或重建图未核对通过；不能当作远端一致图。完整记录提供双方哈希和 Pillow 版本 |
| `READ_STARTED` | 记录到 Read 开始，尚未确认成功 |
| `READ_COMPLETED` | 仅记录 Read 调用完成；不证明返回中有图片，也不证明模型理解 |
| `READ_FAILED` | 记录到读取失败 |
| `UNKNOWN` | 记录尚未齐全，不能判断是否读取 |
| `NO_READ_RECORDED` | 没有 Read 记录；直接使用 MCP 图片时属正常情况 |
| `returned_image_status=VERIFIED` | PostToolUse 返回中有可解码、与该视图匹配的图片 |
| `stream_image_status=VERIFIED` | 关联的 CLI tool_result 中有与该视图匹配的图片 |
| `image_delivery_status=VERIFIED` | CLI 图片匹配，且同次调用没有已知 hook 内容失败；不证明服务端收到或模型理解 |
| 内容状态 `VERIFIED_TRANSCODE` | 同尺寸 JPEG 通过来源关联、重新编码及有界像素比较；这是近似内容校验，不是像素哈希相等 |
| 内容状态 `SIZE_MISMATCH` / `CONTENT_MISMATCH` / `IDENTITY_MISMATCH` | 返回尺寸变化 / 像素内容校验失败 / 来源元数据不一致 |
| 内容状态 `UNAVAILABLE` / `UNKNOWN` | 已观察到空、损坏或失败返回 / 尚无足够证据 |

原始 PNG 和本地重建视图仍要求远端／本地 RGB 哈希完全一致。JPEG 校验使用返回文件的量化表和色度采样参数重新编码源图，再逐像素比较解码结果，不做对齐、缩放、模糊或感知哈希匹配。每个颜色通道的整图平均绝对差不超过 2、均方根差不超过 4、任意 32×32 分块均方根差不超过 6，同时返回图相对原图的平均绝对差不超过 15（均为 0–255 单位）。非标准 EXIF 方向和不支持的颜色模式会被拒绝。阈值允许编码器差异，也意味着容差以内的细微变化无法被完全排除；它不验证衣领识别是否正确。

通过校验后，Molmo 使用 CLI 实际返回图片解码得到的 RGB，以 PNG 无损保存；坐标映射仍使用已验证的同尺寸源视图。交接前再次检查源图像素哈希和返回文件字节／像素哈希。Viser 分别显示重建源图与 `CLI RETURNED IMAGE`。错误分类与转码差异分开记录，不增加重试或重置编辑预算。

Ctrl-C、超时或网络中断后，保留已收到的图片和事件，并标明 `INTERRUPTED`/`FAILED`。尚未收到的数据不会补造为成功，未收到结束标记时 `audit_complete=false`。缺少 hook 时对应状态为 UNKNOWN；CLI 内的实际图片仍可独立验证。若 CLI 没有输出可验证图片，方向交接和在线图片测试不会仅凭工具成功状态放行。

保存结构：

```text
iteration_001/claude_image_tools/<阶段>_<ID>/
    request.json
    image_debug.json
    events.jsonl
    image_delivery.jsonl # CLI 图片内容摘要，不含 base64
    stdout.log
    stderr.log
    exception.log       # 异常时写入
    images/             # 原图快照及处理图
    returned_images/    # CLI 实际返回的图片原始字节，按 encoded_sha256 命名
    points/             # 本地选点调试标记
```

也可以在完全不连接硬件时查看在线工具测试结果：

```bash
python scripts/remote_image_tools_test.py test.png
python -m cloth_agent.fold_exploration_viser results/image_tools_smoke/<测试输出的时间目录>
```

第二条命令打开已保存的调试界面；测试进行中也能用同一输出目录启动查看。仅 `--offline` 的变换测试不包含 Claude 的 Read 记录。

`--local-claude` 使用同一份生产工具注册、Read hooks、远端工作目录清理和本地调试流，只将图片传输换成本机文件读取。它调用真正的 Claude，仍需 Claude 登录及模型网络访问，结果中明确标记未测试 HTTPS/SSH。不指定图片时生成带方向文字的四色测试图，不上传工作场景照片。

2026-09-15 的旧版测试曾记录到原图/旋转图/放大图的成功 Read hook，不能据此证明返回含图片。2026-09-17 的新验证使用真实 Claude CLI 2.1.228、合成图片与本地模拟 API：`view_image` 和 `rotate_image` 的图片在 PostToolUse、CLI tool_result、下一次本地 API 请求中均可解码且像素哈希一致，方向交接校验通过，全程没有额外 Read。这验证本机 CLI/MCP/校验接线，不验证远端生产 API、模型视觉准确率或机器人动作。

同日使用仓库真实衣服照片（720×1280 PNG）重复该 CLI／本地模拟 API 测试，复现原图及旋转图自动转为 JPEG。两图均通过 `VERIFIED_TRANSCODE`，重新编码比较的最大通道平均绝对差约 0.69，最差分块均方根差约 2.24；下一次本地 API 请求也包含相应 JPEG，Molmo 输入准备成功。此测试没有调用真实模型或机器人。

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

### 夹爪完成反馈

需要跳过模型和相机、直接复现已记录轨迹时，可运行 `python scripts/replay_gripper_test.py`。默认模拟，不连接机器人；真实执行用 `python scripts/replay_gripper_test.py --real --confirm-real`。脚本固定复现这次记录的 10 步（抓取 `[410.527, -149.384, 36.690]`，放置 `[382.736, -79.166, 42.690]`，单位 mm），不根据当前衣物重新定位。它使用当前 `config/robot.example.json`，也可用 `--config` 指定折叠运行所用的配置；保留工作区、TCP、控制器 IK 与夹爪反馈检查。

每次创建 `runs/gripper_replay_<UTC时间戳>/`，保存配置快照、轨迹源码、逐步完成日志和 `results/recorded_gripper_test.trace.json`。Home 仅在记录的第 10 步执行，失败或 Ctrl+C 后不额外发送释放或回 Home。模拟只能检查轨迹与调用链；实际夹爪反馈需真实执行验证。

真实执行时，open/close 发出命令后由本地持续读取夹爪位置、状态和错误码；只有实际位置进入目标容差内，且有效状态为 stop/grasp（不是 moving），才允许后续动作。打开容差由 `gripper.open_tolerance_pulse` 配置，默认 15 pulse（允许范围 0–25）；目标 850、反馈 840 且 stop 时可通过，目标本身不变。这个允许误差不代表已经标定实际全开端点。闭合仍使用独立的 5 pulse 容差，打开容差不影响闭合判断。

真机曾在关闭仅从 840 移动到 834 时就返回 grasp；打开过程中也会短暂返回 grasp。因此已删除“闭合位移超过 5 pulse 且 grasp 就完成”的旁路。grasp、发生过位移、短暂位置不变都不能替代到达目标的检查。夹住较厚衣物而无法到达闭合目标时会持续等待，需要人工 Ctrl+C；在可靠接触判据验证前，不自动放行这类情况，也不加力重发命令。视觉 hold 检查仍需执行。

夹爪未完成、正在运动或暂时读取失败时持续等待，不再因 10 秒超时或单次读失败退出。反馈始终不可用时也保持等待，可以 Ctrl+C 中断。仅明确的夹爪硬件错误、发送命令失败或人工中断会终止当前轨迹，并阻止自动释放、自动 Home 和后续折叠动作。命令前也会等待有效静止反馈；关闭／打开命令只发送一次，不因读失败重发。

旧配置字段 `gripper.completion_timeout_s` 保留兼容，现仅作为慢等待日志阈值（默认 10 秒），不再是退出期限；`settle_s` 不参与完成判断。正常等待每 0.5 秒输出状态，慢等待每 5 秒提示一次；日志同时显示在终端并保存。完成条件仍来自有效状态和实际位置，不能把 stop 或“位置不变”单独当作闭合成功。

动作记录的 `gripper_result.completion` 保存命令前状态、目标 pulse、最近 200 次反馈、总采样数、耗时及放行／失败原因，失败和 Ctrl+C 记录也保留；限制内存中的采样历史，避免长时间等待不断占用内存。对应实验的 `.trace.json` 和 `.stdout.txt` 可用于确认是否确实等到闭合；实际反馈必须在 Alienware 真机上验证。

左右袖现在统一为“衣领朝上、下摆朝下时的图像左／右”。默认 remote 折叠链路在袖子定位前有一次 Claude 图像准备调用：Claude 自己选择旋转、裁剪或缩放，直接检查工具附带的结果图片；本地验证返回像素、来源及方向声明后，把同一张 RGB 交给 Molmo。Molmo 只提供这张图上的区域提示，本地映射回原始 Cam A，再把标注图交给 Claude 最终判断。固定相机显示旋转不再被当成衣服已经摆正。原图已摆正时允许直接选原图，无需强制重复旋转。

方向准备、视觉规划、动作提案各自最多尝试 6 次新编辑、16 轮模型调用。状态监督和执行后评价（含抓取探测评价）各自最多 2 次新编辑、8 轮模型调用。旋转／裁剪／缩放共用额度，参数错误也计数。每次工具响应和 Viser 都显示余额；用尽后只可读取、查询和选择已有图，不能继续编辑。方向准备没有合适结果则返回 UNCERTAIN 并停止本轮，不允许 unattended 自动重开 Claude 刷新额度。其他阶段遵循各自 schema，不能捏造动作。模型轮数通过 CLI `--max-turns` 限制，防止编辑额度耗尽后仍无限读图；这不是工具调用次数或固定秒数，原有超时仍生效。没有合法结果就不执行动作。

方向准备的 `UNCERTAIN` 必须附带 `failure_reason`：`IMAGE_UNAVAILABLE` 表示没看到图片内容，`TOOL_ERROR` 表示实际工具调用失败，`VISUAL_AMBIGUITY` 表示图片可见但衣领／下摆或方向不明确。READY 必须使用 null。`selection.json` 保存分类和来源，区分模型自述与本地内容校验。缺少图片时停止编辑，不再尝试用更多缩放、裁剪修复传递故障。

旧的“Read 已完成，所以要求 Claude 纠正”的 hook 已停用：调用完成不足以证明图片内容有效。历史 `orientation_correction` 参数目前仅注册分类审计，不拒绝 StructuredOutput、不要求改口；`same_session_correction_limit=0`，无自动补救调用或额度重置。即使 CLI 内容检查通过，模型仍报告不可见，也保留这个差异并停止交接，不强制 READY。

工具响应附带 `inspection_history`，列出已有图片的 ID、路径、父图、操作参数与经内容校验的图片返回次数；Claude 可用 `list_images` 查询完整目录。远端 job 内保存 `inspection_history.json` 快照和 `image_tool_calls.jsonl` 日志。相同源 image_id、相同操作和参数会返回已有图片及 `reused=true`，不新增文件、不消耗新编辑额度，但仍计入工具调用次数。MCP 重启从日志恢复图片目录、编辑缓存和调用计数。历史只属于当前 job，不自动跨阶段共享；本地 `claude_image_tools/<stage>/events.jsonl`、`image_debug.json`、`image_delivery.jsonl` 和 `images/` 保留操作、内容验证与经像素校验的重建图，远端目录按原流程清理。历史计数也不等于模型理解。

Viser 的 `Claude image operations` 显示操作过程及每张图的内容验证状态；iteration 摘要显示 `Claude → Molmo → Claude` 交接状态、实际输入路径／哈希和三个坐标系的点。`claude_molmo_orientation/molmo_input/camera_0_A.png` 是 Molmo 实际输入，`molmo_handoff.json` 保存提示词，`molmo_sleeve_locator/pixel_mapping.json` 保存映射。方向不明确、缺少可验证的返回图片、哈希／来源不合法或调用失败会停止交接，不会退回旧图。其他本地安全检查继续执行。完整文件说明见 [fold_garment_frame.md](fold_garment_frame.md)。

远端仅收到白名单中的 RGB（包括 RGB 上的 Rxxx 标注、RGB 视频接触图）和筛选后的语义任务/历史。不会发送深度图、热图、XYZ、标定矩阵、工作区数值、完整机器人状态或本地文件清单。上传 PNG 原字节，远端下载后先验证 SHA256。

远端先选择现有 Rxxx，再显式提出完整动作序列。move 使用选定 grasp 或当前 upright RGB 的 pixel，加相对抓取高度与相对 Home yaw；这些是运动提议，不是远端测量的坐标。本地将 upright pixel 逆旋转回原图，查询保存的深度/XYZ；抓取高度仍由原 `resolve_grasp_height` 决定。不会补默认动作、猜缺失深度或把完整 fold 自动改成 probe。该远程接口要求 closure 精确对应选定 Rxxx，不使用旧本地生成器可选的偏移抓取。

原有 Rxxx 可执行性、任务模式、工作区、IK 和执行校验继续生效。网络/Claude/schema/grounding 失败不返回 Proposal；重试耗尽后远程 supervisor/evaluator 不使用旧的默认状态 fallback。每个远端 job 使用 UUID 临时目录、远端 EXIT trap 和本地 finally 的有界尽力清理。公网中转服务的文件按一小时有效期处理，删除远端目录不会删除公网副本。

## 离线回归

## Claude 对话与传输计时

远端调用使用 `--output-format stream-json --verbose --include-partial-messages`，持续接收 CLI 公开消息和正文增量；只把终止 `result` 信封交给原 planner parser，中间 assistant 消息或正文片段不能替代最终方案。没有最终结果或最终结果为错误时，不执行动作。

终端中 `claude_text` 约每秒显示一次新增正文的末尾摘要（最多 160 字符，标明新增字符数及是否截取）；没有正文时不编造“思考内容”。`claude_stream` 每 5 秒显示累计原始事件数、实际类型／子类型、模型发出的工具调用数、距上条事件的秒数。没有新事件时也显示等待状态。工具请求／返回、错误、重试通知和最终结果即时显示。`raw_events=1500` 只是事件数，不能当成模型调用轮数；真实 `num_turns` 从最终结果读取。

每条原始消息仍逐条追加到 `stdout.log` 和 `claude_events.jsonl`。高频系统通知和增量片段不再各占一个对话标题；`claude_transcript.md` 保留正文摘要、完整 assistant 消息、工具结果、重要通知和状态汇总。`image_debug.json` 常规更新最多每秒一次，最终结果、退出、超时和中断时强制保存最终状态。图片返回校验即时更新内存，写盘节流不改变 Molmo 的放行条件。

在 Viser 的 `Claude image operations / <stage>` 中可展开以下文件。磁盘位置为本轮 `iteration_*/claude_image_tools/<stage>_<id>/`：

- `prompt.txt`、`system_prompt.txt`：应用实际发送的完整文本；`request.json` 包含图片清单、schema、远端路径和命令。原始输入图片与处理图保存在 `images/`。
- `claude_transcript.md`：Claude 公开文字、接口提供的 reasoning（如果有）、工具输入与返回结果。图片块显示尺寸、字节数和哈希摘要，完整内容仍保存在原始消息流。隐藏或被删减的内部推理不可获取，不会补写或猜测。
- `image_delivery.jsonl`：每次 CLI tool_result 的图片内容验证，区分空返回、损坏图片与有效图片；`image_debug.json` 关联 tool_use_id、视图、hook 和 stream 证据。
- `claude_events.jsonl`：每条 CLI 消息、接收时间、相对耗时和距上条消息的间隔；`stdout.log` 保留原始流。消息间隔包含网络、排队、工具和模型等待，不是纯推理时长，也不是完整的底层 API 请求日志。
- `timing.md` / `timing.json`：逐图字节数、Alienware HTTPS 上传、公司电脑 HTTPS 下载、SHA256 校验，以及 SSH、Claude、清理、总调用耗时。未知时长不填零；SSH 等父阶段包含子阶段，不能直接相加。
- `claude_result.json`：最终 CLI 返回，包含 CLI 实际提供的模型、token、轮次、API 耗时和费用统计。没有最终结果（如超时）时此文件不存在，之前的逐条记录仍保留。

Viser 显示原始事件数（不是模型轮数）、事件子类型、工具调用数和最后消息时间；详细输入、对话和计时均可展开。跨机器耗时使用各侧测量值，消息接收时间使用本地时钟；不能据此精确拆分服务端排队和模型计算。日志功能不增加模型调用，不要求模型额外生成解释。

## 单腕部 Cam A 抬升取证与最终评估

正式 FOLD/REPAIR_SLEEVE 轨迹的首次抓后动作必须垂直抬升至少 **30 mm**（相对夹爪闭合处的机器人 Z）。原计划抬升更高则保留，不足 30 mm 的正向垂直抬升提高到 30 mm；编译后的完整轨迹仍须通过工作区、预执行和 IK 校验。`grasp_capture_plan.json` 记录原计划是否被调整及拍照触发动作。没有额外插入 10 mm 停顿或在线视觉关卡。

闭合完成时，动作回调从 Cam A 录像缓存中保留一张已经取得的接触位姿帧，后台保存为 `hold_check/camera_A_grasp_before_lift.png`。帧时间必须晚于最后一次接触移动完成、不晚于闭合完成，且距闭合完成不超过 0.5 秒；它可能是在夹爪闭合过程中取得，不能声称是完全闭合后的照片。不等待新帧或 PNG 编码，也不暂停抬升。没有合适帧或未开启录像时，抬升前照片记录为 UNAVAILABLE，不使用已经开始抬升后的帧冒充。

至少 30 mm 的抬升动作完成后，动作回调只提交后台拍照任务，立即继续后续搬运、放下、松爪和回位，不等待取帧、图片编码或 Claude 判断。保存路径为 `hold_check/camera_A_grasp_after_lift.png`。`grasp_snapshots.json` 记录两张照片的状态、帧时间和动作边界。开启录像时复用 Cam A 录像流的新帧（最多等 3 秒）；`--no-video` 时抬升后照片仍在后台单独打开配置中的 A 相机拍 RGB。后台任务串行使用相机，动作结束后收齐结果并关闭设备，再进入最终评估。模拟模式不打开相机。可选 Cam C 同样在后台拍照。

视频评价使用各相机 RGB 视频，旧的空 AB 合成文件不会遮蔽有效 A 视频。单 A 录制不创建 AB 合成视频；归档历史录像时，AB 加标签失败会改试 A RGB，并在归档记录中保留原始错误。有动作时间戳时，最多 16 张视频取证帧优先包含闭合与首次抬升的开始、中间、结束，再补充全程均匀采样；没有时间戳则保持均匀采样。评价同时对比本轮抬升前后两张照片，考虑腕部相机自身运动，寻找布料相对背景的位移、拉紧或形变；照片相似本身不能证明成功或空抓。

照片通过 RGB 白名单与 before/after、视频关键帧一起送入最后的 evaluation，判断抓取、滑脱及折叠结果。腕部视角随机械臂运动，异步取得的图片可能已经包含搬运动作，不能标成“运输前静止图”；画面位移或夹爪闭合不等于抓取成功。已到释放完成之后的迟到帧标记为 `MISSED_WINDOW`，不当作抬升抓取证据上传；拍照失败记录 `FAILED`，均不改变已校验轨迹的继续执行。缺失、遮挡或不明确的证据要求评价保持 UNKNOWN。原有夹爪硬件反馈等待和本地运动校验保留。

ACQUISITION_PROBE 仍按模型明确给出的可逆探测轨迹执行；首次抓后抬升至少 30 mm，取消原先 15–30 mm 范围中的固定上限，40 mm 等更高抬升不会仅因超过 30 mm 被拒绝。所有位姿仍须通过工作区、预执行和 IK 校验，仍禁止抓后横向搬运。初次规划及纠错请求均包含同一抬升规则，主机不截断或重写探测高度。抬升完成后异步取证并继续反转、释放和回位。旧 `grasp_checkpoint` 视觉判断辅助接口留作兼容，正式 fold 流程不再调用它，不产生中途 `grasp_decision.json`。

## Overnight 经验与恢复

`bash scripts/start_fold_exploration.sh --viser --real --confirm-real` 默认启用 unattended。`--no-unattended` 可恢复遇错停止的方式。所有动作仍需原有实机确认、工作区、IK 和夹爪硬件反馈。

- 空抓/无法判断：在最终 evaluation 中评估，保存 `evaluation.json`、`failure_detection.json`、`record.json`、evidence 索引和 `workspace/fold_experience/experiences.jsonl`。阶段切换还保存 `partial_record.json`，便于异常退出后定位进度。执行、释放及 Home 必须确认完成才能进入下一轮；未成功的折叠不推进步骤。中途照片不再触发关卡中止或关卡重试，下一轮沿用现有评估反馈和探测预算。
- 执行后评估异常：unattended 模式先用同一批已存图片重试一次，不重放机械臂动作。仍失败则保存部分执行经验；只有释放/Home 已确认才允许重新观察。
- 规划失败：保存失败和纠正要求供下一轮使用；默认最多一次轨迹纠正。连续三轮规划失败或三次外围非物理阶段失败后停止并保留诊断。硬件异常、未确认回位、人工 Ctrl+C 不自动重启。watchdog 尊重主流程的 `restart_safe=false`，不以最后一条普通日志覆盖执行状态。
- 评估中的 `skill_update` 进入 run-local 候选；当前 run 的候选指导会进入下一轮。结束时由原 skill 审核/汇总机制处理持久化。已验证的规划纠正也会形成候选。确认空抓可记录失败检测 skill；黑图、遮挡、API 故障不能学习成空抓。
- 远端规划和评估现在收到 skill 指导正文，而不只是名称。历史展开任务经验只可用于可迁移的抓取/失败诊断，不能覆盖当前折叠目标。
- 默认从最近一个 run 继承最多 64 条经验作为历史教训，**不继承已完成折叠步骤或衣服物理状态**。`--no-inherit-experience` 可禁用；`--experience-dir` 显式延续同一实验状态，watchdog 沿用此模式。
- 抓取探测动作逐段保存 Cam A 图片到 `lift_checkpoints/`，包括回退阶段，送给执行后评价；正式折叠继续使用闭爪和微抬升的在线关卡。无需额外相机。
- 每轮录像加入带轮次/动作标记的 32 倍累计 `combined_rollout.mp4`。默认保留原片；`--prune-evaluated-video` 可在评估和归档成功后清理正常完成轮次的视频/原生录像，失败轮次保留原片。归档失败不会阻止经验保存，也不删除原片。

Viser 顶部用红色保留最近错误，并显示已恢复或已停止状态；每轮显示失败类别、释放/Home 确认和经验保存情况。终端支持颜色时同样显示红色错误；磁盘日志保持纯文本。仍只展示最近两个 iteration，不删除图片证据。

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
