# 实验口径与限制

- 固定证据：fold_20261005T092836441022198Z / iteration_001 / image_preparation_30cbdea82a72_snapshot。
- 9 张图片逐个核对 SHA-256；不重拍、不重做图像编辑、不执行机器人、不修改正式 skill。
- 所有主对照显式请求 claude-opus-5，并保留远端实际 modelUsage。
- A/B 的 motion 接收本组新生成的 visual_plan，不能复用上次选点答案。
- A 是按需读取文件；B 提供相同可用内容的完整内联版本。两者实际读入的 token 和图像子集可能不同，这是交付方式改变后的行为结果。
- C 将选点和动作合并；为了避免两个 schema 相互冲突，联合输出必须包装为 visual / motion。任务内容保持，不能宣称提示词逐字相同。
- D 仅加明确的信息任务步骤；E 再加入从已有候选注册表与 affine 计算的所有候选坐标，不提供人工选择或旧答案。
- F 缩短说明文本，仍保留 schema 与校验；不是以牺牲字段完整性换速度。
- G 是直接图片输入与无探索工具的执行组合。现有 direct_images 后端同时启用 safe-mode，因此它是机制组合对照，不能把收益全归给某一个开关。
- C′ 重复 C，观察服务/agent/缓存波动；单次筛查不做显著性检验。
- 补充 H 使用同一模型与 C 相同流程，只显式指定 CLI --effort low；远端配置未显式设置 effortLevel，不能称对照是 high/max。
- 原始完整轮次包含前置拍照、编辑、机器人、评价和经验更新；新实验仅比较编辑结束后的 reasoning，不能把二者总时间直接相减作为提速。
- 离线质量检查：schema、eligible Rxx、来源引用链、图像范围、提拉/释放顺序；并用人工式近似区域判断目标袖子/袖口/胸前落点/向内方向。后者不是像素级真值。
- 不执行 controller IK、深度重新采集或真实机器人，因此物理精度未验证。
- 响应前等待含服务与输入处理；输出窗口含 thinking/文本/工具参数。不得据此推断隐含思考主题或 provider 纯算力时间。
- 总 token = input + output + cache-read + cache-creation。thinking 已在 output 中，不重复计算。费用为 CLI 声明值，未与账单核对。
- SSH 空请求 3 次只衡量连接与空命令，不代表模型 API 首字节或图像工具设置时间。

- 已观察到 C 的一次 system.api_retry：状态 524，发生在远端记录第 168.15 秒；重试延迟字段 509 ms。事件时刻不是失败请求的耗时，不直接扣除。
- 原始真实轮次的 8 次模型调用均未记录 api_retry；因此不能用 C 的服务错误解释原始 1521.7 秒。
- 基准 A 实际读取：selection 14 个数据文件（另读 manifest/tool_list），motion 16 个数据文件；两阶段均未读 experience_context，motion 未读 prepared_visual_evidence。I 使用这些实际 Read 文本作为补充控制，motion 的旧 visual_plan 替换为本组新答案。

- 在 C′ 结果产生前约定：若 C 有 API 重试而 C′ 没有，方法/Agent 对比优先引用 C′，同时完整保留 C 的原始耗时，不做失败时长扣除。

- 这是冻结规划接口输入的离线重放：保留已保存的 contact_height_contract，并替换本组新 visual_plan；不重新计算新候选的深度 grounding。因此 contract/区域检查不能当作完整生产编译与物理精度验证。
- D 的某个 message_delta 记录 output_tokens=5、iterations=[]，同时该轮响应窗口约177.9秒；终端usage可能存在不完整记录。报告保留CLI原数并标注，绝不把估算thinking token补加到正式总数。
