# 单次离线 RSI 策略代码迭代

复用现有 `cloth_harness` 的 Observation、分割、outcome_observer 和配准。
与已有连续探索入口不同，本入口没有硬件执行分支，没有自动下一轮。
原有实机路径 `experiments/rsi_exploration_01/run.py:prepare` 和
`one_shot_goal_full_episode_v1.prepare_action` 仅作为未来集成接口，未调用或修改。

已冻结输出放在 `reports/rsi_code_iteration_01_20261029/`。

流程：

1. `run.py prepare`：验证真实 dispatch/执行/图片来源，冻结现有 evaluator 文件哈希；准备 3 条种子、3 条回放池、2 条跨 run heldout。
2. `native.py 1`：独立 Codex 原生会话自行提出假设、为池内动作预测结果、选一个最有信息量的回放、写初始代码。不给池内 after、heldout 或旧 planner 理由。
3. `run.py reveal`：安装并测试 v0，保存预反馈输出，再揭示所选动作的固定反馈和真实 before/after。
4. `native.py 2`：只接受所选一条反馈，输出一次 v1 修改及下一实验建议。
5. `run.py evaluate`：以全新模块加载 v0/v1；测试新动作合法像素和可见抓点、历史动作面积变化预测、90度旋转及平移一致性。
6. `report.py`：生成结果、代码差异和本地 HTML。

Python 使用 `/home/sja/miniconda3/envs/cali/bin/python`；原生调用脚本仅需标准库。
不要重复执行 prepare 覆盖冻结数据。已完成记录不用于伪造新物理反馈。

策略接口：`policy({'rgb': uint8 RGB, 'mask': bool}) -> action`。
辅助接口：`predict_delta(observation, recorded_action) -> float | None`。
模型只输出代码，禁用工具，不能编辑 evaluator。代码版本以 sha256 和原始模型响应绑定；执行前限制导入和文件/设备访问。该静态检查不是通用安全沙箱。

评价数值仅为固定分割下的可见面积代理。配准或区域对应未知时保持 UNKNOWN。
历史动作预测误差与新动作物理效果不同；生成动作只有像素检查，未验证深度、工作空间、碰撞或 IK。
变换测试只检测几何行为，不证明旋转衣物后的物理成功。
如进入后续实机阶段，仍须通过既有安全检查、场景绑定和当前会话授权；本入口不借用其他实验的自动执行授权。

## 本次结果

闭环已完成，策略未得到改善：v0 在 8 个原始状态中输出 5 个动作，v1 全部弃权。
P2 回放的可见面积下降 14.73%，超出两种预报区间；配准和局部对应不足以确认物理机制。
两个未见状态均无数值预测。变换后仍弃权不是动作泛化证据。
完整图片、预报、固定反馈、代码差异和审计见报告 `review.html`。

保留原生调用日志中的异常：第二阶段模型报告额外的 `-` 图片读取错误，同时确认收到所需的两张图。
后续调用构造已加参数分隔符；未重跑学习或改写原始调用记录。
原生入口拒绝重复调用已记录的阶段，避免覆盖本次失败结果。
