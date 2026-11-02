# Intention–Outcome Mismatch Driven Skill Discovery

## 1. 研究目标

### 1.1 核心问题

在真实柔性物体操作中，Agent 对物体结构及动作效果的预测往往不准确。

这种不准确不仅可能导致操作失败，也可能使 Agent 在执行某个动作时，**意外产生原本没有预测到的有益效果**。

我们希望利用这些现象，使 Agent：

1. 识别原始意图与实际结果之间的差异。
2. 发现原本没有意识到的有益操作效果。
3. 将意外效果重新定义为可以主动追求的目标。
4. 通过物理实验归纳能够重复产生该效果的操作规律。
5. 将规律转化为可执行技能，并利用后续真实反馈不断改进。

### 1.2 当前任务

当前研究对象为真实 T 恤操作。

我们希望发现能够使被遮挡的语义端点重新显露的操作技能。

定义目标集合：

\[
\mathcal G=
\{
\operatorname{Reveal}(LS),
\operatorname{Reveal}(RS),
\operatorname{Reveal}(LH),
\operatorname{Reveal}(RH)
\}
\]

其中：

- LS：穿着者左袖口目标点。
- RS：穿着者右袖口目标点。
- LH：穿着者左下摆角。
- RH：穿着者右下摆角。

目标集合由研究者预先定义。

**研究者规定需要获得什么能力，但不规定应该通过什么动作获得该能力。**

特别地：

- 不要求四角同时显露。
- 不要求其他角点的可见性单调增加。
- 不以衣物面积增长作为技能成功的必要条件。
- 不预先规定抓取侧、牵拉方向或几何关系。

---

# 2. 算法总体结构

整个算法分为六个阶段：

\[
\boxed{
\begin{aligned}
&\text{1. Experience Collection}\\
&\downarrow\\
&\text{2. Intention–Outcome Comparison}\\
&\downarrow\\
&\text{3. Unexpected Benefit Discovery}\\
&\downarrow\\
&\text{4. Hindsight Goal Relabeling}\\
&\downarrow\\
&\text{5. Skill Hypothesis Induction}\\
&\downarrow\\
&\text{6. Active Validation \& Skill Refinement}
\end{aligned}
}
\]

其中：

**Stage 2–3 是整个算法的核心发现机制。**

Stage 4–6 负责将发现转化为能够主动使用的技能。

算法不要求每次 Rollout 都产生新的 Skill。

如果某次操作不存在与目标相关的意外收益，则可以正常结束本轮发现过程。

---

# 3. Stage 1：Experience Collection

## 3.1 目标

记录 Agent 在真实环境中的原始意图、执行动作以及实际结果。

每次操作保存：

\[
e_t=(o_t,i_t,a_t,o_{t+1})
\]

其中：

| 变量 | 含义 |
|---|---|
| \(o_t\) | 操作前 RGB 观测 |
| \(i_t\) | 原始操作意图及预期效果 |
| \(a_t\) | 实际执行的机器人动作 |
| \(o_{t+1}\) | 操作后 RGB 观测 |

必要时保存执行过程帧及机器人执行记录。

## 3.2 原始意图的记录

这是算法的重要前提。

在动作执行之前，必须记录 Agent 对本次操作的理解。

包括：

**Intended Object**

Agent 认为自己正在操作什么？

**Intended Effect**

Agent 希望该动作改变什么？

**Expected Outcome**

Agent 预测操作后哪些部位会发生什么变化？

如果历史记录只有原始动作理由，没有明确预测，则只能从该理由中提取受到支持的意图，不能事后补写预测。

## 3.3 重要约束

原始意图必须在观察实际结果之前保存。

\[
i_t\perp o_{t+1}
\]

这里表示意图记录不能利用未来观测的信息，而不是要求两者在统计上独立。

后续分析不允许根据实际结果重新解释原始意图。

例如，Agent 原本认为自己抓住袖口，就必须保留这一原始判断，即使后来发现它可能抓住的是衣身。

**Agent 的原始语义判断属于历史认知，不自动构成物理事实。**

---

# 4. Stage 2：Intention–Outcome Comparison

## 4.1 目标

分别分析 Agent 原本希望发生什么，以及真实操作实际造成了什么。

这两个判断必须独立完成。

### A. Intended Effects

从操作前保存的原始意图中提取预期效果集合：

\[
\hat{\mathcal E}_t=
\operatorname{ParseIntention}(i_t)
\]

例如：

- 预期右侧袖部展开。
- 预期下摆向外移动。
- 预期胸前折叠减少。

### B. Observed Effects

根据真实 Before/After 观测，提取实际效果：

\[
\mathcal E_t=
\operatorname{ObserveEffects}(o_t,o_{t+1})
\]

关键要求是：

**不能只检查原本想操作的区域，必须观察整件衣服发生了什么变化。**

例如：

- 原目标区域是否发生改变？
- 其他袖部是否显露？
- 下摆是否出现新的可见端点？
- 是否有布料重新被遮挡？
- 衣身是否发生整体平移或旋转？
- 是否发生局部相对形变？

必要时使用图像配准、分割及局部特征比较。

这些工具用于描述观察，不直接推断隐藏层序。

## 4.2 结果对比

建立：

\[
M_t=
\operatorname{Compare}
(\hat{\mathcal E}_t,\mathcal E_t)
\]

比较结果至少包含：

| 类型 | 含义 |
|---|---|
| EXPECTED_OBSERVED | 预期变化实际出现 |
| EXPECTED_MISSING | 预期变化未得到确认 |
| UNEXPECTED_OBSERVED | 出现原始意图没有预期的变化 |
| INDETERMINATE | 原始意图或实际结果不足以比较 |

其中，UNEXPECTED_OBSERVED 不自动代表有益。

例如，牵拉袖子时意外遮挡了下摆，也属于预期之外的变化，但不构成我们需要学习的收益。

同时，原始意图没有描述某个部位，并不必然证明 Agent 明确预测该部位不会变化。证据不足时应保留不确定性。

---

# 5. Stage 3：Unexpected Benefit Discovery

## 5.1 目标

从意图与实际结果的差异中，找到值得进一步学习的有益效果。

定义：

\[
\boxed{
\mathcal U_t=
\left(
\mathcal E_t\setminus\hat{\mathcal E}_t
\right)
\cap\mathcal G
}
\]

其中：

- \(\mathcal E_t\)：真实观察到的效果。
- \(\hat{\mathcal E}_t\)：原本预期的效果。
- \(\mathcal G\)：当前任务关心的目标效果。

集合差异需要通过具体语义部位、变化类型和原始意图进行判断，不能仅通过文本匹配实现。

只有当某个**原本未被预期、但与 Corner Reveal 相关的有益效果**出现时，才触发 Hindsight Discovery。

## 5.2 Corner Reveal 事件判定

针对某个语义端点 \(c_i\)，检查：

\[
\operatorname{Reveal}(c_i):
\quad
\text{Occluded Before}
\rightarrow
\text{Exposed After}
\]

判定分为：

**CONFIRMED**

有充分证据支持目标语义端点从自遮挡状态重新显露。

**PROVISIONAL**

出现了相应语义区域、端边或连接结构显露的较强线索，但材料身份或遮挡转换尚未得到充分确认。

**UNCERTAIN**

无法判断是否发生真实 Corner Reveal。

**NO_REVEAL**

没有支持目标端点由遮挡到显露的证据，或证据表明端点原本已经可见。

特别注意：

- 普通轮廓凸点不等于语义端点。
- 袖状布片出现不自动证明袖口端点恢复。
- 整体移动、端点重新进入视野不等于解除自遮挡。
- 其他角点暂时被隐藏，不自动否定当前目标端点的显露。

## 5.3 发现结果

如果存在值得研究的意外收益：

\[
\mathcal U_t\neq\varnothing
\]

则进入 Stage 4。

如果没有：

记录本次意图与结果的比较，结束这一条 Hindsight Discovery 分支。

如果只有暂定证据，可以进入候选假设分析，但不能将其当作已确认的技能成功。

**不能为了完成流程而把普通布片展开、缝边变长或标签显露重新定义为 Corner Reveal。**

---

# 6. Stage 4：Hindsight Goal Relabeling

## 6.1 核心思想

将 Agent 原本没有预期到、却实际产生的有益效果，重新定义为一个可以主动追求的目标。

原始经验：

\[
(o_t,i_t,a_t,o_{t+1})
\]

经过目标重标记：

\[
\boxed{
(o_t,g'_t,a_t,o_{t+1})
}
\]

其中：

\[
g'_t=\operatorname{Reveal}(c_i)
\]

这里的 \(c_i\) 是意外显露的目标端点。

## 6.2 关键问题

完成目标重标记后，Agent 必须回答：

> 如果我在执行动作之前，就希望实现这个意外效果，我应该怎样选择动作？

这是从被动经验分析进入主动技能发现的转折点。

此前的问题是：

**为什么这个动作产生了这样的结果？**

现在的问题是：

**在新的衣物状态下，我怎样有意识地再次产生这种结果？**

## 6.3 输出

形成一条新的技能学习经验：

\[
d_t=(o_t,g'_t,a_t,\mathcal E_t)
\]

并保留：

- 原始意图。
- 实际效果。
- 目标重标记的理由。
- 显露证据及不确定性。
- 对原始操作的完整溯源。

---

# 7. Stage 5：Skill Hypothesis Induction

## 7.1 目标

从已经发现的有益效果中，归纳能够预测并主动产生该效果的操作规律。

这一阶段不应仅仅总结历史抓点或运动方向。

而是需要研究：

\[
\boxed{
\text{Observable Geometry}
\longrightarrow
\text{Action}
\longrightarrow
\text{Target Effect}
}
\]

## 7.2 机制假设

Agent 首先尝试解释：

**为什么操作一个可见部位，会使目标端点显露？**

要求至少考虑两个能够被实际实验区分的解释。

可能涉及布料连接、遮挡变化、局部展开或重新落布等机制，但不预设哪一种成立。

所有解释必须区分：

- 图像直接支持的事实。
- 基于事实提出的物理假设。
- 尚需验证的因果关系。

## 7.3 动作规律归纳

然后尝试寻找：

\[
a=f(o,g)
\]

其中：

- \(o\)：当前观测。
- \(g\)：需要显露的语义端点。
- \(a\)：操作动作。

模型应当自主研究当前图像中哪些可观测性质能够决定：

1. 应该操作哪块可见布料？
2. 在哪里抓取？
3. 沿什么方向牵拉？
4. 预计会对目标端点产生什么变化？

允许使用几何算法、视觉模型、物理假设和历史经验。

但是：

**不得预先规定具体抓取规则，也不得直接复用历史动作坐标。**

## 7.4 生成候选 Skill

尝试实现：

`generate_action(rgb, target_corner) -> action`

候选 Skill 应当具有明确的：

- 输入条件。
- 动作生成过程。
- 适用范围。
- 预期效果。
- 可证伪条件。

如果暂时无法找到唯一的操作规律，可以保留多个竞争假设。

不能因为已经找到一次有效动作，就宣称归纳出了通用技能。

## 7.5 输出

本阶段输出：

**Skill Hypothesis**

一种能够解释历史收益、并预测新状态中潜在有效动作的假设。

**Candidate Skill**

根据当前观测生成实验动作的可执行策略。

两者都不自动代表经过验证的技能。

---

# 8. Stage 6：Active Validation & Skill Refinement

## 8.1 目标

让 Agent 在新的真实衣物状态下，主动检验自己归纳的操作规律。

这是区别于普通历史经验总结的关键步骤。

## 8.2 实验设计

对于候选 Skill，Agent 自主选择一个具有信息价值的验证实验。

在动作执行前冻结：

\[
\hat{\mathcal E}_{t+1}
=
\operatorname{Predict}(o_t,a_t)
\]

明确记录：

- 当前希望显露哪个端点。
- 选择什么抓点和方向。
- 为什么选择这个动作。
- 预计目标区域产生什么变化。
- 什么结果能够支持或反驳当前假设。

## 8.3 在线执行

使用已有真实机器人系统执行实验。

复用现有：

- 视觉观测与坐标标定。
- 抓取和牵拉接口。
- 运动规划。
- 碰撞与工作空间检查。
- 执行状态和反馈机制。

沿用已经验证的夹爪 Z=0 接触配置，不重复要求额外接触高度实验。

不能修改或绕过已有安全机制。

每轮首先执行一个通过现有安全流程的动作，再依据真实结果决定后续实验。

## 8.4 实际结果评价

比较：

\[
\hat{\mathcal E}_{t+1}
\quad\text{与}\quad
\mathcal E_{t+1}
\]

重点判断：

**目标端点是否发生了事先预测的显露？**

同时记录其他意外变化。

这里再次回到 Stage 2 的核心：

**预测效果与实际效果比较。**

因此，验证实验本身也可能产生新的 Hindsight Discovery 事件。

## 8.5 技能更新

根据真实反馈，Agent 可以：

- 保留当前假设。
- 修改适用条件。
- 修正动作生成规则。
- 提出新的竞争解释。
- 放弃被证据反驳的假设。
- 对新出现的意外收益启动另一条发现分支。

只有形成了可执行规则，并在独立状态上获得实际支持，才能逐步将候选 Skill 提升为已验证技能。

一次成功仅提供初步支持，不直接证明泛化。

---

# 9. 在线运行方式

历史离线数据可以用于初始化发现算法。

但最终系统采用在线增量运行方式。

### 9.1 普通操作循环

\[
o_t
\rightarrow i_t
\rightarrow a_t
\rightarrow o_{t+1}
\]

机器人继续完成原本的衣物操作任务。

每一步自动记录原始意图、实际执行和真实结果。

### 9.2 Rollout 后的发现循环

每个 Rollout 结束后：

1. 遍历本次实际操作记录。
2. 独立提取各步的原始意图和实际效果。
3. 检测意图与结果之间的差异。
4. 筛选与 Corner Reveal 有关的意外收益。
5. 对有价值的事件启动 Hindsight 分析。
6. 归纳候选操作规律。
7. 将候选 Skill 和验证计划保存，供后续在线实验检验。

不需要每次重新扫描全部历史数据。

不需要每次都完整修改 Skill。

### 9.3 增量改进

当新的真实反馈出现时，Agent 根据已有证据修正技能假设。

最终形成：

\[
\boxed{
\text{Rollout}
\rightarrow
\text{Mismatch Discovery}
\rightarrow
\text{Skill Hypothesis}
\rightarrow
\text{Active Experiment}
\rightarrow
\text{Updated Skill}
}
\]

后续实验中的新意外收益，也可以重新进入这一循环。

---

# 10. 算法约束

## 10.1 允许的研究者引导

研究者可以预先规定：

- 任务目标：Corner Reveal。
- 四个语义端点的定义。
- 意图与结果的比较流程。
- 什么证据能够支持目标显露。
- 如何验证假设。
- 机器人允许的操作接口。

这些属于算法与任务定义。

## 10.2 不允许的答案引导

研究者不能直接向负责技能发现的 Agent 提供：

- 预期的正确抓取侧。
- 预期的正确牵拉方向。
- 预先设计的几何抓点规律。
- 希望 Agent 最终发现的特定操作技能。
- 某个历史案例为什么成功的人工结论。

具体操作规律必须由 Agent 从证据中提出，并接受真实物理验证。

## 10.3 不允许目标漂移

在 Corner Reveal 任务下：

- 面积增长不能替代端点显露。
- 标签出现不能替代语义角点显露。
- 普通折叠展开不能直接替代 Corner Reveal。
- 意外效果的存在不自动意味着它具有技能学习价值。

允许将其他现象保留为辅助观测或待研究假设，但不能未经证据支持就更改本轮任务目标。

---

# 11. 当前算法的成功标准

整套算法需要分别评价三个层次。

### Level 1：Discovery

Agent 是否自主发现了原始意图之外、与 Corner Reveal 相关的有益效果？

### Level 2：Induction

Agent 是否从这些效果中提出了新的、具有可检验预测的操作规律？

### Level 3：Acquisition

Agent 是否能够利用归纳出的规律，在新衣物状态中主动实现目标，并通过后续反馈形成可复用的执行策略？

三个层次分别报告。

不能因为生成了 Skill 代码，就宣称技能已经学会。

---

# 12. 核心算法概括

整个算法可以压缩为：

\[
\boxed{
\begin{aligned}
&\textbf{Intend}\\
&\downarrow\\
&\textbf{Act}\\
&\downarrow\\
&\textbf{Observe}\\
&\downarrow\\
&\textbf{Compare Intention \& Outcome}\\
&\downarrow\\
&\textbf{Discover Unexpected Benefits}\\
&\downarrow\\
&\textbf{Relabel Goals}\\
&\downarrow\\
&\textbf{Induce Skill Hypotheses}\\
&\downarrow\\
&\textbf{Actively Validate}\\
&\downarrow\\
&\textbf{Refine Skills}
\end{aligned}
}
\]

**核心原则：**

Agent 不仅根据动作是否完成原始目标来学习，还应当观察动作实际产生的全部效果。

当某个原本没有预期的效果能够实现研究目标时，Agent 将它转化为新的学习机会，通过主动物理验证，尝试发现并掌握此前没有意识到的操作规律。
