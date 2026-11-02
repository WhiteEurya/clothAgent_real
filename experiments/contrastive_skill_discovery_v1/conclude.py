from run import *
def main():
 p=OUT/'report.md';report=p.read_text();stats=read(OUT/'planner_replay/summary.json')['statistics'];base=stats['Base'];c=stats['C']
 delta_t=c['mean_wall_s']-base['mean_wall_s'];delta_i=c['mean_input_tokens']-base['mean_input_tokens']
 conclusion=f'''Q1. **无经验 GPT 提出了什么？** A 提出“向外翻开内折衣片”：辨认自由角与折线，抓自由角内侧，越过折线向空桌面翻开。它没有历史证据，Development 后选择保留原规则。部署核心为97个汉字。

Q2. **只看动作有何不同？** B 提出“外展收拢袖口”：抓袖口外露边角，朝衣身外侧移动。它从历史选点归纳了部位与方向，但明确不把被执行动作当作已验证有效；一次修订后保留，核心82个汉字。

Q3. **真实结果是否带来新规律？** C 在本次三组输出中独有“分瓣收拢下摆的单角外展”：用两端间内凹收拢作为Trigger，抓可见端角。它以Train正向案例及E06无明确收益案例支持结构差异，再依据Development E23仅修改方向：取消必须增大两端间距，改为沿局部瓣根指向端角的方向外展。最终核心114个汉字。这里的“新”仅指相对本次A/B样本不同；每组只有一次归纳，不能证明该规则不在模型原有先验内，也不能证明结构—结果关系具有物理因果性。

Q4. **是否更好预测独立效果？** 有限的正例召回改善信号，但**没有验证更强的正负区分能力**。Development中A/B/C正例召回为2/6、6/6、4/6，低收益识别均2/2；C未胜过B。Test中变为1/7、1/7、4/7，但只有7个YES、1个UNKNOWN、0个NO_CLEAR_EVIDENCE，无法估计低收益识别能力。事后常数YES参照在这7个已知标签上也能得到100%召回/准确率，所以不能将C的4/7视为判别优势的充分证据。C的7个已知正例分别是4次正向预测、2次低收益预测、1次UNKNOWN弃权；不将弃权称为失败。C在E29/E32/E33明确触发且符合规则的三个状态上预测正向并与标签一致，跨越两个与Train隔离的运行组；这只是有限跨状态适用现象，缺少负例检验。

Q5. **是否真正改变Planner动作？** 输出坐标与方向确有差异，而且不只改变解释。B四次均抓模型自述的右袖口，C四次均抓下摆端角；两者均报告触发4/4，A为0/4（3次NO、1次UNKNOWN）。C相对同重复编号Base的平均抓点差124.6px、方向差44.4°；C组内两次重复的平均抓点差4.8px、方向差3.1°，而Base为224.1px、51.7°。但C有些动作接近Base已有采样，例如E29第二次抓点仅差约5.1px；因此更稳妥的解释是“输出集中于特定部位”的迹象，不能把全部差异归因于Skill或称为新动作能力。B相对Base平均抓点差301.7px，组内差1.9px，部位偏好变化更明显。所有16次Pa均落入复用的启发式衣物掩码，二维坐标检查通过；未执行的新动作一律不判物理成功。模型自述的衣物部位不是人工标注真值。

Q6. **在线开销是否增加？** 所有Replay仍为每次一个逻辑GPT请求，0次工具调用、无检索、无额外Research Agent。C平均输入Token由{base['mean_input_tokens']:.0f}增至{c['mean_input_tokens']:.0f}，增加{delta_i:.0f}（{delta_i/base['mean_input_tokens']*100:.1f}%）；平均输出Token由389增至673。C平均端到端推理耗时由{base['mean_wall_s']:.1f}秒增至{c['mean_wall_s']:.1f}秒，增加{delta_t:.1f}秒（{delta_t/base['mean_wall_s']*100:.1f}%）；中位数36.5秒对47.8秒。额外输入很小，但观测到的耗时增加并不小，不能声称没有明显延迟代价。每组仅4次、相同并发上限但无受控服务器负载，输出推理长度也不同；这些是本次观察值，不是稳定开销估计。

Q7. **限制更可能来自哪里？** 最直接证据是验证数据缺乏类别区分度：Train为14YES/4NO_CLEAR_EVIDENCE/2UNKNOWN，Development只有2个低收益例，Test没有低收益例；样本少且同组相关。其次是评价器对规则范围的处理不稳定：Development B在未触发时也能靠其他视觉理由预测收益，而Test多次在不符合规则时给出NO_CLEAR_EVIDENCE；A在Test有7/8个Trigger UNKNOWN，却没有任何效果预测UNKNOWN。部分输出混淆“该Skill无法支持收益判断”和“动作无明确收益”，使成绩同时受评价器先验及规则解释方式影响。E30的C Trigger在历史评价为UNKNOWN，Replay为YES，也说明解释存在波动。强模型先验是合理解释之一，但没有无Skill评价器对照或多次Teacher归纳，不能与归纳能力不足分开。反馈不是完全无信息——C给出了结构候选并作了证据支持的方向修订——但不能据此确认归纳成功或宣布归纳本身失败。

### 结论边界与协议偏差

- **Skill生成：完成。** 三条不同候选、各一次修订预算、最终冻结及版本Diff齐全。
- **经验预测改善：未充分证实。** C只有Test正例召回优势；Development未胜B，Test缺负例，评价器的范围/弃权语义有风险。保留全部原始预测，不事后重标以美化结果。
- **Planner行为改变：有可见输出及稳定性信号。** 两状态、两重复不足以把动作变化与所有随机波动分离，更不能判断哪条新动作更有效。
- **真实机器人任务改善：未评估。** 没有实机执行，没有成功率提升证据。

初始编排会话曾在候选冻结前误读Test标签，已在context_audit.json披露；正式Teacher/评价器使用清洁独立会话，未收到这些标签，但全流程不能宣称严格双盲。原始数据的效果标签来自同模型的独立视觉复核，一致性仅说明可重复性，不是独立物理真值。历史动作仍可能间接受人工基础Prompt影响，source_audit只核对所选13个运行来源的显式目标规则标志均为false，未读取Prompt语义。

历史评价使用完整七字段JSON（C初始expected_effect还带Train证据叙述），Replay仅使用四字段短规则；因此该实验不是纯粹的150字规则预测能力消融，预测结果不能全部归因于短规则本身。此范围差异已在Test前写入protocol，并对所有组采取相同字段提取，无人工改写Skill。C修订所用Development E23的结构来自本组盲评文字，未重新附Development图像；这是反馈分辨率的额外限制。

一次额度失败按用户retry指令仅重试原Development B，提示与附件不变；B/C的大图请求出现原生传输重连，日志完整保留。没有因成绩扩大样本，没有D负对照，没有Test后修订。所有Source/Skill/预测冻结和附件允许范围校验共421项通过。上述校验验证文件与边界一致性，不替代对模型推理正确性的验证。

**本轮未达到“真实反馈产生更强独立判别能力且无明显在线代价”的完整证据标准。** 它提供了可复现的候选归纳、有限正例召回信号与行为约束现象；主要研究问题仍未被这批Test充分检验。若后续另开实验，应优先预先冻结含足够明确低收益例的独立验证集，并使评价器明确区分“规则不适用/无法预测”与“预测低收益”；本轮没有为此重混数据或追加调用。
'''
 report=report.replace('待基于已冻结的实际输出补充逐项解释。',conclusion)
 lead='**结果：C生成了与先验/动作模仿不同的候选规则，Test正例召回4/7（A/B均1/7），但Test没有明确低收益例，无法验证判别优势。Planner输出更集中，C平均耗时增加约49%；未做实机验证。**\n\n'
 report=report.replace('# Contrastive Skill Discovery v1\n\n','# Contrastive Skill Discovery v1\n\n'+lead,1)
 p.write_text(report)
 save(OUT/'research_conclusions.json',{'skill_generation':'COMPLETE','predictive_discrimination_advantage':'NOT_ESTABLISHED','positive_recall_signal':{'A':'1/7','B':'1/7','C':'4/7'},'test_negative_count':0,'planner_behavior':'Different outputs and lower within-condition variation; limited stochastic control','physical_improvement':'NOT_EVALUATED','online_C_input_token_delta':delta_i,'online_C_mean_wall_delta_s':delta_t,'full_success_criterion':'NOT_ESTABLISHED'})
if __name__=='__main__':main()
