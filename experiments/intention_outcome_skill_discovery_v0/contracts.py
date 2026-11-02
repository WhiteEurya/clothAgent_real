"""Task definitions, not learned actions. No expert prompt or historical findings."""
S={'type':'string'}
B={'type':'boolean'}
def en(*x):return {'type':'string','enum':list(x)}
def obj(x):return {'type':'object','properties':x,'required':list(x),'additionalProperties':False}
def arr(x):return {'type':'array','items':x}
GOALS=['Reveal(LS)','Reveal(RS)','Reveal(LH)','Reveal(RH)']
POINT={'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':2}
SCORE={'type':'integer','minimum':0,'maximum':3}
INT_ITEM=obj({'id':S,'intention_available':B,'intended_object':S,'intended_effects':arr(obj({'quote':S,'effect':S,'reference_frame':en('IMAGE','WEARER','UNSPECIFIED'),'region_in_before':S,'semantic_goal':{'type':['string','null'],'enum':GOALS+[None]},'semantic_mapping_evidence':S,'mapping_confidence':en('SUPPORTED','PROVISIONAL','UNKNOWN')})),'explicit_expected_outcome':S,'not_inferable':S,'side_ambiguity':S})
INT_SCHEMA=obj({'records':arr(INT_ITEM)})
CORNER=obj({'goal':en(*GOALS),'classification':en('CONFIRMED','PROVISIONAL','UNCERTAIN','NO_REVEAL'),'before_state':en('OCCLUDED','SUSPECTED_OCCLUDED','VISIBLE','OUT_OF_VIEW','UNKNOWN'),'after_state':en('EXPOSED','PARTIALLY_EXPOSED','NOT_EXPOSED','OUT_OF_VIEW','UNKNOWN'),'semantic_identity_basis':S,'wearer_side_basis':S,'before_image_region':S,'after_image_region':S,'before_evidence':S,'after_evidence':S,'material_correspondence':S,'alternative_explanation':S,'evidence_score':SCORE})
OBS_SCHEMA=obj({'id':S,'whole_garment_changes':arr(S),'global_motion':S,'relative_deformation':S,'new_occlusions':arr(S),'view_change_confound':S,'corners':arr(CORNER),'uncertainty':S})
CMP_ITEM=obj({'id':S,'expected_effect_comparisons':arr(obj({'intended_effect':S,'classification':en('EXPECTED_OBSERVED','EXPECTED_MISSING','INDETERMINATE'),'evidence':S})),'corner_comparisons':arr(obj({'goal':en(*GOALS),'classification':en('EXPECTED_OBSERVED','EXPECTED_MISSING','UNEXPECTED_OBSERVED','INDETERMINATE'),'unexpected_evidence':en('SUPPORTED','PROVISIONAL','NONE'),'side_mapping':S,'evidence_for_mismatch':S,'benefit_related_to_goal':B,'absence_of_mention_only':B,'uncertainty':S})),'other_unexpected_effects':arr(S)})
CMP_SCHEMA=obj({'records':arr(CMP_ITEM)})
def hypothesis_schema(goal):
 return obj({'status':en('EXPERIMENTAL_HYPOTHESIS','INSUFFICIENT_EVIDENCE'),'goal':en(goal),'relabel_reason':S,'intended_actual_observed':S,'success_criterion':S,'hypotheses':arr(obj({'name':S,'observed_facts':S,'mechanism':S,'unobserved_assumptions':S,'distinguishing_prediction':S,'falsifier':S})),'observable_geometry':arr(S),'action_generation_rules':arr(S),'applicability':S,'limitations':S,'next_informative_experiment':S})
def action_schema(goal):
 return obj({'decision':en('ACTION','ABSTAIN'),'goal':en(goal),'target_state':en('OCCLUDED','SUSPECTED_OCCLUDED','VISIBLE','OUT_OF_VIEW','UNKNOWN'),'target_identity_basis':S,'target_occlusion_evidence':S,'observable_geometry':S,'grasp_region':S,'grasp_px':POINT,'target_px':POINT,'grasp_angle_deg':{'type':'integer','minimum':0,'maximum':180},'direction_reason':S,'expected_target_change':S,'success_criterion':S,'falsifier':S,'uncertainty':S,'abstain_reason':S,'experimental_only':B})
def synthesis_schema(goal):
 return obj({'status':en('EXPERIMENTAL_SKILL','HYPOTHESIS_ONLY'),'goal':en(goal),'python_code':S,'inference_prompt':S,'algorithm_explanation':S,'limitations':S})
INT_PROMPT='''独立提取历史规划者在操作前的认知。只给Before和原始决策文字，不给After、效果分析或旧研究结论。逐条提取Intended Object、Intended Effect、Expected Outcome，引用原文。只保存文字明确支持的预期；缺失标UNKNOWN，不能补写。Before与Pa仅帮助指示原话指向的画面区域，不能替换规划者的原始材料判断。原规划说某处是袖口，即使你怀疑也必须保留为原认知，不当物理真值。
严格区分画面左右和穿着者左右。“右侧”等含糊文字不能默认是穿着者右侧。记录reference_frame及Before中的可见区域；语义端点映射不清就semantic_goal=null、mapping_confidence=UNKNOWN。不能仅因为同一个“右”字就映射为Reveal(RS)。也不能预设它一定是画面右侧，必须说明原话、Pa或图像支持什么。四端部仅用LS/RS左右袖口边缘中点、LH/RH真实下摆交角定义，不诱导操作规则。逐条输出全部指定ID，用中文。'''
OBS_PROMPT='''这是独立的真实效果观察，不提供原始意图或期望结论。第一张Before、第二张After，完整观察整件T恤，不只看某个目标。描述整体平移/旋转、局部相对形变、显露、新遮挡及视角混淆。不要解释物理机制，不提出动作规则，不猜原始意图。分别审查四个语义目标Reveal(LS)、Reveal(RS)、Reveal(LH)、Reveal(RH)，每个必须恰好出现一次。
LS/RS是穿着者左/右袖口开口边缘中点；LH/RH是穿着者左/右真实下摆与侧缝交角。说明衣物朝向、画面区域和语义侧别依据，不能把画面左右直接当穿着者左右；不清楚时UNKNOWN。
CONFIRMED=图像较强支持真实端点从自遮挡到显露，但仍不是独立材料真值；PROVISIONAL=存在具体袖/摆端边和连接结构显露的较强线索，材料或转换尚不确定；UNCERTAIN=证据不足；NO_REVEAL=无支持或端点原已可见。不要因无法完全确认材料身份就抹去具体暂定线索，也不能将面积增长、轮廓凸点、普通缝边、标签出现等当端点显露。端点重新进入视野不是解除自遮挡。其他角点被遮挡不否定目标显露。每个目标给出Before/After证据、对应、反解释与0无/1弱/2具体暂定/3强视觉证据评分。中文输出。'''
CMP_PROMPT='''独立比较已冻结的原始意图提取与整件衣物效果观察。不得修改任一上游记录，不根据After补写原始预期。本步只做Intention–Outcome Comparison和Unexpected Benefit Discovery，不提出技能。
对每个预期变化判断EXPECTED_OBSERVED/EXPECTED_MISSING/INDETERMINATE；EXPECTED_MISSING只是未确认，不等于证明没发生。对四个语义角点逐一输出比较，允许UNEXPECTED_OBSERVED。
关键：使用语义部位、原话指向的Before图像区域、变化类型建立对应，不能按“左/右”文字匹配。原意图reference_frame未知时不得擅自变成WEARER。若有具体区域证据，可说明在相应语义映射下为何不一致；若不同解释会改变是否意外，保留INDETERMINATE。
没有提到某部位，并不自动证明预测其不变。absence_of_mention_only必须忠实判断；仅凭未提及不能标SUPPORTED。SUPPORTED/PROVISIONAL意外收益需要明确的操作目标、区域关系或预期变化与额外观察之间的具体差异。普通轮廓变化不能升级为Reveal。已预期的Reveal不进入Hindsight分支。保留暂定证据，不强凑意外收益。
每个ID完整输出四个corner_comparisons；只使用给定记录，中文。'''
HYP_PROMPT='''依据本轮自动发现的意外Corner Reveal候选进行Hindsight Goal Relabeling与Skill Hypothesis Induction。locked_goal不可修改，不能换成普通布片、缝边、面积或标签显露。原始意图与实际效果已经独立提取并比较；引用其证据与不确定性，不事后改写原意图，不把画面侧别偷换成穿着者侧别。若复查发现无法支持该意外收益，返回INSUFFICIENT_EVIDENCE，不强行完成。
先解释观察到了什么，再提出至少两个竞争性物理机制，分别说明事实、隐藏层序等假设、能够区分机制的预测及反证。不得预设抓取侧、固定图像方向、几何抓点规律或操作顺序。自主从当前可见几何归纳“操作哪块布料、如何选Pa、如何决定方向和距离、预计目标端点发生什么”，不能复用历史坐标。
若事先希望主动实现该意外效果，在新RGB中应该怎样选择动作？可以保留条件性的实验假设，不要求唯一机制。给出可见成功判据和下一次最有信息价值的单动作实验；不要声称单例已得到可靠技能或物理因果。中文输出。'''
SYNTH_PROMPT='''把本轮假设实现为 generate_action(rgb, target_corner)。唯一目标locked_goal，target_corner为LS/RS/LH/RH短码。rgb是numpy RGB数组；全局注入LOCKED_GOAL、FROZEN_PROMPT和VISION_MODEL(rgb,prompt)。最多一次视觉调用，返回给定action_schema。可用numpy、math、cv2；不得访问文件、网络、环境、机器人或其他报告；不得反射、eval/exec。仅允许import numpy as np、import math、import cv2。不得记忆历史坐标、案例编号、固定图像方向。
明确哪些是VLM语义/几何推理，哪些检查在Python里实际实现，不伪称独立CV检测器。保持指定语义端点目标；目标已可见时可以ABSTAIN，不能改成进一步展平；目标身份/遮挡证据不足时允许ABSTAIN并说明。机制尚有竞争解释但存在具体依据时，可以提出有预测及反证的实验动作。若确实不能形成可执行候选则HYPOTHESIS_ONLY。
Pa/Pb使用原图整数像素；Pb是牵引终点不是隐藏目标坐标。夹爪角度沿用已有图像角定义0..180。不得宣称毫米/深度/Host可行性已检查。experimental_only=true。推理Prompt只带本轮假设与任务接口，不带历史原始意图作为当前目标。目标锁定，输出必要材料证据、动作理由、预期效果、反证及不确定性。'''
