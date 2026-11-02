S={'type':'string'};B={'type':'boolean'}
def enum(*x):return {'type':'string','enum':list(x)}
def obj(x):return {'type':'object','properties':x,'required':list(x),'additionalProperties':False}
def arr(x):return {'type':'array','items':x}
POINT={'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':2}
ENDPOINT=enum('LEFT_CUFF','RIGHT_CUFF','LEFT_HEM','RIGHT_HEM','SEMANTIC_SIDE_UNKNOWN','NONE')
EFFECT=obj({'id':S,'classification':enum('CONFIRMED_SEMANTIC_REVEAL','SUSPECTED_REVEAL','ORDINARY_DEFORMATION','VIEW_CHANGE','INSUFFICIENT_EVIDENCE','NO_REVEAL'),'endpoint':ENDPOINT,'image_region':S,'before_identity':S,'after_identity':S,'actual_visible_change':S,'global_vs_local':S,'uncertainty':S,'confidence':enum('HIGH','MEDIUM','LOW')})
EFFECT_SCHEMA=obj({'effects':arr(EFFECT)})
MISMATCH=obj({'id':S,'intended_region':S,'intended_change':S,'intention_known':B,'mismatch':enum('UNPLANNED','PLANNED','AMBIGUOUS','NO_BENEFIT'),'reason':S})
MISMATCH_SCHEMA=obj({'comparisons':arr(MISMATCH)})
REVIEW=obj({'id':S,'reveal_verified':B,'endpoint':ENDPOINT,'unexpected_verified':B,'evidence':S,'uncertainty':S})
REVIEW_SCHEMA=obj({'reviews':arr(REVIEW)})
HYP=obj({'id':S,'effect_explanation':S,'observable_condition':S,'competing_mechanism':S,'different_action_prediction':S,'supporting_ids':arr(S),'contradicting_ids':arr(S),'unknowns':S,'falsifier':S})
HYP_SCHEMA=obj({'status':enum('HYPOTHESES','DATA_INSUFFICIENT'),'intention_actual_effect_analysis':S,'hypotheses':arr(HYP),'reason':S})
SKILL_SCHEMA=obj({'status':enum('CANDIDATE','NOT_YET_EXECUTABLE'),'goal':S,'procedure':arr(S),'observable_prerequisites':arr(S),'abstention_conditions':arr(S),'competing_hypotheses':arr(S),'reason':S})
ACTION_SCHEMA=obj({'decision':enum('ACTION','ABSTAIN'),'abstain_reason':enum('NONE','NO_EXECUTABLE_SKILL','INSUFFICIENT_OBSERVATION','UNRESOLVED_HYPOTHESES'),'target_corner':ENDPOINT,'visible_evidence':S,'grasp_region':S,'grasp_px':POINT,'target_px':POINT,'grasp_angle_deg':{'type':'integer','minimum':0,'maximum':180},'expected_change':S,'hypothesis_id':S,'falsifier':S,'uncertainty':S,'exploration_candidate':B})
OBS_PROMPT='''独立分析真实操作前后RGB。每个ID有一张Before和一张After，顺序固定。没有原始规划理由；不要推测意图。判断袖口或下摆真实语义端点是否从遮挡变成可见，必须同时说明前后身份依据。分清已证实语义显露、疑似袖状轮廓、普通形变、视野变化、证据不足。画面左右不等于语义左右，不能确认则SEMANTIC_SIDE_UNKNOWN。不用面积增长替代显露，不要求其他端点维持可见，不把EXECUTED当任务成功。每个ID输出一条最有证据的效果；若多处显露在证据字段列出但不虚构身份。保持每个字段简洁，只用所给图像。'''
MISMATCH_PROMPT='''只从原始决策字段提取原始意图，不从After或效果倒推意图。和独立观察结果比较，判定受益是否原意图之外。没有意图记录则intention_known=false和AMBIGUOUS。语义侧别或措辞不明确不能强行UNPLANNED。预期相同部位变化则PLANNED；无有益显露则NO_BENEFIT。不得事后把间接效果补进原始意图。'''
REVIEW_PROMPT='''独立审核真实前后图和原始意图，不提供筛选器结论。核验是否有真实语义端点显露及是否未被原始意图预期。疑似轮廓不算已证实。使用过程图时注意腕载相机视角变化；无法确定材料身份保留不确定。不输出操作规则。'''
HYP_PROMPT='''从给定真实经验中开展Hindsight Skill Discovery。先解释原意图、实际影响、意外收益；再提出至少两个相互竞争的可检验解释及不同动作预测。不要预设抓点、方向或机制，不把隐藏层序当事实。参考反例，混合结果不是失败。不按照历史共同方向直接归纳规则。只引用给定ID；证据不足可以DATA_INSUFFICIENT。'''
SYNTH_PROMPT='''把自主发现的假设转换为可用于当前新RGB的动作生成程序说明。程序将通过冻结的视觉模型定位Pa/Pb。根据当前可见证据选择区域、方向，不能包含历史坐标、run ID、针对单张图的条件分支。至多8条主要步骤；仍未区分机制时明确作为实验候选，不冒充已验证Skill；允许三类ABSTAIN。不要引入经验未支持的人工规则。'''
ACTION_PROMPT='''根据当前RGB、目标和冻结候选策略，重新定位一个lift_pull_place动作。只能看当前图，不读取历史图或After。1280x720等原图像素坐标，图像夹爪角0..180，实测XY10..150mm由Host检查。证据不足允许ABSTAIN；若仅机制未区分但存在有依据的动作，可exploration_candidate=true并说明可证伪预测。不能宣称已通过安全检查。不允许工具。'''
