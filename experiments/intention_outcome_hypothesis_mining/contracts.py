"""Fixed analysis instructions and output contract; no learned manipulation rules."""
S = {'type': 'string'}
def enum(*values): return {'type': 'string', 'enum': list(values)}
def array(items): return {'type': 'array', 'items': items}
def obj(fields):
    return {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}

GOALS = ['Reveal(LS)', 'Reveal(RS)', 'Reveal(LH)', 'Reveal(RH)']
PROMPT = '''独立分析一条真实T恤操作，建立Intention–Outcome Hypothesis Bank。第一张是Before，第二张是After。仅使用本条经验；禁止工具调用、文件读取、网页访问或读取项目指令。不要调用机器人。
任务只包括意图—结果比较及可检验假设，不生成动作策略或Pa/Pb，不归纳通用Skill，不合并其他案例，不更新知识状态。你没有其他案例的结果，也不应猜测。
1. Intended Effect：先忠实提取原始Agent认为抓住什么、希望改变什么及明确预期。原始文字是历史认知，不自动等于材料事实。original_quotes只能逐字引用所给原始意图（每个引用是连续原文片段，不使用省略号拼接），缺失时空列表。不能根据After补写或合理化原意图。没有提到某部位是UNSPECIFIED，不是预测它保持不变；EXPLICIT_UNCHANGED须给原文证据。
2. Observed Effect：检查整件衣服，分别记录整体平移/转向、局部相对形变、显露、新遮挡、视野/相机变化、执行不确定性。执行完成不代表抓取成功，计划坐标不证明材料身份。保留实际抬升等物理条件，不把不同参数的经验当作相同动作。
逐一检查四个目标：LS/RS为穿着者左/右袖口开口边缘中点，LH/RH为穿着者左/右下摆与侧缝交角。严格区分画面左右与穿着者左右，说明依据；无法确定则UNKNOWN。端点重新进入视野、面积增加、轮廓扩张、普通缝边/标签清晰不自动等于Corner Reveal。CONFIRMED只表示本次视觉分析中的强证据，不是独立物理真值；不能把隐藏层序、夹持层或连接关系的推测当事实。
3. Mismatch：对原始预期和整件衣服效果逐项比较。区分EXPECTED_OBSERVED、EXPECTED_NOT_CONFIRMED、UNEXPECTED_CHANGE、UNSPECIFIED_CHANGE、INDETERMINATE。UNEXPECTED_CHANGE须说明具体预期与观察的差异，不能仅因原文没提该处。未确认预期不等于证实没有变化。没有值得解释的Mismatch允许空列表；也可保留比较项但不生成假设。
4. Hypothesis：仅对本条经验值得解释的差异提出可证伪物理/几何机制。可以有竞争解释；没有依据时hypotheses=[]。不要为了简短丢掉本条有根据的竞争假设，也不要凑数量。observable_conditions仅描述Before可见线索，不把After知识伪装成前验特征。predicted_effect描述未来相似条件下可观察的结果，falsification_condition须区分机制反证、抓取失败、作用未到达、观察不足。未经确认的结构保留为假设。假设可涉及显露、意外遮挡、形变或未达预期，不限定必须有益；普通形变不能冒充角点恢复。
每条假设保留所要求全部字段，source_experience_id必须等于输入experience_id；hypothesis_id依次使用输入ID加-H001、-H002等。mismatch_id使用输入ID加-M001等，hypothesis.mismatch_ids引用本条比较。evidence_level是这条假设所依赖的观察证据强度，不是机制已验证。related_corner_goals仅列实际关联的四目标，无则空列表。categories用于报告索引，不是已有技能名称。
逐对完整输出四个角点，不遗漏。所有事实、模型解释、不确定性明确区分，用中文回答。'''

CORNER = obj({
    'goal': enum(*GOALS), 'wearer_side_basis': S,
    'before_state': enum('VISIBLE','OCCLUDED','SUSPECTED_OCCLUDED','OUT_OF_VIEW','UNKNOWN'),
    'after_state': enum('VISIBLE','OCCLUDED','SUSPECTED_OCCLUDED','OUT_OF_VIEW','UNKNOWN'),
    'change': enum('REVEAL','NEW_OCCLUSION','VISIBILITY_IMPROVED','VISIBILITY_REDUCED','UNCHANGED','VIEW_CHANGE','UNKNOWN'),
    'evidence_level': enum('CONFIRMED','PROVISIONAL','UNKNOWN','NO_EVIDENCE'),
    'before_evidence': S, 'after_evidence': S, 'material_correspondence': S,
    'alternative_explanation': S,
    'original_expectation': enum('EXPLICIT_CHANGE','EXPLICIT_UNCHANGED','UNSPECIFIED','UNKNOWN'),
    'original_expectation_quote': {'type':['string','null']}, 'uncertainty': S,
})
MISMATCH = obj({
    'mismatch_id': S,
    'type': enum('EXPECTED_OBSERVED','EXPECTED_NOT_CONFIRMED','UNEXPECTED_CHANGE','UNSPECIFIED_CHANGE','INDETERMINATE'),
    'intended_effect': S, 'observed_effect': S, 'comparison_evidence': S,
    'related_corner_goals': array(enum(*GOALS)), 'worth_explaining': {'type':'boolean'}, 'uncertainty': S,
})
HYPOTHESIS = obj({
    'hypothesis_id': S, 'source_experience_id': S, 'intended_effect': S, 'observed_effect': S,
    'mismatch': S, 'observable_conditions': S, 'hypothesized_mechanism': S,
    'predicted_effect': S, 'falsification_condition': S,
    'evidence_level': enum('CONFIRMED','PROVISIONAL','UNKNOWN'), 'uncertainty': S,
    'mismatch_ids': array(S), 'related_corner_goals': array(enum(*GOALS)),
    'categories': array(enum('CORNER_REVEAL','UNEXPECTED_OCCLUSION','LOCAL_DEFORMATION','GLOBAL_MOTION','UNMET_EXPECTATION','VIEW_CHANGE','OTHER')),
})
SCHEMA = obj({
    'experience_id': S,
    'intended_effect': obj({'intended_object':S,'intended_changes':array(S),'explicit_expected_outcome':S,
                            'original_quotes':array(S),'unspecified_elements':array(S),'uncertainty':S}),
    'observed_effect': obj({'whole_garment_changes':array(S),'global_motion':S,'relative_deformation':S,
                            'new_occlusions':array(S),'view_confound':S,'execution_evidence_limits':S,
                            'effect_determinacy':enum('CLEAR','PARTIAL','UNKNOWN'),'uncertainty':S}),
    'corners': array(CORNER), 'mismatches': array(MISMATCH), 'hypotheses': array(HYPOTHESIS),
    'no_hypothesis_reason': S,
})

def validate_semantics(output, record):
    """Check IDs, complete corner coverage, and literal intention quotations."""
    rid = record['experience_id']
    if output['experience_id'] != rid: raise ValueError('Wrong experience_id')
    corners = output['corners']
    if len(corners) != 4 or {c['goal'] for c in corners} != set(GOALS):
        raise ValueError('Must cover each of the four corners exactly once')
    originals = [str(v) for v in record['original_intention'].values()]
    quotes = list(output['intended_effect']['original_quotes'])
    for c in corners:
        if c['original_expectation'] == 'EXPLICIT_UNCHANGED' and not c['original_expectation_quote']:
            raise ValueError('EXPLICIT_UNCHANGED requires original quotation')
        if c['original_expectation_quote']: quotes.append(c['original_expectation_quote'])
    for q in quotes:
        if not q.strip() or not any(q in original for original in originals):
            raise ValueError('Intention quotation not present verbatim in original record')
    mids = [m['mismatch_id'] for m in output['mismatches']]
    if mids != [f'{rid}-M{i:03d}' for i in range(1,len(mids)+1)]:
        raise ValueError('Invalid mismatch IDs')
    eligible = {m['mismatch_id'] for m in output['mismatches'] if m['worth_explaining'] and m['type']!='EXPECTED_OBSERVED'}
    for i,h in enumerate(output['hypotheses'],1):
        if h['hypothesis_id'] != f'{rid}-H{i:03d}' or h['source_experience_id'] != rid:
            raise ValueError('Invalid hypothesis provenance')
        if not h['mismatch_ids'] or not set(h['mismatch_ids']).issubset(eligible):
            raise ValueError('Hypothesis must cite a worth-explaining mismatch in this experience')
    if not output['hypotheses'] and not output['no_hypothesis_reason'].strip():
        raise ValueError('Empty hypotheses require a reason')
