"""Two independent scoring requests and a mandatory human Top10 choice."""
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from . import action_preference as pref, table_target_depth
from .experiments import one_shot_goal_planner_probe_v1 as io
from .dual_arm.geometry import DualArmError

SCORE=dict(type='integer',minimum=0,maximum=100)
POINT=dict(type='array',items=dict(type='integer'),minItems=2,maxItems=2)
POINT_SCHEMA=dict(type='object',additionalProperties=False,properties=dict(
    id=dict(type='string',enum=[f'P{i:02d}' for i in range(1,11)]),grasp_px=POINT,Ps=SCORE,
    grasp_angle_deg=io.ROTATION_SCHEMA['properties']['grasp_angle_deg'],reason=dict(type='string')),
    required=['id','grasp_px','Ps','grasp_angle_deg','reason'])
PS_SCHEMA=dict(type='object',additionalProperties=False,properties=dict(
    decision=dict(type='string',enum=['ACTION','DONE','ABSTAIN']),
    points=dict(type='array',items=POINT_SCHEMA,minItems=0,maxItems=10),short_reason=dict(type='string')),
    required=['decision','points','short_reason'])
COMPLETION_CHECK_SCHEMA=dict(type='object',additionalProperties=False,properties=dict(
    torso_not_covered=dict(type='boolean'),both_sleeves_out=dict(type='boolean'),
    hem_open=dict(type='boolean'),no_major_fold=dict(type='boolean'),evidence=dict(type='string')),
    required=['torso_not_covered','both_sleeves_out','hem_open','no_major_fold','evidence'])
AUTO_PS_SCHEMA=dict(type='object',additionalProperties=False,properties=dict(
    **PS_SCHEMA['properties'],completion_check=COMPLETION_CHECK_SCHEMA),
    required=[*PS_SCHEMA['required'],'completion_check'])
PREVIOUS_EFFECT_SCHEMA=dict(type='object',additionalProperties=False,properties=dict(
    verdict=dict(type='string',enum=['IMPROVED','UNCHANGED','WORSE','UNCERTAIN']),
    changed_relation=dict(type='string'),evidence=dict(type='string')),
    required=['verdict','changed_relation','evidence'])
PS_CONTEXT_SCHEMA=dict(type='object',additionalProperties=False,properties=dict(
    **PS_SCHEMA['properties'],previous_effect=PREVIOUS_EFFECT_SCHEMA),
    required=[*PS_SCHEMA['required'],'previous_effect'])
AUTO_PS_CONTEXT_SCHEMA=dict(type='object',additionalProperties=False,properties=dict(
    **PS_CONTEXT_SCHEMA['properties'],completion_check=COMPLETION_CHECK_SCHEMA),
    required=[*PS_CONTEXT_SCHEMA['required'],'completion_check'])
COMMON='''你是衣物展开实验中的评分器。第一张是perception当前RGB，第二张是展平目标RGB。
所有像素坐标仅使用第一张原图，左上为原点，x向右、y向下；不把目标图坐标搬到当前图。
保持现有lift_pull_place：抓取、抬升约50毫米、平移10–150毫米、放下；仅允许抓前调整夹爪，夹住后不旋转。
目标是展开主要衣身与袖部、减少折叠和遮挡；允许整体平移/旋转和普通小褶皱，不追求背景或绝对桌面位置一致。
“差不多展开”有四个必须同时满足的结构条件：
1. 衣身正面从领口下方到下摆基本是一片连续露出的布面，不能有边界明确的大片翻折布片盖住胸腹或印花；
2. 两只袖子都基本向外舒展，袖身不能明显缩在肩部或腋下附近、堆拢在衣身旁或压在衣身下面；袖口可见、袖子轮廓可辨不等于袖子展开。分别检查左右袖从肩部、腋下到袖口的布面是否舒展，不要求绷紧、对称或消除普通皱纹；
3. 下摆大部分边线及左右两侧角都基本向外展开；分别沿同侧腋下、衣身侧边到下摆角检查轮廓，不能仍有明显内收的衣角或局部侧边向内收窄，导致一侧下部衣身横向宽度未释放。下摆边线连续、两角可见、没有中央深凹，都不能单独证明本项满足；也不能有深凹缺口分成两片或衣身下部大块堆叠；
4. 没有边界明确、横跨约三分之一衣身宽度、或覆盖约十分之一可见衣物面积的翻折布片。
这些是图像近似判断，不要求精确测量；看不清是否满足就不要声称DONE。
第二张goal图是本实验的合格参考：其中也有长斜向皱纹与局部下摆卷曲。不要把单独一条长皱纹、阴影或目标图同程度的卷曲误判成翻折布片。
区分两类未展开证据：布层覆盖，以及轮廓/部位相对位置仍收缩。判定大片翻折覆盖需要可见的布片边缘和被遮区域；但袖身或下摆角明显内收，不需要出现独立翻折自由边也能判为未展开。即使布面连续可见、印花没有遮挡，只要下摆角及相连侧边仍明显收缩，hem_open也应为false。
允许不伴随明显轮廓内收的普通皱纹、窄折痕和袖口轻微卷边，也允许整体平移/旋转。沿衣服自身领口至下摆的方向判断相对位置，不要求左右严格对称、侧边笔直、固定宽度或匹配目标图绝对像素；不能仅因旋转、透视或设计上的宽窄差异判未展开。
反例：虽然T恤外形、领口和两袖都能认出，但胸腹被独立的翻折布片遮盖、下摆中间深凹并分成两片时，不能判DONE。
没有深度输入，不臆造毫米换算，不声称通过Host的几何或实机检查。分数0–100是主观评价，不是成功概率，不要求加和为100。
只输出指定JSON，不调用工具。
'''+io.GRASP_ANGLE_PROMPT+io.GRIPPER_BOUNDARY_PROMPT
PS_PROMPT=COMMON+'''
现在只提出10个位置不同、空间上有分布的可见候选抓点，并评价抓点本身Ps；不要提出Pb或比较拉动方向。
Ps主要衡量位置清楚、夹取可行、可见外露边角/可分离布层的依据；多层聚拢、遮挡和难定位会降低分数。
优先清楚可见的外露边角，避免猜测隐藏角点；也保留可见折边作为备选，不把10点集中在同一小区域。
Ps不包含尚未指定的移动方向或展开收益，后续另评Pt，避免重复计算。
ACTION返回恰好10个不重复的整数抓点，id为P01至P10；每点给抓前夹爪角度和一句中文reason。
先逐项检查衣身无大块覆盖、两袖主要向外、下摆展开、无大折层；四项均有可见证据才DONE。
只认出T恤轮廓、领口和袖子不足以DONE；胸腹有独立翻折布片或下摆分成两片属于ACTION，目标图也有的长皱纹不属于此类。
ACTION的short_reason必须指出未通过的结构条件和仍需释放的具体布层。
无法看清状态或无法提出有依据的候选时ABSTAIN；DONE/ABSTAIN的points为空。
short_reason最多一句中文。
'''
PT_PROMPT=COMMON+'''
现在只评给定固定动作的条件轨迹分Pt(x,Pb|x)：假设已经成功抓住x，沿指定方向移动到Pb的预期展开收益。
重点比较打开哪一处覆盖/折叠、改善范围，以及新增折叠或只搬动整件衣服的风险；抓点本身的可抓性另由Ps评价，不在Pt重复加分。
候选是同一状态的互斥备选，不是连续执行；每个候选的Pa、Pb及抓前角度全部固定，不得生成或改动。
只对提供的每个id返回一个Pt和一句中文reason，具体说明预期局部变化或风险。编号和顺序不表示优先级。
'''
CONTEXT_PROMPT='''
第三张是上一轮动作前的干净RGB；第一张是本轮当前RGB，也是上一轮动作后的新观测。
上一轮真实派发的Pa→Pb和执行状态以文字提供，坐标属于第三张图；不要把这些坐标当作本轮候选。
先对比第三张与第一张，判断上一动作实际改变了哪处布料覆盖/折叠关系，再评价本轮候选。
不要把上一轮的预期当作已经发生的效果；若图像不足以判断，明确写UNCERTAIN。
本轮Pa、Pb仍只使用第一张当前图的坐标。'''
PS_CONTEXT_PROMPT=PS_PROMPT+CONTEXT_PROMPT+'''
previous_effect中给出上一动作的局部效果判断：verdict、changed_relation及可见图像证据。
即使判断为UNCERTAIN，仍可按本轮当前图提出有依据的候选。'''
PT_CONTEXT_PROMPT=CONTEXT_PROMPT+'''
上一轮效果判断来自本轮独立抓点评分阶段，仅作为视觉假设；结合图像核查，不要机械沿用。
根据当前布料状态评价本轮固定轨迹及完整动作。'''
COMPLETION_OUTPUT_PROMPT='''
特别检查袖子：只要任一袖身仍明显收缩在肩部或腋下附近，即使袖口外露，both_sleeves_out也必须为false，不能DONE；evidence分别说明左右袖依据。无法确认袖子已舒展时也不能将此项设为true。
自动模式另返回completion_check：torso_not_covered、both_sleeves_out、hem_open、no_major_fold四项分别给true/false，
evidence指出具体翻折覆盖关系，或袖身/衣角相对同侧肩腋及侧边的内收证据；分别检查左右下摆角，不能仅以边线连续或没有深凹认定hem_open。普通皱纹不等于大折层，但也不能用“没有自由折边”否定明显的轮廓收缩。
只有四项都明确为true才可决定DONE；任一项false时不要DONE。
如果四项都为true，就应DONE，不要为了生成候选而继续操作。'''
AUTO_PS_PROMPT=PS_PROMPT+COMPLETION_OUTPUT_PROMPT
AUTO_PS_CONTEXT_PROMPT=PS_CONTEXT_PROMPT+COMPLETION_OUTPUT_PROMPT


def trajectory_schema(ids):
    item=dict(type='object',additionalProperties=False,properties=dict(
        id=dict(type='string',enum=ids),Pt=SCORE,reason=dict(type='string')),required=['id','Pt','reason'])
    return dict(type='object',additionalProperties=False,properties=dict(
        scores=dict(type='array',items=item,minItems=len(ids),maxItems=len(ids))),required=['scores'])


def make_payload(root, folder, before, prompt, schema, extra='', *, previous_context=None, previous_effect=None, assess_previous_effect=True):
    size=list(Image.open(before.image).size)
    body=io.payload(before.image,root/'goal/target.png',size)
    body['instructions']=prompt
    if extra:body['input'][0]['content'].append(dict(type='input_text',text=extra))
    if previous_context is not None:
        plan=previous_context['plan']
        body['input'][0]['content'].append(dict(type='input_text',text=(
            '第三张图是上一轮动作前的原图。上一轮实际执行状态：EXECUTED；'
            f'Pa={plan["grasp_px"]}，Pb={plan["target_px"]}，'
            f'抓前夹爪角度={plan.get("grasp_angle_deg")}，'
            f'已完成运动命令数={previous_context.get("completed_commands")}；'
            f'是否小幅整理={plan.get("is_smoothing",False)}。'
            + ('第一张图是当前新观测；请先判断上一动作的实际局部效果。' if assess_previous_effect
               else '第一张图是当前新观测；历史图和动作仅供下一动作规划参考，无需输出上一动作效果评价。'))))
        body['input'][0]['content'].append(dict(type='input_image',
            image_url=io.image_data(previous_context['before_image']),detail='high'))
        if previous_effect is not None:
            import json
            body['input'][0]['content'].append(dict(type='input_text',
                text='抓点阶段的上一动作效果判断（仅作可核查的视觉假设）：'+json.dumps(previous_effect,ensure_ascii=False)))
    body['text']['format'].update(name='preference_scores',schema=schema)
    return body


def validate_points(output,size,*,contextual=False,automatic=False):
    schema=(AUTO_PS_CONTEXT_SCHEMA if contextual else AUTO_PS_SCHEMA) if automatic else (
        PS_CONTEXT_SCHEMA if contextual else PS_SCHEMA)
    io.jsonschema.validate(output,schema)
    if automatic:
        check=output['completion_check']
        ready=all(check[key] for key in ('torso_not_covered','both_sleeves_out','hem_open','no_major_fold'))
        if output['decision']=='DONE' and not ready:
            raise ValueError('DONE conflicts with incomplete structural checklist')
        if output['decision']=='ACTION' and ready:
            raise ValueError('ACTION conflicts with completed structural checklist; no action dispatched')
    if output['decision']!='ACTION':
        if output['points']:raise ValueError('non-ACTION cannot contain candidates')
        plan=dict(decision=output['decision'],grasp_px=None,target_px=None,motion_type=None,
                  grasp_angle_deg=None,short_reason=output['short_reason'])
        io.validate_plan(plan,size,require_grasp_rotation=True)
        return plan
    points=output['points']
    if len(points)!=10 or {p['id'] for p in points}!={f'P{i:02d}' for i in range(1,11)}:
        raise ValueError('must provide exactly 10 uniquely identified points')
    if len({tuple(p['grasp_px']) for p in points})!=10:raise ValueError('duplicate grasp positions')
    for p in points:
        x,y=p['grasp_px']
        if not 2<=x<size[0]-2 or not 2<=y<size[1]-2:raise ValueError('grasp outside supported image')
    return None


def generate_candidates(points,before,config,folder,length_px=120):
    """Pure recorded-observation checks; no runtime or robot calls."""
    accepted=[];rejected=[]
    (folder/'candidate_depth').mkdir(parents=True,exist_ok=True)
    arm=config.arms['left'];world_to_base=np.linalg.inv(arm.world_from_base)
    spread=config.limits['max_surface_spread_mm'];w,h=Image.open(before.image).size
    for p in points:
        for direction,dx,dy in [('R',length_px,0),('D',0,length_px),('L',-length_px,0),('U',0,-length_px)]:
            pa=p['grasp_px'];pb=[pa[0]+dx,pa[1]+dy];identifier=p['id']+direction
            r=dict(id=identifier,point_id=p['id'],grasp_px=pa,target_px=pb,
                grasp_angle_deg=p['grasp_angle_deg'],Ps=p['Ps'],point_reason=p['reason'])
            try:
                if not 2<=pb[0]<w-2 or not 2<=pb[1]<h-2:raise ValueError('target outside image')
                a=before.sample(pa,spread)
                b=table_target_depth.sample_target(before,pb,pa,spread,folder/'candidate_depth'/f'{identifier}.json')
                distance=float(np.linalg.norm(b[:2]-a[:2]))
                if not 10<=distance<=150:raise ValueError('metric translation outside 10–150 mm')
                z=max(a[2],b[2])+50
                for point in (a+[0,0,50],a+[0,0,-15],np.r_[a[:2],z],np.r_[b[:2],z],b+[0,0,5],b+[0,0,30]):
                    arm.validate_point((world_to_base@np.r_[point,1])[:3])
                r['metric_translation_mm']=distance;accepted.append(r)
            except (DualArmError,ValueError) as error:
                rejected.append(dict(r,rejection=f'{type(error).__name__}: {error}'))
    io.save(folder/'candidate_filter.json',dict(accepted=accepted,rejected=rejected,
        scope='historical depth/target fallback, metric distance and TCP workspace; not full IK/collision certification'))
    if len(accepted)<10:raise ValueError(f'only {len(accepted)} eligible actions; cannot display Top10; no retry')
    return accepted


def score_round(root,folder,before,headers,timeout_s,config,request_once,*,automatic=False,previous_context=None):
    import json
    if previous_context is not None and not automatic:
        raise ValueError('previous-action context is supported only in automatic Top5 mode')
    started=time.monotonic();stage_metrics=[]
    if previous_context is not None:
        copied=folder/'context_previous_before.png'
        shutil.copyfile(previous_context['before_image'],copied)
        previous_context=dict(previous_context,before_image=copied)
        io.save(folder/'previous_action_context.json',dict(
            source_round=previous_context['source_round'],execution_status='EXECUTED',
            completed_commands=previous_context.get('completed_commands'),
            previous_before_sha256=io.sha(copied),current_sha256=io.sha(before.image),
            previous_action=previous_context['plan'],
            input_images=['current RGB','goal RGB','previous action before RGB']))
    with (folder/'attempt.json').open('x') as f:
        json.dump(dict(started_at=io.utc(),planner='GPTAutomaticTop5' if automatic else 'PsPtHumanPreference',
                       previous_action_context=previous_context is not None,max_requests=2,retries=0),f)
    try:
        point_prompt=(AUTO_PS_CONTEXT_PROMPT if previous_context else AUTO_PS_PROMPT) if automatic else (
            PS_CONTEXT_PROMPT if previous_context else PS_PROMPT)
        point_schema=(AUTO_PS_CONTEXT_SCHEMA if previous_context else AUTO_PS_SCHEMA) if automatic else (
            PS_CONTEXT_SCHEMA if previous_context else PS_SCHEMA)
        first=make_payload(root,folder,before,point_prompt,point_schema,
                           previous_context=previous_context)
        points,m=request_once(folder/'point_scores',first,headers,timeout_s);stage_metrics.append(m)
        io.save(folder/'point_scores/output.json',points)
        stopping=validate_points(points,list(Image.open(before.image).size),
                                 contextual=previous_context is not None,automatic=automatic)
        if automatic:io.save(folder/'completion_check.json',points['completion_check'])
        if previous_context is not None:io.save(folder/'previous_effect.json',points['previous_effect'])
        if automatic:
            from . import automatic_top5 as auto
            auto.record_points(folder,before,points)
        if stopping is not None:return dict(stop_plan=stopping),dict(wall_latency_s=sum(m['wall_latency_s'] for m in stage_metrics))
        before.validate_for(config,live=False)
        candidates=generate_candidates(points['points'],before,config,folder)
        # Ps and point reasons are deliberately withheld from the Pt request.
        inputs=[{k:c[k] for k in ('id','grasp_px','target_px','grasp_angle_deg')} for c in candidates]
        schema=(auto.trajectory_schema if automatic else trajectory_schema)([c['id'] for c in candidates])
        prompt=(auto.PROMPT+PT_CONTEXT_PROMPT if previous_context else auto.PROMPT) if automatic else PT_PROMPT
        body=make_payload(root,folder,before,prompt,schema,json.dumps(inputs,ensure_ascii=False),
                          previous_context=previous_context,
                          previous_effect=points.get('previous_effect') if previous_context else None)
        if automatic:
            body['input'][0]['content'].append(dict(type='input_text',text=(
                '抓点阶段的结构完成检查（请独立复核，可修正）：'+
                json.dumps(points['completion_check'],ensure_ascii=False))))
        output,m=request_once(folder/'trajectory_scores',body,headers,timeout_s);stage_metrics.append(m)
        io.save(folder/'trajectory_scores/output.json',output);io.jsonschema.validate(output,schema)
        scores={r['id']:r for r in output['scores']}
        if len(scores)!=len(candidates) or set(scores)!={c['id'] for c in candidates}:raise ValueError('Pt ids missing/duplicated')
        rows=[dict(c,Pt=scores[c['id']]['Pt'],trajectory_reason=scores[c['id']]['reason']) for c in candidates]
        for row in rows:io.validate_plan(pref.plan_for(row),list(Image.open(before.image).size),require_grasp_rotation=True)
        pref.ranked(rows,1.)  # Reject nonfinite/out-of-scale scores before review.
        io.save(folder/'scored_candidates.json',rows)
        bundle=dict(candidates=rows)
        if automatic:bundle['completion_check']=points['completion_check']
        if previous_context is not None:bundle['previous_effect']=points['previous_effect']
        if automatic:bundle['selection']=output
        return bundle,dict(wall_latency_s=sum(m['wall_latency_s'] for m in stage_metrics))
    finally:
        # Include failed HTTP attempts too, while excluding human waiting time.
        metrics=[io.read(p) for p in (folder/'point_scores/model_metrics.json',folder/'trajectory_scores/model_metrics.json') if p.exists()]
        io.save(folder/'model_metrics.json',dict(planner='GPTAutomaticTop5' if automatic else 'PsPtHumanPreference',
            wall_latency_s=sum(m['wall_latency_s'] for m in (metrics or stage_metrics)),
            scoring_pipeline_s=time.monotonic()-started,model_calls=len(metrics or stage_metrics),
            retries=0,stages=metrics or stage_metrics))


def choose_plan(root,folder,before,bundle,runtime,timing,state_path,port,timeout_s,*,residual_state_path=None):
    if 'stop_plan' in bundle:return bundle['stop_plan']
    state=pref.load_state(state_path)
    review=folder/'preference_review'
    if residual_state_path is not None:
        from . import ranking_residual as residual, residual_features
        correction = residual.load_state(residual_state_path, state['alpha'])
        features = residual_features.extract(before.image, bundle['candidates'], folder/'residual_features')
        timing['residual_features_s'] = features['wall_s']
        started = time.monotonic()
        order, context = residual.prepare(bundle['candidates'], correction, features)
        timing['residual_ranking_s'] = time.monotonic()-started
        # Persist predictions for every historical checkpoint before accepting labels.
        pref.atomic_save(folder/'prediction_before_feedback.json', context)
        pref.atomic_save(folder/'residual_before.json', dict(revision=correction['revision'],
            alpha=correction['alpha'], identity=correction['identity'], checkpoints=correction['checkpoints']))
        manifest=pref.make_review(review,bundle['candidates'],state,
            {'current.png':before.image,'goal.png':root/'goal/target.png'},
            ranked_rows=order, ranking_context=context)
        io.save(folder/'alpha_before.json',dict(alpha=state['alpha'],revision=state['revision'],
            state_path=str(Path(state_path).resolve()),fixed=True))
        choice,elapsed=pref.wait_for_choice(review,manifest,runtime.cancel,port=port,timeout_s=timeout_s)
        timing['human_review_wait_s']=elapsed
        if choice['decision']=='STOP':raise KeyboardInterrupt('user rejected Top10; no action or learning')
        if runtime.cancel.is_set():raise KeyboardInterrupt('cancelled before residual update')
        # Freeze execution independently of training, re-ranking and human order.
        selected=next(r for r in manifest['displayed'] if r['id']==choice['candidate_id'])
        feedback=residual.apply_feedback(residual_state_path,manifest,choice,folder)
        timing['residual_training_s']=feedback['training_s']
        residual.write_progress(residual.load_state(residual_state_path,state['alpha']),root)
        e=feedback['evaluation']
        fmt=lambda v:'无跨组比较' if v is None else f'{v:.3f}'
        print(f"RESIDUAL {feedback['revision_before']} -> {feedback['revision_after']}; fixed alpha {state['alpha']:.7f}; "
              f"pre-update pair accuracy base={fmt(e['baseline']['pairwise_accuracy'])}, current={fmt(e['current']['pairwise_accuracy'])}; "
              f"training {feedback['training_s']:.3f}s; execute human choice {choice['candidate_id']}",flush=True)
        if runtime.cancel.is_set():raise KeyboardInterrupt('cancelled after training; selected action not dispatched')
        return pref.plan_for(selected)
    manifest=pref.make_review(review,bundle['candidates'],state,
        {'current.png':before.image,'goal.png':root/'goal/target.png'})
    io.save(folder/'alpha_before.json',dict(alpha=state['alpha'],revision=state['revision'],state_path=str(Path(state_path).resolve())))
    choice,elapsed=pref.wait_for_choice(review,manifest,runtime.cancel,port=port,timeout_s=timeout_s)
    timing['human_review_wait_s']=elapsed
    if choice['decision']=='STOP':raise KeyboardInterrupt('user rejected Top10; no action or alpha update')
    if runtime.cancel.is_set():raise KeyboardInterrupt('cancelled before preference update')
    feedback=pref.apply_feedback(state_path,manifest,choice['candidate_id'])
    io.save(folder/'alpha_feedback.json',feedback)
    io.save(folder/'ranking_after_feedback.json',pref.ranked(bundle['candidates'],feedback['alpha_after']))
    print(f"ALPHA {feedback['alpha_before']:.4f} -> {feedback['alpha_after']:.4f}; chosen {choice['candidate_id']}; preference only",flush=True)
    if not feedback['chosen_can_be_top1_with_scalar']:
        print('NOTE: this choice cannot lead the displayed Top10 by changing alpha alone within [0,5].',flush=True)
    selected=next(r for r in manifest['displayed'] if r['id']==choice['candidate_id'])
    # Selection is frozen: updated alpha only affects subsequent rounds.
    return pref.plan_for(selected)


def verify_scene(before,fresh,plan,folder,*,automatic=False):
    """Re-observe after a human pause; fail closed instead of silently re-grounding."""
    from .experiments.trajectory_memory_full_episode_v1 import unchanged
    if before.meta['calibration_id']!=fresh.meta['calibration_id']:raise DualArmError('review calibration changed')
    if Image.open(before.image).size!=Image.open(fresh.image).size:raise DualArmError('review image dimensions changed')
    # Automatic execution checks only the selected grasp neighborhood.
    # People/chairs elsewhere in the frame must not veto a stationary grasp.
    same,global_delta=(True,None) if automatic else unchanged(before.image,fresh.image)
    a=np.asarray(Image.open(before.image).convert('RGB'),dtype=float)
    b=np.asarray(Image.open(fresh.image).convert('RGB'),dtype=float)
    x,y=plan['grasp_px'];sl=(slice(max(0,y-30),y+31),slice(max(0,x-30),x+31))
    local_delta=float(np.quantile(np.abs(a[sl]-b[sl]),.95))
    # Human-review checks retain their existing stricter policy.
    local_threshold=25 if automatic else 15
    local_ok=local_delta<=local_threshold if automatic else local_delta<local_threshold
    ok=bool(same and local_ok)
    io.save(folder/('pre_execution_scene_check.json' if automatic else 'human_review_scene_check.json'),dict(passed=ok,global_rgb_q95=global_delta,
        local_grasp_rgb_q95=local_delta,global_threshold=None if automatic else 8,
        global_check_enabled=not automatic,local_threshold=local_threshold,
        local_comparison="<=" if automatic else "<",
        original_sha256=io.sha(before.image),fresh_sha256=io.sha(fresh.image),
        limitation='conservative RGB change check, not a guarantee of identical hidden cloth layers'))
    if not ok:raise DualArmError('scene changed during '+('automatic planning' if automatic else 'human review')+'; selected action not dispatched')


def append_report(root,summary):
    if summary.get('ranking_correction_enabled'):
        lines=['\n\n人工排名修正：S = Ps + 固定 alpha × Pt + Rθ。',
               '人工分组只监督Top10的跨组偏好，组内并列；全同组不更新权重。执行动作单独锁定为人的选择。先保存训练前评估，再更新Rθ，更新仅用于后续轮次。',
               '[后续新状态效果](learning_progress.md)。无后续人工评审时，不能声称泛化改善。', '',
               '|轮次|视觉特征秒|排名秒|人工等待秒|训练秒|人工选择|', '|---|---:|---:|---:|---:|---|']
        for item in summary['rounds']:
            folder=root/item['round'];p=folder/'residual_feedback.json';t=item['timing']
            event=io.read(p) if p.exists() else {}
            lines.append(f"|{item['round']}|{t.get('residual_features_s',0):.3f}|{t.get('residual_ranking_s',0):.3f}|"
                f"{t.get('human_review_wait_s',0):.2f}|{t.get('residual_training_s',0):.3f}|{event.get('chosen','—')}|")
        with (root/'report.md').open('a') as f:f.write('\n'.join(lines)+'\n')
        return
    lines=['\n\n人工候选实验：P = Ps + alpha * Pt。每轮两阶段评分（停止决定时仅一阶段），候选返回后等待人工选择；等待时间不计入模型耗时。',
           'Ps评抓点本身，Pt评成功抓住后的条件展开收益。只有被展示的候选参与选择偏好学习；更新alpha不代表实机成功，也不替换本轮人工选定动作。',
           '人工选择后使用新RGB-D检查场景及原有Host检查；未通过则停止，不自动改选。',
           '\n|轮次|人工等待秒|alpha变化|人工选择|\n|---|---:|---|---|']
    for item in summary['rounds']:
        folder=root/item['round'];p=folder/'alpha_feedback.json'
        event=io.read(p) if p.exists() else {}
        lines.append(f"|{item['round']}|{item['timing'].get('human_review_wait_s',0):.2f}|{event.get('alpha_before','—')} → {event.get('alpha_after','—')}|{event.get('chosen','—')}|")
        if (folder/'preference_review/index.html').exists():lines.append(f"\n[{item['round']} Top10]({item['round']}/preference_review/index.html)\n")
    with (root/'report.md').open('a') as f:f.write('\n'.join(lines)+'\n')
