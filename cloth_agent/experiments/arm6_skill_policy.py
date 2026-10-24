"""Arm6 unfolding from demonstrated visual skills, without graph state matching."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal

import numpy as np
from PIL import Image, ImageDraw
from .cloth_skill_library import load_library, PROTOCOL
from .discovery_schema import obj, array, S, N, enum
from .discovery_data import save, file_hash
from .discovery_model import CodexMultimodal
from .graph_policy import Requests
from .arm6_corner_pull import SCHEMA as ACTION_SCHEMA, execute_proposal
from .arm6_replay_media import ReplayMedia
from .replay_failures import capture_cli_failure, stop_after_failure, close_cli_runtime, cli_failed

SELECT = obj(decision=enum('ACT','DONE','STOP'),skill_id=S,confidence=N,
    assessment=S,subgoal=S,applicability_evidence=array(S),uncertainties=array(S),reason=S,
    previous_effect=enum('INITIAL','SUBGOAL_REACHED','PARTIAL','NO_CHANGE','WORSE','UNCERTAIN'),
    previous_effect_evidence=array(S),completion_evidence=array(S))
GROUND = obj(**ACTION_SCHEMA['properties'],skill_id=S,subgoal=S,
             grasp_role=S,target_role=S,adaptation_reason=S)
SELECT_PROMPT = '''根据当前照片选择一个有用的展开子目标和对应 skill_id。不是判断当前衣服匹配哪张旧图，
也不要求先认出整体布局；只要当前可见关系支持技能的适用条件即可。用衣物自身部位与布层关系推理。
先评估 previous_action 的子目标是否达成，结合 previous_before 与 current 图，说明实际变化。
移开覆盖层可能暂时把布料向内折，使后续可抓袖缘显露；不以轮廓面积或向外移动作为唯一进步指标。
结合最近几步避免来回翻同一层或反复无效抓取。不要把有运动就说成成功。
ACT 表示提出待案例核对的候选技能，不是执行授权。必须选择 catalog 中一个技能并列出当前图的适用证据，说明这一步的子目标。
没有适用技能或抓取依据就 STOP，说明欠缺什么。主要衣身已打开、袖子和下摆不再明显回折覆盖时可 DONE；
允许小褶皱，需给出当前图像的完成证据。DONE/STOP 的 skill_id 为空。只输出 JSON。'''
GROUND_PROMPT = '''核对候选技能是否适用，再规划一个动作，服务于 selection.subgoal。查看 current 原图及给定案例的
完整前后片段；案例是局部关系与目的的经验，不是需要匹配的整件衣服状态。
selection 是尚未经过案例核对的候选，可能低置信度。逐项检查其中的不确定性：
部位名称不明确不必阻止基于可见布边的操作，但若技能依赖的覆盖关系或可抓边缘无法确认，则 STOP。
你的 confidence 应同时考虑当前技能适用性、可见抓取依据和具体点位可靠性，不能只评坐标清晰度，
也不能为了继续执行而提高分数。说明案例支持了什么、哪些疑问仍未解决。
按照当前实际可见布层选择抓取对象和让开/展开的目标位置，允许有目的的向内翻动。
说明案例与当前的关键差异、为何仍适用，以及当前抓点/落点分别承担什么角色。若不适用则 STOP。
不要照搬案例像素、夹爪角度或隐藏部位；输出 CURRENT 原图像素坐标。夹爪开合方向以图像向右0度、
顺时针模180度。抓取、抬升、平移、放下由现有执行器完成，一次只做一个动作。
遵守 observation_context 的物理位移范围，程序会检查实测深度和完整 IK。
历史案例中的多步操作只用来理解目的，不能一次输出多次抓取。若无法可靠定位可抓边缘就 STOP。
STOP 点位填[0,0]。保留 skill_id，expected_change 要写可从下一张照片检验的子目标。'''

BASELINE_SELECT_PROMPT = '''这是用户明确要求的无置信度门槛摸底实验，目标是通过尝试把衣服展开。
根据当前照片选一个最有希望的子目标和 catalog 技能作为经验参考，不要求当前布层关系与案例完全一致。
看不清领口、袖口、夹持层数，或不能保证会展开，都可以记录为不确定性，同时尝试操作清楚可见的布边。
ACT 选择一个已有 skill_id，说明可见依据、子目标和尝试理由；如有上一步，结合前后图如实评估其效果，
避免重复无效动作。允许为了露出下层而暂时向内翻动。成功与否由后续照片验证，不需提前证明。
主要衣身、袖子和下摆已展开时 DONE，给出可见证据。仅在没有可定位的布边、图像不可用或无法提出
任何具体操作时 STOP。DONE/STOP 的 skill_id 为空。confidence 如实填写，仅供记录，不决定执行。'''

BASELINE_GROUND_PROMPT = '''这是用户明确要求的无置信度门槛摸底实验。根据 current 原图，结合所选技能
和案例，尽力提出一次最有希望的抓取、抬升、移动、放下动作，以展开衣物或露出更多结构。
案例是经验参考，不要求证明当前衣物与案例相同，也不要求先确认隐藏布层、领口或夹持层数。
对这些不确定性如实说明；只要能定位可见布边，就可以选择一步尝试，效果由下一张照片验证。
允许向内翻开覆盖层。输出 CURRENT 原图上的可见抓点、落点、夹爪开合方向（向右0度、顺时针模180度）。
不照搬案例坐标，不猜不可见抓点。遵守 observation_context 给出的位移范围；程序负责实测深度和 IK。
只规划一次抓放，说明抓点/落点角色、案例差异及下一张照片应该检查什么。
只有图像不可用、没有可定位的布边或无法提供具体操作时才 STOP，STOP 点位填[0,0]。
保留所选 skill_id。confidence 如实填写，不需要提高分数，低置信度动作也可作为本次尝试。'''


def make_requests(output,args,entrypoint,backend=None):
    baseline=getattr(args,'confidence_threshold',.8)==0
    return Requests(output/'model',backend or CodexMultimodal(args.model,args.timeout_s),max_calls=args.max_calls,
        schemas={'select_skill':SELECT,'ground_skill':GROUND},
        prompts={'select_skill':BASELINE_SELECT_PROMPT if baseline else SELECT_PROMPT,
                 'ground_skill':BASELINE_GROUND_PROMPT if baseline else GROUND_PROMPT},
        common='Arm6 visual skill policy. Follow this skill entrypoint; return schema-valid JSON.\n'+entrypoint,
        protocol=PROTOCOL+'-policy')


def select_examples(library,skill,max_images=8):
    """Load complete short sequences, never a truncated success-only thumbnail."""
    cases=library['cases'];chosen=[];used=set();count=0
    def order(cid):return (-cases[cid]['confidence'],len(cases[cid]['images']),cid)
    # One positive first, one counterexample if available, then a second positive.
    supports=sorted(skill['support_case_ids'],key=order)
    counters=sorted(skill['counterexample_case_ids'],key=order)
    for candidates in (supports,counters,supports):
        for cid in candidates:
            case=cases[cid]
            if cid not in used and count+len(case['images'])<=max_images:
                chosen.append(case);used.add(cid);count+=len(case['images']);break
    if not chosen:raise ValueError('no complete skill example fits image budget')
    return chosen


def validate_selection(value,library):
    skills={s['skill_id'] for s in library['skills']}
    if value['decision']=='ACT':
        if value['skill_id'] not in skills:raise ValueError('unknown skill')
        if not value['applicability_evidence'] or not value['subgoal'].strip():raise ValueError('missing current applicability evidence')
    elif value['skill_id']:raise ValueError('DONE/STOP must not select a skill')
    if value['decision']=='DONE' and not value['completion_evidence']:raise ValueError('DONE requires current visual evidence')


def assess(requests,library,current,folder,previous=None,history=()):
    images=[('current',Path(current))]
    if previous:images.append(('previous_before',Path(previous['before'])))
    payload=dict(goal='展开衣身、袖子和下摆，允许普通小褶皱',
        catalog=[{k:v for k,v in s.items() if k not in ('support_case_ids','counterexample_case_ids')} for s in library['skills']],
        previous_action=previous['proposal'] if previous else None,recent_history=list(history)[-4:])
    def validate(value):
        validate_selection(value,library)
        if previous and value['previous_effect']=='INITIAL':raise ValueError('must evaluate previous action subgoal')
        if not previous and value['previous_effect']!='INITIAL':raise ValueError('no previous action to evaluate')
        if previous and value['previous_effect']!='UNCERTAIN' and not value['previous_effect_evidence']:
            raise ValueError('previous action assessment requires image evidence')
    selection=requests.ask('select_skill',payload,images,validator=validate)
    save(Path(folder)/'selection.json',selection)
    return selection


def ground(requests,library,package,current,selection,context,folder):
    skill=next(s for s in library['skills'] if s['skill_id']==selection['skill_id'])
    examples=select_examples(library,skill);images=[('current',Path(current))]
    for case in examples:
        for ref in case['images']:images.append((case['case_id']+'_'+ref['step_id'],Path(package)/ref['path']))
    # Drop absolute source paths and historical pixels from the model payload.
    example_payload=[{k:v for k,v in case.items() if k not in ('images','provenance')} for case in examples]
    def validate(value):
        if value['skill_id']!=selection['skill_id']:raise ValueError('grounding changed selected skill')
        if value['decision']=='ACT':
            if not all(value[k].strip() for k in ('visible_corner_evidence','grasp_role','target_role','expected_change')):
                raise ValueError('action needs visible grasp and subgoal evidence')
            with Image.open(current) as im:w,h=im.size
            for key in ('grasp_px','target_px'):
                x,y=value[key]
                if not 2<=x<w-2 or not 2<=y<h-2:raise ValueError('grounding point outside current image')
    proposal=requests.ask('ground_skill',dict(selection=selection,skill=skill,cases=example_payload,
                         observation_context=context),images,validator=validate)
    save(Path(folder)/'grounding.json',dict(proposal=proposal,example_case_ids=[c['case_id'] for c in examples]))
    return proposal


def selection_stop(selection,threshold):
    # ACT is a retrieval hypothesis. Grounding sees the examples and remains the
    # confidence gate for dispatch; a tentative catalog choice is not a motion.
    if selection['decision']=='DONE':
        return 'MODEL_REPORTED_UNFOLDED' if selection['confidence']>=threshold else 'SKILL_COMPLETION_UNCERTAIN'
    if selection['decision']=='STOP':return 'NO_APPLICABLE_SKILL'
    return None


def motion_context(observation,args):
    from cloth_agent.cloth_smoothing import observation_context
    context=observation_context(observation,args.grasp_offset_mm)
    context['configured_motion']=dict(grasp_offset_mm=args.grasp_offset_mm,lift_height_mm=50.,
        place_offset_mm=5.,min_translation_mm=10.,max_translation_mm=args.max_pull_mm)
    context['host_checks']=[c for c in context['host_checks'] if c!='height difference']
    return context


def preview(image,proposal,destination):
    with Image.open(image) as im:canvas=im.convert('RGB')
    draw=ImageDraw.Draw(canvas)
    if proposal['decision']=='ACT':
        a=tuple(proposal['grasp_px']);b=tuple(proposal['target_px'])
        draw.line((a,b),fill='#ffcf40',width=4)
        for p,c,label in ((a,'#ff4f58','GRASP'),(b,'#22e8e2','TARGET')):
            x,y=p;draw.ellipse((x-8,y-8,x+8,y+8),outline=c,width=3)
            draw.text((x+10,y),label,fill=c,stroke_width=1,stroke_fill='black')
        angle=np.deg2rad(proposal['opening_angle_image_deg']);d=24*np.array([np.cos(angle),np.sin(angle)])
        draw.line((tuple(np.array(a)-d),tuple(np.array(a)+d)),fill='white',width=3)
    canvas.save(destination)


def loop(runtime,requests,library,package,output,media,result,args):
    before=runtime.observe(output/'initial_observation');previous=None;history=[];stalls=0
    for index in range(args.max_steps+1):
        if runtime.cancel.is_set():raise KeyboardInterrupt('skill experiment cancelled')
        step=output/f'step_{index:04d}';step.mkdir()
        media.begin(index,before.image);media.phase('skill_selection_and_feedback')
        selection=assess(requests,library,before.image,step,previous,history)
        result['assessments'].append(selection);result['last_observation']=str(before.image)
        print('ARM6 SKILL '+json.dumps(selection,ensure_ascii=False),flush=True)
        if runtime.cancel.is_set():raise KeyboardInterrupt('cancelled after skill selection')
        status=selection_stop(selection,args.confidence_threshold)
        if status:
            result['status']=status;media.snapshot('after',before.image);break
        if index==args.max_steps:
            result['status']='STEP_LIMIT';media.snapshot('after',before.image);break
        stalls=stalls+1 if previous and selection['previous_effect'] in ('NO_CHANGE','WORSE','UNCERTAIN') else 0
        if stalls>=args.max_stalled_steps:
            result['status']='SUBGOAL_STALLED';media.snapshot('after',before.image);break
        media.phase('skill_grounding')
        if selection['confidence']<args.confidence_threshold:
            print('候选技能尚不确定：先核对案例及当前抓取依据，再判断是否执行。',flush=True)
        proposal=ground(requests,library,package,before.image,selection,motion_context(before,args),step)
        print(f'技能点位核对：{proposal["decision"]}，最终动作置信度 {proposal["confidence"]:.2f}；'
              f'{proposal["reason"]}',flush=True)
        preview(before.image,proposal,step/'proposal_preview.png')
        if runtime.cancel.is_set():raise KeyboardInterrupt('cancelled after skill grounding')
        status,after,new_previous=execute_proposal(runtime,before,proposal,step/'action',media,result,
            kind='skill',threshold=args.confidence_threshold,max_distance_mm=args.max_pull_mm,
            grasp_offset_mm=args.grasp_offset_mm)
        if status!='EXECUTED':
            result['status']=status;media.snapshot('after',before.image);break
        history.append(dict(skill_id=selection['skill_id'],subgoal=selection['subgoal'],
                            expected_change=proposal['expected_change'],previous_effect=selection['previous_effect']))
        previous=new_previous;before=after;result['last_observation']=str(before.image)
        save(output/'result.json',dict(result,status='OBSERVING_SUBGOAL_EFFECT'))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--image',type=Path,help='offline selection and pixel plan only; no robot/depth/IK execution')
    parser.add_argument('--config',type=Path,default=Path('config/dual_arm.local.json'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-steps',type=int,default=12)
    parser.add_argument('--max-stalled-steps',type=int,default=2)
    parser.add_argument('--max-pull-mm',type=float,default=150.)
    parser.add_argument('--grasp-offset-mm',type=float,default=-15.)
    parser.add_argument('--confidence-threshold',type=float,default=.8,
                        help='minimum confidence for grounded motion and completion; 0 disables confidence gating')
    parser.add_argument('--max-calls',type=int,default=52)
    parser.add_argument('--model',default='gpt-6-astra')
    parser.add_argument('--timeout-s',type=float,default=180.)
    args=parser.parse_args(argv)
    if (not 1<=args.max_steps<=30 or not 1<=args.max_stalled_steps<=10 or not 10<=args.max_pull_mm<=150
        or not -30<=args.grasp_offset_mm<=10 or not 0<=args.confidence_threshold<=1 or args.max_calls<1
        or not np.isfinite(args.timeout_s) or args.timeout_s<=0):parser.error('invalid experiment limits')
    if args.image and not args.image.is_file():parser.error('image does not exist')
    # Validate before even creating a robot runtime.
    library=load_library(args.library);package=args.library.resolve().parent
    output=(args.output or Path('reports')/('arm6_skill_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    output.mkdir(parents=True,exist_ok=False)
    result=dict(status='PREPARING',experiment='experience_guided_skill',robot='Arm6',physical_actions=0,
        confidence_gating_enabled=args.confidence_threshold>0,
        planning_mode='best_effort_baseline' if args.confidence_threshold==0 else 'evidence_checked',
        skill_actions=0,steps=[],assessments=[],library=str(args.library.resolve()),library_sha256=file_hash(args.library),
        parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    save(output/'result.json',result)
    if args.confidence_threshold==0:
        print('摸底测试：置信度门槛已关闭，仍记录模型原始置信度。',flush=True)
    runtime=None;requests=None;media=ReplayMedia(output,title='Arm6 skill policy')
    prior=signal.getsignal(signal.SIGTERM)
    def interrupted(*_):raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        requests=make_requests(output,args,(package/library['entrypoint']['path']).read_text())
        if args.image:
            with Image.open(args.image) as im:im.convert('RGB').save(output/'input.png');size=list(im.size)
            selection=assess(requests,library,output/'input.png',output)
            result['assessments'].append(selection)
            status=selection_stop(selection,args.confidence_threshold)
            if status:result['proposed_next']=status
            else:
                proposal=ground(requests,library,package,output/'input.png',selection,
                    dict(image_size_px=size,measured_depth_available=False,
                         note='Offline RGB only: metric displacement, depth and IK are NOT validated.'),output)
                preview(output/'input.png',proposal,output/'proposal_preview.png')
                result['proposal']=proposal
                result['proposed_next']='PIXEL_PLAN_ONLY' if proposal['decision']=='ACT' and proposal['confidence']>=args.confidence_threshold else 'GROUNDING_STOP_OR_UNCERTAIN'
            result.update(status='OFFLINE_PLAN_ONLY',robot_calls=0)
        else:
            from cloth_agent.gamepad_collect_runtime import CollectionRuntime
            runtime=CollectionRuntime(args.config);runtime.config.home_recover_stopped=True
            loop(runtime,requests,library,package,output,media,result,args)
    except (Exception,KeyboardInterrupt) as exc:
        result.update(capture_cli_failure(output,exc))
        if runtime is not None:runtime.cancel.set();stop_after_failure(runtime,result)
    finally:
        signal.signal(signal.SIGTERM,prior);close_cli_runtime(runtime,result)
        if requests is not None:result['model_metrics']=requests.metrics()
        try:
            video=media.finish(result['status']);result['timing_file']=str(output/'timing.json')
            if video:result['video']=video
        except Exception as exc:result['media_error']=str(exc)
        save(output/'result.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 1 if cli_failed(result) else 0


if __name__=='__main__':raise SystemExit(main())
