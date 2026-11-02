"""Existing Arm6 full episode plus non-routing post-action reason analysis."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
from datetime import datetime
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
_spec=importlib.util.spec_from_file_location('live_analysis_io',Path(__file__).with_name('legacy_skill_loop.py'))
io=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(io)
OBSERVATION_PROMPT='''你只负责第一层：记录这次操作前后可观察到的物理变化。第一张Before，第二张After。区域以Before图像方位为准，尽量用印花、缝线或轮廓等可见标志追踪同一布片；无法可靠对应就写UNKNOWN。
分别记录左上区域、中央衣身、右侧布片、下摆及新出现的形变。描述位置/朝向变化、边缘轮廓、可见折线、遮挡关系；观察不到的区域标为未知，不强行识别衣物部位。相对稳定也必须有可比的图像证据。
区分整件衣服平移旋转与局部相对变化；未经配准或标定不要编造精确位移。不把阴影变化直接等同层数变化。只描述前后图，不推测中途材料运动、抓空、摩擦或传力原因。
不判断成功、失败、收益、目标完成或动作优劣；不生成Skill或下一动作。'''
REGION_SCHEMA=io.obj({'before':io.S,'after':io.S,'visible_change':io.S,
    'correspondence':io.enum('CONFIRMED','UNKNOWN'),'image_evidence':io.S,'unknowns':io.S})
OBSERVATION_SCHEMA=io.obj({'regions':io.obj({k:REGION_SCHEMA for k in
    ['upper_left','central_body','right_panel','lower_hem','new_deformation']}),
    'whole_garment_motion':io.S,'local_relative_changes':io.S,'observation_limits':io.S})
ANALYSIS_PROMPT='''你只负责第二层：这些变化意味着什么？输入第一张Before、第二张After、已经独立保存的第一层观察，以及原计划理由和实际动作。原计划理由只是预期，不能当成实际结果。若原定目标不够明确，目标完成项必须UNKNOWN。
独立判断三项：原定目标是否完成、是否存在可确认的局部进展、是否存在可确认的不利变化。每项输出YES/NO/UNKNOWN，并引用第一层的区域键和具体图像证据。另列尚不可判断的结果。
目标仍然存在或尚未完成，不代表没有局部进展。target_completed=NO可以与local_progress=YES同时成立；进展和不利变化也可同时存在。不要将三项合并为整体成功/失败分数。
NO需要清晰可比的证据；没有确认YES的证据不等于NO。遮挡、布片对应不明或仅见整体位姿变化且无法判断局部状态时应UNKNOWN。UNKNOWN不代表失败、零收益或无进展，不得转换成这些标签。
第一层与图像不一致时明确指出，相关判断保留UNKNOWN，不擅自改写第一层。区分观察与解释，不推测不可见物理机制。不生成Skill或下一动作。'''
JUDGMENT_SCHEMA=io.obj({'label':io.enum('YES','NO','UNKNOWN'),'evidence':io.S,
    'observation_keys':{'type':'array','items':io.enum('upper_left','central_body','right_panel','lower_hem','new_deformation','whole_garment_motion','local_relative_changes')},
    'uncertainty':io.S})
ANALYSIS_SCHEMA=io.obj({'original_target':io.S,'target_completed':JUDGMENT_SCHEMA,
    'local_progress':JUDGMENT_SCHEMA,'adverse_change':JUDGMENT_SCHEMA,
    'undetermined_results':{'type':'array','items':io.S},'observation_conflicts':io.S})
REGION_NAMES={'upper_left':'左上区域','central_body':'中央衣身','right_panel':'右侧布片',
    'lower_hem':'下摆','new_deformation':'新的形变'}

def observation_markdown(result):
    def cell(value):return str(value).replace('|',chr(92)+'|').replace('\n','<br>')
    lines=['第一层：实际发生了什么？','',
        '| 观察维度 | 可见变化 | 对应关系 | 图像证据与未知项 |',
        '|---|---|---|---|']
    for key,name in REGION_NAMES.items():
        row=result['regions'][key]
        lines.append('| '+' | '.join(map(cell,[name,row['visible_change'],row['correspondence'],row['image_evidence']+'；'+row['unknowns']]))+' |')
    lines+=['','整体位姿变化：'+result['whole_garment_motion'],
        '','局部相对变化：'+result['local_relative_changes'],'','观察限制：'+result['observation_limits']]
    return '\n'.join(lines)+'\n'

def analyze_round(folder):
    execution=io.read(folder/'execution.json')
    if execution.get('status')!='EXECUTED':return
    start=__import__('time').perf_counter();stage='observation'
    try:
        images=[folder/'before/rgb.png',folder/'after/rgb.png']
        # Blind to planner rationale and desired outcome in the observation layer.
        observed=io.request(folder/'physical_observation',OBSERVATION_PROMPT,
            '仅比较这两张原始图像，按区域记录可见物理变化。',images,OBSERVATION_SCHEMA)
        io.save(folder/'physical_observation.json',dict(observed,source='model_visual_observation',
            human_verified=False,layer=1,planner_intent_supplied=False))
        (folder/'physical_observation.md').write_text(observation_markdown(observed))
        stage='interpretation'
        details={'original_plan':io.read(folder/'plan.json'),'execution':execution,
            'layer_1_observation':observed}
        result=io.request(folder/'reason_analysis',ANALYSIS_PROMPT,
            json.dumps(details,ensure_ascii=False),images,ANALYSIS_SCHEMA)
        io.save(folder/'reason_analysis.json',dict(result,source='model_visual_interpretation',
            human_verified=False,layer=2,planner_feedback=False,
            layer_1_sha256=io.sha(folder/'physical_observation.json'),
            label_semantics={'UNKNOWN':'unresolved; neither failure nor zero benefit',
                'dimensions_independent':True,'aggregate_success_label':False},
            elapsed_s=__import__('time').perf_counter()-start))
        lines=['第二层：这些变化意味着什么？','','原定目标：'+result['original_target']]
        for key,name in [('target_completed','目标完成'),('local_progress','可确认的局部进展'),('adverse_change','可确认的不利变化')]:
            item=result[key];lines+=['',name+'：'+item['label'],'证据：'+item['evidence'],'未知项：'+item['uncertainty']]
        lines+=['','尚不可判断：'+'；'.join(result['undetermined_results']),
            '','观察冲突：'+result['observation_conflicts'],'','UNKNOWN不得作为失败或零收益标签。']
        (folder/'reason_analysis.md').write_text('\n'.join(lines)+'\n')
        print('分层反馈：目标完成='+result['target_completed']['label']+
            '；局部进展='+result['local_progress']['label']+'；不利变化='+result['adverse_change']['label'],flush=True)
    except Exception as exc:
        io.save(folder/'reason_analysis_error.json',{'stage':stage,'error':f'{type(exc).__name__}: {exc}',
            'planner_feedback':False,'affects_execution':False,'missing_labels_are_unknown':True})
        print('分层反馈未完成，原rollout继续：'+str(exc),flush=True)

def execute(args):
    from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode
    original_runtime=episode.EpisodeCollectionRuntime
    original_validate=episode.planner.validate_plan
    output=args.output.resolve()
    def relaxed_reason(plan,size,**kwargs):
        kwargs['require_single_sentence']=False
        return original_validate(plan,size,**kwargs)
    class AnalysisRuntime(original_runtime):
        def capture_current(self,directory,*pos,**kw):
            observation=super().capture_current(directory,*pos,**kw)
            path=Path(directory)
            if path.name=='after' and path.parent.name.startswith('round_'):
                analyze_round(path.parent)
            return observation
    argv=['one_shot_goal_full_episode_v1','--execute','--output',str(output),
        '--config',str(args.config),'--max-rounds',str(args.max_rounds),'--single-view',
        '--same-prompt-retry','--retry-plan-feedback']
    resume=getattr(args,'resume_from',None)
    reset_radius_mm=None
    if not resume:
        import random
        reset_radius_mm=random.SystemRandom().uniform(20.,40.)
        argv+=['--initialize-center-radius-mm',str(reset_radius_mm)]
    if resume:
        resume=resume.resolve()
        argv+=['--no-reset','--goal-image',str(resume/'goal/target.png')]
    elif args.goal_image:argv+=['--goal-image',str(args.goal_image)]
    if args.max_rounds==0:
        i=argv.index('--max-rounds');del argv[i:i+2]
        argv+=['--unlimited-rounds']
    # Remove only the mistakenly introduced pull-distance instruction on continuation.
    prompt=episode.SINGLE_VIEW_PROMPT
    legacy_suffix='\n本实验Pa到Pb的桌面XY牵拉距离限定为20–40毫米，由Host按标定坐标核验。'
    prompt_correction=False
    if resume:
        source_prompt=(resume/'prompt.txt').read_text()
        if source_prompt not in (prompt,prompt+legacy_suffix):
            raise ValueError('续跑Prompt与来源记录不一致；未连接机器人')
        prompt_correction=source_prompt!=prompt
    try:
        with patch.object(episode,'EpisodeCollectionRuntime',AnalysisRuntime), \
             patch.object(episode.planner,'validate_plan',relaxed_reason), \
             patch.object(episode,'SINGLE_VIEW_PROMPT',prompt),patch.object(sys,'argv',argv):
            return episode.main()
    finally:
        if output.exists():
            io.save(output/'reason_analysis_protocol.json',{'runner':'one_shot_goal_full_episode_v1',
                'argv':argv,'max_rounds':args.max_rounds or None,'single_view':True,'startup_reset':not bool(resume),
                'resume_from':str(resume) if resume else None,'continuation_kind':'current physical state; separate log; no analysis feedback',
                'pull_distance_xy_mm':[10,150],'removed_mistaken_pull_limit':True,
                'source_prompt_corrected':prompt_correction,'reset_radius_bound_range_mm':[20,40],
                'reset_radius_bound_mm':reset_radius_mm,'reset_sampling':'within disk; radius bound sampled uniformly per new episode',
                'analysis_feedback_to_planner':False,
                'skill_discovery':False,'feedback_layers':2,'feedback_calls_per_completed_action':2,
                'observation_prompt':OBSERVATION_PROMPT,'analysis_prompt':ANALYSIS_PROMPT,
                'analysis_error_policy':'record and continue; interrupts propagate'})
            (output/'reason_analysis_entrypoint.py').write_bytes(Path(__file__).read_bytes())

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--output',type=Path);p.add_argument('--config',type=Path,default=ROOT/'config/dual_arm.local.json')
    p.add_argument('--goal-image',type=Path);p.add_argument('--max-rounds',type=int,default=6,help='0 means no round cap')
    p.add_argument('--resume-from',type=Path,help='continue current cloth state, reuse source goal/prompt, no reset; save a separate log')
    args=p.parse_args(argv)
    if args.execute and args.dry_run:p.error('choose execute OR dry-run')
    if not 0<=args.max_rounds<=50:p.error('max-rounds must be 0 (unlimited) or 1–50')
    if args.resume_from:
        if args.goal_image:p.error('resume-from preserves the source goal; do not pass goal-image')
        for name in ['episode_summary.json','goal/target.png','prompt.txt']:
            if not (args.resume_from/name).is_file():p.error('resume source missing '+name)
    if not args.execute:
        print(json.dumps({'runner':'one_shot_goal_full_episode_v1','max_rounds':args.max_rounds or None,
            'resume_from':str(args.resume_from) if args.resume_from else None,
            'single_view':True,'startup_reset':not bool(args.resume_from),'pull_distance_xy_mm':[10,150],
            'reset_radius_bound_range_mm':[20,40],
            'skill_discovery':False,'feedback_layers':2,'feedback_calls_per_completed_action':2,
            'analysis_feedback_to_planner':False,'connected':False},indent=2));return 0
    args.output=args.output or ROOT/'reports'/('live_skill_discovery_01_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if args.output.exists():p.error('output already exists')
    return execute(args)
if __name__=='__main__':raise SystemExit(main())
