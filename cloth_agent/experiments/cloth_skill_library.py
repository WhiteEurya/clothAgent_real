"""Distill sequential Arm6 demonstrations into visual, goal-conditioned skills.

No graph node matching and no robot connection. Model-derived skills remain
provisional; each rule retains sequence images and source hashes for inspection.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil

import jsonschema
from .discovery_data import read, save, file_hash
from .discovery_schema import obj, array, S, N, enum
from .discovery_model import CodexMultimodal
from .graph_policy import Requests, evidence_file, frozen, load_frozen

PROTOCOL = 'arm6-visual-skills-v1'
CASE = obj(title=S, applicability=S, subgoal=S, grasp_relation=S, move_relation=S,
    expected_effect=S, completion_test=S, next_step=S, evidence_step_ids=array(S),
    visual_evidence=array(S), intent_basis=enum('SEQUENCE_INFERENCE','UNKNOWN'),
    outcome=enum('OBSERVED_SUBGOAL','POSSIBLE','COUNTEREXAMPLE'), limitations=array(S), confidence=N)
ANALYSIS = obj(links=array(obj(before_step=S, next_step=S,
    continuity=enum('CONTINUOUS','CHANGED','UNCERTAIN'), evidence=S)),
    cases=array(CASE), limitations=array(S))
SKILL = obj(skill_id=S, name=S, applicability=S, subgoal=S, grasp_relation=S,
    move_relation=S, completion_test=S, next_step=S, support_case_ids=array(S),
    counterexample_case_ids=array(S), limitations=array(S))
SYNTHESIS = obj(skills=array(SKILL), omitted_cases=array(obj(case_id=S, reason=S)))
GUIDANCE = '''目标是展开衣服，经验应解释动作目的和布层关系。用户明确说明：有些向内翻折是为了
移开覆盖层、显露藏在下面的袖子，再抓袖子展开。不能仅因内折、图案遮盖或面积暂时缩小就判失败；
也不能把所有内折都强行解释为暴露袖子。用实际图片验证新露出的可抓部位及后续是否利用它。
意图通常是从序列推断，必须区别于直接观察到的变化。旧 graph 的 PROGRESS 和保存确认都不是任务成功标签。
不要复制旧像素坐标；技能使用相对衣物部位、覆盖关系和子目标。'''
ANALYZE_PROMPT = GUIDANCE + '''\n逐张检查本窗口的动作前后图，按提供的原始时间顺序分析，不匹配 graph。
每幅图左侧为 before（红 start、青 target 是操作选择而非实际夹持证明），右侧为 after。
先比较相邻动作的 after 与下一动作 before，逐对填写 links；明显改变为 CHANGED，不确定为 UNCERTAIN。
连续编号不证明无人干预。next_iteration_before_unverified 表示可能存在人工干预，不能据此确认因果。
提取一到数步的有用片段，每个片段给出适用条件、子目标、抓取对象、移动关系、完成判断及下一步。
片段只能引用本窗口连续的 step IDs，不能跨 CHANGED/UNCERTAIN 边界推断连续效果。
可包括明确的无效尝试，但不能只因单步没变大就视为反例。OBSERVED_SUBGOAL 必须有图片证据支持具体
子目标（如露出可抓袖缘），而不是假定最终衣服会展开。无法确定时用 POSSIBLE，不凑成功案例。
缺少 after 不能产生 OBSERVED_SUBGOAL。不要把折叠演示直接转成展开方法。中文简洁输出。'''
SYNTHESIZE_PROMPT = GUIDANCE + '''\n将已逐图审查的片段归纳为少量可复用技能，按目的与适用条件合并，
保留不同动作变体和限制。你没有新图片，不得新增视觉事实。每个技能至少引用一个 OBSERVED_SUBGOAL、
confidence>=0.8 的支持案例；反例只能引用已标为 COUNTEREXAMPLE 的案例。POSSIBLE 不可变成成功证据。
技能可以暂时向内移动来为后续操作创造条件，不能规定所有抓取必须向外。
不泛化成固定屏幕方向、绝对像素坐标或未测量的高度。最多12个技能。skill_id 用小写英文和连字符。
未被引用的每个案例都列入 omitted_cases 并说明原因。没有可靠证据时允许空 skills。'''
ENTRYPOINT = '''---
name: arm6-cloth-unfold
description: Plan Arm6 cloth unfolding from current RGB using demonstrated subgoals, occlusion changes, and visual examples.
---

# Arm6 cloth unfolding experience

Use the current image to choose a useful subgoal. Whole-garment recognition and
matching a graph node are not prerequisites. Read `library.json` for the skill
catalog, then load only the chosen skill's case sheets under `examples/`.

An inward flip can expose a buried sleeve. Judge whether the intended region
became visible or graspable and whether the next action uses it; area increase
and movement direction alone do not measure progress. This is a possible
strategy, not a claim about every inward movement.

Treat inferred intent separately from visible effects. Cases are observational
examples, not proven success rates. Re-localize visible grasp/placement points
and gripper angle in the current image. Do not copy demonstration coordinates.
Execute one action, observe again, and evaluate its stated subgoal before choosing
the next. Respect uncertain layers and conflicting examples.

The application `cloth_agent.experiments.arm6_skill_policy` loads this entrypoint,
the catalog, and selected visual examples explicitly. Arm6 execution uses the
existing measured-depth and IK pipeline. Compilation and `--image` never move
hardware; physical operation requires the application's `--execute` flag.
'''


def verified(ref):
    if not ref or not Path(ref['path']).is_file() or file_hash(ref['path']) != ref['sha256']:
        raise ValueError(f'evidence changed or missing: {ref}')
    return Path(ref['path'])


def segments(trajectory, sessions=(), holdout_sessions=()):
    """Keep original sequence, split resets, missing/rejected iterations and runs."""
    available={s['session'] for s in trajectory['steps']}
    if (set(sessions)|set(holdout_sessions))-available:raise ValueError('unknown session filter')
    if set(sessions)&set(holdout_sessions):raise ValueError('training and holdout overlap')
    groups=[];last=None;last_manifest=None
    for step in sorted(trajectory['steps'],key=lambda s:(s['session'],s['iteration'])):
        if sessions and step['session'] not in sessions:continue
        if step['session'] in holdout_sessions:continue
        manifest=read(verified(step['commanded_action']['manifest']))
        same=(last is not None and step['session']==last['session'] and
              step['iteration']==last['iteration']+1 and step.get('episode_id')==last.get('episode_id') and
              manifest.get('preceding_reset')==last_manifest.get('preceding_reset'))
        if same:
            folder=Path(last['source_folder']);session=folder.parent.parent if folder.parent.name=='pending' else folder.parent
            if (session/'resets'/f'after_iter_{last["iteration"]:04d}').exists():same=False
        if not same:groups.append([])
        groups[-1].append(step);last=step;last_manifest=manifest
    return groups


def make_windows(groups,size=6,overlap=2):
    if not 2<=size<=8 or not 1<=overlap<size:raise ValueError('window size 2..8, overlap 1..size-1')
    windows=[]
    for group_index,group in enumerate(groups):
        start=0
        while start<len(group):
            part=group[start:start+size]
            windows.append(dict(window_id=f'w{len(windows):03d}',segment=group_index,
                                step_ids=[s['step_id'] for s in part]))
            if start+size>=len(group):break
            start+=size-overlap
    return windows


def validate_analysis(value,window,steps):
    jsonschema.validate(value,ANALYSIS)
    ids=window['step_ids'];expected=list(zip(ids,ids[1:]))
    pairs=[(x['before_step'],x['next_step']) for x in value['links']]
    if len(pairs)!=len(expected) or set(pairs)!=set(expected):raise ValueError('continuity must cover adjacent pairs exactly')
    links={(x['before_step'],x['next_step']):x['continuity'] for x in value['links']}
    for case in value['cases']:
        evidence=case['evidence_step_ids']
        if not evidence or any(s not in ids for s in evidence):raise ValueError('unknown or empty case evidence')
        start=ids.index(evidence[0])
        if evidence!=ids[start:start+len(evidence)]:raise ValueError('case must be a consecutive sequence')
        if any(links[p]!='CONTINUOUS' for p in zip(evidence,evidence[1:])):
            raise ValueError('case crosses an unverified visual continuity boundary')
        if case['outcome']!='POSSIBLE':
            if any(not steps[s]['after_observation'] for s in evidence):raise ValueError('observed outcome requires after images')
            if not case['visual_evidence'] or not case['completion_test'].strip():raise ValueError('observed outcome requires visible evidence')


def validate_synthesis(value,cases):
    jsonschema.validate(value,SYNTHESIS)
    ids=[s['skill_id'] for s in value['skills']]
    if len(ids)>12 or len(ids)!=len(set(ids)) or any(not re.fullmatch('[a-z][a-z0-9-]{0,63}',s) for s in ids):
        raise ValueError('invalid skill identifiers or count')
    used=set()
    for skill in value['skills']:
        if not skill['support_case_ids']:raise ValueError('skill requires supporting cases')
        for key,outcome in [('support_case_ids','OBSERVED_SUBGOAL'),('counterexample_case_ids','COUNTEREXAMPLE')]:
            if len(skill[key])!=len(set(skill[key])):raise ValueError('duplicate case reference')
            for cid in skill[key]:
                if cid not in cases or cases[cid]['outcome']!=outcome:raise ValueError('invalid evidence role')
                if key=='support_case_ids' and cases[cid]['confidence']<.8:raise ValueError('uncertain support')
                used.add(cid)
    omitted=[x['case_id'] for x in value['omitted_cases']]
    if len(omitted)!=len(set(omitted)) or set(omitted)&used or used|set(omitted)!=set(cases):
        raise ValueError('account for every case without unsupported references')


def prepare_archive(trajectory_path,output,*,sessions=(),holdout_sessions=(),size=6,overlap=2):
    from scripts.compile_arm6_archive_graph import action_sheet
    trajectory=load_frozen(trajectory_path)
    if trajectory.get('protocol')!='arm6-archive-experience-v1':raise ValueError('unsupported trajectory snapshot')
    groups=segments(trajectory,sessions,holdout_sessions)
    windows=make_windows(groups,size,overlap)
    if not windows:raise ValueError('no training trajectories')
    steps={s['step_id']:s for g in groups for s in g}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    sheets=output/'evidence';sheets.mkdir(exist_ok=True)
    for step in steps.values():
        # Verify all supplied evidence, including motion, before asking for conclusions.
        def visit(value):
            if isinstance(value,dict):
                if 'path' in value and 'sha256' in value:verified(value)
                else:
                    for v in value.values():visit(v)
            elif isinstance(value,list):
                for v in value:visit(v)
        visit(step['commanded_action'])
        for key in ('before_observation','after_observation'):
            if step[key]:visit(trajectory['observations'][step[key]])
        action_sheet(trajectory,step,sheets/f'{step["step_id"]}.jpg')
    manifest=dict(protocol=PROTOCOL,source=evidence_file(trajectory_path),
        sessions=sorted({s['session'] for s in steps.values()}),holdout_sessions=list(holdout_sessions),
        window_size=size,overlap=overlap,windows=windows,
        steps={sid:dict(source_folder=s['source_folder'],session=s['session'],iteration=s['iteration'],
                       episode_id=s.get('episode_id'),after_provenance=s['outcome']['after_provenance']) for sid,s in steps.items()})
    frozen(output/'dataset.json',manifest)
    return trajectory,steps,windows


def compile_library(trajectory_path,output,*,sessions=(),holdout_sessions=(),size=6,overlap=2,backend=None,
                    model='gpt-6-astra',timeout=360.,max_calls=128,prepare_only=False):
    output=Path(output)
    trajectory,steps,windows=prepare_archive(trajectory_path,output,sessions=sessions,
        holdout_sessions=holdout_sessions,size=size,overlap=overlap)
    if prepare_only:return dict(status='PREPARED',steps=len(steps),windows=len(windows),robot_calls=0)
    req=Requests(output/'model',backend or CodexMultimodal(model,timeout),max_calls=max_calls,
        schemas={'analyze':ANALYSIS,'synthesize':SYNTHESIS},
        prompts={'analyze':ANALYZE_PROMPT,'synthesize':SYNTHESIZE_PROMPT},
        common='Offline evidence-based cloth skill compilation. JSON only; no hardware commands.',protocol=PROTOCOL)
    cases={}
    try:
        for window in windows:
            payload=[]
            for sid in window['step_ids']:
                step=steps[sid];a=step['commanded_action']
                payload.append(dict(step_id=sid,iteration=step['iteration'],collection_mode=a['collection_mode'],
                    selected_points=a['selected_points'],after_provenance=step['outcome']['after_provenance'],
                    motion_sources=[verified(p['motion_source']).read_text() for p in a['automated_phases']
                                    if p['motion_source'] and p['phase']!='return_perception'],
                    handoff=a.get('handoff'),sampled_close_poses=a.get('sampled_close_poses',[])))
            print(f'分析 {window["window_id"]}: '+', '.join(window['step_ids']),flush=True)
            analysis=req.ask('analyze',dict(window=window,steps=payload),
                [(sid,output/'evidence'/f'{sid}.jpg') for sid in window['step_ids']],
                validator=lambda v:validate_analysis(v,window,steps))
            frozen(output/'analyses'/f'{window["window_id"]}.json',analysis)
            for i,case in enumerate(analysis['cases']):
                cid=f'{window["window_id"]}_c{i:02d}'
                cases[cid]=dict(case,case_id=cid,window_id=window['window_id'],
                    provenance=[dict(step_id=sid,session=steps[sid]['session'],iteration=steps[sid]['iteration'],
                        source_folder=steps[sid]['source_folder'],manifest=steps[sid]['commanded_action']['manifest'],
                        before=trajectory['observations'][steps[sid]['before_observation']]['rgb'],
                        after=trajectory['observations'][steps[sid]['after_observation']]['rgb'] if steps[sid]['after_observation'] else None,
                        after_provenance=steps[sid]['outcome']['after_provenance']) for sid in case['evidence_step_ids']])
        synthesis=req.ask('synthesize',dict(cases=list(cases.values())),[],validator=lambda v:validate_synthesis(v,cases))
        frozen(output/'synthesis.json',synthesis)
        package=output/'skill';package.mkdir(exist_ok=True);(package/'examples').mkdir(exist_ok=True)
        # Portable case sheets are independent of the original absolute paths.
        for case in cases.values():
            case['images']=[]
            for sid in case['evidence_step_ids']:
                relative=f'examples/{sid}.jpg';shutil.copyfile(output/'evidence'/f'{sid}.jpg',package/relative)
                case['images'].append(dict(path=relative,sha256=file_hash(package/relative),step_id=sid))
        (package/'SKILL.md').write_text(ENTRYPOINT)
        library=dict(protocol=PROTOCOL,status='PROVISIONAL_VISUAL_EXPERIENCE',
            entrypoint=dict(path='SKILL.md',sha256=file_hash(package/'SKILL.md')),
            source=evidence_file(trajectory_path),dataset=evidence_file(output/'dataset.json'),
            training_sessions=sorted({s['session'] for s in steps.values()}),holdout_sessions=list(holdout_sessions),
            skills=synthesis['skills'],cases=cases,omitted_cases=synthesis['omitted_cases'],
            limitations=['Observational subgoal evidence; not causal success rates.',
                        'Inferred intent is not operator annotation.',
                        'Visual continuity cannot exclude all unrecorded interventions.'])
        frozen(package/'library.json',library)
        save(output/'summary.json',dict(status='COMPILED' if library['skills'] else 'NO_SUPPORTED_SKILLS',
             trajectories=len(steps),windows=len(windows),cases=len(cases),skills=len(library['skills']),
             library=str((package/'library.json').resolve()),robot_calls=0))
        return read(output/'summary.json')
    finally:save(output/'model_metrics.json',req.metrics())


def load_library(path):
    path=Path(path).resolve();library=load_frozen(path)
    if library.get('protocol')!=PROTOCOL:raise ValueError('unsupported skill library')
    validate_synthesis(dict(skills=library['skills'],omitted_cases=library['omitted_cases']),library['cases'])
    if not library['skills']:raise ValueError('library has no supported skills')
    for case in library['cases'].values():jsonschema.validate({k:case[k] for k in CASE['properties']},CASE)
    for ref in [library['entrypoint']]+[i for c in library['cases'].values() for i in c['images']]:
        target=(path.parent/ref['path']).resolve()
        if not target.is_relative_to(path.parent) or not target.is_file() or file_hash(target)!=ref['sha256']:
            raise ValueError('skill package evidence missing, changed, or outside package')
    return library


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory',type=Path,default=Path('reports/arm6_archive_graph_20261019/trajectory.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--session',action='append',default=[],help='restrict training; repeatable')
    parser.add_argument('--holdout-session',action='append',default=[])
    parser.add_argument('--window-size',type=int,default=6)
    parser.add_argument('--overlap',type=int,default=2)
    parser.add_argument('--prepare-only',action='store_true',help='verify evidence and render windows without model calls')
    parser.add_argument('--model',default='gpt-6-astra')
    parser.add_argument('--timeout-s',type=float,default=360.)
    parser.add_argument('--max-calls',type=int,default=128)
    args=parser.parse_args(argv)
    if not math.isfinite(args.timeout_s) or args.timeout_s<=0 or args.max_calls<1:parser.error('positive timeout and call budget required')
    result=compile_library(args.trajectory.resolve(),args.output.resolve(),sessions=args.session,
        holdout_sessions=args.holdout_session,size=args.window_size,overlap=args.overlap,
        prepare_only=args.prepare_only,model=args.model,timeout=args.timeout_s,max_calls=args.max_calls)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
