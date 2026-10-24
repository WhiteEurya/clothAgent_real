"""Compile confirmed Arm6 recordings across sessions, offline and without dispatch.

Import a frozen evidence snapshot, visually partition ALL before/after RGBs, then
bind every original action to exactly one graph edge. Never bridge reset episodes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import copy
import csv
import json
from pathlib import Path
import sys
import subprocess
import html
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloth_agent.experiments.graph_policy import Requests, evidence_file, frozen, load_frozen, graph_validation
from cloth_agent.experiments.graph_policy_schema import NODE
from cloth_agent.experiments.discovery_schema import obj, array, S, enum
from cloth_agent.experiments.discovery_model import CodexMultimodal
from scripts.compile_arm6_traj_graph import rows, read

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
LIMITATIONS=[
    'Only operator-confirmed COMPLETE Arm6/Camera A recordings are included; reset, rejected, aborted and unconfirmed records are excluded.',
    'Dedicated after RGB is preferred. Legacy next-iteration before RGB is subsequent evidence only, with possible intervening manual changes.',
    'No transitions are invented across sessions, deleted/rejected iterations or random reset episodes.',
    'Visual state equivalence is a model judgement; minor wrinkles, translation and rotation alone do not define different states.',
    'Recorded actions remain one-shot evidence, not universally validated policies. This offline graph does not enable robot dispatch.',
    'Folding and unfolding demonstrations coexist. No final goal or terminal node is inferred from a lack of outgoing edges.',
]


def get_folders(session):
    result={}
    for p in sorted(list(session.glob('traj_*/traj.json'))+list(session.glob('pending/iter_*/traj.json'))):
        index=int(p.parent.name.rsplit('_',1)[1])
        if index in result:raise ValueError(f'duplicate iteration: {session}/{index}')
        result[index]=p.parent
    return result


def after_folder(folder, manifest, index, folders):
    if (folder/'after/rgb.png').is_file():return folder/'after','dedicated_after'
    following=folders.get(index+1)
    if following is None:return None,'missing_after'
    other=read(following/'traj.json')
    if (other.get('episode_id')!=manifest.get('episode_id') or
            other.get('preceding_reset')!=manifest.get('preceding_reset')):
        return None,'reset_episode_boundary'
    # Even legacy records without episode metadata cannot bridge a reset operation.
    session=folder.parent if folder.parent.name!='pending' else folder.parent.parent
    if (session/'resets'/f'after_iter_{index:04d}').exists():return None,'reset_boundary'
    if other.get('collection_mode')=='random_reset':return None,'reset_boundary'
    if not (following/'observation/rgb.png').is_file():return None,'missing_after'
    return following/'observation','next_iteration_before_unverified'


def import_archive(root):
    observations={};obs_paths={};steps=[];excluded=[];sessions=[]
    def add_observation(directory):
        directory=directory.resolve()
        if str(directory) in obs_paths:return obs_paths[str(directory)]
        meta=read(directory/'observation.json')
        evidence={k:evidence_file(directory/name) for k,name in
                  [('rgb','rgb.png'),('observation','observation.json'),('surface','world_xyz_mm.npy')]}
        if not all(evidence.values()):raise ValueError(f'incomplete RGB-D: {directory}')
        if meta.get('synthetic'):raise ValueError('synthetic observation')
        oid=f'o{len(observations):03d}'
        observations[oid]=dict(**evidence,captured_at=meta['captured_at'],arm='left',camera='A')
        obs_paths[str(directory)]=oid
        return oid
    for session in sorted(Path(root).iterdir()):
        if not session.is_dir():continue
        folders=get_folders(session);included=0
        for index,folder in sorted(folders.items()):
            manifest=read(folder/'traj.json');mode=manifest.get('collection_mode','assisted')
            reason=None
            if manifest.get('status')!='COMPLETE' or manifest.get('operator_accepted') is not True:
                reason='not_operator_confirmed_COMPLETE'
            elif manifest.get('arm')!='left' or manifest.get('camera')!='A':reason='not_Arm6_Camera_A'
            elif manifest.get('not_a_demonstration') or mode=='random_reset':reason='random_reset_not_demonstration'
            if reason:
                excluded.append(dict(folder=str(folder.resolve()),reason=reason,status=manifest.get('status'),mode=mode));continue
            try:
                states=rows(folder/'states.jsonl');events=rows(folder/'events.jsonl')
                if len(states)!=manifest['samples'] or not states or not np.isfinite([s['world_tcp_mm_deg'] for s in states]).all():
                    raise ValueError('invalid feedback or sample count')
                free=mode in ('manual_only','single_point_manual')
                automatic=mode=='automatic_pick_place'
                if mode not in ('assisted','automatic_pick_place','manual_only','single_point_manual'):
                    raise ValueError(f'unsupported collection mode: {mode}')
                phase_names=(['automatic_pick_place','return_perception'] if automatic else
                    (['approach'] if mode=='single_point_manual' else []) if free else
                    ['approach','transfer_place','return_perception'])
                phases=[]
                for phase in phase_names:
                    result=read(folder/phase/'result.json')
                    if result['status'] not in ('EXECUTED','ALREADY_AT_PERCEPTION'):raise ValueError(f'incomplete {phase}')
                    plan=evidence_file(folder/phase/'native_plan.json');motion=evidence_file(folder/phase/'motion.py')
                    if result['status']=='EXECUTED' and (not plan or not motion):raise ValueError(f'missing plan for {phase}')
                    phases.append(dict(phase=phase,result=result,actual_native_plan=plan,motion_source=motion))
                points=read(folder/'points.json') if (folder/'points.json').exists() else []
                if not free and (len(points)!=2 or {p['name'] for p in points}!={'start','target'}):
                    raise ValueError('missing start/target')
                action=dict(type='ARM6_AUTOMATIC_PICK_PLACE_DEMONSTRATION' if automatic else
                            'ARM6_FREE_MANUAL_DEMONSTRATION' if free else 'ARM6_ASSISTED_DEMONSTRATION',
                    arm='left',axis=6,units='world_mm_deg_and_gripper_pulse',collection_mode=mode,
                    selected_points=points,automated_phases=phases,
                    manual_input_trace=evidence_file(folder/'events.jsonl'),
                    measured_trajectory=evidence_file(folder/'states.jsonl'),
                    controller_events=evidence_file(folder/'controller/events.jsonl'),
                    manifest=evidence_file(folder/'traj.json'),parameters=manifest.get('parameters',{}))
                if not action['controller_events']:raise ValueError('missing controller log')
                if free or not automatic:
                    finish=[e for e in events if e['kind']==('manual_recording_finished' if free else 'manual_confirmed')]
                    if len(finish)!=1:raise ValueError('missing or duplicate manual completion')
                    action['manual_finish' if free else 'handoff']=finish[0]
                    action['sampled_close_poses']=[]
                    for e in events:
                        if e.get('intent',{}).get('gripper')=='close':
                            sample=min(states,key=lambda s:abs(s['t_s']-e['t_s']))
                            action['sampled_close_poses'].append(dict(button_time_s=e['t_s'],sample_time_s=sample['t_s'],
                                measured_tcp_mm_deg=sample['world_tcp_mm_deg'],gripper_pulse=sample['gripper_pulse']))
                after,provenance=after_folder(folder,manifest,index,folders)
                before_id=add_observation(folder/'observation');after_id=add_observation(after) if after else None
                sid=f's{len(steps):03d}'
                steps.append(dict(step_id=sid,session=session.name,iteration=index,episode_id=manifest.get('episode_id'),
                    source_folder=str(folder.resolve()),before_observation=before_id,after_observation=after_id,
                    commanded_action=action,actual_execution=dict(status='EXECUTED',physical_execution=True,operator_accepted=True),
                    telemetry=dict(samples=len(states),phases=dict(Counter(s['phase'] for s in states))),
                    outcome=dict(after_provenance=provenance,continuity_verified=provenance=='dedicated_after',
                                 physical_cloth_outcome='requires visual assessment')))
                included+=1
            except (ValueError,KeyError,OSError) as exc:
                excluded.append(dict(folder=str(folder.resolve()),status=manifest.get('status'),mode=mode,reason=f'invalid_evidence: {exc}'))
        reset_manifests=list(session.glob('resets/**/traj.json'))
        excluded.extend(dict(folder=str(p.parent.resolve()),reason='random_reset_not_demonstration',status=read(p).get('status'),mode='random_reset') for p in reset_manifests)
        if included:sessions.append(dict(session=session.name,included=included))
    used={s[k] for s in steps for k in ('before_observation','after_observation') if s[k]}
    observations={k:v for k,v in observations.items() if k in used}
    if not steps:raise ValueError('no confirmed usable trajectories')
    return dict(protocol='arm6-archive-experience-v1',snapshot_at=datetime.now(timezone.utc).isoformat(),
        goal='Organize recorded folding and unfolding transitions without imposing a new goal.',
        observations=observations,steps=steps,sessions=sessions,excluded_iterations=excluded,limitations=LIMITATIONS)


def font(size):return ImageFont.truetype(FONT,size)


def atlas(trajectory, output):
    output.mkdir(parents=True,exist_ok=True);images=[];items=list(trajectory['observations'].items())
    for start in range(0,len(items),12):
        page=Image.new('RGB',(1920,918),'#111b2a');draw=ImageDraw.Draw(page)
        for j,(oid,obs) in enumerate(items[start:start+12]):
            im=Image.open(obs['rgb']['path']).convert('RGB');im.thumbnail((480,270))
            x=(j%4)*480;y=(j//4)*306
            page.paste(im,(x,y+36));draw.text((x+10,y+3),oid,font=font(23),fill='white')
        path=output/f'atlas_{start//12:02d}.jpg';page.save(path,quality=95);images.append((path.stem,path))
    return images


def validate_partition(value, observations):
    nodes=value['nodes'];ids=[n['node_id'] for n in nodes]
    if len(set(ids))!=len(ids) or not nodes:raise ValueError('duplicate or empty nodes')
    assigned=[o for n in nodes for o in n['representative_observations']]
    if len(assigned)!=len(set(assigned)) or set(assigned)!=set(observations):
        raise ValueError('partition must assign every supplied observation exactly once')
    for n in nodes:
        if not n['representative_observations'] or not n['observation_condition']['positive_evidence']:
            raise ValueError('observed state requires image references and positive evidence')


def partition(trajectory,out,backend):
    path=out/'partition.json'
    if path.exists():
        result=read(path);validate_partition(result,trajectory['observations']);return result
    images=atlas(trajectory,out/'atlas')
    schema=obj(nodes=array(NODE),limitations=array(S))
    req=Requests(out/'state_model',backend,max_calls=2,schemas={'partition':schema},
        common='Offline Arm6 cloth-state image clustering. Chinese descriptions. Return schema-valid JSON. No robot commands.',
        prompts={'partition':'''Inspect EVERY labeled RGB tile on all attached contact sheets. Partition ALL observation IDs into visually equivalent garment states. This is image matching, not matching descriptions or time order. Each supplied image appears exactly once in representative_observations, including all images in large groups. Do not create one state per photo by default. Ignore rigid rotation/translation and small wrinkles. Merge when visible sleeve/hem/torso coverage, exposed regions, layered boundaries and accessible cloth regions are equivalent for choosing the next action. Do NOT merge substantial hidden-corner/overlap differences simply because all states are crumpled, or because the next action is similar. Use garment-relative orientation determined by collar and hem, never screen-left alone; uncertain orientation must be explicit. Folding and unfolding are mixed. No predefined state count or goal. Use short IDs n000,n001,... and concise Chinese task_context, conditions grounded in visible images, confidence and uncertainty. Do not infer invisible layers. Include uncertain images conservatively rather than invent matches. Input has original RGBs scaled uniformly; tile header is its observation ID. The host binds actions afterwards; no synthetic connections are allowed.'''},protocol='arm6-archive-partition-v1')
    result=req.ask('partition',dict(observation_ids=list(trajectory['observations']),count=len(trajectory['observations'])),
                   images,validator=lambda v:validate_partition(v,trajectory['observations']))
    frozen(path,result);frozen(out/'state_metrics.json',req.metrics());return result


def action_sheet(trajectory,step,destination):
    page=Image.new('RGB',(1280,420),'#111b2a');draw=ImageDraw.Draw(page)
    for j,key in enumerate(('before_observation','after_observation')):
        oid=step[key]
        if not oid:
            draw.text((650,100),'Missing after RGB',font=font(24),fill='orange');continue
        im=Image.open(trajectory['observations'][oid]['rgb']['path']).convert('RGB')
        if j==0:
            ink=ImageDraw.Draw(im)
            pts=step['commanded_action']['selected_points']
            for p in pts:
                x,y=p['pixel_xy'];color='#ff4f58' if p['name']=='start' else '#22e8e2'
                ink.ellipse((x-9,y-9,x+9,y+9),outline=color,width=4)
                ink.text((x+12,y-18),p['name'],font=font(22),fill=color,stroke_width=1,stroke_fill='black')
        im.thumbnail((640,360));page.paste(im,(640*j,50))
        draw.text((640*j+8,8),f'{step["step_id"]} {key}: {oid}',font=font(20),fill='white')
    page.save(destination,quality=95)


def action_semantics(trajectory,out,backend,batch_size=14):
    schema=obj(actions=array(obj(step_id=S,action_family=S,grasp_relation=S,target_relation=S,
        observed_outcome=S,seed_outcome=enum('NO_PROGRESS','PROGRESS','UNKNOWN'),limitations=array(S))))
    folder=out/'action_sheets';folder.mkdir(exist_ok=True)
    def run_batch(start):
        batch=trajectory['steps'][start:start+batch_size];path=out/f'actions_{start//batch_size:02d}.json'
        def validate(value):
            ids=[a['step_id'] for a in value['actions']]
            if len(ids)!=len(set(ids)) or set(ids)!={s['step_id'] for s in batch}:raise ValueError('action coverage mismatch')
        if path.exists():result=read(path);validate(result);return result['actions']
        images=[];payload=[]
        for step in batch:
            p=folder/f'{step["step_id"]}.jpg';action_sheet(trajectory,step,p);images.append((step['step_id'],p))
            a=step['commanded_action']
            payload.append(dict(step_id=step['step_id'],type=a['type'],points=a['selected_points'],
                after_provenance=step['outcome']['after_provenance'],
                close_pose_samples=a.get('sampled_close_poses',[]),handoff=a.get('handoff'),
                source_motion=[Path(p['motion_source']['path']).read_text() for p in a['automated_phases'] if p['motion_source'] and p['phase']!='return_perception']))
        req=Requests(out/f'action_model_{start//batch_size:02d}',backend,max_calls=2,schemas={'actions':schema},
            common='Offline visual descriptions of recorded Arm6 actions. Chinese concise text, schema-valid JSON, no robot commands.',
            prompts={'actions':'''For each supplied step inspect its BEFORE/AFTER RGB pair directly. Red start and cyan target show recorded user selections, not proof of actual cloth contact. Describe visible garment-relative grasp and target relations supported by markers and the recorded motion; never invent hidden cloth corners. For manual mode only one start may exist; do not invent a target, describe uncertainty. Body frame uses collar/hem, not fixed screen axes. Give a concise action family and observed cloth effect. NO_PROGRESS if only rigid displacement/minor wrinkles changed; PROGRESS only for a clear material unfolding/folding change; UNKNOWN when uncertain or no after. A next-iteration before photo is only subsequent evidence and possible intervention MUST be stated. Physical execution/accepted save is not proof of grasp or task success. Do not infer final goal. One result for each step, no dropped steps.'''},protocol='arm6-archive-actions-v1')
        result=req.ask('actions',dict(steps=payload),images,validator=validate)
        frozen(path,result);frozen(out/f'action_metrics_{start//batch_size:02d}.json',req.metrics())
        print(f'action batch {start//batch_size+1} complete ({len(batch)} trajectories)',flush=True)
        return result['actions']
    with ThreadPoolExecutor(max_workers=2) as pool:
        batches=list(pool.map(run_batch,range(0,len(trajectory['steps']),batch_size)))
    return {a['step_id']:a for batch in batches for a in batch}


def apply_merge_review(trajectory, initial, candidates, review):
    """Apply only confident, disjoint, image-reviewed pairs; retain all evidence."""
    validate_partition(initial,trajectory['observations'])
    nodes={n['node_id']:copy.deepcopy(n) for n in initial['nodes']}
    pairs={p['pair_id']:p for p in candidates}
    decisions={d['pair_id']:d for d in review['decisions']}
    if len(decisions)!=len(review['decisions']) or set(decisions)!=set(pairs):
        raise ValueError('merge review coverage mismatch')
    seen=set();merges=[]
    for pid,candidate in pairs.items():
        ids=candidate['nodes']
        if len(ids)!=2 or len(set(ids))!=2 or seen.intersection(ids) or set(ids)-set(nodes):
            raise ValueError('merge candidates must be disjoint existing pairs')
        seen.update(ids)
        for nid in ids:
            if candidate['observation_members'][nid]!=nodes[nid]['representative_observations']:
                raise ValueError('merge review evidence mismatch')
        decision=decisions[pid]
        if decision['decision']!='MERGE' or decision['confidence']<.85:continue
        if not decision['positive_evidence']:raise ValueError('merge requires shared visible evidence')
        left,right=ids
        nodes[left].update(task_context=decision['task_context'],confidence=decision['confidence'],
            representative_observations=nodes[left]['representative_observations']+nodes[right]['representative_observations'],
            observation_condition={k:decision[k] for k in ('positive_evidence','negative_evidence','uncertainty_conditions')})
        del nodes[right]
        merges.append(dict(pair_id=pid,source_nodes=ids,target_node=left,reason=decision['reason']))
    result=dict(nodes=list(nodes.values()),limitations=initial['limitations']+
                ['A targeted RGB review consolidated confidently equivalent candidate pairs; all member images are retained.'])
    validate_partition(result,trajectory['observations'])
    return result,merges


def build_graph(trajectory,partition_result,semantics):
    validate_partition(partition_result,trajectory['observations'])
    nodes=copy.deepcopy(partition_result['nodes']);mapping={o:n['node_id'] for n in nodes for o in n['representative_observations']}
    edges=[]
    for step in trajectory['steps']:
        sid=step['step_id'];target=mapping.get(step['after_observation']);a=semantics[sid]
        if target is None:
            target='unknown_'+sid
            nodes.append(dict(node_id=target,task_context='缺少动作后照片',representative_observations=[],confidence=0,
                observation_condition=dict(positive_evidence=[],negative_evidence=[],uncertainty_conditions=['未观测结果，不用于状态匹配或自动执行'])))
        edges.append(dict(edge_id='e'+sid[1:],source_node=mapping[step['before_observation']],target_node=target,
            action_family=a['action_family'],action_template=dict(grasp_relation=a['grasp_relation'],target_relation=a['target_relation'],
                grasp_z_rule='原始深度/偏移和实测轨迹见 seed_action；重放需重新采样当前深度，不直接复制绝对 Z。',
                yaw_rule='保留原始夹爪角度及朝向记录；重放需验证当前图像方向与 IK。',other_constraints=a['limitations']),
            seed_step_id=sid,seed_action=dict(exact_command=copy.deepcopy(step['commanded_action']),source_observation=step['before_observation']),
            observed_outcome=a['observed_outcome'],outcome_evidence=[sid,step['before_observation']]+([step['after_observation']] if step['after_observation'] else []),
            seed_outcome=a['seed_outcome'] if step['after_observation'] else 'UNKNOWN',generalization_status='ONE_SHOT_HYPOTHESIS'))
    graph=dict(graph_id='arm6_archive_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),nodes=nodes,edges=edges,
        limitations=LIMITATIONS+partition_result['limitations'])
    graph_validation(graph,trajectory)
    return graph


def render(graph,trajectory,out):
    out.mkdir(exist_ok=True);node_dir=out/'nodes';node_dir.mkdir(exist_ok=True)
    step_map={s['step_id']:s for s in trajectory['steps']}
    with (out/'edges.csv').open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['edge','source','target','session','iteration','mode','after_evidence','outcome','action','trajectory'])
        for edge in graph['edges']:
            step=step_map[edge['seed_step_id']]
            writer.writerow([edge['edge_id'],edge['source_node'],edge['target_node'],step['session'],step['iteration'],
                step['commanded_action']['collection_mode'],step['outcome']['after_provenance'],
                edge['seed_outcome'],edge['action_family'],step['source_folder']])
    lines=['digraph G { rankdir=LR; bgcolor="#101b2a"; nodesep=.3; ranksep=.7; size="60,60";',
           f'graph [fontname="Noto Sans CJK SC",fontcolor="white",fontsize=24,labelloc=t,label="Arm6 · {len(graph["nodes"])} 个状态 / {len(graph["edges"])} 条示范\\n同向连线按轨迹数量汇总；原始边完整保留在 graph.json"];',
           'node [shape=plain,fontname="Noto Sans CJK SC",fontcolor="white"];',
           'edge [color="#76b9d4",fontcolor="#b9dcec",fontname="Noto Sans CJK SC",fontsize=10];']
    for node in graph['nodes']:
        nid=node['node_id'];refs=node['representative_observations'];im=Image.new('RGB',(320,250),'#203049');d=ImageDraw.Draw(im)
        d.text((9,5),f'{nid} · {len(refs)} 张',font=font(19),fill='white')
        if refs:
            photo=Image.open(trajectory['observations'][refs[0]]['rgb']['path']).convert('RGB');photo.thumbnail((320,180));im.paste(photo,(0,34))
        label=node['task_context'];d.text((8,216),label[:19],font=font(15),fill='#dbe8fc')
        path=node_dir/f'{nid}.png';im.save(path)
        lines.append(f'{json.dumps(nid)} [label=<<TABLE BORDER="0" CELLBORDER="0"><TR><TD><IMG SRC="{html.escape(str(path.resolve()))}"/></TD></TR></TABLE>>];')
    counts=Counter((e['source_node'],e['target_node']) for e in graph['edges'])
    for (a,b),count in counts.items():lines.append(f'{json.dumps(a)} -> {json.dumps(b)} [label="{count} traj",penwidth={min(5,1+count/3):.2f}];')
    lines.append('}');(out/'overview.dot').write_text('\n'.join(lines))
    for format in ('svg','png'):
        subprocess.run(['dot','-T'+format,str(out/'overview.dot'),'-o',str(out/('overview.'+format))],check=True)
    # Paginated node cards remain readable in VS Code even for a large cyclic graph.
    for start in range(0,len(graph['nodes']),12):
        page=Image.new('RGB',(1280,800),'#101b2a');d=ImageDraw.Draw(page)
        d.text((15,8),f'Arm6 状态图 · {len(graph["edges"])} 条轨迹 · 状态 {start+1}–{min(start+12,len(graph["nodes"]))}',font=font(22),fill='white')
        for j,n in enumerate(graph['nodes'][start:start+12]):page.paste(Image.open(node_dir/f'{n["node_id"]}.png'),((j%4)*320,50+(j//4)*250))
        page.save(out/f'states_{start//12+1:02d}.jpg',quality=95)
    # Per-state contact sheets expose ALL merged members, not just the representative.
    for n in graph['nodes']:
        refs=n['representative_observations']
        for start in range(0,len(refs),12):
            subset=dict(observations={oid:trajectory['observations'][oid] for oid in refs[start:start+12]})
            atlas(subset,out/'members'/n['node_id']/f'page_{start//12+1:02d}')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('data/demonstrations'))
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--import-only',action='store_true')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args(argv);out=args.output.resolve();out.mkdir(parents=True,exist_ok=args.resume)
    path=out/'trajectory.json'
    if args.resume and path.exists():trajectory=load_frozen(path)
    else:trajectory=import_archive(args.root);frozen(path,trajectory)
    audit=dict(trajectories=len(trajectory['steps']),observations=len(trajectory['observations']),sessions=trajectory['sessions'],
        modes=dict(Counter(s['commanded_action']['collection_mode'] for s in trajectory['steps'])),
        after_sources=dict(Counter(s['outcome']['after_provenance'] for s in trajectory['steps'])),
        excluded=trajectory['excluded_iterations'],robot_calls=0)
    (out/'import_summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k!='excluded'},ensure_ascii=False),flush=True)
    if args.import_only:
        atlas(trajectory,out/'atlas');return
    backend=CodexMultimodal(timeout=900)
    print('Visually clustering all observation photos...',flush=True)
    groups=partition(trajectory,out,backend)
    print(f'Visual partition: {len(groups["nodes"])} states. Describing recorded actions...',flush=True)
    semantics=action_semantics(trajectory,out,backend)
    graph=build_graph(trajectory,groups,semantics)
    graph.update(protocol='arm6-archive-graph-v1',robot='Arm6',camera='A',source_trajectory=evidence_file(path),terminal_node_ids=[])
    for edge in graph['edges']:
        edge.update(replayable=False,replay_block_reason='Offline evidence graph; requires action-template replay preflight and goal selection.')
    frozen(out/'graph.json',graph)
    render(graph,trajectory,out/'preview')
    metrics=[read(p) for p in sorted(out.glob('*metrics*.json'))]
    summary={**audit,'status':'COMPILED','nodes':len(graph['nodes']),'edges':len(graph['edges']),
        'self_loops':sum(e['source_node']==e['target_node'] for e in graph['edges']),
        'merged_states':sum(len(n['representative_observations'])>1 for n in graph['nodes']),
        'unobserved_sinks':sum(not n['representative_observations'] for n in graph['nodes']),
        'outcomes':dict(Counter(e['seed_outcome'] for e in graph['edges'])),
        'model_calls':sum(m.get('model_calls',0) for m in metrics),
        'model_elapsed_s':sum(m.get('reasoning_time_s',0) for m in metrics),'replay_enabled':False}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('excluded','sessions')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
