"""Small, prospective A/B skill-gate discovery; never imported by live policy.

Uses the existing skills and RobotAPI execution chain. Discovery is a separate,
tool-free request containing only G01/G02. No learned gate is auto-deployed.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import json
import shutil

from .discovery_data import save, read, file_hash, digest

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ('G01', 'G02', 'G03', 'G04')
FROZEN_FILES = (
    'cloth_agent/experiments/partitioned_unfold.py',
    'cloth_agent/experiments/unfold_skill_planner.py',
    'cloth_agent/contour_normal_grasp.py',
    'cloth_agent/cloth_harness/cv_harness/segmentation.py',
    'cloth_agent/replay_dual_arm_runtime.py',
    'cloth_agent/robot_api.py', 'cloth_agent/robot_collision.py',
    'skills/garment-tangent-grasp/SKILL.md',
    'config/dual_arm.zed_dual_xarm7_20261103.json',
    'config/fusion.unfold_dual.json', 'config/topdown.unfold_zed.json',
    'cloth_agent/experiments/skill_gate_pilot.py',
)


def now():
    return datetime.now(timezone.utc).isoformat()


def init(root):
    from .one_shot_goal_planner_probe_v1 import MODEL,EFFORT
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    protocol = dict(version=1, created=now(), max_actions=4,
        train=['G01', 'G02'], test=['G03', 'G04'],
        garment='current black garment; capture its own spread goal before trials',
        state_recipes={'G01':'single-side fold', 'G02':'central pile',
                       'G03':'two-side fold', 'G04':'pile with changed orientation'},
        order={'G01': ['A', 'B'], 'G02': ['B', 'A'],
               'G03': ['A', 'B'], 'G04': ['B', 'A']},
        arms={'A': ['SIMPLE'] * 4, 'B': ['TANGENT'] + ['SIMPLE'] * 3},
        observation='existing single current planning RGB plus fixed goal; existing RGB-D for execution',
        model=MODEL, reasoning_effort=EFFORT,
        selection_retries='unchanged existing invalid-candidate/depth/distance reselection; logged; no extra semantic review',
        reset='operator matched reset; no automated random reset',
        applicability='ABSTAIN, invalid output, invalid geometry/depth, or rejected planning: no dispatch, no replacement skill',
        model_review_per_step=False, automatic_deployment=False,
        comparison='Report final task completion, identifiable semantic parts, flatness, side effects, and first-step effect separately. If criteria conflict, initial states mismatch, evidence is missing, or either first skill is unavailable: INCOMPARABLE; do not label the other skill superior.',
        flatness_scale={'0': 'strong pile/large overlaps', '1': 'partly spread, major folds',
                        '2': 'mostly spread with remaining folds', '3': 'spread, ordinary wrinkles only'},
        holdout='freeze gate before G03/G04 prediction; predict from canonical Before RGB before either test trajectory; reveal both test outcomes only after both predictions',
        frozen_files={f: file_hash(ROOT/f) for f in FROZEN_FILES})
    save(root/'protocol.json', protocol, immutable=True)
    save(root/'protocol_seal.json', dict(sha256=file_hash(root/'protocol.json')), immutable=True)
    return root


def capture(root, purpose, group=None, note=''):
    """Existing fixed-camera pipeline only; no robot reads, startup or motion."""
    from types import SimpleNamespace
    import fcntl
    import numpy as np
    from cloth_agent.perception import PerceptionConfig, capture_two_view_rgbd
    from cloth_agent.dual_cloud_observation import (
        load_config, validate_primary, capture_secondary, scale_world, attach_secondary)
    from cloth_agent.dual_arm.observation import save_observation
    from .unfold_dual_topdown import reconstruct
    verify(root)
    if purpose not in ('goal','before'):
        raise ValueError('invalid capture purpose')
    if purpose=='before' and group not in GROUPS:
        raise ValueError('select group')
    if purpose=='goal' and (root/'goal.json').exists():
        raise ValueError('goal already frozen; no capture needed')
    if purpose=='before' and (root/group/'before').exists():
        raise ValueError('canonical Before already frozen; no capture needed')
    out=root/'captures'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    # Same lease as existing runtime; never race a running trajectory.
    with open('/tmp/clothagent_codex_dual_arm.lock','a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=SimpleNamespace(raw=read(ROOT/'config/dual_arm.zed_dual_xarm7_20261103.json'))
        fusion=load_config(ROOT/'config/fusion.unfold_dual.json')
        perception=PerceptionConfig.load(ROOT,ROOT/'config/perception.zed_dual_xarm7_20261103.json')
        frame,=capture_two_view_rgbd(perception)
        validate_primary(frame,fusion)
        y,x=np.indices(frame.depth_m.shape);k,d=frame.intrinsics,frame.depth_m
        xyz=np.stack(((x-k[0,2])/k[0,0]*d,(y-k[1,2])/k[1,1]*d,d),axis=-1)
        t=np.array(frame.X_base_camera,copy=True)
        raw=((xyz@t[:3,:3].T+t[:3,3])*1000).astype(np.float32);t[:3,3]*=1000
        secondary=capture_secondary(fusion)
        before=save_observation(out/'observation',frame.rgb,scale_world(raw,t,fusion['depth_scale']['zed']),cfg,{},
            extra=dict(camera_serial=frame.serial,intrinsics=k.tolist(),world_from_camera_mm=t.tolist(),
                       purpose='pilot_reference_camera_only',depth_scale=fusion['depth_scale']['zed']))
        np.save(before.directory/'raw_world_xyz_mm.npy',raw)
        before=attach_secondary(before,secondary,fusion,cfg)
        view=reconstruct(before,out,read(ROOT/'config/topdown.unfold_zed.json'))
    if purpose=='goal':bind_goal(root,view.image,note)
    else:canonical_before(root,group,view.image,note)
    save(out/'capture.json',dict(purpose=purpose,group=group,robot_calls=0,planning_rgb=str(view.image)))
    return dict(output=str(out),robot_calls=0)


def verify(root):
    root = Path(root)
    if file_hash(root/'protocol.json') != read(root/'protocol_seal.json')['sha256']:
        raise ValueError('pilot protocol changed')
    p = read(root/'protocol.json')
    for f, h in p['frozen_files'].items():
        if file_hash(ROOT/f) != h:
            raise ValueError('frozen skill/config changed: '+f)
    goal = read(root/'goal.json')
    if goal and file_hash(root/'goal.png') != goal['sha256']:
        raise ValueError('pilot goal changed')
    return p


def bind_goal(root, image, note):
    verify(root)
    if list(root.glob('G*/**/trajectory.json')):
        raise ValueError('cannot change goal after starting trajectories')
    if (root/'goal.json').exists():
        raise ValueError('goal already bound')
    shutil.copyfile(image, root/'goal.png')
    save(root/'goal.json', dict(source=str(Path(image).resolve()), sha256=file_hash(root/'goal.png'),
                               note=note, bound=now()), immutable=True)


def canonical_before(root, group, image, note):
    """Operator-provided initial snapshot; no outcomes or chosen action attached."""
    verify(root)
    if group not in GROUPS:
        raise ValueError('invalid group')
    if list((root/group).glob('*/trajectory.json')):
        raise ValueError('initial snapshot must precede both trajectories')
    target = root/group/'before'
    target.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(image, target/'rgb.png')
    save(target/'manifest.json', dict(sha256=file_hash(target/'rgb.png'),
         source=str(Path(image).resolve()), operator_note=note, created=now()), immutable=True)


def assigned_skill(arm, step):
    return 'TANGENT' if arm == 'B' and step == 0 else 'SIMPLE'


def plan_assigned(root, folder, before, skill, headers, timeout, direct, runtime):
    """External experimental assignment, not an applicability gate or new skill."""
    from . import partitioned_unfold as unfold
    if skill == 'SIMPLE':
        def simple_request(destination, payload, auth, seconds):
            payload=copy.deepcopy(payload)
            payload['text']['format']['schema']['properties']['skill']={'type':'null'}
            value,metrics=direct(destination,payload,auth,seconds)
            if value.get('skill') is not None:
                raise ValueError('assigned Simple returned another skill; not applicable')
            return value,metrics
        return unfold.plan_round_pure(root, folder, before, headers, timeout, simple_request, runtime=runtime)
    from PIL import Image
    from . import one_shot_goal_planner_probe_v1 as io
    from .unfold_topdown import primary, PRIMARY_GUIDANCE
    from .unfold_planning_skills import workspace_prompt, planning_text
    from .unfold_skill_planner import run
    view = primary(before, folder)
    raw = (ROOT/'skills/garment-tangent-grasp/SKILL.md').read_text(encoding='utf-8')
    # The randomized assignment replaces only the *selection criterion*.
    # Retain the current skill's geometry/action instructions verbatim.
    method = planning_text('garment-tangent-grasp', raw).split('\n\n', 1)[1]
    method = method.replace('每轮根据新图重新选择普通抓放或技能。','')
    prompt = ('第一张是当前衣物，第二张是展开目标。本轮对照实验指定调用 Tangent Skill，'
              '不比较或选择其他技能。根据当前图输出该技能的一步候选；若无法提出有效候选，'
              '输出ABSTAIN，已达到目标则DONE，不强行生成动作。\n'+method+'\n'+PRIMARY_GUIDANCE)
    prompt += '\n'+workspace_prompt(runtime, before, view, folder)
    body = io.payload(view.image, root/'goal/target.png', Image.open(view.image).size)
    body['instructions'] = prompt
    contract = ('ACTION时skill必须为garment-tangent-grasp，给可见边界点boundary_point_px，'
                'grasp_px、target_px为null，chord_endpoints_px=[]，pull_distance_mm为10–150。'
                'region_name/evidence说明定位区域和依据，不输出隐藏点。非ACTION时skill及所有点为null，'
                'chord_endpoints_px=[]，pull_distance_mm=10。')
    save(folder/'planning_skills.json', dict(mode='pilot_assigned_tangent',
         skills=['garment-tangent-grasp'], local_cv_executed=False, semantic_review=False))

    def assigned_request(destination, payload, auth, seconds):
        payload = copy.deepcopy(payload)
        payload['text']['format']['schema']['properties']['skill'] = {
            'anyOf': [{'type':'string','enum':['garment-tangent-grasp']},{'type':'null'}]}
        value, metrics = direct(destination, payload, auth, seconds)
        if value.get('decision') == 'ACTION' and value.get('skill') != 'garment-tangent-grasp':
            raise ValueError('assigned Tangent returned no Tangent action; not applicable')
        return value, metrics

    value, metrics = run(root, folder, view, body, headers, timeout, assigned_request,
                         selection_prompt=contract)
    return unfold.finish_plan(folder, before, view, value, metrics)


def trial(root, group, arm, note, execute=False):
    """One operator-arranged trajectory, existing episode/stop/validation code."""
    protocol = verify(root)
    if group not in GROUPS or arm not in ('A', 'B'):
        raise ValueError('invalid group/arm')
    if not (root/'goal.json').exists() or not (root/group/'before/manifest.json').exists():
        raise ValueError('bind same-garment goal and canonical Before RGB first')
    order = protocol['order'][group]
    if arm == order[1] and not (root/group/order[0]/'trajectory.json').exists():
        raise ValueError('follow preregistered A/B order')
    if group in protocol['test']:
        prediction = read(root/'predictions'/f'{group}.json')
        if not prediction or prediction['gate_sha256'] != file_hash(root/'gate.json'):
            raise ValueError('freeze gate and predict this Before before test execution')
    folder = root/group/arm
    if folder.exists():
        raise ValueError('trajectory already exists; never overwrite/rerun silently')
    if not execute:
        return dict(status='READY_CHECK_ONLY', group=group, arm=arm, robot_calls=0)
    from .partitioned_unfold import configure_runtime, episode, DEFAULT_CONFIG
    from cloth_agent.replay_dual_arm_runtime import PartitionedReplayRuntime
    from cloth_agent.dual_cloud_observation import load_config
    from .one_shot_goal_full_episode_v1 import direct_once
    from .one_shot_goal_planner_probe_v1 import auth_headers
    import signal
    folder.mkdir(parents=True)
    (folder/'goal').mkdir()
    shutil.copyfile(root/'goal.png', folder/'goal/target.png')
    record = dict(group=group, arm=arm, started=now(), status='STARTED',
                  reset_note=note, assigned_sequence=protocol['arms'][arm], max_actions=4)
    save(folder/'trajectory.json', record)
    runtime = None
    old_handler = signal.getsignal(signal.SIGTERM)
    def terminate(*args):
        raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM, terminate)
    try:
        headers = auth_headers()
        runtime = configure_runtime(PartitionedReplayRuntime(DEFAULT_CONFIG, planning_only=False))
        runtime.dual_cloud_config = load_config(ROOT/'config/fusion.unfold_dual.json')
        def request(step_folder, before, context):
            step = int(next(p.name[6:] for p in [step_folder, *step_folder.parents]
                            if p.name.startswith('round_')))
            skill = assigned_skill(arm, step)
            save(step_folder/'assignment.json', dict(skill=skill, step=step,
                 reason='preregistered intervention; no learned gate during A/B'))
            return plan_assigned(folder, step_folder, before, skill, headers, 300,
                                 direct_once, runtime)
        outcome = episode(runtime, folder, max_actions=4, request=request)
        record.update(status=outcome['status'], physical_actions=outcome['physical_actions'])
        if outcome['physical_actions'] == 0:
            record['first_step_applicability'] = ('NO_ACTION_DONE' if outcome['status']=='MODEL_REPORTED_DONE'
                                                 else 'NOT_APPLICABLE')
        else:
            first = outcome['steps'][0].get('execution', {})
            record['first_step_applicability'] = 'EXECUTED' if first.get('status')=='EXECUTED' else 'EXECUTION_FAILURE'
    except BaseException as exc:
        record.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',
                      error=f'{type(exc).__name__}: {exc}')
        # Controller completion does not establish cloth acquisition.
        record['execution_evidence'] = str(folder/'episode_summary.json')
        summary=read(folder/'episode_summary.json')
        record['physical_actions']=summary.get('physical_actions',0)
        if summary.get('steps'):
            first=summary['steps'][0].get('execution',{})
            record['first_step_applicability']=('EXECUTED' if first.get('status')=='EXECUTED'
                else 'EXECUTION_FAILURE' if first.get('physical_execution') else 'NOT_APPLICABLE')
        raise
    finally:
        if runtime is not None:
            runtime.close()
        signal.signal(signal.SIGTERM, old_handler)
        record['ended'] = now()
        save(folder/'trajectory.json', record)
    return record


def record_comparison(root, group, feedback):
    """Offline evidence, not a per-step VLM review or controller success proxy."""
    p = verify(root)
    if group not in GROUPS:
        raise ValueError('invalid group')
    if group in p['test']:
        if not all((root/'predictions'/f'{g}.json').exists() for g in p['test']):
            raise ValueError('do not reveal test outcomes before both predictions')
    for arm in ('A', 'B'):
        if read(root/group/arm/'trajectory.json').get('status') in (None, 'STARTED'):
            raise ValueError('both actual trajectories must be recorded')
    needed = {'reviewer','matching','winner','reason','A','B','evidence'}
    if set(feedback) != needed:
        raise ValueError('comparison fields must be '+str(sorted(needed)))
    if feedback['matching'] not in ('MATCHED','MISMATCHED','UNCERTAIN'):
        raise ValueError('invalid matching assessment')
    if feedback['winner'] not in ('A','B','TIE','INCOMPARABLE'):
        raise ValueError('invalid comparison')
    if feedback['matching']!='MATCHED' and feedback['winner']!='INCOMPARABLE':
        raise ValueError('unmatched pairs cannot support a preference')
    for arm in ('A','B'):
        v = feedback[arm]
        if set(v) != {'semantic_parts_by_step','final_flatness','task_complete','first_step_effect','side_effects','applicability'}:
            raise ValueError('missing outcome dimensions for '+arm)
        if v['final_flatness'] not in (None,0,1,2,3) or v['task_complete'] not in (True,False,None):
            raise ValueError('invalid observed outcome rating')
        if not isinstance(v['semantic_parts_by_step'],list) or not v['semantic_parts_by_step']:
            raise ValueError('record observed semantic parts, including UNKNOWN, by step')
        if v['applicability'] != 'EXECUTED' and feedback['winner']!='INCOMPARABLE':
            raise ValueError('unavailable first skill is not comparative evidence')
        actual=read(root/group/arm/'trajectory.json')
        if v['applicability']=='EXECUTED' and actual.get('first_step_applicability')!='EXECUTED':
            raise ValueError('feedback cannot promote unexecuted skill to executed')
    if not feedback['reviewer'] or not feedback['evidence']:
        raise ValueError('require reviewer and real observation evidence')
    for ref in feedback['evidence']:
        f=(root/group/ref).resolve()
        if not f.is_relative_to((root/group).resolve()) or not f.is_file():
            raise ValueError('evidence must be an existing file from this pair')
    save(root/group/'comparison.json',dict(feedback,recorded=now()),immutable=True)


def review_package(root, group):
    """Local side-by-side normal observations; never a live verification call."""
    import html
    import os
    protocol=verify(root)
    if group not in GROUPS:raise ValueError('invalid group')
    if group in protocol['test'] and not all((root/'predictions'/f'{g}.json').exists() for g in protocol['test']):
        raise ValueError('both predictions must be frozen before revealing test outcomes')
    folder=root/group
    document=['<!doctype html><meta charset="utf-8"><title>Pilot evidence</title>',
              '<style>body{font:17px sans-serif;margin:25px}img{max-width:48%}pre{white-space:pre-wrap}section{border-top:1px solid #aaa}</style>',
              '<h1>'+group+' offline evidence</h1>',
              '<p>Motor completion is not grasp success. Rate actual garment change; use UNKNOWN when unclear. '
              'Do not attribute an unmatched initial state or an inapplicable skill to skill benefit.</p>']
    template=dict(reviewer='',matching='UNCERTAIN',winner='INCOMPARABLE',reason='',evidence=[])
    for arm in ('A','B'):
        trialdir=folder/arm
        summary=read(trialdir/'episode_summary.json')
        if not summary:raise ValueError('trajectory missing: '+arm)
        document+=['<h2>'+arm+'</h2><pre>'+html.escape(json.dumps(read(trialdir/'trajectory.json'),ensure_ascii=False,indent=2))+'</pre>']
        images=[trialdir/'initial_observation/topdown/planning_rgb.png']
        for step in summary.get('steps',[]):
            n=step['round'];rd=trialdir/f'round_{n:03d}'
            candidates=sorted(rd.glob('**/after/topdown/planning_rgb.png'))
            images.extend(candidates)
            document.append('<section><h3>Step '+str(n)+'</h3>')
            for assignment in sorted(rd.glob('**/assignment.json')):
                document.append('<pre>'+html.escape(assignment.read_text(encoding='utf-8'))+'</pre>')
            for plan in sorted(rd.glob('**/plan.json')):
                document.append('<pre>'+html.escape(plan.read_text(encoding='utf-8'))+'</pre>')
            document.append('</section>')
        frames=[]
        for i,path in enumerate(images):
            if path.exists():
                rel=os.path.relpath(path,folder)
                document.append('<p>'+html.escape(rel)+'</p><img src="'+html.escape(rel)+'">')
                template['evidence'].append(rel)
                frames.append(dict(observation=rel,parts={k:'UNKNOWN' for k in
                    ('collar','left_sleeve','right_sleeve','left_hem_corner','right_hem_corner')}))
        actual=read(trialdir/'trajectory.json')
        template[arm]=dict(semantic_parts_by_step=frames,final_flatness=None,task_complete=None,
            first_step_effect='UNKNOWN',side_effects=[],applicability=actual.get('first_step_applicability','UNKNOWN'))
    (folder/'review.html').write_text('\n'.join(document),encoding='utf-8')
    target=folder/'feedback_template.json'
    if not target.exists():save(target,template)
    return dict(review=str(folder/'review.html'),feedback_template=str(target))


def image_part(path):
    from .one_shot_goal_planner_probe_v1 import image_data
    return dict(type='input_image',image_url=image_data(path),detail='high')


GATE_SCHEMA = dict(type='object', additionalProperties=False, properties={
    'status': {'type':'string','enum':['DISCOVERED','INCONCLUSIVE']},
    'rule': {'type':'string'}, 'evidence_groups': {'type':'array','items':{'type':'string','enum':['G01','G02']}},
    'evidence': {'type':'string'}, 'limitations': {'type':'string'}},
    required=['status','rule','evidence_groups','evidence','limitations'])
PREDICT_SCHEMA = dict(type='object', additionalProperties=False, properties={
    'skill': {'type':'string','enum':['SIMPLE','TANGENT','UNDECIDED']}, 'reason': {'type':'string'}},
    required=['skill','reason'])


def discovery_payload(root):
    """Whitelist-only packaging. No directory enumeration or test access."""
    from PIL import Image
    from . import one_shot_goal_planner_probe_v1 as io
    protocol = verify(root)
    content = [dict(type='input_text',text='展开目标（所有轨迹相同）：'),image_part(root/'goal.png')]
    evidence = {}
    for group in protocol['train']:
        outcome = read(root/group/'comparison.json')
        if not outcome:
            raise ValueError('training pair feedback missing: '+group)
        # Paths/notes do not encode intended actions. Only actual outcomes and RGB.
        evidence[group] = {'comparison': outcome}
        content += [dict(type='input_text',text=group+' 初态'),image_part(root/group/'before/rgb.png')]
        for arm in ('A','B'):
            f=root/group/arm/'initial_observation/topdown/planning_rgb.png'
            if not f.exists():
                raise ValueError('actual trajectory Before missing')
            content += [dict(type='input_text',text=f'{group}/{arm} 实际初态'),image_part(f)]
        content.append(dict(type='input_text',text=json.dumps({'group':group,'real_feedback':outcome},ensure_ascii=False)))
    body = io.payload(root/'goal.png',root/'goal.png',Image.open(root/'goal.png').size)
    body['input'][0]['content']=content
    body['instructions']=('你是独立 Skill Gate Discovery Harness，只能使用本请求中的两个真实对照经验。'
        'A为Simple首步、后续固定Simple；B为Tangent首步、后续固定Simple；最多四步。'
        '请自主归纳简洁的首步选择规则，输出SIMPLE或TANGENT的依据必须是执行前RGB和目标图可观察的信息。'
        '不根据颜色、组号或动作后才知道的事实记忆答案；不假设某技能天然更适合某类情况。'
        '先判断反馈是否确实支持偏好，不能把执行成功当作展开成功，不适用和不可比不是另一技能胜出。'
        '无需修改任何技能或提出新的动作方法。规则须能注入现有单次规划请求，不增加观察、工具或推理轮数。'
        '只给一段简洁规则并引用G01/G02的实际证据；无法归纳则INCONCLUSIVE，rule为空。'
        '没有留出状态或结果可供使用，不能声称泛化。')
    body['text']['format'].update(name='skill_gate_discovery',schema=GATE_SCHEMA)
    return body, evidence


def discover(root, direct=None):
    from .one_shot_goal_full_episode_v1 import direct_once
    from .one_shot_goal_planner_probe_v1 import auth_headers
    if (root/'gate.json').exists():
        raise ValueError('gate already frozen; no iterative tuning on test')
    body,evidence=discovery_payload(root)
    output,metrics=(direct or direct_once)(root/'discovery_request',body,auth_headers(),300)
    from jsonschema import validate
    validate(output,GATE_SCHEMA)
    if output['status']=='DISCOVERED' and (not output['rule'].strip() or not output['evidence_groups']):
        raise ValueError('discovered gate requires rule and training evidence')
    gate=dict(output, frozen=now(),training_digest=digest(evidence),model=body['model'],
              request_sha256=digest(body), protocol_sha256=file_hash(root/'protocol.json'),metrics=metrics)
    save(root/'gate.json',gate,immutable=True)
    save(root/'gate_seal.json',dict(sha256=file_hash(root/'gate.json')),immutable=True)
    return gate


def predict(root,group):
    from PIL import Image
    from . import one_shot_goal_planner_probe_v1 as io
    from .one_shot_goal_full_episode_v1 import direct_once
    protocol=verify(root)
    if group not in protocol['test']:
        raise ValueError('predictions only for prespecified held-out groups')
    gate=read(root/'gate.json')
    if file_hash(root/'gate.json')!=read(root/'gate_seal.json')['sha256']:
        raise ValueError('gate changed after freeze')
    if list((root/group).glob('*/trajectory.json')):
        raise ValueError('prediction must precede test trajectories')
    out=root/'predictions'/f'{group}.json'
    if out.exists():raise ValueError('held-out prediction already saved')
    before=root/group/'before/rgb.png'
    if gate['status']!='DISCOVERED':
        prediction=dict(skill='UNDECIDED',reason='Training evidence produced no frozen gate')
    else:
        body=io.payload(before,root/'goal.png',Image.open(before).size)
        body['instructions']=('第一张为当前RGB，第二张为目标。只根据以下冻结Gate选择首步技能，'
            '不修改规则，不需要工具或额外观察。不能确定则UNDECIDED。\n'+gate['rule'])
        body['text']['format'].update(name='frozen_gate_prediction',schema=PREDICT_SCHEMA)
        prediction,_=direct_once(root/'prediction_requests'/group,body,io.auth_headers(),300)
        io.jsonschema.validate(prediction,PREDICT_SCHEMA)
    save(out,dict(prediction,group=group,saved=now(),before_sha256=file_hash(before),
                  gate_sha256=file_hash(root/'gate.json')),immutable=True)
    return read(out)


def report(root):
    """No invented gate or preference when physical trials/feedback are absent."""
    p=verify(root);gate=read(root/'gate.json');rows=[]
    for group in GROUPS:
        pred=read(root/'predictions'/f'{group}.json')
        revealed=group in p['train'] or all((root/'predictions'/f'{g}.json').exists() for g in p['test'])
        feedback=read(root/group/'comparison.json') if revealed else {}
        winner=feedback.get('winner','PENDING')
        choice={'SIMPLE':'A','TANGENT':'B'}.get(pred.get('skill'))
        correct=(choice==winner) if choice and winner in ('A','B') else None
        rows.append(dict(group=group,split='train' if group in p['train'] else 'test',
            trials={arm:read(root/group/arm/'trajectory.json').get('status','PENDING') for arm in ('A','B')},
            winner=winner,prediction=pred.get('skill'),correct=correct,comparison=feedback))
    result=dict(status='COMPLETE' if all(r['winner']!='PENDING' for r in rows) and gate else 'INCOMPLETE',
        gate=gate or {'status':'NOT_LEARNED','reason':'Awaiting real training feedback'},groups=rows,
        auto_deployed=False,generalization_claim=False)
    training=[r for r in rows if r['split']=='train']
    tests=[r for r in rows if r['split']=='test']
    if result['status']=='INCOMPLETE':result['diagnosis']='DATA_COLLECTION_INCOMPLETE'
    elif not any(r['winner'] in ('A','B') for r in training):result['diagnosis']='TRAINING_PREFERENCE_UNCLEAR'
    elif gate.get('status')!='DISCOVERED':result['diagnosis']='NO_GATE_INFERRED'
    elif any(r['correct'] is False for r in tests):result['diagnosis']='HELD_OUT_CHOICE_ERROR'
    elif any(r['correct'] is None for r in tests):result['diagnosis']='HELD_OUT_INCOMPARABLE_OR_UNDECIDED'
    else:result['diagnosis']='TWO_HELD_OUT_CHOICES_SUPPORTED_ONLY'
    result['training_preference_diversity']=len({r['winner'] for r in training if r['winner'] in ('A','B')})
    result['held_out_assessable']=sum(r['correct'] is not None for r in tests)
    result['held_out_correct']=sum(r['correct'] is True for r in tests)
    result['single_call_integration']='Frozen rule can be text in an existing single decision request; no production integration or physical gate validation in this pilot.'
    save(root/'report.json',result)
    text=['# Skill Gate Discovery Pilot','',
          '四组，前两组学习、后两组留出。每轨迹最多四步。未修改 Skill 算法，未部署 Gate。','',
          '|组|划分|A 状态|B 状态|真实偏好|预测|选对|','|---|---|---|---|---|---|---|']
    for r in rows:text.append(f"|{r['group']}|{r['split']}|{r['trials']['A']}|{r['trials']['B']}|{r['winner']}|{r['prediction']}|{r['correct']}|")
    text+=['','冻结 Gate：'+json.dumps(result['gate'],ensure_ascii=False),
           '', '诊断：'+result['diagnosis'],
           f"留出可比较：{result['held_out_assessable']}/2；选对：{result['held_out_correct']}。",
           '', 'PENDING/INCOMPARABLE 不计为成功。若实验未完成，不能回答技能收益或留出选择正确率。']
    for r in rows:
        if r['comparison']:
            text+=['','## '+r['group']+' 真实收益与副作用',
                   '```json',json.dumps(r['comparison'],ensure_ascii=False,indent=2),'```']
    text+=['','Gate 可作为现有单次决策请求中的文本规则；本轮未部署，未验证接入后的真机收益。',
           '若两组学习偏好相同，只能支持有限的选择偏好，不能据此宣称已学到两种技能的完整边界。']
    (root/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['init','goal','before','capture-goal','capture-before','trial','review','feedback','discover','predict','report'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--group',choices=GROUPS);p.add_argument('--arm',choices=['A','B'])
    p.add_argument('--image',type=Path);p.add_argument('--feedback',type=Path)
    p.add_argument('--note',default='');p.add_argument('--execute',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    if a.command in ('before','capture-before','trial','review','feedback','predict') and not a.group:
        p.error('--group required')
    if a.command=='trial' and not a.arm:p.error('--arm required')
    if a.command in ('goal','before') and not a.image:p.error('--image required')
    if a.command=='feedback' and not a.feedback:p.error('--feedback required')
    if a.command=='init':result={'root':str(init(root))}
    elif a.command=='goal':result=bind_goal(root,a.image,a.note)
    elif a.command=='before':result=canonical_before(root,a.group,a.image,a.note)
    elif a.command=='capture-goal':result=capture(root,'goal',note=a.note)
    elif a.command=='capture-before':result=capture(root,'before',a.group,a.note)
    elif a.command=='trial':result=trial(root,a.group,a.arm,a.note,a.execute)
    elif a.command=='review':result=review_package(root,a.group)
    elif a.command=='feedback':result=record_comparison(root,a.group,read(a.feedback))
    elif a.command=='discover':result=discover(root)
    elif a.command=='predict':result=predict(root,a.group)
    else:result=report(root)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
