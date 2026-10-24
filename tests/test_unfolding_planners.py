import copy
import json
from pathlib import Path
from types import SimpleNamespace
import threading

from PIL import Image
import pytest

pytestmark = pytest.mark.usefixtures("fake_one_shot_view_hardware")

from cloth_agent.experiments import one_shot_goal_full_episode_v1 as runtime_module
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as wire
from cloth_agent.pre_action_gate import PreActionGate, GATE_SCHEMA
from cloth_agent.unfolding_planners import (
    ShadowPlannerRouter, DeliberativePlanner, MemoryPlanner, PlannerDisabledError)
from cloth_agent import planner_shadow_review as review


def proposal(decision='ACTION'):
    return dict(decision=decision, grasp_px=[20,30] if decision=='ACTION' else None,
        target_px=[70,40] if decision=='ACTION' else None,
        motion_type='lift_pull_place' if decision=='ACTION' else None, short_reason='拉开布边。')


def images(tmp_path):
    for name in ('current.png','goal.png'):
        Image.new('RGB',(100,80),'purple').save(tmp_path/name)
    return tmp_path/'current.png',tmp_path/'goal.png'


def test_gate_exact_inputs_no_post_action_leakage_or_fast_rationale(tmp_path):
    current,goal=images(tmp_path)
    (tmp_path/'after').mkdir()
    (tmp_path/'after/forbidden.txt').write_text('FUTURE_SECRET')
    p=proposal();p['short_reason']='FAST_RATIONALE_SECRET。'
    calls=[]
    def request(folder, body, headers, timeout):
        calls.append(body)
        assert folder==tmp_path/'gate_request'
        assert body['model']=='gpt-6-astra' and body['reasoning']=={'effort':'high'}
        assert body['tools']==[] and body['tool_choice']=='none'
        assert body['text']['format']['schema']==GATE_SCHEMA
        assert len(body['input'])==1
        content=body['input'][0]['content']
        assert len(content)==4
        assert [c['image_url'] for c in content[1:]]==[
            wire.image_data(current),wire.image_data(goal),wire.image_data(tmp_path/'gate_input_overlay.png')]
        assert 'FUTURE_SECRET' not in json.dumps(body)
        assert 'FAST_RATIONALE_SECRET' not in json.dumps(body)
        assert 'previous_response_id' not in body
        return {'decision':'DELIBERATE'},{'wall_latency_s':1.}
    gate=PreActionGate(tmp_path,{},request)
    assert gate.evaluate(current,goal,p)=={'decision':'DELIBERATE'}
    assert len(calls)==1 and gate.metrics['status']=='VALID'
    assert set(wire.read(tmp_path/'gate_result.json'))=={'decision'}
    manifest=wire.read(tmp_path/'gate_input_manifest.json')
    assert not any(manifest[k] for k in ('after_supplied','execution_supplied','history_supplied','graph_supplied','previous_rejection_supplied'))


@pytest.mark.parametrize('failure',['timeout','schema','refusal'])
def test_gate_error_uncertain_no_retry(tmp_path,failure):
    current,goal=images(tmp_path);calls=[]
    def request(*args):
        calls.append(1)
        if failure=='timeout':raise TimeoutError('offline synthetic')
        if failure=='refusal':raise ValueError('refusal')
        return {'decision':'TRUST','extra':'not allowed'},{'wall_latency_s':1}
    gate=PreActionGate(tmp_path,{},request)
    assert gate.evaluate(current,goal,proposal())=={'decision':'UNCERTAIN'}
    assert gate.metrics['status']=='ERROR' and len(calls)==1


def test_gate_user_stop_propagates(tmp_path):
    current,goal=images(tmp_path)
    def request(*args):raise KeyboardInterrupt()
    gate=PreActionGate(tmp_path,{},request)
    with pytest.raises(KeyboardInterrupt):gate.evaluate(current,goal,proposal())
    assert gate.metrics['status']=='INTERRUPTED'


def test_router_cannot_replace_action_or_escalate_on_host_reject(tmp_path):
    plan=proposal();router=ShadowPlannerRouter(tmp_path);router.accept_fast(plan)
    class Gate:
        metrics={'status':'VALID'}
        def evaluate(self,current,goal,action):
            action['grasp_px'][:]=[99,99]
            return {'decision':'DELIBERATE'}
    router.shadow(Gate(),None,None)
    assert router.execution_plan==plan
    router.host_reject('synthetic depth failure')
    states=[r['state'] for r in wire.read(tmp_path/'planner_states.json')['events']]
    assert states==['FAST_ACTION','GATE_DELIBERATE','HOST_REJECT']
    for planner in (DeliberativePlanner(),MemoryPlanner()):
        assert not planner.enabled
        with pytest.raises(PlannerDisabledError):planner.plan(None)


def install_runtime(monkeypatch,tmp_path,gate_label='TRUST',host_reject=False,decision='ACTION'):
    m=runtime_module;calls=[];expected=proposal(decision)
    class Runtime:
        def __init__(self,*args):self.cancel=threading.Event()
        def capture_current(self,path):
            calls.append('observe');path.mkdir()
            Image.new('RGB',(100,80),'purple').save(path/'rgb.png')
            return SimpleNamespace(image=path/'rgb.png')
        def arrival(self,*args):calls.append('arrival')
        def _execute_operations(self,operations,folder):
            assert operations==expected
            calls.append('execute');return {'status':'EXECUTED'}
        def move_to_perception(self,*args):calls.append('return')
        def close(self):calls.append('close')
        def stop(self):calls.append('stop')
    class Recorder:
        def __init__(self,*args):pass
        def start(self):pass
        def check(self):pass
        def stage(self,*args):pass
        def close(self):pass
    def init(runtime,folder,*args):
        folder.mkdir();m.save(folder/'result.json',dict(physical_actions=1,robot_execution_s=1))
    def plan(root,folder,before,headers,timeout):
        calls.append('fast')
        m.save(folder/'plan.json',expected);m.save(folder/'attempt.json',{'http_attempts':1})
        m.save(folder/'model_metrics.json',{'wall_latency_s':22.})
        wire.overlay(before.image,expected,folder/'proposed_action_overlay.png','Fast')
        return copy.deepcopy(expected),{'wall_latency_s':22.}
    def request(folder,body,headers,timeout):
        calls.append('gate')
        assert not (folder.parent/'after').exists()
        assert wire.read(folder.parent/'host_validation.json')['status']=='NOT_RUN'
        if gate_label=='ERROR':raise TimeoutError('test gate error')
        if gate_label=='INTERRUPT':raise KeyboardInterrupt()
        return {'decision':gate_label},{'wall_latency_s':3.}
    def prepare(runtime,before,plan,folder):
        calls.append('host');assert plan==expected
        if host_reject:raise m.DualArmError('synthetic depth rejection')
        return copy.deepcopy(plan),{}
    for name,obj in [('EpisodeCollectionRuntime',Runtime),('TrajectoryRecorder',Recorder),
                     ('initialize',init),('plan_round',plan),('direct_once',request),('prepare_action',prepare)]:
        monkeypatch.setattr(m,name,obj)
    monkeypatch.setattr(m.planner,'auth_headers',lambda:{})
    goal=tmp_path/'goal.png';Image.new('RGB',(100,80),'purple').save(goal)
    args=SimpleNamespace(output=tmp_path/'episode',goal_image=goal,config=Path('unused'),
        max_rounds=1,timeout_s=2,gate_timeout_s=2)
    return m,args,calls,expected


@pytest.mark.parametrize('gate_label',['TRUST','DELIBERATE','UNCERTAIN','ERROR'])
@pytest.mark.parametrize('host_reject',[False,True])
def test_end_to_end_shadow_does_not_change_action_or_host(monkeypatch,tmp_path,gate_label,host_reject):
    m,args,calls,expected=install_runtime(monkeypatch,tmp_path,gate_label,host_reject)
    m.run(args);root=args.output;r=root/'round_000'
    assert calls.count('fast')==calls.count('gate')==1
    assert calls.index('gate')<calls.index('host')
    assert calls.count('execute')==int(not host_reject)
    assert m.read(r/'plan.json')==m.read(r/'fast_plan.json')==expected
    s=m.read(root/'episode_summary.json');layer=s['planner_layers']
    assert s['termination_reason']==('SAFETY_STOP' if host_reject else 'ROUND_LIMIT')
    assert layer['host_reject_count']==int(host_reject)
    assert layer['fast_planner_latency_s']['median']==22.
    assert layer['total_fast_planner_wall_s']==22.
    assert layer['gate_status_counts']=={'ERROR' if gate_label=='ERROR' else 'VALID':1}
    assert sum(layer['gate_decision_counts'].values())==int(gate_label!='ERROR')
    assert layer['gate_human_correspondence'][0]['human_label'] is None
    for name in ('fast_plan.json','fast_action_overlay.png','gate_input_overlay.png',
                 'gate_result.json','host_validation.json','execution.json','timing.json'):
        assert (r/name).exists()
    timing=m.read(r/'timing.json')
    assert all(k in timing for k in ('fast_planner_wall_s','gate_wall_s','host_validation_s','robot_execution_s','total_round_s'))
    assert m.read(r/'host_validation.json')['status']==('REJECT' if host_reject else 'PASS')
    assert (root/'review.html').exists() and not s.get('report_error')


@pytest.mark.parametrize('decision',['DONE','ABSTAIN'])
def test_nonaction_has_no_gate_or_host_call(monkeypatch,tmp_path,decision):
    m,args,calls,_=install_runtime(monkeypatch,tmp_path,decision=decision)
    m.run(args)
    assert calls.count('fast')==1
    assert not any(x in calls for x in ('gate','host','execute'))
    s=m.read(args.output/'episode_summary.json')
    assert s['termination_reason']==decision and s['planner_layers']['gate_decision_counts']==dict(TRUST=0,DELIBERATE=0,UNCERTAIN=0)


def test_gate_interrupt_stops_before_host(monkeypatch,tmp_path):
    m,args,calls,_=install_runtime(monkeypatch,tmp_path,gate_label='INTERRUPT')
    m.run(args)
    assert 'host' not in calls and 'execute' not in calls and 'stop' in calls
    assert m.read(args.output/'episode_summary.json')['termination_reason']=='USER_STOP'


def make_ratings(root,choice='NEED_DELIBERATION'):
    manifest=wire.read(root/'review_manifest.json')
    return dict(schema_version=1,episode_id=manifest['episode_id'],
        protocol_sha256=manifest['protocol_sha256'],goal_sha256=manifest['goal_sha256'],
        final_fold_ready='UNCERTAIN',reviewer='synthetic test',
        rounds=[dict(round_id=r['round_id'],binding_sha256=r['binding_sha256'],choice=choice,
                     reasonable=None,progress=None,note='test') for r in manifest['rounds']])


def test_review_false_trust_uses_human_not_host_and_preserves_records(monkeypatch,tmp_path):
    m,args,calls,_=install_runtime(monkeypatch,tmp_path,gate_label='TRUST',host_reject=True)
    m.run(args);root=args.output
    hashes={str(p.relative_to(root)):wire.sha(p) for p in root.rglob('*') if p.is_file()}
    ratings=tmp_path/'ratings.json';wire.save(ratings,make_ratings(root))
    result=review.evaluate_review(root,ratings,tmp_path/'analysis')
    assert result['false_trust_count']==1 and result['false_trust_over_human_bad']==1.
    assert result['false_deliberate_over_human_normal'] is None
    assert all(wire.sha(root/k)==v for k,v in hashes.items())
    wire.save(ratings,make_ratings(root,'TRUST_FAST'))
    result=review.evaluate_review(root,ratings,tmp_path/'analysis_normal')
    assert result['false_trust_count']==0 # same Host reject, different human policy label
    assert result['normal_action_count']==1
    wire.save(ratings,make_ratings(root,None))
    result=review.evaluate_review(root,ratings,tmp_path/'analysis_pending')
    assert result['false_trust_count'] is None and result['pending_rounds']==1
    wire.save(ratings,make_ratings(root,'UNCERTAIN'))
    result=review.evaluate_review(root,ratings,tmp_path/'analysis_uncertain')
    assert result['false_trust_count'] is None and result['decisive_human_gate_pairs']==0
    bad=make_ratings(root);bad['rounds'][0]['binding_sha256']={}
    wire.save(ratings,bad)
    with pytest.raises(ValueError,match='evidence mismatch'):
        review.evaluate_review(root,ratings,tmp_path/'bad_analysis')


def test_review_false_deliberate_and_gate_error_exclusion(monkeypatch,tmp_path):
    m,args,calls,_=install_runtime(monkeypatch,tmp_path,gate_label='DELIBERATE')
    m.run(args);ratings=tmp_path/'ratings.json'
    wire.save(ratings,make_ratings(args.output,'TRUST_FAST'))
    result=review.evaluate_review(args.output,ratings,tmp_path/'false_deliberate')
    assert result['false_deliberate_count']==1
    assert result['false_deliberate_over_human_normal']==1.
    # A synthetic fallback is never treated as a model UNCERTAIN in quality metrics.
    other=tmp_path/'error_case';other.mkdir()
    m,args,calls,_=install_runtime(monkeypatch,other,gate_label='ERROR')
    m.run(args);wire.save(ratings,make_ratings(args.output,'NEED_DELIBERATION'))
    result=review.evaluate_review(args.output,ratings,tmp_path/'error_review')
    assert result['valid_gate_human_pairs']==0 and result['false_trust_count'] is None


def test_fast_prompt_only_adds_authorized_boundary_and_keeps_host_snapshot():
    # Saved original run supplies a durable fixture independent of current constants.
    original=Path('reports/one_shot_goal_full_episode_v1_restart02/protocol.json')
    if not original.exists():pytest.skip('local original-run fixture unavailable')
    protocol=wire.read(original)
    assert wire.PROMPT==protocol['prompt']+'\n\n'+wire.GRIPPER_BOUNDARY_PROMPT
    assert wire.SCHEMA==protocol['schema']
    assert wire.MODEL==protocol['model'] and wire.EFFORT==protocol['effort']
    for p,h in protocol['protected_sources'].items():
        if Path(p).resolve() == Path(wire.__file__).resolve():
            continue  # User-authorized prompt-only boundary addition above.
        assert wire.sha(p)==h,p
