import json
from types import SimpleNamespace
from pathlib import Path
import pytest
from cloth_agent.experiments.gravity_unfold import (
    command, evaluation, unknown, rollout, freeze_baseline, latest_completed, READY_KEYS, metrics, main)


def decision(p='LiftAndReveal', holding=False):
    d = dict(primitive=p, reason='visible cuff boundary', grasp=None if holding else [.5, 0, .05, 3.14, 0, 0],
             targets=[[.5, 0, .2, 3.14, 0, 0]],
             evidence=dict(exposed_extremity=True, safe_lift=True, hanging_structure=holding,
                           meaningful_extremity=holding, overlap_edge=False),
             fold_ready=dict.fromkeys(READY_KEYS, False))
    if p == 'LiftDrapeRelease':
        d['targets'] = [[.5, 0, .25, 3.14, 0, 0], [.5, .1, .25, 3.14, 0, 0], [.5, .1, .06, 3.14, 0, 0]]
    return d


def test_hierarchy_geometry_and_holding():
    assert command(decision()).startswith('motion.grasp')
    d = decision(); d['targets'][0][0] += .1
    with pytest.raises(ValueError, match='vertical'): command(d)
    d = decision('ShortDragReveal')
    with pytest.raises(ValueError, match='hierarchy'): command(d)
    d['evidence'].update(exposed_extremity=False, safe_lift=False)
    with pytest.raises(ValueError, match='limits'): command(d)
    d['targets'] = [[.52, 0, .05, 3.14, 0, 0]]
    assert command(d)
    code = command(decision('LiftDrapeRelease', True), holding=True, held_pose=[.5, 0, .2, 3.14, 0, 0])
    assert 'grasp' not in code and 'action="open"' in code
    with pytest.raises(ValueError, match='held continuation'):
        command(decision(), holding=True)


def test_progress_execution_success_is_not_progress():
    d = unknown('only global translation'); d.update(outcome='PROGRESS', whole_garment_translation_only=True)
    assert evaluation(d)['outcome'] == 'NO_PROGRESS'
    d = unknown('new sleeve'); d.update(outcome='PROGRESS', new_landmarks=['sleeve cuff'])
    assert evaluation(d)['outcome'] == 'STRUCTURE_REVEALED'
    d['new_landmarks'] = []
    with pytest.raises(ValueError): evaluation(d)


class Runtime:
    holding = False
    def __init__(self, status='EXECUTED'):
        self.status=status; self.executed=[]; self.captures=[]; self.stopped=False; self.closed=False
    def observe(self, path):
        path.mkdir(); (path/'rgb.png').write_bytes(b'fake rgb')
        self.captures.append((path, self.holding))
        return SimpleNamespace(directory=path, image=path/'rgb.png')
    def execute_primitive(self, primitive, source, before, folder):
        self.executed.append(source)
        self.holding = primitive == 'LiftAndReveal'
        return dict(status=self.status, physical_execution=True)
    def stop(self): self.stopped=True
    def close(self): self.closed=True


def model(answers, seen):
    def call(prompt, output, images, **kwargs):
        seen.append((prompt, images)); output.mkdir()
        (output/'answer.txt').write_text(json.dumps(answers.pop(0)))
        return {'status':'AGENT_FINISHED', 'session_id':'unfold-test-session'}
    return call


def test_lift_observe_drape_evaluate_fold_ready(tmp_path):
    r=Runtime(); seen=[]
    hanging=unknown('new sleeve'); hanging.update(outcome='STRUCTURE_REVEALED', new_landmarks=['sleeve'])
    final=unknown('torso open'); final.update(outcome='PROGRESS', fold_ready=dict.fromkeys(READY_KEYS, True))
    terminal=decision('FOLD_READY'); terminal.update(grasp=None,targets=[],fold_ready=dict.fromkeys(READY_KEYS,True))
    terminal['previous_progress']=final
    drape=decision('LiftDrapeRelease',True); drape['previous_progress']=hanging
    answers=[decision(), drape, terminal]
    result=rollout(r,tmp_path/'v1',{},model_call=model(answers,seen),max_actions=3)
    assert result['status']=='FOLD_READY' and r.closed
    assert [held for _,held in r.captures]==[False,True,False]
    assert result['metrics']['physical_actions']==2 and result['metrics']['gravity_assisted_actions']==2
    assert result['metrics']['actions_to_fold_ready']==2
    assert len(seen)==3 and len(seen[1][1])==2
    assert not list((tmp_path/'v1').glob('*/evaluator'))
    assert (tmp_path/'v1/unfolding_rollout_v1.md').exists()


def test_negative_evidence_and_repeat_rejected(tmp_path):
    r=Runtime(); seen=[]
    d=decision('ShortDragReveal'); d['evidence'].update(exposed_extremity=False,safe_lift=False)
    d['targets']=[[.52,0,.05,3.14,0,0]]
    no=unknown('wrinkle only'); no['outcome']='NO_PROGRESS'
    repeat=dict(d,previous_progress=no)
    result=rollout(r,tmp_path/'v1',{},model_call=model([d,repeat],seen))
    assert result['status']=='FAILED' and len(r.executed)==1
    assert len(result['failed_evidence'])==1 and result['metrics']['no_progress_ratio']==1
    assert 'NO_PROGRESS' in seen[-1][0]


def test_action_limit_does_not_trigger_extra_analysis(tmp_path):
    r=Runtime(); seen=[]
    result=rollout(r,tmp_path/'v1',{},model_call=model([decision()],seen),max_actions=1)
    assert result['status']=='ACTION_LIMIT' and len(seen)==1
    assert result['steps'][0]['after_rgb'] and result['metrics']['unknown_actions']==1


def test_fault_does_not_move_for_after_photo(tmp_path):
    r=Runtime('EXECUTION_FAILED')
    result=rollout(r,tmp_path/'v1',{},model_call=model([decision()],[]))
    assert result['status']=='EXECUTION_FAILED' and len(r.captures)==1
    assert result['steps'][0]['progress_evaluation']['outcome']=='UNKNOWN'
    assert result['metrics']['physical_actions']==1


def test_baseline_preserved_and_missing_completion_not_fabricated(tmp_path):
    source=tmp_path/'unfold_old'; folder=source/'round_001'; folder.mkdir(parents=True)
    (folder/'outcome.json').write_text(json.dumps(dict(status='EXECUTED',physical_execution=True)))
    (source/'result.json').write_text('{"status":"MODEL_FAILED"}')
    original=(folder/'outcome.json').read_bytes()
    frozen=freeze_baseline(source,tmp_path/'v0')
    assert frozen['label']=='Experience Graph v0' and frozen['metrics']['physical_actions']==1
    assert frozen['metrics']['no_progress_actions'] is None
    assert (folder/'outcome.json').read_bytes()==original
    with pytest.raises(FileExistsError): freeze_baseline(source,tmp_path/'v0')
    with pytest.raises(ValueError,match='No completed'): latest_completed([tmp_path])
    (source/'result.json').write_text('{"status":"MODEL_REPORTED_COMPLETE"}')
    assert latest_completed([tmp_path])==source


def test_cli_does_not_construct_runtime_without_execute(tmp_path):
    with pytest.raises(SystemExit): main(['run','--output',str(tmp_path/'out')])
    assert not (tmp_path/'out').exists()


def test_runtime_validates_full_plan_then_retains_grasp(monkeypatch):
    from cloth_agent.experiments.gravity_runtime import GravityRuntime
    from cloth_agent.codex_arm7_runtime import Arm7Runtime
    from cloth_agent.dual_arm.sequential_home import PositionSegment
    move=PositionSegment('right', {}, {}, 10., 20., 30.)
    ops=[('gripper',dict(action='open')), ('move',move), ('gripper',dict(action='close')),
         ('move',move), ('gripper',dict(action='open')), ('move',move)]
    calls=[]
    def plan(self,source,initial,**kwargs):
        calls.append(kwargs); return ops
    monkeypatch.setattr(Arm7Runtime,'plan',plan)
    runtime=object.__new__(GravityRuntime); runtime.primitive='LiftAndReveal'
    retained=runtime.plan('',{})
    assert len(retained)==4 and retained[-1][0]=='move'
    assert retained[-1][1].speed==pytest.approx(10.)
    assert retained[-1][1].acceleration==pytest.approx(20.)
    assert retained[-1][1].timeout==pytest.approx(30.)
    assert calls==[dict(preserve_command_boundaries=True)]
    runtime.primitive='PeelOverlap'
    assert len(runtime.plan('',{}))==len(ops)
    runtime.primitive='ShortDragReveal'
    assert runtime.plan('',{})[1][1].speed==pytest.approx(3.5/1.30)


def test_hanging_observation_does_not_prepare_or_release(monkeypatch, tmp_path):
    from cloth_agent.experiments.gravity_runtime import GravityRuntime
    from cloth_agent.codex_arm7_runtime import Arm7Runtime
    from cloth_agent.dual_arm import observation
    def forbidden(*args):
        pytest.fail('hanging capture called ordinary observation/preparation')
    monkeypatch.setattr(Arm7Runtime, 'observe', forbidden)
    sentinel=SimpleNamespace(directory=tmp_path/'hanging', image=tmp_path/'hanging/rgb.png', meta={})
    monkeypatch.setattr(observation, 'capture', lambda *args, **kwargs: sentinel)
    runtime=object.__new__(GravityRuntime)
    runtime.holding=True; runtime.config=SimpleNamespace(raw={'camera':{'support_height_patch':{'enabled':True}}, 'calibration_id':'test'}); runtime.models=None; runtime.camera=None
    runtime.coordinator=SimpleNamespace(snapshots=None)
    observed=runtime.observe(tmp_path/'hanging')
    assert observed.observation is sentinel
    assert 'support_height_patch' not in observed.capture_config.raw['camera']
    assert 'support_height_patch' in runtime.config.raw['camera']
    assert observed.capture_config.raw['calibration_id']=='test_hanging_unpatched'


def test_no_fold_ready_while_holding():
    d=decision('FOLD_READY'); d.update(grasp=None, targets=[], fold_ready=dict.fromkeys(READY_KEYS,True))
    with pytest.raises(ValueError,match='released'): command(d,holding=True)


def test_resume_baseline_preserves_and_validates_evidence(tmp_path):
    source=tmp_path/'old'; folder=source/'round_001'; folder.mkdir(parents=True)
    evidence=folder/'outcome.json'
    evidence.write_text(json.dumps(dict(status='EXECUTED',physical_execution=True)))
    destination=tmp_path/'review'
    freeze_baseline(source,destination)
    frozen=destination/'evidence/round_001/outcome.json'
    before=frozen.stat().st_mtime_ns
    assert freeze_baseline(source,destination,resume=True)['metrics']['physical_actions']==1
    assert frozen.stat().st_mtime_ns==before
    frozen.write_text('changed')
    with pytest.raises(ValueError,match='differs'): freeze_baseline(source,destination,resume=True)
    assert frozen.read_text()=='changed'


def test_completed_review_reused_with_structured_area(tmp_path):
    from cloth_agent.experiments.gravity_unfold import baseline_review_answer
    folder=tmp_path/'evaluate_round_001'; folder.mkdir()
    progress=unknown('unchanged'); progress['outcome']='NO_PROGRESS'
    progress['visible_area_change']=dict(viewpoint_comparability='same view', evidence='same contour')
    answer=dict(progress_evaluation=progress,gravity_assisted=None,gravity_evidence='not visible')
    (folder/'answer.txt').write_text(json.dumps(answer))
    (folder/'result.json').write_text('{"status":"AGENT_FINISHED"}')
    (folder/'prompt.txt').write_text('本次仅做只读分析，不连接或操作机器人，不修改项目文件。\nprompt\n')
    def forbidden(*args,**kwargs): pytest.fail('unnecessary model call')
    result=baseline_review_answer('prompt',[],folder,forbidden)
    assert result['progress_evaluation']['outcome']=='NO_PROGRESS'
    assert 'same contour' in result['progress_evaluation']['visible_area_change']
    assert json.loads((folder/'answer.txt').read_text())==answer


def test_failed_review_gets_new_attempt_directory(tmp_path):
    from cloth_agent.experiments.gravity_unfold import baseline_review_answer
    folder=tmp_path/'review'; folder.mkdir(); (folder/'answer.txt').write_text('incomplete')
    seen=[]
    result=baseline_review_answer('prompt',[],folder,model([{'ok': True}],seen))
    assert result=={'ok':True}
    assert (folder/'answer.txt').read_text()=='incomplete'
    assert (tmp_path/'review_retry_001/answer.txt').exists()


def test_hanging_observation_provenance_and_grasp_rejection(tmp_path):
    from cloth_agent.experiments.gravity_runtime import GravityRuntime, HangingObservation
    from cloth_agent.dual_arm.observation import Observation
    observation=Observation(tmp_path, {'calibration_id':'base_hanging_unpatched','synthetic':False}, None)
    config=SimpleNamespace(raw={'calibration_id':'base_hanging_unpatched'})
    wrapped=HangingObservation(observation,config,'base')
    wrapped.validate_for(SimpleNamespace(raw={'calibration_id':'base'}),live=True,check_age=False)
    with pytest.raises(ValueError,match='parent calibration'):
        wrapped.validate_for(SimpleNamespace(raw={'calibration_id':'other'}))
    runtime=object.__new__(GravityRuntime); runtime.holding=True
    with pytest.raises(ValueError,match='held drape'):
        runtime.execute_primitive('LiftAndReveal',command(decision()),wrapped,tmp_path)
    with pytest.raises(ValueError,match='new grasp'): wrapped.sample([3,3],1)


def test_progress_structured_evidence_with_additional_fields():
    value=unknown('same contour')
    value['visible_area_change']=dict(viewpoint_comparability='same view', change='none', evidence='same contour')
    value['outcome']='NO_PROGRESS'
    assert evaluation(value)['outcome']=='NO_PROGRESS'
    assert 'same contour' in value['visible_area_change']
    for invalid in ({}, {'evidence': ''}, {'evidence': True}, {'amount': 3}):
        value=unknown('insufficient'); value['visible_area_change']=invalid
        with pytest.raises(ValueError): evaluation(value)


def test_nonholding_observation_uses_original_unfold(monkeypatch,tmp_path):
    from cloth_agent.experiments.gravity_runtime import GravityRuntime
    from cloth_agent.codex_arm7_runtime import Arm7Runtime
    seen=[]; sentinel=object()
    def original(self,path): seen.append(path); return sentinel
    monkeypatch.setattr(Arm7Runtime,'observe',original)
    runtime=object.__new__(GravityRuntime); runtime.holding=False
    assert runtime.observe(tmp_path/'initial') is sentinel
    assert seen==[tmp_path/'initial']


@pytest.mark.parametrize('flags,expected_wait', [([],False), (['--auto-execute'],False), (['--manual-execute'],True)])
def test_main_observes_before_offline_review_and_closes_robot(monkeypatch,tmp_path,flags,expected_wait):
    import cloth_agent.experiments.gravity_unfold as module
    import cloth_agent.experiments.gravity_runtime as runtime_module
    import cloth_agent.trajectory_preview as preview_module
    events=[]
    class Fake(Runtime):
        config=None
        def __init__(self,*args): super().__init__(); events.append('runtime')
        def close(self): events.append('close')
    class Preview:
        def __init__(self,*args,**kwargs):
            assert kwargs['wait'] is expected_wait
        def close(self): events.append('preview_close')
    monkeypatch.setattr(runtime_module,'GravityRuntime',Fake)
    monkeypatch.setattr(preview_module,'TrajectoryPreview',Preview)
    def fake_rollout(runtime,output,baseline,**kwargs):
        events.append('observe'); return dict(status='FOLD_READY',metrics={})
    def review(*args): events.append('review')
    monkeypatch.setattr(module,'rollout',fake_rollout)
    monkeypatch.setattr(module,'finish_baseline_review',review)
    baseline=tmp_path/'baseline.json'
    baseline.write_text(json.dumps(dict(label='Experience Graph v0',files=[])))
    assert main(['run','--baseline',str(baseline),'--output',str(tmp_path/'run'),'--execute',*flags])==0
    assert events==['runtime','observe','close','preview_close','review']


def test_offline_review_failure_preserves_rollout(monkeypatch,tmp_path):
    import cloth_agent.experiments.gravity_unfold as module
    def fail(*args,**kwargs): raise ValueError('bad review response')
    monkeypatch.setattr(module,'freeze_baseline',fail)
    record=dict(status='FOLD_READY',baseline={},steps=[],failed_evidence=[])
    module.finish_baseline_review(record,tmp_path,tmp_path/'v0.json')
    saved=json.loads((tmp_path/'unfolding_rollout_v1.json').read_text())
    assert saved['status']=='FOLD_READY' and saved['baseline_review_status']=='PENDING'


def test_user_verified_speed_applies_before_planning_without_global_changes(monkeypatch):
    from copy import deepcopy
    import numpy as np
    from cloth_agent.experiments.gravity_runtime import GravityRuntime
    from cloth_agent.codex_arm7_runtime import Arm7Runtime
    from cloth_agent.dual_arm.sequential_home import command_limits
    raw=dict(limits=dict(joint_speed_deg_s=5.,joint_accel_deg_s2=10.),
             safety=dict(arms={k:dict(max_joint_speed_deg_s=5.,stop_excursion_deg=[1.25]*n)
                              for k,n in [('left',6),('right',7)]}))
    original=deepcopy(raw)
    config=SimpleNamespace(raw=raw,limits=raw['limits'])
    monkeypatch.setattr(Arm7Runtime,'load_config',lambda self,path:config)
    runtime=object.__new__(GravityRuntime)
    changed=runtime.load_config(None)
    assert raw==original
    assert changed.limits['joint_speed_deg_s']==5.
    assert changed.raw['gravity_speed_policy']['joint_speed_deg_s']==6.5
    assert changed.limits['joint_accel_deg_s2']==10.
    assert changed.raw['safety']['arms']['left']==original['safety']['arms']['left']
    assert changed.raw['safety']['arms']['right']['stop_excursion_deg']==pytest.approx([2.1125]*7)
    speed,accel,timeout=command_limits(changed,None,SimpleNamespace(arm_id='right'),np.zeros(7),np.ones(7))
    assert speed==5. and accel==10. and timeout>0


def test_real_gravity_config_passes_startup_and_retimes_only_arm7(monkeypatch):
    import numpy as np
    from cloth_agent.experiments.gravity_runtime import GravityRuntime
    from cloth_agent.codex_native_runtime import NativeRuntime
    from cloth_agent.dual_arm.sequential_home import PositionSegment
    from cloth_agent.codex_astra_pipeline import ROOT
    runtime=object.__new__(GravityRuntime)
    runtime.config=runtime.load_config(ROOT/'config/dual_arm.local.json')
    runtime.config.execution_mode='controller_sequential'
    # Reproduce the exact failing startup check without constructing hardware.
    runtime.config.require_real(homing=True)
    runtime.models={'right':None}
    start={'right':np.zeros(7),'left':np.zeros(6)}
    end={'right':np.ones(7),'left':np.ones(6)}
    operations=[('move',PositionSegment(arm,start,end,5.,10.,60.)) for arm in ('right','left')]
    monkeypatch.setattr(NativeRuntime,'_join_moves',lambda self,ops,boundaries:ops)
    result=runtime._join_moves(operations)
    assert result[0][1].speed==6.5
    assert result[0][1].acceleration==10.
    assert result[0][1].timeout>0
    assert result[1][1].speed==5.
    assert runtime.config.raw['safety']['arms']['left']['max_joint_speed_deg_s']==5.
    assert runtime.config.raw['safety']['arms']['right']['max_joint_speed_deg_s']==6.5


def test_rollout_reuses_planner_session_without_evaluator_calls(tmp_path):
    runtime=Runtime(); sessions=[]
    terminal=decision('BLOCKED'); terminal.update(grasp=None,targets=[],previous_progress=unknown('occluded'))
    inner=model([decision(),terminal],[])
    def call(prompt,output,images,**kwargs):
        assert output.name=='planner'
        sessions.append(kwargs.get('session_id'))
        return inner(prompt,output,images,**kwargs)
    result=rollout(runtime,tmp_path/'v1',{},model_call=call,max_actions=2)
    assert result['status']=='ACTION_LIMIT'
    assert len(result['blocked_decisions'])==1
    assert sessions==[None,'unfold-test-session']
    assert len(runtime.executed)==1


def test_blocked_continues_same_session_without_moving(tmp_path):
    r=Runtime(); seen=[]
    blocked=decision('BLOCKED'); blocked.update(grasp=None,targets=[])
    result=rollout(r,tmp_path/'v1',{},model_call=model([blocked,decision()],seen),max_actions=2)
    assert result['status']=='ACTION_LIMIT'
    assert len(result['blocked_decisions'])==1
    assert len(r.executed)==1 and len(seen)==2
    assert len(r.captures)==3


@pytest.mark.parametrize('status,holding',[('ACTION_LIMIT',True),('FAILED',False),('INTERRUPTED',True),('BLOCKED',True),('FOLD_READY',True)])
def test_incomplete_rollout_never_starts_baseline_review(monkeypatch,tmp_path,status,holding):
    import cloth_agent.experiments.gravity_unfold as module
    def forbidden(*args,**kwargs): pytest.fail('offline model review must not start')
    monkeypatch.setattr(module,'freeze_baseline',forbidden)
    record=dict(status=status,holding_at_stop=holding,baseline={},steps=[],failed_evidence=[])
    module.finish_baseline_review(record,tmp_path,tmp_path/'v0.json')
    assert record['baseline_review_status']=='DEFERRED'
    assert record['status']==status
