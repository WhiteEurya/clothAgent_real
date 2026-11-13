from types import SimpleNamespace
import json

import numpy as np
from PIL import Image
import pytest

from cloth_agent.experiments import partitioned_unfold as unfold
from cloth_agent.dual_arm.geometry import pose_matrix, DualArmError
from cloth_agent.codex_motion import parse_code
from cloth_agent.experiment import ExperimentRunner


def action():
    return dict(decision='ACTION', grasp_px=[40,40], target_px=[60,40],
                motion_type='lift_pull_place', short_reason='展开衣袖',
                grasp_angle_deg=90., is_smoothing=False,
                action_purpose='展开衣袖', expected_stop_state='衣袖展开')


@pytest.mark.parametrize('arm', ['left','right'])
@pytest.mark.parametrize('smoothing', [False,True])
@pytest.mark.parametrize('skill', [None, 'garment-chord-grasp', 'garment-tangent-grasp', 'cloth-contour-normal-grasp', 'make-space-for-skill'])
def test_current_geometry_and_single_contact_offset(tmp_path, arm, smoothing, skill):
    runtime=unfold.configure_runtime(unfold.PartitionedReplayRuntime(unfold.DEFAULT_CONFIG,planning_only=True))
    joints={k:a.home_joints.copy() for k,a in runtime.config.arms.items()}
    a=(runtime.config.arms[arm].world_from_base @ pose_matrix(runtime.models[arm].forward(joints[arm])))[:3,3]
    b=a+np.array([20.,0.,3.])
    picture=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(picture)
    camera=np.eye(4);camera[:3,:3]=np.diag([1.,-1.,-1.]);camera[:3,3]=a+[0.,0.,500.]
    before=SimpleNamespace(image=picture,meta=dict(joints_deg=joints,intrinsics=[[500,0,50],[0,500,50],[0,0,1]],world_from_camera_mm=camera.tolist()),
                           sample=lambda pixel,spread:a if pixel[0]==40 else b,
                           validate_for=lambda *a,**k:None)
    plan=action();plan['is_smoothing']=smoothing
    (tmp_path/'planning_skills.json').write_text(json.dumps({'selected_skill':skill}))
    try:
        source=unfold.prepare_action(runtime,before,plan,tmp_path)
        config,program,assignment=runtime.compile(source,before)
        assert assignment['active_arm']==arm
        points=[np.array(args['right'][:3])*1000 for _,args in parse_code(source,allow_grasp=True)]
        np.testing.assert_allclose(points[0],a)
        assert points[1][2]==pytest.approx(b[2]+(10 if smoothing else (30 if skill in ('garment-chord-grasp','garment-tangent-grasp','cloth-contour-normal-grasp') else 20)))
        np.testing.assert_allclose(points[1][:2],a[:2])
        np.testing.assert_allclose(points[2][:2],b[:2])
        assert points[2][2]==points[1][2]
        np.testing.assert_allclose(points[-1],b+[0,0,5])
        runner=ExperimentRunner(tmp_path/'preview',config)
        script=runner.workspace/'action.py';script.write_text(program)
        preflight=runner.preflight(script)
        assert preflight.error is None
        base=runtime.models[arm].forward(joints[arm])[:3]
        up=runtime.config.arms[arm].world_from_base[:3,:3].T @ [0,0,1]
        assert [a['name'] for a in preflight.actions[:4]] == ['home', 'open_gripper', 'move', 'move']
        np.testing.assert_allclose(preflight.actions[3]['actual_ee_pose'][:3],base-20*up)
        assert preflight.actions[-1]['name']=='home'
    finally:runtime.close()


@pytest.mark.parametrize('outcome', ['EXECUTED','EXECUTION_FAILED'])
def test_serial_loop_stops_after_failure_or_done(tmp_path,monkeypatch,outcome):
    events=[]
    before=SimpleNamespace(directory=tmp_path,image=tmp_path/'rgb.png')
    Image.new('RGB',(100,100)).save(before.image)
    runtime=SimpleNamespace(observe=lambda path:events.append('observe') or before,
        execute=lambda *a:events.append('execute') or dict(status=outcome,physical_execution=True),
        close=lambda:events.append('close'))
    monkeypatch.setattr(unfold,'prepare_action',lambda *a:'source')
    def request(folder,before,context):
        events.append('request')
        return (action() if context is None else dict(decision='DONE')),{}
    result=unfold.episode(runtime,tmp_path,max_actions=3,request=request)
    assert result['physical_actions']==1
    if outcome=='EXECUTED':
        assert result['status']=='MODEL_REPORTED_DONE'
        assert events==['observe','request','execute','observe','request','close']
    else:
        assert result['status']=='EXECUTION_FAILED'
        assert events==['observe','request','execute','close']


def test_transient_capture_failure_retries_without_planning_or_motion(tmp_path):
    calls=[];image=tmp_path/'rgb.png';Image.new('RGB',(20,20)).save(image)
    before=SimpleNamespace(directory=tmp_path/'captured',image=image)
    def observe(path):
        calls.append(path)
        if len(calls)<3:raise RuntimeError("Frame didn't arrive within 5000")
        return before
    runtime=SimpleNamespace(observe=observe,execute=lambda *a:pytest.fail('no action'),close=lambda:None)
    result=unfold.episode(runtime,tmp_path,max_actions=2,
        request=lambda *a:(dict(decision='DONE',short_reason='complete'),{}))
    assert result['status']=='MODEL_REPORTED_DONE' and len(calls)==3
    recovery=json.loads((tmp_path/'initial_observation_capture_recovery.json').read_text())
    assert [row['status'] for row in recovery['attempts']]==['FAILED','FAILED','CAPTURED']


def test_controller_error_is_never_treated_as_camera_retry(tmp_path):
    calls=[]
    def observe(path):
        calls.append(path);raise DualArmError('controller not stationary/ready')
    runtime=SimpleNamespace(observe=observe,close=lambda:None)
    with pytest.raises(DualArmError,match='controller'):
        unfold.episode(runtime,tmp_path,max_actions=1,request=lambda *a:pytest.fail('no planning'))
    assert len(calls)==1


def test_planner_service_backoff_only_for_explicit_transient_status():
    assert unfold.planner_retry_delay(RuntimeError('429 Too Many Requests'),0)==30
    assert unfold.planner_retry_delay(RuntimeError('429 Too Many Requests'),1)==60
    assert unfold.planner_retry_delay(RuntimeError('503 Service Unavailable'),0)==10
    assert unfold.planner_retry_delay(RuntimeError('invalid schema'),0)==0


def test_identical_action_without_image_change_not_dispatched_twice(tmp_path,monkeypatch):
    calls=[];before=SimpleNamespace(directory=tmp_path,image=tmp_path/'rgb.png')
    Image.new('RGB',(100,100)).save(before.image)
    runtime=SimpleNamespace(observe=lambda path:before,
        execute=lambda *a:calls.append('execute') or dict(status='EXECUTED',physical_execution=True),close=lambda:None)
    monkeypatch.setattr(unfold,'prepare_action',lambda *a:'source')
    result=unfold.episode(runtime,tmp_path,max_actions=3,request=lambda *a:(action(),{}))
    assert calls==['execute']
    assert result['status']=='PLANNING_REJECTED'
    assert 'repeated action' in result['error']
    assert len(result['steps'][-1]['planning_rejections'])==3


def test_planner_adds_previous_topdown_without_previous_action(tmp_path,monkeypatch):
    from cloth_agent.experiments import unfold_topdown as td
    monkeypatch.setattr(td,'primary',lambda before,folder:before)
    (tmp_path/'goal').mkdir();folder=tmp_path/'round';folder.mkdir()
    image=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(image)
    history_image=tmp_path/'previous.png';Image.new('RGB',(100,100),'red').save(history_image)
    Image.new('RGB',(100,100)).save(tmp_path/'goal/target.png')
    (tmp_path/'hem_sleeve_pull_rule.json').write_text('{"enabled": false, "rgb_only": true}')
    before=SimpleNamespace(image=image,raw_plan=lambda plan:dict(plan))
    (folder/'rejection_feedback.json').write_text(json.dumps(dict(rejection_reason="XY_TRANSLATION_OUT_OF_RANGE",distance_mm=168.3276,maximum_mm=150)))
    def direct(folder,body,headers,timeout):
        assert '168.3276' in body['instructions']
        assert '不设固定拉距上下限' in body['instructions']
        assert '本次候选未执行' in body['instructions']
        from cloth_agent.experiments.unfold_completion import ARM6_DONE_RULE
        assert ARM6_DONE_RULE in body['instructions']
        assert '衣领必须显露' not in body['instructions']
        assert '袖子相对衣身的伸出长度应尽量接近目标图' not in body['instructions']
        assert '完整衣物外轮廓明确可辨' not in body['instructions']
        assert 'Arm6 基座坐标' not in body['instructions']
        assert '-250.578537' not in body['instructions']
        assert '优先保留原抓点' in body['instructions']
        assert '不代表抓点或目标点越界' in body['instructions']
        assert 'world_mm是标定世界坐标' in body['instructions']
        assert '两台 Arm7' in body['instructions']
        for name in ('cloth-contour-normal-grasp','garment-tangent-grasp','garment-chord-grasp'):
            assert f'【规划参考 skill：{name}】' in body['instructions']
        assert '/home/sja/' not in body['instructions']
        assert 'contour_normal_grasp_once' not in body['instructions']
        assert 'Simple只能抓当前图中可见的衣袖或者下摆' in body['instructions']
        assert '结构距离报告' not in body['instructions']
        assert body['text']['format']['schema']['properties']['skill']
        from cloth_agent.experiments.one_shot_goal_planner_probe_v1 import image_data
        images=[content['image_url'] for content in body['input'][0]['content'] if content['type']=='input_image']
        assert images==[image_data(image),image_data(tmp_path/'goal/target.png'),image_data(history_image)]
        assert '第三张图是上一轮动作前' in body['instructions']
        assert '不提供历史图' not in body['instructions']
        assert '所有抓放点、边界点和夹爪角度只使用第一张图坐标' in body['instructions']
        assert '上一轮实际执行状态' not in json.dumps(body,ensure_ascii=False)
        return dict(decision='DONE',skill=None,region_name='',evidence='已经展开',
            boundary_point_px=None,chord_endpoints_px=[],
            grasp_angle_deg=None,pull_distance_mm=10,grasp_px=None,target_px=None),{'wall_latency_s':0.}
    plan,_=unfold.plan_round(tmp_path,folder,before,{},1,direct,
        dict(plan=action(),before_image=image,completed_commands=11,planning_plan=action(),planning_before_image=history_image))
    assert plan['decision']=='DONE'
    assert (folder/'proposed_action_overlay.png').exists()
    manifest=json.loads((folder/'planning_skills.json').read_text())
    assert {x['name'] for x in manifest['skills']}=={'cloth-contour-normal-grasp','garment-tangent-grasp','garment-chord-grasp'}
    assert manifest['local_cv_executed'] is False
    assert (folder/'planner_prompt.txt').exists()


@pytest.mark.parametrize('arm',['left','right'])
@pytest.mark.parametrize('surface_z',[50.,130.6])
def test_reset_drops_on_midline_with_selected_arm(tmp_path,arm,surface_z):
    from cloth_agent.random_fold_reset import sample_release
    from cloth_agent.dual_arm.geometry import matrix_pose
    runtime=unfold.configure_runtime(unfold.PartitionedReplayRuntime(unfold.DEFAULT_CONFIG,planning_only=True))
    try:
        joints={k:a.home_joints.copy() for k,a in runtime.config.arms.items()}
        home=matrix_pose(runtime.config.arms[arm].world_from_base @ pose_matrix(runtime.models[arm].forward(joints[arm])))
        grasp=home[:3].copy();grasp[2]=surface_z
        before=SimpleNamespace(meta={'joints_deg':joints})
        release=dict(release_world_z_mm=325.,yaw_delta=0.,drop_offset_xy=[0.,0.])
        source=unfold.prepare_reset(runtime,before,dict(world_xyz_mm=grasp.tolist()),release,tmp_path)
        points=[np.array(args['right'][:3])*1000 for _,args in parse_code(source,allow_grasp=True)]
        np.testing.assert_allclose(points[0],grasp)
        np.testing.assert_allclose(points[1],np.r_[grasp[:2],surface_z+70.])
        assert len(points)==4
        np.testing.assert_allclose(points[3],np.r_[points[2][:2],325.])
        geometry=json.loads((tmp_path/'geometry.json').read_text())
        routing=geometry['assignment']
        axis=np.array(routing['right_axis_xy'])
        midpoint=np.array(routing['midpoint_world_mm'])[:2]
        assert (points[2][:2]-midpoint) @ axis == pytest.approx(0.,abs=1e-8)
        np.testing.assert_allclose(points[2][:2],grasp[:2]-((grasp[:2]-midpoint) @ axis)*axis)
        assert np.linalg.norm(points[2][:2]-home[:2])>1.
        assert points[2][2]==points[1][2]
        config,program,assignment=runtime.compile(source,before,post_release_descent_mm=50.)
        assert assignment['active_arm']==arm
        contact=runtime.config.arms[arm].world_from_base @ np.r_[assignment['contact_base_mm_deg'][:3],1.]
        np.testing.assert_allclose(contact[:3],grasp-[0,0,20])
        runner=ExperimentRunner(tmp_path/'preview',config)
        script=runner.workspace/'action.py';script.write_text(program)
        result=runner.preflight(script)
        assert result.error is None
        retreat=runtime.config.arms[arm].world_from_base @ np.r_[result.actions[-2]['actual_ee_pose'][:3],1.]
        np.testing.assert_allclose(retreat[:3],points[-1]-[0,0,50.])
        assert [a['name'] for a in result.actions]==['home','open_gripper','move','move','close_gripper','move','move','move','open_gripper','move','home']
    finally:runtime.close()


@pytest.mark.parametrize('success',[False,True])
def test_reset_before_planning_fresh_observation_and_failure_stop(tmp_path,success):
    events=[]
    runtime=SimpleNamespace(observe=lambda p:events.append(p.name) or SimpleNamespace(directory=p),close=lambda:events.append('close'))
    def reset(before,folder):
        events.append('reset')
        return dict(status='EXECUTED' if success else 'PLANNING_FAILED',physical_execution=success)
    def request(folder,before,context):
        assert before.directory.name=='after_reset'
        assert context is None
        events.append('request')
        return dict(decision='DONE'),{}
    result=unfold.episode(runtime,tmp_path,max_actions=1,request=request,reset_request=reset)
    assert result['physical_actions']==int(success)
    assert events==(['initial_observation','reset','after_reset','request','close'] if success else ['initial_observation','reset','close'])
    assert result['status']==('MODEL_REPORTED_DONE' if success else 'RESET_FAILED')


def test_unreliable_reset_region_never_dispatches(tmp_path):
    image=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(image)
    before=SimpleNamespace(image=image,validate_for=lambda *a,**k:None)
    runtime=SimpleNamespace(config=None,execute=lambda *a:pytest.fail('must not execute'))
    def direct(*args):return dict(center_px=[50,50],radius_px=0,confidence=.5,reason='occluded'),{}
    with pytest.raises(DualArmError):
        unfold.initialize(runtime,before,tmp_path/'reset',headers={},timeout=1,direct=direct,seed=123,radius_mm=20)
    assert json.loads((tmp_path/'reset/result.json').read_text())['status']=='ERROR'


def test_reset_sampling_dispatch_and_settling(tmp_path,monkeypatch):
    image=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(image)
    before=SimpleNamespace(image=image,validate_for=lambda *a,**k:None,
        sample=lambda pixel,spread:np.array([pixel[0],pixel[1],50.]))
    events=[]
    runtime=SimpleNamespace(config=None,execute=lambda *a,**kw:events.append('execute') or dict(status='EXECUTED',physical_execution=True))
    def prepare(runtime,before,point,release,folder):
        assert np.linalg.norm(np.array(point['world_xyz_mm'])[:2]-[50,50])<=20
        assert release['release_world_z_mm']==325.
        assert release['drop_offset_xy']==[0.,0.]
        events.append('prepare');return 'source'
    monkeypatch.setattr(unfold,'prepare_reset',prepare)
    monkeypatch.setattr(unfold.time,'sleep',lambda seconds:events.append(('settle',seconds)))
    def direct(*args):return dict(center_px=[50,50],radius_px=20,confidence=.9,reason='visible body'),{}
    result=unfold.initialize(runtime,before,tmp_path/'reset',headers={},timeout=1,direct=direct,seed=123,radius_mm=20)
    assert result['status']=='EXECUTED'
    assert result['physical_execution']
    assert events==['prepare','execute',('settle',2.)]


@pytest.mark.parametrize('profile',['body-up','body-drop'])
def test_body_up_reset_plans_both_paths_before_any_motion(tmp_path,monkeypatch,profile):
    from cloth_agent.experiments import reset_parking
    image=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(image)
    before=SimpleNamespace(image=image,validate_for=lambda *a,**k:None,
        sample=lambda pixel,spread:np.array([pixel[0],pixel[1],50.]))
    parked=object();events=[]
    runtime=SimpleNamespace(config=None,reset_profile=profile)
    def prepare(runtime,before,point,release,folder,**kw):
        assert release['release_world_z_mm']==400.
        expected=dict(grasp_yaw_offset_deg=90.,release_world_z_mm=400.,far_translation_mm=100.)
        if profile=='body-drop':expected.update(far_translation_mm=0.,release_at_grasp=False)
        assert kw==expected
        events.append('prepare');return 'source'
    def find(*a):events.append('check_both_paths');return {'arm':'left'}
    def park(*a):events.append('park');return parked
    def execute(source,observation,*a,**kw):
        assert observation is parked
        events.append('drop');return dict(status='EXECUTED',physical_execution=True)
    runtime.execute=execute
    monkeypatch.setattr(unfold,'prepare_reset',prepare)
    monkeypatch.setattr(reset_parking,'find_parking',find)
    monkeypatch.setattr(reset_parking,'execute_parking',park)
    monkeypatch.setattr(unfold.time,'sleep',lambda _:None)
    calls=[]
    def direct(folder,body,*args):
        calls.append(body)
        if profile=='body-up':
            assert 'one third' in body['instructions']
        if profile=='body-drop':
            assert 'one third' not in body['instructions']
            assert 'halfway from neckline to hem' in body['instructions']
            assert 'do not shift to one side to avoid a print' in body['instructions']
            assert '100 mm' not in body['instructions']
            assert 'moves to the arm partition line at that clearance height, then raises to world Z=400 mm' in body['instructions']
        return dict(center_px=[50,50],radius_px=10,confidence=.9,reason='visible body'),{}
    unfold.initialize(runtime,before,tmp_path/'reset',headers={},timeout=1,direct=direct,seed=123,radius_mm=0)
    assert len(calls)==1
    assert events==['prepare','check_both_paths','park','drop']
    def reject(*a):raise ValueError('path rejected')
    monkeypatch.setattr(reset_parking,'find_parking',reject)
    events.clear()
    with pytest.raises(ValueError,match='path rejected'):
        unfold.initialize(runtime,before,tmp_path/'reject',headers={},timeout=1,direct=direct,seed=123,radius_mm=0)
    assert events==['prepare']


def test_reset_midline_shift_preserves_grasp_lift_and_moves_release_leg():
    from cloth_agent.experiments.reset_parking import shift_midline_drop
    source = ('motion.grasp(right=[0.4, -0.2, 0.05, 0, 0, 0])\n'
              'motion.move_tcp(right=[0.4, -0.2, 0.4, 0, 0, 0])\n'
              'motion.move_tcp(right=[0.5, -0.2, 0.4, 0, 0, 0])\n'
              'motion.move_tcp(right=[0.5, -0.3, 0.4, 0, 0, 0])\n')
    calls = parse_code(shift_midline_drop(source,[0.,1.],20.),allow_grasp=True)
    assert calls[0][1]['right'][:3] == [0.4,-0.2,0.05]
    assert calls[1][1]['right'][:3] == [0.4,-0.2,0.4]
    np.testing.assert_allclose(calls[2][1]['right'][:3],[0.5,-0.18,0.4])
    np.testing.assert_allclose(calls[3][1]['right'][:3],[0.5,-0.28,0.4])


@pytest.mark.parametrize('always_reject',[False,True])
def test_translation_feedback_replans_without_motion(tmp_path,monkeypatch,always_reject):
    image=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(image)
    before=SimpleNamespace(directory=tmp_path,image=image)
    dispatched=[];requests=[]
    runtime=SimpleNamespace(observe=lambda path:before,close=lambda:None,
        execute=lambda *a:dispatched.append(a) or dict(status='EXECUTED',physical_execution=True))
    def prepare(*args):
        if always_reject or len(requests)==1:
            raise unfold.TranslationRejected(dict(distance_mm=168.3,maximum_mm=150))
        return 'source'
    def request(folder,obs,context):
        if context and context.get('final_completion_only'):
            return dict(decision='ABSTAIN',short_reason='not complete'),{}
        assert obs is before
        if requests:
            feedback=json.loads((folder/'rejection_feedback.json').read_text())
            assert feedback['distance_mm']==168.3
            assert feedback['rejection_reason']=='XY_TRANSLATION_OUT_OF_RANGE'
            assert feedback['coordinate_frame']=='world_mm'
            assert feedback['workspace_check']=='NOT_EVALUATED'
            assert not dispatched
        requests.append(folder)
        return action(),{}
    monkeypatch.setattr(unfold,'prepare_action',prepare)
    result=unfold.episode(runtime,tmp_path,max_actions=1,request=request)
    assert len(requests)==(3 if always_reject else 2)
    assert len(dispatched)==(0 if always_reject else 1)
    assert result['status']==('PLANNING_REJECTED' if always_reject else 'ACTION_LIMIT')


@pytest.mark.parametrize('decision',['ACTION','DONE','ABSTAIN'])
@pytest.mark.parametrize('with_history',[False,True])
def test_pure_oneshot_images_and_one_call(tmp_path,monkeypatch,decision,with_history):
    from cloth_agent.experiments import unfold_topdown as td, unfold_planning_skills as skills
    from cloth_agent.experiments.unfold_skill_planner import SELECT_SCHEMA
    monkeypatch.setattr(td,'primary',lambda before,folder:before)
    def forbidden(*args,**kwargs):raise AssertionError('Pure baseline must not load skills')
    monkeypatch.setattr(skills,'append',forbidden)
    (tmp_path/'goal').mkdir();folder=tmp_path/'round';folder.mkdir()
    image=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(image)
    Image.new('RGB',(100,100)).save(tmp_path/'goal/target.png')
    before=SimpleNamespace(image=image,raw_plan=lambda plan:dict(plan),sample=lambda p,s:np.r_[p,50.])
    calls=[]
    def direct(folder,body,headers,timeout):
        calls.append(body)
        assert '把第一张图中的衣物展平' in body['instructions']
        assert body['text']['format']['schema']==SELECT_SCHEMA
        assert '【规划参考 skill' not in body['instructions']
        assert len(body['input'])==1
        assert sum(x['type']=='input_image' for x in body['input'][0]['content'])==(3 if with_history else 2)
        assert body['tools']==[]
        active=decision=='ACTION'
        return dict(decision=decision,grasp_px=[20,30] if active else None,
                    target_px=[40,30] if active else None,skill=None,boundary_point_px=None,
                    chord_endpoints_px=[],grasp_angle_deg=None,pull_distance_mm=10,region_name='衣物',
                    evidence='第一句。第二句。'),dict(wall_latency_s=1.)
    plan,_=unfold.plan_round(tmp_path,folder,before,{},1,direct,
        previous_context={'before_image':image} if with_history else None,planning_mode='pure-oneshot')
    assert len(calls)==1 and plan['decision']==decision
    assert plan['is_smoothing'] is False
    assert (folder/'pixel_mapping.json').exists()
    assert json.loads((folder/'planning_skills.json').read_text())['skills']==[]


@pytest.mark.parametrize('mode',['skills','pure-oneshot'])
@pytest.mark.parametrize('recover',[True,False])
def test_both_modes_share_depth_reselection(tmp_path,monkeypatch,mode,recover):
    from cloth_agent.experiments import unfold_topdown as td
    from cloth_agent.dual_arm.observation import DepthUnavailableError
    monkeypatch.setattr(td,'primary',lambda before,folder:before)
    from cloth_agent.experiments import unfold_planning_skills as skills
    boundary_calls=[]
    def bounds(runtime,before,view,folder):
        boundary_calls.append(runtime)
        return 'CURRENT_BOUNDARY_SENTINEL: left and right pixel inequalities'
    monkeypatch.setattr(skills,'workspace_prompt',bounds)
    (tmp_path/'goal').mkdir();folder=tmp_path/'round';folder.mkdir()
    image=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(image)
    Image.new('RGB',(100,100)).save(tmp_path/'goal/target.png')
    def sample(p,spread):
        if p[0]==20 or not recover:raise DepthUnavailableError('insufficient fused support')
        return np.r_[p,50.]
    before=SimpleNamespace(image=image,raw_plan=lambda p:dict(p),sample=sample)
    calls=[]
    def direct(folder,body,headers,timeout):
        calls.append(body)
        assert 'CURRENT_BOUNDARY_SENTINEL' in body['instructions']
        if len(calls)>1:assert 'insufficient fused support' in body['instructions']
        return dict(decision='ACTION',skill=None,region_name='衣物',evidence='展开',
                    grasp_px=[20 if len(calls)==1 else 30,30],target_px=[50,30],
                    boundary_point_px=None,chord_endpoints_px=[],grasp_angle_deg=None,pull_distance_mm=10),dict(wall_latency_s=1.)
    plan,metrics=unfold.plan_round(tmp_path,folder,before,{},1,direct,planning_mode=mode,runtime=object())
    assert len(boundary_calls)==1
    assert len(calls)==(2 if recover else 3)
    assert plan['decision']==('ACTION' if recover else 'ABSTAIN')
    assert metrics['selection_attempts']==len(calls)
    assert metrics['wall_latency_s']==len(calls)


def test_dual_reset_uses_same_fused_image_and_depth(tmp_path,monkeypatch):
    from cloth_agent.experiments import unfold_topdown
    source=tmp_path/'source.png';fused=tmp_path/'fused.png'
    Image.new('RGB',(100,100),'red').save(source)
    Image.new('RGB',(120,120),'blue').save(fused)
    before=SimpleNamespace(image=source,meta={'dual_cloud':True},validate_for=lambda *a,**k:None,
        sample=lambda *a:pytest.fail('reset must not sample ZED alone'))
    view=SimpleNamespace(image=fused,plan_coordinates='topdown',
        sample=lambda p,s:np.array([p[0],p[1],48.77]),
        reference=lambda p:dict(camera_id=1,pixel_xy=[588,327]))
    monkeypatch.setattr(unfold_topdown,'primary',lambda obs,folder:view)
    calls=[]
    def direct(folder,body,*args):
        from cloth_agent.experiments.one_shot_goal_planner_probe_v1 import image_data
        assert body['input'][0]['content'][1]['image_url']==image_data(fused)
        assert 'center_px和radius_px均使用该图坐标' in body['instructions']
        calls.append('model')
        return dict(center_px=[60,60],radius_px=20,confidence=.9,reason='visible body'),{}
    def prepare(runtime,obs,point,release,folder):
        assert obs is before
        assert point['world_xyz_mm'][2]==48.77
        assert point['source_reference']['camera_id']==1
        assert point['coordinate_frame']=='topdown'
        calls.append('prepare');return 'source'
    monkeypatch.setattr(unfold,'prepare_reset',prepare)
    runtime=SimpleNamespace(config=None,execute=lambda *a,**k:dict(status='PLANNING_FAILED',physical_execution=False))
    unfold.initialize(runtime,before,tmp_path/'reset',headers={},timeout=1,direct=direct,seed=123,radius_mm=20)
    assert calls==['model','prepare']


def test_dual_reset_never_falls_back_to_zed_when_fusion_fails(tmp_path,monkeypatch):
    from cloth_agent.experiments import unfold_topdown
    before=SimpleNamespace(meta={'dual_cloud':True},validate_for=lambda *a,**k:None)
    def fail(*a):raise DualArmError('fusion unavailable')
    monkeypatch.setattr(unfold_topdown,'primary',fail)
    runtime=SimpleNamespace(config=None,execute=lambda *a:pytest.fail('no execution'))
    with pytest.raises(DualArmError,match='fusion unavailable'):
        unfold.initialize(runtime,before,tmp_path/'reset',headers={},timeout=1,
                          direct=lambda *a:pytest.fail('no model'),seed=123,radius_mm=20)

@pytest.mark.parametrize('mode',['recover','exhausted','moved','unmarked','workspace','collision'])
def test_all_undispatched_errors_replan(tmp_path,monkeypatch,mode):
    image=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(image)
    before=SimpleNamespace(directory=tmp_path,image=image);calls=[];requests=[]
    failure=dict(kind='IK_REJECTED',stage='complete_pre_execution_path_planning',physical_execution=False)
    def execute(*args):
        calls.append(args)
        if mode=='recover' and len(calls)>1:return dict(status='EXECUTED',physical_execution=True)
        outcome=dict(status='PLANNING_FAILED',physical_execution=mode=='moved',error={'workspace':'TCP projection 568.061 exceeds halfspace 0 limit 563.195','collision':'self-collision clearance'}.get(mode,'test IK failure'))
        if mode not in ('unmarked','workspace','collision'):outcome['planning_failure']=failure
        return outcome
    runtime=SimpleNamespace(observe=lambda p:before,execute=execute,close=lambda:None)
    monkeypatch.setattr(unfold,'prepare_action',lambda *args:'source')
    def request(folder,obs,context):
        requests.append(folder)
        if len(requests)>1:
            assert json.loads((folder/'rejection_feedback.json').read_text())['rejection_reason']=='ACTION_FAILED'
        return action(),{}
    result=unfold.episode(runtime,tmp_path,max_actions=1,request=request)
    assert len(calls)=={'recover':2,'exhausted':3,'moved':1,'unmarked':3,'workspace':3,'collision':3}[mode]
    assert result['physical_actions']==(1 if mode in ('recover','moved') else 0)
    assert result['status']=={'recover':'ACTION_LIMIT','exhausted':'PLANNING_REJECTED','moved':'EXECUTION_FAILED','unmarked':'PLANNING_REJECTED','workspace':'PLANNING_REJECTED','collision':'PLANNING_REJECTED'}[mode]


def test_previous_action_is_sent_with_matching_history_image(tmp_path):
    image=tmp_path/'previous.png';Image.new('RGB',(12,12)).save(image)
    record=dict(source_round='round_002',source_attempt='round_002/replan_01',
                skill='garment-chord-grasp',grasp_px=[3,4],target_px=[8,9],
                planned_xy_distance_mm=95.2,execution_status='EXECUTED',
                physical_execution=True,grasp_success='NOT_ASSESSED')
    body={'instructions':'test','input':[{'role':'user','content':[]}]}
    unfold.add_previous_image(body,tmp_path,{'before_image':image,'action_record':record})
    texts=[v['text'] for v in body['input'][0]['content'] if v['type']=='input_text']
    assert json.loads(texts[-1].split('\n',1)[1])==record
    assert json.loads((tmp_path/'planning_context.json').read_text())['previous_action']==record
    assert '不证明抓住布料或展开成功' in body['instructions']


@pytest.mark.parametrize('mode',['skills','pure-oneshot','SIMPLE','TANGENT'])
def test_planner_receives_topdown_and_goal_without_raw_camera(tmp_path,monkeypatch,mode):
    from cloth_agent.experiments import unfold_topdown as td, openai_cli_planner as cli
    (tmp_path/'goal').mkdir();folder=tmp_path/'round';folder.mkdir()
    zed=tmp_path/'zed.png';topdown=tmp_path/'topdown.png';goal=tmp_path/'goal/target.png'
    for path,color,size in [(zed,'red',(128,72)),(topdown,'blue',(100,100)),(goal,'green',(100,100))]:
        Image.new('RGB',size,color).save(path)
    before=SimpleNamespace(image=zed)
    view=SimpleNamespace(image=topdown,raw_plan=lambda p:dict(p),plan_coordinates='topdown')
    monkeypatch.setattr(td,'primary',lambda *a:view)
    history=tmp_path/'previous_topdown.png';Image.new('RGB',(100,100),'yellow').save(history)
    previous_context={'planning_before_image':history,'source_round':'round_000'} if mode in ('skills','pure-oneshot') else None
    calls=[]
    def direct(destination,body,*args):
        calls.append(body)
        destination.mkdir(parents=True,exist_ok=True)
        prompt,images,_=cli.prepare(destination,body)
        expected=[topdown,goal]+([history] if previous_context else [])
        assert [p.read_bytes() for p in images]==[p.read_bytes() for p in expected]
        if previous_context:
            assert '第三张图是上一轮动作前的俯视图' in prompt
            assert '第四张' not in prompt
        assert '当前ZED原始RGB' not in prompt
        assert '【当前双视图】' not in prompt
        assert zed.read_bytes() == before.image.read_bytes()
        return dict(decision='DONE',skill=None,region_name='',evidence='已展开',
                    grasp_px=None,target_px=None,boundary_point_px=None,
                    chord_endpoints_px=[],grasp_angle_deg=None,pull_distance_mm=10),{'wall_latency_s':0}
    unfold.plan_round(tmp_path,folder,before,{},1,direct,previous_context=previous_context,
        planning_mode=mode if mode in ('skills','pure-oneshot') else 'skills',
        forced_skill=mode if mode in ('SIMPLE','TANGENT') else None)
    assert len(calls)==1
    context=json.loads((folder/'planning_context.json').read_text())
    assert context['input_images']==['current','goal']+(['previous_before'] if previous_context else [])
    assert 'current_zed' not in context


def test_dual_wrist_reset_uses_zed_then_returns_to_wrist_observation(tmp_path):
    calls=[]
    initial=SimpleNamespace(directory=tmp_path/'initial',image=tmp_path/'zed.png')
    after=SimpleNamespace(directory=tmp_path/'after',image=tmp_path/'wrist.png')
    def observe(path,**kwargs):
        calls.append(('observe',kwargs))
        return initial if kwargs.get('reset_zed') else after
    def reset(before,folder):
        assert before is initial
        calls.append(('reset',{}))
        return dict(status='EXECUTED',physical_execution=True)
    def request(folder,before,context):
        assert before is after
        return {'decision':'DONE','short_reason':'done'},{}
    runtime=SimpleNamespace(dual_wrist_config={'configured':True},observe=observe,close=lambda:None)
    result=unfold.episode(runtime,tmp_path,max_actions=1,request=request,reset_request=reset)
    assert result['status']=='MODEL_REPORTED_DONE'
    assert calls==[('observe',{'reset_zed':True}),('reset',{}),('observe',{})]


@pytest.mark.parametrize('stage',['request','prepare'])
@pytest.mark.parametrize('recover',[True,False])
def test_pre_dispatch_errors_feedback_and_bounded_retry(tmp_path,monkeypatch,stage,recover):
    image=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(image)
    before=SimpleNamespace(directory=tmp_path,image=image)
    requests=[];prepared=[];executed=[]
    def request(folder,*args):
        if args[-1] and args[-1].get('final_completion_only'):
            return dict(decision='ABSTAIN',short_reason='not complete'),{}
        requests.append(folder)
        if len(requests)>1:
            feedback=json.loads((folder/'rejection_feedback.json').read_text())
            assert 'test failure' in feedback['error']
            assert feedback['physical_execution'] is False
        if stage=='request' and (not recover or len(requests)==1):raise ValueError('test failure')
        return action(),{}
    def prepare(*args):
        prepared.append(args)
        if stage=='prepare' and (not recover or len(prepared)==1):raise DualArmError('test failure')
        return 'source'
    def execute(*args):
        executed.append(args)
        return dict(status='EXECUTED',physical_execution=True)
    monkeypatch.setattr(unfold,'prepare_action',prepare)
    runtime=SimpleNamespace(observe=lambda p:before,execute=execute,close=lambda:None)
    result=unfold.episode(runtime,tmp_path,max_actions=1,request=request)
    assert len(requests)==(2 if recover else 3)
    assert len(executed)==(1 if recover else 0)
    expected='ACTION_LIMIT' if recover else ('PLANNER_UNAVAILABLE' if stage=='request' else 'PLANNING_REJECTED')
    assert result['status']==expected


@pytest.mark.parametrize('failure',[False,True])
def test_double_reset_only_recaptures_without_unfold(tmp_path,failure):
    events=[]
    def observe(path,**kwargs):
        events.append(('observe',path.name,kwargs))
        return SimpleNamespace(directory=path,image=path/'rgb.png')
    runtime=SimpleNamespace(observe=observe,dual_wrist_config={'enabled':True},close=lambda:None)
    def reset_request(before,folder):
        events.append(('reset',before.directory.name,folder.name))
        return dict(status='PLANNING_FAILED' if failure else 'EXECUTED',physical_execution=not failure)
    result=unfold.episode(runtime,tmp_path,max_actions=20,request=lambda *a:pytest.fail('no unfold'),
        reset_request=reset_request,reset_repeats=2,reset_only=True)
    assert result['status']==('RESET_FAILED' if failure else 'RESET_ONLY_COMPLETED')
    assert len([e for e in events if e[0]=='reset'])==(1 if failure else 2)
    if not failure:
        assert events[3]==('reset','after_reset_01','initialization_02')
        assert result['physical_actions']==2
        assert result['final_image'].endswith('after_reset_02/rgb.png')


@pytest.mark.parametrize('case', ['success','first_failed','abstain','done','repeat_space'])
def test_compound_skill_round(tmp_path,monkeypatch,case):
    observations=[];calls=[];executions=[]
    def observe(path):
        path.mkdir(parents=True,exist_ok=True)
        image=path/'rgb.png';Image.new('RGB',(100,100),(len(observations)*40,0,0)).save(image)
        obs=SimpleNamespace(directory=path,image=image);observations.append(obs);return obs
    def execute(*args, **kwargs):
        assert kwargs == {}  # Space preparation uses the normal execution profile.
        executions.append(args)
        return dict(status='EXECUTION_FAILED' if case=='first_failed' else 'EXECUTED',physical_execution=True)
    runtime=SimpleNamespace(observe=observe,execute=execute,close=lambda:None)
    monkeypatch.setattr(unfold,'prepare_action',lambda *a:'source')
    def request(folder,before,context):
        if context and context.get('final_completion_only'):
            return dict(decision='ABSTAIN',short_reason='not complete'),{}
        calls.append(folder)
        if len(calls)>1:
            assert before is observations[1]
            assert context['skill_followup'] is True
            assert context['action_record']['execution_status']=='EXECUTED'
            assert folder==tmp_path/'round_000/skill_followup'
        skill='make-space-for-skill' if len(calls)==1 or case=='repeat_space' else 'garment-tangent-grasp'
        (folder/'planning_skills.json').write_text(json.dumps({'selected_skill':skill}))
        if len(calls)>1 and case in ('abstain','done'):return dict(decision=case.upper()),{}
        return action(),{}
    result=unfold.episode(runtime,tmp_path,max_actions=1,request=request,max_plan_attempts=1)
    assert len(calls)==(1 if case=='first_failed' else 2)
    assert len(result['steps'])==1
    assert len(executions)==result['physical_actions']==(2 if case=='success' else 1)
    assert len(result['steps'][0]['sub_actions'])==len(executions)
    assert result['status']==dict(success='ACTION_LIMIT',first_failed='EXECUTION_FAILED',abstain='ABSTAIN',done='MODEL_REPORTED_DONE',repeat_space='PLANNING_REJECTED')[case]


def test_continue_saved_run_recaptures_without_reset(tmp_path,monkeypatch):
    parent=tmp_path/'old';parent.mkdir();output=tmp_path/'continued';output.mkdir()
    protocol=dict(max_actions=20)
    (parent/'protocol.json').write_text(json.dumps(protocol))
    original=json.dumps(dict(status='EXECUTION_FAILED',physical_actions=9,error='outside halfspace',
        steps=[dict(round=8,execution=dict(status='PLANNING_FAILED',physical_execution=False))]))
    (parent/'episode_summary.json').write_text(original)
    state=unfold.continuation_state(parent)
    assert state['start_round']==8
    observations=[]
    def observe(path):
        observations.append(path);path.mkdir();image=path/'rgb.png';Image.new('RGB',(100,100)).save(image)
        return SimpleNamespace(directory=path,image=image)
    def request(folder,before,context):
        assert folder.name=='round_008'
        assert before.directory==output/'initial_observation'
        assert json.loads((folder/'rejection_feedback.json').read_text())['error']=='outside halfspace'
        return dict(decision='DONE',short_reason='complete'),{}
    runtime=SimpleNamespace(observe=observe,close=lambda:None)
    result=unfold.episode(runtime,output,max_actions=20,request=request,continuation=state)
    assert result['status']=='MODEL_REPORTED_DONE'
    assert result['prior_physical_actions']==9 and result['physical_actions']==0
    assert (parent/'episode_summary.json').read_text()==original
    assert len(observations)==1


@pytest.mark.parametrize('skill,distance,smoothing', [
    ('make-space-for-skill',250.,False),
    (None,184.,False),
    ('garment-tangent-grasp',300.,False),
    (None,5.,False),
    ('make-space-for-skill',80.,True),
])
def test_translation_reaches_workspace_compiler_without_fixed_distance_cap(tmp_path,skill,distance,smoothing):
    picture=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(picture)
    class ReachedCompile(Exception):pass
    def compile(*args):raise ReachedCompile()
    arms={key:SimpleNamespace(world_from_base=np.eye(4),home_joints=[0]*7) for key in ('left','right')}
    models={key:SimpleNamespace(forward=lambda joints,x=x:[x,0,100,0,0,0]) for key,x in [('left',-100),('right',100)]}
    runtime=SimpleNamespace(config=SimpleNamespace(arms=arms),models=models,compile=compile)
    before=SimpleNamespace(image=picture,meta={'joints_deg':{'left':[0]*7,'right':[0]*7}},
        sample=lambda pixel,spread:np.array([0 if pixel[0]==40 else distance,0,50.]),validate_for=lambda *a,**kw:None)
    plan=action();plan.update(grasp_angle_deg=None,is_smoothing=smoothing)
    (tmp_path/'planning_skills.json').write_text(json.dumps({'selected_skill':skill}))
    with pytest.raises(ReachedCompile):unfold.prepare_action(runtime,before,plan,tmp_path)


@pytest.mark.parametrize('verdict', ['DONE','ABSTAIN','ACTION','ERROR'])
def test_action_limit_checks_final_observation_without_more_motion(tmp_path,monkeypatch,verdict):
    observed=[];executed=[];checks=[]
    def observe(path):
        obs=SimpleNamespace(directory=path,image=path/'rgb.png');observed.append(obs);return obs
    runtime=SimpleNamespace(observe=observe,close=lambda:None,
        execute=lambda *a:executed.append(a) or dict(status='EXECUTED',physical_execution=True))
    monkeypatch.setattr(unfold,'prepare_action',lambda *a:'source')
    def request(folder,before,context):
        if context and context.get('final_completion_only'):
            checks.append(before)
            if verdict=='ERROR':raise RuntimeError('service unavailable')
            return dict(decision=verdict,short_reason='final assessment'),{}
        return action(),{}
    result=unfold.episode(runtime,tmp_path,max_actions=1,request=request)
    assert len(executed)==1 and len(observed)==2 and checks==[observed[-1]]
    assert result['physical_actions']==1
    assert result['status']==('MODEL_REPORTED_DONE' if verdict=='DONE' else 'ACTION_LIMIT')
    assert result['final_completion_check']['status']==('ERROR' if verdict=='ERROR' else 'COMPLETED')


@pytest.mark.parametrize('mode',['skills','pure-oneshot'])
def test_final_completion_uses_one_visual_request_no_geometry(tmp_path,monkeypatch,mode):
    from cloth_agent.experiments import unfold_topdown as td
    monkeypatch.setattr(td,'primary',lambda before,folder:before)
    (tmp_path/'goal').mkdir();folder=tmp_path/'round_020';folder.mkdir()
    image=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(image)
    Image.new('RGB',(100,100)).save(tmp_path/'goal/target.png')
    before=SimpleNamespace(image=image,sample=lambda *a:pytest.fail('completion must not sample depth'))
    calls=[]
    def direct(folder,body,headers,timeout):
        calls.append(body)
        assert body['text']['format']['schema']['properties']['decision']['enum']==['DONE','ABSTAIN']
        assert sum(c['type']=='input_image' for m in body['input'] for c in m['content'])==2
        return dict(decision='DONE',short_reason='complete'),{}
    value,_=unfold.plan_round(tmp_path,folder,before,{},1,direct,
        previous_context={'final_completion_only':True},planning_mode=mode)
    assert value['decision']=='DONE' and len(calls)==1


def test_continuation_preserves_self_collision_evidence_after_abstain(tmp_path):
    (tmp_path/'protocol.json').write_text(json.dumps(dict(max_actions=20)))
    (tmp_path/'episode_summary.json').write_text(json.dumps(dict(status='ABSTAIN',reason='cannot choose another action',
        physical_actions=16,steps=[dict(round=16,decision='ABSTAIN',execution=dict(status='PLANNING_FAILED',
            physical_execution=False,error='PyRoki IK rejected target (self-collision: link1 vs link5)'))])))
    state=unfold.continuation_state(tmp_path)
    assert state['start_round']==16
    assert 'self-collision' in state['feedback']['last_execution_error']
    assert state['feedback']['last_execution_physical_execution'] is False
