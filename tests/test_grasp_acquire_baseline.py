"""Offline tests: no camera/model/controller connections."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cloth_agent.experiments import grasp_acquire_baseline as b
from cloth_agent.dual_arm.observation import Observation
from cloth_agent.codex_motion import parse_code
from tests.test_dual_arm_runtime import scene
from tests.test_codex_native_runtime import runtime_for


def observation(tmp_path, invalid=False):
    xyz=np.zeros((9,9,3),dtype=np.float32);xyz[:,:,2]=650
    if invalid:xyz[2:7,2:7]=np.nan
    np.save(tmp_path/'uncorrected_world_xyz_mm.npy',xyz)
    transform=np.eye(4);transform[2,2]=-1;transform[2,3]=1200
    affine=np.eye(4);affine[2,3]=-600
    corrected=xyz.copy();corrected[:,:,2]-=600
    meta=dict(world_from_camera_mm=transform.tolist(),camera_serial='testB',
        support_height_patch=dict(plane_z_ax_by_c=[0,0,640],matrix_mm=affine.tolist(),
            contact_reference=dict(semantics='empirical_open_gripper_contact_tcp_height')))
    return Observation(tmp_path,meta,corrected)


def test_depth_stats_and_existing_sample_no_invented_fusion(tmp_path):
    obs=observation(tmp_path)
    result=b.measure_pa(obs,[4,4],8)
    assert result['B']['raw_world_z_mm']==650
    assert result['B']['raw_camera_depth_z_mm']==550
    assert result['B']['depth_patch']['median']==550
    assert result['current_surface_z_mm']==50
    assert result['table_z_mm']==40
    assert result['table']['physical_table_z_mm'] is None
    assert result['fused_surface_z_mm'] is None
    assert result['surface_fusion_mode']=='B_ONLY_NO_AB_FUSION'
    assert result['current_surface_z_patch']['valid_ratio']==1


def test_invalid_patch_is_logged_without_valid_target(tmp_path):
    result=b.measure_pa(observation(tmp_path,True),[4,4],8)
    assert result['surface_xyz_mm'] is None and result['sample_error']
    assert result['B']['depth_patch']['valid_ratio']==0
    assert result['B']['depth_patch']['median'] is None
    b.save(tmp_path/'measurement.json',result)  # strictly JSON, no NaN


def test_statistics_exclude_invalid_and_retain_clipped_denominator():
    a=np.arange(25,dtype=float).reshape(5,5);a[0,0]=np.nan
    result=b.patch_stats(a,[2,2])
    assert result['valid_ratio']==24/25
    assert result['min']==1 and result['max']==24 and result['median']==12.5


def test_a_projection_does_not_reuse_b_pixel():
    transform=np.eye(4);transform[0,3]=100
    k=np.array([[100,0,50],[0,100,50],[0,0,1]])
    assert b.project_point([200,0,1000],transform,k,(100,100))==[60,50]
    assert b.project_point([0,0,-1000],transform,k,(100,100)) is None


def test_numbering_preserves_previous_trials(tmp_path):
    a=b.allocate_trial(tmp_path);(a/'marker').write_text('keep')
    c=b.allocate_trial(tmp_path)
    assert a.name=='trial_000001' and c.name=='trial_000002'
    assert (a/'marker').read_text()=='keep'


def test_low_speed_caps_never_raise_limits():
    limits=dict(joint_speed_deg_s=1.,joint_accel_deg_s2=20.,cartesian_speed_mm_s=15.,cartesian_accel_mm_s2=30.)
    runtime=SimpleNamespace(config=SimpleNamespace(limits=limits))
    b.low_speed(runtime)
    assert limits==dict(joint_speed_deg_s=1.,joint_accel_deg_s2=4.,cartesian_speed_mm_s=5.,cartesian_accel_mm_s2=10.)


def test_prepare_source_applies_existing_offset_once_and_lifts_exactly25(tmp_path):
    from cloth_agent.codex_grasp_task import lower_task_grasps
    calls=[]
    def adjust(source,folder):
        calls.append(source);return lower_task_grasps(source,folder,offset_mm=3)
    runtime=SimpleNamespace(config=SimpleNamespace(raw={},arms={'right':SimpleNamespace(world_from_base=np.eye(4))}),
        models={'right':SimpleNamespace(forward=lambda q:[0,0,100,180,0,0])},adjust_motion=adjust)
    source,grasp,lift=b.prepare_source(runtime,SimpleNamespace(meta={'joints_deg':{'right':[0]}}),[300,100,60],0,tmp_path)
    assert len(calls)==1
    assert grasp[2]==pytest.approx(.057)
    assert np.array(lift[:3])-grasp[:3]==pytest.approx([0,0,.025])
    assert [k for k,_ in parse_code(source,allow_grasp=True)]==['grasp','move_tcp']


@pytest.mark.parametrize('staged',[False,True])
def test_native_phase_plan_closes_once_and_preplans_putback(scene,monkeypatch,staged):
    from cloth_agent.codex_arm7_runtime import Arm7Runtime
    runtime=runtime_for(scene);runtime.__class__=Arm7Runtime
    runtime.staged_release=staged
    original_speed=runtime.config.limits['joint_speed_deg_s']
    b.low_speed(runtime)
    monkeypatch.setattr(runtime.paths,'joint_path',lambda a,c,**kw:[a,c])
    initial={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    before=SimpleNamespace(meta={'joints_deg':initial})
    source='motion.grasp(right=[.25,.12,.18,0,0,0])\nmotion.move_tcp(right=[.25,.12,.205,0,0,0])\n'
    try:
        phases=b.plan_trial(runtime,before,source,[.25,.12,.18,0,0,0])
        assert [n for n,_ in phases]==['pregrasp_descent','close','lift','replace','release_retreat','return_home']
        grippers=[d['action'] for _,ops in phases for k,d in ops if k=='gripper']
        assert grippers==(['open','close','release_partial','open'] if staged else ['open','close','open'])
        assert b.end_joints(phases[2][1])['right'][2]==pytest.approx(205)
        assert b.end_joints(phases[3][1])['right'][2]==pytest.approx(180)
        if staged:
            release=phases[4][1]
            assert release[0][0]=='gripper' and release[0][1]['action']=='release_partial'
            assert release[0][1]['joints']['right'][2]==pytest.approx(180)
            assert release[0][1]['opening_delta_pulse']==120.
            assert release[-1][0]=='gripper' and release[-1][1]['action']=='open'
            assert release[-1][1]['joints']['right'][:3]==pytest.approx([250,120,240])
            assert all(kind=='move' for kind,_ in release[1:-1])
        assert all(d.speed<=2 for _,ops in phases[:-1] for k,d in ops if k=='move')
        assert all(d.speed<=original_speed for k,d in phases[-1][1] if k=='move')
        assert runtime.config.limits['joint_speed_deg_s']==min(2,original_speed)
        assert all(not c.commands for c in scene.connections.values())
    finally:runtime.close()


def test_staged_putback_rejects_nonvertical_clearance_before_dispatch(scene,monkeypatch):
    from cloth_agent import codex_grasp_task as task
    from cloth_agent.dual_arm.geometry import DualArmError
    runtime=runtime_for(scene)
    placed={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    runtime.planning_context={}
    monkeypatch.setattr(task,'_clearance_targets',lambda *a:[(10.,0.,[]),(20.,20.,[])])
    try:
        with pytest.raises(DualArmError,match='vertical clearance'):
            b.retreat_and_home(runtime,placed,complete_release=True)
        assert all(not c.commands for c in scene.connections.values())
    finally:runtime.close()


def test_return_planning_failure_never_dispatches(scene,monkeypatch):
    runtime=runtime_for(scene)
    monkeypatch.setattr(runtime.paths,'joint_path',lambda a,c,**kw:[a,c])
    initial={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    original=runtime.plan
    def fail(source,*a,**kw):
        if source.startswith('motion.move_tcp'):raise ValueError('putback cannot be certified')
        return original(source,*a,**kw)
    monkeypatch.setattr(runtime,'plan',fail)
    try:
        with pytest.raises(ValueError,match='putback'):
            b.plan_trial(runtime,SimpleNamespace(meta={'joints_deg':initial}),
                'motion.grasp(right=[.25,.12,.18,0,0,0])\nmotion.move_tcp(right=[.25,.12,.205,0,0,0])\n',
                [.25,.12,.18,0,0,0])
        assert all(not c.commands for c in scene.connections.values())
    finally:runtime.close()


@pytest.mark.parametrize('photo_fails',[False,True])
def test_hold_photo_between_lift_and_place_no_retry(tmp_path,photo_fails):
    events=[]
    def execute(ops,folder):
        events.append(folder.name);return {'status':'EXECUTED'}
    def photo(runtime,folder,hold):
        events.append('PHOTO')
        if photo_fails:raise RuntimeError('camera timeout')
        folder.mkdir();Image.new('RGB',(8,8)).save(folder/'rgb.png')
    runtime=SimpleNamespace(_execute_operations=execute,coordinator=SimpleNamespace(snapshots=lambda:{}))
    report={'phases':[]};names=['pregrasp_descent','close','lift','replace','release_retreat','return_home']
    error=b.dispatch_trial(runtime,[(n,[]) for n in names],tmp_path,report,2,photo)
    assert events==['pregrasp_descent','close','lift','PHOTO','replace','release_retreat','return_home']
    assert bool(error)==photo_fails and report['returned_safely']


def test_motion_failure_does_not_blindly_replace_or_retry(tmp_path):
    events=[]
    def execute(ops,folder):
        events.append(folder.name)
        return {'status':'EXECUTION_FAILED'}
    runtime=SimpleNamespace(_execute_operations=execute)
    with pytest.raises(RuntimeError):
        b.dispatch_trial(runtime,[('close',[]),('lift',[]),('replace',[])],tmp_path,{'phases':[]},2)
    assert events==['close']


def test_home_limits_applied_only_for_home_dispatch_and_restored_on_failure(tmp_path):
    runtime=SimpleNamespace(config=SimpleNamespace(limits=dict(joint_speed_deg_s=5.,joint_accel_deg_s2=10.,
        cartesian_speed_mm_s=15.,cartesian_accel_mm_s2=30.)),coordinator=SimpleNamespace(snapshots=lambda:{}))
    b.low_speed(runtime);speeds=[]
    def execute(ops,folder):
        speeds.append(runtime.config.limits['joint_speed_deg_s'])
        return {'status':'EXECUTION_FAILED' if folder.name=='return_home' else 'EXECUTED'}
    runtime._execute_operations=execute
    report={'phases':[]}
    with pytest.raises(RuntimeError):
        b.dispatch_trial(runtime,[('release_retreat',[]),('return_home',[])],tmp_path,report,0)
    assert speeds==[2.,5.]
    assert runtime.config.limits['joint_speed_deg_s']==2.
    assert not report.get('returned_safely')


def test_hold_photo_has_no_observe_or_motion(tmp_path):
    class Runtime:
        def observe(self,*a):raise AssertionError('must not open gripper/home')
    runtime=Runtime()
    runtime.camera=SimpleNamespace(read=lambda:(np.zeros((8,8,3),np.uint8),np.ones((8,8))),intrinsics=np.eye(3))
    runtime.coordinator=SimpleNamespace(snapshots=lambda:{'right':{'joints':[0,0],'gripper':{'position_pulse':7}}})
    runtime.config=SimpleNamespace(limits={'stationary_tolerance_deg':.2})
    b.hold_photo(runtime,tmp_path/'hold',0)
    assert (tmp_path/'hold/rgb.png').exists()


def test_label_is_human_only_and_cannot_be_overwritten(tmp_path):
    trial=b.allocate_trial(tmp_path)
    b.save(trial/'trial.json',dict(operator_label=None,returned_safely=True))
    b.set_label(trial,'MISS')
    assert json.loads((trial/'trial.json').read_text())['operator_label']=='MISS'
    with pytest.raises(ValueError):b.set_label(trial,'HELD')
    assert (tmp_path/'labels.csv').read_text().count('MISS')==1


def test_no_execute_does_not_initialize_hardware(monkeypatch):
    monkeypatch.setattr(b,'run',lambda *a:pytest.fail('hardware branch'))
    assert b.main(['run'])==0
