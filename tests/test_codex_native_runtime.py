from types import SimpleNamespace
import numpy as np
import pytest
from cloth_agent.codex_native_runtime import NativeRuntime
from cloth_agent.dual_arm.motion.home import HomeValidator
from cloth_agent.dual_arm.motion.path import PathPlanner
from cloth_agent.dual_arm.geometry import DualArmError
from .test_dual_arm_runtime import scene, coordinator


def runtime_for(scene):
    scene.config.execution_mode='controller_sequential'
    r=object.__new__(NativeRuntime)
    r.config=scene.config;r.models=scene.models;r.cancel=scene.cancel
    for k,m in r.models.items():
        m.lower=np.full(scene.config.arms[k].axis,-20.)
        m.upper=np.full(scene.config.arms[k].axis,20.)
    r.paths=PathPlanner(SimpleNamespace(models=r.models),r,HomeValidator(r.config,r.models,None))
    r.coordinator=coordinator(scene);r.prepared=False;r._lease=None
    return r


def test_ik_failure_preserves_step_target_candidate_and_effective_code(scene,tmp_path,monkeypatch):
    r=runtime_for(scene)
    def reject(*args):
        error=DualArmError('PyRoki IK rejected target (2.1 mm, 0.8 deg)')
        error.ik_candidate=dict(joints_deg=scene.config.arms['left'].home_joints.tolist(),distance_mm=2.1,angle_deg=.8)
        error.ik_diagnostics=dict(seed_count=2,max_iterations_per_seed=40,attempts=[{'reason':'POSE_TOLERANCE_NOT_MET'}])
        raise error
    monkeypatch.setattr(scene.models['left'],'inverse',reject)
    def plan(source,initial):
        r.planning_context=dict(command_index=3,command='move_tcp',arm='left',route='tcp_rrtconnect')
        r._inverse_target('left',[700,-85,146,180,0,-77],initial['left'],'rrt_goal_ik')
    monkeypatch.setattr(r,'plan',plan)
    obs=SimpleNamespace(meta={'joints_deg':{k:a.home_joints.copy() for k,a in scene.config.arms.items()}},validate_for=lambda *a,**kw:None)
    source='motion.move_tcp(left=[.7,-.085,.146,3.14,0,-1.3])'
    try:
        outcome=r.execute(source,obs,tmp_path)
        assert outcome['failure_kind']=='IK_FAILED'
        assert outcome['ik_failure']['command_index']==3
        assert outcome['ik_failure']['target_base_mm_deg']==[700,-85,146,180,0,-77]
        assert outcome['ik_failure']['candidate']['distance_mm']==2.1
        assert outcome['ik_failure']['diagnostics']['max_iterations_per_seed']==40
        assert len(outcome['ik_failure']['target_world_m_rad'])==6
        assert outcome['rejected_motion_source']==source
        assert outcome['physical_execution'] is False
        assert all(not c.commands for c in scene.connections.values())
    finally:r.close()


def test_shared_grasp_height_applied_before_planning(scene,tmp_path,monkeypatch):
    from cloth_agent.codex_motion import parse_code
    import json
    r=runtime_for(scene)
    r.config.raw['astra_grasp_height']={'mode':'shared_arm7_target_z'}
    obs=SimpleNamespace(meta={'joints_deg':{k:a.home_joints.copy() for k,a in scene.config.arms.items()}},
                        validate_for=lambda *a,**kw:None)
    seen=[]
    def plan(source,initial):
        seen.append(parse_code(source))
        raise ValueError('test stops before dispatch')
    monkeypatch.setattr(r,'plan',plan)
    source=('motion.move_tcp(left=[.4,.2,.03,0,0,0],right=[.7,.2,.035,0,0,0])\n'
            'motion.gripper(arm="left",action="close")\n'
            'motion.gripper(arm="right",action="close")')
    try:
        result=r.execute(source,obs,tmp_path)
        assert result['status']=='PLANNING_FAILED'
        assert seen[0][0][1]['left'][2]==.035
        assert seen[0]==parse_code((tmp_path/'effective_motion.py').read_text())
        assert json.loads((tmp_path/'grasp_height_policy.json').read_text())['changes'][0]['original_left_z_mm']==30
        assert all(not c.commands for c in scene.connections.values())
        result=r.execute('motion.gripper(arm="left",action="close")',obs,tmp_path)
        assert result['status']=='PLANNING_FAILED'
        assert len(seen)==1
    finally:r.close()


def test_native_code_plans_all_before_dispatch_and_preserves_gripper_order(scene,tmp_path):
    r=runtime_for(scene)
    initial={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    target=np.radians(initial['left']);target[0]+=.001
    obs=SimpleNamespace(meta={'joints_deg':initial},validate_for=lambda *a,**kw:None)
    source=f'motion.gripper(arm="left",action="open")\nmotion.move_joints(left={target.tolist()})\nmotion.gripper(arm="left",action="close")'
    try:
        result=r.execute(source,obs,tmp_path)
        assert result['status']=='EXECUTED', result
        assert result['completed_commands']==3
        assert np.allclose(scene.connections['left'].joints,np.degrees(target))
    finally:r.close()


def test_rejected_final_target_does_not_execute_earlier_gripper(scene,tmp_path):
    r=runtime_for(scene)
    obs=SimpleNamespace(meta={'joints_deg':{k:a.home_joints.copy() for k,a in scene.config.arms.items()}},
                        validate_for=lambda *a,**kw:None)
    try:
        result=r.execute('motion.gripper(arm="left",action="close")\nmotion.move_joints(left=[99,0,0,0,0,0])',obs,tmp_path)
        assert result['status']=='PLANNING_FAILED'
        assert result['failure_context']['command_index']==2
        assert result['failure_context']['command']=='move_joints'
        assert all(not c.commands for c in scene.connections.values())
    finally:r.close()


def test_native_fault_stops_both_arms(scene,tmp_path,monkeypatch):
    r=runtime_for(scene)
    obs=SimpleNamespace(meta={'joints_deg':{k:a.home_joints.copy() for k,a in scene.config.arms.items()}},
                        validate_for=lambda *a,**kw:None)
    def fail(*args):
        error=DualArmError('injected gripper fault')
        error.gripper_completion={'samples':[{'position':500,'state':'moving'}]}
        raise error
    monkeypatch.setattr(scene.connections['left'],'gripper',fail)
    try:
        result=r.execute('motion.gripper(arm="left",action="close")',obs,tmp_path)
        assert result['status']=='EXECUTION_FAILED'
        assert all(c.stopped for c in scene.connections.values())
        assert result['failed_command']['data']['action']=='close'
        assert result['gripper_completion']['samples'][0]['position']==500
        assert (tmp_path/'gripper_001.json').exists()
    finally:r.close()


def test_unfinished_proof_is_not_a_collision_or_executed(scene,tmp_path,monkeypatch):
    from cloth_agent.dual_arm.stop_sweep import ProofIncompleteError
    r=runtime_for(scene)
    obs=SimpleNamespace(meta={'joints_deg':{k:a.home_joints.copy() for k,a in scene.config.arms.items()}},
                        validate_for=lambda *a,**kw:None)
    def unfinished(*args):raise ProofIncompleteError('proof still pending')
    monkeypatch.setattr(r,'plan',unfinished)
    try:
        result=r.execute('motion.gripper(arm="left",action="open")',obs,tmp_path)
        assert result['status']=='PLANNING_INCOMPLETE'
        assert result['physical_execution'] is False
        assert all(not c.commands for c in scene.connections.values())
    finally:r.close()


def test_joint_search_retries_exhausted_candidates(scene,monkeypatch):
    r=runtime_for(scene);attempts=[]
    initial={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    def search(a,b,**kw):
        attempts.append(kw['order'])
        if len(attempts)<2:raise DualArmError('OMPL found no exact collision-certified path in the time budget')
        return [a,b]
    monkeypatch.setattr(r.paths,'joint_path',search)
    try:
        r.plan(f'motion.move_joints(left={np.radians(initial["left"]+.01).tolist()})',initial)
        assert attempts==['left_first','left_first']
        assert all(not c.commands for c in scene.connections.values())
    finally:r.close()


def test_observe_moves_to_perception_before_camera_capture(scene,tmp_path,monkeypatch):
    from cloth_agent.dual_arm import observation
    r=runtime_for(scene)
    target=scene.config.arms['left'].home_joints.copy();target[0]+=1.
    monkeypatch.setattr(r,'perception_target',lambda:('left',target))
    calls=[]
    camera=SimpleNamespace(stop=lambda:calls.append("stop_camera"))
    monkeypatch.setattr(observation,"create_camera",lambda config:camera)
    def capture(directory,config,models,snapshot,*,camera):
        assert camera is r.camera
        calls.append('capture')
        assert np.allclose(scene.connections['left'].joints,target)
        assert np.allclose(scene.connections['right'].joints,config.arms['right'].home_joints)
        directory.mkdir()
        return SimpleNamespace(meta={'joints_deg':{k:v['joints'] for k,v in snapshot().items()}})
    monkeypatch.setattr(observation,'capture',capture)
    try:
        r.observe(tmp_path/'before')
        first_count=len(scene.connections['left'].commands)
        r.observe(tmp_path/'after')
        assert calls==['capture','capture']
        assert len(scene.connections['left'].commands)==first_count
        assert (tmp_path/'before_perception/native_plan.json').exists()
        assert 'ALREADY_AT_PERCEPTION' in (tmp_path/'after_perception/result.json').read_text()
    finally:r.close()
    assert calls==['capture','capture','stop_camera']
    r.close()
    assert calls.count('stop_camera')==1


def test_perception_failure_prevents_capture(scene,tmp_path,monkeypatch):
    from cloth_agent.dual_arm import observation
    r=runtime_for(scene)
    monkeypatch.setattr(r,'perception_target',lambda:('left',np.full(6,9999.)))
    monkeypatch.setattr(observation,'capture',lambda *a:pytest.fail('camera called before arrival'))
    try:
        with pytest.raises(DualArmError):r.observe(tmp_path/'before')
        assert all(not c.commands for c in scene.connections.values())
    finally:r.close()


@pytest.mark.parametrize('wrong_identity',[False,True])
def test_perception_record_validates_identity_and_trims_sdk_padding(scene,tmp_path,wrong_identity):
    import json
    r=runtime_for(scene);arm=scene.config.arms['left']
    path=tmp_path/'pose.json'
    path.write_text(json.dumps(dict(robot_ip=arm.ip,axis=6,
        control_box_sn='wrong' if wrong_identity else arm.serial,camera_serial='test-camera',
        joint_angles_deg=[1,2,3,4,5,6,0],tcp_offset_mm_deg=arm.tcp_offset.tolist())))
    scene.config.raw['camera'].update(mount_arm='left',serial='test-camera',perception_pose_file=str(path))
    try:
        if wrong_identity:
            with pytest.raises(DualArmError,match='identity'):r.perception_target()
        else:
            owner,q=r.perception_target()
            assert owner=='left' and q.tolist()==[1,2,3,4,5,6]
    finally:r.close()


def test_compaction_rechecks_whole_box_and_reduces_stops(scene,monkeypatch):
    from cloth_agent.dual_arm.sequential_home import PositionSegment
    r=runtime_for(scene);r.plan_stats={'checked_segments':0,'dispatched_segments':0}
    initial={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    operations=[]
    for i in range(10):
        a={k:q.copy() for k,q in initial.items()};b={k:q.copy() for k,q in initial.items()}
        a['left'][0]+=i*.1;b['left'][0]+=(i+1)*.1
        operations.append(('move',PositionSegment('left',a,b,5,10,30)))
    checked=[]
    monkeypatch.setattr(r.paths.validator,'certify_box',lambda a,b,**kw:checked.append((a,b)))
    try:
        compact=r._compact_moves(operations)
        assert len(compact)==1 and len(checked)==1
        assert np.allclose(compact[0][1].end['left'],operations[-1][1].end['left'])
        assert compact[0][1].speed==scene.config.limits['joint_speed_deg_s']
        assert r.plan_stats=={'checked_segments':10,'dispatched_segments':1}
        def reject(*a,**kw):raise DualArmError('larger box collides')
        monkeypatch.setattr(r.paths.validator,'certify_box',reject)
        assert len(r._compact_moves(operations))==10
    finally:r.close()


def test_compaction_preserves_joint_reversal(scene):
    from cloth_agent.dual_arm.sequential_home import PositionSegment
    r=runtime_for(scene);r.plan_stats={'checked_segments':0,'dispatched_segments':0}
    a={k:v.home_joints.copy() for k,v in scene.config.arms.items()}
    b={k:q.copy() for k,q in a.items()};b['left'][0]+=1
    try:
        operations=[('move',PositionSegment('left',a,b,5,10,30)),('move',PositionSegment('left',b,a,5,10,30))]
        assert len(r._compact_moves(operations))==2
    finally:r.close()


@pytest.mark.parametrize('arm7_only',[False,True])
def test_first_observation_skips_motion_when_already_at_perception(scene,tmp_path,monkeypatch,arm7_only):
    import json
    r=runtime_for(scene)
    owner='left'
    if arm7_only:
        from cloth_agent.codex_arm7_runtime import Arm7Runtime
        r.__class__=Arm7Runtime
        owner='right'
        scene.connections['left'].joints[0]+=2
    target=scene.config.arms[owner].home_joints.copy();target[0]+=1
    scene.connections[owner].joints=target.copy()
    monkeypatch.setattr(r,'perception_target',lambda:(owner,target))
    monkeypatch.setattr(r,'_plan_preparation',lambda *a:pytest.fail('already at perception must not plan motion'))
    try:
        r.move_to_perception(tmp_path/'transition')
        result=json.loads((tmp_path/'transition/result.json').read_text())
        assert result==dict(status='ALREADY_AT_PERCEPTION',physical_execution=False)
        assert all(not c.commands for c in scene.connections.values())
    finally:r.close()


def test_adjacent_source_moves_execute_without_intermediate_stop(scene,tmp_path):
    import json
    r=runtime_for(scene)
    initial={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    a=initial['left'].copy();a[0]+=.1
    b=initial['left'].copy();b[0]+=.2
    source=(f'motion.move_joints(left={np.radians(a).tolist()})\n'
            f'motion.move_joints(left={np.radians(b).tolist()})\n'
            'motion.gripper(arm="left",action="close")')
    obs=SimpleNamespace(meta={'joints_deg':initial},validate_for=lambda *a,**kw:None)
    try:
        result=r.execute(source,obs,tmp_path)
        assert result['status']=='EXECUTED',result
        positions=[q for kind,q in scene.connections['left'].commands if kind=='position']
        assert len(positions)==1
        assert np.allclose(positions[0],b)
        stats=json.loads((tmp_path/'planning_stats.json').read_text())
        assert stats['removed_stops']>=1
        assert scene.connections['left'].commands[-1]==('gripper','close')
    finally:r.close()


def test_joining_preserves_gripper_switch_and_explicit_arrival_boundaries(scene):
    from cloth_agent.dual_arm.sequential_home import PositionSegment
    r=runtime_for(scene);r.plan_stats={'checked_segments':0,'dispatched_segments':0}
    q={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    def move(arm):
        nonlocal q
        end={k:v.copy() for k,v in q.items()};end[arm][0]+=.1
        result=('move',PositionSegment(arm,q,end,5,10,30));q=end
        return result
    operations=[move('left'),move('left')]
    operations.append(('gripper',dict(arm='left',action='close',joints={k:v.tolist() for k,v in q.items()})))
    operations.extend([move('left'),move('right'),move('left')])
    try:
        joined=r._join_moves(operations,boundaries={1})
        assert len(joined)==len(operations)
        assert [kind for kind,_ in joined]==[kind for kind,_ in operations]
        assert r.plan_stats['removed_stops']==0
    finally:r.close()


@pytest.mark.parametrize('failure',['unsafe','unknown'])
def test_joining_rejected_full_box_keeps_original_commands(scene,monkeypatch,failure):
    from cloth_agent.dual_arm.sequential_home import PositionSegment
    from cloth_agent.dual_arm.stop_sweep import ProofIncompleteError
    r=runtime_for(scene);r.plan_stats={'checked_segments':0,'dispatched_segments':0}
    a={k:v.home_joints.copy() for k,v in scene.config.arms.items()}
    b={k:q.copy() for k,q in a.items()};b['left'][0]+=.1
    c={k:q.copy() for k,q in b.items()};c['left'][0]+=.1
    operations=[('move',PositionSegment('left',a,b,5,10,30)),('move',PositionSegment('left',b,c,5,10,30))]
    def reject(*args,**kw):
        raise (ProofIncompleteError if failure=='unknown' else DualArmError)('merge not certified')
    monkeypatch.setattr(r.paths.validator,'certify_box',reject)
    try:
        joined=r._join_moves(operations)
        assert len(joined)==2
        assert joined[0][1] is operations[0][1]
        assert joined[1][1] is operations[1][1]
        assert not any(c.commands for c in scene.connections.values())
    finally:r.close()


@pytest.mark.parametrize('changed',['none','calibration','synthetic','joints'])
def test_astra_old_observation_preserves_provenance_and_arrival_checks(scene,tmp_path,changed):
    import time
    r=runtime_for(scene)
    obs=scene.obs
    obs.meta['synthetic']=False
    obs.meta['captured_unix_s']=time.time()-86400
    if changed=='calibration':obs.meta['calibration_id']='different-calibration'
    if changed=='synthetic':obs.meta['synthetic']=True
    if changed=='joints':scene.connections['left'].joints[0]+=2
    try:
        result=r.execute('motion.gripper(arm="left",action="open")',obs,tmp_path)
        if changed=='none':
            assert result['status']=='EXECUTED',result
        else:
            assert result['status']=='PLANNING_FAILED',result
            assert 'expired' not in result['error']
            assert all(not c.commands for c in scene.connections.values())
    finally:r.close()


@pytest.mark.parametrize('failure',['collision','unknown','ik'])
def test_tcp_replans_from_original_start_and_certifies_detour(scene,monkeypatch,failure):
    from cloth_agent.dual_arm.sequential_home import PositionSegment
    from cloth_agent.dual_arm.stop_sweep import ProofIncompleteError
    from cloth_agent.dual_arm.motion.home import pack,unpack
    r=runtime_for(scene);r.planning_context={};r.plan_stats={}
    initial={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    target=initial['left'][:6].copy();target[0]+=15
    pose=np.r_[target[:3]/1000,np.radians(target[3:])]
    checked=[];searches=[]
    monkeypatch.setattr(r,'check',lambda *a:{'safe':True})
    inverse=r.models['left'].inverse
    ik_calls=[]
    def ik(p,seed):
        ik_calls.append(p)
        if failure=='ik' and len(ik_calls)==2:raise DualArmError('intermediate IK failed')
        return inverse(p,seed)
    monkeypatch.setattr(r.models['left'],'inverse',ik)
    def segments(a,b):
        checked.append((a,b))
        if failure!='ik' and len(checked)==2:
            raise (ProofIncompleteError if failure=='unknown' else DualArmError)('direct path rejected')
        return [PositionSegment('left',a,b,5,10,30)]
    monkeypatch.setattr(r,'_segments',segments)
    def search(a,b,**kw):
        searches.append((a,b,kw))
        via=a.copy();via[1]+=.1
        return [a,via,b]
    monkeypatch.setattr(r.paths,'joint_path',search)
    try:
        result,goal=r._tcp_segments('left',pose,initial)
        assert len(searches)==1 and searches[0][2]['order']=='left_first'
        assert np.allclose(searches[0][0],pack(initial))
        assert np.allclose(goal['left'][:6],target)
        assert np.allclose(goal['right'],initial['right'])
        assert len(result)==2 # failed direct prefix is discarded
        assert np.allclose(result[0].start['left'],initial['left'])
        assert np.allclose(result[0].end['left'],unpack(searches[0][0]+np.eye(13)[1]*.1)['left'])
        assert np.allclose(result[-1].end['left'],goal['left'])
        assert r.plan_stats['tcp_rrt_detours']==1
        assert not any(c.commands for c in scene.connections.values())
    finally:r.close()


def test_tcp_failed_detour_never_dispatches_prior_commands(scene,tmp_path,monkeypatch):
    from cloth_agent.dual_arm.stop_sweep import ProofIncompleteError
    r=runtime_for(scene)
    initial={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    obs=SimpleNamespace(meta={'joints_deg':initial},validate_for=lambda *a,**kw:None)
    monkeypatch.setattr(r,'check',lambda *a:{'safe':True})
    def reject(*a,**kw):raise ProofIncompleteError('direct proof incomplete')
    monkeypatch.setattr(r,'_segments',reject)
    searches=[]
    def no_route(*a,**kw):
        searches.append(kw)
        raise DualArmError('OMPL found no exact collision-certified path')
    monkeypatch.setattr(r.paths,'joint_path',no_route)
    try:
        result=r.execute('motion.gripper(arm="left",action="open")\nmotion.move_tcp(left=[.265,-.12,.18,0,0,0])',obs,tmp_path)
        assert result['status']=='PLANNING_FAILED'
        assert result['failure_context']['route']=='tcp_rrtconnect'
        assert 'direct proof incomplete' in result['failure_context']['direct_failure']
        assert len(searches)==1
        assert not any(c.commands for c in scene.connections.values())
    finally:r.close()


def test_tcp_valid_direct_path_does_not_search_rrt(scene,monkeypatch):
    r=runtime_for(scene);r.planning_context={};r.plan_stats={}
    initial={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    monkeypatch.setattr(r,'check',lambda *a:{'safe':True})
    monkeypatch.setattr(r.paths,'joint_path',lambda *a,**kw:pytest.fail('unnecessary RRT search'))
    try:
        segments,goal=r._tcp_segments('left',[.260,-.120,.180,0,0,0],initial)
        assert segments and np.allclose(goal['left'][:3],[260,-120,180])
        assert np.allclose(goal['right'],initial['right'])
        assert r.planning_context['route']=='tcp_direct'
    finally:r.close()


def test_execution_timings_report_real_stages(scene,tmp_path):
    r=runtime_for(scene);scene.obs.meta['synthetic']=False
    try:
        result=r.execute('motion.gripper(arm="right",action="open")',scene.obs,tmp_path)
        assert result['status']=='EXECUTED'
        for key in ('trajectory_planning_s','preview_s','preview_wait_s','pre_dispatch_check_s','execution_s'):
            assert result['timings_s'][key]>=0
    finally:r.close()


def test_joint_search_stops_after_two_timeout_rounds(scene,tmp_path,monkeypatch):
    r=runtime_for(scene);attempts=[]
    initial={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    def search(a,b,**kw):
        attempts.append(kw['order'])
        raise DualArmError('collision proof time budget exhausted')
    monkeypatch.setattr(r.paths,'joint_path',search)
    obs=SimpleNamespace(meta={'joints_deg':initial},validate_for=lambda *a,**kw:None)
    try:
        outcome=r.execute(f'motion.move_joints(right={np.radians(initial["right"]+.01).tolist()})',obs,tmp_path)
        assert outcome['status']=='PLANNING_FAILED'
        assert outcome['failure_kind']!='IK_FAILED'
        assert outcome['failure_context']['path_search_failure']['attempt_rounds']==2
        assert attempts==['right_first','right_first']
        assert all(not c.commands for c in scene.connections.values())
    finally:r.close()
