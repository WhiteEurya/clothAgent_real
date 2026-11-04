"""Offline single-gripper ownership, graph continuity and geometric regressions."""
import copy
import json
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cloth_agent.dual_arm.geometry import pose_matrix
from cloth_agent.dual_arm.observation import save_observation
from cloth_agent.site_observation import load_site
from cloth_agent.experiments.partitioned_folding import (
    PartitionError, choose_arm, load_policy, partition_frame, plan_step, main,
)


@pytest.fixture
def scene(tmp_path):
    profile = load_site()
    # Analytic tabletop in the site's common frame: pixels map to mm exactly.
    y,x = np.indices((160,180))
    xyz = np.stack((x*5., -y*5., np.full_like(x,30.)), axis=-1)
    config = NS(raw={'calibration_id':profile['calibration_id']})
    obs = save_observation(tmp_path/'observation',np.zeros((160,180,3),np.uint8),xyz,config,{},
        synthetic=True,extra=dict(common_frame='right_arm_base',camera_serial=profile['camera']['serial'],execution_allowed=False))
    graph = dict(edges=[dict(edge_id='e',source_node='n0',target_node='n1',action_family='test fold')])
    templates = {'e':dict(arm='left',axis=6)}
    request = dict(observation_id=obs.meta['observation_id'],rgb_sha256=obs.meta['sha256']['rgb.png'],
        current_node='n0',edge_id='e',centerline_px=[[100,30],[100,130]],
        grasp_px=[130,70],target_px=[80,70],confidence=1.,anchor_source='synthetic_demo')
    return profile,obs,graph,templates,request,load_policy()


def plan(scene, request=None):
    profile,obs,graph,templates,original,policy=scene
    return plan_step(graph,templates,profile,obs,request or original,policy,[0,0,1])


@pytest.mark.parametrize('grasp,arm', [([130,70],'left'),([70,70],'right')])
def test_side_owns_whole_action_and_correct_base_pose(scene,grasp,arm):
    request=copy.deepcopy(scene[4]); request.update(grasp_px=grasp,target_px=[80,100])
    result=plan(scene,request)
    assert result['selected_arm']==arm and result['axis']==7
    assert {op['arm'] for op in result['operations']}=={arm}
    assert len(result['operations'])==9
    assert result['execution_allowed'] is False
    transform=np.array(scene[0]['arms'][arm]['common_from_base_m'])
    transform[:3,3]*=1000
    for op in result['operations']:
        if op['kind']=='move_tcp':
            np.testing.assert_allclose(transform @ pose_matrix(op['pose_base_mm_deg']),
                                       pose_matrix(op['pose_common_mm_deg']),atol=1e-7)


def test_target_crossing_does_not_switch_arm(scene):
    result=plan(scene)
    assert result['crosses_centerline'] and result['selected_arm']=='left'
    assert result['target_signed_distance_mm']<0
    assert all(op['arm']=='left' for op in result['operations'])


def test_reversing_centerline_does_not_swap_arm(scene):
    request=copy.deepcopy(scene[4]);request['centerline_px'].reverse()
    assert plan(scene,request)['selected_arm']==plan(scene)['selected_arm']


@pytest.mark.parametrize('arm',[None,'left','right'])
def test_center_band_needs_explicit_arm(scene,arm):
    request=copy.deepcopy(scene[4]);request.update(grasp_px=[100,70],center_arm=arm)
    result=plan(scene,request)
    assert result['selected_arm']==arm
    assert (result['status']=='CENTER_ARM_REQUIRED')==(arm is None)
    assert bool(result['operations'])==(arm is not None)


def test_cannot_force_wrong_half(scene):
    request=copy.deepcopy(scene[4]);request['center_arm']='right'
    with pytest.raises(PartitionError,match='cannot override'):
        plan(scene,request)


@pytest.mark.parametrize('change,match',[
    ({'observation_id':'old'},'different observation'),
    ({'rgb_sha256':'old'},'different observation'),
    ({'current_node':'n1'},'not outgoing'),
    ({'edge_id':'missing'},'unsupported'),
    ({'confidence':.2},'confidence'),
    ({'confidence':float('nan')},'confidence'),
    ({'grasp_px':[130.5,70]},'integer'),
    ({'centerline_px':[[100,70],[100,71]]},'too short'),
])
def test_stale_or_invalid_inputs_rejected(scene,change,match):
    request=copy.deepcopy(scene[4]);request.update(change)
    with pytest.raises(ValueError,match=match):
        plan(scene,request)


def test_missing_depth_rejected(scene):
    scene[1].xyz[68:73,128:133]=np.nan
    with pytest.raises(ValueError,match='depth'):
        plan(scene)


def test_rotated_centerline_uses_world_geometry(scene):
    profile,_,_,_,_,policy=scene
    frame=partition_frame(profile,[400,-200,30],[600,200,30],[0,0,1],policy)
    assert choose_arm(frame,[750,0,30])[0]=='left'
    assert choose_arm(frame,[250,0,30])[0]=='right'
    # A nearly horizontal line cannot assign opposite bases to opposite cloth halves.
    with pytest.raises(PartitionError,match='does not separate'):
        partition_frame(profile,[200,200,30],[800,200,30],[0,0,1],policy)


def test_no_old_height_or_pose_reuse(scene):
    result=plan(scene)
    moves={op['phase']:op for op in result['operations'] if op['kind']=='move_tcp'}
    assert moves['grasp']['pose_common_mm_deg'][2]==30.
    assert moves['lifted']['pose_common_mm_deg'][2]==110.
    assert moves['placed']['pose_common_mm_deg'][2]==35.
    assert moves['grasp']['pose_base_mm_deg'][3:]==pytest.approx(scene[0]['arms']['left']['grasp_rpy_deg'])


def test_two_separate_steps_can_choose_different_arms(scene):
    first=plan(scene)
    request=copy.deepcopy(scene[4]);request.update(grasp_px=[70,70],target_px=[110,70])
    second=plan(scene,request)
    assert [first['selected_arm'],second['selected_arm']]==['left','right']
    assert all(len({op['arm'] for op in step['operations']})==1 for step in (first,second))


def test_real_graph_offline_without_robot_or_camera_sdk(monkeypatch,capsys):
    monkeypatch.setitem(sys.modules,'xarm',None)
    monkeypatch.setitem(sys.modules,'pyzed',None)
    assert main([])==0
    result=json.loads(capsys.readouterr().out)
    assert result['nodes']==9 and result['edges']==8
    assert result['robot_calls']==0 and result['execution_allowed'] is False


def test_cli_has_no_execute_option():
    with pytest.raises(SystemExit) as error:
        main(['--execute'])
    assert error.value.code==2
