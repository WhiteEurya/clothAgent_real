from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from PIL import Image

from cloth_agent.contour_normal_grasp import NormalGraspError, build, geometry, execute_once


def rectangle():
    mask = np.zeros((220, 300), np.uint8)
    mask[40:180, 60:220] = 1
    return mask


@pytest.mark.parametrize('region,normal', [([60,110],[1,0]),([219,110],[-1,0]),
                                        ([140,40],[0,1]),([140,179],[0,-1])])
def test_opposite_intersection_and_outward_pull(region, normal):
    mask=rectangle()
    g=geometry(mask,region,pull_px=35)
    assert np.allclose(g['normal_unit_xy'],normal)
    pa,pb=np.array(g['grasp_px']),np.array(g['target_px'])
    assert np.dot(pb-pa,normal)>0
    assert mask[pa[1],pa[0]] and not mask[pb[1],pb[0]]
    assert np.linalg.norm(np.array(g['intersection_px'])-pa)==pytest.approx(8)


def test_rotation_equivariance_of_normal():
    mask=np.zeros((400,400),np.uint8)
    corners=cv2.boxPoints(((200,200),(160,100),30)).astype(int)
    cv2.fillPoly(mask,[corners],1)
    point=(corners[0]+corners[1])/2
    g=geometry(mask,point,pull_px=40)
    a,b=np.array(g['tangent_unit_xy']),np.array(g['normal_unit_xy'])
    assert abs(np.dot(a,b))<1e-8
    assert abs(np.dot(b,(np.array([200,200])-point)/np.linalg.norm(np.array([200,200])-point)))>.99


def test_concave_multiple_crossings_abstain():
    mask=np.zeros((220,300),np.uint8)
    mask[30:190,40:80]=1;mask[30:190,180:220]=1;mask[170:190,40:220]=1
    with pytest.raises(NormalGraspError,match='intersections'):
        geometry(mask,[40,100],pull_px=50)


def test_hole_at_grasp_is_not_filled_to_invent_material():
    mask=rectangle();mask[103:118,205:217]=0
    with pytest.raises(NormalGraspError,match='visible segmented material'):
        geometry(mask,[60,110],pull_px=35)


@pytest.mark.parametrize('kwargs', [dict(pull_px=2),dict(inset_px=-1),dict(pull_px=float('nan')),
                                  dict(pull_px=500),dict(tangent_window_px=0)])
def test_invalid_or_non_table_target_rejected(kwargs):
    with pytest.raises(NormalGraspError):geometry(rectangle(),[60,110],**kwargs)


def test_sharp_corner_and_clipped_foreground_rejected():
    with pytest.raises(NormalGraspError,match='ambiguous'):geometry(rectangle(),[60,40])
    mask=rectangle();mask[:,0]=1;mask[100,:80]=1
    with pytest.raises(NormalGraspError,match='clipped'):geometry(mask,[0,100])


def fixture_proposal(tmp_path):
    image=tmp_path/'rgb.png';Image.new('RGB',(300,220),'purple').save(image)
    mask=tmp_path/'visible.png';Image.fromarray(rectangle()*255).save(mask)
    proposal=build(image,[60,110],'左侧下摆','左侧下摆内收',tmp_path/'preview',mask_path=mask,pull_px=35)
    before=SimpleNamespace(image=image,validate_for=Mock())
    runtime=SimpleNamespace(config=SimpleNamespace(),capture_current=Mock(return_value=before),
        arrival=Mock(),_execute_operations=Mock(return_value={'status':'EXECUTED'}),
        move_to_perception=Mock(),cancel=SimpleNamespace(set=Mock()),stop=Mock())
    return before,proposal,runtime


@pytest.mark.parametrize('check_spread', [True, False])
def test_only_one_dispatch_and_existing_checks_retained(tmp_path,monkeypatch,check_spread):
    before,proposal,runtime=fixture_proposal(tmp_path)
    from cloth_agent import preference_one_shot
    from cloth_agent.experiments import trajectory_memory_full_episode_v1 as host
    verify=Mock();prepare=Mock(return_value=(['checked'],{'initial':'joints'}))
    monkeypatch.setattr(preference_one_shot,'verify_scene',verify)
    monkeypatch.setattr(host,'prepare_action',prepare)
    folder=tmp_path/'execute'
    execute_once(runtime,before,proposal,folder,check_grasp_height_spread=check_spread)
    verify.assert_called_once();prepare.assert_called_once()
    assert prepare.call_args.kwargs['check_grasp_height_spread'] is check_spread
    assert prepare.call_args.kwargs['target_sampler'].__name__=='project_target'
    runtime._execute_operations.assert_called_once_with(['checked'],folder)
    with pytest.raises(Exception,match='already been dispatched'):execute_once(runtime,before,proposal,folder)
    assert runtime._execute_operations.call_count==1


def test_host_rejection_never_dispatches(tmp_path,monkeypatch):
    before,proposal,runtime=fixture_proposal(tmp_path)
    from cloth_agent import preference_one_shot
    from cloth_agent.experiments import trajectory_memory_full_episode_v1 as host
    monkeypatch.setattr(preference_one_shot,'verify_scene',Mock())
    monkeypatch.setattr(host,'prepare_action',Mock(side_effect=ValueError('depth unavailable')))
    with pytest.raises(ValueError,match='depth unavailable'):execute_once(runtime,before,proposal,tmp_path/'execute')
    runtime._execute_operations.assert_not_called()
    assert not (tmp_path/'execute/dispatch_intent.json').exists()


def test_mismatched_image_and_changed_plan_do_not_capture_or_dispatch(tmp_path):
    before,proposal,runtime=fixture_proposal(tmp_path)
    proposal['plan']['target_px'][0]+=1
    with pytest.raises(Exception,match='coordinates changed'):execute_once(runtime,before,proposal,tmp_path/'execute')
    runtime.capture_current.assert_not_called()
    proposal['observation_image_sha256']='wrong'
    with pytest.raises(Exception,match='does not belong'):execute_once(runtime,before,proposal,tmp_path/'execute')
    runtime._execute_operations.assert_not_called()


def test_returned_execution_failure_stops_without_return_move(tmp_path,monkeypatch):
    before,proposal,runtime=fixture_proposal(tmp_path)
    from cloth_agent import preference_one_shot
    from cloth_agent.experiments import trajectory_memory_full_episode_v1 as host
    monkeypatch.setattr(preference_one_shot,'verify_scene',Mock())
    monkeypatch.setattr(host,'prepare_action',Mock(return_value=([],{})))
    runtime._execute_operations.return_value={'status':'EXECUTION_FAILED'}
    result=execute_once(runtime,before,proposal,tmp_path/'execute')
    assert result['status']=='EXECUTION_FAILED'
    runtime.stop.assert_called_once();runtime.move_to_perception.assert_not_called()
