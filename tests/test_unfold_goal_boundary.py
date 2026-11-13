import copy
import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from cloth_agent.experiments import unfold_goal_boundary as boundary
from cloth_agent.experiments import unfold_skill_planner as planner


def fixture(tmp_path):
    goal=tmp_path/'goal';goal.mkdir()
    image=goal/'target.png';Image.new('RGB',(640,640),'gray').save(image)
    (goal/'target.boundary.json').write_text(json.dumps(dict(goal_sha256=boundary.digest(image),world_xy_mm=[[0,0],[20,0],[20,20]])))
    view=tmp_path/'view';view.mkdir()
    (view/'view.json').write_text(json.dumps(dict(center_world_xy_mm=[0,0],mm_per_pixel=2)))
    before=SimpleNamespace(image=image,directory=view,plan_coordinates='topdown',sample=lambda xy,spread:np.r_[xy,50.])
    folder=tmp_path/'round';folder.mkdir();(folder/'planning_skills.json').write_text('{}')
    body=dict(instructions='test',input=[dict(content=[dict(type='input_image',image_url='current'),dict(type='input_image',image_url='goal')])],text=dict(format={}))
    return before,folder,body


def test_world_alignment():
    np.testing.assert_allclose(boundary.project([[100,80]],dict(center_world_xy_mm=[80,100],mm_per_pixel=2)),[[330,330]])


def test_goal_boundary_does_not_enable_removed_reorientation(tmp_path,monkeypatch):
    before,folder,body=fixture(tmp_path);calls=[];samples=[]
    before.sample=lambda xy,spread:samples.append(xy)
    def forbidden(*args):raise AssertionError('simple must not run geometry')
    monkeypatch.setattr(planner,'segment_observation',forbidden)
    def direct(folder,request,headers,timeout):
        calls.append(request)
        offered=request['text']['format']['schema']['properties']['skill']['anyOf'][0]['enum']
        assert boundary.CHORD_NAME not in offered
        assert 'garment-reposition' not in offered
        assert 'garment-reposition' not in request['instructions']
        assert boundary.CHORD_NAME not in request['instructions']
        assert request['input'][0]['content'][0]['image_url']=='current'
        return dict(decision='ACTION',skill=None,region_name='袖口',evidence='可见袖口可直接展开',grasp_px=[340,320],target_px=[320,320],boundary_point_px=None,chord_endpoints_px=[],pull_distance_mm=40),{}
    plan,_=planner._run_once(tmp_path,folder,before,body,{},1,direct,{})
    assert len(calls)==1 and len(calls[0]['input'][0]['content'])==2
    assert calls[0]['input'][0]['content'][1]['image_url']=='goal'
    assert body['input'][0]['content'][0]['image_url']=='current'
    assert samples==[[340,320],[320,320]] and plan['decision']=='ACTION'
    assert boundary.CHORD_NAME not in planner.SELECT_SCHEMA['properties']['skill']['anyOf'][0]['enum']
    assert json.loads((folder/'planning_skills.json').read_text())['local_cv_executed'] is False


def test_missing_goal_boundary_does_not_offer_reorientation(tmp_path,monkeypatch):
    before,folder,body=fixture(tmp_path)
    (tmp_path/'goal/target.boundary.json').unlink()
    def direct(folder,request,headers,timeout):
        offered=request['text']['format']['schema']['properties']['skill']['anyOf'][0]['enum']
        assert boundary.CHORD_NAME not in offered
        assert request['input'][0]['content'][0]['image_url']=='current'
        return dict(decision='DONE',skill=None,region_name='',evidence='已展开',grasp_px=None,target_px=None,boundary_point_px=None,chord_endpoints_px=[],pull_distance_mm=10),{}
    plan,_=planner._run_once(tmp_path,folder,before,body,{},1,direct,{})
    assert plan['decision']=='DONE'


def test_wrong_target_is_not_overlaid(tmp_path):
    before,folder,body=fixture(tmp_path)
    Image.new('RGB',(640,640),'red').save(before.image)
    with pytest.raises(ValueError,match='different target'):boundary.attach(tmp_path,folder,before,body)


def test_planner_error_is_in_next_request_without_distance_diagnosis(tmp_path,monkeypatch):
    before,folder,body=fixture(tmp_path)
    (folder/'rejection_feedback.json').write_text(json.dumps(dict(
        rejection_reason='PLANNER_ERROR',physical_execution=False,error='ValueError: incomplete response')))
    calls=[]
    def direct(folder,request,headers,timeout):
        calls.append(request)
        assert 'ValueError: incomplete response' in request['instructions']
        assert '仅因XY平移距离' not in request['instructions']
        return dict(decision='DONE',skill=None,region_name='',evidence='已展开',grasp_px=None,target_px=None,boundary_point_px=None,chord_endpoints_px=[],pull_distance_mm=10),{}
    plan,_=planner.run(tmp_path,folder,before,body,{},1,direct)
    assert len(calls)==1 and plan['decision']=='DONE'
