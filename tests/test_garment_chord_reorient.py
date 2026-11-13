import json
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from PIL import Image
from cloth_agent.garment_chord_reorient import geometry,contour_crossings,NAME
from cloth_agent.contour_normal_grasp import NormalGraspError
from cloth_agent.experiments import unfold_skill_planner as planner
from cloth_agent.experiments import unfold_goal_boundary as boundary


def scene():
    mask=np.zeros((300,300),np.uint8)
    cv2.ellipse(mask,(155,150),(50,85),35,0,360,1,-1)
    target=np.array([[100,60],[200,60],[200,240],[100,240]],float)
    return mask,target


def test_extreme_crossings_and_parallel_pull():
    mask,target=scene();g=geometry(mask,target,2.,40.)
    ys=[p[1] for p in g['all_intersections_px']]
    assert g['A_px'][1]==min(ys) and g['B_px'][1]==max(ys)
    direction=np.array(g['pull_unit_xy']);chord=np.array(g['B_px'])-g['A_px']
    assert abs(direction[0]*chord[1]-direction[1]*chord[0])<1e-8
    reverse=geometry(mask,target[::-1],2.,40.)
    assert reverse['grasp_px']==g['grasp_px'] and reverse['target_px']==g['target_px']


def test_no_intersection_is_unavailable():
    mask,_=scene()
    with pytest.raises(NormalGraspError,match='at least two'):
        geometry(mask,[[1,1],[299,1],[299,299],[1,299]],2.)


@pytest.mark.parametrize('missing_depth',[False,True])
def test_skill_routes_one_model_request_to_cv_and_measured_depth(tmp_path,monkeypatch,missing_depth):
    mask,target=scene();goal=tmp_path/'goal';goal.mkdir();image=goal/'target.png'
    Image.new('RGB',(640,640),'gray').save(image)
    (goal/'target.boundary.json').write_text(json.dumps(dict(goal_sha256=boundary.digest(image),world_xy_mm=((target-320)*2*[1,-1]).tolist())))
    view=tmp_path/'view';view.mkdir();(view/'view.json').write_text(json.dumps(dict(center_world_xy_mm=[0,0],mm_per_pixel=2)))
    before=SimpleNamespace(image=image,directory=view,plan_coordinates='topdown')
    samples=[]
    def sample(xy,spread):
        samples.append(xy)
        if missing_depth:
            from cloth_agent.dual_arm.observation import DepthUnavailableError
            raise DepthUnavailableError('missing source depth')
    before.sample=sample
    monkeypatch.setattr(planner,'segment_observation',lambda before:(mask,{'method':'fixture'}))
    folder=tmp_path/'round';folder.mkdir();(folder/'planning_skills.json').write_text('{}')
    body=dict(instructions='test',input=[dict(content=[dict(type='input_image',image_url='current'),dict(type='input_image',image_url='goal')])],text=dict(format={}))
    calls=[]
    def direct(folder,request,headers,timeout):
        calls.append(request)
        assert NAME in request['text']['format']['schema']['properties']['skill']['anyOf'][0]['enum']
        return dict(decision='ACTION',skill=NAME,region_name='朝向纠正',evidence='明显倾斜',grasp_px=None,target_px=None,boundary_point_px=None,chord_endpoints_px=[],pull_distance_mm=40),{}
    plan,_=planner._run_once(tmp_path,folder,before,body,{},1,direct,{})
    assert len(calls)==1
    if missing_depth:
        assert plan['decision']=='ABSTAIN'
        assert json.loads((folder/'skill_failure.json').read_text())['stage']=='FUSED_DEPTH'
        return
    assert plan['decision']=='ACTION'
    assert samples==[plan['grasp_px'],plan['target_px']]
    assert (folder/'chord_reorientation.json').is_file()
    assert json.loads((folder/'planning_skills.json').read_text())['local_cv_executed']
