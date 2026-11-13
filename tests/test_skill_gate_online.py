import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from cloth_agent.experiments import skill_gate_online as online
from cloth_agent.experiments import partitioned_unfold as unfold


@pytest.fixture
def scene(tmp_path,monkeypatch):
    from cloth_agent.experiments import unfold_topdown,unfold_planning_skills
    (tmp_path/'goal').mkdir();image=tmp_path/'rgb.png'
    Image.new('RGB',(300,200),'white').save(image)
    Image.new('RGB',(300,200),'white').save(tmp_path/'goal/target.png')
    before=SimpleNamespace(directory=tmp_path,image=image,meta={'observation_id':'obs-fixed'},
        raw_plan=lambda plan:plan,sample=lambda p,s:np.r_[p,50.])
    monkeypatch.setattr(unfold_topdown,'primary',lambda *args:before)
    monkeypatch.setattr(unfold_planning_skills,'workspace_prompt',lambda *args:'existing workspace')
    return tmp_path,before


def response(skill=None,decision='ABSTAIN'):
    return dict(decision=decision,skill=skill,region_name='test',evidence='test reason',
        grasp_px=[100,100] if decision=='ACTION' and skill is None else None,
        target_px=[140,100] if decision=='ACTION' and skill is None else None,
        boundary_point_px=[60,100] if decision=='ACTION' and skill else None,
        chord_endpoints_px=[],pull_distance_mm=40,grasp_angle_deg=None)


def test_reproducible_draw_not_posthoc():
    a=[online.assignment(2026110601,r,i) for r in range(1,7) for i in range(12)]
    b=[online.assignment(2026110601,r,i) for r in range(1,7) for i in range(12)]
    assert a==b
    assert {r['forced_skill'] for r in a}=={'SIMPLE','TANGENT'}
    for r in a:
        assert (r['forced_skill']=='TANGENT')==(r['draw']<.3)
        assert r['selection_probability']==online.PROBABILITIES[r['forced_skill']]
    with pytest.raises(ValueError):online.assignment(1,7,0)


@pytest.mark.parametrize('skill',['SIMPLE','TANGENT'])
def test_forced_one_request_abstain_no_second_branch(scene,skill):
    root,before=scene;folder=root/'round_000';folder.mkdir();calls=[]
    def direct(folder,body,*args):
        calls.append(body)
        assert sum(x['type']=='input_image' for x in body['input'][0]['content'])==3
        assert '该部位未显露且附近有明显叠层' not in body['instructions']
        from cloth_agent.experiments.unfold_topdown import PRIMARY_GUIDANCE
        assert PRIMARY_GUIDANCE in body['instructions']
        from cloth_agent.experiments.unfold_skill_planner import ACTION_GUIDANCE,TANGENT_GUIDANCE
        assert ACTION_GUIDANCE in body['instructions']
        assert (TANGENT_GUIDANCE in body['instructions']) == (skill == 'TANGENT')
        if skill == 'SIMPLE':
            assert '普通展开10–150 mm，抚平10–30 mm' in body['instructions']
            assert '不是像素距离' in body['instructions']
        return response(),{'wall_latency_s':1}
    value,metrics=unfold.plan_round(root,folder,before,{},1,direct,runtime=object(),forced_skill=skill)
    assert value['decision']=='ABSTAIN' and len(calls)==1
    assert json.loads((folder/'depth_reselection.json').read_text())['max_attempts']==1


@pytest.mark.parametrize('skill',['SIMPLE','TANGENT'])
def test_invalid_candidate_never_reselects(scene,monkeypatch,skill):
    from cloth_agent.experiments import unfold_skill_planner as skills
    from cloth_agent.dual_arm.observation import DepthUnavailableError
    root,before=scene;folder=root/'round_000';folder.mkdir();calls=[]
    if skill=='TANGENT':
        def fail(*a):raise skills.normal.NormalGraspError('invalid segmentation')
        monkeypatch.setattr(skills,'segment',fail)
    else:
        def fail(*a):raise DepthUnavailableError('unsupported pixel')
        before.sample=fail
    def direct(*args):
        calls.append(1)
        return response(None if skill=='SIMPLE' else 'garment-tangent-grasp','ACTION'),{'wall_latency_s':1}
    value,_=unfold.plan_round(root,folder,before,{},1,direct,runtime=object(),forced_skill=skill)
    assert value['decision']=='ABSTAIN' and len(calls)==1
    assert not list(folder.glob('depth_reselection_*'))


def test_draw_logged_before_model_and_unselected_is_unknown(scene):
    root,before=scene;folder=root/'round_000';folder.mkdir()
    def direct(*args):
        row=json.loads((folder/'exploration.json').read_text())
        assert row['planning_status']=='PENDING' and row['observation_id']=='obs-fixed'
        return response(),{'wall_latency_s':1}
    req=online.RandomizedRequest(root,seed=3,run_id=1,runtime=object(),headers={},timeout=1,direct=direct)
    plan,_=req(folder,before)
    row=json.loads((folder/'exploration.json').read_text())
    assert row['model_requests']==1 and plan['decision']=='ABSTAIN'
    assert row['planning_success'] is False
    other=({'SIMPLE','TANGENT'}-{row['forced_skill']}).pop()
    assert row['availability'][other]['status']=='NOT_EVALUATED'
    with pytest.raises(ValueError,match='no redraw'):req(folder,before)


def test_transport_failure_is_logged_without_retry(scene):
    root,before=scene;folder=root/'round_000';folder.mkdir();calls=[]
    def direct(*args):calls.append(1);raise RuntimeError('transport failed')
    req=online.RandomizedRequest(root,seed=3,run_id=1,runtime=object(),headers={},timeout=1,direct=direct)
    plan,_=req(folder,before)
    assert plan['decision']=='ABSTAIN' and calls==[1]
    assert 'transport failed' in json.loads((folder/'exploration.json').read_text())['abstain_reason']


def test_wrong_goal_rejected_before_any_runtime_call(scene):
    root,before=scene
    with pytest.raises(ValueError,match='goal is not configured'):
        online.run_episode(object(),root,seed=3,run_id=1,headers={},timeout=1,
            direct=lambda *a:pytest.fail('model must not be called'),initialize_seed=1,radius_mm=20)


def test_single_distance_failure_does_not_replan(scene,monkeypatch):
    root,before=scene;requests=[];executions=[]
    def request(*a):
        requests.append(1)
        return dict(decision='ACTION',grasp_px=[20,20],target_px=[200,100]),{}
    def reject(*a):raise unfold.TranslationRejected({'distance_mm':180,'maximum_mm':150})
    monkeypatch.setattr(unfold,'prepare_action',reject)
    runtime=SimpleNamespace(observe=lambda path:before,execute=lambda *a:executions.append(1),close=lambda:None)
    outcome=unfold.episode(runtime,root,max_actions=12,request=request,max_plan_attempts=1)
    assert requests==[1] and executions==[] and outcome['status']=='PLANNING_REJECTED'


@pytest.mark.parametrize('skill',['SIMPLE','TANGENT'])
def test_both_branches_send_explicit_run_goal(scene,skill):
    from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
    root,before=scene;folder=root/'round_000';folder.mkdir()
    Image.new('RGB',(300,200),(20,80,40)).save(root/'goal/target.png')
    expected=io.image_data(root/'goal/target.png');calls=[]
    def direct(destination,body,*args):
        images=[x['image_url'] for x in body['input'][0]['content'] if x['type']=='input_image']
        assert len(images)==3 and images[1]==expected and images[2]==io.image_data(before.image)
        calls.append(1)
        return response(),{'wall_latency_s':1}
    unfold.plan_round(root,folder,before,{},1,direct,runtime=object(),forced_skill=skill)
    assert calls==[1]


def test_collect_records_keeps_action_geometry_execution_and_no_counterfactual(scene):
    root,before=scene;folder=root/'round_000';folder.mkdir()
    record=online.assignment(3,1,0)
    record.update(planning_status='ACTION',before_rgb=str(before.image),action={'decision':'ACTION'})
    (folder/'exploration.json').write_text(json.dumps(record))
    geometry={'grasp_world_mm':[100,200,30],'target_world_mm':[140,200,30]}
    (folder/'geometry.json').write_text(json.dumps(geometry))
    executed=[{'name':'close_gripper','status':'completed'}]
    outcome={'status':'EXECUTED','physical_execution':True,'experiment':{'actual_robot_actions':executed}}
    (root/'episode_summary.json').write_text(json.dumps({'status':'ACTION_LIMIT','steps':[{'round':0,'execution':outcome}]}))
    rows=online.collect_records(root)
    assert rows[0]['action_geometry']==geometry and rows[0]['actual_robot_actions']==executed
    assert rows[0]['execution']==outcome and rows[0]['planning_success'] is True
    assert json.loads((root/'exploration_summary.json').read_text())['counterfactual_preference'] is None
