import json
from pathlib import Path
import pytest
from PIL import Image
from cloth_agent.experiments import skill_gate_pilot as p


@pytest.fixture
def pilot(tmp_path):
    root=p.init(tmp_path/'pilot')
    image=tmp_path/'rgb.png';Image.new('RGB',(32,32),'white').save(image)
    p.bind_goal(root,image,'same garment target')
    for group in p.GROUPS:p.canonical_before(root,group,image,'operator arranged')
    return root


def test_assignment_is_first_step_only():
    assert [p.assigned_skill('A',i) for i in range(4)]==['SIMPLE']*4
    assert [p.assigned_skill('B',i) for i in range(4)]==['TANGENT','SIMPLE','SIMPLE','SIMPLE']


def test_preregistered_order_and_holdout_block_before_hardware(pilot):
    with pytest.raises(ValueError,match='order'):p.trial(pilot,'G01','B','ready')
    with pytest.raises(ValueError,match='predict'):p.trial(pilot,'G03','A','ready',execute=True)
    assert p.trial(pilot,'G01','A','ready')['robot_calls']==0
    assert not (pilot/'G01/A').exists()


def test_no_fake_completed_report(pilot):
    result=p.report(pilot)
    assert result['status']=='INCOMPLETE'
    assert result['gate']['status']=='NOT_LEARNED'
    assert not (pilot/'gate.json').exists()
    assert all(r['correct'] is None for r in result['groups'])


def test_discovery_whitelists_only_train(pilot):
    for group in ('G01','G02'):
        p.save(pilot/group/'comparison.json',{'winner':'INCOMPARABLE','reason':'no clear preference'})
        for arm in ('A','B'):
            folder=pilot/group/arm/'initial_observation/topdown';folder.mkdir(parents=True)
            Image.new('RGB',(32,32),'white').save(folder/'planning_rgb.png')
    for group in ('G03','G04'):
        p.save(pilot/group/'comparison.json',{'secret':'DO_NOT_LEAK_HELD_OUT_RESULT'})
    body,evidence=p.discovery_payload(pilot)
    text=json.dumps(body,ensure_ascii=False)
    assert set(evidence)=={'G01','G02'}
    assert 'DO_NOT_LEAK' not in text and 'G03' not in text and 'G04' not in text
    assert body['tools']==[] and body['tool_choice']=='none'
    assert '隐藏部位→Tangent' not in body['instructions']


def test_no_reveal_before_both_predictions(pilot):
    with pytest.raises(ValueError,match='both predictions'):
        p.record_comparison(pilot,'G03',{})


def test_protocol_and_goal_tampering_detected(pilot):
    data=p.read(pilot/'protocol.json');data['train']=['G03','G04']
    p.save(pilot/'protocol.json',data)
    with pytest.raises(ValueError,match='protocol changed'):p.verify(pilot)


def test_unavailable_skill_cannot_be_called_a_loss(pilot):
    for arm in ('A','B'):
        p.save(pilot/'G01'/arm/'trajectory.json',{'status':'ABSTAIN','first_step_applicability':'NOT_APPLICABLE'})
    value=dict(reviewer='offline reviewer',matching='MATCHED',winner='A',reason='test',evidence=['A/trajectory.json'])
    for arm in ('A','B'):
        value[arm]=dict(semantic_parts_by_step=[{'step':0,'status':'UNKNOWN'}],final_flatness=None,
            task_complete=None,first_step_effect='UNKNOWN',side_effects=[],applicability='NOT_APPLICABLE')
    with pytest.raises(ValueError,match='unavailable'):p.record_comparison(pilot,'G01',value)
    value['winner']='INCOMPARABLE'
    p.record_comparison(pilot,'G01',value)


@pytest.mark.parametrize('invalid_first',[False,True])
def test_real_episode_adapter_four_step_limit_and_no_fallback(pilot,monkeypatch,invalid_first):
    from types import SimpleNamespace
    from cloth_agent.experiments import partitioned_unfold as unfold
    from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
    from cloth_agent import replay_dual_arm_runtime,dual_cloud_observation
    calls=[];counter=[0]
    def observe(folder):
        folder.mkdir(parents=True,exist_ok=True)
        image=folder/'rgb.png';counter[0]+=1
        Image.new('RGB',(128,128),(counter[0]*40,0,0)).save(image)
        return SimpleNamespace(directory=folder,image=image)
    runtime=SimpleNamespace(observe=observe,close=lambda:None,
        execute=lambda *args: calls.append('EXECUTE') or dict(status='EXECUTED',physical_execution=True))
    monkeypatch.setattr(replay_dual_arm_runtime,'PartitionedReplayRuntime',lambda *a,**k:runtime)
    monkeypatch.setattr(unfold,'configure_runtime',lambda x:x)
    monkeypatch.setattr(dual_cloud_observation,'load_config',lambda p:{})
    monkeypatch.setattr(io,'auth_headers',lambda:{})
    monkeypatch.setattr(unfold,'prepare_action',lambda *a:'existing-source')
    # G02 is preregistered B then A.
    def plan(root,folder,before,skill,*args):
        calls.append(skill)
        if invalid_first and skill=='TANGENT':return dict(decision='ABSTAIN'),{}
        return dict(decision='ACTION',grasp_px=[counter[0]*20,5],target_px=[counter[0]*20+10,10],
            motion_type='lift_pull_place',grasp_angle_deg=None,is_smoothing=False,
            action_purpose='test',expected_stop_state='test',short_reason='test'),{}
    monkeypatch.setattr(p,'plan_assigned',plan)
    result=p.trial(pilot,'G02','B','operator reset complete',execute=True)
    if invalid_first:
        assert calls==['TANGENT']
        assert result['first_step_applicability']=='NOT_APPLICABLE'
        assert result['physical_actions']==0
    else:
        assert calls==['TANGENT','EXECUTE','SIMPLE','EXECUTE','SIMPLE','EXECUTE','SIMPLE','EXECUTE']
        assert result['physical_actions']==4
        assert counter[0]==5  # initial + normal after each action; no extra observations


@pytest.mark.parametrize('skill',['SIMPLE','TANGENT'])
def test_assigned_planner_single_request_and_no_handwritten_gate(tmp_path,monkeypatch,skill):
    from types import SimpleNamespace
    import numpy as np
    from cloth_agent.experiments import unfold_topdown,unfold_planning_skills
    root=tmp_path;(root/'goal').mkdir();folder=root/'round';folder.mkdir()
    image=root/'rgb.png';Image.new('RGB',(100,100),'white').save(image)
    Image.new('RGB',(100,100),'white').save(root/'goal/target.png')
    before=SimpleNamespace(image=image,raw_plan=lambda v:v,sample=lambda pixel,spread:np.r_[pixel,50.])
    monkeypatch.setattr(unfold_topdown,'primary',lambda *a:before)
    monkeypatch.setattr(unfold_planning_skills,'workspace_prompt',lambda *a:'existing workspace')
    calls=[]
    def direct(folder,body,headers,timeout):
        calls.append(body)
        assert body['tools']==[]
        assert sum(c['type']=='input_image' for c in body['input'][0]['content'])==2
        assert '仅当衣角或袖口' not in body['instructions']
        assert '该部位未显露且附近有明显叠层' not in body['instructions']
        return dict(decision='ABSTAIN',skill=None,region_name='',evidence='no valid candidate',
            grasp_px=None,target_px=None,boundary_point_px=None,chord_endpoints_px=[],pull_distance_mm=10),{'wall_latency_s':1}
    value,_=p.plan_assigned(root,folder,before,skill,{},1,direct,object())
    assert value['decision']=='ABSTAIN'
    assert len(calls)==1
