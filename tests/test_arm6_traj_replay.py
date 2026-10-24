import numpy as np
import pytest
from cloth_agent.experiments.arm6_traj_replay import instantiate_poses, compile_templates


def test_rejected_demo_requires_specific_authorization():
    from cloth_agent.experiments.arm6_traj_replay import validate_demonstration_authorization
    action = dict(manifest=dict(path='/record/traj.json', sha256='original'))
    manifest = dict(status='REJECTED', operator_accepted=False)
    with pytest.raises(ValueError):
        validate_demonstration_authorization(action, manifest)
    action['replay_authorization'] = dict(kind='USER_REQUESTED_REJECTED_DEMONSTRATION',
        manifest=action['manifest'].copy(), reason='User requested this iteration')
    validate_demonstration_authorization(action, manifest)
    with pytest.raises(ValueError):
        validate_demonstration_authorization(action, dict(status='ABORTED', operator_accepted=False))
    action['replay_authorization']['manifest']['sha256'] = 'another-record'
    with pytest.raises(ValueError):
        validate_demonstration_authorization(action, manifest)


def template():
    return dict(seed_start_mm=[100,100,30],seed_target_mm=[200,100,30],
        grasp_offset_mm=[1,2,-15], measured_grasp_mm_deg=[101,102,15,180,0,0],
        lift_above_surfaces_mm=30, approach_height_mm=50,place_offset_mm=5,clearance_mm=30)


def test_same_anchors_reproduce_measured_grasp_and_lift():
    poses=instantiate_poses(template(),[100,100,30],[200,100,30])
    assert poses['grasp'][:3]==pytest.approx([101,102,15])
    assert poses['lifted'][:3]==pytest.approx([101,102,60])
    assert poses['placed'][:3]==pytest.approx([200,100,35])
    assert poses['retreat'][2]==65


def test_live_anchors_rotate_offsets_and_use_current_depth():
    poses=instantiate_poses(template(),[300,200,40],[300,300,45])
    assert poses['grasp'][:3]==pytest.approx([298,201,25])
    assert poses['lifted'][2]==75
    assert poses['placed'][:3]==pytest.approx([300,300,50])
    from scipy.spatial.transform import Rotation
    assert Rotation.from_euler('xyz',poses['grasp'][3:],degrees=True).as_matrix()==pytest.approx(
        (Rotation.from_euler('z',90,degrees=True)*Rotation.from_euler('xyz',[180,0,0],degrees=True)).as_matrix())


def test_bad_anchor_or_insufficient_lift_rejected():
    with pytest.raises(ValueError):instantiate_poses(template(),[0,0,30],[0,1,30])
    with pytest.raises(ValueError):instantiate_poses(template(),[float('nan'),0,30],[0,100,30])
    value=template();value['lift_above_surfaces_mm']=0
    with pytest.raises(ValueError):instantiate_poses(value,[0,0,30],[0,100,30])


def test_arm6_cli_routes_without_calling_arm7_runtime(tmp_path,monkeypatch):
    from cloth_agent.experiments import graph_policy_replay as cli
    from cloth_agent.experiments import arm6_traj_replay as arm6
    monkeypatch.setattr(cli,'load_frozen',lambda _:dict(protocol=arm6.PROTOCOL))
    seen=[]
    monkeypatch.setattr(arm6,'run_cli',lambda args,parser:seen.append(args.execute) or dict(status='DRY_RUN_OK'))
    assert cli.main(['--graph',str(tmp_path/'graph.json'),'--dry-run'])==0
    assert seen==[False]


def test_matching_excludes_unobserved_sink():
    from cloth_agent.experiments.arm6_traj_replay import matching_graph
    graph=dict(nodes=[dict(node_id='observed',representative_observations=['rgb']),
                      dict(node_id='unknown',representative_observations=[])])
    matched=matching_graph(graph)
    assert [n['node_id'] for n in matched['match_nodes']]==['observed']
    assert len(graph['nodes'])==2


def test_expanded_choices_exclude_blocked_and_tried_actions():
    from cloth_agent.experiments.arm6_traj_replay import executable_choices
    graph=dict(edges=[dict(edge_id=k,source_node='n',target_node=t)
                      for k,t in [('blocked','x'),('loop','n'),('progress','p')]])
    templates={'loop':{},'progress':{}}
    outgoing,supported,choices=executable_choices(graph,templates,'n',set())
    assert len(outgoing)==3 and len(supported)==2
    assert [e['edge_id'] for e in choices]==['progress']
    assert [e['edge_id'] for e in executable_choices(graph,templates,'n',{'progress'})[2]]==['loop']
    assert executable_choices(graph,templates,'n',{'progress','loop'})[2]==[]


def test_expanded_cli_stops_before_unsupported_dispatch(tmp_path,monkeypatch):
    from cloth_agent.experiments import arm6_traj_replay as replay
    from cloth_agent.experiments import graph_policy_replay as cli
    from cloth_agent.experiments import replay_live_graph
    from cloth_agent import gamepad_collect_runtime
    from types import SimpleNamespace
    graph=dict(protocol=replay.PROTOCOL,nodes=[dict(node_id='x',representative_observations=['o'])],
               edges=[dict(edge_id='manual',source_node='x',target_node='y')],terminal_node_ids=[],
               replay_unavailable_edges={'manual':'unsupported manual trajectory'})
    path=tmp_path/'graph.json';path.write_text('{}')
    rgb=tmp_path/'rgb.png';rgb.write_bytes(b'fixture')
    monkeypatch.setattr(cli,'load_frozen',lambda _:graph)
    monkeypatch.setattr(replay,'compile_templates',lambda *a,**kw:(graph,{'observations':{}},{}))
    monkeypatch.setattr(gamepad_collect_runtime,'CollectionRuntime',lambda *a:SimpleNamespace(observe=lambda *a:SimpleNamespace(image=rgb)))
    monkeypatch.setattr(replay_live_graph,'create_live_graph',lambda *a:SimpleNamespace(update=lambda *a,**kw:None,close=lambda:None))
    monkeypatch.setattr(replay,'close_cli_runtime',lambda *a:None)
    seen=[]
    def match(*a,**kw):
        seen.append(kw['pending'])
        return dict(status='MATCH',node_id='x',candidate_nodes=['x'],confidence=.95,
                    matched_evidence=['RGB correspondence'],violated_conditions=[],unresolved_conditions=[])
    monkeypatch.setattr(replay,'topology_match',match)
    cli.main(['--graph',str(path),'--arm6-expanded-graph','--execute','--no-live-graph',
              '--output',str(tmp_path/'run')])
    import json
    result=json.loads((tmp_path/'run/result.json').read_text())
    assert result['status']=='ACTION_TEMPLATE_UNAVAILABLE'
    assert result['physical_actions']==0
    assert seen==[None]


@pytest.mark.parametrize('protocol',['arm6-traj-experience-graph-v1','arm6-archive-graph-v1'])
def test_archive_cli_routes_to_arm6(tmp_path,monkeypatch,protocol):
    from cloth_agent.experiments import graph_policy_replay as cli
    from cloth_agent.experiments import arm6_traj_replay as arm6
    monkeypatch.setattr(cli,'load_frozen',lambda _:dict(protocol=protocol))
    seen=[]
    monkeypatch.setattr(arm6,'run_cli',lambda args,parser:seen.append(args.execute) or dict(status='DRY_RUN_OK'))
    assert cli.main(['--graph',str(tmp_path/'graph.json'),'--arm6-expanded-graph','--dry-run'])==0
    assert seen==[False]


def automatic_action(tmp_path):
    import json
    from cloth_agent.experiments.graph_policy import evidence_file
    grasp=[100,100,15,180,0,0];lift=[100,100,80,180,0,0]
    across=[200,100,80,180,0,0];placed=[200,100,35,180,0,0];retreat=[200,100,65,180,0,0]
    def grip(action,**kw):return dict(kind='gripper',data=dict(arm='left',action=action,**kw))
    def move(pose):return dict(kind='move',data=dict(arm='left',target_pose_mm_deg=pose))
    plan=[grip('open'),move([100,100,80,180,0,0]),move(grasp),grip('close'),move(lift),move(across),move(placed),
          grip('release_partial',opening_delta_pulse=180,settle_s=.5),move(retreat)]
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    action=dict(type='ARM6_AUTOMATIC_PICK_PLACE_DEMONSTRATION',selected_points=[
        dict(name='start',world_xyz_mm=[100,100,30],pixel_xy=[100,100]),
        dict(name='target',world_xyz_mm=[200,100,30],pixel_xy=[200,100])],
        automated_phases=[dict(phase='automatic_pick_place',result=dict(status='EXECUTED'),actual_native_plan=evidence_file(path))])
    return action,dict(parameters=dict(approach_height_mm=50)),plan,path


def test_automatic_replay_preserves_executed_poses_and_release(tmp_path):
    from cloth_agent.experiments.arm6_traj_replay import compile_demonstration_template
    action,manifest,plan,path=automatic_action(tmp_path)
    result=compile_demonstration_template({'edge_id':'e'},action,manifest,[],[])
    poses=instantiate_poses(result,[100,100,30],[200,100,30])
    for key,index in [('grasp',2),('lifted',4),('across',5),('placed',6),('retreat',8)]:
        assert poses[key][:3]==pytest.approx(plan[index]['data']['target_pose_mm_deg'][:3])
    assert result['release_delta_pulse']==180 and result['release_settle_s']==.5
    assert result['grasp_pose_source']=='executed_native_plan_target'


@pytest.mark.parametrize('fault',['full_open','changed_rotation','wrong_target','missing_pose','wrong_arm'])
def test_automatic_template_rejects_unrepresentable_dispatch(tmp_path,fault):
    import json
    from cloth_agent.experiments.arm6_traj_replay import compile_automatic_template
    from cloth_agent.experiments.graph_policy import evidence_file
    action,manifest,plan,path=automatic_action(tmp_path)
    if fault=='full_open':plan[7]['data']['action']='open'
    if fault=='changed_rotation':plan[5]['data']['target_pose_mm_deg'][5]=30
    if fault=='wrong_target':plan[5]['data']['target_pose_mm_deg'][0]=300
    if fault=='missing_pose':del plan[5]['data']['target_pose_mm_deg']
    if fault=='wrong_arm':plan[5]['data']['arm']='right'
    path.write_text(json.dumps(plan));action['automated_phases'][0]['actual_native_plan']=evidence_file(path)
    with pytest.raises(ValueError):compile_automatic_template(action,manifest)
