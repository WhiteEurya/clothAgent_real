"""Independent seed-derived geometry and conservative eligibility; no robot/model."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cloth_agent.experiments import graph_template_compiler as tc
from cloth_agent.experiments import graph_policy as gp
from cloth_agent.experiments import graph_policy_replay as pr
from cloth_agent.experiments import graph_minimal_grounding as mg
from cloth_agent.dual_arm.observation import Observation
from tests.test_graph_policy import fixture_graph, write_graph


def fixture(tmp_path):
    g,t=fixture_graph(tmp_path)
    folder=tmp_path/'real';folder.mkdir()
    rng=np.random.default_rng(3)
    Image.fromarray(rng.integers(0,256,(160,160,3),dtype=np.uint8)).save(folder/'rgb.png')
    y,x=np.mgrid[:160,:160]
    xyz=np.stack((x*2.,y*2.,np.full_like(x,50.)),axis=-1)
    np.save(folder/'world_xyz_mm.npy',xyz)
    (folder/'observation.json').write_text(json.dumps(dict(schema_version=1,frame_id='world_mm',camera_serial='test',
        sha256={n:gp.file_hash(folder/n) for n in ['rgb.png','world_xyz_mm.npy']})))
    obs=Observation.load(folder)
    src='motion.grasp(right=[.08,.08,.035,3.14,0,0])\nmotion.move_tcp(right=[.10,.14,.11,3.14,.1,.2])\nmotion.move_transfer(right=[.18,.08,.05,3.14,0,.3])\n'
    step=t['steps'][0];step['commanded_action']=gp.command(src)
    step['intermediate_evidence']={'grasp_height_policy.json':dict(content=dict(mode='declared_grasp_lowering',offset_mm=15.,changes=[dict(arm='right',original_z_mm=50.,applied_z_mm=35.)]))}
    for key in t['observations']:
        t['observations'][key]=dict(rgb=gp.evidence_file(obs.image),observation=gp.evidence_file(folder/'observation.json'))
    e=g['edges'][0];e['edge_id']='any_edge_42';e['seed_action']['exact_command']=step['commanded_action']
    e['minimal_visual_evidence']=dict(status='RESOLVED',confidence=.95,
        garment_frame=dict(collar_pixel_xy=[80,30],hem_center_pixel_xy=[80,130],visual_evidence='test garment'),registration_roi=[10,10,150,150])
    # fixture_graph wrote a frozen first version; replace only the test fixture files.
    for p in [tmp_path/'seed_trajectory.json',tmp_path/'seed_trajectory.json.sha256.json']:p.unlink()
    gp.frozen(tmp_path/'seed_trajectory.json',t)
    return g,t,obs


def test_non_e001_compile_own_command_and_replay_modes(tmp_path):
    g,t,obs=fixture(tmp_path);path=write_graph(tmp_path,g,t)
    policy=tc.compile_policy(path,tmp_path/'compiled',frame_mode='garment')
    edge=policy['edges'][0]
    assert edge['minimal_status']=='MINIMAL_SUPPORTED'
    template=edge['host_motion_template']
    assert template['primitive_sequence']==['grasp','move_tcp','move_transfer']
    assert [s['z_above_surface_mm'] for s in template['profile']]==[0,60,0]
    assert template['profile'][1]['across']==pytest.approx(.6)
    grounder=mg.MinimalGrounder(tmp_path/'compiled/host_templates.json')
    candidate=grounder.expand(dict(grasp_px=[40,40],target_px=[90,40],confidence=.9),edge,obs,tmp_path/'ground')
    assert [s['pixel_xy'] for s in candidate['slots']]==[[40,40],[50,70],[90,40]]
    _,source,_=gp.instantiate(pr.primitive_candidate(candidate),[edge],obs)
    # Host applies grasp lowering later; recover all other actual seed parameters.
    ops=gp.command(source)['operations'];expected=t['steps'][0]['commanded_action']['operations']
    for i in range(1,len(ops)):
        assert ops[i]['method']==expected[i]['method']
        np.testing.assert_allclose(ops[i]['arguments']['right'],expected[i]['arguments']['right'],atol=1e-6)
    assert pr.validate_policy(policy)==t
    assert pr.dry_run(tmp_path/'compiled/policy_graph.json',tmp_path/'dry')['robot_calls']==0
    assert tc.compile_policy(path,tmp_path/'compiled',frame_mode='garment')==policy
    assert pr.main(['--graph',str(tmp_path/'compiled/policy_graph.json'),'--grounding-mode','minimal','--output',str(tmp_path/'cli')])==0


@pytest.mark.parametrize('outcome',['FAILURE','NO_PROGRESS','UNKNOWN'])
def test_negative_experience_never_default(tmp_path,outcome):
    g,t,obs=fixture(tmp_path);g['edges'][0]['seed_outcome']=outcome
    e=tc.policy_projection(g,t)['edges'][0]
    assert not e['replayable']
    r=tc.compile_edge(e,t['steps'][0],t)
    assert r['status']=='MINIMAL_UNSUPPORTED'
    assert r['seed_grasp_px']==[40,40] # negative geometry remains available as experience
    assert 'not selected' in r['reason']


@pytest.mark.parametrize('damage,reason',[
    ('frame','frame'),('lowering','lowering'),('depth','surface'),('skeleton','skeleton'),('command','command mismatch')])
def test_unreliable_inputs_fail_closed(tmp_path,damage,reason):
    g,t,obs=fixture(tmp_path);e=tc.policy_projection(g,t)['edges'][0];s=t['steps'][0]
    if damage=='frame':e.pop('minimal_visual_evidence')
    if damage=='lowering':s['intermediate_evidence']={}
    if damage=='depth':s['intermediate_evidence']['grasp_height_policy.json']['content']['changes'][0]['original_z_mm']=90.
    if damage=='skeleton':
        s['commanded_action']=copy.deepcopy(s['commanded_action']);s['commanded_action']['operations'][0]['method']='move_tcp';e['seed_action']['exact_command']=s['commanded_action']
    if damage=='command':e['seed_action']['exact_command']=dict(operations=[])
    r=tc.compile_edge(e,s,t)
    assert r['status']=='MINIMAL_UNSUPPORTED' and reason in r['reason']


def test_no_action_family_special_cases_and_no_progress_self_loop(tmp_path):
    g,t,obs=fixture(tmp_path)
    for family in ['left_sleeve_inward_fold','right_torso_side_inward_fold','bottom_hem_toward_collar_fold','new_family']:
        g['edges'][0]['action_family']=family
        e=tc.policy_projection(g,t)['edges'][0]
        assert tc.compile_edge(e,t['steps'][0],t)['status']=='MINIMAL_SUPPORTED'
    g['edges'][0]['target_node']='N0'
    assert not tc.policy_projection(g,t)['edges'][0]['replayable']


def test_forged_replayable_or_geometry_rejected(tmp_path):
    g,t,obs=fixture(tmp_path);path=write_graph(tmp_path,g,t)
    p=tc.compile_policy(path,tmp_path/'compiled',frame_mode='garment');p['edges'][0]['seed_action']['exact_command']['operations'][0]['arguments']['right'][0]+=1
    with pytest.raises(ValueError,match='changed'):pr.validate_policy(p)


def test_two_edges_keep_distinct_executed_geometry(tmp_path):
    g,t,obs=fixture(tmp_path)
    first=tc.policy_projection(g,t)['edges'][0]
    second=copy.deepcopy(first);second['edge_id']='second_seed_edge'
    step=copy.deepcopy(t['steps'][0]);step['step_id']='s_second'
    step['commanded_action']['operations'][1]['arguments']['right']=[.12,.10,.15,3.14,.2,.4]
    second['seed_step_id']='s_second';second['seed_action']['exact_command']=step['commanded_action']
    a=tc.compile_edge(first,t['steps'][0],t);b=tc.compile_edge(second,step,t)
    assert a['status']==b['status']=='MINIMAL_SUPPORTED'
    assert a['host_motion_template']['profile']!=b['host_motion_template']['profile']
    assert a['surface_relative_height_profile']==[0,60,0]
    assert b['surface_relative_height_profile']==[0,100,0]
    assert a['host_motion_template']['exact_command_sha256']!=b['host_motion_template']['exact_command_sha256']


def test_fresh_compile_accepts_missing_legacy_metadata(tmp_path):
    import jsonschema
    from cloth_agent.experiments.graph_policy_schema import seed_compiler_schemas,SCHEMAS
    g,t,obs=fixture(tmp_path)
    jsonschema.validate(g,seed_compiler_schemas()['compile'])
    g['edges'][0].pop('minimal_visual_evidence')
    jsonschema.validate(g,SCHEMAS['compile'])
    jsonschema.validate(g,seed_compiler_schemas()['compile'])


def test_single_command_seed_then_compile_closes_robot_first(tmp_path,monkeypatch):
    from types import SimpleNamespace
    events=[]
    runtime=SimpleNamespace(close=lambda:events.append('close'),preview=None)
    monkeypatch.setattr(gp,'make_runtime',lambda *a:runtime)
    monkeypatch.setattr(gp,'seed',lambda *a,**kw:events.append('seed') or dict(status='ROUND_LIMIT'))
    def compile_run(path,requests):
        assert events==['seed','close']
        assert requests.no_cache
        assert 'minimal_visual_evidence' not in requests.schemas['compile']['properties']['edges']['items']['required']
        events.append('compile');return {}
    monkeypatch.setattr(gp,'compile_graph',compile_run)
    assert gp.main(['seed','--output',str(tmp_path/'new'),'--compile-after'])==0
    assert events==['seed','close','compile']
