import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from cloth_agent.experiments import action_local_frame as al
from cloth_agent.experiments import graph_template_compiler as tc
from cloth_agent.experiments import graph_minimal_grounding as mg
from cloth_agent.experiments import graph_policy_replay as pr
from cloth_agent.experiments import graph_policy as gp
from tests.test_graph_template_compiler import fixture
from tests.test_graph_policy import write_graph


def test_frame_projects_horizontal_and_is_right_handed():
    basis,length=al.frame([10,20,30],[40,60,300])
    np.testing.assert_allclose(basis.T@basis,np.eye(3),atol=1e-12)
    assert np.linalg.det(basis)==pytest.approx(1.) and length==50
    np.testing.assert_array_equal(basis[:,2],[0,0,1])


@pytest.mark.parametrize('a,b', [([0,0,0],[0,0,100]),([0,0,0],[9,0,0]),([0,0,0],[np.nan,10,0])])
def test_degenerate_frame_rejected(a,b):
    with pytest.raises(ValueError):al.frame(a,b)


def test_compile_without_any_landmarks_and_exact_seed_roundtrip(tmp_path,monkeypatch):
    g,t,obs=fixture(tmp_path);g['edges'][0].pop('minimal_visual_evidence')
    def forbidden(*a,**k):pytest.fail('must not register garment')
    monkeypatch.setattr(mg,'register_frame',forbidden);monkeypatch.setattr(tc,'register_frame',forbidden)
    path=write_graph(tmp_path,g,t);policy=tc.compile_policy(path,tmp_path/'new',frame_mode='action_local')
    e=policy['edges'][0];assert e['minimal_status']=='MINIMAL_SUPPORTED'
    grounder=mg.MinimalGrounder(tmp_path/'new/host_templates.json')
    candidate=grounder.expand(dict(grasp_px=[40,40],target_px=[90,40],confidence=1.),e,obs,tmp_path/'exp')
    assert 'garment_frame' not in candidate
    _,motion,_=pr.instantiate_grounded(candidate,e,obs)
    ops=gp.command(motion)['operations'];ops[0]['arguments']['right'][2]-=.015
    for a,b in zip(ops,t['steps'][0]['commanded_action']['operations']):
        np.testing.assert_allclose(a['arguments']['right'][:3],b['arguments']['right'][:3],atol=1e-10)
        np.testing.assert_allclose(Rotation.from_euler('xyz',a['arguments']['right'][3:]).as_matrix(),Rotation.from_euler('xyz',b['arguments']['right'][3:]).as_matrix(),atol=1e-10)
    assert pr.grounded_side_check(candidate,e,obs)['status']=='NOT_INDEPENDENTLY_EVALUATED'
    assert pr.validate_policy(policy)==t


def test_rotation_scaling_and_height_follow_local_frame(tmp_path):
    g,t,obs=fixture(tmp_path);p=tc.compile_policy(write_graph(tmp_path,g,t),tmp_path/'new',frame_mode='action_local');e=p['edges'][0]
    grounder=mg.MinimalGrounder(tmp_path/'new/host_templates.json');template=grounder.template(e)
    # Same requested pixels, new measured world positions: translate + rotate 90deg + double XY distance.
    a=np.array([200.,300.,70.]);b=np.array([200.,500.,90.])
    current=SimpleNamespace(xyz=obs.xyz,meta=obs.meta,sample=lambda pixel,spread:a if pixel==[40,40] else b)
    c=grounder.expand(dict(grasp_px=[40,40],target_px=[90,40],confidence=1.),e,current,tmp_path/'rotated')
    basis,_=al.frame(a,b)
    for r,pose in zip(template['profile'],c['poses']):
        xyz=np.array(pose['world_m_rad'][:3])*1000
        np.testing.assert_allclose(basis.T@(xyz-a),[r['along_fraction']*200,r['across_fraction']*200,r['height_from_grasp_mm']],atol=1e-9)
        np.testing.assert_allclose(Rotation.from_euler('xyz',pose['world_m_rad'][3:]).as_matrix(),basis@np.array(r['tool_rotation_relative']),atol=1e-9)
    # Pb Z does not tilt the basis or silently change the frozen Pa-relative height profile.
    assert c['poses'][-1]['world_m_rad'][2]*1000==pytest.approx(70.)


def test_single_minimal_model_call_and_unchanged_native_dispatch(tmp_path,monkeypatch):
    g,t,obs=fixture(tmp_path);p=tc.compile_policy(write_graph(tmp_path,g,t),tmp_path/'new',frame_mode='action_local');e=p['edges'][0]
    grounder=mg.MinimalGrounder(tmp_path/'new/host_templates.json');calls=[]
    class Requests:
        minimal_grounder=grounder
        def ask(self,kind,payload,images,validator):
            calls.append(kind);assert set(payload)=={'edge_id','grasp','target'}
            value=dict(grasp_px=[40,40],target_px=[90,40],confidence=.9);validator(value);return value
    candidate=pr.selected_edge_grounding(Requests(),p,e,obs,tmp_path/'ground')
    _,motion,_=pr.instantiate_grounded(candidate,e,obs)
    dispatched=[]
    class Runtime:
        grasp_offset_mm=15.
    monkeypatch.setattr(pr,'execute_recorded',lambda r,s,b,f:dispatched.append(s) or {'status':'TEST_ONLY'})
    folder=tmp_path/'dispatch';folder.mkdir()
    pr.execute_policy_edge(Runtime(),motion,obs,folder,p,e)
    assert calls==['ground_edge_minimal'] and dispatched==[motion]


def test_no_progress_local_geometry_saved_but_not_replayable(tmp_path):
    g,t,obs=fixture(tmp_path);g['edges'][0]['seed_outcome']='NO_PROGRESS'
    p=tc.compile_policy(write_graph(tmp_path,g,t),tmp_path/'new',frame_mode='action_local')
    records=gp.load_frozen(tmp_path/'new/edge_experiences.json')
    assert records[0]['action_local_profile'] and not p['edges'][0]['replayable']
    assert gp.load_frozen(tmp_path/'new/host_templates.json')['edges']=={}
