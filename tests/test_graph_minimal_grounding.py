import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import jsonschema
from cloth_agent.experiments import graph_minimal_grounding as mg
from cloth_agent.experiments import graph_policy_replay as pr
from tests.test_graph_policy_replay import policy,Model,candidate


def test_schema_only_visual_parameters_and_bounds():
    value={'grasp_px':[10,20],'target_px':[40,50],'confidence':.9};mg.validate(value,(100,100))
    with pytest.raises(jsonschema.ValidationError):mg.validate({**value,'move_tcp':[]})
    with pytest.raises(ValueError):mg.validate({**value,'target_px':[100,50]},(100,100))
    mg.validate({'grasp_px':None,'target_px':None,'confidence':0})


def test_host_profile_preserves_path_heights_and_rotations():
    slots=[dict(operation_index=i,pixel_xy=p,z_above_surface_mm=z,rpy_rad=[np.pi,0,0]) for i,(p,z) in enumerate([([10,20],0),([10,20],60),([30,15],120),([50,20],60),([50,20],0)])]
    profile=mg.relative_slots(slots);r=mg.expand_slots(profile,[100,100],[100,180],np.pi/2)
    assert [s['pixel_xy'] for s in r]==[[100,100],[100,100],[110,140],[100,180],[100,180]]
    assert [s['z_above_surface_mm'] for s in r]==[0,60,120,60,0]
    from scipy.spatial.transform import Rotation
    for s in r:np.testing.assert_allclose(Rotation.from_euler('xyz',s['rpy_rad']).as_matrix(),Rotation.from_euler('z',np.pi/2).as_matrix()@Rotation.from_euler('xyz',[np.pi,0,0]).as_matrix(),atol=1e-8)


def test_old_requests_still_use_original_full_payload(tmp_path):
    _,graph=policy(tmp_path);model=Model();edge=graph['edges'][0]
    obs=SimpleNamespace(image=tmp_path/'rgb.png',xyz=np.zeros((20,20,3)),meta={})
    result=pr.selected_edge_grounding(model,graph,edge,obs,tmp_path)
    assert result==candidate()
    assert model.calls[0][0]=='ground_edge'
    assert model.calls[0][1]==pr.grounding_payload(graph,edge,obs)


def test_minimal_routing_does_not_construct_full_payload(tmp_path,monkeypatch):
    _,graph=policy(tmp_path);edge=graph['edges'][0];expected=candidate()
    def forbidden(*a):raise AssertionError('Full context leaked')
    monkeypatch.setattr(pr,'grounding_payload',forbidden)
    class Stub:
        def ground(self,requests,selected,before,folder):assert selected is edge;return expected
    model=SimpleNamespace(minimal_grounder=Stub())
    assert pr.selected_edge_grounding(model,graph,edge,None,tmp_path)==expected


def test_sleeve_guard_unchanged_rejects_wrong_side():
    c=candidate();c['slots'][0]['pixel_xy']=[15,10]
    edge={'action_family':'left_sleeve_inward_fold'};obs=SimpleNamespace(xyz=np.zeros((20,20,3)))
    with pytest.raises(ValueError,match='opposite'):pr.validate_sleeve_side(c,edge,obs)


def test_bundle_hash_and_edge_identity_enforced(tmp_path):
    bundle=Path('reports/minimal_edge_grounding/host_templates.json')
    if not bundle.exists():pytest.skip('offline bundle not generated')
    grounder=mg.MinimalGrounder(bundle);graph=mg.read('runs/graph_policy_full02/policy_graph_v2.json')
    edge=next(e for e in graph['edges'] if e['edge_id']=='e001');grounder.template(edge)
    altered=copy.deepcopy(edge);altered['action_family']='another'
    with pytest.raises(ValueError,match='changed'):grounder.template(altered)
    with pytest.raises(ValueError,match='no compiled'):grounder.template({'edge_id':'e002'})
    t=tmp_path/'bundle.json';t.write_text(bundle.read_text()+' ');Path(str(t)+'.sha256.json').write_text(Path(str(bundle)+'.sha256.json').read_text())
    with pytest.raises(ValueError,match='checksum'):mg.MinimalGrounder(t)


def test_no_extra_model_calls_when_host_frame_fails(tmp_path,monkeypatch):
    bundle=Path('reports/minimal_edge_grounding/host_templates.json')
    if not bundle.exists():pytest.skip('offline bundle not generated')
    grounder=mg.MinimalGrounder(bundle);edge=next(e for e in mg.read('runs/graph_policy_full02/policy_graph_v2.json')['edges'] if e['edge_id']=='e001')
    calls=[]
    class Requests:
        def ask(self,kind,payload,images,validator):
            calls.append((kind,payload));v=dict(grasp_px=[400,530],target_px=[590,520],confidence=.9);validator(v);return v
    def reject(*a):raise ValueError('unresolved Host frame')
    monkeypatch.setattr(mg,'register_frame',reject)
    with pytest.raises(ValueError,match='unresolved Host frame'):grounder.ground(Requests(),edge,SimpleNamespace(image=tmp_path/'rgb.png'),tmp_path)
    assert calls==[('ground_edge_minimal',mg.SPEC_E001)]


def test_replay_minimal_retains_original_execution_validation_path(tmp_path):
    from tests.test_graph_policy import Runtime
    path,graph=policy(tmp_path);model=Model();seen=[]
    class Minimal:
        path=tmp_path/'host_templates.json'
        def template(self,edge):seen.append('host_template')
        def ground(self,requests,edge,before,folder):seen.append('minimal_ground');return candidate()
    model.minimal_grounder=Minimal();model.grounding_mode='minimal';runtime=Runtime()
    result=pr.replay(path,tmp_path/'run',runtime,model)
    assert seen==['host_template','minimal_ground']
    assert len(runtime.executions)==1
    assert all(call[0]!='ground_edge' for call in model.calls)
    assert result['transitions'][0]['reached_expected_target_node']


def test_ambiguous_registration_does_not_reuse_stale_frame(tmp_path):
    from PIL import Image
    a=tmp_path/'reference.png';b=tmp_path/'rgb.png'
    Image.new('RGB',(1280,720),'gray').save(a);Image.new('RGB',(1280,720),'gray').save(b)
    t=dict(reference_image=str(a),reference_sha256=mg.sha(a),camera_serial='x',image_size=[1280,720],registration_roi=[500,180,850,450])
    with pytest.raises(ValueError,match='missing descriptors'):mg.register_frame(t,SimpleNamespace(meta={'camera_serial':'x'},image=b))
