from types import SimpleNamespace
import numpy as np
import pytest
from cloth_agent.experiments.graph_two_point_grounding import template, SCHEMA
from cloth_agent.experiments.action_local_frame import expand, instantiate_local
from cloth_agent.experiments.graph_minimal_grounding import GroundingUncertain


def fixture():
    poses=[[.1,.2,.06,3.141592653589793,0,0], [.15,.2,.10,3.141592653589793,0,0], [.2,.2,.065,3.141592653589793,0,0]]
    edge=dict(edge_id='e',source_node='n',action_family='fold',action_template={},seed_action={'exact_command':{'operations':[dict(method='grasp' if i==0 else 'move_tcp',arguments={'right':p}) for i,p in enumerate(poses)]}})
    obs=SimpleNamespace(xyz=np.zeros((100,100,3)),meta={'camera_serial':'test'},sample=lambda p,s: np.array([100,200,60]) if p==[10,10] else np.array([200,200,65]))
    return edge,obs,poses


def test_same_anchors_reproduce_frozen_commands_without_lowering(tmp_path):
    edge,obs,poses=fixture()
    candidate=expand(template(edge,obs),dict(grasp_px=[10,10],target_px=[20,20],confidence=1),edge,obs,tmp_path)
    np.testing.assert_allclose([p['world_m_rad'] for p in candidate['poses']],poses,atol=1e-12)
    _,source,_=instantiate_local(candidate,edge,obs)
    assert source.count('motion.grasp')==1


def test_rotated_scaled_anchors_preserve_heights(tmp_path):
    edge,obs,_=fixture();compiled=template(edge,obs)
    obs.sample=lambda p,s: np.array([300,400,80]) if p==[10,10] else np.array([300,600,90])
    result=expand(compiled,dict(grasp_px=[10,10],target_px=[20,20],confidence=1),edge,obs,tmp_path)
    np.testing.assert_allclose([p['world_m_rad'][:3] for p in result['poses']],[[.3,.4,.08],[.3,.5,.12],[.3,.6,.085]])


def test_schema_forbids_generated_waypoints():
    import jsonschema
    assert set(SCHEMA['properties'])=={'grasp_px','target_px','confidence'}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(grasp_px=[1,2],target_px=[3,4],confidence=1,slots=[]),SCHEMA)


def test_degenerate_seed_rejected():
    edge,obs,_=fixture();edge['seed_action']['exact_command']['operations'][-1]['arguments']['right'][:2]=[.1,.2]
    with pytest.raises(ValueError,match='separation'):template(edge,obs)
