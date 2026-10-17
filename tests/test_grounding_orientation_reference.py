"""Recorded orientation evidence reaches grounding without copying seed targets."""
import copy
import json

import pytest

from cloth_agent.experiments import graph_policy_replay as pr
from cloth_agent.experiments.graph_policy import file_hash
from tests.test_graph_policy import observation
from tests.test_graph_policy_replay import policy, Model


@pytest.mark.parametrize('source_kind',['seed_id','growth_path'])
def test_reference_rgb_and_each_recorded_orientation_reach_grounding(tmp_path,source_kind):
    _,graph=policy(tmp_path);edge=graph['edges'][0]
    if source_kind=='growth_path':
        reference=observation(tmp_path/'reference')
        edge['seed_action']['source_observation']=str(reference.image)
        meta={'frame_id':'world_mm','world_from_camera_mm':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
        (reference.directory/'observation.json').write_text(json.dumps(meta))
        edge['seed_action']['exact_command']['operations'][0]['arguments']['right'][5]=-1.570796
    expected=copy.deepcopy(edge)
    current=observation(tmp_path/'current');model=Model()
    pr.selected_edge_grounding(model,graph,edge,current,tmp_path)
    _,payload,images=model.calls[0]
    assert images[0]==('current',current.image)
    assert images[1][0]=='action_reference' and images[1][1].is_file()
    ref=payload['reference_orientation']
    assert ref['reference_image_id']=='action_reference'
    assert [row['world_rpy_rad'] for row in ref['operation_orientations']]==[
        op['arguments']['right'][3:] for op in edge['seed_action']['exact_command']['operations']]
    if source_kind=='growth_path':
        assert ref['reference_geometry']['world_from_camera_mm']==meta['world_from_camera_mm']
    assert 'source' not in json.dumps(ref) and 'world_xyz' not in json.dumps(ref)
    assert edge==expected


def test_changed_growth_reference_is_rejected_before_model_call(tmp_path):
    _,graph=policy(tmp_path);edge=graph['edges'][0];reference=observation(tmp_path/'ref')
    edge['seed_action']['source_observation']=str(reference.image)
    (reference.directory/'observation.json').write_text(json.dumps({'sha256':{'rgb.png':file_hash(reference.image)}}))
    reference.image.write_bytes(b'changed')
    model=Model()
    with pytest.raises(ValueError,match='checksum'):
        pr.selected_edge_grounding(model,graph,edge,observation(tmp_path/'current'),tmp_path)
    assert not model.calls


def test_unavailable_reference_is_explicit_and_still_supplies_numeric_rpy(tmp_path):
    _,graph=policy(tmp_path);edge=graph['edges'][0]
    edge['seed_action']['source_observation']=str(tmp_path/'missing.png')
    model=Model();pr.selected_edge_grounding(model,graph,edge,observation(tmp_path/'current'),tmp_path)
    _,payload,images=model.calls[0]
    assert len(images)==1
    assert payload['reference_orientation']['reference_image_id'] is None
    assert payload['reference_orientation']['operation_orientations']
