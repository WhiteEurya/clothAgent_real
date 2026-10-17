import copy
from types import SimpleNamespace
import numpy as np
import pytest
from tests.test_graph_template_compiler import fixture
from cloth_agent.experiments import graph_template_compiler as tc,action_local_frame as al


def setup(tmp_path):
    g,t,o=fixture(tmp_path);edge=tc.policy_projection(g,t)['edges'][0];step=t['steps'][0]
    slots,profile,errors,lowering=tc.seed_slots(edge,step,o)
    record=dict(seed_grasp_px=slots[0]['pixel_xy'],seed_target_px=slots[-1]['pixel_xy'],actual_grasp_lowering=lowering,
                grasp_semantic_relation='grasp',target_semantic_relation='target')
    o.meta['support_height_patch']=dict(enabled=True,mode='support_plane_z',z_policy='raw_plane_to_reference_height',reference_height_mm=40.,
        plane_z_ax_by_c=[0,0,50],input_coordinate_policy='uncorrected_world_xyz_only',
        contact_reference=dict(semantics='empirical_open_gripper_contact_tcp_height',user_confirmed=True))
    template=al.compile_local(edge,step,o,record,contact_z=True)
    return edge,step,o,template


def test_visual_depth_residual_never_becomes_grasp_tcp_height(tmp_path):
    e,step,o,t=setup(tmp_path)
    value=dict(grasp_px=t['seed_grasp_px'],target_px=t['seed_target_px'],confidence=1.)
    old=al.expand(t,value,e,o,tmp_path/'seed')
    def sample(px,spread):
        v=np.asarray(o.sample(px,spread)).copy();v[2]+=25;return v
    noisy=SimpleNamespace(sample=sample,meta=o.meta,xyz=o.xyz)
    new=al.expand(t,value,e,noisy,tmp_path/'noisy')
    assert new['poses']==old['poses']
    assert new['height_semantics']['visual_anchor_z_mm'][0]==75
    assert new['height_semantics']['commanded_grasp_tcp_z_mm']==50
    for index,(row,op) in enumerate(zip(new['poses'],step['commanded_action']['operations'])):
        expected=op['arguments']['right'][2]+(.015 if index==0 else 0)
        assert row['world_m_rad'][2]==pytest.approx(expected)


@pytest.mark.parametrize('change',['missing','disabled','unconfirmed','changed_baseline','changed_plane'])
def test_contact_contract_mismatch_rejects(tmp_path,change):
    e,step,o,t=setup(tmp_path);meta=copy.deepcopy(o.meta);p=meta['support_height_patch']
    if change=='missing':meta.pop('support_height_patch')
    elif change=='disabled':p['enabled']=False
    elif change=='unconfirmed':p['contact_reference']['user_confirmed']=False
    elif change=='changed_baseline':p['reference_height_mm']+=5
    else:p['plane_z_ax_by_c'][2]+=5
    obs=SimpleNamespace(sample=o.sample,meta=meta,xyz=o.xyz)
    with pytest.raises(ValueError):al.expand(t,dict(grasp_px=t['seed_grasp_px'],target_px=t['seed_target_px'],confidence=1.),e,obs,tmp_path/'bad')
