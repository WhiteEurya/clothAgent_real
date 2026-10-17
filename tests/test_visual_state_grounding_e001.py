import copy
import pytest
from cloth_agent.experiments.visual_state_grounding_e001 import host,PRED_IDS

@pytest.fixture
def program():
    return dict(regions=[dict(id='upper',semantic_region='upper cloth')],predicates=[dict(id=k,region='upper',question='Visible?') for k in PRED_IDS],
        rules=[{'if':'left_inward_flap == YES && torso_open == YES && right_extended == YES && right_rounded_cuff == NO','result':'S1'},
               {'if':'left_inward_flap == YES && torso_open == YES && right_extended == NO && right_rounded_cuff == YES','result':'S2'}],fallback='UNKNOWN')

@pytest.fixture
def value():
    return dict(entities=dict(right_cuff=dict(visible=True,bbox=[10,20,30,40],anchor_px=[20,30],confidence=.95),
        right_sleeve_root=dict(anchor_px=[40,30],confidence=.9),
        right_upper_torso=dict(bbox=[50,20,80,50],target_anchor_px=[60,30],confidence=.95)),
        predicates=dict(zip(PRED_IDS,['YES','YES','NO','YES'])))

def test_s1_only_semantic_grounding(program,value):
    before=copy.deepcopy(value);result=host(value,program,(100,100))
    assert result['node']['decision']=='S1' and result['grounding']['grasp_anchor_px']==[20,30]
    assert result['grounding']['physical_execution'] is False and value==before


def test_s2_does_not_ground_even_with_visible_entities(program,value):
    value['predicates'].update(right_extended='NO',right_rounded_cuff='YES')
    result=host(value,program,(100,100))
    assert result['node']['decision']=='S2' and result['grounding'] is None

@pytest.mark.parametrize('failure',['invisible','low_confidence','outside_bbox','outside_image','missing_root','unknown_predicate'])
def test_unresolved_entities_or_state_block_semantic_grounding(program,value,failure):
    if failure=='invisible':value['entities']['right_cuff'].update(visible=False,bbox=None,anchor_px=None,confidence=0)
    if failure=='low_confidence':value['entities']['right_upper_torso']['confidence']=.4
    if failure=='outside_bbox':value['entities']['right_cuff']['anchor_px']=[35,30]
    if failure=='outside_image':value['entities']['right_upper_torso']['target_anchor_px']=[101,30]
    if failure=='missing_root':value['entities']['right_sleeve_root']['anchor_px']=None
    if failure=='unknown_predicate':value['predicates']['right_extended']='UNKNOWN'
    assert host(value,program,(100,100))['grounding'] is None
