import copy,json
from pathlib import Path
import pytest
from cloth_agent.experiments.unified_visual_representation_e001 import features,derive,host,DEFAULT,SCHEMA,PIDS

def sample():
    return dict(garment=dict(collar_anchor=[100,20],hem_anchor=[100,180],confidence=.95),
      left_sleeve=dict(bbox=[40,35,90,80],polygon=[[40,35],[90,35],[90,80],[40,80]],distal_anchor=[80,60],root_anchor=[50,45],confidence=.95),
      right_sleeve=dict(bbox=[140,35,200,80],polygon=[[140,35],[200,35],[200,80],[140,80]],distal_anchor=[195,55],root_anchor=[150,45],confidence=.95),
      right_cuff=dict(bbox=[180,40,200,65],grasp_anchor=[190,55],opening_visible='NO',confidence=.95),
      torso=dict(bbox=[50,40,150,181],left_boundary=[[50,40],[50,110],[50,180]],right_boundary=[[150,40],[150,110],[150,180]],target_anchor=[125,65],print_visible_fraction=.95,surface_ridges=[],confidence=.95))

def program():return json.loads(Path('reports/visual_program_S1_S2/visual_program.json').read_text())

def test_geometry_and_grounding_share_anchors():
    v=sample();h=host(v,DEFAULT,program(),(240,220))
    assert h['predicates']==dict(zip(PIDS,['YES','YES','NO','YES']))
    assert h['node']['decision']=='S1'
    assert h['grounding']['grasp_anchor_px']==v['right_cuff']['grasp_anchor']
    assert h['grounding']['target_anchor_px']==v['torso']['target_anchor']
    assert not h['physical_execution']

def test_missing_orientation_is_unknown_and_blocks_grounding():
    v=sample();v['garment']['collar_anchor']=None
    h=host(v,DEFAULT,program(),(240,220));assert set(h['predicates'].values())=={'UNKNOWN'}
    assert h['grounding'] is None

def test_aperture_unknown_and_low_confidence_propagate():
    v=sample();v['right_cuff']['opening_visible']='UNKNOWN';v['left_sleeve']['confidence']=.1
    h=host(v,DEFAULT,program(),(240,220))
    assert h['predicates']['right_rounded_cuff']=='UNKNOWN'
    assert h['predicates']['left_inward_flap']=='UNKNOWN'
    assert h['node']['decision']=='UNKNOWN'

def test_long_interior_ridge_rejects_torso_open():
    v=sample();v['torso']['surface_ridges']=[dict(start=[90,50],end=[90,170],confidence=.95)]
    assert host(v,DEFAULT,program(),(240,220))['predicates']['torso_open']=='NO'

def test_schema_forbids_direct_predicates():
    import jsonschema
    v=sample();v['right_extended']='YES'
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(v,SCHEMA)

def test_s2_has_no_grounding():
    v=sample();v['right_sleeve']['distal_anchor']=[155,55];v['right_cuff'].update(bbox=[145,40,165,65],grasp_anchor=[155,55],opening_visible='YES')
    h=host(v,DEFAULT,program(),(240,220));assert h['node']['decision']=='S2';assert h['grounding'] is None

def test_invalid_anchor_never_yields_grounding():
    v=sample();v['right_cuff']['grasp_anchor']=[239,219]
    assert host(v,DEFAULT,program(),(240,220))['grounding'] is None

def test_flat_geometry_can_disprove_round_cuff_with_unknown_opening():
    v=sample();v['right_cuff'].update(bbox=[170,50,210,60],opening_visible='UNKNOWN')
    assert host(v,DEFAULT,program(),(240,220))['predicates']['right_rounded_cuff']=='NO'

def test_garment_frame_survives_half_turn():
    v=sample();rot=copy.deepcopy(v);width,height=240,220
    def point(p):return [width-1-p[0],height-1-p[1]]
    for entity in rot.values():
        for k,val in list(entity.items()):
            if 'anchor' in k:entity[k]=point(val)
            elif k=='bbox':
                x0,y0,x1,y1=val;entity[k]=[width-1-x1,height-1-y1,width-1-x0,height-1-y0]
            elif k in ['polygon','left_boundary','right_boundary']:entity[k]=[point(p) for p in val]
    assert host(v,DEFAULT,program(),(width,height))['predicates']==host(rot,DEFAULT,program(),(width,height))['predicates']
