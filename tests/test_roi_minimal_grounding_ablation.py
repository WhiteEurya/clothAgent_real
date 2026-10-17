import copy
import numpy as np
import pytest
import jsonschema
from cloth_agent.experiments import roi_minimal_grounding_ablation as r


def test_identity_and_translated_roi():
    b=[320,390,570,660]
    v=r.transform_roi(b,[[1,0,0],[0,1,0]],(1280,720))
    assert v['crop_xyxy']==b and v['crop_size']==[250,270]
    v=r.transform_roi(b,[[1,0,10.2],[0,1,-3.2]],(1280,720))
    assert v['crop_xyxy']==[330,386,581,657]


def test_rotation_encloses_all_corners_and_invalid_rejects():
    # Use a sufficiently large ROI for the predeclared 16px minimum.
    v=r.transform_roi([10,10,40,30],[[0,-1,60],[1,0,0]],(100,100))
    assert v['crop_xyxy']==[30,10,50,40]
    with pytest.raises(ValueError):r.transform_roi([0,0,30,30],[[1,0,-1],[0,1,0]],(100,100))


def test_uv_mapping_pixel_centers_no_off_by_one():
    b=[100,200,201,401]
    assert r.map_uv([0,0],b)==[100,200]
    assert r.map_uv([1,1],b)==[200,400]
    assert r.map_uv([.5,.5],b)==[150,300]
    assert r.map_uv(None,b) is None
    for bad in [[-0.1,0],[1.01,1],[float('nan'),0],[True,0]]:
        with pytest.raises(ValueError):r.map_uv(bad,b)


def test_only_normalized_coordinates_contract():
    v=dict(grasp_uv=[.2,.3],target_uv=[.5,.6],confidence=.9);r.validate(v)
    for bad in [{**v,'XYZ':[0,0,0]},{**v,'grasp_uv':[100,200]}]:
        with pytest.raises(jsonschema.ValidationError):r.validate(bad)
    r.validate(dict(grasp_uv=None,target_uv=None,confidence=0))


def test_crop_pair_only_and_protocol_hashes():
    if not (r.OUT/'manifest.json').exists():pytest.skip('not prepared')
    m=r.verify();assert len(m['queries'])==7
    assert m['prompt']==r.PROMPT
    for q in m['queries']:
        folder=r.OUT/'samples'/q['sample_id'];roi=r.read(folder/'roi.json')
        if roi['status']!='OK':continue
        from PIL import Image
        for name in ['grasp','target']:
            a=Image.open(folder/(name+'.png'));b=Image.open(folder/(name+'_2x.png'))
            assert b.size==(a.width*2,a.height*2)
        for request in folder.glob('*/request.json'):
            v=r.read(request);assert v['prompt']==r.PROMPT and v['cache_disabled']
            assert len(v['images'])==2
            assert all('rgb.png' not in im['path'] for im in v['images'])
