import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
from cloth_agent import geometry_report as g


def test_path_length_follows_surface_and_never_bridges_missing_depth():
    y,x=np.indices((100,100));xyz=np.stack([x,y,10*np.sin(x/10)],axis=-1).astype(float)
    mask=np.ones((100,100),dtype='uint8');path=np.array([(i,50) for i in range(10,91,5)])
    a=g.measure_path(path,xyz,mask)
    assert a['visible_path_mm']>a['chord_mm']
    xyz[45:56,40:51]=np.nan
    b=g.measure_path(path,xyz,mask)
    assert b['visible_path_mm'] is None and b['status']=='INCOMPLETE_DEPTH_OR_LAYER_JUMP'


def test_color_segmentation_depth_and_cropping():
    rgb=np.full((180,240,3),40,dtype='uint8');rgb[20:180,50:190]=[180,120,200]
    y,x=np.indices(rgb.shape[:2]);xyz=np.stack([x,y,np.zeros_like(x)],axis=-1).astype(float)
    report,mask,paths=g.analyze(rgb,xyz)
    assert report['touches_image_border']
    assert len(paths)==4 and report['depth_valid_fraction']==1
    assert any(p['visible_path_mm'] is None for p in report['contour_candidates'])
    rgb[165:]=40
    complete,_,_=g.analyze(rgb,xyz)
    assert all(p['visible_path_mm'] is not None for p in complete['contour_candidates'])
    xyz[:]=np.nan
    report,_,_=g.analyze(rgb,xyz)
    assert report['status']=='PARTIAL' and 'contour_candidates' not in report


def test_unknown_goal_no_reference_and_outputs_bound_to_rgb(tmp_path):
    rgb=np.full((180,240,3),40,dtype='uint8');rgb[20:160,50:190]=[180,120,200]
    image=tmp_path/'rgb.png';Image.fromarray(rgb).save(image)
    y,x=np.indices(rgb.shape[:2]);xyz=np.stack([x,y,np.zeros_like(x)],axis=-1).astype(float)
    before=SimpleNamespace(image=image,xyz=xyz,meta={'frame_id':'world_mm'})
    report=g.build(before,image,tmp_path)
    assert report['source_rgb_sha256']==g.sha(image)
    assert report['reference']['status']=='UNAVAILABLE'
    assert 'global_span_ratios_to_reference' not in report
    assert (tmp_path/'geometry_overlay.png').exists()
    assert json.loads((tmp_path/'geometry_report.json').read_text())['role']=='advisory_only'
    assert 'NOT identified' in g.prompt_report(report)
