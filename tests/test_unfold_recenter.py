import json
from types import SimpleNamespace
import numpy as np
import pytest
from cloth_agent.experiments import unfold_recenter as m
from cloth_agent.contour_normal_grasp import NormalGraspError


def test_world_center_translation_preserves_collar_grasp_and_segment(tmp_path,monkeypatch):
    (tmp_path/'view.json').write_text(json.dumps(dict(mode='RGB_FIRST_MULTI_CAMERA',mm_per_pixel=2.,center_world_xy_mm=[0,0])))
    mask=np.zeros((300,300),bool);mask[180:220,40:80]=True
    view=SimpleNamespace(directory=tmp_path,sample=lambda p,_:np.r_[(np.array(p)-149.5)*[2,-2],45])
    polygon=np.array([[-280,-280],[280,-280],[280,280],[-280,280]],float)
    monkeypatch.setattr(m,'table_polygon',lambda _:polygon)
    r=m.candidate(view,mask,[50,190])
    np.testing.assert_allclose(r['garment_center_world_xy_mm'],[-180,-100])
    np.testing.assert_allclose(r['table_center_world_xy_mm'],[0,0])
    np.testing.assert_allclose(r['requested_center_delta_mm'],[180,100])
    assert r['grasp_px']==[50,190]
    assert np.linalg.norm(r['planned_step_delta_mm'])==pytest.approx(140)
    assert r['segmentation_reason'] is not None
    assert 10<=r['measured_distance_mm']<=150
    mask[0,45]=True
    with pytest.raises(NormalGraspError,match='uncut'):m.candidate(view,mask,[50,190])


def test_polygon_center_independent_of_vertex_order():
    p=np.array([[0,0],[100,0],[100,50],[0,50]])
    np.testing.assert_allclose(m.polygon_center(p),[50,25])
    np.testing.assert_allclose(m.polygon_center(p[::-1]),[50,25])


def test_recenter_rejects_missing_or_off_garment_collar_point(tmp_path,monkeypatch):
    (tmp_path/'view.json').write_text(json.dumps(dict(mode='RGB_FIRST_MULTI_CAMERA',mm_per_pixel=2.,center_world_xy_mm=[0,0])))
    mask=np.zeros((300,300),bool);mask[180:220,40:80]=True
    view=SimpleNamespace(directory=tmp_path,sample=lambda *a:pytest.fail('invalid grasp must not sample depth'))
    monkeypatch.setattr(m,'table_polygon',lambda _:np.array([[-280,-280],[280,-280],[280,280],[-280,280]],float))
    for grasp in (None,[20,20],[float('nan'),190]):
        with pytest.raises(NormalGraspError,match='collar'):
            m.candidate(view,mask,grasp)
