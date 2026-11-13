import cv2
import numpy as np
import pytest
from cloth_agent.cloth_harness.cv_harness.segmentation import segment_light_garment, segment_auto_garment, external_contour_band


def scene():
    rgb=np.full((200,300,3),[180,150,30],np.uint8)
    rgb[40:160,60:180]=[210,210,210]
    rgb[180:187,20:280]=[225,225,225]  # separate narrow metal rail
    return rgb


def test_independent_segmentation_excludes_mat_and_separate_rail():
    mask,report=segment_light_garment(scene())
    assert mask[100,100] and not mask[20,20] and not mask[183,150]
    assert mask[40:160,60:180].mean()>.98
    assert not report['model_polygon_used'] and not report['depth_filled']


def test_unknown_cells_never_fabricate_source_depth_or_change_rgb():
    rgb=scene();unknown=np.zeros((200,300),bool);unknown[100,100]=True;unknown[80:100,110:130]=True
    rgb[unknown]=[24,30,39];original=rgb.copy();u=unknown.copy()
    mask,report=segment_light_garment(rgb,unknown)
    assert np.array_equal(rgb,original) and np.array_equal(unknown,u)
    assert mask[100,100]  # tiny projection hole may support silhouette only
    assert not mask[90,120]  # a large unseen region is not invented
    assert report['depth_filled'] is False


def test_no_garment_does_not_make_up_a_polygon():
    with pytest.raises(ValueError,match='no light garment'):
        segment_light_garment(np.full((200,300,3),[180,150,30],np.uint8))


def test_model_schema_accepts_only_local_points_not_silhouette():
    import jsonschema
    from cloth_agent.experiments.unfold_skill_planner import SELECT_SCHEMA
    value=dict(decision='ACTION',skill='garment-tangent-grasp',region_name='不可见下摆角',
        evidence='附近边界可见',boundary_point_px=[60,100],chord_endpoints_px=[],pull_distance_mm=40,grasp_px=None,target_px=None)
    jsonschema.validate(value,SELECT_SCHEMA)
    for key in ['visible_polygon_px','background_points_px']:
        with pytest.raises(jsonschema.ValidationError):jsonschema.validate(dict(value,**{key:[]}),SELECT_SCHEMA)


def test_auto_profile_prefers_measured_elevated_surface_over_color_candidates():
    h,w=160,220
    yy,xx=np.mgrid[:h,:w]
    xyz=np.stack([xx.astype(np.float32),yy.astype(np.float32),
                  (0.01*xx+0.02*yy).astype(np.float32)],axis=2)
    rgb=np.full((h,w,3),[180,150,30],np.uint8)
    garment=(slice(45,115),slice(70,155))
    xyz[garment+(2,)]+=25
    rgb[garment]=[210,210,210]
    mask,report=segment_auto_garment(rgb,xyz,np.zeros((h,w),bool))
    assert report['selected_profile']=='elevated'
    assert mask[80,100] and not mask[20,20]
    assert report['candidate_scores']['elevated']['score'] > report['candidate_scores']['light']['score']


def test_external_edge_band_excludes_garment_interior():
    mask=np.zeros((100,120),np.uint8);mask[20:80,30:90]=1
    edge=external_contour_band(mask,thickness=2)
    assert edge.sum()<mask.sum()/4
    assert not edge[50,60]
    assert edge[20,60]
