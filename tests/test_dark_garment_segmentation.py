from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from cloth_agent.cloth_harness.cv_harness.segmentation import segment_dark_garment
from cloth_agent.experiments.unfold_skill_planner import segment_observation
from cloth_agent.contour_normal_grasp import NormalGraspError


def test_black_cloth_print_and_unknown_background():
    rgb=np.full((100,100,3),190,np.uint8)
    rgb[25:75,30:70]=20
    rgb[40:55,40:55]=[230,90,100]
    unknown=np.zeros((100,100),bool);unknown[:15]=True;rgb[:15]=0
    original=unknown.copy()
    mask,report=segment_dark_garment(rgb,unknown)
    assert mask[45,45] and mask[30,35]
    assert not mask[:15].any() and not mask[:, :20].any()
    assert np.array_equal(unknown,original) and not report['depth_filled']


def test_black_profile_uses_uncorrected_same_grid(tmp_path):
    rgb=np.full((100,100,3),190,np.uint8);rgb[25:75,30:70]=20
    Image.fromarray(rgb).save(tmp_path/'unmatched_rgb.png')
    Image.fromarray(np.full_like(rgb,200)).save(tmp_path/'planning_rgb.png')
    before=SimpleNamespace(image=tmp_path/'planning_rgb.png',segmentation_profile='black',unknown_mask=np.zeros((100,100),bool))
    mask,report=segment_observation(before)
    assert mask[50,50] and not mask[10,10]
    assert report['source_image'].endswith('unmatched_rgb.png')
    (tmp_path/'unmatched_rgb.png').unlink()
    with pytest.raises(NormalGraspError,match='uncorrected'):segment_observation(before)


def test_print_touching_outer_edge_is_not_a_background_notch():
    rgb=np.full((140,140,3),190,np.uint8)
    rgb[35:105,40:100]=20
    # Unlike an enclosed hole, this print touches the garment's outer edge.
    rgb[55:80,80:100]=[210,60,70]
    unknown=np.zeros((140,140),bool)
    mask,_=segment_dark_garment(rgb,unknown)
    assert mask[57:78,82:99].all()
    assert not mask[:,110:].any()


def test_rendered_mask_does_not_turn_shadow_into_colored_print(tmp_path):
    rgb=np.full((160,160,3),180,np.uint8)
    rgb[35:125,40:120]=20
    rgb[70:100,100:120]=[120,115,75]  # muted yellow print at the silhouette
    rgb[35:125,120:130]=95  # neutral cast shadow, not a foreground color
    Image.fromarray(rgb).save(tmp_path/'rendered_rgb.png')
    Image.fromarray(rgb).save(tmp_path/'planning_rgb.png')
    np.save(tmp_path/'appearance_valid.npy',np.ones((160,160),bool))
    unknown=np.zeros((160,160),bool);unknown[80,60]=True
    before=SimpleNamespace(image=tmp_path/'planning_rgb.png',segmentation_profile='black',unknown_mask=unknown)
    mask,report=segment_observation(before)
    assert mask[80,115] and not mask[80,125]
    assert report['source_image'].endswith('rendered_rgb.png')
    assert report['silhouette_cells_without_measured_depth']==1
    assert before.unknown_mask[80,60]
