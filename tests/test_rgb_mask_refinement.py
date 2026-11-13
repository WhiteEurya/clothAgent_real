import numpy as np
from cloth_agent.cloth_harness.cv_harness.segmentation import refine_mask_with_rgb


def test_rgb_removes_thin_table_branch_and_preserves_dark_print():
    rgb=np.full((100,100,3),[150,145,115],np.uint8)
    rgb[25:75,30:70]=[220,230,240];rgb[40:60,40:60]=[15,20,25]
    coarse=np.zeros((100,100),np.uint8);coarse[25:75,30:70]=1
    coarse[35:39,10:32]=1
    original=coarse.copy();rgb_copy=rgb.copy()
    mask,report=refine_mask_with_rgb(rgb,coarse)
    assert not mask[35:39,10:29].any()
    assert mask[25:75,30:70].all()
    assert np.array_equal(coarse,original) and np.array_equal(rgb,rgb_copy)
    assert report['depth_modified'] is False


def test_unknown_pixels_are_not_claimed_as_foreground():
    rgb=np.full((100,100,3),[150,145,115],np.uint8);rgb[20:80,20:80]=240
    coarse=np.zeros((100,100),np.uint8);coarse[20:80,20:80]=1
    unknown=np.zeros((100,100),bool);unknown[45:50,45:50]=True
    mask,_=refine_mask_with_rgb(rgb,coarse,unknown)
    assert not mask[unknown].any()
