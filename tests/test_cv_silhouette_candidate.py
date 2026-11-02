import cv2
import numpy as np
from cloth_agent.cloth_harness.candidates.silhouette_area import footprint


def test_enclosed_print_change_does_not_fake_outer_area_change():
    before=np.zeros((120,120),bool);before[20:100,20:100]=True
    after=before.copy();before[50:70,50:70]=False;after[40:80,40:80]=False
    assert before.sum()!=after.sum()
    a,ha=footprint(before);b,hb=footprint(after)
    assert np.array_equal(a,b) and ha.sum()==400 and hb.sum()==1600


def test_external_fold_gap_not_filled():
    mask=np.zeros((120,120),bool);mask[20:100,20:100]=True;mask[20:60,50:70]=False
    out,holes=footprint(mask)
    assert np.array_equal(out,mask) and not holes.any()


def test_real_enclosed_hole_is_footprint_not_material_area():
    mask=np.zeros((120,120),bool);mask[20:100,20:100]=True;mask[50:70,50:70]=False
    out,holes=footprint(mask)
    assert out.sum()>mask.sum() and holes.any() # explicitly not a material membership mask
    assert not mask[60,60] # input remains unchanged; cannot authorize a grasp there


def test_rigid_translation_keeps_footprint_area():
    a=np.zeros((120,120),np.uint8);a[20:80,20:90]=1;a[30:40,30:45]=0
    b=cv2.warpAffine(a,np.float32([[1,0,8],[0,1,7]]),(120,120))
    x,_=footprint(a);y,_=footprint(b)
    assert x.sum()==y.sum()
