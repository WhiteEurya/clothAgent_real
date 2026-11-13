"""Depth-scale calibration must generalize to a distinct plane without refitting."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts.evaluate_depth_scale import fit_depth_scale


def plane_points(translation, angles):
    t=np.eye(4);t[:3,:3]=Rotation.from_euler('xyz',angles,degrees=True).as_matrix();t[:3,3]=translation
    x,y=np.meshgrid(np.linspace(20,260,30),np.linspace(20,180,20))
    local=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    return local@t[:3,:3].T+t[:3,3],t


@pytest.mark.parametrize('scale',[.995,1.017])
def test_scale_fitted_on_a_recovers_new_slanted_plane_b(scale):
    a,ta=plane_points([10,-50,750],[20,10,0])
    b,tb=plane_points([-120,40,900],[10,-20,40])
    fitted=fit_depth_scale(a/scale,ta)
    assert fitted==pytest.approx(scale)
    assert np.max(np.abs((b/scale*fitted-tb[:3,3])@tb[:3,2]))<1e-9
    assert np.median(np.abs((b/scale-tb[:3,3])@tb[:3,2]))>2


def test_invalid_depth_not_accepted():
    a,t=plane_points([0,0,750],[0,0,0]);a[0]=np.nan
    with pytest.raises(ValueError):fit_depth_scale(a,t)
