from types import SimpleNamespace

import numpy as np

from cloth_agent.depth_level import rotate_cloud
from cloth_agent.dual_arm.observation import save_observation


def test_rotation_levels_known_plane_preserves_anchor_and_relief():
    pivot=np.array([700.,220.,33.])
    x=np.array([400.,550.,700.])
    points=np.c_[x,np.full(3,220.),33.+(x-700)*np.tan(np.radians(1))]
    leveled,rotation=rotate_cloud(points,pivot,1.)
    np.testing.assert_allclose(leveled[:,2],33.,atol=1e-10)
    np.testing.assert_allclose(leveled[-1],pivot)
    np.testing.assert_allclose(np.linalg.norm(leveled[0]-leveled[1]),np.linalg.norm(points[0]-points[1]))
    elevated=points.copy();elevated[:,2]+=10
    upper,_=rotate_cloud(elevated,pivot,1.)
    np.testing.assert_allclose(upper[:,2]-leveled[:,2],10*np.cos(np.radians(1)))


def test_observation_saves_original_and_consistent_corrected_transform(tmp_path):
    correction={'mode':'world_y_rotation','pivot_world_mm':[700,220,33],'angle_deg':1.}
    config=SimpleNamespace(raw={'calibration_id':'corrected-test','camera':{'world_tilt_correction':correction}})
    xyz=np.array([[[400.,220.,28.],[700.,220.,33.]],[[float('nan')]*3,[600.,200.,30.]]])
    obs=save_observation(tmp_path/'obs',np.zeros((2,2,3),dtype=np.uint8),xyz,config,{},
                         synthetic=True,extra={'world_from_camera_mm':np.eye(4).tolist()})
    expected,_=rotate_cloud(xyz,correction['pivot_world_mm'],1.)
    np.testing.assert_allclose(obs.xyz,expected,rtol=1e-6,equal_nan=True)
    np.testing.assert_allclose(np.load(obs.directory/'uncorrected_world_xyz_mm.npy'),xyz,equal_nan=True)
    matrix=np.array(obs.meta['world_from_camera_mm'])
    np.testing.assert_allclose(xyz[0]@matrix[:3,:3].T+matrix[:3,3],obs.xyz[0],rtol=1e-6)
    assert np.isnan(obs.xyz[1,0]).all()
    assert 'uncorrected_world_xyz_mm.npy' in obs.meta['sha256']


def test_no_correction_keeps_existing_observation_behavior(tmp_path):
    config=SimpleNamespace(raw={'calibration_id':'original'})
    xyz=np.ones((2,2,3))
    obs=save_observation(tmp_path/'obs',np.zeros((2,2,3),dtype=np.uint8),xyz,config,{},synthetic=True)
    np.testing.assert_array_equal(obs.xyz,xyz)
    assert 'world_tilt_correction' not in obs.meta
