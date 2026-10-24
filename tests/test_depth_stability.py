import numpy as np
import pytest
from types import SimpleNamespace

from cloth_agent.depth_stability import depth_window_report
from cloth_agent.perception import CameraSpec, RealSenseRGBD


@pytest.mark.parametrize('unstable', [False, True])
@pytest.mark.parametrize('camera_b', [False, True])
def test_capture_waits_for_depth_or_rejects_observation(tmp_path, monkeypatch, unstable, camera_b):
    from cloth_agent.dual_arm import observation
    from cloth_agent.dual_arm.geometry import DualArmError
    class Camera:
        started=False
        intrinsics=np.eye(3)
        reads=0
        def start(self):self.started=True
        def stop(self):self.started=False
        def read(self):
            self.reads+=1
            depth=.7+(.002*self.reads if unstable else 0)
            return np.zeros((240,320,3),dtype=np.uint8),np.full((240,320),depth)
    camera=Camera()
    config=SimpleNamespace(arms={},limits={},raw={'calibration_id':'test', 'camera':{
        'mount_arm':None,'world_from_camera_mm':np.eye(4).tolist(),'serial':'test',
        'depth_exposure_us':4000., 'depth_stability':{
            'warmup_frames':30,'max_frames':30,'window_frames':5,
            'stable_windows':3,'max_range_mm':3.}}})
    if camera_b:
        from pathlib import Path
        from cloth_agent.codex_camera_b import with_arm7_camera
        from cloth_agent.dual_arm.config import DualConfig
        root = Path(__file__).resolve().parents[1]
        real_config = with_arm7_camera(DualConfig.load(root/'config/dual_arm.local.json', root=root, homing_only=True))
        config.raw['camera']['depth_stability'] = real_config.raw['camera']['depth_stability']
    monkeypatch.setattr(observation,'create_camera',lambda config:camera)
    if unstable:
        with pytest.raises(DualArmError,match='did not stabilize'):
            observation.capture(tmp_path/'obs',config,{},lambda:{})
        assert not (tmp_path/'obs').exists()
    else:
        obs=observation.capture(tmp_path/'obs',config,{},lambda:{})
        assert obs.meta['depth_stability']['stable']
        assert obs.meta['depth_capture_policy']['depth_exposure_us']==4000.
        assert camera.reads==(75 if camera_b else 50)
    assert not camera.started


def test_stable_depth_passes_but_local_drift_does_not():
    frames = np.full((5, 240, 320), .7)
    assert depth_window_report(frames, 3.)['stable']
    for i in range(5):
        frames[i, 65:110, 130:180] += i*.004
    assert not depth_window_report(frames, 3.)['stable']


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('base_m', [.7, 2.4])
@pytest.mark.parametrize('limit_mm', [2., 3.])
def test_depth_threshold_accepts_float_roundoff(dtype, base_m, limit_mm):
    frames = np.full((10, 240, 320), base_m, dtype=dtype)
    frames[5:] = base_m + limit_mm / 1000
    report = depth_window_report(frames, limit_mm)
    assert report['stable']
    assert report['max_range_mm'] == limit_mm
    assert report['worst_tile_range_mm'] == pytest.approx(limit_mm, abs=.001)
    # Preserve the measured value in evidence instead of rounding it away.
    if dtype == np.float32 and base_m == .7 and limit_mm == 3.:
        assert report['worst_tile_range_mm'] == 3.000020980834961


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('change_mm', [3.002, 3.01, 4.])
def test_depth_threshold_rejects_change_beyond_roundoff(dtype, change_mm):
    frames = np.full((10, 240, 320), .7, dtype=dtype)
    frames[5:] = .7 + change_mm / 1000
    assert not depth_window_report(frames, 3.)['stable']


def test_missing_cloth_depth_cannot_hide_in_stable_background():
    frames = np.full((5, 240, 320), .7)
    frames[:, 65:110, 130:180] = 0
    assert not depth_window_report(frames, 3.)['stable']


def test_fixed_depth_settings_restored_on_close_and_partial_failure(tmp_path):
    for fail in (False, True):
        options=SimpleNamespace(enable_auto_exposure='auto',exposure='exposure',gain='gain')
        values=dict(auto=1.,exposure=8500.,gain=16.)
        original=values.copy();writes=[]
        def set_option(key,value):
            writes.append((key,value))
            if fail and key=='exposure' and value==4000.:
                raise RuntimeError('injected setting failure')
            values[key]=value
        sensor=SimpleNamespace(supports=lambda key:True,
            get_option_range=lambda key:SimpleNamespace(min=0,max=100000),
            get_option=lambda key:values[key],set_option=set_option)
        camera=object.__new__(RealSenseRGBD)
        camera.rs=SimpleNamespace(option=options)
        camera.spec=CameraSpec('A','test',tmp_path,depth_exposure_us=4000.,depth_gain=16.)
        camera._depth_restore=None;camera.started=True
        stops=[];camera.pipeline=SimpleNamespace(stop=lambda:stops.append(True))
        try:
            if fail:
                with pytest.raises(RuntimeError):camera._configure_depth(sensor)
            else:
                camera._configure_depth(sensor)
                assert values==dict(auto=0.,exposure=4000.,gain=16.)
        finally:camera.stop()
        assert values==original
        assert writes[-1]==('auto',1.)
        assert stops==[True]
