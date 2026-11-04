"""SDK contract, recorded geometry, and no-robot integration regressions."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from cloth_agent import perception
from cloth_agent.dual_arm.observation import Observation, create_camera
from cloth_agent.site_observation import (
    DEFAULT_PROFILE, ROOT, base_point_mm, export_observation, load_recording,
    load_site, make_frame, world_xyz_mm,
    convert_pose_mm_deg,
)
from cloth_agent.zed_rgbd import ZEDRGBD


def test_sdk_import_keeps_loader_error_and_interpreter(monkeypatch):
    import builtins
    from cloth_agent.zed_rgbd import load_zed_sdk
    original = builtins.__import__
    def failing(name, *args, **kwargs):
        if name == 'pyzed.sl':
            raise ImportError('libsl_zed.so: cannot open shared object file')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', failing)
    with pytest.raises(perception.PerceptionError, match='libsl_zed.so') as error:
        load_zed_sdk()
    assert sys.executable in str(error.value)


@pytest.fixture
def fake_sdk(monkeypatch):
    status = NS(open=0, grab=0, retrieve=0, closed=0, serial=11483267)
    class Mat:
        def get_data(self):
            return self.data
    class Camera:
        def open(self, params):
            status.params = params
            return status.open
        def get_camera_information(self):
            return NS(serial_number=status.serial, camera_configuration=NS(
                resolution=NS(width=1280, height=720), calibration_parameters=NS(
                    left_cam=NS(fx=730., fy=731., cx=640., cy=360.))))
        def grab(self, runtime):
            status.runtime = runtime
            return status.grab
        def retrieve_image(self, mat, view):
            mat.data = np.full((720, 1280, 4), [10, 20, 30, 255], dtype=np.uint8)
            return status.retrieve
        def retrieve_measure(self, mat, measure):
            mat.data = np.full((720, 1280), .8 if measure == 'DEPTH' else 20., dtype=np.float32)
            if measure == 'DEPTH':
                mat.data[0, :3] = [np.inf, 0., 3.]
            else:
                mat.data[0, 3:5] = [51., np.nan]
            return status.retrieve
        def get_timestamp(self, ref):
            return NS(get_nanoseconds=lambda: 1793716524872723783)
        def close(self):
            status.closed += 1
    class Init(NS):
        def set_from_serial_number(self, serial):
            self.serial = serial
    sl = NS(Camera=Camera, Mat=Mat, InitParameters=Init, RuntimeParameters=NS,
        RESOLUTION=NS(HD720='HD720'), COORDINATE_SYSTEM=NS(IMAGE='IMAGE'), UNIT=NS(METER='METER'),
        DEPTH_MODE=NS(ULTRA='ULTRA'), ERROR_CODE=NS(SUCCESS=0), VIEW=NS(LEFT='LEFT'),
        MEASURE=NS(DEPTH='DEPTH', CONFIDENCE='CONFIDENCE'), TIME_REFERENCE=NS(IMAGE='IMAGE'))
    monkeypatch.setitem(sys.modules, 'pyzed', NS(sl=sl))
    monkeypatch.setitem(sys.modules, 'pyzed.sl', sl)
    return status


def spec():
    return perception.CameraSpec('A', '11483267', Path('.'), backend='zed')


def test_zed_rgb_and_metric_depth_contract(fake_sdk):
    camera = ZEDRGBD(spec(), 1280, 720, 30)
    camera.start()
    try:
        rgb, depth = camera.read()
        assert rgb[10, 10].tolist() == [30, 20, 10]
        assert depth[10, 10] == pytest.approx(.8)
        assert np.isnan(depth[0, :5]).all()
        assert camera.intrinsics[1, 1] == 731.
        assert fake_sdk.params.coordinate_units == 'METER'
        assert fake_sdk.params.coordinate_system == 'IMAGE'
        assert fake_sdk.runtime.enable_fill_mode is False
    finally:
        camera.stop()
    assert fake_sdk.closed == 1


@pytest.mark.parametrize('failure', ['open', 'serial', 'grab', 'retrieve'])
def test_sdk_failures_close_camera(fake_sdk, failure):
    setattr(fake_sdk, failure, 99)
    camera = ZEDRGBD(spec(), 1280, 720, 30)
    with pytest.raises(perception.PerceptionError):
        try:
            camera.start()
            camera.read()
        finally:
            camera.stop()
    assert fake_sdk.closed == 1


def test_config_dispatch_and_legacy_default(monkeypatch):
    profile = load_site()
    camera = create_camera(NS(raw=profile))
    assert isinstance(camera, ZEDRGBD)
    monkeypatch.setattr(perception, 'RealSenseRGBD', lambda *args: 'legacy')
    assert perception.create_rgbd_camera(perception.CameraSpec('A', 'old', Path('.')), 1, 1, 1) == 'legacy'
    config = perception.PerceptionConfig.load(ROOT, ROOT/'config/perception.zed_dual_xarm7_20261103.json')
    assert config.cameras[0].backend == 'zed'
    assert config.manual_base_y_offset_mm == 0


def test_capture_pipeline_uses_zed_and_releases(fake_sdk):
    config = perception.PerceptionConfig(cameras=(spec(),), width=1280, height=720,
        warmup_frames=0, temporal_median_frames=1)
    # Use an actual fixed-base extrinsic instead of the dummy path.
    from dataclasses import replace
    profile = load_site()
    config = replace(config, cameras=(replace(spec(), extrinsics_file=ROOT/profile['camera']['extrinsics_file']),))
    frame, = perception.capture_two_view_rgbd(config)
    assert frame.rgb[10, 10].tolist() == [30, 20, 10]
    assert fake_sdk.closed == 1


def test_common_frame_projection_units_and_execution_gate(tmp_path, monkeypatch):
    # These checks must stay independent of pyzed/xarm installation or connectivity.
    monkeypatch.setitem(sys.modules, 'xarm', None)
    monkeypatch.setitem(sys.modules, 'pyzed', None)
    profile = load_site()
    k = np.array([[730., 0., 640.], [0., 730., 360.], [0., 0., 1.]])
    frame = make_frame(profile, np.zeros((720,1280,3), np.uint8),
        np.full((720,1280), .8), k, np.zeros((720,1280)))
    xyz = world_xyz_mm(frame)
    expected = (frame.X_base_camera @ [0., 0., .8, 1.])[:3] * 1000.
    np.testing.assert_allclose(xyz[360,640], expected, atol=1e-4)
    left = yaml.safe_load((ROOT/profile['arms']['left']['extrinsics_file']).read_text())
    expected_left = (np.array(left['X_CammountCam']) @ [0., 0., .8, 1.])[:3] * 1000.
    np.testing.assert_allclose(base_point_mm(profile, expected, 'left'), expected_left, atol=1e-7)
    obs = export_observation(profile, frame, tmp_path/'obs',
        dict(source='recorded_rgbd', image_timestamp_ns=1793716524872723783))
    assert Observation.load(obs.directory).meta['joints_deg'] == {}
    assert obs.meta['captured_unix_s'] == pytest.approx(1793716524.8727238)
    with pytest.raises(ValueError, match='observation-only'):
        obs.validate_for(NS(raw={'calibration_id': profile['calibration_id']}), live=True)


def test_reject_inconsistent_common_frame(tmp_path):
    profile = json.loads(DEFAULT_PROFILE.read_text())
    profile['arms']['left']['common_from_base_m'][0][3] += .1
    path = tmp_path/'site.json'
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match='inconsistent'):
        load_site(path)


def test_full_pose_roundtrip_and_home():
    from cloth_agent.dual_arm.geometry import pose_matrix
    profile = load_site()
    pose = profile['arms']['left']['home']['pose_base_mm_deg']
    common = convert_pose_mm_deg(profile, pose, 'left', 'common')
    restored = convert_pose_mm_deg(profile, common, 'right', 'left')
    np.testing.assert_allclose(pose_matrix(restored), pose_matrix(pose), atol=1e-8)
    np.testing.assert_allclose(base_point_mm(profile, common[:3], 'left'), pose[:3], atol=1e-8)


@pytest.mark.parametrize('field,value', [('serial','wrong'), ('depth_units','millimeters'),
    ('coordinate_system','Y_UP'), ('fill_mode',True), ('image_size',[640,480])])
def test_recording_rejects_wrong_camera_or_geometry(tmp_path, field, value):
    metadata = dict(serial='11483267', image_size=[1280,720], coordinate_system='IMAGE: x right, y down, z forward',
        depth_units='meters', fill_mode=False, dist=[0.]*5)
    metadata[field] = value
    (tmp_path/'metadata.yaml').write_text(yaml.safe_dump(metadata))
    with pytest.raises(ValueError, match='Recording'):
        load_recording(load_site(), tmp_path)
