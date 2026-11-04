"""Current-site camera and calibration bridge. No robot connection or motion API."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import yaml

from .dual_arm.observation import save_observation
from .perception import CameraSpec, RGBDFrame, create_rgbd_camera, load_extrinsics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / 'config/site.zed_dual_xarm7_20261103.json'
OPTICAL_FRAME = 'IMAGE: x right, y down, z forward'


def rigid(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6)):
        raise ValueError('Expected a finite rigid transform')
    return matrix


def load_site(path=DEFAULT_PROFILE, root=ROOT):
    root = Path(root)
    profile = json.loads(Path(path).read_text())
    if (profile.get('profile_kind') != 'calibrated_observation_only'
            or profile.get('robot_connection_enabled') is not False
            or profile.get('common_frame') != 'right_arm_base'):
        raise ValueError('Expected observation-only profile in right_arm_base')
    manifest_path = root / profile['manifest']
    manifest = json.loads(manifest_path.read_text())
    for name, record in manifest['files'].items():
        if hashlib.sha256((manifest_path.parent / name).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Imported calibration changed: {name}')
    camera = profile['camera']
    if camera['backend'] != 'zed' or camera['mount_arm'] is not None:
        raise ValueError('This profile requires a fixed ZED camera')
    transforms = {}
    if set(profile['arms']) != {'left', 'right'}:
        raise ValueError('Both calibrated arms are required')
    for arm, config in profile['arms'].items():
        session_dir = root / config['calibration_session']
        session = yaml.safe_load((session_dir / 'session.yaml').read_text())
        if (session['robot_ip'] != config['robot_ip'] or session['robot_axis'] != config['robot_axis']
                or session['mode'] != 'thirdview' or session['cammount_link'] != 'link_base'
                or str(session['serial']) != camera['serial']
                or session['coordinate_system'] != OPTICAL_FRAME
                or session['translation_units'] != 'meters' or session['stream'] != 'left_rectified'
                or session['image_size'] != [camera['width'], camera['height']]):
            raise ValueError(f'{arm}: camera/robot calibration mismatch')
        data = yaml.safe_load((root / config['extrinsics_file']).read_text())
        if data.get('camera_mount', 'link_base') != 'link_base':
            raise ValueError('Wrist transforms require robot access; unsupported here')
        if (root / config['extrinsics_file']).resolve() != (session_dir / 'extrinsics.yaml').resolve():
            raise ValueError('Extrinsics must belong to the imported session')
        transforms[arm] = rigid(data['X_CammountCam'])
    if camera['extrinsics_file'] != profile['arms']['right']['extrinsics_file']:
        raise ValueError('Camera must use the right-base extrinsics')
    for arm in profile['arms']:
        common_from_base = rigid(profile['arms'][arm]['common_from_base_m'])
        if not np.allclose(common_from_base @ transforms[arm], transforms['right'], atol=1e-8):
            raise ValueError(f'{arm}: inconsistent common-frame transform')
    return profile


def make_frame(profile, rgb, depth, intrinsics, confidence, root=ROOT):
    camera = profile['camera']
    shape = (camera['height'], camera['width'])
    rgb, depth = np.asarray(rgb), np.array(depth, dtype=np.float32, copy=True)
    k, confidence = np.asarray(intrinsics, dtype=float), np.asarray(confidence)
    if rgb.shape != (*shape, 3) or depth.shape != shape or confidence.shape != shape:
        raise ValueError('RGB/depth/confidence dimensions differ from calibration')
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0):
        raise ValueError('Invalid rectified intrinsics')
    depth[(~np.isfinite(depth) | (depth < .15) | (depth > 2.) | ~np.isfinite(confidence)
           | (confidence > camera['confidence_threshold']))] = np.nan
    if not np.isfinite(depth).any():
        raise ValueError('Capture contains no valid measured depth')
    return RGBDFrame(camera['label'], camera['serial'], rgb, depth, k,
                     load_extrinsics(Path(root) / camera['extrinsics_file']))


def load_recording(profile, directory, root=ROOT):
    directory = Path(directory).resolve()
    metadata = yaml.safe_load((directory / 'metadata.yaml').read_text())
    camera = profile['camera']
    if (str(metadata['serial']) != camera['serial']
            or metadata['image_size'] != [camera['width'], camera['height']]
            or metadata['coordinate_system'] != OPTICAL_FRAME
            or metadata['depth_units'] != 'meters' or metadata.get('fill_mode') is not False
            or not np.allclose(metadata['dist'], 0)):
        raise ValueError('Recording camera, coordinate frame, units or rectification mismatch')
    with Image.open(directory / 'left_rectified.png') as image:
        rgb = np.asarray(image.convert('RGB')).copy()
    frame = make_frame(profile, rgb, np.load(directory / 'depth_m.npy', allow_pickle=False),
                       metadata['K'], np.load(directory / 'confidence.npy', allow_pickle=False), root)
    return frame, dict(source='recorded_rgbd', source_directory=str(directory),
        image_timestamp_ns=metadata['timestamp_ns'], depth_mode=metadata['depth_mode'])


def world_xyz_mm(frame):
    y, x = np.indices(frame.depth_m.shape)
    k, depth = frame.intrinsics, frame.depth_m
    xyz = np.stack(((x-k[0, 2])/k[0, 0]*depth, (y-k[1, 2])/k[1, 1]*depth, depth), axis=-1)
    matrix = frame.X_base_camera
    return ((xyz @ matrix[:3, :3].T + matrix[:3, 3]) * 1000.).astype(np.float32)


def base_point_mm(profile, point_common_mm, arm):
    """Convert a common-frame target to the chosen arm's base, without IK/motion."""
    point = np.asarray(point_common_mm, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Expected a finite 3D point in millimeters')
    base_from_common = np.linalg.inv(rigid(profile['arms'][arm]['common_from_base_m']))
    return (base_from_common @ np.r_[point / 1000., 1.])[:3] * 1000.


def convert_pose_mm_deg(profile, pose, source, destination):
    """Convert complete XYZ/RPY poses between left/right/common; no robot access."""
    from .dual_arm.geometry import matrix_pose, pose_matrix
    def common_from(frame):
        if frame == 'common':
            return np.eye(4)
        matrix = rigid(profile['arms'][frame]['common_from_base_m']).copy()
        matrix[:3, 3] *= 1000.
        return matrix
    return matrix_pose(np.linalg.inv(common_from(destination)) @ common_from(source) @ pose_matrix(pose))


def export_observation(profile, frame, output, provenance):
    transform_mm = frame.X_base_camera.copy()
    transform_mm[:3, 3] *= 1000.
    config = SimpleNamespace(raw={'calibration_id': profile['calibration_id'], 'camera': {}})
    extra = dict(provenance, common_frame=profile['common_frame'], camera_backend='zed',
        camera_serial=frame.serial, intrinsics=frame.intrinsics.tolist(),
        world_from_camera_mm=transform_mm.tolist(), execution_allowed=False,
        robot_connection_enabled=False, physical_alignment_validated=False,
        robot_snapshot_available=False)
    if 'image_timestamp_ns' in provenance:
        timestamp = provenance['image_timestamp_ns'] / 1e9
        extra.update(captured_unix_s=timestamp, captured_at=datetime.fromtimestamp(timestamp, timezone.utc).isoformat())
    observation = save_observation(output, frame.rgb, world_xyz_mm(frame), config, {}, extra=extra)
    np.save(Path(output) / 'depth_m.npy', frame.depth_m, allow_pickle=False)
    return observation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--recording', type=Path, help='Import a saved RobotCamCalib RGB-D folder, offline')
    mode.add_argument('--live-camera', action='store_true', help='Open only the ZED, never a robot')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    profile = load_site(args.profile)
    if not args.recording and not args.live_camera:
        print(json.dumps(dict(status='CALIBRATION_OK', calibration_id=profile['calibration_id'],
            common_frame=profile['common_frame'], robot_calls=0, camera_calls=0,
            robot_integration='deferred'), indent=2))
        return 0
    if args.output and args.output.exists():
        parser.error('output already exists')
    if args.recording:
        frame, provenance = load_recording(profile, args.recording)
    else:
        c = profile['camera']
        spec = CameraSpec(c['label'], c['serial'], ROOT / c['extrinsics_file'], backend='zed',
                          depth_mode=c['depth_mode'], confidence_threshold=c['confidence_threshold'])
        camera = create_rgbd_camera(spec, c['width'], c['height'], c['fps'])
        try:
            camera.start()
            for _ in range(30):
                rgb, depth = camera.read()
            frame = make_frame(profile, rgb, depth, camera.intrinsics, camera.confidence)
            provenance = dict(source='live_camera_only', image_timestamp_ns=camera.timestamp_ns,
                              depth_mode=c['depth_mode'])
        finally:
            camera.stop()
    output = args.output or ROOT / 'outputs' / ('zed_observation_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    observation = export_observation(profile, frame, output, provenance)
    print(json.dumps(dict(status='OBSERVATION_OK', output=str(observation.directory),
        valid_depth_fraction=float(np.isfinite(frame.depth_m).mean()), robot_calls=0,
        execution_allowed=False), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
