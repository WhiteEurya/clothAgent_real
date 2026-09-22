#!/usr/bin/env python3
"""Diagnose RGB/depth/wrist geometry using the RobotCamCalib 4x4 AprilTag board.

Live acquisition is read-only: move with the pendant, hold still, press Enter.
Re-analyze a saved session with --session. Never changes calibration or TCP.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import sys
import time

import cv2
import numpy as np
from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.apriltag_diagnostic import A4_SCALE, analyze_frame, draw_detection, load_board, save_plots, summarize
from cloth_agent.config import RobotConfig
from cloth_agent.perception import PerceptionConfig, RealSenseRGBD
from cloth_agent.robot_api import _validated_live_tcp_offset
from cloth_agent.run_storage import auxiliary_dir
from scripts.manual_pixel_move import write_json


class WristGeometry:
    """Same URDF composition as perception.load_extrinsics, with saved joints."""

    def __init__(self, spec, robot_ip):
        self.source = yaml.safe_load(spec.extrinsics_file.read_text())
        self.hand_eye = np.asarray(self.source['X_CammountCam'], float)
        if (self.hand_eye.shape != (4, 4) or not np.isfinite(self.hand_eye).all()
                or not np.allclose(self.hand_eye[3], [0, 0, 0, 1])
                or not np.allclose(self.hand_eye[:3, :3].T @ self.hand_eye[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(self.hand_eye[:3, :3]), 1, atol=1e-4)):
            raise ValueError('Invalid camera hand-eye transform')
        if self.source.get('camera_serial', spec.serial) != spec.serial:
            raise ValueError('Extrinsics camera serial differs from selected camera')
        self.mount = self.source.get('camera_mount', 'link_base')
        self.robot = None
        self.metadata = {'extrinsics': self.source,
                         'extrinsics_sha256': hashlib.sha256(spec.extrinsics_file.read_bytes()).hexdigest()}
        if self.mount != 'link_base':
            if self.source['robot_ip'] != robot_ip:
                raise ValueError('Extrinsics robot IP differs from robot config')
            from yourdfpy import URDF
            path = spec.extrinsics_file.parent / self.source['robot_urdf']
            self.robot = URDF.load(path, load_meshes=False, load_collision_meshes=False)
            self.metadata.update(urdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 joint_names=list(self.robot.actuated_joint_names))

    def transform(self, angles_rad):
        if self.robot is None:
            return self.hand_eye.copy()
        names = self.robot.actuated_joint_names
        if len(angles_rad) < len(names) or not np.isfinite(angles_rad[:len(names)]).all():
            raise ValueError('Invalid sampled joints for camera FK')
        self.robot.update_cfg(dict(zip(names, angles_rad[:len(names)])))
        return self.robot.get_transform(self.mount, 'link_base') @ self.hand_eye


def robot_sample(arm, config):
    started = time.monotonic()
    if not arm.connected:
        raise RuntimeError('Robot disconnected')
    qcode, joints = arm.get_servo_angle(is_radian=True)
    pcode, pose = arm.get_position(is_radian=False)
    if qcode != 0 or pcode != 0:
        raise RuntimeError(f'Failed robot feedback: joints={qcode}, pose={pcode}')
    if (len(joints) < 6 or len(pose) != 6 or not np.isfinite(joints).all()
            or not np.isfinite(pose).all()):
        raise ValueError('Invalid robot feedback')
    config.validate_live_tcp_offset(arm.tcp_offset)
    return {'timestamp': datetime.now(timezone.utc).isoformat(),
            'read_duration_s': time.monotonic()-started, 'joints_rad': list(joints),
            'tcp_pose_mm_deg': list(pose), 'tcp_offset_mm_deg': list(arm.tcp_offset)}


def stationary(before, after):
    q1, q2 = np.asarray(before['joints_rad']), np.asarray(after['joints_rad'])
    if q1.shape != q2.shape:
        raise ValueError('Joint feedback sizes changed')
    joint_delta = float(np.max(np.abs(np.degrees((q2-q1+np.pi) % (2*np.pi)-np.pi))))
    p1, p2 = np.asarray(before['tcp_pose_mm_deg']), np.asarray(after['tcp_pose_mm_deg'])
    delta_mm = float(np.linalg.norm(p2[:3]-p1[:3]))
    delta_deg = float(np.max(np.abs((p2[3:]-p1[3:]+180) % 360-180)))
    if (joint_delta > .2 or delta_mm > 1 or delta_deg > .5
            or max(before['read_duration_s'], after['read_duration_s']) > .5):
        raise ValueError(f'Unstable/slow capture: joints {joint_delta:.3f} deg, TCP {delta_mm:.3f} mm/{delta_deg:.3f} deg')


def capture_one(camera, arm, config, geometry, directory, sample_id, group):
    # Flush queued frames while recording a stationary bracket. This is not
    # hardware-triggered synchronization, and movement out-and-back is undetectable.
    before = robot_sample(arm, config)
    for _ in range(4):
        rgb, depth = camera.read()
    after = robot_sample(arm, config)
    directory.mkdir()
    Image.fromarray(rgb).save(directory / 'rgb.png')
    np.save(directory / 'depth_m.npy', depth)
    record = {'sample_id': sample_id, 'pose_group': group,
              'before': before, 'after': after, 'status': 'CAPTURED',
              'intrinsics': camera.intrinsics.tolist(), 'camera_label': camera.spec.label}
    try:
        stationary(before, after)
        record['X_base_camera'] = geometry.transform(before['joints_rad']).tolist()
    except ValueError as exc:
        record.update(status='REJECTED_MOTION', error=str(exc))
    write_json(directory / 'capture.json', record)
    if record['status'] == 'CAPTURED':
        write_json(directory / 'result.json', {'views': [{
            'label': camera.spec.label, 'serial': camera.spec.serial, 'image': 'rgb.png',
            'depth_m': 'depth_m.npy', 'intrinsics': record['intrinsics'],
            'X_base_camera': record['X_base_camera'], 'base_z_offset_mm': 0.0}]})
    return record, rgb, depth


def analyze_capture(record, rgb, depth, board, detector, directory):
    result = {key: record[key] for key in ('sample_id', 'pose_group', 'status')}
    if record['status'] == 'REJECTED_MOTION':
        result['error'] = record['error']
    else:
        try:
            result.update(analyze_frame(rgb, depth, np.asarray(record['intrinsics']),
                                        np.asarray(record['X_base_camera']), board, detector))
        except (ValueError, RuntimeError, cv2.error) as exc:
            result.update(status='DETECTION_FAILED', error=str(exc))
    write_json(directory / 'analysis.json', result)
    Image.fromarray(draw_detection(rgb, result)).save(directory / 'detections.png')
    print(f"Sample {record['sample_id']}: {result['status']} "
          f"fit={result.get('reprojection_px', {}).get('rms', '-')} px "
          f"depth residual={result.get('depth_minus_pnp_mm', {}).get('median', '-')} mm "
          f"{result.get('error', '')}", flush=True)
    return result


def save_report(directory, records, board):
    report = summarize(records)
    report['board'] = {k: v for k, v in board.items() if k != 'corners'}
    report['limitations'] = [
        'Board must remain fixed and flat. Print scaling and geometry affect PnP metric accuracy.',
        'Pinhole model matches the pipeline; distortion is not corrected.',
        'Fit and held-out residuals do not independently certify intrinsics.',
        'Depth-vs-PnP disagreement does not uniquely separate print scale, depth and RGB intrinsics.',
        'Joint/TCP reads bracket frames but are not hardware-synchronized.',
        'Hand-eye, FK, board motion and RGB pose error can all cause cross-view drift.',
        'Absolute accuracy and physical jaw/TCP offset remain unverified.',
    ]
    write_json(directory / 'report.json', report)
    save_plots(records, report, directory / 'diagnostic.png')
    print(f"Report: {directory / 'report.json'} | {report['status']} | "
          f"max board drift={report.get('max_pairwise_board_drift_mm', '-')} mm", flush=True)


def wait_for_capture(camera, window):
    """Pump the preview and terminal on one thread, sharing the capture pipeline."""
    print('Keep board FIXED. Move wrist manually, hold still. '
          'Enter/Space in preview or Enter in terminal=capture; q/Esc=finish.', flush=True)
    while True:
        rgb, _ = camera.read()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, 'LIVE | Hold still: Enter/Space capture | Q/Esc finish',
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xff
        if key in (27, ord('q'), ord('Q')):
            return 'q'
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            return 'q'
        if key in (10, 13, 32):
            return ''
        # Only poll interactive terminals: redirected stdin at EOF should not
        # immediately close an otherwise usable GUI preview.
        if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if not line:
                return 'q'
            command = line.strip().lower()
            if command in ('', 'q'):
                return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board', type=Path, default=PROJECT_ROOT / 'config/apriltag_board_4x4_tag48mm.yaml')
    parser.add_argument('--board-scale', type=float, default=None, help='Default: A4 shrink 0.70710678; offline inherits saved scale')
    parser.add_argument('--robot-config', type=Path, default=PROJECT_ROOT / 'config/robot.example.json')
    parser.add_argument('--perception-config', type=Path, default=PROJECT_ROOT / 'config/perception.free_exploration.json')
    parser.add_argument('--camera', default='A')
    parser.add_argument('--session', type=Path, help='Re-analyze saved session; no hardware connections')
    parser.add_argument('--output-dir', type=Path, help='Parent for a new timestamped diagnostic directory')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--no-preview', action='store_true',
                        help='Disable live RGB window (terminal-only/headless capture)')
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 20:
        parser.error('--repeats must be 1..20')
    board_path = args.session / 'board.yaml' if args.session else args.board
    if args.session:
        old_manifest = json.loads((args.session / 'session.json').read_text())
        scale = args.board_scale if args.board_scale is not None else old_manifest['board']['scale']
    else:
        scale = args.board_scale if args.board_scale is not None else A4_SCALE
    board = load_board(board_path, scale)
    from pupil_apriltags import Detector
    # Import plotting before opening hardware; missing dependencies fail early.
    import matplotlib
    matplotlib.use('Agg')
    detector = Detector(families=board['family'], quad_decimate=1.0, nthreads=2)
    parent = args.output_dir or (args.session / 'reanalyzed' if args.session else auxiliary_dir(PROJECT_ROOT, 'apriltag_diagnostics'))
    directory = parent / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'board.yaml').write_bytes(board_path.read_bytes())
    manifest = {'status': 'STARTING', 'mode': 'OFFLINE' if args.session else 'READ_ONLY_CAPTURE',
                'source_session': str(args.session) if args.session else None,
                'board': {k: v for k, v in board.items() if k != 'corners'}}
    write_json(directory / 'session.json', manifest)
    print(f"Board {board['family']}, scale={scale:.9f}, black edge={board['tag_size_mm']:.3f} mm\nOutput: {directory}", flush=True)
    records = []
    arm = camera = None
    window = None
    try:
        if args.session:
            captures = sorted(args.session.glob('sample_*/capture.json'))
            if not captures:
                raise ValueError('Session has no captured frames')
            for capture_path in captures:
                record = json.loads(capture_path.read_text())
                rgb = np.asarray(Image.open(capture_path.parent / 'rgb.png').convert('RGB'))
                depth = np.load(capture_path.parent / 'depth_m.npy', allow_pickle=False)
                target = directory / capture_path.parent.name
                target.mkdir()
                records.append(analyze_capture(record, rgb, depth, board, detector, target))
        else:
            if not args.no_preview:
                if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
                    raise RuntimeError('Live preview needs a graphical desktop. Use --no-preview for terminal-only capture.')
                window = 'AprilTag camera preview'
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window, 960, 540)
            config = RobotConfig.load(PROJECT_ROOT, args.robot_config)
            perception = PerceptionConfig.load(PROJECT_ROOT, args.perception_config)
            spec = next((s for s in perception.cameras if s.label == args.camera.upper()), None)
            if spec is None:
                raise ValueError('Camera not configured')
            geometry = WristGeometry(spec, config.robot_ip)
            manifest.update(geometry=geometry.metadata, robot_config=asdict(config), camera=spec.label,
                            camera_serial=spec.serial, repeats=args.repeats)
            write_json(directory / 'session.json', manifest)
            from xarm.wrapper import XArmAPI
            arm = XArmAPI(config.robot_ip, is_radian=False)
            _validated_live_tcp_offset(arm, config)
            camera = RealSenseRGBD(spec, perception.width, perception.height, perception.fps)
            camera.start()
            profile = camera.pipeline.get_active_profile().get_stream(camera.rs.stream.color).as_video_stream_profile()
            intr = profile.get_intrinsics()
            manifest['runtime_intrinsics'] = {'width': intr.width, 'height': intr.height,
                                              'K': camera.intrinsics.tolist(),
                                              'distortion_model': str(intr.model),
                                              'distortion_coefficients': list(intr.coeffs),
                                              'distortion_applied': False}
            write_json(directory / 'session.json', manifest)
            for _ in range(perception.warmup_frames):
                camera.read()
            group = 0
            while True:
                try:
                    command = (wait_for_capture(camera, window) if window else
                               input('Keep board FIXED. Move wrist manually, hold still. Enter=capture, q=finish: ').strip().lower())
                except EOFError:
                    break
                if command == 'q':
                    break
                if command:
                    continue
                if window:
                    # Make the temporary pause explicit while capture/analysis
                    # owns this same pipeline; no second camera connection.
                    rgb, _ = camera.read()
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.putText(frame, 'CAPTURING / ANALYZING - hold still', (12, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
                    cv2.imshow(window, frame)
                    cv2.waitKey(1)
                group += 1
                for _ in range(args.repeats):
                    sample_id = len(records)+1
                    target = directory / f'sample_{sample_id:03d}'
                    record, rgb, depth = capture_one(camera, arm, config, geometry, target, sample_id, group)
                    records.append(analyze_capture(record, rgb, depth, board, detector, target))
                save_report(directory, records, board)
        manifest['status'] = 'COMPLETE'
    except KeyboardInterrupt:
        manifest['status'] = 'INTERRUPTED'
    except Exception as exc:
        manifest.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            try:
                if window is not None:
                    cv2.destroyAllWindows()
            finally:
                if camera is not None:
                    camera.stop()
        finally:
            if arm is not None:
                arm.disconnect()
            write_json(directory / 'session.json', manifest)
            save_report(directory, records, board)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
