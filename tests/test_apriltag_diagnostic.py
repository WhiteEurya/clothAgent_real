"""Run with cali Python -m unittest tests.test_apriltag_diagnostic (no pytest needed)."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from cloth_agent.apriltag_diagnostic import (
    A4_SCALE, analyze_frame, load_board, project, summarize,
)
from scripts import diagnose_apriltag_mapping as script


BOARD = script.PROJECT_ROOT / 'config/apriltag_board_4x4_tag48mm.yaml'


def synthetic_scene(board, pose=None, depth_bias=0):
    k = np.array([[700., 0, 320], [0, 710, 240], [0, 0, 1]])
    if pose is None:
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(np.array([.18, -.25, .08]))[0]
        pose[:3, 3] = [.02, -.01, .65]
    tags = [SimpleNamespace(tag_id=i, corners=project(pose, corners, k), hamming=0, decision_margin=80)
            for i, corners in board['corners'].items()]
    yy, xx = np.indices((480, 640))
    rays = np.stack(((xx-320)/700, (yy-240)/710, np.ones_like(xx)), axis=-1)
    normal = pose[:3, 2]
    depth = (normal @ pose[:3, 3]) / (rays @ normal) + depth_bias
    rgb = np.full((480, 640, 3), 180, np.uint8)
    detector = SimpleNamespace(detect=lambda *args, **kwargs: tags)
    return rgb, depth, k, detector, pose


class AprilTagDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.board = load_board(BOARD)

    def test_matches_original_yaml_geometry_and_a4_scale(self):
        self.assertEqual(self.board['family'], 'tag36h11')
        self.assertEqual(set(self.board['corners']), set(range(16)))
        self.assertAlmostEqual(self.board['tag_size_mm'], 48*A4_SCALE)
        edge = np.linalg.norm(self.board['corners'][0][1]-self.board['corners'][0][0])
        self.assertAlmostEqual(edge, .048*A4_SCALE)
        for invalid in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                load_board(BOARD, invalid)

    def test_known_board_pose_rgb_and_depth_recovered(self):
        rgb, depth, k, detector, pose = synthetic_scene(self.board)
        result = analyze_frame(rgb, depth, k, np.eye(4), self.board, detector)
        self.assertEqual(result['status'], 'USABLE')
        np.testing.assert_allclose(result['X_camera_board'], pose, atol=1e-6)
        self.assertLess(result['held_out_reprojection_px']['rms'], 1e-4)
        self.assertLess(result['depth_minus_pnp_mm']['max_abs'], 1e-4)

    def test_print_scale_and_depth_bias_are_visible(self):
        rgb, depth, k, detector, pose = synthetic_scene(self.board, depth_bias=.012)
        result = analyze_frame(rgb, depth, k, np.eye(4), self.board, detector)
        self.assertAlmostEqual(result['depth_minus_pnp_mm']['median'], 12, places=4)
        rgb, depth, k, detector, pose = synthetic_scene(self.board)
        wrong_board = load_board(BOARD, 1.0)
        wrong = analyze_frame(rgb, depth, k, np.eye(4), wrong_board, detector)
        # Scaling the board preserves RGB reprojection but changes metric depth!
        self.assertLess(wrong['reprojection_px']['rms'], 1e-4)
        self.assertAlmostEqual(wrong['median_depth_over_pnp'], A4_SCALE, places=5)

    def test_fixed_board_stays_fixed_and_bad_hand_eye_drifts(self):
        records, bad_records = [], []
        fixed = np.eye(4)
        fixed[:3, 3] = [.4, .1, .03]
        for index, angle in enumerate([-.25, 0, .25]):
            pose = np.eye(4)
            pose[:3, :3] = cv2.Rodrigues(np.array([.15, angle, .07]))[0]
            pose[:3, 3] = [.02, -.01, .65]
            rgb, depth, k, detector, _ = synthetic_scene(self.board, pose)
            camera = fixed @ np.linalg.inv(pose)
            result = analyze_frame(rgb, depth, k, camera, self.board, detector)
            result.update(sample_id=index+1, pose_group=index+1)
            records.append(result)
            offset = np.eye(4)
            offset[:3, 3] = [.05, 0, 0]
            bad = analyze_frame(rgb, depth, k, camera @ offset, self.board, detector)
            bad.update(sample_id=index+1, pose_group=index+1)
            bad_records.append(bad)
        good = summarize(records)
        self.assertEqual(good['status'], 'CONSISTENCY_MEASURED')
        self.assertLess(good['max_pairwise_board_drift_mm'], .001)
        self.assertGreater(summarize(bad_records)['max_pairwise_board_drift_mm'], 20)
        self.assertEqual(good['absolute_accuracy'], 'NOT_VERIFIED')
        repeated = [dict(records[0], sample_id=i, pose_group=1) for i in range(3)]
        self.assertEqual(summarize(repeated)['status'], 'INSUFFICIENT_POSE_VARIATION')

    def test_wrong_family_missing_depth_and_motion(self):
        rgb, depth, k, detector, _ = synthetic_scene(self.board)
        with self.assertRaisesRegex(ValueError, 'Need >=4'):
            analyze_frame(rgb, depth, k, np.eye(4), self.board, SimpleNamespace(detect=lambda *a, **kw: []))
        result = analyze_frame(rgb, np.full_like(depth, np.nan), k, np.eye(4), self.board, detector)
        self.assertEqual(result['depth_minus_pnp_mm']['count'], 0)
        before = {'joints_rad': [0]*7, 'tcp_pose_mm_deg': [0]*6, 'read_duration_s': .01}
        after = copy.deepcopy(before)
        after['joints_rad'][0] = .1
        with self.assertRaisesRegex(ValueError, 'Unstable'):
            script.stationary(before, after)

    def test_read_only_capture_preserves_evidence_and_rejects_movement(self):
        rgb, depth, k, _, _ = synthetic_scene(self.board)
        class Arm:
            connected = True
            tcp_offset = [0, 0, 172, 0, 0, 0]
            q = 0

            def get_servo_angle(self, **kwargs):
                return 0, [self.q]*7

            def get_position(self, **kwargs):
                return 0, [400, 100, 400, 0, 0, 0]
        arm = Arm()
        def read():
            arm.q += .02
            return rgb, depth
        camera = SimpleNamespace(read=read, intrinsics=k, spec=SimpleNamespace(label='A', serial='test'))
        config = script.RobotConfig.load(script.PROJECT_ROOT, script.PROJECT_ROOT/'config/robot.example.json')
        geometry = SimpleNamespace(transform=lambda q: np.eye(4))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'sample_001'
            result, _, _ = script.capture_one(camera, arm, config, geometry, path, 1, 1)
            self.assertEqual(result['status'], 'REJECTED_MOTION')
            self.assertTrue((path/'rgb.png').is_file())
            self.assertFalse((path/'result.json').exists())

    def test_real_detector_on_original_board_asset(self):
        if importlib.util.find_spec('pupil_apriltags') is None:
            self.skipTest('pupil_apriltags is available in cali hardware environment')
        asset = script.PROJECT_ROOT.parent/'RobotCamCalib/assets/apriltag_grid/compact_apriltag_grid_4x4_tag48mm_board_only.png'
        if not asset.exists():
            self.skipTest('Original local RobotCamCalib rendering not available')
        from pupil_apriltags import Detector
        image = Image.open(asset).convert('RGB')
        image.thumbnail((900, 900))
        rgb = np.pad(np.asarray(image), ((60, 60), (60, 60), (0, 0)), constant_values=255)
        height, width = rgb.shape[:2]
        k = np.array([[1000., 0, width/2], [0, 1000, height/2], [0, 0, 1]])
        result = analyze_frame(rgb, np.full((height, width), np.nan), k, np.eye(4), self.board,
                               Detector(families=self.board['family'], quad_decimate=1.0))
        self.assertEqual(len(result['tag_ids']), 16)
        self.assertLess(result['reprojection_px']['rms'], 1)

    def test_offline_cli_never_loads_robot_geometry_or_camera(self):
        if any(importlib.util.find_spec(name) is None for name in ('pupil_apriltags', 'matplotlib')):
            self.skipTest('Full diagnostic dependencies are available in cali environment')
        rgb, depth, k, detector, _ = synthetic_scene(self.board)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'session'
            sample = source/'sample_001'
            sample.mkdir(parents=True)
            (source/'board.yaml').write_bytes(BOARD.read_bytes())
            (source/'session.json').write_text(json.dumps({'board': {'scale': A4_SCALE}}))
            (sample/'capture.json').write_text(json.dumps({'sample_id': 1, 'pose_group': 1, 'status': 'CAPTURED',
                'intrinsics': k.tolist(), 'X_base_camera': np.eye(4).tolist()}))
            Image.fromarray(rgb).save(sample/'rgb.png')
            np.save(sample/'depth_m.npy', depth)
            with patch('pupil_apriltags.Detector', return_value=detector), \
                 patch.object(script, 'WristGeometry', side_effect=AssertionError('hardware')), \
                 patch.object(script, 'RealSenseRGBD', side_effect=AssertionError('hardware')):
                self.assertEqual(script.main(['--session', str(source), '--output-dir', str(root/'out')]), 0)
            output, = (root/'out').iterdir()
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(report['board']['scale'], A4_SCALE)
            self.assertEqual(report['status'], 'INSUFFICIENT_POSE_VARIATION')
            self.assertTrue((output/'diagnostic.png').is_file())


if __name__ == '__main__':
    unittest.main()
