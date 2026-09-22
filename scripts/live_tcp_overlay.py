#!/usr/bin/env python3
"""Freeze observation RGB and overlay TCP feedback; optionally scan at fixed X=420."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.config import RobotConfig
from cloth_agent.perception import PerceptionConfig, camera_base_xyz_map_mm, capture_two_view_rgbd
from cloth_agent.robot_api import _validated_live_tcp_offset, move_robot_to_perception_position
from cloth_agent.run_storage import auxiliary_dir
from scripts.manual_pixel_move import save_capture, write_json


class FrozenCameraProjection:
    """Use the camera pose at capture time, even after the wrist has moved."""

    def __init__(self, intrinsics, X_base_camera, image_size):
        self.k = np.array(intrinsics, dtype=float, copy=True)
        transform = np.array(X_base_camera, dtype=float, copy=True)
        if (self.k.shape != (3, 3) or not np.isfinite(self.k).all()
                or min(self.k[0, 0], self.k[1, 1]) <= 0
                or not np.allclose(self.k[2], [0, 0, 1])):
            raise ValueError('Invalid intrinsics')
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-4)):
            raise ValueError('Invalid camera transform')
        self.X_camera_base = np.linalg.inv(transform)
        self.width, self.height = image_size

    def project(self, base_xyz_mm):
        point = np.asarray(base_xyz_mm, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('TCP XYZ must contain three finite numbers')
        camera = self.X_camera_base @ np.r_[point / 1000, 1.0]
        if camera[2] <= 1e-6:
            return {'status': 'BEHIND_CAMERA', 'raw_pixel_xy': None}
        uvw = self.k @ camera[:3]
        uv = uvw[:2] / uvw[2]
        if not np.isfinite(uv).all():
            raise ValueError('Nonfinite projected pixel')
        inside = 0 <= uv[0] < self.width and 0 <= uv[1] < self.height
        return {'status': 'VISIBLE' if inside else 'OUTSIDE_IMAGE',
                'raw_pixel_xy': uv.tolist(), 'camera_depth_m': float(camera[2])}


def read_pose(arm, config):
    """No enable, mode, state, gripper, or motion commands are permitted here."""
    if not arm.connected:
        raise RuntimeError('Robot disconnected')
    result = arm.get_position(is_radian=False)
    if not isinstance(result, (tuple, list)) or len(result) != 2 or result[0] != 0:
        raise RuntimeError(f'get_position failed: {result}')
    pose = np.asarray(result[1], dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise RuntimeError('Invalid robot TCP pose')
    config.validate_live_tcp_offset(arm.tcp_offset)
    return pose.tolist()


def check_capture_stationary(before, after):
    translation = float(np.linalg.norm(np.asarray(after[:3]) - before[:3]))
    rotation = float(np.max(np.abs((np.asarray(after[3:]) - before[3:] + 180) % 360 - 180)))
    if translation > 1.0 or rotation > .5:
        raise RuntimeError(f'Robot moved during capture: {translation:.2f} mm / {rotation:.2f} deg; recapture')


def poll_sample(arm, config, projection):
    started = time.monotonic()
    sample = {'requested_at': datetime.now(timezone.utc).isoformat(),
              'sample_monotonic': started}
    try:
        pose = read_pose(arm, config)
        sample.update(tcp_pose_mm_deg=pose, projection=projection.project(pose[:3]), status='OK')
    except Exception as exc:
        sample.update(status='READ_ERROR', error=f'{type(exc).__name__}: {exc}')
    sample['read_duration_s'] = time.monotonic() - started
    return sample


def telemetry_loop(arm, config, projection, latest, stop, log_path, interval):
    """Worker owns network reads; a single-slot mailbox prevents stale backlogs."""
    try:
        with log_path.open('a', encoding='utf-8') as log:
            while not stop.is_set():
                sample = poll_sample(arm, config, projection)
                log.write(json.dumps(sample, allow_nan=False) + '\n')
                log.flush()
                publish_latest(latest, sample)
                stop.wait(interval)
    except Exception as exc:
        publish_latest(latest, {'status': 'READ_ERROR', 'error': f'Telemetry stopped: {exc}',
                                'sample_monotonic': time.monotonic()})


def publish_latest(latest, sample):
    try:
        latest.get_nowait()
    except queue.Empty:
        pass
    latest.put_nowait(sample)


def overlay_status(sample, now, stale_s=1.0):
    if sample is None:
        return 'WAITING', None
    if sample['status'] != 'OK':
        return 'READ_ERROR', None
    if now - sample['sample_monotonic'] > stale_s:
        return 'STALE', None
    projection = sample['projection']
    return projection['status'], projection['raw_pixel_xy'] if projection['status'] == 'VISIBLE' else None


def biased_projection(sample, now, image_size, pixel_offset=(0.0, 0.0)):
    """Display-only offset in ORIGINAL RGB pixels; never modifies robot data."""
    offset = np.asarray(pixel_offset, dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError('Pixel offset must contain two finite numbers')
    status, _ = overlay_status(sample, now)
    result = {'pixel_offset_xy': offset.tolist(), 'status': status, 'raw_pixel_xy': None,
              'biased_pixel_xy': None, 'display_only': True}
    if status not in {'VISIBLE', 'OUTSIDE_IMAGE'}:
        return result
    original = np.asarray(sample['projection']['raw_pixel_xy'], dtype=float)
    corrected = original + offset
    width, height = image_size
    result.update(raw_pixel_xy=original.tolist(), biased_pixel_xy=corrected.tolist(),
                  status='VISIBLE' if 0 <= corrected[0] < width and 0 <= corrected[1] < height else 'OUTSIDE_IMAGE')
    return result


def render_overlay(rgb, sample, now, pixel_offset=(0.0, 0.0)):
    image = rgb.copy()
    draw = ImageDraw.Draw(image)
    status, pixel = overlay_status(sample, now)
    if pixel is not None:
        u, v = pixel
        draw.ellipse((u-9, v-9, u+9, v+9), outline='#00ff80', width=3)
        draw.line((u-15, v, u+15, v), fill='#00ff80', width=2)
        draw.line((u, v-15, u, v+15), fill='#00ff80', width=2)
        draw.text((u+12, v+10), 'TCP', fill='#00ff80', stroke_width=1, stroke_fill='black')
    biased = biased_projection(sample, now, rgb.size, pixel_offset)
    if biased['status'] == 'VISIBLE' and any(pixel_offset):
        u, v = biased['biased_pixel_xy']
        if pixel is not None:
            draw.line((*pixel, u, v), fill='#ffbb40', width=1)
        draw.rectangle((u-7, v-7, u+7, v+7), outline='#ffbb40', width=3)
        draw.text((u+12, v-20), 'OFFSET', fill='#ffbb40', stroke_width=1, stroke_fill='black')
    return image, status


class YProjectionPlot:
    """Plot displayed telemetry, not an independent visual measurement."""

    def __init__(self, root, directory):
        import tkinter as tk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.directory = directory
        self.rows = []
        self.last_timestamp = None
        self.window = tk.Toplevel(root)
        self.window.title('TCP base Y vs projected RGB y')
        self.window.protocol('WM_DELETE_WINDOW', self.window.withdraw)
        self.figure = Figure(figsize=(7, 4), tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel('Actual TCP base Y (mm)')
        self.axes.set_ylabel('Projected RGB y / v (original pixels)')
        self.axes.set_title('Fixed reference camera projection (not visual tracking)')
        self.axes.grid(True, alpha=.3)
        self.lines = {phase: self.axes.plot([], [], '.', markersize=3, label=phase)[0]
                      for phase in ('APPROACHING', 'SCANNING', 'MONITORING')}
        self.axes.legend()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.log = (directory / 'y_projection.jsonl').open('w', encoding='utf-8')

    def update(self, sample, now):
        status, _ = overlay_status(sample, now)
        if status not in ('VISIBLE', 'OUTSIDE_IMAGE'):
            return
        timestamp = sample['sample_monotonic']
        if timestamp == self.last_timestamp:
            return
        uv = sample['projection']['raw_pixel_xy']
        xyz = sample['tcp_pose_mm_deg'][:3]
        if uv is None or not np.isfinite([*xyz, *uv]).all():
            return
        self.last_timestamp = timestamp
        phase = sample.get('scan_status', 'MONITORING')
        if phase not in self.lines:
            phase = 'MONITORING'
        row = dict(sample_monotonic=timestamp, phase=phase, base_xyz_mm=list(xyz),
                   rgb_pixel_xy=list(uv), projection_status=status)
        self.log.write(json.dumps(row, allow_nan=False)+'\n')
        self.log.flush()
        self.rows.append(row)
        for name, line in self.lines.items():
            rows = [r for r in self.rows if r['phase'] == name]
            line.set_data([r['base_xyz_mm'][1] for r in rows],
                          [r['rgb_pixel_xy'][1] for r in rows])
        self.axes.relim()
        self.axes.autoscale_view()
        self.canvas.draw_idle()

    def save(self):
        self.figure.savefig(self.directory / 'y_projection.png', dpi=160)

    def close(self):
        try:
            self.save()
        finally:
            self.log.close()


def run_viewer(root, frame, latest, directory, pixel_offset=(0.0, 0.0), scan_begin=None):
    import tkinter as tk
    from PIL import ImageTk

    root.title('Frozen observation photo | ' + ('automatic approach + Y scan' if scan_begin is not None else 'live TCP projection (read-only)'))
    rgb = Image.fromarray(frame.rgb)
    offset = [float(value) for value in pixel_offset]
    scale = min(1.0, (root.winfo_screenwidth()-80)/rgb.width,
                (root.winfo_screenheight()-270)/rgb.height)
    size = (max(1, int(rgb.width*scale)), max(1, int(rgb.height*scale)))
    image_label = tk.Label(root)
    image_label.pack()
    info = tk.StringVar(value='Waiting for robot feedback')
    tk.Label(root, textvariable=info, justify='left', font=('TkDefaultFont', 12)).pack(pady=8)
    tk.Label(root, text='Move using pendant/vendor controls. Marker = configured TCP, not jaw tips.\n'
             'Green: original TCP. Orange: display offset. Arrows: 1 px; Shift+arrows: 10 px; R: reset.\n'
             'Fixed photo / fixed camera transform. S: save screenshot. ' +
             ('Esc: stop motion and close.' if scan_begin is not None else 'Esc: close (no robot command).')).pack()
    current = None
    y_plot = YProjectionPlot(root, directory)
    if scan_begin is not None:
        tk.Label(root, text='SCAN: select position mode. B automatically approaches X=420 and configured start Y/Z.\n'
                 'Elevated transit, vertical descent, then one Y sweep. Esc/close: request stop.',
                 fg='red').pack()
        root.bind('<b>', lambda event: scan_begin.set())
        root.bind('<B>', lambda event: scan_begin.set())

    def record_offset():
        write_json(directory / 'display_offset.json', {
            'pixel_offset_xy': offset, 'units': 'original_RGB_pixels', 'display_only': True,
            'convention': 'biased_uv = original_uv + offset; +u right, +v down',
            'updated_at': datetime.now(timezone.utc).isoformat()})

    def nudge(event):
        step = 10 if event.state & 0x0001 else 1
        dx, dy = {'Left': (-step, 0), 'Right': (step, 0),
                  'Up': (0, -step), 'Down': (0, step)}[event.keysym]
        offset[0] += dx
        offset[1] += dy
        record_offset()
        return 'break'

    def reset(event=None):
        offset[:] = [0.0, 0.0]
        record_offset()

    record_offset()

    def refresh():
        nonlocal current
        try:
            current = latest.get_nowait()
        except queue.Empty:
            pass
        now = time.monotonic()
        y_plot.update(current, now)
        image, status = render_overlay(rgb, current, now, offset)
        photo = ImageTk.PhotoImage(image.resize(size))
        image_label.configure(image=photo)
        image_label.image = photo
        text = status
        if current is not None and 'scan_status' in current:
            text += f' | Scan: {current["scan_status"]}'
            if 'x_error_mm' in current:
                text += f' | dX={current["x_error_mm"]:+.3f} mm'
            if 'measured_y_speed_mm_s' in current:
                text += f' | Vy={current["measured_y_speed_mm_s"]:+.2f} mm/s'
        if current is not None and status not in {'READ_ERROR', 'STALE'} and current['status'] == 'OK':
            x, y, z = current['tcp_pose_mm_deg'][:3]
            text += f' | TCP base XYZ: {x:.2f}, {y:.2f}, {z:.2f} mm'
            uv = current['projection']['raw_pixel_xy']
            if uv is not None:
                text += f' | raw pixel: {uv[0]:.1f}, {uv[1]:.1f}'
        elif status == 'READ_ERROR':
            text += f' | {current["error"]}'
        elif status == 'STALE':
            text += ' | feedback older than 1 second; marker hidden'
        biased = biased_projection(current, now, rgb.size, offset)
        text += f'\nDisplay offset (raw pixels): du={offset[0]:+g}, dv={offset[1]:+g} | {biased["status"]}'
        if biased['biased_pixel_xy'] is not None:
            u, v = biased['biased_pixel_xy']
            text += f' | offset pixel: {u:.1f}, {v:.1f}'
        info.set(text)
        root.after(50, refresh)

    def snapshot(event=None):
        y_plot.save()
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        now = time.monotonic()
        image, status = render_overlay(rgb, current, now, offset)
        # Include the offset legend in saved images as well as the JSON record.
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, min(image.width, 610), 36), fill='black')
        draw.text((6, 4), f'Display only: du={offset[0]:+g}, dv={offset[1]:+g} raw px | {status}', fill='white')
        draw.text((6, 20), 'Green=original TCP; orange=offset. Robot/calibration unchanged.', fill='white')
        image.save(directory / f'overlay_{timestamp}.png')
        write_json(directory / f'overlay_{timestamp}.json', {
            'display_status': status, 'sample': current,
            'display_bias': biased_projection(current, now, rgb.size, offset)})
        print(f'Saved overlay_{timestamp}.png ({status})', flush=True)

    root.bind('<s>', snapshot)
    root.bind('<S>', snapshot)
    for key in ('Left', 'Right', 'Up', 'Down'):
        root.bind(f'<{key}>', nudge)
        root.bind(f'<Shift-{key}>', nudge)
    root.bind('<r>', reset)
    root.bind('<R>', reset)
    root.bind('<Escape>', lambda event: root.quit())
    root.protocol('WM_DELETE_WINDOW', root.quit)
    root.deiconify()
    refresh()
    try:
        root.mainloop()
    finally:
        y_plot.close()
    return offset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-config', type=Path, default=PROJECT_ROOT / 'config/robot.example.json')
    parser.add_argument('--perception-config', type=Path,
                        default=PROJECT_ROOT / 'config/perception.free_exploration.json')
    parser.add_argument('--camera', default='A')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--pixel-offset', nargs=2, type=float, default=[0.0, 0.0], metavar=('DU', 'DV'),
                        help='Display-only original RGB pixel offset: +DU right, +DV down; adjustable with arrow keys')
    parser.add_argument('--capture-current', action='store_true', help='Capture at current stationary pose; no initial motion')
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--confirm-real', action='store_true')
    parser.add_argument('--scan-y', nargs=2, type=float, metavar=('START', 'END'), help='One fixed-X=420 Y sweep in mm')
    parser.add_argument('--scan-z', type=float, help='Explicit fixed scan height in base mm')
    parser.add_argument('--scan-speed', type=float, help='Explicit commanded scan speed in mm/s')
    parser.add_argument('--scan-x-tolerance', type=float, default=.5, help='Abort on measured X error exceeding this mm, maximum 0.5')
    args = parser.parse_args(argv)
    if not np.isfinite(args.pixel_offset).all():
        parser.error('--pixel-offset must contain finite numbers')
    if not args.capture_current and not (args.real and args.confirm_real):
        parser.error('Moving to observation requires --real --confirm-real; or use --capture-current')
    scan_settings = None
    if args.scan_y is not None:
        if not (args.real and args.confirm_real) or args.scan_z is None or args.scan_speed is None:
            parser.error('--scan-y requires --real --confirm-real --scan-z and --scan-speed')
        from cloth_agent.fixed_x_scan import ScanSettings
        scan_settings = ScanSettings(*args.scan_y, args.scan_z, args.scan_speed, args.scan_x_tolerance)
    elif args.scan_z is not None or args.scan_speed is not None:
        parser.error('--scan-z/--scan-speed require --scan-y')
    config = RobotConfig.load(PROJECT_ROOT, args.robot_config)
    if scan_settings is not None:
        scan_settings.validate(config)
    perception = PerceptionConfig.load(PROJECT_ROOT, args.perception_config)
    label = args.camera.upper()
    if label not in {camera.label for camera in perception.cameras}:
        parser.error(f'Camera {label} is not configured')
    perception = replace(perception, active_camera_labels=(label,))
    # Display must work before any motion. The SDK import also precedes motion.
    import tkinter as tk
    from xarm.wrapper import XArmAPI
    root = tk.Tk()
    root.withdraw()
    arm = None
    worker = None
    stop = threading.Event()
    report = {'status': 'STARTING', 'monitor_policy': 'read_only', 'capture_current': args.capture_current,
              'initial_display_pixel_offset_xy': args.pixel_offset, 'offset_policy': 'display_only_original_RGB_pixels'}
    if scan_settings is not None:
        from dataclasses import asdict
        report.update(monitor_policy='operator_triggered_fixed_x_scan', scan_settings=asdict(scan_settings))
    directory = None
    try:
        parent = args.output_dir or auxiliary_dir(PROJECT_ROOT, 'live_tcp_overlay')
        directory = parent / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / 'session.json', report)
        observation = {}
        if not args.capture_current:
            print('Moving Home -> observation position...', flush=True)
            observation = move_robot_to_perception_position(config)
        # Deliberately not XArmBackend: it enables motion and changes mode/state.
        arm = XArmAPI(config.robot_ip, is_radian=False)
        # A new SDK connection initially exposes an all-zero report cache.
        # Reuse the normal backend's read-only startup guard before read_pose
        # performs strict per-sample checks. Never write/replace the live offset.
        print('Waiting for the initial TCP offset report...', flush=True)
        report['startup_tcp_offset_mm_deg'] = list(_validated_live_tcp_offset(arm, config))
        print('Hold still while capturing the reference image...', flush=True)
        before = read_pose(arm, config)
        frame, = capture_two_view_rgbd(perception)
        after = read_pose(arm, config)
        check_capture_stationary(before, after)
        observation.update(capture_before_tcp_pose_mm_deg=before, capture_after_tcp_pose_mm_deg=after)
        xyz, _ = camera_base_xyz_map_mm(frame, perception)
        save_capture(directory, frame, xyz, observation, config)
        projection = FrozenCameraProjection(frame.intrinsics, frame.X_base_camera,
                                            (frame.rgb.shape[1], frame.rgb.shape[0]))
        report.update(status='MONITORING', tcp_offset_mm_deg=list(arm.tcp_offset),
                      frozen_camera_transform=frame.X_base_camera.tolist())
        write_json(directory / 'session.json', report)
        latest = queue.Queue(maxsize=1)
        scan_begin = None
        if scan_settings is None:
            worker = threading.Thread(target=telemetry_loop,
                                      args=(arm, config, projection, latest, stop, directory / 'tcp_samples.jsonl', .1),
                                      daemon=True)
        else:
            from cloth_agent.fixed_x_scan import scan_monitor
            scan_begin = threading.Event()
            worker = threading.Thread(target=scan_monitor,
                                      args=(arm, config, projection, latest, stop, scan_begin, scan_settings,
                                            directory, poll_sample, publish_latest, read_pose), daemon=True)
        worker.start()
        print(f'Reference captured. {"Press B to approach the start and scan." if scan_settings else "You may now move manually."} Records: {directory}', flush=True)
        if scan_begin is None:
            report['final_display_pixel_offset_xy'] = run_viewer(root, frame, latest, directory, args.pixel_offset)
        else:
            report['final_display_pixel_offset_xy'] = run_viewer(root, frame, latest, directory, args.pixel_offset, scan_begin)
        report['status'] = 'CLOSED'
        return 0
    except BaseException as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        stop.set()
        if scan_settings is not None and worker is not None:
            # Give the scan worker a chance to issue its stop before disconnect.
            worker.join(timeout=2)
            if worker.is_alive() and arm is not None:
                try:
                    report['scan_exit_stop_code'] = arm.set_state(4)
                except Exception as exc:
                    report['scan_exit_stop_error'] = str(exc)
        if arm is not None:
            arm.disconnect()  # No Home, stop, mode change, or gripper command on exit.
        if worker is not None:
            worker.join(timeout=2)
        root.destroy()
        if directory is not None:
            write_json(directory / 'session.json', report)


if __name__ == '__main__':
    raise SystemExit(main())
