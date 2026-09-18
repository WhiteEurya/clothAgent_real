#!/usr/bin/env python3
"""Observe, click one raw RGB pixel, then position the TCP above its RGB-D point.

One fresh capture and at most one target move per invocation. No gripper commands.
See docs/manual_pixel_move.md. Hardware is only connected with --real --confirm-real.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.config import RobotConfig
from cloth_agent.perception import (
    PerceptionConfig, camera_base_xyz_map_mm, capture_two_view_rgbd,
)
from cloth_agent.robot_api import (
    RobotAPI, XArmBackend, move_robot_to_perception_position,
    validate_controller_trajectory,
)
from cloth_agent.run_storage import auxiliary_dir


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')


def raw_pixel(x, y, display_size, raw_size):
    """Invert the displayed resize; no rotation, cropping, or snapping."""
    dw, dh = display_size
    rw, rh = raw_size
    if not (0 <= x < dw and 0 <= y < dh):
        raise ValueError('Click outside the image')
    return int(x * rw / dw), int(y * rh / dh)


def select_target(frame, xyz, valid, pixel, config, clearance_mm):
    u, v = pixel
    if not (0 <= u < frame.rgb.shape[1] and 0 <= v < frame.rgb.shape[0]):
        raise ValueError('Pixel outside raw RGB')
    if not valid[v, u] or not np.isfinite(xyz[v, u]).all():
        raise ValueError('No valid depth at this pixel; select another point')
    if not np.isfinite(clearance_mm) or clearance_mm < 0:
        raise ValueError('Clearance must be finite and nonnegative')
    surface = xyz[v, u].astype(float)
    target = surface + [0, 0, clearance_mm]
    config.validate_workspace_pose(*target, relative_yaw_deg=0)
    return {'raw_pixel_xy': [u, v], 'depth_m': float(frame.depth_m[v, u]),
            'measured_base_xyz_mm': surface.tolist(), 'target_tcp_xyz_mm': target.tolist(),
            'clearance_mm': float(clearance_mm), 'relative_yaw_deg': 0.0}


def build_actions(selection, config):
    x, y, z = selection['target_tcp_xyz_mm']
    # The controller IK validator starts from configured Home. Execute that
    # same route, keeping the lateral transit at least as high as Home.
    approach_z = max(config.init_pose_mm_deg[2], z + 80.0)
    points = [(x, y, approach_z), (x, y, z)]
    for point in points:
        if not np.isfinite(point).all():
            raise ValueError('Nonfinite motion target')
        config.validate_workspace_pose(*point, relative_yaw_deg=0)
    return [{'name': 'home', 'args': {}}] + [
        {'name': 'move', 'args': dict(x=x, y=y, z=height, yaw=0.0)}
        for x, y, height in points]


def execute_target(config, actions, report, report_path):
    # Read-only IK and live TCP checks finish before creating an enabled backend.
    report['controller_ik'] = asdict(validate_controller_trajectory(config, actions))
    report['status'] = 'EXECUTING'
    write_json(report_path, report)
    robot = RobotAPI(config, XArmBackend(config))
    try:
        for action in actions:
            if action['name'] == 'home':
                robot.home()
            elif action['name'] == 'move':
                robot.move(**action['args'])
            else:
                raise ValueError(f"Unsupported manual action: {action['name']}")
        report['status'] = 'AT_TARGET'
    finally:
        report['actual_robot_actions'] = robot.action_dicts()
        try:
            write_json(report_path, report)
        finally:
            robot.close()  # Disconnect only: leave the TCP at the target for inspection.


def save_capture(directory, frame, xyz, observation, config):
    Image.fromarray(frame.rgb).save(directory / 'rgb.png')
    np.save(directory / 'depth_m.npy', frame.depth_m)
    np.save(directory / f'camera_{frame.label}_base_xyz_mm.npy', xyz)
    write_json(directory / 'result.json', {
        'schema_version': 1, 'captured_at': datetime.now(timezone.utc).isoformat(),
        'observation': observation, 'robot_config': asdict(config),
        'views': [{'label': frame.label, 'serial': frame.serial, 'image': 'rgb.png',
                   'depth_m': 'depth_m.npy', 'intrinsics': frame.intrinsics.tolist(),
                   'X_base_camera': frame.X_base_camera.tolist(), 'base_z_offset_mm': 0.0,
                   'base_xyz_map': f'camera_{frame.label}_base_xyz_mm.npy'}],
        'note': 'Stationary temporal RGB-D aggregate; no online Z-bias correction.'})


def choose_point(root, frame, xyz, valid, config, clearance_mm):
    import tkinter as tk
    from PIL import ImageTk

    root.title('Manual RGB-D point move — raw RGB')
    rgb = Image.fromarray(frame.rgb)
    scale = min(1.0, (root.winfo_screenwidth() - 100) / rgb.width,
                (root.winfo_screenheight() - 230) / rgb.height)
    size = (max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale)))
    photo = ImageTk.PhotoImage(rgb.resize(size))
    canvas = tk.Canvas(root, width=size[0], height=size[1], highlightthickness=0)
    canvas.pack()
    canvas.create_image(0, 0, image=photo, anchor='nw')
    info = tk.StringVar(value=f'Click a point. TCP clearance above measured surface: {clearance_mm:g} mm.')
    tk.Label(root, textvariable=info, justify='left').pack(padx=10, pady=10)
    chosen = None
    confirmed = False

    def confirm(event=None):
        nonlocal confirmed
        if chosen is not None:
            confirmed = True
            root.quit()

    button = tk.Button(root, text='Move to selected point (Enter)', command=confirm, state='disabled')
    button.pack(pady=5)

    def click(event):
        nonlocal chosen
        chosen = None
        button.configure(state='disabled')
        canvas.delete('marker')
        try:
            pixel = raw_pixel(event.x, event.y, size, rgb.size)
            candidate = select_target(frame, xyz, valid, pixel, config, clearance_mm)
            build_actions(candidate, config)
            chosen = candidate
            canvas.create_oval(event.x-5, event.y-5, event.x+5, event.y+5,
                               outline='red', width=2, tags='marker')
            surface = chosen['measured_base_xyz_mm']
            target = chosen['target_tcp_xyz_mm']
            info.set(f'Raw pixel {pixel}; depth {chosen["depth_m"]*1000:.1f} mm\n'
                     f'Surface XYZ: {surface[0]:.2f}, {surface[1]:.2f}, {surface[2]:.2f} mm\n'
                     f'Target TCP: {target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f} mm\n'
                     'Enter: validate IK, Home, approach, descend and hold. Esc: cancel.')
            button.configure(state='normal')
        except (ValueError, RuntimeError) as exc:
            info.set(str(exc))

    canvas.bind('<Button-1>', click)
    root.bind('<Return>', confirm)
    root.bind('<Escape>', lambda event: root.quit())
    root.protocol('WM_DELETE_WINDOW', root.quit)
    root.deiconify()
    root.mainloop()
    return chosen if confirmed else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-config', type=Path, default=PROJECT_ROOT / 'config/robot.example.json')
    parser.add_argument('--perception-config', type=Path,
                        default=PROJECT_ROOT / 'config/perception.free_exploration.json')
    parser.add_argument('--camera', default='A')
    parser.add_argument('--clearance-mm', type=float, default=30.0,
                        help='TCP height above measured point; 0 reaches the measured surface')
    parser.add_argument('--output-dir', type=Path, help='Parent for a new timestamped record directory')
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--confirm-real', action='store_true')
    args = parser.parse_args(argv)
    if not args.real or not args.confirm_real:
        parser.error('This interactive hardware script requires --real --confirm-real')
    if not np.isfinite(args.clearance_mm) or args.clearance_mm < 0:
        parser.error('--clearance-mm must be finite and nonnegative')
    config = RobotConfig.load(PROJECT_ROOT, args.robot_config)
    perception = PerceptionConfig.load(PROJECT_ROOT, args.perception_config)
    label = args.camera.upper()
    if label not in {camera.label for camera in perception.cameras}:
        parser.error(f'Camera {label} is not configured')
    perception = replace(perception, active_camera_labels=(label,))
    # Check GUI availability before issuing any robot command (including Home).
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    report = {'schema_version': 1, 'status': 'STARTING', 'actual_robot_actions': []}
    report_path = None
    try:
        parent = args.output_dir or auxiliary_dir(PROJECT_ROOT, 'manual_pixel_move')
        directory = parent / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        directory.mkdir(parents=True, exist_ok=False)
        report_path = directory / 'execution.json'
        write_json(report_path, report)
        print(f'Records: {directory}\nMoving Home -> observation position...', flush=True)
        report['observation'] = move_robot_to_perception_position(config)
        report['status'] = 'CAPTURING'
        write_json(report_path, report)
        print('Capturing fresh aligned RGB-D; keep the scene stationary.', flush=True)
        frame, = capture_two_view_rgbd(perception)
        xyz, valid = camera_base_xyz_map_mm(frame, perception)
        save_capture(directory, frame, xyz, report['observation'], config)
        selection = choose_point(root, frame, xyz, valid, config, args.clearance_mm)
        root.destroy()
        root = None
        if selection is None:
            report['status'] = 'CANCELLED_AT_OBSERVATION'
            return 0
        report['selection'] = selection
        report['planned_actions'] = build_actions(selection, config)
        report['status'] = 'VALIDATING'
        write_json(report_path, report)
        annotated = Image.fromarray(frame.rgb)
        u, v = selection['raw_pixel_xy']
        ImageDraw.Draw(annotated).ellipse((u-7, v-7, u+7, v+7), outline='red', width=2)
        annotated.save(directory / 'selected_pixel.png')
        print('Checking live TCP and IK, then moving to target...', flush=True)
        execute_target(config, report['planned_actions'], report, report_path)
        print(f'At target. Gripper unchanged. Records: {directory}', flush=True)
        return 0
    except BaseException as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if report_path is not None:
            write_json(report_path, report)
        if root is not None:
            root.destroy()


if __name__ == '__main__':
    raise SystemExit(main())
