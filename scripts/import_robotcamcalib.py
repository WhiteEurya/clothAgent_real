#!/usr/bin/env python3
"""Import saved calibration only; never imports a robot or camera SDK."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--name', default='zed_dual_xarm7_20261103')
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('name must be a directory basename')
    destination = ROOT / 'config/calibration' / args.name
    destination.mkdir(parents=True, exist_ok=False)
    records = {}

    def copy(path, relative):
        path = path.resolve()
        path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        records[str(relative)] = dict(source=str(path.relative_to(source)),
            sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        return str(target.relative_to(ROOT))

    arms, sessions, transforms = {}, {}, {}
    for arm in ('left', 'right'):
        original = source / f'config/point_task_{arm}.yaml'
        config = yaml.safe_load(original.read_text())
        session_dir = Path(config['calibration_session'])
        if not session_dir.is_absolute():
            session_dir = source / session_dir
        session = yaml.safe_load((session_dir / 'session.yaml').read_text())
        if session['mode'] != 'thirdview' or session['cammount_link'] != 'link_base':
            raise ValueError('Only fixed thirdview calibration is supported')
        if session['robot_ip'] != config['robot_ip'] or session['robot_axis'] != config['robot_axis']:
            raise ValueError('Robot configuration/session mismatch')
        sessions[arm] = session
        file_paths = {name: copy(session_dir / name, Path(arm) / name)
                      for name in ('session.yaml', 'intrinsics_left.yaml', 'extrinsics.yaml')}
        copy(original, Path(arm) / 'point_task_source.yaml')
        for summary in session_dir.glob('validation_*/summary.yaml'):
            copy(summary, Path(arm) / summary.parent.name / 'summary.yaml')
        urdf = Path(session['robot_urdf'])
        model = copy(urdf, Path('robot_model') / urdf.name)
        for mesh in ET.parse(urdf).getroot().iter('mesh'):
            relative = Path(mesh.attrib['filename'])
            copy(urdf.parent / relative, Path('robot_model') / relative)
        transforms[arm] = np.asarray(yaml.safe_load((session_dir / 'extrinsics.yaml').read_text())['X_CammountCam'])
        arms[arm] = dict(config, calibration_session=str((destination / arm).relative_to(ROOT)),
                         urdf=model, extrinsics_file=file_paths['extrinsics.yaml'])
    for field in ('serial', 'stream', 'coordinate_system', 'translation_units', 'image_size'):
        if sessions['left'][field] != sessions['right'][field]:
            raise ValueError(f'Camera calibration mismatch: {field}')
    home = yaml.safe_load((source / 'config/dual_arm_home.yaml').read_text())
    home_path = copy(source / 'config/dual_arm_home.yaml', Path('dual_arm_home_source.yaml'))
    common = {'right': np.eye(4), 'left': transforms['right'] @ np.linalg.inv(transforms['left'])}
    for arm in arms:
        if home['arms'][arm]['robot_ip'] != arms[arm]['robot_ip']:
            raise ValueError('Home robot identity mismatch')
        if not np.allclose(common[arm], home['T_common_base'][arm], atol=1e-8):
            raise ValueError('Home and calibration frames differ')
        arms[arm]['home'] = home['arms'][arm]
        # Keep the imported point-task rectangle as provenance only.  The
        # current ZED execution policy enforces the measured minimum Z; X/Y
        # and maximum Z are intentionally left to IK and collision proofs.
        if 'target_bounds_mm' in arms[arm]:
            arms[arm]['source_target_bounds_mm'] = arms[arm]['target_bounds_mm']
            arms[arm]['target_bounds_mm'] = [[None, None], [None, None],
                                              [arms[arm]['source_target_bounds_mm'][2][0], None]]
        arms[arm]['workspace_policy'] = 'minimum_base_z_only'
        arms[arm]['home']['calibration_session'] = arms[arm]['calibration_session']
        arms[arm]['common_from_base_m'] = common[arm].tolist()
    for name in ('alignment_validation_20261103', 'left_home_20261103', 'dual_parameter_scan_20261103'):
        folder = source / 'outputs' / name
        for filename in ('RESULTS.md', 'report.json', 'execution.json'):
            if (folder / filename).exists():
                copy(folder / filename, Path('evidence') / name / filename)
    camera = dict(backend='zed', label='A', serial=str(sessions['right']['serial']),
                  mount_arm=None, width=1280, height=720, fps=30, depth_mode='ULTRA',
                  confidence_threshold=50, extrinsics_file=arms['right']['extrinsics_file'])
    profile = dict(schema_version=1, profile_kind='calibrated_observation_only',
        calibration_id=args.name, common_frame='right_arm_base', camera=camera, arms=arms,
        robot_connection_enabled=False, physical_alignment_validated=False,
        home_source=home_path, source_root=str(source),
        pending_robot_integration=['controller identities', 'fresh state and TCP checks',
            'full task workspace and collision geometry', 'home/perception path commissioning',
            'six-axis graph action adaptation to seven-axis robots'])
    (destination / 'manifest.json').write_text(json.dumps(dict(source_root=str(source), files=records), indent=2)+'\n')
    profile['manifest'] = str((destination / 'manifest.json').relative_to(ROOT))
    profile_path = ROOT / 'config' / f'site.{args.name}.json'
    with profile_path.open('x') as handle:
        json.dump(profile, handle, indent=2)
        handle.write('\n')
    perception = dict(cameras=[camera], active_cameras=['A'], width=1280, height=720, fps=30,
        warmup_frames=10, temporal_median_frames=5, min_depth_m=.15, max_depth_m=2.,
        table_appearance_mode='border_background', table_plane_mode='background_fit',
        table_roi_xyxy=[0., 0., 1., 1.], manual_base_y_offset_mm=0.)
    # ROI is intentionally uncommissioned; this config supplies RGB-D, not a validated garment mask.
    perception['table_roi_status'] = 'pending_current_scene'
    with (ROOT / 'config' / f'perception.{args.name}.json').open('x') as handle:
        json.dump(perception, handle, indent=2)
        handle.write('\n')
    print(profile_path)


if __name__ == '__main__':
    main()
