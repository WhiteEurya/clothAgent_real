"""Manual-Pa single trial or fixed-world-XY batch. No model, graph, folding or retry.

Uses Arm7Runtime's existing observation, height adjustment, approach, close,
Cartesian planning, release and retreat. All motion phases are certified before
dispatch. A stationary RGB read at the lift checkpoint never calls observe().
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import shutil
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
LABELS = ('HELD', 'MISS', 'UNCERTAIN')
LIFT_MM = 25.


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def allocate_trial(root):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    with (root/'.index.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        numbers = [int(p.name[6:]) for p in root.glob('trial_*') if p.is_dir() and p.name[6:].isdigit()]
        path = root/f'trial_{max(numbers, default=0)+1:06d}'
        path.mkdir(exist_ok=False)
        return path


def patch_stats(values, pa, radius=2, valid=None):
    """Statistics over the clipped pixel window, never replace the runtime sampler."""
    x, y = map(int, pa); h, w = values.shape
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError('Pa outside image')
    bounds = [max(0,x-radius),max(0,y-radius),min(w,x+radius+1),min(h,y+radius+1)]
    x0,y0,x1,y1 = bounds
    a = np.asarray(values[y0:y1,x0:x1], dtype=float)
    mask = np.isfinite(a)
    if valid is not None: mask &= valid[y0:y1,x0:x1]
    good = a[mask]
    quantiles = np.percentile(good,[0,25,50,75,100]).tolist() if good.size else [None]*5
    return dict(zip(('min','p25','median','p75','max'),quantiles),
        valid_ratio=float(mask.mean()),valid_count=int(mask.sum()),pixel_count=int(mask.size),
        bounds_xyxy=bounds,units='mm')


def raw_cloud(observation):
    path = observation.directory/'uncorrected_world_xyz_mm.npy'
    return np.load(path,allow_pickle=False) if path.exists() else observation.xyz


def camera_measurement(observation, pa):
    xyz = raw_cloud(observation)
    transform = np.asarray(observation.meta['world_from_camera_mm'],dtype=float)
    # Rotation correction stores its physical pre-correction extrinsic separately.
    transform = np.asarray(observation.meta.get('uncorrected_world_from_camera_mm',transform),dtype=float)
    camera_z = (xyz-transform[:3,3]) @ transform[:3,2]
    valid = np.isfinite(xyz).all(axis=2) & (camera_z>100) & (camera_z<=2500)
    x,y = map(int,pa)
    return dict(pixel_xy=[x,y],raw_world_z_mm=float(xyz[y,x,2]) if valid[y,x] else None,
        raw_camera_depth_z_mm=float(camera_z[y,x]) if valid[y,x] else None,
        raw_world_z_patch=patch_stats(xyz[:,:,2],pa,valid=valid),
        depth_patch=patch_stats(camera_z,pa,valid=valid),
        raw_semantics='Temporal depth aggregate before support-height/world-tilt correction; existing extrinsic calibration retained',
        camera_serial=observation.meta['camera_serial'],observation=str(observation.directory))


def measure_pa(observation, pa, max_spread):
    # Exactly the existing arm7/replay pixel -> surface sampler (5x5 median).
    b = camera_measurement(observation,pa)
    sample_error=None
    try:xyz=observation.sample(pa,max_spread)
    except Exception as exc:
        xyz=None;sample_error=f'{type(exc).__name__}: {exc}'
    raw = raw_cloud(observation)
    x,y = map(int,pa)
    patch = observation.meta.get('support_height_patch')
    table = dict(z_mm=None,raw_plane_z_mm=None,physical_table_z_mm=None,
        status='UNAVAILABLE',reason='No active calibrated table/support plane in this observation')
    if patch and np.isfinite(raw[y,x,:2]).all():
        a,beta,c = patch['plane_z_ax_by_c']
        raw_xy = raw[y,x,:2]
        z = float(a*raw_xy[0]+beta*raw_xy[1]+c)
        matrix = np.asarray(patch['matrix_mm'])
        mapped = matrix @ np.array([*raw_xy,z,1.])
        table = dict(z_mm=float(mapped[2]),raw_plane_z_mm=z,physical_table_z_mm=None,
            status='CONFIGURED_SUPPORT_REFERENCE',
            semantics=patch.get('contact_reference',{}).get('semantics','configured support reference plane'),
            source=patch.get('source'),
            limitation='This may be an empirical contact TCP baseline, not a measured physical tabletop surface.')
    return dict(Pa=[x,y],surface_xyz_mm=xyz.tolist() if xyz is not None else None,
        current_surface_z_mm=float(xyz[2]) if xyz is not None else None,sample_error=sample_error,
        command_grasp_z_mm=None,
        fused_surface_z_mm=None,surface_fusion_mode='B_ONLY_NO_AB_FUSION',
        surface_source='Arm7Runtime observation.sample: stabilized Camera B plus existing configured correction',
        B=b,A=dict(status='UNAVAILABLE',raw_world_z_mm=None,raw_camera_depth_z_mm=None),
        table=table,table_z_mm=table['z_mm'],
        current_surface_z_patch=patch_stats(observation.xyz[:,:,2],pa,valid=np.isfinite(observation.xyz).all(axis=2)),
        sample_max_spread_mm=float(max_spread))


def project_point(point, transform, intrinsics, shape):
    camera = np.linalg.solve(np.asarray(transform),np.r_[point,1.])[:3]
    if not np.isfinite(camera).all() or camera[2]<=0:return None
    pixel = np.asarray(intrinsics) @ camera
    uv = np.rint(pixel[:2]/pixel[2]).astype(int)
    h,w = shape
    return uv.tolist() if 2<=uv[0]<w-2 and 2<=uv[1]<h-2 else None


def capture_a(runtime, config_path, before, pa, folder):
    """Diagnostic only: A stays at its current mount pose, never drives arm6."""
    from cloth_agent.dual_arm.config import DualConfig
    from cloth_agent.dual_arm.observation import capture, create_camera
    config = DualConfig.load(config_path,root=ROOT,homing_only=True)
    if config.raw['camera']['mount_arm']!='left':
        raise ValueError('Camera A diagnostic requires the existing left-mounted calibration')
    camera = create_camera(config)
    try:
        obs = capture(folder,config,runtime.models,runtime.coordinator.snapshots,camera=camera)
        intrinsics = camera.intrinsics.copy()
        point = raw_cloud(before)[pa[1],pa[0]]
        uv = project_point(point,obs.meta['world_from_camera_mm'],intrinsics,obs.xyz.shape[:2])
        np.save(folder/'intrinsics.npy',intrinsics,allow_pickle=False)
        if uv is None:
            return dict(status='OUT_OF_VIEW',raw_world_z_mm=None,raw_camera_depth_z_mm=None,
                observation=str(folder),reason='B Pa projected outside Camera A view')
        result = camera_measurement(obs,uv)
        result.update(status='MEASURED_PROJECTED_LOCATION',
            registration='B uncorrected 3D point projected through existing A calibration; not semantic correspondence',
            mount_pose_source='Existing coordinator snapshot; arm6 may be represented by saved static obstacle pose',
            projection_world_distance_mm=float(np.linalg.norm(raw_cloud(obs)[uv[1],uv[0]]-point))
                if np.isfinite(raw_cloud(obs)[uv[1],uv[0]]).all() else None,
            used_for_command=False)
        return result
    finally:
        camera.stop()


def select_pa(image_path):
    import cv2
    image = cv2.imread(str(image_path))
    if image is None:raise ValueError('Cannot open current RGB')
    chosen=[]; name='Pa: click, Enter=confirm, Esc=cancel'
    def mouse(event,x,y,*_):
        if event==cv2.EVENT_LBUTTONDOWN:chosen[:]=[int(x),int(y)]
    cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE);cv2.setMouseCallback(name,mouse)
    try:
        while True:
            frame=image.copy()
            if chosen:
                cv2.drawMarker(frame,tuple(chosen),(0,0,255),cv2.MARKER_CROSS,20,2)
                cv2.putText(frame,f'Pa={chosen}',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
            cv2.imshow(name,frame);key=cv2.waitKey(40)&255
            if key in (10,13) and chosen:return chosen
            if key==27 or cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1:
                raise KeyboardInterrupt('Operator cancelled Pa selection')
    finally:cv2.destroyWindow(name)


def low_speed(runtime):
    caps=dict(joint_speed_deg_s=2.,joint_accel_deg_s2=4.,cartesian_speed_mm_s=5.,cartesian_accel_mm_s2=10.)
    prior=dict(runtime.config.limits)
    runtime.acquire_home_limits={k:prior[k] for k in ('joint_speed_deg_s','joint_accel_deg_s2')}
    for key,value in caps.items():runtime.config.limits[key]=min(prior[key],value)
    return dict(original={k:prior[k] for k in caps},effective={k:runtime.config.limits[k] for k in caps},
        return_home=runtime.acquire_home_limits,
        scope='This runtime only; gripper parameters and safety configuration unchanged')


@contextmanager
def home_speed(runtime):
    """Use original configured joint dynamics for empty-hand Home, including proof."""
    limits=getattr(runtime,'acquire_home_limits',{})
    previous={k:runtime.config.limits[k] for k in limits}
    try:
        if limits:runtime.config.limits.update(limits)
        yield
    finally:
        if previous:runtime.config.limits.update(previous)


def retreat_and_home(runtime, placed, *, complete_release=False):
    from cloth_agent.codex_grasp_task import _clearance_targets, _context, _joint_goal, _grippers
    from cloth_agent.dual_arm.geometry import DualArmError, matrix_pose, pose_matrix
    world=runtime.config.arms['right'].world_from_base @ pose_matrix(runtime.models['right'].forward(placed['right']))
    pose=matrix_pose(world);pose=np.r_[pose[:3]/1000,np.radians(pose[3:])]
    failures=[]
    last=DualArmError('No vertical clearance available before full release')
    for height,inward,hover in _clearance_targets(runtime,'right',pose):
        if complete_release and (inward or height<20.):continue
        if runtime.cancel.is_set():raise KeyboardInterrupt('planning cancelled')
        try:
            _context(runtime,1,'host_retreat','right',retreat_height_mm=height,inward_mm=inward)
            segments,raised=runtime._tcp_segments('right',hover,placed,allow_detour=False)
            release=_grippers(runtime,raised,('right',),'open') if complete_release else []
            goal={k:v.copy() for k,v in raised.items()}
            goal['right']=runtime.config.arms['right'].home_joints.copy()
            _context(runtime,1,'host_return_home','right',retreat_height_mm=height)
            with home_speed(runtime):
                home,_=_joint_goal(runtime,'right',goal,raised)
            return [('move',s) for s in segments]+release,home
        except DualArmError as exc:
            failures.append(dict(height_mm=height,inward_mm=inward,error=str(exc)))
            last=exc
    runtime.planning_context['host_attempts']=failures
    raise last


def prepare_source(runtime, before, xyz, yaw_deg, folder):
    from cloth_agent.codex_motion import parse_code
    from cloth_agent.grasp_orientation import nearest_grasp_orientation
    from cloth_agent.dual_arm.geometry import pose_matrix
    pose=[*(np.asarray(xyz)/1000).tolist(),math.pi,0.,math.radians(yaw_deg)]
    lifted=pose.copy();lifted[2]+=LIFT_MM/1000
    source=f'motion.grasp(right={pose!r})\nmotion.move_tcp(right={lifted!r})\n'
    joints=before.meta['joints_deg']['right']
    reference=runtime.config.arms['right'].world_from_base @ pose_matrix(runtime.models['right'].forward(joints))
    source,orientation=nearest_grasp_orientation(source,reference[:3,:3],prefer_ccw=True)
    save(folder/'orientation.json',orientation)
    # Reuse native height adjustment exactly once; never add a trial Z patch.
    source=runtime.adjust_motion(source,folder)
    if runtime.config.raw.get('astra_grasp_height',{}).get('mode')=='shared_arm7_target_z':
        from cloth_agent.codex_grasp_height import share_arm7_grasp_z
        source,changes=share_arm7_grasp_z(source)
        save(folder/'shared_height_policy.json',changes)
    grasp=list(parse_code(source,allow_grasp=True)[0][1]['right'])
    lifted=grasp.copy();lifted[2]+=LIFT_MM/1000
    source=f'motion.grasp(right={grasp!r})\nmotion.move_tcp(right={lifted!r})\n'
    (folder/'acquire_motion.py').write_text(source)
    return source,grasp,lifted


def end_joints(operations):
    kind,data=operations[-1]
    return {k:np.asarray(v).copy() for k,v in (data.end if kind=='move' else data['joints']).items()}


def plan_trial(runtime, before, source, grasp):
    """Reuse the standard grasp prefix and certify the entire put-back first."""
    initial={k:np.asarray(v) for k,v in before.meta['joints_deg'].items()}
    operations=runtime.plan(source,initial,preserve_command_boundaries=True)
    grippers=[(i,d['action']) for i,(kind,d) in enumerate(operations) if kind=='gripper']
    staged=getattr(runtime,'staged_release',False)
    expected=['open','close','release_partial','open'] if staged else ['open','close','open']
    if [a for _,a in grippers]!=expected:
        raise ValueError('Unexpected native grasp skeleton; no trial dispatched')
    close=grippers[1][0];release=grippers[2][0]
    approach=operations[:close];closing=operations[close:close+1];lift=operations[close+1:release]
    if not approach or not lift:raise ValueError('Native grasp lacks approach/lift')
    lifted=end_joints(lift)
    replace=runtime.plan(f'motion.move_tcp(right={grasp!r})\n',lifted,preserve_command_boundaries=True)
    placed=end_joints(replace)
    if staged:
        from cloth_agent.codex_grasp_task import _grippers
        from cloth_agent.dual_arm.single_arm_backend import validate_release_delta
        delta=runtime.release_open_delta_pulse
        validate_release_delta(runtime.config.arms['right'].gripper,delta)
        opening=_grippers(runtime,placed,('right',),'release_partial',opening_delta_pulse=delta)
    else:
        opening=runtime.plan('motion.gripper(arm="right",action="open")\n',placed,preserve_command_boundaries=True)
    retreat,home=retreat_and_home(runtime,placed,complete_release=staged)
    return [('pregrasp_descent',approach),('close',closing),('lift',lift),
        ('replace',replace),('release_retreat',opening+retreat),('return_home',home)]


def hold_photo(runtime, folder, hold_s):
    """Direct camera read only. In particular, no open/Home/perception operation."""
    folder.mkdir(exist_ok=False)
    before=runtime.coordinator.snapshots()
    deadline=time.monotonic()+hold_s
    while True:
        rgb,depth=runtime.camera.read()
        if time.monotonic()>=deadline:break
    after=runtime.coordinator.snapshots()
    for arm in before:
        if np.max(np.abs(np.asarray(before[arm]['joints'])-after[arm]['joints']))>runtime.config.limits['stationary_tolerance_deg']:
            raise RuntimeError('Robot moved while taking hold image')
    Image.fromarray(rgb).save(folder/'rgb.png')
    np.save(folder/'raw_depth_m.npy',depth,allow_pickle=False)
    save(folder/'capture.json',dict(captured_at=datetime.now(timezone.utc).isoformat(),
        capture_mode='DIRECT_CAMERA_READ_NO_MOTION_NO_HEIGHT_PATCH',
        intrinsics=runtime.camera.intrinsics.tolist(),snapshot_before=before,snapshot_after=after))


def dispatch_trial(runtime, phases, folder, report, hold_s, photo_fn=hold_photo):
    """No phase is replanned/retried. Camera-only failure still allows planned placement."""
    capture_error=None
    for name,operations in phases:
        directory=folder/name;directory.mkdir()
        started=time.monotonic()
        if name=='return_home':
            with home_speed(runtime):
                result=runtime._execute_operations(operations,directory)
        else:
            result=runtime._execute_operations(operations,directory)
        row=dict(phase=name,result=result,elapsed_s=time.monotonic()-started)
        report['phases'].append(row)
        save(folder/'trial.json',report)
        if result.get('status')!='EXECUTED':
            raise RuntimeError(f'{name}: {result.get("error",result.get("status"))}')
        row['snapshot']=runtime.coordinator.snapshots()
        if name=='lift':
            try:
                photo_fn(runtime,folder/'lift_observation',hold_s)
                shutil.copyfile(folder/'lift_observation/rgb.png',folder/'lift.png')
            except Exception as exc:
                capture_error=f'{type(exc).__name__}: {exc}'
                report['lift_capture_error']=capture_error
        if name=='return_home':report['returned_safely']=True
        save(folder/'trial.json',report)
    return capture_error


def set_label(folder, label):
    if label not in LABELS:raise ValueError('Label must be HELD/MISS/UNCERTAIN')
    folder=Path(folder);report=json.loads((folder/'trial.json').read_text())
    if not report.get('returned_safely'):raise ValueError('Trial has not completed safe placement')
    if report.get('operator_label') is not None:raise ValueError('Human label already saved; do not overwrite')
    report.update(operator_label=label,labelled_at=datetime.now(timezone.utc).isoformat())
    save(folder/'trial.json',report)
    with (folder.parent/'labels.csv').open('a+',newline='') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        stream.seek(0,2);empty=stream.tell()==0
        writer=csv.writer(stream)
        if empty:writer.writerow(['trial','label','labelled_at'])
        writer.writerow([folder.name,label,report['labelled_at']])


def run(args):
    # All hardware imports/initialization are confined to the explicit execute branch.
    from cloth_agent.codex_arm7_runtime import Arm7Runtime
    from cloth_agent.dual_arm.observation import create_camera
    folder=allocate_trial(args.output)
    report=dict(protocol='grasp-acquire-baseline-v1',trial=folder.name,status='STARTED',
        started_at=datetime.now(timezone.utc).isoformat(),lift_mm=LIFT_MM,hold_s=args.hold_s,
        automatic_retry=False,folding=False,phases=[],operator_label=None,returned_safely=False)
    save(folder/'trial.json',report);print(f'Trial: {folder}',flush=True)
    runtime=None;started=time.monotonic()
    try:
        runtime=Arm7Runtime(args.config)
        runtime.auto_grasp_debug=False;runtime.auto_grasp_branch=None
        report['speed']=low_speed(runtime)
        save(folder/'runtime_config.json',runtime.config.raw)
        runtime.camera=create_camera(runtime.config)
        before=runtime.observe(folder/'before')
        np.save(folder/'before/intrinsics.npy',runtime.camera.intrinsics,allow_pickle=False)
        shutil.copyfile(before.image,folder/'before.png')
        pa=select_pa(before.image)
        report['Pa']=pa
        from PIL import ImageDraw
        marked=Image.open(before.image).convert('RGB')
        draw=ImageDraw.Draw(marked);x,y=pa
        draw.line((x-10,y,x+10,y),fill='red',width=2)
        draw.line((x,y-10,x,y+10),fill='red',width=2)
        draw.text((x+12,y),f'Pa {pa}',fill='red')
        marked.save(folder/'before_pa.png')
        save(folder/'trial.json',report)
        measurement=measure_pa(before,pa,runtime.config.limits['max_surface_spread_mm'])
        try:measurement['A']=capture_a(runtime,args.config,before,pa,folder/'camera_A')
        except Exception as exc:
            measurement['A'].update(reason=f'{type(exc).__name__}: {exc}',used_for_command=False)
        report['measurement']=measurement
        save(folder/'measurement.json',measurement)
        if measurement['sample_error']:
            raise ValueError('Existing depth sampler rejected Pa: '+measurement['sample_error'])
        source,grasp,lifted=prepare_source(runtime,before,measurement['surface_xyz_mm'],args.yaw_deg,folder)
        measurement.update(command_grasp_z_mm=grasp[2]*1000,command_grasp_world_m_rad=grasp,
            command_lift_world_m_rad=lifted,existing_grasp_offset_mm=runtime.grasp_offset_mm)
        save(folder/'measurement.json',measurement)
        before.validate_for(runtime.config,live=True,check_age=False)
        phases=plan_trial(runtime,before,source,grasp)
        save(folder/'certified_phases.json',[dict(phase=name,operations=[dict(kind=k,data=d.to_dict() if k=='move' else d)
            for k,d in ops]) for name,ops in phases])
        runtime.arrival({k:np.asarray(v) for k,v in before.meta['joints_deg'].items()})
        report['status']='EXECUTING';save(folder/'trial.json',report)
        capture_error=dispatch_trial(runtime,phases,folder,report,args.hold_s)
        report['status']='COMPLETED_CAPTURE_FAILED' if capture_error else 'COMPLETED'
    except BaseException as exc:
        report.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',
            error=f'{type(exc).__name__}: {exc}')
        if runtime is not None:report['stop_results']=runtime.stop()
    finally:
        if runtime is not None:
            try:runtime.close()
            except Exception as exc:report['cleanup_error']=f'{type(exc).__name__}: {exc}'
        report['elapsed_s']=time.monotonic()-started
        report['finished_at']=datetime.now(timezone.utc).isoformat()
        report['gripper_feedback_files']=[str(p.relative_to(folder)) for p in folder.glob('*/gripper_*.json')]
        save(folder/'trial.json',report)
    print(f"{report['status']}: {folder}",flush=True)
    if report.get('error'):print(report['error'],flush=True)
    if report['returned_safely']:
        print('已放回并退出运动。请根据试抬时的实际观察填写；不确定请选 UNCERTAIN。',flush=True)
        print(f'抓点图：{folder/"before_pa.png"}\n试抬图：{folder/"lift.png"}',flush=True)
        try:
            while True:
                label=input('结果 HELD / MISS / UNCERTAIN：').strip().upper()
                if label in LABELS:set_label(folder,label);break
                print('请输入 HELD、MISS 或 UNCERTAIN。')
        except (EOFError,KeyboardInterrupt):
            print(f'人工标签尚未填写。稍后用 label --trial {folder} --value HELD|MISS|UNCERTAIN 补录。')
    return 0 if report['status']=='COMPLETED' else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    live=sub.add_parser('run',help='one manually selected Pa and one acquisition trial')
    live.add_argument('--execute',action='store_true')
    live.add_argument('--output',type=Path,default=ROOT/'runs/grasp_acquire_baseline')
    live.add_argument('--config',type=Path,default=ROOT/'config/dual_arm.local.json')
    live.add_argument('--yaw-deg',type=float,default=0.,help='world yaw; vertical tool, existing jaw-symmetry choice retained')
    live.add_argument('--hold-s',type=float,default=2.)
    batch=sub.add_parser('batch',help='fixed-XY Z sweep; manual garment reset and Enter between trials')
    batch.add_argument('--execute',action='store_true')
    batch.add_argument('--output',type=Path,default=ROOT/'runs/grasp_acquire_baseline')
    batch.add_argument('--config',type=Path,default=ROOT/'config/dual_arm.local.json')
    batch.add_argument('--yaw-deg',type=float,default=0.)
    batch.add_argument('--schedule',choices=('compare9','compare6','shakedown','balanced12'),default='compare9',
        help='default compare9: 0/-2/-4 mm, three trials each, shuffled; confirm after lift and reset between trials')
    batch.add_argument('--random-seed',type=int,default=0)
    batch.add_argument('--shakedown',type=Path,help='completed shakedown with manual PASS review')
    review=sub.add_parser('review-shakedown',help='offline operator review, required before balanced12')
    review.add_argument('--session',type=Path,required=True)
    review.add_argument('--result',choices=('PASS','FAIL'),required=True)
    label=sub.add_parser('label',help='offline human label entry after safe placement')
    label.add_argument('--trial',type=Path,required=True);label.add_argument('--value',choices=LABELS,required=True)
    args=parser.parse_args(argv)
    if args.command=='label':set_label(args.trial,args.value);return 0
    if args.command=='review-shakedown':
        from .grasp_acquire_batch import review_shakedown
        review_shakedown(args.session,args.result);return 0
    if args.command=='batch':
        from . import grasp_acquire_batch
        if not math.isfinite(args.yaw_deg):parser.error('yaw must be finite')
        try:offsets=grasp_acquire_batch.schedule(args.schedule,args.random_seed,args.shakedown)
        except (ValueError,OSError) as exc:parser.error(str(exc))
        if not args.execute:
            print(f'No robot/camera initialized. Batch offsets (mm): {offsets}. Add --execute to start.');return 0
        if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            parser.error('Pa selection needs a desktop display.')
        return grasp_acquire_batch.run(args)
    if not math.isfinite(args.yaw_deg) or not 0<=args.hold_s<=5:
        parser.error('yaw must be finite; hold-s must be in [0,5]')
    if not args.execute:
        print('No robot/camera initialized. Add --execute to run one manually selected Pa trial.');return 0
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        parser.error('Pa selection needs a desktop display. Run this command in the robot workstation desktop terminal.')
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
