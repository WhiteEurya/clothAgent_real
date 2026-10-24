"""Codex-guided central cloth pickup and gravity-drop reset for Arm6."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import secrets
import time
import numpy as np
from PIL import Image, ImageDraw

from .dual_arm.geometry import DualArmError, matrix_pose, pose_matrix
from .gamepad_collect import (save_json, approach_program, tcp_command,
                             execute_program, TrajectoryRecorder)
from .experiments.discovery_schema import obj, S, N, enum

POINT = {'type':'array','items':{'type':'integer','minimum':0},'minItems':2,'maxItems':2}
REGION = obj(center_px=POINT, radius_px={'type':'number','minimum':0}, confidence=N, reason=S)
PROMPT = '''Inspect the supplied CURRENT RGB of a garment on a work surface. Locate a visible,
solid fabric patch near the center of the garment body for lifting the cloth and dropping it.
Return center_px in original image pixels and radius_px for the largest reliable circular sampling
region ENTIRELY on visible cloth. The whole circle must avoid neckline/cuff openings, table,
cloth boundary, gripper, people, and occlusions. Prefer body fabric near the garment center,
not sleeves or a hem corner. This is an area for sampling a grasp, not a cloth bounding circle.
Prefer a broad body region to diversify grasps; shrink only to avoid unsafe pixels; do not invent hidden fabric. If no reliable central patch is
visible, return confidence below 0.8 and radius_px=0. Explain visible evidence and uncertainty.
Do not output robot commands. No desired folded result is assumed.'''


def locate_region(image, folder, *, backend=None):
    from .experiments.discovery_model import CodexMultimodal
    from .experiments.graph_policy import Requests
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    requests=Requests(folder/'model', backend or CodexMultimodal(), max_calls=2,
        schemas={'cloth_center':REGION}, prompts={'cloth_center':PROMPT},
        common='Return schema-valid JSON based only on the attached image.', protocol='arm6-random-fold-v2')
    with Image.open(image) as im: size=im.size
    region=requests.ask('cloth_center',dict(image_width=size[0],image_height=size[1]),[('current',image)])
    save_json(folder/'region.json',region)
    save_json(folder/'model_metrics.json',requests.metrics())
    validate_region(region,size)
    return region


def validate_region(region, size):
    center=np.asarray(region['center_px'],dtype=float);radius=float(region['radius_px'])
    confidence=float(region['confidence'])
    if (center.shape!=(2,) or not np.isfinite(center).all() or not np.isfinite(radius)
            or not .8 <= confidence <= 1 or radius <= 0):
        raise DualArmError('Codex 未找到可信的衣服中心抓取区域')
    if any(center-radius < 2) or any(center+radius >= np.asarray(size)-2):
        raise DualArmError('中心采样区域超出照片边界')


def sample_grasp(observation, region, max_spread, radius_mm, seed):
    with Image.open(observation.image) as im: validate_region(region,im.size)
    if not np.isfinite(radius_mm) or not 0 < radius_mm <= 80:
        raise ValueError('random center radius must be in (0,80] mm')
    center=np.asarray(region['center_px'],dtype=float)
    origin=observation.sample(center.astype(int).tolist(),max_spread)
    rng=np.random.default_rng(seed)
    # Uniform disk sampling in image coordinates, bounded again in measured world XY.
    for _ in range(200):
        radius=region['radius_px']*np.sqrt(rng.random());angle=rng.uniform(0,2*np.pi)
        pixel=np.rint(center+radius*np.array([np.cos(angle),np.sin(angle)])).astype(int)
        if np.linalg.norm(pixel-center)>region['radius_px']:continue
        try: xyz=observation.sample(pixel.tolist(),max_spread)
        except DualArmError:continue
        if np.linalg.norm(np.asarray(xyz)[:2]-np.asarray(origin)[:2])<=radius_mm:
            return dict(pixel_xy=pixel.tolist(),world_xyz_mm=np.asarray(xyz).tolist(),
                        center_world_xyz_mm=np.asarray(origin).tolist(),seed=int(seed),radius_mm=radius_mm)
    raise DualArmError('中心区域内未找到有效深度的随机抓取点；不执行重置')


def validate_randomization(lift_height, height_jitter, yaw_range, drop_radius, max_retries):
    if (not np.isfinite([lift_height,height_jitter,yaw_range,drop_radius]).all()
            or not 50 < lift_height <= 300 or not 0 <= height_jitter < lift_height-50
            or not 0 <= yaw_range <= 90 or not 0 <= drop_radius <= 60
            or isinstance(max_retries,bool) or not isinstance(max_retries,int)
            or not 0 <= max_retries <= 2):
        raise ValueError('reset requires lift (50,300], jitter [0,lift-50), yaw [0,90], drop radius [0,60], retries 0..2')


def sample_release(seed, lift_height, height_jitter, yaw_range, drop_radius):
    rng=np.random.default_rng(seed)
    height=float(rng.uniform(lift_height-height_jitter,lift_height))
    yaw=float(rng.uniform(-yaw_range,yaw_range))
    radius=drop_radius*np.sqrt(rng.random());angle=rng.uniform(0,2*np.pi)
    return dict(lift_height=height,yaw_delta=yaw,
                drop_offset_xy=(radius*np.array([np.cos(angle),np.sin(angle)])).tolist())


def compare_reset(before, after, folder, *, backend=None):
    from .experiments.discovery_model import CodexMultimodal
    from .experiments.graph_policy import Requests
    schema=obj(change=enum('CHANGED','TOO_SIMILAR','UNCERTAIN'),confidence=N,reason=S)
    prompt=('Compare the garment BEFORE and AFTER a pickup/drop reset. Assess changes in folds, '
            'overlaps, exposed parts and silhouette after accounting for rigid translation and rotation. '
            'Translation or rotation alone does not count as a new folding state. '
            'Return TOO_SIMILAR only when clearly nearly unchanged; CHANGED for clear fold changes, '
            'otherwise UNCERTAIN. Do not infer successful grasp from robot commands.')
    requests=Requests(Path(folder)/'comparison_model',backend or CodexMultimodal(),max_calls=2,
                      schemas={'compare':schema},prompts={'compare':prompt},common='Use only visible evidence.',
                      protocol='arm6-reset-compare-v1')
    result=requests.ask('compare',{},[('before',before),('after',after)])
    save_json(Path(folder)/'comparison.json',result)
    save_json(Path(folder)/'comparison_metrics.json',requests.metrics())
    return result


def reset_program(runtime, xyz, rpy, *, grasp_offset=-15., lift_height=200., approach_height=50.,
                  yaw_delta=0., drop_offset_xy=(0.,0.)):
    xyz=np.asarray(xyz,dtype=float);rpy=np.asarray(rpy,dtype=float)
    if (xyz.shape!=(3,) or rpy.shape!=(3,) or not np.isfinite(np.r_[xyz,rpy]).all()
            or not -30 <= grasp_offset <= 10 or not 0 < approach_height <= 200
            or not max(grasp_offset,approach_height) < lift_height <= 300):
        raise ValueError('invalid random reset geometry')
    offset=np.asarray(drop_offset_xy,dtype=float)
    if (offset.shape!=(2,) or not np.isfinite(offset).all() or np.linalg.norm(offset)>60.000001
            or not np.isfinite(yaw_delta) or abs(yaw_delta)>90):
        raise ValueError('invalid randomized release geometry')
    above=np.r_[xyz+[0,0,approach_height],rpy]
    grasp=np.r_[xyz+[0,0,grasp_offset],rpy]
    lifted=np.r_[xyz+[0,0,lift_height],rpy]
    arm=runtime.config.arms['left']
    home=matrix_pose(arm.world_from_base @ pose_matrix(runtime.models['left'].forward(arm.home_joints)))
    rotated=lifted.copy();rotated[5]=(rotated[5]+yaw_delta+180)%360-180
    drop=rotated.copy();drop[:2]=home[:2]+offset
    if not np.isfinite(drop).all():raise ValueError('invalid home XY drop target')
    return (approach_program(runtime,above)+tcp_command(grasp)
            +'motion.gripper(arm="left", action="close")\n'+tcp_command(lifted)
            +(tcp_command(rotated) if yaw_delta else '')+tcp_command(drop)
            +'motion.gripper(arm="left", action="open")\n')


def preview(image, region, point, destination):
    with Image.open(image) as raw: im=raw.convert('RGB')
    draw=ImageDraw.Draw(im);x,y=region['center_px'];r=region['radius_px']
    draw.ellipse((x-r,y-r,x+r,y+r),outline='cyan',width=3)
    x,y=point['pixel_xy'];draw.ellipse((x-7,y-7,x+7,y+7),outline='red',width=3)
    im.save(destination)


def run_reset(runtime, folder, *, radius_mm=80., lift_height=300., grasp_offset=-15.,
              seed=None, backend=None, recorder=None, height_jitter=50., yaw_range=45.,
              drop_radius=40., max_retries=1):
    validate_randomization(lift_height,height_jitter,yaw_range,drop_radius,max_retries)
    if not np.isfinite(radius_mm) or not 0 < radius_mm <= 80:
        raise ValueError('random center radius must be in (0,80] mm')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    seed=secrets.randbits(63) if seed is None else seed
    report=dict(status='PREPARING',kind='random_fold_reset',seed=seed,lift_height_mm=lift_height,
                radius_mm=radius_mm,grasp_offset_mm=grasp_offset,not_a_demonstration=True,release_location='random_offset_from_home_xy',
                height_jitter_mm=height_jitter,yaw_range_deg=yaw_range,drop_radius_mm=drop_radius,
                max_retries=max_retries,attempts=[])
    owns_recorder=recorder is None;started=time.monotonic()
    save_json(folder/'result.json',report)
    try:
        coordinator=getattr(runtime,'coordinator',None)
        if coordinator is not None and owns_recorder:
            with coordinator.log_lock:
                coordinator.directory=folder/'controller';coordinator.directory.mkdir()
        before=runtime.observe(folder/'before')
        before.validate_for(runtime.config,live=True,check_age=False)
        if owns_recorder:
            recorder=TrajectoryRecorder(runtime,folder);recorder.start()
        seeds=np.random.SeedSequence(seed).spawn(max_retries+1)
        for index,sequence in enumerate(seeds):
            if runtime.cancel.is_set():raise KeyboardInterrupt('reset cancelled')
            attempt=folder if index==0 else folder/f'retry_{index:02d}'
            attempt.mkdir(parents=True,exist_ok=True)
            before.validate_for(runtime.config,live=True,check_age=False)
            region=locate_region(before.image,attempt,backend=backend)
            if runtime.cancel.is_set():raise KeyboardInterrupt('reset cancelled')
            grasp_seed,release_seed=[int(v) for v in sequence.generate_state(2,dtype=np.uint64)]
            point=sample_grasp(before,region,runtime.config.limits['max_surface_spread_mm'],radius_mm,grasp_seed)
            release=sample_release(release_seed,lift_height,height_jitter,yaw_range,drop_radius)
            save_json(attempt/'grasp.json',point);preview(before.image,region,point,attempt/'grasp_preview.png')
            save_json(attempt/'release.json',dict(seed=release_seed,**release))
            rpy=matrix_pose(runtime.config.arms['left'].world_from_base @
                pose_matrix(runtime.models['left'].forward(runtime.config.arms['left'].home_joints)))[3:]
            source=reset_program(runtime,point['world_xyz_mm'],rpy,grasp_offset=grasp_offset,**release)
            recorder.event('random_reset',attempt=index,release=release,**point)
            entry=dict(index=index,directory=str(attempt.relative_to(folder)),release=release)
            report['attempts'].append(entry)
            save_json(folder/'result.json',report)
            # Entire lift, rotation and transfer is planned before the first physical command.
            execute_program(runtime,recorder,attempt,'lift_and_drop',source)
            entry['executed']=True
            recorder.stage('settle')
            if runtime.cancel.wait(2.):raise KeyboardInterrupt('reset cancelled while settling')
            recorder.check();recorder.stage('return_perception')
            runtime.move_to_perception(attempt/'return_perception')
            recorder.check();recorder.stage('after_observation')
            after=runtime.capture_current(attempt/'after')
            recorder.check()
            report['final_after']=str((attempt/'after/rgb.png').relative_to(folder))
            recorder.stage('compare_random_reset')
            comparison=compare_reset(before.image,after.image,attempt,backend=backend)
            entry['comparison']=comparison
            if runtime.cancel.is_set():raise KeyboardInterrupt('reset cancelled during comparison')
            recorder.check()
            similar=comparison['change']=='TOO_SIMILAR' and comparison['confidence']>=.8
            report.update(status='EXECUTED',outcome=(
                'TOO_SIMILAR' if similar else 'CHANGED' if comparison['change']=='CHANGED'
                and comparison['confidence']>=.8 else 'UNCERTAIN'))
            save_json(folder/'result.json',report)
            if not similar or index==max_retries:break
            print('落下后的折叠形态变化较小，重新采样并重置一次。',flush=True)
            before=after
    except BaseException as exc:
        report.update(status='ABORTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error=f'{type(exc).__name__}: {exc}')
        runtime.cancel.set()
        try:runtime.stop()
        except Exception as stop_exc:report['stop_error']=str(stop_exc)
        raise
    finally:
        if recorder is not None:
            if owns_recorder:recorder.close()
            report['samples']=recorder.samples
            if recorder.error:report.update(status='FAILED',recording_error=recorder.error)
        report['elapsed_s']=time.monotonic()-started
        save_json(folder/'result.json',report)
    if recorder is not None:recorder.check()
    return report


def reset_choice():
    """An independent explicit 'unfold complete' control, not save confirmation."""
    import tkinter as tk
    root=tk.Tk();root.title('展开采集：下一步');choice=[]
    tk.Label(root,text='本条轨迹已处理。衣服是否已经展开完成？',padx=30,pady=20).pack()
    def choose(value):choice.append(value);root.destroy()
    for label,value in [('继续展开（下一条轨迹）','continue'),
                        ('展开完成：随机抓起并落下','reset'),('结束采集','finish')]:
        tk.Button(root,text=label,command=lambda v=value:choose(v),width=38,pady=8).pack(padx=20,pady=5)
    root.protocol('WM_DELETE_WINDOW',lambda:choose('finish'));root.mainloop()
    return choice[0]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,
                        help='evidence directory; default reports/random_reset_<UTC timestamp>')
    parser.add_argument('--image',type=Path,help='only test Codex center detection on a saved image; no hardware')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--config',type=Path,default=Path('config/dual_arm.local.json'))
    parser.add_argument('--radius-mm',type=float,default=80.)
    parser.add_argument('--lift-height-mm',type=float,default=300.)
    parser.add_argument('--grasp-offset-mm',type=float,default=-15.)
    parser.add_argument('--height-jitter-mm',type=float,default=50.,help='sample height from lift-height minus jitter up to lift-height')
    parser.add_argument('--yaw-range-deg',type=float,default=45.)
    parser.add_argument('--drop-radius-mm',type=float,default=40.)
    parser.add_argument('--max-retries',type=int,default=1)
    args=parser.parse_args(argv)
    try:validate_randomization(args.lift_height_mm,args.height_jitter_mm,args.yaw_range_deg,args.drop_radius_mm,args.max_retries)
    except ValueError as exc:parser.error(str(exc))
    if args.output is None:
        args.output=Path('reports')/('random_reset_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if args.image:
        if args.execute:parser.error('--image cannot be used with --execute')
        args.output.mkdir(parents=True,exist_ok=False)
        region=locate_region(args.image,args.output)
        preview(args.image,region,dict(pixel_xy=region['center_px']),args.output/'center_preview.png')
        print(region);return
    if not args.execute:parser.error('use --image for offline vision or --execute for one physical reset')
    if not (0 < args.radius_mm <= 80 and 50 < args.lift_height_mm <= 300 and -30 <= args.grasp_offset_mm <= 10):
        parser.error('invalid reset parameters')
    if args.output.exists():parser.error('output already exists')
    print(f'单次 Arm6 随机重置：抬升 {args.lift_height_mm:g} mm，中心采样范围 {args.radius_mm:g} mm',flush=True)
    print(f'记录目录：{args.output.resolve()}',flush=True)
    from .gamepad_collect_runtime import CollectionRuntime
    runtime=CollectionRuntime(args.config)
    try:
        runtime.config.home_recover_stopped=True
        result=run_reset(runtime,args.output,radius_mm=args.radius_mm,lift_height=args.lift_height_mm,
                  grasp_offset=args.grasp_offset_mm,height_jitter=args.height_jitter_mm,
                  yaw_range=args.yaw_range_deg,drop_radius=args.drop_radius_mm,max_retries=args.max_retries)
        print(f'测试执行完成。请查看落下效果：{(args.output/result["final_after"]).resolve()}',flush=True)
        print(f'抓取点预览：{(args.output/"grasp_preview.png").resolve()}',flush=True)
    finally:runtime.close()


if __name__=='__main__':main()
