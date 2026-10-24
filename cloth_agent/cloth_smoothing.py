"""Bounded, image-feedback smoothing using Arm6's existing pick/place executor."""
from pathlib import Path
import time
import shutil
import json
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from .dual_arm.geometry import DualArmError
from .experiments.discovery_schema import obj, S, N, enum
from .gamepad_collect import save_json, automatic_program, execute_program
from .grasp_angle import transform_for, image_angle_to_rpy, draw_preview

POINT={'type':'array','items':{'type':'integer','minimum':0},'minItems':2,'maxItems':2}
SCHEMA=obj(decision=enum('ACT','DONE','STOP'),assessment=S,reason=S,confidence=N,
           progress=enum('INITIAL','IMPROVED','NO_PROGRESS','WORSE','SLIPPED','UNCERTAIN'),
           grasp_px=POINT,target_px=POINT,
           opening_angle_image_deg={'type':'number','minimum':0,'maximum':180},expected_change=S)
PROMPT='''你的目标是把当前衣服做 smooth，让布面更平整、褶皱更少。
根据当前照片选择下一步最有帮助的动作；如果提供了动作前照片，就结合前后变化调整方案。
你可控制 Arm6 夹爪抓取、低抬、移动和放下。输出原图上的抓取点、目标点及夹爪开合方向。
开合方向角度以图像向右为 0°、顺时针计，模 180°。程序负责实测深度转换和 IK 执行。
需要继续整理时返回 ACT，已经足够平整时返回 DONE，无法提出下一步时返回 STOP。
按给定 JSON 格式回复，简要说明判断与预期效果。DONE/STOP 的点位填 [0,0]、角度填 0。
'''



def observation_context(observation,grasp_offset):
    """Small factual geometry payload; actual selected pixels are still checked later."""
    xyz=np.asarray(observation.xyz)
    valid=np.isfinite(xyz).all(axis=2)
    h,w=valid.shape
    samples=[]
    for y in np.linspace(0,h-1,7,dtype=int):
        for x in np.linspace(0,w-1,9,dtype=int):
            if valid[y,x]:
                samples.append(dict(pixel_xy=[int(x),int(y)],world_xyz_mm=np.round(xyz[y,x],1).tolist()))
    scales={}
    for name,axis in [('image_x',1),('image_y',0)]:
        delta=np.linalg.norm(np.diff(xyz[...,:2],axis=axis),axis=2)
        values=delta[np.isfinite(delta)&(delta>.05)&(delta<5)]
        scales[name]=float(np.median(values)) if len(values) else None
    return dict(image_size_px=[w,h],measured_depth_available=bool(valid.any()),
        valid_depth_fraction=float(valid.mean()),approx_world_xy_mm_per_pixel=scales,
        sparse_measured_geometry=samples,
        configured_motion=dict(grasp_offset_mm=grasp_offset,lift_height_mm=15.,place_offset_mm=5.,
                               min_translation_mm=10.,max_translation_mm=25.,max_surface_height_difference_mm=5.),
        host_checks=['exact selected 5x5 measured depth patches','surface spread','metric displacement',
                     'height difference','calibrated image-angle conversion','full IK before dispatch'],
        note='Sparse samples and global median scale are approximate context, not replacement for exact point validation.')


def prepare_step(runtime,observation,proposal,rpy,grasp_offset,folder):
    observation.validate_for(runtime.config,live=True,check_age=False)
    with Image.open(observation.image) as im:width,height=im.size
    points=[]
    for name,key in [('start','grasp_px'),('target','target_px')]:
        pixel=proposal[key]
        if len(pixel)!=2 or not (2<=pixel[0]<width-2 and 2<=pixel[1]<height-2):
            raise DualArmError('展平点位在图像边缘或超出范围')
        xyz=observation.sample(pixel,runtime.config.limits['max_surface_spread_mm'])
        points.append(dict(name=name,pixel_xy=pixel,world_xyz_mm=np.asarray(xyz).tolist()))
    save_json(folder/'points.json',points)
    start,target=(np.asarray(p['world_xyz_mm']) for p in points)
    distance=float(np.linalg.norm(target[:2]-start[:2]))
    if not 10<=distance<=25:raise DualArmError(f'局部去皱位移 {distance:.1f} mm 不在 10–25 mm 范围内')
    if abs(float(target[2]-start[2]))>5:
        raise DualArmError('局部去皱两点表面高度差超过 5 mm，不跨层搬运')
    k=np.asarray(runtime.camera.intrinsics,dtype=float);world=transform_for(observation)
    axis=Rotation.from_euler('xyz',runtime.config.arms['left'].tcp_offset[3:],degrees=True).inv().apply([0,1,0])
    chosen,delta=image_angle_to_rpy(proposal['opening_angle_image_deg'],proposal['grasp_px'],start,rpy,k,world,axis)
    save_json(folder/'points.json',points)
    save_json(folder/'geometry.json',dict(rpy_world_deg=chosen.tolist(),yaw_delta_deg=delta,
                                         distance_mm=distance,lift_height_mm=15.,grasp_offset_mm=grasp_offset))
    draw_preview(observation.image,points,chosen,k,world,axis,folder/'preview.png')
    return automatic_program(runtime,start,target,chosen,approach_height=50.,grasp_offset=grasp_offset,
                             lift_height=15.,clearance=30.,place_offset=5.)


def smooth_cloth(runtime,recorder,observation,folder,rpy,*,max_steps=3,grasp_offset=-15.,backend=None):
    """Execute only after the operator selects Codex smoothing; never retry motion."""
    if not isinstance(max_steps,int) or not 1<=max_steps<=6:raise ValueError('smoothing steps must be 1..6')
    if not -30<=grasp_offset<=10:raise ValueError('invalid smoothing grasp offset')
    from .experiments.graph_policy import Requests
    from .experiments.discovery_model import CodexMultimodal
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    requests=Requests(folder/'model',backend or CodexMultimodal(timeout=180),max_calls=4*(max_steps+1),
        schemas={'smooth':SCHEMA},prompts={'smooth':PROMPT},
        common='让衣服更平整。根据照片规划下一步，按 JSON 格式返回。',
        protocol='arm6-smoothing-goal-v4')
    shutil.copyfile(observation.image,folder/'initial.png')
    report=dict(status='RUNNING',completed_actions=0,max_steps=max_steps,steps=[],
                initial_observation='initial.png',grasp_offset_mm=grasp_offset)
    started=time.monotonic();previous=None;last_action=None
    save_json(folder/'result.json',report)
    try:
        for index in range(max_steps+1):
            if runtime.cancel.is_set():raise KeyboardInterrupt('smoothing cancelled')
            recorder.check();step=folder/f'step_{index:04d}';step.mkdir()
            recorder.stage('smoothing_observe_and_plan')
            images=[('current',observation.image)]
            if previous is not None:images.append(('previous_before',previous.image))
            t=time.monotonic()
            proposal=requests.ask('smooth',dict(last_action=last_action,remaining_actions=max_steps-index,
                observation_context=observation_context(observation,grasp_offset)),images)
            model_s=time.monotonic()-t
            save_json(step/'proposal.json',proposal)
            shutil.copyfile(observation.image,step/'before.png')
            entry=dict(index=index,proposal=proposal,model_s=model_s,before=str((step/'before.png').relative_to(folder)))
            report['steps'].append(entry)
            if runtime.cancel.is_set():raise KeyboardInterrupt('smoothing cancelled during model call')
            recorder.check()
            if proposal['confidence']<.8:
                report['status']='UNCERTAIN';break
            if previous is not None and proposal['progress'] in ('NO_PROGRESS','WORSE','SLIPPED','UNCERTAIN','INITIAL'):
                report['status']='NO_PROGRESS' if proposal['progress']=='NO_PROGRESS' else 'REASSESSMENT_STOP';break
            if proposal['decision']!='ACT':
                report['status']='MODEL_REPORTED_DONE' if proposal['decision']=='DONE' else 'MODEL_STOP';break
            if index==max_steps:report['status']='STEP_LIMIT';break
            recorder.stage('smoothing_plan')
            source=None
            for attempt in range(2):
                if runtime.cancel.is_set():raise KeyboardInterrupt('smoothing cancelled before planning')
                recorder.check()
                try:
                    source=prepare_step(runtime,observation,proposal,rpy,grasp_offset,step)
                    break
                except (DualArmError,ValueError) as exc:
                    entry['validation_error']=str(exc)
                    entry.setdefault('geometry_rejections',[]).append(dict(proposal=proposal,error=str(exc)))
                    save_json(folder/'result.json',report)
                    print('几何检查：'+str(exc),flush=True)
                    if attempt==1:
                        report['status']='GEOMETRY_REJECTED';break
                    recorder.stage('smoothing_replan_geometry')
                    t=time.monotonic()
                    measured=json.loads((step/'points.json').read_text()) if (step/'points.json').exists() else None
                    proposal=requests.ask('smooth',dict(last_action=last_action,remaining_actions=max_steps-index,
                        observation_context=observation_context(observation,grasp_offset),
                        correction=dict(rejected_proposal=proposal,measured_points=measured,error=str(exc),
                          instruction='尚未执行动作。请根据实测结果修正一次像素点位，仍以 smooth 为目标；无法修正则 STOP。')),images)
                    model_s+=time.monotonic()-t;entry.update(model_s=model_s,proposal=proposal)
                    save_json(step/'revised_proposal.json',proposal)
                    if runtime.cancel.is_set():raise KeyboardInterrupt('smoothing cancelled during geometric replan')
                    recorder.check()
                    if proposal['decision']!='ACT' or proposal['confidence']<.8:
                        report['status']='GEOMETRY_REJECTED';break
            if source is None:break
            entry.pop('validation_error',None)
            t=time.monotonic()
            execute_program(runtime,recorder,step,'automatic_pick_place',source)
            entry['action_s']=time.monotonic()-t;report['completed_actions']+=1
            recorder.stage('smoothing_return_perception')
            runtime.open_at_home_on_return=True
            try:runtime.move_to_perception(step/'return_perception')
            finally:runtime.open_at_home_on_return=False
            recorder.check();previous=observation
            observation=runtime.capture_current(step/'after')
            entry['after']=str(observation.image.relative_to(folder));last_action=proposal
            report['last_observation']=str(observation.image.relative_to(folder))
            save_json(step/'timing.json',dict(model_s=model_s,action_s=entry['action_s'],
                                             return_and_observe_s=time.monotonic()-t-entry['action_s']))
            save_json(folder/'result.json',report)
        print(f'Codex 局部去皱结束：{report["status"]}，执行 {report["completed_actions"]} 步',flush=True)
        if report['steps']:
            last=report['steps'][-1]
            if last.get('validation_error'):print('几何拒绝原因：'+last['validation_error'],flush=True)
            print('模型判断：'+last['proposal']['reason'],flush=True)
    except BaseException as exc:
        report.update(status='ABORTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error=f'{type(exc).__name__}: {exc}')
        # The collection owner handles stop/recorder cleanup and failed trajectory status.
        raise
    finally:
        report.update(elapsed_s=time.monotonic()-started,model_metrics=requests.metrics())
        save_json(folder/'result.json',report)
    return report
