"""Bounded view changes for missing depth, before a grasp has been dispatched."""
from pathlib import Path
import numpy as np
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.dual_arm.observation import DepthUnavailableError
from .discovery_data import save


def lateral_poses(origin, world_from_camera, distance_mm=15.):
    origin=np.asarray(origin,float);frame=np.asarray(world_from_camera,float)
    if origin.shape!=(6,) or frame.shape!=(4,4) or not np.isfinite(np.r_[origin,frame.ravel()]).all():
        raise DualArmError('invalid depth recovery geometry')
    if not 0<distance_mm<=20:
        raise DualArmError('depth recovery offset exceeds 20 mm bound')
    direction=frame[:2,0].copy()
    norm=np.linalg.norm(direction)
    if norm<.5:raise DualArmError('camera horizontal direction is not suitable for lateral recovery')
    direction/=norm
    result=[]
    for sign in (-1.,1.):
        pose=origin.copy();pose[:2]+=sign*distance_mm*direction;result.append(pose)
    return result


def move_view(runtime, pose, folder):
    from cloth_agent.gamepad_collect import tcp_command
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    if runtime.cancel.is_set():raise KeyboardInterrupt('depth recovery cancelled')
    source=tcp_command(pose)
    (folder/'motion.py').write_text(source)
    initial={k:np.asarray(v['joints']) for k,v in runtime.coordinator.snapshots().items()}
    speed=runtime.config.limits['cartesian_speed_mm_s']
    runtime.config.limits['cartesian_speed_mm_s']=min(speed,10.)
    try:operations=runtime.plan(source,initial,preserve_command_boundaries=True)
    finally:runtime.config.limits['cartesian_speed_mm_s']=speed
    if any(kind!='move' for kind,_ in operations):raise DualArmError('viewpoint move must not operate the gripper')
    runtime.arrival(initial)
    result=runtime._execute_operations(operations,folder);save(folder/'result.json',result)
    if result['status']!='EXECUTED':raise DualArmError(f'depth viewpoint move failed: {result}')


def prepare_with_view_retry(runtime, observation, anchors, folder, prepare, reground, *, distance_mm=15.):
    """Return (observation, anchors, prepared motion); only retry missing-depth errors.

    reground must re-recognize the state and choose NEW pixels in the new RGB.
    prepare must finish depth validation before dispatching any grasp action.
    """
    try:
        return observation,anchors,prepare(observation,anchors)
    except DepthUnavailableError as exc:
        initial_error=str(exc)
    from cloth_agent.gamepad_collect import world_pose
    if getattr(runtime,'active_arms',())!=('left',):raise DualArmError('depth view recovery is Arm6-only')
    if runtime.cancel.is_set():raise KeyboardInterrupt('depth recovery cancelled')
    runtime.arrival({k:np.asarray(v) for k,v in observation.meta['joints_deg'].items()})
    origin=world_pose(runtime)
    poses=lateral_poses(origin,observation.meta['world_from_camera_mm'],distance_mm)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    log=dict(initial_error=initial_error,offset_mm=distance_mm,attempts=[],status='RETRYING',
             original_world_tcp_mm_deg=origin.tolist(),gripper_commands=False)
    save(folder/'result.json',log)
    print(f'选点深度无效：在当前拍照位置左移 {distance_mm:g} mm 重拍；必要时改为原位置右侧 {distance_mm:g} mm。',flush=True)
    for index,pose in enumerate(poses):
        if runtime.cancel.is_set():raise KeyboardInterrupt('depth recovery cancelled')
        attempt=folder/f'view_{index:02d}';attempt.mkdir()
        move_view(runtime,pose,attempt/'move')
        fresh=runtime.capture_current(attempt/'observation')
        new_anchors=reground(fresh,attempt)
        entry=dict(view=index,observation=str(fresh.image),anchors=new_anchors,world_tcp_mm_deg=pose.tolist())
        log['attempts'].append(entry)
        try:
            prepared=prepare(fresh,new_anchors)
        except DepthUnavailableError as exc:
            entry['error']=str(exc);save(folder/'result.json',log);continue
        no_action=new_anchors.get('decision') in ('DONE','ABSTAIN')
        entry['status']='NO_ACTION_'+new_anchors['decision'] if no_action else 'DEPTH_VALID'
        log['status']='REPLANNED_NO_ACTION' if no_action else 'RECOVERED';save(folder/'result.json',log)
        print('换视角后决定不执行抓取。' if no_action else '换视角后深度有效，使用新图像和新标定姿态继续。',flush=True)
        return fresh,new_anchors,prepared
    if not runtime.cancel.is_set():move_view(runtime,origin,folder/'restore')
    log['status']='DEPTH_UNAVAILABLE_AFTER_TWO_VIEWS';save(folder/'result.json',log)
    raise DepthUnavailableError(f'depth unavailable after left/right {distance_mm:g} mm view retries; restored original observation pose')
