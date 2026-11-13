"""One recorded cloth pickup and configurable image-down moving release.

Experimental entry only; uses original full collision/IK checks and monitored
RobotAPI motion. Optional isolated timing scale; no automatic retries after physical motion.
"""
from copy import deepcopy
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
from cloth_agent.experiments.partitioned_unfold import ROOT,configure_runtime,prepare_reset
from cloth_agent.replay_dual_arm_runtime import PartitionedReplayRuntime
from cloth_agent.perception import PerceptionConfig
from cloth_agent.unfold_video import UnfoldVideo
from cloth_agent.codex_motion import parse_code
from cloth_agent.experiments.reset_parking import _program_actions,shift_midline_drop,find_parking,execute_parking
from cloth_agent.robot_collision import RobotCollision
from cloth_agent.experiments.full_grasp_adapter import build_robot_config
from cloth_agent.site_robot_check import read_arm
from cloth_agent.robot_api import XArmBackend,_connect_xarm
from cloth_agent.experiments.moving_release_probe import move_with_release,scale_timing,fast_release_config


def image_down_direction(meta, anchor):
    """Image-vertical tangent on a constant-world-Z plane using calibration."""
    K=np.asarray(meta['intrinsics'],float)
    T=np.asarray(meta['world_from_camera_mm'],float)
    anchor=np.asarray(anchor,float)
    camera=np.linalg.solve(T,np.r_[anchor,1.])[:3]
    if camera[2]<=0:raise ValueError('release point behind camera')
    pixel=K@camera;pixel=pixel[:2]/pixel[2]
    ray=T[:3,:3]@np.linalg.solve(K,np.r_[pixel+[0.,60.],1.])
    if abs(ray[2])<1e-8:raise ValueError('image-down ray parallel to release plane')
    distance=(anchor[2]-T[2,3])/ray[2]
    if distance<=0:raise ValueError('release plane intersection behind camera')
    point=T[:3,3]+distance*ray
    direction=(point-anchor)[:2]
    if not np.isfinite(direction).all() or np.linalg.norm(direction)<1e-6:
        raise ValueError('invalid calibrated direction')
    return direction/np.linalg.norm(direction),[anchor,point]


def checked_program(r,before,source,direction,timing_scale=1,travel_mm=60.):
    calls=parse_code(source,allow_grasp=True)
    end=list(calls[-1][1]['right']);end[0]+=float(direction[0]*travel_mm/1000.);end[1]+=float(direction[1]*travel_mm/1000.)
    extended=source+f'motion.move_tcp(right={end!r})\n'
    config,program,assignment=r.compile(extended,before,post_release_descent_mm=50.)
    config=scale_timing(config,timing_scale)
    actions=_program_actions(program)
    opens=[i for i,a in enumerate(actions) if a['name']=='open_gripper']
    release_index=opens[-1]
    sweep=actions[release_index-1]
    start=actions[release_index-2]
    if start['name']!='move' or sweep['name']!='move':raise ValueError('unexpected release program')
    if not config.continuous_joint_stream:raise ValueError('checked streaming mode required')
    collision=RobotCollision(r.config_path,{k:np.asarray(v) for k,v in before.meta['joints_deg'].items()},assignment['active_arm'],root=r.config.root)
    collision.require_execution(config,actions)
    return config,program,assignment,actions,collision,release_index


def run_actions(backend,config,actions,records):
    pending=[]
    def flush():
        if pending:
            records.append(backend.joint_sequence(config,pending)[1]);pending.clear()
    for a in actions:
        if a['name'] in ('move','home'):pending.append(a)
        else:
            flush()
            if a['name']=='open_gripper':records.append(backend.open_gripper(config)[1])
            elif a['name']=='close_gripper':records.append(backend.close_gripper(config)[1])
            else:raise ValueError('unexpected action')
    flush()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true')
    ap.add_argument('--grasp-descent-mm',type=float,choices=(0.,17.),default=0.)
    ap.add_argument('--timing-scale',type=int,choices=(1,2),default=1)
    ap.add_argument('--release-z-mm',type=float,choices=(200.,400.),default=400.)
    ap.add_argument('--travel-mm',type=float,choices=(60.,100.,150.),default=100.)
    ap.add_argument('--park-other',action='store_true')
    ap.add_argument('--fast-release',action='store_true')
    args=ap.parse_args()
    if not args.execute:ap.error('--execute required')
    name='cloth_moving_release_'+datetime.now().strftime('%Y%m%d_%H%M%S');root=ROOT/'reports'/name;root.mkdir()
    r=configure_runtime(PartitionedReplayRuntime(ROOT/'config/dual_arm.dual_wrist_xarm7_20261110.json'))
    # The previous cloth trial lifted the foam pad. For this isolated trial,
    # default to measured surface; 17 mm is an explicit operator trial setting.
    # Do not change the formal unfold/reset policy or retry deeper on failure.
    r.task_policy=dict(r.task_policy,grasp_surface_offset_mm=-args.grasp_descent_mm)
    backend=observer=None
    release_frames=[]
    result=dict(status='PREPARING',physical_actions=0,events=[],samples=[],records=[],
                grasp_surface_offset_mm=-args.grasp_descent_mm,automatic_deeper_retry=False,timing_scale=args.timing_scale,
                release_z_mm=args.release_z_mm,travel_mm=args.travel_mm,release_trigger_progress=.1 if args.fast_release else .2)
    def save():
        (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        config_camera=PerceptionConfig.load(ROOT,ROOT/'config/perception.zed_dual_xarm7_20261103.json')
        r.video_recorder=UnfoldVideo(config_camera,ROOT/'videos'/name,root);r.video_recorder.start()
        before=r.observe(root/'before',reset_zed=True)
        print(json.dumps(dict(stage='SELECT',image=str(before.image),expected='pixel; image-down direction comes from saved calibration')),flush=True)
        selection=json.loads(input());repairs=[]
        pixel=selection['pixel'];xyz,resolved=before.sample_nearby_measured(pixel,pixel,5,repairs)
        folder=root/'planning';folder.mkdir()
        point=dict(pixel_xy=resolved,world_xyz_mm=xyz.tolist())
        source=prepare_reset(r,before,point,dict(release_world_z_mm=400.,yaw_delta=0.,drop_offset_xy=[0.,0.]),folder,
                             grasp_yaw_offset_deg=90.,release_world_z_mm=400.)
        final=list(parse_code(source,allow_grasp=True)[-1][1]['right']);final[2]=args.release_z_mm/1000.
        if args.release_z_mm!=400.:
            source+=f'motion.move_tcp(right={final!r})\n'
        direction,points=image_down_direction(before.meta,np.array(final[:3])*1000.)
        _,_,assignment=r.compile(source,before,post_release_descent_mm=50.)
        axis=np.asarray(assignment['right_axis_xy']);tangent=np.array([-axis[1],axis[0]])
        rejected=[];selected=None;parking=None
        if args.park_other:
            end=list(final);end[0]+=float(direction[0]*args.travel_mm/1000.);end[1]+=float(direction[1]*args.travel_mm/1000.)
            extended=source+f'motion.move_tcp(right={end!r})\n'
            parking=find_parking(r,before,extended,folder)
            parked=deepcopy(before)
            parked.meta['joints_deg'][parking['arm']]=parking['joints_deg']
            config,program,assignment=r.compile(parking['reset_source'],parked,post_release_descent_mm=50.)
            config=scale_timing(config,args.timing_scale)
            actions=_program_actions(program);index=[i for i,a in enumerate(actions) if a['name']=='open_gripper'][-1]
            collision=RobotCollision(r.config_path,parked.meta['joints_deg'],assignment['active_arm'],root=r.config.root)
            collision.require_execution(config,actions)
            inactive=parking['arm']
            returned=deepcopy(parked.meta['joints_deg'])
            returned[assignment['active_arm']]=list(r.config.arms[assignment['active_arm']].home_joints)
            pose=r.models[inactive].forward(np.array(returned[inactive]))
            return_config=build_robot_config(r.config.raw,{},inactive,*pose[3:5],config_path=r.config_path)
            RobotCollision(r.config_path,returned,inactive,root=r.config.root).require_execution(return_config,[dict(name='home',args={})])
            selected=(config,program,assignment,actions,collision,index);shift=parking['midline_shift_mm']
            result['parking']=parking
        else:
            for shift in (0.,10.,-10.,20.,-20.,30.,-30.,40.,-40.,50.,-50.):
                try:
                    selected=checked_program(r,before,shift_midline_drop(source,tangent,shift),direction,args.timing_scale,args.travel_mm);break
                except (ValueError,RuntimeError) as exc:
                    rejected.append(dict(shift_mm=shift,error=str(exc)))
        result.update(selection=selection,grasp=point,depth_repairs=repairs,direction_method='CALIBRATED_CONSTANT_WORLD_Z_IMAGE_DOWN',direction_world_xy=direction.tolist(),direction_points_world_mm=[p.tolist() for p in points],rejected=rejected)
        if selected is None:raise ValueError('No complete path for moving release: '+rejected[-1]['error'])
        config,program,assignment,actions,collision,index=selected
        release_config=fast_release_config(config) if args.fast_release else config
        collision.require_execution(release_config,actions)
        result['release_timing']=dict(speed_deg_s=release_config.home_speed_deg_s,acceleration_deg_s2=release_config.home_acceleration_deg_s2)
        result.update(status='PATH_CHECKED',assignment=assignment,actions=actions,shift_mm=shift)
        (folder/'checked_program.py').write_text(program,encoding='utf-8');save()
        if parking and parking['parking_required']:
            before=execute_parking(r,before,parking,folder)
            collision=RobotCollision(r.config_path,before.meta['joints_deg'],assignment['active_arm'],root=r.config.root)
            collision.require_execution(config,actions)
        from xarm.wrapper import XArmAPI
        observer=_connect_xarm(XArmAPI,config.robot_ip)
        backend=XArmBackend(config,collision_checker=collision)
        result['motion_started']=True;save()
        run_actions(backend,config,actions[:index-1],result['records'])
        start=[actions[index-2]['args'][k] for k in ('x','y','z')]
        def sample_frame(elapsed):
            recorder=r.video_recorder
            with recorder.condition:
                if recorder.latest is not None and (not release_frames or recorder.seq!=release_frames[-1][1]):
                    if len(release_frames)<400:
                        release_frames.append((elapsed,recorder.seq,recorder.latest[0].copy()))
        move_with_release(backend,observer,release_config,actions[index-1],start,result,sample_callback=sample_frame,expected_distance_mm=args.travel_mm,trigger_progress=.1 if args.fast_release else .2)
        result['physical_actions']=1
        # Opening was measured by the release worker; do not issue a second open.
        run_actions(backend,config,actions[index+1:],result['records'])
        backend.close();backend=None;observer.disconnect();observer=None
        if parking and parking['parking_required']:
            rows={k:read_arm(r.profile,k) for k in r.models}
            if any(v['status']!='READ_ONLY_CHECKS_PASS' for v in rows.values()):
                raise RuntimeError('controller not ready for parked-arm return')
            current={k:v['joints_deg'] for k,v in rows.items()}
            return_check=RobotCollision(r.config_path,current,parking['arm'],root=r.config.root)
            return_check.require_execution(return_config,[dict(name='home',args={})])
            backend=XArmBackend(return_config,collision_checker=return_check)
            result['parking_return']=backend.home(return_config)[1]
            backend.close();backend=None
        time.sleep(2)
        after=r.observe(root/'after',reset_zed=True)
        result.update(status='COMPLETED',after_image=str(after.image),cloth_release_time_measured=False)
    except BaseException as exc:
        result.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
        if backend:backend.stop()
    finally:
        if observer:observer.disconnect()
        if backend:backend.close()
        r.close()
        if release_frames:
            frames_dir=root/'release_frames';frames_dir.mkdir(exist_ok=True)
            manifest=[]
            for i,(elapsed,seq,rgb) in enumerate(release_frames):
                filename=f'{i:04d}.png';Image.fromarray(rgb).save(frames_dir/filename)
                manifest.append(dict(file=filename,elapsed_s=elapsed,source_sequence=seq))
            (frames_dir/'timestamps.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            result['release_frame_count']=len(manifest)
        save()
        print(json.dumps(dict(output=str(root),status=result['status'],error=result.get('error'),events=result['events']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
