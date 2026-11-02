"""Independent Arm6 automatic upper-edge pickup / slow laydown test.
No automatic retries and no production batch integration. Defaults to offline geometry preview.
"""
import argparse
from contextlib import contextmanager
from unittest.mock import patch
from datetime import datetime
from pathlib import Path
import json
import time
import numpy as np
from PIL import Image, ImageDraw
from cloth_agent.gamepad_collect import save_json, tcp_command, approach_program, execute_program, TrajectoryRecorder
from cloth_agent.dual_arm.geometry import matrix_pose, pose_matrix, DualArmError
from cloth_agent.experiments.discovery_schema import obj, S, N, enum

SHORT_SHAKE_PROFILE=Path(__file__).resolve().parents[2]/'config/short_stroke_shake.json'

def short_shake_profile(stroke,speed,cycles):
    profile=json.loads(SHORT_SHAKE_PROFILE.read_text())
    if not np.isfinite([stroke,speed]).all() or not 0<stroke<=profile['max_stroke_mm'] or not 0<speed<=profile['target_speed_mm_s'] or type(cycles) is not int or not 0<=cycles<=profile['max_cycles']:
        raise ValueError('short-stroke shake requires <=30 mm, <=150 mm/s, and 0..3 cycles')
    return profile

PROMPT='''Locate the garment's LOWER/FRONT COLLAR RIM at its CENTER in the CURRENT RGB: the visible solid collar fabric BELOW the neck opening, on the chest/front-body side. Follow the garment's orientation rather than blindly choosing the image's largest y. Pick the central lower neckline rim itself, slightly inward onto solid collar fabric so the gripper can hold enough cloth. Do NOT choose the upper/back collar rim, a left/right shoulder, a sleeve, the empty neck opening, a label, or an unrelated chest wrinkle. If the lower/front collar center is hidden or cannot be identified reliably, return ABSTAIN with null pick_px rather than substituting another part. Return original-image pick_px and briefly explain the visible collar boundary and how the chosen fabric lies below the neck opening toward the chest. Do not generate robot commands.'''



SCHEMA=obj(decision=enum('PICK','ABSTAIN'),pick_px={'anyOf':[{'type':'array','items':{'type':'integer','minimum':0},'minItems':2,'maxItems':2},{'type':'null'}]},confidence=N,reason=S)

def locate(image, folder):
    from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
    from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once
    with Image.open(image) as im:size=im.size
    body=io.payload(image,image,list(size))
    body['instructions']=PROMPT
    body['input']=[dict(role='user',content=[dict(type='input_text',text=f'Current RGB width={size[0]}, height={size[1]}.'),dict(type='input_image',image_url=io.image_data(image))])]
    body['text']['format']['schema']=SCHEMA
    result,metrics=direct_once(folder,body,io.auth_headers(),240.)
    save_json(folder/'collar.json',result)
    if result['decision']!='PICK' or result['pick_px'] is None or result['confidence']<.8:
        raise DualArmError('visible lower/front collar center not verified; no retry motion')
    return result

def geometry(point,base_xy,table_xy,*,lift=250.,back=60.,forward=120.,table_z=20.,lift_world_z=None):
    p=np.asarray(point,float);base=np.asarray(base_xy,float);center=np.asarray(table_xy,float)
    if p.shape!=(3,) or base.shape!=(2,) or center.shape!=(2,) or not np.isfinite(np.r_[p,base,center,lift,back,forward,table_z]).all():raise ValueError('invalid geometry')
    if not 0<back<=100 or not 0<forward<=250:raise ValueError('test distance outside configured bounds')
    if lift_world_z is None:
        if not 50<=lift<=300:raise ValueError('relative lift outside configured bounds')
    elif not np.isfinite(lift_world_z) or lift_world_z<=p[2]+5:
        raise ValueError('Home world Z must be finite and above grasp surface')
    if p[2]<table_z:raise ValueError('measured collar surface is below configured support plane; inspect height')
    grasp=p.copy();grasp[2]-=5.
    if grasp[2]<0:raise ValueError('grasp would descend below world Z=0 hard-table floor')
    direction=base-p[:2]
    if np.linalg.norm(direction)<1:raise ValueError('base retreat direction undefined')
    direction/=np.linalg.norm(direction)
    raised=p.copy();raised[2]=p[2]+lift if lift_world_z is None else lift_world_z
    retreat=raised.copy();retreat[:2]+=back*direction
    # Forward must reverse the baseward retreat, even if a nominal
    # table-center point lies behind the retreated collar position.
    inward=-direction
    target=retreat.copy();target[:2]+=forward*inward;target[2]=table_z+5
    return dict(grasp_surface=p.tolist(),grasp_xyz_mm=grasp.tolist(),grasp_offset_mm=-5.,lifted=raised.tolist(),retreated=retreat.tolist(),target=target.tolist(),table_center_xy=center.tolist(),base_xy=base.tolist(),table_z_mm=table_z,contact_source='UNVERIFIED_OPEN_LOOP',lift_mm=float(raised[2]-p[2]),lift_mode='RELATIVE' if lift_world_z is None else 'HOME_WORLD_Z',retreat_mm=back,laydown_mm=forward)

def laydown_points(contact,target,step_mm=20.):
    a,b=np.asarray(contact,float),np.asarray(target,float)
    if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite(np.r_[a,b,step_mm]).all() or step_mm<=0 or a[2]<b[2]:raise ValueError('invalid descending laydown')
    n=max(1,int(np.ceil(np.linalg.norm(b-a)/step_mm)))
    return [a+(b-a)*i/n for i in range(1,n+1)]

def preview(rgb,point,g,folder):
    with Image.open(rgb) as im:
        im=im.convert('RGB');d=ImageDraw.Draw(im);x,y=point;d.ellipse((x-9,y-9,x+9,y+9),outline='cyan',width=3);d.text((x+12,y),'LOWER COLLAR',fill='cyan');im.save(folder/'collar_preview.png')
    # SVG is a geometric schematic, not a prediction of the cloth shape.
    points=[g['grasp_surface'],g['lifted'],g['retreated'],g['target']]
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="400"><rect width="100%" height="100%" fill="white"/><text x="20" y="25">World XY / Z schematic (mm); contact height requires operator confirmation</text>']
    for panel,axes in enumerate([(0,1),(0,2)]):
        arr=np.array(points)[:,axes];lo=arr.min(0)-30;span=np.maximum(arr.max(0)-lo+30,1);mapped=(arr-lo)/span*[420,270];mapped[:,1]=350-mapped[:,1];mapped[:,0]+=30+panel*500
        svg.append('<polyline points="'+' '.join(f'{x:.1f},{y:.1f}' for x,y in mapped)+'" fill="none" stroke="teal" stroke-width="3"/>')
        for i,(x,y) in enumerate(mapped):svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="orange"/><text x="{x+7}" y="{y-8}">{i}</text>')
    svg.append('</svg>');(folder/'trajectory.svg').write_text(''.join(svg))

def automatic_laydown_tail(g,rpy,transition_height_mm=100.):
    if not np.isfinite(transition_height_mm) or not 5<transition_height_mm<=200:
        raise ValueError('transition height must be (5,200] mm above support')
    transition=np.array(g['retreated'],float)
    transition[2]=min(transition[2],g['table_z_mm']+transition_height_mm)
    source=tcp_command(np.r_[transition,rpy])
    source+=''.join(tcp_command(np.r_[v,rpy]) for v in laydown_points(transition,g['target']))
    source+='motion.gripper(arm="left", action="open")\n'+tcp_command(np.r_[np.array(g['target'])+[0,0,30],rpy])
    return source,dict(source='FIXED_HEIGHT_OPEN_LOOP',contact_verified=False,
                       transition_world_xyz_mm=transition.tolist(),transition_height_mm=transition_height_mm)

def command_stages(source):
    from cloth_agent.codex_motion import parse_code
    calls=parse_code(source)
    fixed=['prepare_open','approach','grasp_descend','close_gripper','lift','retreat','vertical_descent']
    if len(calls)<10:raise ValueError('unexpected retry command layout')
    labels=fixed+['laydown_forward_descend']*(len(calls)-9)+['release','post_release_retreat']
    return [dict(command_index=i,phase=label,command=kind,arguments=kw)
            for i,((kind,kw),label) in enumerate(zip(calls,labels),1)]

def failure_diagnostics(runtime,error,stages):
    # Preserve the exact rejected interpolation target, not just its endpoint.
    context=dict(getattr(runtime,'planning_context',{}) or {})
    ik=getattr(error,'ik_failure',None) or context.get('ik_failure')
    index=ik.get('command_index') if ik else context.get('command_index')
    command=next((c for c in stages if c['command_index']==index),None)
    phase=command['phase'] if command else context.get('stage','unknown')
    if ik and ik.get('phase')=='host_approach_ik':phase='approach_endpoint_ik'
    result=dict(error=f'{type(error).__name__}: {error}',phase=phase,
        command_index=index,command_endpoint=command,planning_context=context,
        ik_failure=ik,ik_candidate=getattr(error,'ik_candidate',None),
        solver_diagnostics=getattr(error,'ik_diagnostics',None),
        dispatched_segments=getattr(runtime,'plan_stats',{}).get('dispatched_segments'),
        thresholds_modified=False)
    if ik and ik.get('target_world_m_rad'):
        pose=np.array(ik['target_world_m_rad'],float)
        result['failed_target_world_mm_deg']=np.r_[pose[:3]*1000,np.degrees(pose[3:])].tolist()
    # Normalize NumPy values in diagnostic-only fields without losing numbers.
    return json.loads(json.dumps(result,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)))

def home_stop_program(runtime,point,home,table_z):
    point=np.asarray(point,float);home=np.asarray(home,float)
    if point.shape!=(3,) or home.shape!=(6,) or not np.isfinite(np.r_[point,home,table_z]).all():raise ValueError('invalid home-return geometry')
    grasp=point.copy();grasp[2]-=5
    if point[2]<table_z or grasp[2]<0 or home[2]<=grasp[2]:raise ValueError('invalid home-return height')
    rpy=home[3:]
    source=approach_program(runtime,np.r_[point+[0,0,50],rpy])
    source+=tcp_command(np.r_[grasp,rpy])+'motion.gripper(arm="left", action="close")\n'
    # One Cartesian segment simultaneously retracts and rises to calibrated Home TCP.
    source+=tcp_command(home)
    from cloth_agent.codex_motion import parse_code
    calls=parse_code(source);labels=['prepare_open','approach','grasp_descend','close_gripper','lift_retract_to_home']
    if len(calls)!=len(labels):raise ValueError('unexpected home-return command layout')
    stages=[dict(command_index=i,phase=label,command=kind,arguments=kw) for i,((kind,kw),label) in enumerate(zip(calls,labels),1)]
    g=dict(mode='HOME_STOP',grasp_surface=point.tolist(),grasp_xyz_mm=grasp.tolist(),grasp_offset_mm=-5.,home_world_mm_deg=home.tolist(),lifted=home[:3].tolist(),retreated=home[:3].tolist(),target=home[:3].tolist(),release=False,return_to_perception=False)
    return source,stages,g

def vertical_drop_program(runtime,point,home,table_z,*,shake_cycles=0,shake_mm=10.):
    target=np.asarray(home,float).copy()
    target[:2]=np.asarray(point,float)[:2]
    source,stages,g=home_stop_program(runtime,point,target,table_z)
    stages[-1]['phase']='vertical_lift_to_home_height'
    if type(shake_cycles) is not int or not 0<=shake_cycles<=3 or not np.isfinite(shake_mm) or not 0<shake_mm<=60:
        raise ValueError('shake requires 0..3 cycles and amplitude (0,60] mm')
    if shake_cycles and target[2]-shake_mm<=max(float(point[2]),table_z)+5:
        raise ValueError('insufficient vertical clearance for shake')
    from cloth_agent.codex_motion import parse_code
    for cycle in range(shake_cycles):
        lower=target.copy();lower[2]-=shake_mm
        for pose,phase in [(lower,'shake_down'),(target,'shake_up')]:
            line=tcp_command(pose);source+=line
            kind,kw=parse_code(line)[0]
            stages.append(dict(command_index=len(stages)+1,phase=phase,cycle=cycle+1,command=kind,arguments=kw))
    source+='motion.gripper(arm="left", action="open")\n'
    stages.append(dict(command_index=len(stages)+1,phase='release_drop',command='gripper',arguments=dict(arm='left',action='open')))
    g.update(mode='VERTICAL_DROP',release=True,home_world_mm_deg=np.asarray(home,float).tolist(),lift_target_world_mm_deg=target.tolist(),horizontal_displacement_mm=0.,shake_cycles=shake_cycles,shake_amplitude_mm=shake_mm,shake_axis="world_z")
    return source,stages,g

@contextmanager
def shake_speed_profile(runtime,stages,requested_speed,configured_cap,*,independent=False,acceleration=500.):
    if not np.isfinite([requested_speed,configured_cap]).all() or requested_speed<=0 or configured_cap<=0:
        raise ValueError('invalid shake speed')
    if independent and (not 0<requested_speed<=150 or not np.isfinite(acceleration) or not 0<acceleration<=500):
        raise ValueError('independent shake requires speed <=150 mm/s and acceleration <=500 mm/s^2')
    shake_indices={s['command_index'] for s in stages if s['phase'] in ('shake_down','shake_up')}
    original=runtime._tcp_segments
    def segment(*a,**kw):
        old=runtime.config.limits.copy()
        is_shake=getattr(runtime,'planning_context',{}).get('command_index') in shake_indices
        try:
            if is_shake:
                runtime.config.limits['cartesian_speed_mm_s']=requested_speed if independent else min(requested_speed,configured_cap)
                if independent:
                    runtime.config.limits['cartesian_accel_mm_s2']=acceleration
                # Shake timing uses Cartesian speed, not the legacy joint cap.
                kw['apply_joint_speed_cap']=False
            return original(*a,**kw)
        finally:
            runtime.config.limits.clear();runtime.config.limits.update(old)
    with patch.object(runtime,'_tcp_segments',segment):yield

def run_live(args):
    from cloth_agent.experiments.trajectory_memory_full_episode_v1 import EpisodeCollectionRuntime
    runtime=None;recorder=None;stages=[];report=dict(status='PREPARING',automatic_retries=0,grasp_status='UNKNOWN',speed_mm_s=args.speed_mm_s)
    try:
        runtime=EpisodeCollectionRuntime(args.config)
        runtime.config.home_recover_stopped=True
        configured_speed_cap=runtime.config.limits['cartesian_speed_mm_s']
        runtime.config.limits['cartesian_speed_mm_s']=min(runtime.config.limits['cartesian_speed_mm_s'],args.speed_mm_s)
        before=runtime.observe(args.output/'before');before.validate_for(runtime.config,live=True,check_age=False)
        result=locate(before.image,args.output/'grounding')
        point=before.sample(result['pick_px'],runtime.config.limits['max_surface_spread_mm'])
        home=matrix_pose(runtime.config.arms['left'].world_from_base@pose_matrix(runtime.models['left'].forward(runtime.config.arms['left'].home_joints)))
        if args.mode in ('home-stop','vertical-drop'):
            builder=vertical_drop_program if args.mode=='vertical-drop' else home_stop_program
            source,stages,g=builder(runtime,point,home,args.table_z_mm,**(dict(shake_cycles=args.shake_cycles,shake_mm=args.shake_mm) if args.mode=="vertical-drop" else {}))
            save_json(args.output/'geometry.json',g);preview(before.image,result['pick_px'],g,args.output)
            save_json(args.output/'command_stages.json',stages)
            save_json(args.output/'execution_policy.json',dict(mode=g['mode'],shake_speed_cap_mm_s=args.shake_speed_mm_s if args.independent_shake else min(args.shake_speed_mm_s,configured_speed_cap),independent_shake=args.independent_shake,shake_acceleration_mm_s2=args.shake_accel_mm_s2 if args.independent_shake else runtime.config.limits['cartesian_accel_mm_s2'],configured_cartesian_cap_mm_s=configured_speed_cap,grasp_offset_mm=-5.,release=g['release'],return_to_perception=False,operator_prompts=0))
            print(f'自动执行：抓取下领口中央布料 → XY不变，升高Z至Home高度 → 上下{args.shake_mm:g}mm往返{args.shake_cycles}次 → 松爪。' if args.mode=='vertical-drop' else '自动执行：抓取下领口中央布料 → 同时抬升并收回至Home → 保持夹爪闭合并停止。',flush=True)
            recorder=TrajectoryRecorder(runtime,args.output);recorder.start()
            with shake_speed_profile(runtime,stages,args.shake_speed_mm_s,configured_speed_cap,independent=args.independent_shake,acceleration=args.shake_accel_mm_s2):
                execute_program(runtime,recorder,args.output,'vertical_pickup_drop' if args.mode=='vertical-drop' else 'pickup_home_stop',source)
            report.update(status='EXECUTED',end_state='VERTICAL_LIFT_RELEASED' if args.mode=='vertical-drop' else 'HOME_TCP_HOLDING_COMMAND',grasp_status='UNKNOWN',release_commanded=g['release'])
            try:runtime.capture_current(args.output/'after_at_end_pose')
            except DualArmError as capture_error:
                report['post_action_observation_error']=str(capture_error)
                print('动作完成，结束位置RGB-D采集失败：'+str(capture_error),flush=True)
        else:
            g=geometry(point,runtime.config.arms['left'].world_from_base[:2,3],args.table_center_xy,
                lift=args.lift_mm,back=args.back_mm,forward=args.forward_mm,table_z=args.table_z_mm,
                lift_world_z=home[2] if args.lift_mode=="home" else None)
            save_json(args.output/'geometry.json',g);preview(before.image,result['pick_px'],g,args.output)
            rpy=home[3:]
            source=approach_program(runtime,np.r_[point+[0,0,50],rpy])+tcp_command(np.r_[g["grasp_xyz_mm"],rpy])+'motion.gripper(arm="left", action="close")\n'+tcp_command(np.r_[g['lifted'],rpy])+tcp_command(np.r_[g['retreated'],rpy])
            tail,report['contact']=automatic_laydown_tail(g,rpy,args.transition_height_mm)
            source+=tail
            stages=command_stages(source)
            save_json(args.output/'command_stages.json',stages)
            save_json(args.output/'execution_policy.json',dict(mode='AUTOMATIC_SINGLE_TEST',contact=report['contact'],grasp_offset_mm=-5.,operator_prompts=0))
            print('自动执行：抓取下领口中央布料 → 提起 → 后退 → 缓降 → 前进下降铺放；接触状态未测量。',flush=True)
            recorder=TrajectoryRecorder(runtime,args.output);recorder.start()
            # Plan the complete pickup-through-release path before any grasp command.
            execute_program(runtime,recorder,args.output,'automatic_retry_laydown',source)
            runtime.move_to_perception(args.output/'return_perception')
            runtime.capture_current(args.output/'after');report['status']='EXECUTED';report['unfolding_outcome']='UNVERIFIED'
    except BaseException as exc:
        report.update(status='STOPPED',error=f'{type(exc).__name__}: {exc}')
        diagnostic=failure_diagnostics(runtime,exc,stages)
        save_json(args.output/'failure_diagnostics.json',diagnostic)
        report['failure_phase']=diagnostic['phase']
        report['failure_diagnostics']='failure_diagnostics.json'
        print('RETRY FAILURE: phase='+str(diagnostic['phase'])+' command='+str(diagnostic['command_index'])+'; diagnostics: '+str(args.output/'failure_diagnostics.json'),flush=True)
        if diagnostic.get('failed_target_world_mm_deg'):
            print('Rejected interpolated world XYZ(mm), RPY(deg): '+str(diagnostic['failed_target_world_mm_deg']),flush=True)
        if runtime is not None:
            runtime.cancel.set()
            try:report['stop_result']=runtime.stop()
            except Exception as stop:report['stop_error']=str(stop)
        raise
    finally:
        if recorder is not None:recorder.close()
        if runtime is not None:runtime.close()
        save_json(args.output/'result.json',report)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--onsite-confirmed',action='store_true')
    p.add_argument('--mode',choices=['vertical-drop','home-stop','laydown'],default='vertical-drop')
    p.add_argument('--config',type=Path,default=Path('config/dual_arm.local.json'))
    p.add_argument('--output',type=Path,default=Path('reports')/('collar_retry_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    p.add_argument('--table-center-xy',type=float,nargs=2,required=True,help='verified table interior center in world mm; not pixels')
    p.add_argument('--table-z-mm',type=float,required=True,help='verified support plane world Z')
    p.add_argument('--lift-mode',choices=['home','relative'],default='home')
    p.add_argument('--lift-mm',type=float,default=250,help='only used with --lift-mode relative');p.add_argument('--back-mm',type=float,default=60);p.add_argument('--forward-mm',type=float,default=120);p.add_argument('--speed-mm-s',type=float,default=5)
    p.add_argument('--shake-cycles',type=int,default=3,help='vertical-drop only; 0 disables gentle vertical shake')
    p.add_argument('--shake-mm',type=float,default=30.)
    p.add_argument('--independent-shake',action='store_true',help='isolated suspended shake Cartesian speed/acceleration profile; geometry checks remain active')
    p.add_argument('--shake-accel-mm-s2',type=float,default=500.)
    p.add_argument('--shake-speed-mm-s',type=float,default=None,help='shake stage only; default 150 mm/s for independent profile, otherwise 15 mm/s; planned speed may be lower')
    p.add_argument('--transition-height-mm',type=float,default=100.,help='open-loop transition height above support; not detected cloth contact')
    p.add_argument('--observation',type=Path);p.add_argument('--pick-px',type=int,nargs=2)
    p.add_argument('--base-xy',type=float,nargs=2,help='offline world base origin')
    a=p.parse_args()
    if a.shake_speed_mm_s is None:a.shake_speed_mm_s=150. if a.independent_shake else 15.
    if not 0<=a.shake_cycles<=3 or not 0<a.shake_mm<=60:p.error('shake cycles 0..3, amplitude (0,60] mm')
    if not 0<a.shake_speed_mm_s<=(150 if a.independent_shake else 30):p.error('shake speed exceeds selected profile range')
    if not 0<a.shake_accel_mm_s2<=500:p.error('shake acceleration must be (0,500] mm/s^2')
    if a.independent_shake:
        try:short_shake_profile(a.shake_mm,a.shake_speed_mm_s,a.shake_cycles)
        except ValueError as e:p.error(str(e))
    if a.independent_shake and a.mode!='vertical-drop':p.error('independent shake only supports vertical-drop')
    if not 5<a.transition_height_mm<=200:p.error('transition height must be (5,200] mm')
    if not 0<a.speed_mm_s<=5:p.error('first-test speed must be (0,5] mm/s')
    if a.execute and (not a.onsite_confirmed or a.observation or a.pick_px):p.error('execute requires onsite confirmation and fresh automatic upper-edge grounding')
    a.output.mkdir(parents=True,exist_ok=False)
    save_json(a.output/'parameters.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    if a.independent_shake:save_json(a.output/'short_stroke_profile.json',short_shake_profile(a.shake_mm,a.shake_speed_mm_s,a.shake_cycles))
    if a.execute:return run_live(a)
    if not a.observation or a.pick_px is None or a.base_xy is None:p.error('offline preview requires observation, pick-px, base-xy')
    from cloth_agent.dual_arm.observation import Observation
    obs=Observation.load(a.observation);point=obs.sample(a.pick_px,None)
    from cloth_agent.gamepad_collect_runtime import CollectionRuntime
    from types import SimpleNamespace
    offline=CollectionRuntime(a.config)
    arm=offline.config.arms['left']
    home=matrix_pose(arm.world_from_base@pose_matrix(offline.models['left'].forward(arm.home_joints)))
    if a.mode in ('home-stop','vertical-drop'):
        offline.coordinator=SimpleNamespace(snapshots=lambda:{k:{'joints':np.asarray(q)} for k,q in obs.meta['joints_deg'].items()})
        builder=vertical_drop_program if a.mode=='vertical-drop' else home_stop_program
        source,stages,g=builder(offline,point,home,a.table_z_mm,**(dict(shake_cycles=a.shake_cycles,shake_mm=a.shake_mm) if a.mode=="vertical-drop" else {}))
        (a.output/'motion.py').write_text(source);save_json(a.output/'command_stages.json',stages)
    else:
        g=geometry(point,a.base_xy,a.table_center_xy,lift=a.lift_mm,back=a.back_mm,forward=a.forward_mm,table_z=a.table_z_mm,lift_world_z=home[2] if a.lift_mode=='home' else None)
    save_json(a.output/'geometry.json',g);preview(obs.image,a.pick_px,g,a.output)
    save_json(a.output/'result.json',dict(status='OFFLINE_GEOMETRY_ONLY',robot_calls=0,model_calls=0,host_kinematics_checked=False))
    print(a.output)

if __name__=='__main__':main()
