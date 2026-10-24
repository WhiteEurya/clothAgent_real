"""One current-image corner pull for the standalone recognition experiment."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from .discovery_schema import obj, S, N, enum
from .discovery_data import save
from .graph_policy import evidence_file
from cloth_agent.dual_arm.geometry import DualArmError, matrix_pose, pose_matrix
from cloth_agent.grasp_angle import transform_for, image_angle_to_rpy, draw_preview
from cloth_agent.gamepad_collect import automatic_program, TrajectoryRecorder
from cloth_agent.cloth_smoothing import observation_context

POINT={'type':'array','items':{'type':'integer','minimum':0},'minItems':2,'maxItems':2}
SCHEMA=obj(decision=enum('ACT','STOP'),confidence=N,grasp_px=POINT,target_px=POINT,
           opening_angle_image_deg={'type':'number','minimum':0,'maximum':180},
           visible_corner_evidence=S,reason=S,expected_change=S)
PROMPT='''当前无法可靠辨认衣服的整体布局和折叠遮挡关系。你的任务是选择一个清楚可见的外露边角，
抓住它并向外拉开一次，让衣服更展开、露出更多可辨认区域。只规划这一步，不需要完成整件展开。
根据当前 RGB 的实际布料边界选择抓取点和目标点，不猜测被盖住的衣角，不要求先判断领口和左右。
目标应是图像内有实测深度的落点。用抓取、抬升、平移、放下实现拉开，不做贴桌强拖。
输出原图像素坐标，夹爪开合方向以向右为0度、顺时针，模180度；程序验证深度、距离和IK。
若有上次未知状态拉动的前图和动作，比较其效果；避免无变化地重复同一抓取。
看不清可抓的外露边角或无法提出可靠动作时返回STOP，点位填[0,0]。
简述抓取边角的可见证据和预期变化；不生成代码。'''


UNFOLD_PROMPT='''你已能辨认当前衣物的整体布局。根据当前照片和识别结果，选择下一步展开动作：
抓住一个可见的袖口、下摆角或折回的外露边缘，向外移动，让收拢或叠压的衣物部位展开。
只规划一步。利用当前识别的衣物部位，不能把画面左右当作衣服固定左右，不猜被遮住的抓取点。
输出原图抓取点、目标点及夹爪开合方向（向右0度，顺时针，模180度）。
动作由抓取、抬升、平移、放下组成。目标必须在图内且可取得实测深度，位移范围由输入指定。
若提供上次动作前图，比较变化，避免重复无效果的动作。无法可靠提出下一步时返回STOP。
不生成代码；说明可见抓取依据、动作目的和预期效果。'''


def prepare(runtime,observation,proposal,folder,*,max_distance_mm=80.,grasp_offset_mm=-15.):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    observation.validate_for(runtime.config,live=True,check_age=False)
    with Image.open(observation.image) as im:width,height=im.size
    points=[]
    for name,key in [('start','grasp_px'),('target','target_px')]:
        x,y=proposal[key]
        if not (2<=x<width-2 and 2<=y<height-2):raise DualArmError('unknown pull point outside image interior')
        xyz=np.asarray(observation.sample([x,y],runtime.config.limits['max_surface_spread_mm']),float)
        if xyz.shape!=(3,) or not np.isfinite(xyz).all():raise DualArmError('unknown pull requires measured depth')
        points.append(dict(name=name,pixel_xy=[x,y],world_xyz_mm=xyz.tolist()))
    save(folder/'points.json',points)
    start,target=[np.asarray(p['world_xyz_mm']) for p in points]
    distance=float(np.linalg.norm(target[:2]-start[:2]))
    if not 10<=distance<=max_distance_mm:
        raise DualArmError(f'unknown pull distance {distance:.1f} mm outside 10–{max_distance_mm:g} mm')
    arm=runtime.config.arms['left']
    rpy=matrix_pose(arm.world_from_base @ pose_matrix(runtime.models['left'].forward(arm.home_joints)))[3:]
    k=np.asarray(runtime.camera.intrinsics);world=transform_for(observation)
    axis=Rotation.from_euler('xyz',arm.tcp_offset[3:],degrees=True).inv().apply([0,1,0])
    chosen,delta=image_angle_to_rpy(proposal['opening_angle_image_deg'],proposal['grasp_px'],start,rpy,k,world,axis)
    draw_preview(observation.image,points,chosen,k,world,axis,folder/'preview.png')
    source=automatic_program(runtime,start,target,chosen,approach_height=50.,grasp_offset=grasp_offset_mm,
                             lift_height=50.,clearance=30.,place_offset=5.)
    (folder/'motion.py').write_text(source)
    save(folder/'geometry.json',dict(distance_mm=distance,rpy_world_deg=chosen.tolist(),yaw_delta_deg=delta,
                                    lift_height_mm=50.,grasp_offset_mm=grasp_offset_mm))
    initial={k:np.asarray(s['joints']) for k,s in runtime.coordinator.snapshots().items()}
    operations=runtime.plan(source,initial,preserve_command_boundaries=True)
    openings=[i for i,(kind,data) in enumerate(operations) if kind=='gripper' and data['action']=='open']
    if len(openings)!=2:raise ValueError('unexpected unknown-pull gripper sequence')
    i=openings[-1];operations[i]=('gripper',dict(operations[i][1],action='release_partial',opening_delta_pulse=180.,settle_s=.5))
    return operations,initial


def run_once(runtime,observation,requests,folder,media,result,*,threshold=.8,max_distance_mm=80.,previous=None,grasp_offset_mm=-15.,kind="unknown_pull",recognition=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if runtime.cancel.is_set():raise KeyboardInterrupt('unknown pull cancelled')
    media.phase(kind+'_planning')
    context=observation_context(observation,grasp_offset_mm)
    context['configured_motion']=dict(grasp_offset_mm=grasp_offset_mm,lift_height_mm=50.,place_offset_mm=5.,
                                     min_translation_mm=10.,max_translation_mm=max_distance_mm)
    context['host_checks']=[x for x in context['host_checks'] if x!='height difference']
    images=[('current',observation.image)]
    payload=dict(observation_context=context,recognition=recognition)
    if previous:
        images.append(('previous_before',Path(previous['before'])))
        payload['previous_action']=previous['proposal']
    proposal=requests.ask(kind,payload,images)
    return execute_proposal(runtime,observation,proposal,folder,media,result,
        threshold=threshold,max_distance_mm=max_distance_mm,grasp_offset_mm=grasp_offset_mm,kind=kind)


def execute_proposal(runtime,observation,proposal,folder,media,result,*,threshold=.8,
                     max_distance_mm=80.,grasp_offset_mm=-15.,kind='unknown_pull'):
    """Execute one already grounded proposal, using the same Arm6 motion path."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    prefix={'unknown_pull':'UNKNOWN_PULL','unfold':'UNFOLD','skill':'SKILL'}[kind]
    save(folder/'proposal.json',proposal)
    if runtime.cancel.is_set():raise KeyboardInterrupt('unknown pull cancelled after model call')
    if proposal['decision']!='ACT':return prefix+'_MODEL_STOP',None,None
    if proposal['confidence']<threshold or not proposal['visible_corner_evidence'].strip():
        return prefix+'_UNCERTAIN',None,None
    try:operations,initial=prepare(runtime,observation,proposal,folder,max_distance_mm=max_distance_mm,grasp_offset_mm=grasp_offset_mm)
    except (DualArmError,ValueError) as exc:
        save(folder/'rejection.json',dict(error=str(exc),physical_execution=False))
        return prefix+'_GEOMETRY_REJECTED',None,None
    if runtime.cancel.is_set():raise KeyboardInterrupt('unknown pull cancelled before dispatch')
    label={'unknown_pull':'UNKNOWN_CORNER_PULL','unfold':'KNOWN_STATE_UNFOLD',
           'skill':'EXPERIENCE_GUIDED_SKILL'}[kind]
    media.snapshot('action_before',observation.image);media.edge(label)
    save(folder/'dispatch_intent.json',dict(kind=label,
        source='CURRENT_RGB_SKILL_PLAN' if kind=='skill' else 'CURRENT_RGB_FREE_PLAN',
        before=evidence_file(observation.image),proposal=proposal))
    recorder=TrajectoryRecorder(runtime,folder)
    try:
        recorder.start();recorder.stage(label.lower());recorder.check()
        media.phase('execution');runtime.arrival(initial)
        execution=runtime._execute_operations(operations,folder)
        save(folder/'execution.json',execution)
        result['steps'].append(dict(kind=label,execution=execution,proposal=proposal))
        if execution['status']!='EXECUTED':raise DualArmError('unknown pull execution failed')
        result['physical_actions']+=1;result['pull_actions']=result.get('pull_actions',0)+1
        counter={'unknown_pull':'exploration_pulls','unfold':'unfold_actions','skill':'skill_actions'}[kind]
        result[counter]=result.get(counter,0)+1
        recorder.check();media.phase('return_perception');recorder.stage('return_perception')
        runtime.open_at_home_on_return=True
        try:runtime.move_to_perception(folder/'return_perception')
        finally:runtime.open_at_home_on_return=False
        recorder.check()
    finally:recorder.close()
    recorder.check();media.phase('end_observation')
    after=runtime.capture_current(folder/'after');media.snapshot('after',after.image);media.phase(None)
    save(folder/'transition.json',dict(kind=label,operator_accepted=False,
        before=evidence_file(observation.image),after=evidence_file(after.image),
        experiment='experience_guided_skill' if kind=='skill' else 'standalone_recognize_and_pull'))
    return 'EXECUTED',after,dict(before=str(observation.image),proposal=proposal)
