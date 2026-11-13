"""Current ZED + dual Arm7 adapter for the existing contextual one-shot planner.

No graph or alternate motion backend. RobotAPI owns motion and stops.
"""
from pathlib import Path
import argparse
import json
import shutil
import signal
import traceback
import secrets
import time

import numpy as np
from PIL import Image

from cloth_agent.replay_dual_arm_runtime import PartitionedReplayRuntime, route_grasp
from cloth_agent.dual_arm.geometry import DualArmError, matrix_pose, pose_matrix
from .discovery_data import save
from . import unfold_console as console

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT/'config/dual_arm.zed_dual_xarm7_20261103.json'
DEFAULT_GOAL = ROOT/'data/reference/pink_kirby_goal_20261109/goal.png'
RESET_RELEASE_WORLD_Z_MM = 325.0
RESET_RETURN_DESCENT_MM = 50.0
RESET_CLEARANCE_LIFT_MM = 70.0
GRASP_SURFACE_OFFSET_MM = -20.0
CAPTURE_ATTEMPTS = 3


def retryable_capture_error(exc):
    """Only transient camera acquisition failures; never robot/controller errors."""
    message = str(exc).lower()
    tokens = (
        "frame didn't arrive", "capture stopped or timed out",
        "rgb-d frame timed out", "camera grab failed", "zed grab failed",
        "warmup failed after", "no device connected", "device disconnected",
    )
    return any(token in message for token in tokens)


def planner_retry_delay(exc, failed_attempt):
    """Back off only for explicit transient service responses, never local errors."""
    message=str(exc).lower()
    if '429' in message or 'too many requests' in message:
        return (30.,60.)[min(failed_attempt,1)]
    if any(token in message for token in ('502 bad gateway','503 service unavailable','504 gateway timeout')):
        return (10.,30.)[min(failed_attempt,1)]
    return 0.


class TranslationRejected(DualArmError):
    def __init__(self, feedback):
        self.feedback = feedback
        super().__init__(f"unfold XY translation {feedback['distance_mm']:.1f} mm outside "
                         f"10–{feedback['maximum_mm']} mm")


def prepare_action(runtime, before, plan, folder):
    """Original lift/pull/place heights, with ownership chosen before orientation."""
    from scipy.spatial.transform import Rotation
    from cloth_agent.grasp_angle import image_angle_to_rpy, transform_for
    from cloth_agent import restricted_context_one_shot as planner
    measurement=getattr(before,'planning_view',before)
    planner.validate(plan, Image.open(measurement.image).size, False)
    if plan['decision'] != 'ACTION':
        raise ValueError('only ACTION can be prepared')
    before.validate_for(runtime.config, live=True, check_age=False)
    # Cloth folds can have real local height variation. Keep measured-depth
    # validity checks, without rejecting a patch solely for its Z spread.
    a, b = [np.asarray(measurement.sample(plan[key], None), float)
            for key in ('grasp_px', 'target_px')]
    distance = float(np.linalg.norm(b[:2]-a[:2]))
    smoothing = plan['is_smoothing']
    from .unfold_planning_skills import NAMES, SPACE_PREPARATION
    manifest_path = Path(folder)/'planning_skills.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    selected_skill = manifest.get('selected_skill')
    # Amplitude is constrained by measured workspace and complete motion checks,
    # not by a separate task-level distance cap.
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(distance) or distance <= 0:
        raise DualArmError('grasp/target require finite measurements and nonzero XY translation')
    joints = before.meta['joints_deg']
    tcp = {k: (arm.world_from_base @ pose_matrix(runtime.models[k].forward(joints[k])))[:3, 3]
           for k, arm in runtime.config.arms.items()}
    routing_source = (f'motion.grasp(right={np.r_[a/1000,0.,0.,0.].tolist()!r})\n'
                      f'motion.move_tcp(right={np.r_[b/1000,0.,0.,0.].tolist()!r})\n')
    _, assignment = route_grasp(routing_source, tcp)
    active = assignment['active_arm'];arm = runtime.config.arms[active]
    home = runtime.models[active].forward(arm.home_joints)
    rpy = matrix_pose(arm.world_from_base @ pose_matrix(home))[3:]
    angle = plan['grasp_angle_deg']
    if angle is not None:
        angle_pixel=plan['grasp_px'];camera_transform=transform_for(before)
        K = np.asarray(before.meta['intrinsics'], float)
        if hasattr(measurement,'orientation_inputs'):
            angle,angle_pixel,K,camera_transform=measurement.orientation_inputs(plan['grasp_px'],angle)
        if K.shape != (3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1]) <= 0:
            raise DualArmError('invalid current ZED intrinsics')
        axis = Rotation.from_euler('xyz', arm.tcp_offset[3:], degrees=True).inv().apply([0.,1.,0.])
        rpy, _ = image_angle_to_rpy(angle, angle_pixel, a, rpy, K, camera_transform, axis)
    lift_clearance_mm = 10 if smoothing else (30 if selected_skill in NAMES else 20)
    height = max(a[2],b[2]) + lift_clearance_mm
    points = [a, np.r_[a[:2],height], np.r_[b[:2],height], b+[0,0,5]]
    source = '\n'.join(f'motion.{"grasp" if i==0 else "move_tcp"}(right={np.r_[p/1000,np.radians(rpy)].tolist()!r})'
                       for i,p in enumerate(points))+'\n'
    # Reuse the replay compiler: only the selected arm moves; offset applied once.
    config, program, actual_assignment = runtime.compile(source, before)
    folder = Path(folder)
    save(folder/'geometry.json', dict(grasp_world_mm=a.tolist(),target_world_mm=b.tolist(),
        distance_mm=distance,lift_world_z_mm=height,lift_clearance_mm=lift_clearance_mm,selected_skill=selected_skill,grasp_offset_mm=GRASP_SURFACE_OFFSET_MM,
        target_depth_policy='current measured owning-camera depth; no plane fallback',
        source_references=({k:measurement.reference(plan[k]) for k in ('grasp_px','target_px')} if hasattr(measurement,'reference') else None),
        assignment=actual_assignment))
    (folder/'motion.py').write_text(source,encoding='utf-8')
    (folder/'robotapi_preview.py').write_text(program,encoding='utf-8')
    return source


def prepare_reset(runtime, before, point, release, folder, *, grasp_yaw_offset_deg=0.,
                  release_world_z_mm=RESET_RELEASE_WORLD_Z_MM, far_translation_mm=0.,
                  release_at_grasp=False):
    """Lift clear, transfer to the partition line, then raise to release height."""
    a = np.asarray(point['world_xyz_mm'], float)
    joints = before.meta['joints_deg']
    tcp = {k: (arm.world_from_base @ pose_matrix(runtime.models[k].forward(joints[k])))[:3,3]
           for k, arm in runtime.config.arms.items()}
    raw = np.r_[a/1000, 0.,0.,0.].tolist()
    _, assignment = route_grasp(f'motion.grasp(right={raw!r})\nmotion.move_tcp(right={raw!r})\n', tcp)
    active = assignment['active_arm']; arm = runtime.config.arms[active]
    home = matrix_pose(arm.world_from_base @ pose_matrix(runtime.models[active].forward(arm.home_joints)))
    if grasp_yaw_offset_deg not in (-90., 0., 90.):
        raise ValueError('reset trial yaw offset must be -90, 0 or 90 degrees')
    home[5] += grasp_yaw_offset_deg
    height = float(release['release_world_z_mm'])
    offset = np.asarray(release['drop_offset_xy'],float)
    if (not np.isfinite(height) or height != release_world_z_mm or offset.shape != (2,)
            or not np.isfinite(offset).all() or np.any(offset != 0)
            or release['yaw_delta'] != 0):
        raise DualArmError('invalid midline/Home-yaw initialization release')
    if a[2] >= height:
        raise DualArmError('reset grasp surface is at or above fixed release world Z')
    lifted = np.r_[a[:2], min(a[2] + RESET_CLEARANCE_LIFT_MM, height)]
    axis = np.asarray(assignment['right_axis_xy'])
    drop_xy = a[:2] - assignment['signed_distance_mm'] * axis
    if release_at_grasp:
        if far_translation_mm:raise ValueError('vertical drop cannot include horizontal translation')
        drop_xy=a[:2].copy()
    transfer = np.r_[drop_xy, lifted[2]]
    dropped = np.r_[drop_xy, height]
    points = [a, lifted, transfer, dropped]
    if release_at_grasp:
        lifted = dropped.copy()
        points = [a, lifted, dropped]
    source = ''.join(f'motion.{"grasp" if i==0 else "move_tcp"}(right={np.r_[p/1000,np.radians(home[3:])].tolist()!r})\n'
                     for i,p in enumerate(points))
    midline = dropped.copy()
    if not np.isfinite(far_translation_mm):raise ValueError('invalid reset translation')
    if far_translation_mm:
        dropped = dropped + [0.,-far_translation_mm,0.]
        source += f'motion.move_tcp(right={np.r_[dropped/1000,np.radians(home[3:])].tolist()!r})\n'
    _,program,actual = runtime.compile(source,before,post_release_descent_mm=RESET_RETURN_DESCENT_MM)
    if actual['active_arm'] != active:raise DualArmError('reset arm assignment changed')
    save(folder/'geometry.json',dict(assignment=actual,grasp_world_mm=a.tolist(),
        lift_world_mm=lifted.tolist(),drop_world_mm=dropped.tolist(),release=release,
        release_policy='VERTICAL_AT_GRASP_XY' if release_at_grasp else 'GRASP_PROJECTION_ON_LIVE_TCP_BISECTOR',release_world_z_mm=height,
        grasp_yaw_offset_deg=grasp_yaw_offset_deg,far_translation_mm=far_translation_mm,
        transfer_world_mm=transfer.tolist(),clearance_lift_mm=RESET_CLEARANCE_LIFT_MM,
        midline_before_translation_world_mm=midline.tolist()))
    (folder/'motion.py').write_text(source,encoding='utf-8')
    (folder/'robotapi_preview.py').write_text(program,encoding='utf-8')
    return source


def initialize(runtime, before, folder, *, headers, timeout, direct, seed, radius_mm):
    """One original center-sampling reset, without retries or an Arm6 backend."""
    from cloth_agent import random_fold_reset as reset
    from . import one_shot_goal_planner_probe_v1 as planner
    folder.mkdir()
    profile = getattr(runtime,'reset_profile','legacy')
    offset_reset = profile in ('body-up','body-drop')
    height = 400. if offset_reset else RESET_RELEASE_WORLD_Z_MM
    minimum_radius = getattr(runtime,'reset_minimum_radius_mm',0.)
    report=dict(status='PREPARING',seed=seed,center_radius_mm=radius_mm,minimum_radius_mm=minimum_radius,
                physical_execution=False,acquisition='NOT_ASSESSED',max_retries=0,
                release_policy='GRASP_PROJECTION_ON_LIVE_TCP_BISECTOR',release_world_z_mm=height,
                reset_profile=profile)
    try:
        before.validate_for(runtime.config,live=True,check_age=False)
        measurement=before
        guidance=''
        if getattr(before,'meta',{}).get('dual_cloud'):
            from .unfold_topdown import primary, guidance_for
            measurement=primary(before,folder)
            guidance='\n'+guidance_for(before)+'\n本次只提供一张当前融合俯视图；没有目标图。center_px和radius_px均使用该图坐标。'
        save(folder/'observation_used.json',dict(
            source_image=str(before.image),planning_image=str(measurement.image),
            coordinate_frame=getattr(measurement,'plan_coordinates','source_rgb'),
            depth_source='dual_measured_fusion' if measurement is not before else 'source_camera'))
        with Image.open(measurement.image) as image:size=list(image.size)
        body=planner.payload(measurement.image,measurement.image,size)
        body['instructions']=reset.PROMPT+guidance+' Return only schema-valid JSON.'
        if offset_reset:
            body['instructions']=('Locate a solid visible body-fabric patch roughly one third of the way '
                'from the neckline toward the hem, toward the body center; avoid the neck opening, '
                'sleeves, seams at the outer edge and bare table. Return center_px and a small reliable '
                'radius_px entirely on this fabric patch. If unavailable, return confidence below 0.8. '
                'This is a single pickup/drop reset; the host lifts to world Z=400 mm, moves to the '
                'arm partition line then 100 mm toward the far side of the table (image up), and releases. '
                'Do not infer that hidden corners are guaranteed. Return only schema-valid JSON.'+guidance)
            if profile=='body-drop':
                body['instructions']=body['instructions'].replace(
                    'Locate a solid visible body-fabric patch roughly one third of the way '
                    'from the neckline toward the hem, toward the body center; avoid the neck opening, ',
                    'Locate the center of the garment body: halfway from neckline to hem and halfway '
                    'between the left and right body edges, excluding the sleeves. Set center_px at '
                    'this central visible fabric, not at the center of the image or table. Printed fabric '
                    'is graspable; do not shift to one side to avoid a print. Avoid the neck opening, ')
                body['instructions']=body['instructions'].replace(
                    'moves to the arm partition line then 100 mm toward the far side of the table (image up), and releases.',
                    'moves to the arm partition line at the same height, then releases without any additional forward/backward offset.')
            body['instructions']=body['instructions'].replace(
                'the host lifts to world Z=400 mm, moves to the arm partition line',
                'the host first lifts 70 mm above the measured cloth surface, moves to the arm partition line at that clearance height, then raises to world Z=400 mm').replace(
                'world Z=400 mm at the same height', 'world Z=400 mm')
        if minimum_radius > 0:
            body['instructions']+=f' Host samples a grasp {minimum_radius:g}–{radius_mm:g} mm from this body center in measured world XY. Return the largest reliable fabric disk around this center, large enough to include that ring where visible; keep it inside the garment and do not move the center to an edge.'
        body['input'][0]['content']=[dict(type='input_text',text=f'CURRENT RGB dimensions: {size} pixels.'),
            dict(type='input_image',image_url=planner.image_data(measurement.image),detail='high')]
        body['text']['format'].update(name='initialization_cloth_region',schema=reset.REGION)
        region,metrics=direct(folder/'region_request',body,headers,timeout)
        save(folder/'region.json',region);save(folder/'model_metrics.json',metrics)
        planner.jsonschema.validate(region,reset.REGION)
        reset.validate_region(region,size)
        seeds=np.random.SeedSequence(seed).spawn(1)[0].generate_state(2,dtype=np.uint64)
        repairs=[]
        try:
            point=reset.sample_grasp(measurement,region,None,radius_mm,int(seeds[0]),depth_repairs=repairs,
                                    **({'minimum_radius_mm':minimum_radius} if minimum_radius else {}))
        finally:
            save(folder/'local_depth_repairs.json',dict(repairs=repairs,raw_cloud_modified=False))
        point['coordinate_frame']=getattr(measurement,'plan_coordinates','source_rgb')
        if hasattr(measurement,'reference'):
            point['source_reference']=measurement.reference(point['pixel_xy'])
        release=dict(release_world_z_mm=height,yaw_delta=0.,drop_offset_xy=[0.,0.])
        save(folder/'grasp.json',point);save(folder/'release.json',release)
        reset.preview(measurement.image,region,point,folder/'grasp_preview.png')
        if offset_reset:
            source=prepare_reset(runtime,before,point,release,folder,
                grasp_yaw_offset_deg=90.,release_world_z_mm=height,
                **(dict(far_translation_mm=0.,release_at_grasp=False) if profile=='body-drop'
                   else dict(far_translation_mm=100.)))
            from .reset_parking import find_parking, execute_parking
            parking=find_parking(runtime,before,source,folder)
            source=parking.get('reset_source',source)
            if parking.get('parking_required',True):
                before=execute_parking(runtime,before,parking,folder)
        else:
            source=prepare_reset(runtime,before,point,release,folder)
        report.update(runtime.execute(source,before,folder/'execution',post_release_descent_mm=RESET_RETURN_DESCENT_MM))
        if report['status']=='EXECUTED':time.sleep(2.)
        return report
    except BaseException as exc:
        report.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:save(folder/'result.json',report)


def configure_runtime(runtime):
    runtime.workspace_arm_fallback = True
    runtime.task_policy = dict(runtime.task_policy, approach_clearance_mm=50.,
                              retreat_clearance_mm=30., grasp_surface_offset_mm=GRASP_SURFACE_OFFSET_MM,
                              far_grasp_world_y_max_mm=-400., far_grasp_raise_mm=2.)
    return runtime


def add_previous_image(body, folder, previous_context, *, current_zed=None, current_camera_label="ZED"):
    from .one_shot_goal_planner_probe_v1 import image_data, sha
    folder = Path(folder)
    context = dict(history=False, input_images=['current', 'goal'])
    if current_zed is not None:
        destination = folder/'context_current_zed.png'
        shutil.copyfile(current_zed, destination)
        body['instructions'] = body['instructions'].replace(
            '每次独立规划，仅依据这两张图，不提供历史图或上一轮动作。',
            f'结合当前俯视图、目标图和同次观测的{current_camera_label}原图规划。')
        guidance = (f'第三张图是同次观测的当前{current_camera_label}原始RGB，辅助判断衣物结构、折叠和遮挡；'
                    '第一张图是当前融合topdown，第二张图是目标。'
                    '两张当前图视角不同，不能直接混用像素坐标。'
                    '所有抓放点、边界点和图像夹爪角度只使用第一张topdown坐标；'
                    f'{current_camera_label}原图不作为选点坐标系，也不替代实测深度支持。')
        body['instructions'] += '\n【当前双视图】\n'+guidance
        body['input'][0]['content'].extend([
            dict(type='input_text', text=f'第三张：当前{current_camera_label}原始RGB，辅助理解结构；选点仍使用第一张topdown。'),
            dict(type='input_image', image_url=image_data(destination), detail='high')])
        context['current_zed'] = dict(source=str(current_zed), image=str(destination), sha256=sha(destination))
    image_count = sum(item['type']=='input_image' for item in body['input'][0]['content'])
    if image_count == 3:
        context['input_images'].append('current_zed')
    previous_label = '第四张' if image_count == 3 else '第三张'
    if previous_context is not None:
        source = previous_context.get('planning_before_image') or previous_context['before_image']
        destination = folder/'context_previous_before.png'
        shutil.copyfile(source, destination)
        body['instructions'] = body['instructions'].replace(
            '每次独立规划，仅依据这两张图，不提供历史图或上一轮动作。',
            '结合当前图、目标图、上一轮动作前图及上一轮动作记录规划本轮动作。')
        guidance = (f'{previous_label}图是上一轮动作前的俯视图，第一张图是本轮当前俯视图，也是上一轮动作后的观察。'
                    f'对比{previous_label}与第一张，判断衣物哪些部位发生变化，并据此选择本轮动作和拉动幅度。'
                    '本轮所有抓放点、边界点和夹爪角度只使用第一张图坐标，不照搬历史图坐标。'
                    f'上一轮动作记录中的像素坐标属于{previous_label}图；EXECUTED仅表示命令执行，不证明抓住布料或展开成功。'
                    '看不清变化时明确说明不确定。')
        body['instructions'] += '\n【上一轮图像参考】\n'+guidance
        body['input'][0]['content'].extend([
            dict(type='input_text', text=f'{previous_label}：上一轮动作前图，仅用于比较变化；本轮选点使用第一张当前图。'),
            dict(type='input_image', image_url=image_data(destination), detail='high')])
        previous_action = previous_context.get('action_record')
        if previous_action is not None:
            body['input'][0]['content'].append(dict(type='input_text', text=
                '上一轮已执行动作（计划抓放参数，非实测夹爪轨迹；结合前后图判断效果）：\n'
                +json.dumps(previous_action, ensure_ascii=False)))
            context['previous_action'] = previous_action
        context['input_images'].append('previous_before')
        context.update(history=True,
                       source=str(source), image=str(destination), sha256=sha(destination),
                       source_round=previous_context.get('source_round'))
    save(folder/'planning_context.json', context)
    return body


def final_completion_check(root, folder, before, headers, timeout, direct):
    """One final visual decision using the existing transport; never build motion."""
    from . import one_shot_goal_planner_probe_v1 as io
    from .unfold_topdown import primary
    from .unfold_completion import ARM6_DONE_RULE
    from .unfold_goal_annotation import apply_goal_annotation
    view=primary(before,folder)
    body=io.payload(view.image,Path(root)/'goal/target.png',Image.open(view.image).size)
    body['instructions']=('第一张是当前最终图，第二张是展开目标。只判断是否完成，不规划动作。'
        +ARM6_DONE_RULE+'\n满足上述完成条件返回DONE；未满足或无法确认返回ABSTAIN，并简述原因。')
    schema=dict(type='object',additionalProperties=False,
        properties=dict(decision=dict(type='string',enum=['DONE','ABSTAIN']),short_reason=dict(type='string')),
        required=['decision','short_reason'])
    body['text']['format'].update(name='unfold_final_completion',schema=schema)
    apply_goal_annotation(root,folder,body)
    save(Path(folder)/'request_payload.json',body)
    value,metrics=direct(Path(folder)/'completion_check',body,headers,timeout)
    io.jsonschema.validate(value,schema)
    save(Path(folder)/'plan.json',value)
    save(Path(folder)/'model_metrics.json',metrics)
    return value,metrics


def plan_round(root, folder, before, headers, timeout, direct, previous_context=None, *, runtime=None, planning_mode="skills", forced_skill=None):
    if previous_context and previous_context.get('final_completion_only'):
        return final_completion_check(root,folder,before,headers,timeout,direct)
    if forced_skill is not None:
        from .skill_gate_online import plan_forced
        return plan_forced(root,folder,before,headers,timeout,direct,runtime=runtime,forced_skill=forced_skill)
    if planning_mode == "pure-oneshot":
        return plan_round_pure(root,folder,before,headers,timeout,direct,runtime=runtime,previous_context=previous_context)
    from cloth_agent import restricted_context_one_shot as planner, preference_one_shot as shared
    from .unfold_skill_planner import UNFOLD_PROMPT
    prompt = UNFOLD_PROMPT
    prompt += ('\n当前设备是固定 ZED 和相向摆放的两台 Arm7。宿主按两夹爪当前中心的 XY 中垂线分区，'
               '优先由抓点所在侧的臂执行；若完整动作超出该臂工作范围，则检查另一臂，只有通过范围、IK和碰撞检查才切换；始终只有一臂运动。仍只规划一个Pa/Pb，不能指定同时双臂动作。'
               'Host用当前实测深度，抓取表面下探20毫米；不自行扣减。目标图用于展开结构，不直接照搬目标图像素作抓放点。'
               'world_mm是标定世界坐标，不是任一机械臂的基座坐标。未提供对应坐标变换与范围时，'
               '不得凭world坐标数值或正负判断抓点越界；可达性、工作范围和碰撞由Host检查。')
    prompt += '\n不设固定拉距上下限；按当前衣物形态选择抓放点，抓放点及完整路径须处于可用工作范围内，并通过IK和碰撞检查。'
    prompt += '\n抓放点须位于对应执行臂的工作范围及画面内。'
    from .unfold_planning_skills import append as append_planning_skills
    prompt = append_planning_skills(root,folder,prompt)
    from .unfold_topdown import primary, guidance_for
    planning_before=primary(before,folder)
    prompt=prompt.replace('第一张原图','第一张当前俯视图').replace(
        '结合桌边透视判断桌面方向','根据俯视图中的桌面边缘判断桌面方向')
    prompt += '\n'+guidance_for(before)
    if runtime is not None:
        from .unfold_planning_skills import workspace_prompt
        prompt += '\n'+workspace_prompt(runtime,before,planning_before,folder)
    if getattr(planning_before,'plan_coordinates',None)=='topdown':
        prompt += '\n当前主视图来自两台相机；残余接缝不代表衣物折叠。显示投影平面不是抓取高度，选点会回映射到原相机实测深度。Host使用融合点云中的实测坐标，并用两台相机的局部同表面三维观测确认支持；支持不足会反馈失败位置，最多重新选点两次，不填补空洞或猜测深度。'
    if previous_context and previous_context.get('skill_followup'):
        prompt += ('\n【同一round的第二段：腾空间后揭开】上一段已执行反方向腾空间，当前图是执行后新观测。'
                   '请重新定位原覆盖区域和轮廓，优先用几何Skill正向揭开，不沿用上一段像素。'
                   '不得再次选择make-space-for-skill；若覆盖已打开可用Simple或DONE，无法提出可行动作则ABSTAIN。')
    body = shared.make_payload(root,folder,planning_before,prompt,planner.SCHEMA,
                               previous_context=None,assess_previous_effect=False)
    add_previous_image(body,folder,previous_context)
    (Path(folder)/'planner_prompt.txt').write_text(body['instructions'],encoding='utf-8')
    body['text']['format']['name']='partitioned_unfold_action'
    from .unfold_skill_planner import run as run_skill_planner
    value,metrics = run_skill_planner(root,folder,planning_before,body,headers,timeout,direct)
    return finish_plan(folder,before,planning_before,value,metrics)


def plan_round_pure(root,folder,before,headers,timeout,direct,*,runtime=None,max_attempts=3,previous_context=None):
    """Current topdown + goal (+ previous topdown); no skills or auxiliary model calls."""
    from . import one_shot_goal_planner_probe_v1 as io
    from .unfold_topdown import primary, guidance_for
    from .unfold_skill_planner import run as run_skill_planner
    planning_before=primary(before,folder)
    size=Image.open(planning_before.image).size
    body=io.payload(planning_before.image,Path(root)/'goal/target.png',size)
    body['instructions']='把第一张图中的衣物展平至第二张目标图的状态。Simple选点时，避免只夹最外沿的一小撮；在同一可见衣袖或下摆边缘带稍向布面内侧选点，以夹入足够布料，避免明显多层堆叠。这是调整抓点，不是额外下压。衣身和两侧袖子均已展开，袖子不再明显收在衣身内，且没有明显大面积折叠或覆盖时，返回DONE。袖子尚未展开或仍被压在衣身内时不能视为完成。衣领必须显露：衣领区域可辨认、未被大块布层覆盖即可；不要求领口撑开，也不要求完整领圈清晰可见。允许局部皱褶和轻微不对称，但袖子相对衣身的伸出长度应尽量接近目标图；若袖子仍明显短缩，应继续展开，不能仅因袖口可见就判定DONE。伸出程度已接近目标后，不要仅为消除褶皱继续拉扯。未完成时输出抓点grasp_px和放置点target_px，无法选点时ABSTAIN。坐标为当前图整数像素。'
    from .unfold_completion import update_completion
    body['instructions']=update_completion(body['instructions'])
    body['instructions'] += '\n不设固定拉距上下限；按当前衣物形态选择抓放点，抓放点及完整路径须处于可用工作范围内，并通过IK和碰撞检查。'
    body['instructions'] += '\n'+guidance_for(before)
    if runtime is not None:
        from .unfold_planning_skills import workspace_prompt
        body['instructions'] += '\n'+workspace_prompt(runtime,before,planning_before,folder)
    add_previous_image(body,folder,previous_context)
    contract='不使用skill：skill、boundary_point_px为null，chord_endpoints_px=[]，pull_distance_mm=10（未使用占位值）。region_name/evidence简述动作区域及理由；非ACTION时grasp_px、target_px为null。'
    (Path(folder)/'planner_prompt.txt').write_text(body['instructions']+'\n'+contract,encoding='utf-8')
    save(Path(folder)/'planning_skills.json',dict(mode='pure_oneshot',skills=[],local_cv_executed=False,semantic_review=False))
    value,metrics=run_skill_planner(root,folder,planning_before,body,headers,timeout,direct,selection_prompt=contract,max_attempts=max_attempts)
    return finish_plan(folder,before,planning_before,value,metrics)


def finish_plan(folder,before,planning_before,value,metrics):
    from cloth_agent import restricted_context_one_shot as planner
    save(Path(folder)/'topdown_plan.json',value)
    value=planning_before.raw_plan(value)
    save(Path(folder)/'pixel_mapping.json',dict(topdown_plan=json.loads((Path(folder)/'topdown_plan.json').read_text(encoding='utf-8')),
        execution_plan=value,plan_coordinates=getattr(planning_before,'plan_coordinates','source_rgb'),
        source_references=({k:planning_before.reference(value[k]) for k in ('grasp_px','target_px')} if value['decision']=='ACTION' and hasattr(planning_before,'reference') else None),source_image=str(before.image)))
    save(folder/'model_metrics.json',metrics)
    plan_image=planning_before.image if getattr(planning_before,'plan_coordinates',None)=='topdown' else before.image
    plan = planner.validate(value,Image.open(plan_image).size,False)
    save(folder/'plan.json',plan)
    from .one_shot_goal_planner_probe_v1 import overlay
    overlay(plan_image,plan,folder/'proposed_action_overlay.png','DUAL ARM7 UNFOLD')
    return plan,metrics


def continuation_state(directory):
    """Read immutable provenance; never reuse saved RGB-D as a live observation."""
    directory = Path(directory).resolve()
    protocol = json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
    summary = json.loads((directory/'episode_summary.json').read_text(encoding='utf-8'))
    if summary.get('status') in ('STARTED', 'RUNNING'):
        raise ValueError('cannot continue a run still marked active')
    steps = summary.get('steps', [])
    last = steps[-1] if steps else None
    start = int(last['round']) if last else 0
    if last and any(a.get('execution', {}).get('physical_execution') for a in last.get('sub_actions', [])):
        start += 1
    elif last and last.get('execution', {}).get('physical_execution'):
        start += 1
    feedback = dict(rejection_reason='CONTINUED_RUN', parent_run=str(directory),
                    error=summary.get('error') or summary.get('reason'),
                    instruction='已重新观察当前衣物；结合当前图重新规划，不重复被拒动作，不沿用旧图坐标。')
    # Preserve the rejected execution evidence, not only a later ABSTAIN reason.
    last_execution = last.get('execution', {}) if last else {}
    if last_execution.get('physical_execution') is False and last_execution.get('error'):
        feedback['last_execution_error'] = last_execution['error']
        feedback['last_execution_physical_execution'] = False
    return dict(parent_run=str(directory), start_round=start, protocol=protocol,
                prior_physical_actions=summary.get('physical_actions', 0), feedback=feedback)


def episode(runtime, root, *, max_actions, request, reset_request=None, max_plan_attempts=3, progress=False, reset_repeats=1, reset_only=False, continuation=None):
    """One new episode; capture/check/execute one complete action at a time."""
    from .trajectory_memory_full_episode_v1 import repeated_action
    if max_plan_attempts not in (1,2,3):raise ValueError('planning attempts must be 1..3')
    if reset_repeats not in (1,2) or (reset_only and reset_request is None):
        raise ValueError('reset-only requires initialization; reset repeats must be 1 or 2')
    history=[];context=None
    def show(label,message,**kwargs):
        if progress:console.emit(label,message,**kwargs)
    def observe(directory, *, reset_zed=False):
        directory=Path(directory)
        attempts=[]
        for attempt in range(CAPTURE_ATTEMPTS):
            target=directory if attempt==0 else directory.with_name(directory.name+f'_retry_{attempt:02d}')
            try:
                with console.stage('采集图像与深度，生成俯视图','capture'):
                    observation=(runtime.observe(target,reset_zed=True) if reset_zed else runtime.observe(target))
                    if hasattr(observation,'xyz') and not reset_zed:
                        from .unfold_topdown import primary
                        primary(observation,observation.directory)
                attempts.append(dict(attempt=attempt+1,status='CAPTURED',directory=str(observation.directory)))
                if len(attempts)>1:save(root/f'{directory.name}_capture_recovery.json',dict(attempts=attempts))
                return observation
            except Exception as exc:
                retryable=retryable_capture_error(exc)
                attempts.append(dict(attempt=attempt+1,status='FAILED',directory=str(target),
                                     retryable=retryable,error=f'{type(exc).__name__}: {exc}'))
                save(root/f'{directory.name}_capture_recovery.json',dict(attempts=attempts))
                if not retryable or attempt==CAPTURE_ATTEMPTS-1:raise
                show('相机重试',f'采集失败，重新连接并重采 {attempt+2}/{CAPTURE_ATTEMPTS}：{exc}',warning=True)
                time.sleep(1.)
        raise AssertionError('capture retry loop exhausted without result')
    result=dict(status='STARTED',physical_actions=0,steps=[])
    if continuation:
        if reset_request is not None:raise ValueError('continuation must not reset')
        result.update(continued_from=continuation['parent_run'], prior_physical_actions=continuation['prior_physical_actions'])
    try:
        before=observe(root/'initial_observation',reset_zed=reset_request is not None and bool(getattr(runtime,'dual_wrist_config',None)))
        if reset_request is not None:
            for reset_index in range(reset_repeats):
                show('初始化',f'第 {reset_index+1}/{reset_repeats} 次抓起并放下衣物')
                reset_folder=root/('initialization' if reset_index==0 else f'initialization_{reset_index+1:02d}')
                with console.stage('自动初始化衣物','startup'):
                    initialization=reset_request(before,reset_folder)
                result['initialization']=initialization
                result.setdefault('initializations',[]).append(initialization)
                if initialization.get('physical_execution'):result['physical_actions']+=1
                if initialization['status']!='EXECUTED':
                    result.update(status='RESET_FAILED',error=initialization.get('error'))
                    return result
                if reset_only or reset_index<reset_repeats-1:
                    before=observe(root/f'after_reset_{reset_index+1:02d}',reset_zed=True)
                else:
                    before=observe(root/'after_reset')
            if reset_only:
                result.update(status='RESET_ONLY_COMPLETED',final_image=str(before.image),reset_count=reset_repeats)
                return result
        index=continuation['start_round'] if continuation else 0;followup=False
        while index<=max_actions:
            stage_root=root/f'round_{index:03d}'
            if followup:stage_root=stage_root/'skill_followup'
            folder=stage_root;folder.mkdir()
            save(folder/'observation_used.json',dict(directory=str(before.directory)))
            if continuation and index==continuation['start_round']:
                save(folder/'rejection_feedback.json',continuation['feedback'])
            if index==max_actions:
                show('最终完成检查','动作次数已到上限，使用最后一次观测判断DONE，不再执行动作')
                result['status']='ACTION_LIMIT'
                check=dict(status='STARTED',observation=str(before.directory),physical_execution=False)
                result['final_completion_check']=check
                try:
                    with console.stage('最终DONE判断（不执行动作）','final_completion'):
                        final_plan,final_metrics=request(folder,before,dict(final_completion_only=True))
                    check.update(status='COMPLETED',plan=final_plan,metrics=final_metrics)
                    save(folder/'final_completion_check.json',check)
                    if final_plan.get('decision')=='DONE':
                        result['status']='MODEL_REPORTED_DONE'
                        result['reason']=final_plan.get('short_reason','最终观察已达到完成条件')
                    else:
                        result['reason']='达到动作上限；最终检查未确认DONE：'+final_plan.get('short_reason','未确认完成')
                except Exception as exc:
                    check.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
                    result['reason']='达到动作上限；最终DONE检查失败：'+check['error']
                    save(folder/'final_completion_check.json',check)
                show('原因',result['reason'])
                break
            if progress:console.write(f'\n  ── 第 {index+1:02d}/{max_actions} 步 · round_{index:03d} ──')
            if not followup:
                row=dict(round=index,decision=None,execution={'status':'NOT_DISPATCHED'},sub_actions=[])
                result['steps'].append(row)
            phase='skill_followup' if followup else 'initial'
            if followup:show('技能第二段','已重新拍照，重新规划正向揭开')
            for attempt in range(max_plan_attempts):
                show('规划',f'等待模型返回'+(f' · 重规划 {attempt+1}/{max_plan_attempts}' if attempt else ''))
                planning_started=time.monotonic()
                try:
                    with console.stage('规划选点与计算几何动作','planning'):
                        plan,metrics=request(folder,before,context)
                except Exception as exc:
                    # Planning has no motion dispatch: preserve its error for a new request.
                    feedback=dict(rejection_reason='PLANNER_ERROR', physical_execution=False,
                                  error=f'{type(exc).__name__}: {exc}')
                    save(folder/'planning_error.json',feedback)
                    row.setdefault('planning_rejections',[]).append(feedback)
                    show('规划重试',feedback['error'],warning=True)
                    if attempt==max_plan_attempts-1:
                        # The planner itself never produced a valid decision.
                        # Do not repeatedly reset garments while the model transport
                        # or response contract is unavailable.
                        result.update(status='PLANNER_UNAVAILABLE',error=feedback['error'])
                        return result
                    delay=planner_retry_delay(exc,attempt)
                    if delay:
                        show('服务退避',f'模型服务限流/暂不可用，等待 {delay:.0f} 秒后重试',warning=True)
                        time.sleep(delay)
                    folder=stage_root/f'replan_{attempt+1:02d}'
                    folder.mkdir()
                    save(folder/'observation_used.json',dict(directory=str(before.directory)))
                    save(folder/'rejection_feedback.json',feedback)
                    continue
                show('规划完成',f'{time.monotonic()-planning_started:.1f} 秒 · {plan["decision"]}')
                row['decision']=plan['decision']
                save(root/'episode_summary.json',result)
                if plan['decision']!='ACTION':
                    row.setdefault('planning_decisions',[]).append(dict(phase=phase,decision=plan['decision'],reason=plan.get('short_reason'),folder=str(folder)))
                    result['status']='MODEL_REPORTED_DONE' if plan['decision']=='DONE' else 'ABSTAIN'
                    result['reason']=plan.get('short_reason','模型未返回可执行动作')
                    show('原因',console.brief(plan.get('short_reason','')))
                    return result
                manifest_path=folder/'planning_skills.json'
                selected_skill=(json.loads(manifest_path.read_text(encoding='utf-8')).get('selected_skill')
                                if manifest_path.exists() else None)
                from .unfold_planning_skills import SPACE_PREPARATION
                preparing_space=selected_skill==SPACE_PREPARATION
                try:
                    if repeated_action(plan,before.image,history):
                        raise DualArmError('unchanged observation and repeated action; choose a different action')
                    if followup and preparing_space:
                        raise DualArmError('second action cannot prepare space again; replan the reveal or ABSTAIN')
                    with console.stage('换算实测抓放点与机械臂分区','geometry'):
                        source=prepare_action(runtime,before,plan,folder)
                except TranslationRejected as exc:
                    show('距离不符',str(exc),warning=True)
                    feedback=dict(exc.feedback, rejected_plan=plan,
                        rejection_reason='XY_TRANSLATION_OUT_OF_RANGE', coordinate_frame='world_mm',
                        workspace_check='NOT_EVALUATED', collision_check='NOT_EVALUATED')
                    save(folder/'translation_rejection.json',feedback)
                    row.setdefault('planning_rejections',[]).append(feedback)
                    if attempt==max_plan_attempts-1:
                        result.update(status='PLANNING_REJECTED',error=str(exc))
                        return result
                except Exception as exc:
                    # prepare_action only compiles; no robot command has been sent.
                    feedback=dict(rejection_reason='ACTION_PREPARATION_FAILED',
                        physical_execution=False,error=f'{type(exc).__name__}: {exc}',rejected_plan=plan)
                    save(folder/'preparation_rejection.json',feedback)
                    row.setdefault('planning_rejections',[]).append(feedback)
                    show('选点重试',feedback['error'],warning=True)
                    if attempt==max_plan_attempts-1:
                        result.update(status='PLANNING_REJECTED',error=feedback['error'])
                        return result
                else:
                    geometry_file=folder/'geometry.json'
                    if progress and geometry_file.exists():
                        geometry=json.loads(geometry_file.read_text(encoding='utf-8'))
                        arm=geometry.get('assignment',{}).get('active_arm','未知')
                        show('动作',f'{arm} 臂 · XY {geometry["distance_mm"]:.1f} mm · 抬升 {geometry.get("lift_clearance_mm",0):.0f} mm')
                    show('运动','IK / 路径检查，通过后执行；此处尚不代表已开始运动')
                    execution_started=time.monotonic()
                    with console.stage('求解 IK、检查完整路径并执行抓放','motion'):
                        outcome=runtime.execute(source,before,folder/'execution')
                    show('执行结果',f'{outcome.get("status")} · {time.monotonic()-execution_started:.1f} 秒',warning=outcome.get('status')!='EXECUTED')
                    if outcome.get('error'):show('原因',console.explain(outcome['error']),warning=True)
                    save(folder/'execution.json',outcome)
                    failure=outcome.get('planning_failure',{})
                    # Any rejected action can be replanned when explicitly undispatched.
                    # Never infer absence of motion from the error name or missing metadata.
                    if outcome.get('status') == 'EXECUTED' or outcome.get('physical_execution') is not False:
                        break
                    feedback=dict(rejection_reason='ACTION_FAILED', rejected_plan=plan,
                        physical_execution=False, error=outcome.get('error') or outcome.get('status'),
                        planning_failure=failure, execution=outcome)
                    row['execution']=outcome
                    row.setdefault('planning_rejections',[]).append(feedback)
                    save(folder/'execution_rejection.json',feedback)
                    show('动作重规划','未执行，将原始错误交回规划器重新选点',warning=True)
                    if attempt==max_plan_attempts-1:
                        result.update(status='PLANNING_REJECTED',error=outcome.get('error'))
                        return result
                folder=stage_root/f'replan_{attempt+1:02d}'
                folder.mkdir()
                save(folder/'observation_used.json',dict(directory=str(before.directory)))
                save(folder/'rejection_feedback.json',feedback)
            row['execution']=outcome;save(folder/'execution.json',outcome)
            row['sub_actions'].append(dict(phase=phase,skill=selected_skill,plan=plan,
                                          execution=outcome,folder=str(folder)))
            # Preserve possible movement on failure; never report it as zero.
            if outcome.get('physical_execution'):result['physical_actions']+=1
            save(root/'episode_summary.json',result)
            if outcome['status']!='EXECUTED':
                result.update(status='EXECUTION_FAILED',error=outcome.get('error'));break
            after=observe(folder/'after')
            history.append(dict(plan=plan,before=before.image,after=after.image,
                                acquisition='NOT_ASSESSED'))
            actions=outcome.get('experiment',{}).get('actual_robot_actions',[])
            context=dict(plan=plan,before_image=before.image,after_image=after.image,
                         completed_commands=len(actions),source_round=folder.name)
            if (folder/'topdown_plan.json').exists():
                context.update(planning_plan=json.loads((folder/'topdown_plan.json').read_text(encoding='utf-8')),
                               planning_before_image=folder/'topdown/planning_rgb.png')
            planning_plan = context.get('planning_plan', plan)
            geometry_path = folder/'geometry.json'
            geometry = json.loads(geometry_path.read_text(encoding='utf-8')) if geometry_path.exists() else {}
            selection_path = folder/'skill_selection/raw_output.txt'
            selection = json.loads(selection_path.read_text(encoding='utf-8')) if selection_path.exists() else {}
            context['action_record'] = dict(
                source_round=f'round_{index:03d}', source_attempt=str(folder.relative_to(root)),
                skill=selection.get('skill'),
                grasp_px=planning_plan.get('grasp_px'), target_px=planning_plan.get('target_px'),
                pixel_frame='previous_before', motion_type=planning_plan.get('motion_type'),
                purpose=planning_plan.get('action_purpose'), reason=planning_plan.get('short_reason'),
                planned_xy_distance_mm=geometry.get('distance_mm'),
                grasp_world_mm=geometry.get('grasp_world_mm'), target_world_mm=geometry.get('target_world_mm'),
                execution_status=outcome['status'], physical_execution=outcome.get('physical_execution'),
                error=outcome.get('error'), grasp_success='NOT_ASSESSED')
            before=after
            row['sub_actions'][-1]['after_observation']=str(after.directory)
            if preparing_space and not followup:
                followup=True
                context['skill_followup']=True
            else:
                followup=False
                index+=1
            save(root/'episode_summary.json',result)
        return result
    except KeyboardInterrupt:
        result['status']='INTERRUPTED';raise
    except Exception as exc:
        result.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
        (root/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        save(root/'episode_summary.json',result)
        with console.stage('保存录像并关闭本次设备连接','video'):
            runtime.close()


def _main(argv=None):
    from datetime import datetime
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continue-from',type=Path,help='continue a saved run from a fresh observation, without reset; save into a new output directory')
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--goal-image',type=Path,default=DEFAULT_GOAL)
    parser.add_argument('--capture-goal',action='store_true',help='capture currently unfolded garment as the goal before the first reset; sequential runs reuse this goal')
    parser.add_argument('--reset-profile',choices=['legacy','body-up','body-drop'],default='legacy')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-actions',type=int,default=20)
    parser.add_argument('--json-summary',action='store_true',help='also print full final JSON; always saved to episode_summary.json')
    parser.add_argument('--planning-mode',choices=['skills','pure-oneshot'],default='skills',
                        help='pure-oneshot: original minimal current + goal prompt, no skills')
    parser.add_argument('--planner-transport', choices=['openai-cli','rbs'], default='openai-cli',
                        help='model transport; openai-cli uses existing OpenAI login, rbs restores prior HTTP transport')
    parser.add_argument('--timeout-s',type=float,default=900)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--camera-mode',choices=['legacy','dual-wrist'],default='legacy')
    parser.add_argument('--fixed-wrist-config',type=Path,default=ROOT/'config/calibration/dual_wrist_relaxed_20261110/extrinsics.json')
    parser.add_argument('--topdown-config',type=Path,default=ROOT/'config/topdown.dual_wrist_rgb.json')
    parser.add_argument('--capture-only',action='store_true',help='move through existing startup/perception sequence and save RGB-D/topdown only; no planner or cloth action')
    parser.add_argument('--single-camera',action='store_true',help='use original ZED-only observation and top-down')
    parser.add_argument('--fusion-config',type=Path,default=ROOT/'config/fusion.unfold_dual.json')
    parser.add_argument('--no-video',action='store_true',help='disable continuous ZED RGB recording')
    parser.add_argument('--video-dir',type=Path,default=ROOT/'videos',help='video storage root; per-run subdirectory')
    parser.add_argument('--reset-only',action='store_true',help='perform initialization and capture its result; no unfold')
    parser.add_argument('--reset-repeats',type=int,choices=[1,2],default=1)
    parser.add_argument('--no-reset',action='store_true',help='use manually arranged cloth; skip initialization')
    parser.add_argument('--initialize-seed',type=int)
    parser.add_argument('--initialize-center-radius-mm',type=float,help='override sampling disk radius, 0..80 mm; body-drop/up default: 40..60 mm annulus')
    args=parser.parse_args(argv)
    continuation=None
    if args.continue_from:
        if args.capture_goal or args.reset_only or args.capture_only:
            parser.error('continue-from cannot capture a goal, reset-only or capture-only')
        continuation=continuation_state(args.continue_from)
        prior=continuation['protocol']
        args.no_reset=True
        args.goal_image=args.continue_from.resolve()/'goal/target.png'
        args.config=Path(prior['config'])
        args.camera_mode=prior.get('camera_mode','legacy')
        args.planning_mode=prior.get('planning_mode','skills')
        args.planner_transport=prior.get('planner_transport','openai-cli')
        args.max_actions=prior['max_actions']
        if args.camera_mode=='dual-wrist':
            args.fixed_wrist_config=Path(prior['fusion_config'])
            args.topdown_config=Path(prior['topdown_config'])
        elif prior.get('fusion_config'):
            args.fusion_config=Path(prior['fusion_config'])
        else:
            args.single_camera=True
        if continuation['start_round']>=args.max_actions:
            parser.error('source run has reached its action limit')

    if args.reset_only and (args.no_reset or args.capture_only):parser.error('reset-only cannot combine with no-reset/capture-only')
    if args.capture_goal and (args.capture_only or args.no_reset or args.reset_only):parser.error('capture-goal requires a normal unfold run with reset')
    if args.initialize_seed is not None and args.initialize_seed<0:parser.error('seed must be nonnegative')
    if args.initialize_center_radius_mm is not None and not 0<=args.initialize_center_radius_mm<=80:parser.error('reset radius must be 0..80 mm')
    seed=args.initialize_seed if args.initialize_seed is not None else secrets.randbits(63)
    radius=args.initialize_center_radius_mm
    default_annulus = radius is None and args.reset_profile in ('body-up','body-drop')
    if radius is None:radius=60. if default_annulus else float(np.random.default_rng(seed).uniform(20,40))
    if args.max_actions<1 or args.timeout_s<=0:parser.error('invalid action/time limit')
    if not args.capture_goal and not args.goal_image.is_file():parser.error('goal image missing')
    started=time.monotonic()
    root=(args.output or ROOT/'reports'/('arm7_unfold_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    root.mkdir(parents=True,exist_ok=False);(root/'goal').mkdir()
    if continuation:save(root/'continuation.json',continuation)
    console.open_session(root)
    runtime=configure_runtime(PartitionedReplayRuntime(args.config,planning_only=not args.execute))
    runtime.reset_minimum_radius_mm = 40. if default_annulus else 0.
    runtime.reset_profile=args.reset_profile
    if args.camera_mode=='dual-wrist':
        if args.single_camera:parser.error('--dual-wrist mode cannot use --single-camera')
        from cloth_agent.dual_wrist_observation import load_fixed
        runtime.dual_wrist_config=load_fixed(args.fixed_wrist_config)
        runtime.topdown_config=json.loads(args.topdown_config.read_text(encoding='utf-8'))
        if runtime.topdown_config.get('appearance_render')!='rgb_mosaic':parser.error('dual-wrist mode requires RGB-first topdown')
        if runtime.config_path!=(ROOT/runtime.dual_wrist_config['scene']).resolve():parser.error('dual-wrist requires its camera/support collision scene')
        runtime.preparation_description='依次张开夹爪、回 Home；每次采图前两臂依次到固定双俯拍位置。'
    elif not args.single_camera:
        from cloth_agent.dual_cloud_observation import load_config
        runtime.dual_cloud_config=load_config(args.fusion_config)
    if not args.capture_goal:shutil.copyfile(args.goal_image,root/'goal/target.png')
    goal_boundary=args.goal_image.with_suffix('.boundary.json')
    if not args.capture_goal and goal_boundary.is_file():
        shutil.copyfile(goal_boundary,root/'goal/target.boundary.json')
    reorient_enabled=False
    console.start(root,root/'goal/target.png' if args.capture_goal else args.goal_image.resolve(),args.max_actions,not args.no_reset,args.planner_transport,reorient_enabled)
    save(root/'hem_sleeve_pull_rule.json',dict(enabled=False,rgb_only=True))
    save(root/'protocol.json',dict(mode='PARTITIONED_DUAL_ARM7_UNFOLD',planner='single-step current topdown + goal + current ZED RGB + previous-before topdown',
        planning_mode=args.planning_mode,planner_transport=args.planner_transport,
        planning_skills=[] if args.planning_mode=='pure-oneshot' else ['cloth-contour-normal-grasp','garment-tangent-grasp','garment-chord-grasp'],
        reorientation_skill=None,
        planning_skills_mode='disabled' if args.planning_mode=='pure-oneshot' else 'cv_geometry_direct',semantic_review=False,planning_history=True,planning_image_count=dict(first_round=3,later_rounds=4),planning_history_mode='previous_before_image_and_executed_action',
        topdown='primary measured world-XY; each point maps to owning camera pixels; rebuilt every observation',
        fusion_config=None if args.single_camera else str(args.fusion_config),
        post_reset_press=False,
        camera='fixed ZED' if args.single_camera else 'fixed ZED + RealSense; depth scale + per-observation color match',graph=False,reset=not args.no_reset,reset_seed=seed,reset_radius_mm=radius,reset_minimum_radius_mm=runtime.reset_minimum_radius_mm,
        reset_only=args.reset_only,reset_repeats=args.reset_repeats,reset_profile=args.reset_profile,
        reset_release_policy={'body-up':'MIDLINE_THEN_WORLD_MINUS_Y_100_MM','body-drop':'GRASP_PROJECTION_ON_LIVE_TCP_BISECTOR','legacy':'GRASP_PROJECTION_ON_LIVE_TCP_BISECTOR'}[args.reset_profile],reset_release_world_z_mm=400. if args.reset_profile in ('body-up','body-drop') else RESET_RELEASE_WORLD_Z_MM,reset_return_descent_mm=RESET_RETURN_DESCENT_MM,grasp_offset_mm=GRASP_SURFACE_OFFSET_MM,max_actions=args.max_actions,
        ownership='partition preferred; alternate arm on workspace rejection; full IK/collision checks retained',
        execution='existing ExperimentRunner / RobotAPI / RobotCollision',
        geometry_report='disabled: old Arm6/Camera A GGO reference is not current calibration',
        goal_source='CAPTURE_BEFORE_RESET' if args.capture_goal else str(args.goal_image.resolve()),capture_goal=args.capture_goal,config=str(args.config.resolve())))
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    protocol['unattended_policy']=dict(
        capture_attempts=CAPTURE_ATTEMPTS,
        next_reset_after=['MODEL_REPORTED_DONE','HUMAN_REPORTED_DONE','ACTION_LIMIT','ABSTAIN','PLANNING_REJECTED'],
        stop_batch_after=['RESET_FAILED','EXECUTION_FAILED','PLANNER_UNAVAILABLE','ERROR','INTERRUPTED'],
        pre_dispatch_plan_attempts=3,
        camera_stream_reconnect_attempts=3,
        physical_or_unknown_motion_failure_retry=False)
    save(root/'protocol.json',protocol)
    if args.camera_mode=='dual-wrist':
        protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
        protocol.update(camera='dual fixed-pose wrist RealSense',camera_mode='dual-wrist',fusion_config=str(args.fixed_wrist_config),topdown='RGB-first mosaic; original color pixel to measured native depth',topdown_config=str(args.topdown_config),planner='current RGB mosaic + goal + original left wrist RGB + optional previous image',capture_only=args.capture_only,absolute_robot_alignment='Existing left hand-eye and dual-arm base anchor; not newly measured',real_cloth_execution_allowed='world_from_right_color_mm' in runtime.dual_wrist_config)
        protocol.update(zed_topdown_supplement=bool(runtime.topdown_config.get('zed_supplement',{}).get('enabled',False)),
                        topdown_canvas=runtime.topdown_config.get('rgb_mosaic'),
                        perception_pose_name=runtime.dual_wrist_config.get('pose_name'),
                        camera_alignment_validation=runtime.dual_wrist_config.get('validation',{}),
                        reset_observation='disabled' if args.no_reset else 'fixed ZED RGB-D with both arms at Home',
                        reset_partition='live TCP bisector at Home',
                        recording_camera='ZED 11483267')
        save(root/'protocol.json',protocol)
    if not args.execute:
        runtime.close()
        mapping_ready=args.camera_mode!='dual-wrist' or 'world_from_right_color_mm' in runtime.dual_wrist_config
        result=dict(status='DRY_RUN_OK',output=str(root),robot_calls=0,model_calls=0,camera_mode=args.camera_mode,real_execution_ready=mapping_ready,pending=None if mapping_ready else 'display-only camera mapping',motion_checks='Evaluated from live state before each execution')
        console.summary(root,result,time.monotonic()-started)
        if args.json_summary:console.write(json.dumps(result))
        save(root/'dry_run.json',result)
        return 0
    if args.camera_mode=='dual-wrist' and not args.capture_only and 'world_from_right_color_mm' not in runtime.dual_wrist_config:
        runtime.close()
        raise ValueError('所选配置是历史仅显示候选，缺少执行用 world_from_right_color_mm；请使用默认双腕 extrinsics.json 配置。')
    if args.planner_transport == 'openai-cli':
        from .openai_cli_planner import direct_once
    else:
        from .one_shot_goal_full_episode_v1 import direct_once
    old=signal.getsignal(signal.SIGTERM)
    def terminate(*_):raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM,terminate)
    try:
        headers={}
        if args.planner_transport == 'rbs':
            from .one_shot_goal_planner_probe_v1 import auth_headers
            headers=auth_headers()
        if not args.no_video:
            from cloth_agent.perception import PerceptionConfig
            from cloth_agent.unfold_video import UnfoldVideo
            # Fixed ZED records the experiment; wrist cameras retain their own
            # native RGB-D capture and planning mappings in dual-wrist mode.
            video_config=PerceptionConfig.load(ROOT,ROOT/'config/perception.zed_dual_xarm7_20261103.json')
            video_camera=None
            if args.camera_mode=='dual-wrist':
                from cloth_agent.zed_rgbd import ZEDRGBD
                video_spec=next(s for s in video_config.cameras if s.label in video_config.active_camera_labels)
                video_camera=ZEDRGBD(video_spec,video_config.width,video_config.height,video_config.fps,rgb_only=(args.no_reset or args.capture_only) and not getattr(runtime,'topdown_config',{}).get('zed_supplement',{}).get('enabled',False))
            recorder=UnfoldVideo(video_config,args.video_dir.resolve()/root.name,root,camera=video_camera)
            runtime.video_recorder=recorder
            with console.stage('打开相机与录像','capture'):recorder.start()
            console.emit('录像',str(recorder.path))
        if args.capture_only:
            observation=runtime.observe(root/'capture')
            from .unfold_topdown import primary
            view=primary(observation,root/'capture_view')
            result=dict(status='CAPTURE_ONLY_COMPLETED',planning_image=str(view.image),cloth_actions=0,model_calls=0)
            save(root/'episode_summary.json',result);console.summary(root,result,time.monotonic()-started)
            return 0
        if args.capture_goal:
            from .unfold_topdown import primary
            with console.stage('采集当前展开衣服作为本轮目标','capture'):
                goal_observation=runtime.observe(root/'goal_observation')
                goal_view=primary(goal_observation,root/'goal_view')
                shutil.copyfile(goal_view.image,root/'goal/target.png')
                save(root/'goal/capture.json',dict(source=str(goal_view.image),
                    observation=str(goal_observation.directory),captured_before_reset=True))
        result=episode(runtime,root,continuation=continuation,max_actions=args.max_actions,progress=True,reset_repeats=args.reset_repeats,reset_only=args.reset_only,
            request=lambda folder,before,context:plan_round(root,folder,before,headers,args.timeout_s,direct_once,context,runtime=runtime,planning_mode=args.planning_mode),
            reset_request=None if args.no_reset else lambda before,folder:initialize(runtime,before,folder,
                headers=headers,timeout=args.timeout_s,direct=direct_once,seed=seed,radius_mm=radius))
        console.summary(root,result,time.monotonic()-started)
        if args.json_summary:console.write(json.dumps(dict(output=str(root),**result),ensure_ascii=False))
        return 0 if result['status'] in ('MODEL_REPORTED_DONE','ACTION_LIMIT','ABSTAIN','PLANNING_REJECTED','RESET_ONLY_COMPLETED') else 1
    except KeyboardInterrupt:
        saved=root/'episode_summary.json'
        result=json.loads(saved.read_text(encoding='utf-8')) if saved.exists() else dict(status='INTERRUPTED')
        result['status']='INTERRUPTED'
        save(saved,result)
        console.summary(root,result,time.monotonic()-started)
        if args.json_summary:console.write(json.dumps(dict(output=str(root),**result),ensure_ascii=False))
        return 130
    except Exception as exc:
        saved=root/'episode_summary.json'
        result=json.loads(saved.read_text(encoding='utf-8')) if saved.exists() else {}
        result.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
        save(saved,result)
        console.summary(root,result,time.monotonic()-started)
        if args.json_summary:console.write(json.dumps(dict(output=str(root),**result),ensure_ascii=False))
        return 1
    finally:
        runtime.close();signal.signal(signal.SIGTERM,old)
        video_file=root/'video_recording.json'
        if video_file.exists():
            video=json.loads(video_file.read_text(encoding='utf-8'))
            if video.get('video'):console.emit('录像文件',str(video['video']))
            if video.get('video_error'):console.emit('录像错误',console.brief(video['video_error']),warning=True)


def main(argv=None):
    try:
        return _main(argv)
    except Exception as exc:
        if console.ACTIVE is None:raise
        root=console.ACTIVE.root
        error=f'{type(exc).__name__}: {exc}'
        (root/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8')
        saved=root/'episode_summary.json'
        result=json.loads(saved.read_text(encoding='utf-8')) if saved.exists() else {}
        result.update(status='ERROR',error=error);save(saved,result)
        console.summary(root,result,0)
        return 1
    finally:
        console.close_session()

if __name__=='__main__':raise SystemExit(main())
