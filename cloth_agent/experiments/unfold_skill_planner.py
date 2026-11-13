"""Layer-aware direct point selection -> common measured-depth execution contract.

No camera/robot operations. Legacy normal geometry helpers remain for offline demonstrations; direct() is the same model transport used by the unfold experiment.
"""
from pathlib import Path
import copy
from collections import Counter
import json
import cv2
import numpy as np
from PIL import Image
from cloth_agent import contour_normal_grasp as normal
from cloth_agent import restricted_context_one_shot as planner
from . import one_shot_goal_planner_probe_v1 as io
from .discovery_data import save
from .unfold_planning_skills import NAMES, SPACE_PREPARATION, apply_switch_rule
from .unfold_completion import update_completion
from . import unfold_console as console

POINT=dict(type='array',items=dict(type='integer'),minItems=2,maxItems=2)
SELECT_SCHEMA=dict(type='object',additionalProperties=False,properties=dict(
    decision=dict(type='string',enum=['ACTION','DONE','ABSTAIN']),
    skill=dict(anyOf=[dict(type='string',enum=[*NAMES, SPACE_PREPARATION]),dict(type='null')]),
    region_name=dict(type='string'),evidence=dict(type='string'),
    grasp_angle_deg=dict(type=['number','null'],minimum=0,maximum=180),
    grasp_px=dict(anyOf=[POINT,dict(type='null')]),
    target_px=dict(anyOf=[POINT,dict(type='null')]),
    boundary_point_px=dict(anyOf=[POINT,dict(type='null')]),
    chord_endpoints_px=dict(type='array',items=POINT,minItems=0,maxItems=2),
    pull_distance_mm=dict(type='number',exclusiveMinimum=0)),
    required=['decision','skill','region_name','evidence','boundary_point_px',
              'chord_endpoints_px','pull_distance_mm','grasp_px','target_px','grasp_angle_deg'])
SELECT_PROMPT='''
先根据目标图选择本步需要展开的部位。该部位外露时直接抓放；被布层覆盖、无法直接抓取时使用几何skill。不要用别处可见的角替代被覆盖的目标，也不要仅因褶皱或内收启动skill。
技能腾空间：先确定要揭开的覆盖区域、正向拉动方向和所需拉距，再比较该方向的桌面余量；衣服居中不代表余量足够。若余量不足，或Host明确报告所选动作因机械臂自碰撞未通过IK检查，可选择skill="make-space-for-skill"，为原定揭开留足空间或把衣物移离容易自碰撞的操作位置；腾空间自身也必须通过IK和完整路径检查，不能复用被拒绝的路径。纠偏/腾空间时，若衣服整体偏离可用桌面中央，按当前俯视图方向选点：偏上→抓衣服最下沿向下拉；偏下→抓衣服最上沿向上拉；偏左→抓图像右侧袖子向右拉；偏右→抓图像左侧袖子向左拉。这四条仅用于纠偏/腾空间，不限制普通展开；没有整体偏移时，仍按原定展开所缺空间选择反方向的可见外缘。放点应争取补足所缺的空间，遵守工作边界和完整路径约束，不要仅象征性挪动；也不能挤出反方向桌边。boundary_point_px=null、chord_endpoints_px=[]。evidence简述原定揭开方向、空间不足或Host自碰撞反馈，以及本次腾空间方向。Host按普通抓放速度完成动作后重新拍照，在同一round再规划一次揭开；本次不要输出第二段坐标。
几何Skill选择顺序：几何Skill优先使用割线garment-chord-grasp；仅当无法可靠选择同一可见边界段的两个端点，或Host反馈割线无法生成有效候选时，才使用切线garment-tangent-grasp。
普通抓放：skill=null，直接给grasp_px、target_px，boundary_point_px=null，chord_endpoints_px=[]。
几何skill：选择被覆盖目标附近的可见边界boundary_point_px，弦线法另给同一边界段两端chord_endpoints_px，其他方法为[]；grasp_px、target_px=null。Host独立分割轮廓，沿内法线取正向最远交点，计算最终抓放点，不做候选语义复核。
region_name/evidence简述目标边角是否可见。pull_distance_mm为期望水平拉距（正数、毫米），不设固定上下限；根据需要展开的范围选择，执行以实测工作边界和完整路径检查为准。非ACTION时skill和所有点为null，chord_endpoints_px=[]，pull_distance_mm=10。
'''
GRASP_ROTATION_PROMPT = '抓取前可调整夹爪方向，抓住后保持方向完成抬升、平移和放下，不允许持布旋转。grasp_angle_deg表示当前topdown图中两指开合轴：图像向右为0°，向上为90°，逆时针增加，范围0–180°。这不是机器人yaw，也不是拉动方向。按最终抓点附近布边选择方向；几何Skill若无法可靠判断最终抓点处方向则填null，沿用执行臂Home方向。DONE/ABSTAIN必须填null。Host按来源相机标定映射方向，并保留原有IK和完整路径检查。'

UNFOLD_PROMPT='''
你是衣物展开的单步规划器。第一张是当前图，第二张是目标。每次独立规划，仅依据这两张图，不提供历史图或上一轮动作。
以展开为主，不为居中单独操作；当下一步展开受桌面空间限制，或Host明确报告自碰撞导致IK失败时，可使用make-space-for-skill腾空间。
默认直接输出抓点和放点；仅当衣角或袖口明显被其他布层压住时使用。
收尾时若衣领被布层盖住，可直接抓住可见的衣服下摆，稍微向图像下方拉动以显露衣领，使用 Simple。
仅使用lift_pull_place，持布时不旋转；抓取方向保持所属机械臂既有Home姿态。
Simple选点时，避免只夹最外沿的一小撮；在同一可见衣袖或下摆边缘带稍向布面内侧选点，以夹入足够布料，避免明显多层堆叠。这是调整抓点，不是额外下压。衣身和两侧袖子均已展开，袖子不再明显收在衣身内，且没有明显大面积折叠或覆盖时，返回DONE。袖子尚未展开或仍被压在衣身内时不能视为完成。衣领必须显露：衣领区域可辨认、未被大块布层覆盖即可；不要求领口撑开，也不要求完整领圈清晰可见。允许局部皱褶和轻微不对称，但袖子相对衣身的伸出长度应尽量接近目标图；若袖子仍明显短缩，应继续展开，不能仅因袖口可见就判定DONE。伸出程度已接近目标后，不要仅为消除褶皱继续拉扯。无法可靠选点时ABSTAIN。只输出指定JSON。
'''
UNFOLD_PROMPT=update_completion(UNFOLD_PROMPT).replace(
    '仅使用lift_pull_place，持布时不旋转；抓取方向保持所属机械臂既有Home姿态。',
    '仅使用lift_pull_place。'+GRASP_ROTATION_PROMPT)

# Shared execution requirements, supplied before selection in every planner mode.
ACTION_GUIDANCE = """【选点前已知的执行要求】
下摆是衣服底部的开口边；若目标图有粉色标线及端点，它们标示下摆及两角，仅作部位参考。
不设固定拉距上下限。根据当前折叠状态和目标展开形态选择拉动距离，抓放点及完整路径须满足工作边界、IK和碰撞检查。
抓点选可见布料，放点选可见桌面；避开空洞和补色接缝。允许布面局部高度变化和整体重建残缺，不因这些现象放弃；但补出的RGB不等于实测深度，实际抓放点都须有融合点云支持，不能用猜测高度代替。
宿主优先按抓点分区选臂，超出工作范围时检查另一臂；抓放点和完整路径须由同一臂通过范围、IK和碰撞检查；保持衣服大致朝向，抓放点保持在桌面可见范围内。仅用既有lift_pull_place，不自行扣减抓取高度或改变持布姿态。
"""
TANGENT_GUIDANCE = """【若本步使用Tangent：选边界点前的几何要求】
C选在真实衣物外轮廓上较平滑、近似直线的一段，避开尖角、急弯、毛刺、轮廓交汇和图像裁切处；不要把印花或布内折痕当外轮廓。
Host吸附到轮廓的最大距离为25像素，沿轮廓两侧各25像素拟合切线；局部PCA次/主方差比超过0.2会无法生成方向。因此应选附近有足够连续直边支持的C，而不是仅凭单个像素判断方向。
内法线须能区分布内外并与对侧轮廓相交，仍统一取最远交点；对侧抓点须落在可见布料上，放点须越过对侧边缘，二者留在图内并有实测深度及同臂可达空间。分割需要一块连通、未被图像边界裁断的衣物轮廓。
"""

def abstain(reason):
    return dict(decision='ABSTAIN',grasp_px=None,target_px=None,motion_type=None,
                grasp_angle_deg=None,is_smoothing=False,action_purpose=None,
                expected_stop_state=None,short_reason=reason)


def segment(rgb, unknown_mask=None):
    """Independent CV segmentation; no model output is accepted here."""
    from cloth_agent.cloth_harness.cv_harness.segmentation import segment_light_garment
    try:return segment_light_garment(rgb,unknown_mask)
    except ValueError as exc:raise normal.NormalGraspError(str(exc)) from exc


def segment_observation(before):
    """Use an explicit profile or select one from measured CV evidence.

    ``auto`` is independent of model-selected points.  When the planning view
    has measured XYZ it first evaluates a raised-surface candidate against the
    tabletop plane, then compares it with the black/light color candidates.
    """
    import os
    profile=getattr(before,'segmentation_profile',os.environ.get('CLOTH_SEGMENTATION_PROFILE','light'))
    path=Path(before.image)
    if profile=='grounded_sam2':
        from cloth_agent.grounded_sam2 import segment_view
        try:return segment_view(before)
        except (ValueError,RuntimeError,OSError,TimeoutError) as exc:
            raise normal.NormalGraspError('SAM2 segmentation unavailable: '+str(exc)) from exc
    if profile in ('black','auto'):
        from cloth_agent.cloth_harness.cv_harness.segmentation import segment_dark_garment
        rendered=path.parent/'rendered_rgb.png'
        path=rendered if rendered.is_file() else path.parent/'unmatched_rgb.png'
        if not path.is_file():raise normal.NormalGraspError('black segmentation requires uncorrected topdown RGB')
        rgb=np.asarray(Image.open(path).convert('RGB'))
        if rgb.shape[:2]!=np.asarray(Image.open(before.image)).shape[:2]:
            raise normal.NormalGraspError('segmentation/planning image dimensions differ')
        appearance_valid=path.parent/'appearance_valid.npy'
        unknown=(~np.load(appearance_valid,allow_pickle=False) if path.name=='rendered_rgb.png'
                 and appearance_valid.is_file() else getattr(before,'unknown_mask',None))
        try:
            if profile=='auto':
                from cloth_agent.cloth_harness.cv_harness.segmentation import segment_auto_garment
                xyz=getattr(before,'xyz',None)
                mask,report=segment_auto_garment(rgb,xyz,unknown)
            else:mask,report=segment_dark_garment(rgb,unknown)
        except ValueError as exc:raise normal.NormalGraspError(str(exc)) from exc
    elif profile=='light':
        mask,report=segment(np.asarray(Image.open(path).convert('RGB')),getattr(before,'unknown_mask',None))
    else:raise normal.NormalGraspError('unknown segmentation profile: '+str(profile))
    if getattr(before,'unknown_mask',None) is not None:
        report['silhouette_cells_without_measured_depth']=int(((mask>0)&before.unknown_mask).sum())
    report.update(profile=profile,source_image=str(path))
    return mask,report


class CandidateDepthFailure(normal.NormalGraspError):
    def __init__(self, failures):
        self.failures=failures
        super().__init__('所选边界未产生有效局部抓放候选（实测深度/距离）：'+json.dumps(failures,ensure_ascii=False))


def measured_candidate(before, mask, selection):
    options=dict(max_snap_px=25.)
    if hasattr(before,'unknown_mask'):
        options['unknown_mask']=before.unknown_mask
    if selection['skill']=='garment-chord-grasp':
        if len(selection['chord_endpoints_px'])!=2:
            raise normal.NormalGraspError('chord method requires two endpoints')
        options['chord_endpoints']=selection['chord_endpoints_px']
    elif selection['chord_endpoints_px']:
        raise normal.NormalGraspError('non-chord method must not supply endpoints')
    # Search only amplitude along the selected normal; never change Q to get
    # a desired grasp. Every candidate uses current measured depths.
    candidates=[];depth_failures={};failures=Counter()
    from cloth_agent.dual_arm.observation import DepthUnavailableError
    from cloth_agent.dual_arm.geometry import DualArmError
    desired = float(selection['pull_distance_mm'])
    if not np.isfinite(desired) or desired <= 0:
        raise normal.NormalGraspError('pull distance must be positive and finite')
    # Any two supported image points are at most the image diagonal apart.
    # Keep the original normal/opposite intersection; expand amplitude only.
    max_pull_px = int(np.ceil(np.hypot(*mask.shape)))
    for length in range(1, max_pull_px + 1, 4):
        try:
            g=normal.geometry(mask,selection['boundary_point_px'],pull_px=length,**options)
            values=[]
            for k in ('grasp_px','target_px'):
                try:values.append(np.asarray(before.sample(g[k],None),float))
                except DepthUnavailableError as exc:
                    depth_failures[str(g[k])]=dict(role=k,pixel=g[k],error=str(exc))
                    raise
            a,b=values
            distance=float(np.linalg.norm(b[:2]-a[:2]))
            if np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(distance) and distance > 0:
                candidates.append((abs(distance-desired),g,distance,length))
            else:failures['invalid measured XY distance']+=1
        except (normal.NormalGraspError,DualArmError) as exc:
            failures[str(exc)]+=1
            continue
    if not candidates and depth_failures:raise CandidateDepthFailure(list(depth_failures.values()))
    if not candidates:raise normal.NormalGraspError('候选无法计算：'+json.dumps(dict(failures),ensure_ascii=False))
    _,g,distance,length=min(candidates,key=lambda v:v[0])
    return g,distance,dict(options,pull_px=length)


def _run_once(root, folder, before, body, headers, timeout, direct, cv_cache, selection_prompt=SELECT_PROMPT):
    folder=Path(folder);select_folder=folder/'skill_selection';select_folder.mkdir()
    selection_body=copy.deepcopy(body)
    from .unfold_goal_annotation import apply_goal_annotation
    apply_goal_annotation(root, select_folder, selection_body)
    selection_body['instructions']+='\n'+ACTION_GUIDANCE
    if GRASP_ROTATION_PROMPT not in selection_body['instructions']:
        selection_body['instructions']+='\n'+GRASP_ROTATION_PROMPT
    if selection_prompt == SELECT_PROMPT or 'ACTION时skill=garment-tangent-grasp' in selection_prompt:
        selection_body['instructions']+='\n'+TANGENT_GUIDANCE
    selection_body['instructions']+='\n'+selection_prompt
    if selection_prompt==SELECT_PROMPT:
        selection_body['instructions']=apply_switch_rule(selection_body['instructions'])
    schema=copy.deepcopy(SELECT_SCHEMA)
    selection_body['text']['format'].update(name='unfold_skill_selection',schema=schema)
    (select_folder/'planner_prompt.txt').write_text(selection_body['instructions'],encoding='utf-8')
    selection,metrics=direct(select_folder,selection_body,headers,timeout)
    io.jsonschema.validate(selection,schema)
    if selection['decision'] != 'ACTION' and selection['grasp_angle_deg'] is not None:
        raise ValueError('non-ACTION grasp_angle_deg must be null')
    save(select_folder/'selection.json',selection)
    console.selected(selection)
    save(select_folder/'model_metrics.json',metrics)
    save(select_folder/'request_payload.json',selection_body)
    if selection['decision']!='ACTION':
        plan=abstain(selection['evidence']);plan['decision']=selection['decision']
        return plan,metrics
    action_points=selection
    geometry_skill=selection['skill'] in NAMES
    if geometry_skill:
        try:
            if selection['boundary_point_px'] is None:
                raise normal.NormalGraspError('geometry skill requires a visible boundary point')
            if 'segmentation' not in cv_cache:
                cv_cache['segmentation']=segment_observation(before)
            mask,report=cv_cache['segmentation']
            save(folder/'segmentation.json',report)
            mask_path=folder/'mask.png';Image.fromarray(mask*255).save(mask_path)
            from cloth_agent.cloth_harness.cv_harness.segmentation import external_contour_band
            edge_mask=external_contour_band(mask,thickness=int(report.get('edge_mask_thickness_px',2)))
            Image.fromarray(edge_mask*255).save(folder/'edge_mask.png')
            geometry,distance,options=measured_candidate(before,mask,selection)
            proposal=normal.build(before.image,selection['boundary_point_px'],selection['region_name'],
                                  selection['evidence'],folder/'skill_geometry',mask_path=mask_path,**options)
            proposal.update(skill=selection['skill'],measured_xy_distance_mm=distance)
            save(folder/'skill_geometry/proposal.json',proposal)
            action_points=geometry
        except (normal.NormalGraspError,cv2.error) as exc:
            console.failure('候选失败',exc,folder,getattr(exc,'failures',[]))
            save(folder/'skill_failure.json',dict(stage='CV_GEOMETRY',error=str(exc),
                retryable_selection=True,retryable_depth=isinstance(exc,CandidateDepthFailure),
                failed_selection=selection,failed_points=getattr(exc,'failures',[])))
            return abstain('skill几何无法生成有效候选：'+str(exc)),metrics
    plan=dict(decision='ACTION',grasp_px=action_points['grasp_px'],target_px=action_points['target_px'],
              motion_type='lift_pull_place',grasp_angle_deg=selection['grasp_angle_deg'],is_smoothing=False,
              action_purpose=selection['region_name'][:240],
              expected_stop_state=('露出被覆盖的目标部位' if geometry_skill else
                  '接近目标展开形态'),short_reason=selection['evidence'])
    plan=planner.validate(plan,Image.open(before.image).size,False)
    from cloth_agent.dual_arm.observation import DepthUnavailableError
    try:
        for key in ('grasp_px','target_px'):
            before.sample(plan[key],None)
    except DepthUnavailableError as exc:
        console.failure('深度缺失',f'{key}={plan[key]}；{exc}',folder)
        save(folder/'skill_failure.json',dict(stage='FUSED_DEPTH',error=str(exc),
            retryable_selection=True,retryable_depth=True,failed_selection=selection,
            failed_points=[dict(role=key,pixel=plan[key],error=str(exc))]))
        return abstain('所选点缺少融合点云支持：'+str(exc)),metrics
    console.emit('抓放点',f'{plan["grasp_px"]} → {plan["target_px"]}（俯视图像素，待运动检查）')
    manifest=json.loads((folder/'planning_skills.json').read_text(encoding='utf-8'))
    manifest.update(selected_skill=selection['skill'],action_source='measured_cv_candidate' if geometry_skill else 'direct_points',
                    semantic_review=False,automatic_opposite_grasp=geometry_skill,local_cv_executed=geometry_skill,
                    goal_boundary_reposition_available=False)
    save(folder/'planning_skills.json',manifest)
    return plan,metrics


def run(root, folder, before, body, headers, timeout, direct, *, selection_prompt=SELECT_PROMPT, max_attempts=3):
    """At most two new boundary selections after candidate computation failure.

    This is not semantic review: successful candidates go straight to Host.
    Every attempt uses the same unchanged observation and geometry checks.
    """
    if max_attempts not in (1,2,3):raise ValueError('selection attempts must be 1..3')
    folder=Path(folder);request=copy.deepcopy(body);attempts=[];cv_cache={}
    feedback_path=Path(folder)/'rejection_feedback.json'
    if feedback_path.exists():
        feedback=json.loads(feedback_path.read_text(encoding='utf-8'))
        if feedback.get('rejection_reason')=='IK_REJECTED':
            request['instructions'] += ('\n本次候选在执行前IK路径检查失败，未下发任何动作，当前图仍有效。'
                '失败位姿属于所选机械臂基座坐标，不是world或图像坐标；这不证明所有替代路径不可达。'
                '先重新判断是否已达到DONE条件；已完成就停止，不要仅因收到重规划反馈而继续拉扯。'
                '仍需展开时重新选择抓放点，优先尝试更靠该臂可操作区域内部、距离更短的放点，'
                '必要时更换抓点，不要原样重复失败动作。新动作仍须通过原有IK、工作范围和完整路径碰撞检查。'
                '无法提出合适动作时ABSTAIN。失败信息：\n'+json.dumps(feedback,ensure_ascii=False))
        elif feedback.get('rejection_reason')=='XY_TRANSLATION_OUT_OF_RANGE':
            request['instructions'] += ('\n本次候选未执行，衣物与当前照片未改变。以下为Host实测反馈，请修正抓放点，'
                       '本次拒绝仅因XY平移距离不满足范围，不代表抓点或目标点越界，也不代表已验证可达。'
                       '优先保留原抓点，调整目标点使距离进入允许范围：过长则缩短，过短则适当增加。'
                       '仍须选择符合动作语义的可见放置区域；若原抓点无法配合有效目标，可改选抓点。'
                       '不要重复被拒候选，不要把距离超限推断为抓点越界；确实没有合适动作则ABSTAIN：\n'
                       +feedback_path.read_text(encoding='utf-8'))
        else:
            request['instructions'] += ('\n上次规划或动作编译失败，尚未下发动作，当前观测未改变。'
                '以下是实际错误反馈，请据此修正输出或重新选择动作，不要原样重复失败候选。'
                '不得通过忽略深度、工作范围、IK或碰撞要求解决错误；输入格式错误请按当前schema重新输出。'
                '若已完成则DONE；无法提出有效动作则ABSTAIN。\n'+json.dumps(feedback,ensure_ascii=False))

    manifest=json.loads((folder/'planning_skills.json').read_text(encoding='utf-8'))
    for index in range(max_attempts):
        target=folder if index==0 else folder/f'depth_reselection_{index:02d}'
        if index:
            target.mkdir();save(target/'planning_skills.json',manifest)
            console.emit('重新选点',f'第 {index+1}/{max_attempts} 次规划 · 使用同一观测，尚未下发动作',warning=True)
        plan,metrics=_run_once(root,target,before,request,headers,timeout,direct,cv_cache,selection_prompt)
        attempts.append(dict(directory=str(target),decision=plan['decision'],metrics=metrics))
        failure=target/'skill_failure.json'
        detail=json.loads(failure.read_text(encoding='utf-8')) if failure.exists() else {}
        if plan['decision']!='ABSTAIN' or not detail.get('retryable_selection'):break
        request['instructions']+='\nHost候选计算失败，尚未执行，图像未改变。以下给出所选边界和具体失败原因。请按上述使用条件重新选点：方向或抓点无法计算时换一段边界，不要只改变拉动距离；放置点或距离问题可调整幅度；深度缺失则避开该点，不要降低深度要求或猜测高度。这不是语义复核。实在没有其他边界再ABSTAIN。\n'+json.dumps(detail,ensure_ascii=False)
    save(folder/'depth_reselection.json',dict(attempts=attempts,selected_attempt=str(target),
         final_decision=plan['decision'],max_attempts=max_attempts))
    if target!=folder:
        final_manifest=json.loads((target/'planning_skills.json').read_text(encoding='utf-8'))
        final_manifest['selected_attempt']=str(target);save(folder/'planning_skills.json',final_manifest)
    return plan,dict(metrics,semantic_review=False,selection_attempts=len(attempts),
        wall_latency_s=sum(a['metrics'].get('wall_latency_s',0) for a in attempts))
