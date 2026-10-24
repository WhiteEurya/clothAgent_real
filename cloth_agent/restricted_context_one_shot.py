"""Single action, visible sleeve/hem grounding, previous executed action context."""
import copy
import html
import json
import shutil
import time
from PIL import Image
from cloth_agent import preference_one_shot as shared
from cloth_agent import geometry_report
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io

PROMPT = """你是衣物展开的单步规划器。第一张是当前RGB，第二张是展平目标；如有第三张，则是上一动作前图，并附有实际执行动作，仅作为上下文。
只输出一个动作，不生成候选、不评分，不输出结构分析、上一动作效果评价、动作关系解释或风险分析。
只能抓当前图中可见的衣袖或者下摆。不能抓领口、肩部或衣身内部，不猜隐藏点。
避免只夹最外沿的一小撮；在同一可见衣袖或下摆边缘带稍向布面内侧选点，以夹入足够布料，避免明显多层堆叠。这是调整Pa，不是额外下压。
若目标袖子或下摆角不可见，可抓清楚可见的对侧对应部位；一侧收缩且临近桌面边界时，也可通过拉已展开的对侧间接展开。不机械镜像坐标或固定方向。
所有坐标使用第一张原图，左上原点，x向右、y向下，整数像素；目标图或历史图的坐标不能直接复用。
仅使用lift_pull_place。is_smoothing为布尔值：仅对残余小褶皱作小幅整理时true；解除明显折叠、释放内收部位时false。已基本展开时优先DONE，不为了普通小皱纹反复整理。
常规动作抬升至抓点/目标表面较高者上方50毫米，平移10–150毫米；整理动作抬升至较高表面上方10毫米，平移10–30毫米。Host按实测毫米检查，超限不执行；仅抓前可调整夹爪方向，持布时不旋转。附带Geometry Report来自本轮RGB-D的本地计算，不臆造像素到毫米换算；几何可行性仍由Host检查。
几何报告仅作辅助：轮廓候选不是已识别的下摆，需结合RGB辨认；未知长度保持未知，不根据候选跨度比直接决定DONE。报告不可用时按RGB规划，不猜隐藏长度。不要求复述报告或增加输出字段。
结构距离报告按对应部位给出八段直线距离、桌面XY距离/三维距离及相对展平参考比值。选择Pa前必须结合当前RGB复核候选部位身份：例如左下摆清晰但右侧候选不清楚，检查下摆两角跨度和候选到右腋下的距离。多段距离异常或边缘连接关系不符时降低身份置信度，不把折角冒充右下摆；优先改选清晰可见的对侧衣袖/下摆，无可靠抓点则ABSTAIN。短距离可能是真实折叠，也可能认错点，不能单独判假或判DONE。HIDDEN/UNCERTAIN、无效深度的点及相关长度均未知，不补全、不猜毫米。结构识别结果也是视觉假设，须复核，不作为真值；换到未被测量的新候选时不得套用原点距离。short_reason简短说明所抓部位；若因身份不可靠改选对侧，点明原因。
只做一个简单完成判断：主要衣身、双袖和下摆已基本展开、没有明显内收或大片覆盖时DONE，允许普通小皱纹；仍需展开时ACTION；无法可靠选择动作时ABSTAIN。
ACTION输出Pa、Pb、grasp_angle_deg、is_smoothing和一句简短动作理由；DONE/ABSTAIN时is_smoothing=false，坐标、角度和motion_type均为null，short_reason简短说明判断。
历史图可能来自不同视角；不要把相机视角变化当成衣服移动。只输出指定JSON。
""" + io.GRASP_ANGLE_PROMPT + io.GRIPPER_BOUNDARY_PROMPT
SCHEMA = copy.deepcopy(io.ROTATION_SCHEMA)
SCHEMA["properties"]["is_smoothing"]={"type":"boolean"}
SCHEMA["required"].append("is_smoothing")
REPORTS_PREVIOUS_EFFECT = False


def validate(output, size, contextual):
    io.jsonschema.validate(output, SCHEMA)
    io.validate_plan({k:output[k] for k in io.ROTATION_SCHEMA["properties"]},size,require_grasp_rotation=True)
    if output["decision"]!="ACTION" and output["is_smoothing"]:
        raise ValueError("non-ACTION is_smoothing must be false")
    return dict(output)


def plan_round(root,folder,before,headers,timeout_s,direct,previous_context=None,*,view_change=False,rejection_feedback=None):
    body=shared.make_payload(root,folder,before,PROMPT,SCHEMA,
        extra=('本次为抓点缺失深度后的换视角重规划：只移动了相机，尚未执行本轮衣物动作。必须在第一张新图中重新选择Pa/Pb，禁止复用旧像素；深度是否可用仍由Host检查。' if view_change else ''),
        previous_context=previous_context,assess_previous_effect=False)
    geometry=geometry_report.build(before,root/'goal/target.png',folder)
    body['input'][0]['content'].append(dict(type='input_text',text='Geometry Report（当前RGB-D自动计算；辅助证据，不是动作或DONE结论）：\n'+geometry_report.prompt_report(geometry)))
    structure_started=time.perf_counter()
    structure_metrics=None
    structure=dict(status='UNAVAILABLE',reason='no verified current/reference RGB-D')
    if hasattr(before,'xyz') and geometry.get('reference',{}).get('source'):
        structure_folder=folder/'structure_grounding'
        structure_folder.mkdir(parents=True,exist_ok=False)
        structure_body=shared.make_payload(root,structure_folder,before,geometry_report.STRUCTURE_PROMPT,
                                          geometry_report.STRUCTURE_SCHEMA,assess_previous_effect=False)
        structure_body['text']['format']['name']='garment_structure_points'
        landmarks,structure_metrics=direct(structure_folder,structure_body,headers,timeout_s)
        io.jsonschema.validate(landmarks,geometry_report.STRUCTURE_SCHEMA)
        for key,image in [('current',before.image),('reference',root/'goal/target.png')]:
            with Image.open(image) as im:w,h=im.size
            for point in landmarks[key].values():
                px=point['px']
                if point['visibility']!='VISIBLE' and px is not None:
                    raise ValueError('hidden/uncertain structure point must have null coordinates')
                if point['visibility']=='VISIBLE' and (px is None or not (0<=px[0]<w and 0<=px[1]<h)):
                    raise ValueError('visible structure point outside image or missing')
        io.save(structure_folder/'landmarks.json',landmarks)
        io.save(structure_folder/'model_metrics.json',structure_metrics)
        structure=geometry_report.build_structure(before,root/'goal/target.png',landmarks,folder)
    structure['wall_s']=round(time.perf_counter()-structure_started,4)
    io.save(folder/'structure_report.json',structure)
    body['input'][0]['content'].append(dict(type='input_text',text='结构直线距离与候选部位身份复核（视觉假设，不是真值）：\n'+json.dumps(structure,ensure_ascii=False)))
    if rejection_feedback is not None:
        body['input'].append(dict(role='user',content=[dict(type='input_text',text=
            '上一个候选在执行前被Host拒绝，衣服动作没有执行。这不表示动作已产生效果。'
            '请根据新图与以下错误重新规划一次，修正违反约束的坐标，不要重复被拒绝动作；'
            '不可因拒绝而声称衣服已展开，无可行动作时ABSTAIN。\n'+json.dumps(rejection_feedback,ensure_ascii=False))]))
        io.save(folder/'rejection_feedback.json',rejection_feedback)
    body['text']['format']['name']='restricted_context_action'
    if previous_context:
        io.save(folder/'previous_action_context.json',{**previous_context,
            'before_image':str(previous_context['before_image']),'after_image':str(previous_context['after_image'])})
        shutil.copyfile(previous_context['before_image'],folder/'context_previous_before.png')
    io.save(folder/'request_manifest.json',dict(model=io.MODEL,effort=io.EFFORT,
        input_image_count=3 if previous_context else 2,model_calls=2 if structure_metrics is not None else 1,topn_enabled=False,
        current_sha256=io.sha(before.image),goal_sha256=io.sha(root/'goal/target.png'),
        output_mode='compact_action',previous_effect_enabled=False,geometry_report='geometry_report.json',
        geometry_wall_s=geometry['wall_s'],structure_report='structure_report.json',structure_wall_s=structure['wall_s'],
        previous_action_context=bool(previous_context),grasp_regions=['visible_sleeve','visible_bottom_hem']))
    output,metrics=direct(folder,body,headers,timeout_s)
    metrics['action_model_wall_s']=metrics['wall_latency_s']
    metrics['structure_model_wall_s']=(structure_metrics or {}).get('wall_latency_s',0.)
    metrics['structure_wall_s']=structure['wall_s']
    metrics['wall_latency_s']+=metrics['structure_model_wall_s']
    metrics['geometry_wall_s']=geometry['wall_s']
    io.save(folder/'model_metrics.json',metrics)
    io.save(folder/'decision_evidence.json',output)
    with Image.open(before.image) as im:plan=validate(output,im.size,previous_context is not None)
    for name in ('plan.json','fast_plan.json'):io.save(folder/name,plan)
    io.overlay(before.image,plan,folder/'proposed_action_overlay.png',folder.name+' SLEEVE / HEM')
    shutil.copyfile(folder/'proposed_action_overlay.png',folder/'fast_action_overlay.png')
    io.save(folder/'fast_status.json',dict(valid=True,planner='RestrictedContextOneShot',model=io.MODEL,
        effort=io.EFFORT,retries=0,automatic_slow_routing=False))
    render_round(folder)
    return plan,metrics


def replan_rejected(root,folder,before,headers,timeout_s,direct,previous_context,feedback):
    """Keep each HTTP attempt isolated; publish only a completed valid proposal."""
    attempt=folder/'target_replan'/'retry_01'
    attempt.mkdir(parents=True,exist_ok=False)
    shutil.copytree(before.directory,attempt/'before')
    plan,metrics=plan_round(root,attempt,before,headers,timeout_s,direct,
        previous_context=previous_context,rejection_feedback=feedback)
    for path in attempt.iterdir():
        if path.is_file():shutil.copyfile(path,folder/path.name)
    render_round(folder)
    return plan,metrics


def render_round(folder):
    if not (folder/'decision_evidence.json').exists():return
    d=io.read(folder/'decision_evidence.json')
    parts=['<!doctype html><meta charset="utf-8"><title>衣袖/下摆 one-shot</title>',
           '<style>body{font:18px system-ui;max-width:1100px;margin:30px auto}img{max-width:100%}pre{white-space:pre-wrap}</style>',
           '<h1>'+html.escape(folder.name)+' · 单动作上下文实验</h1>']
    if (folder/'progress_stop.json').exists():
        progress=io.read(folder/'progress_stop.json')
        parts.append('<p><b>连续无改善：</b>'+str(progress['consecutive_no_improvement'])+' / '+str(progress['patience'])+'</p>')
        if progress.get('stop_reason'):
            parts.append('<p><b>本轮仅评估，未执行新动作：</b>'+html.escape(progress['stop_reason'])+'</p>')
    if 'primary_structure' in d:
        structure=d['primary_structure']
        parts.append('<h2>当前主要还有哪个结构没有展开？</h2>')
        for key,label in [('status','判断'),('region','主要结构'),('evidence','图像依据'),('priority_reason','优先处理原因')]:
            parts.append('<p><b>'+label+'：</b>'+html.escape(str(structure[key]))+'</p>')
        parts.append('<p><b>动作与结构的关系：</b>'+html.escape(d.get('action_structure_relation',''))+'</p>')
    if d.get('opposite_grasp'):
        opposite=d['opposite_grasp']
        parts.append('<p><b>对向抓取：</b>'+html.escape(opposite['hidden_part']+' 不可见 → '+opposite['selected_part'])+'</p>')
        parts.append('<p><b>可见性依据：</b>'+html.escape(opposite['evidence'])+'</p>')
    for key,label in [('decision','决定'),('is_smoothing','小幅整理'),('short_reason','选择理由'),('grasp_region','抓取部位'),
                      ('grounding_evidence','部位识别依据'),('expected_change','预计变化'),('failure_risk','风险'),
                      ('previous_effect','上一动作实际效果'),('completion_check','停止条件')]:
        if key in d:parts.append('<p><b>'+label+'：</b>'+html.escape(str(d[key]))+'</p>')
    for name,label in [('context_previous_before.png','上一动作前'),('before/rgb.png','本轮原图'),
                       ('structure_current.png','当前结构点与直线距离（仅展示）'),
                       ('structure_reference.png','展平参考对应结构距离'),
                       ('geometry_overlay.png','几何候选轮廓（仅展示，不叠加到模型图像）'),
                       ('proposed_action_overlay.png','选择的 Pa → Pb'),('after/rgb.png','执行后')]:
        if (folder/name).exists():parts.append(f'<h2>{label}</h2><img src="{name}">')
    for name in ('motion_profile.json','structure_report.json','geometry_report.json','depth_recovery/summary.json','progress_stop.json','timing.json','execution.json'):
        if (folder/name).exists():parts.append('<pre>'+html.escape((folder/name).read_text())+'</pre>')
    (folder/'decision.html').write_text('\n'.join(parts))
