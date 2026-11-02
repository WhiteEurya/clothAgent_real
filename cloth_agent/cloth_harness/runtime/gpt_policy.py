"""One call per condition, stable Pa/Pb format. No anchor/verification model calls."""
from cloth_agent import restricted_context_one_shot as existing
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io

# Exact same short policy for A/B/C/D; only observation evidence differs.
PROMPT='''你是衣物展开的单步规划器。第一张是当前图，第二张是相同衣物的展平目标。若提供历史图或反馈，只描述上一步真实执行信息；不能将未知当成成功，也不能把投影面积增加直接当成展开进展。
只抓当前图可见衣袖或下摆；在边缘带稍向内选点以夹牢，不猜隐藏点。根据当前衣服朝向和动作目的自行选择Pa、Pb，不因为有历史反馈就机械重复或禁止某一部位。
只输出一个动作，沿用整数像素Pa、Pb。ACTION时给一句action_purpose和一句expected_stop_state及short_reason；只能lift_pull_place，只在抓前旋转夹爪。is_smoothing仅在残余小褶皱整理时true。
主要衣身、双袖和下摆基本展开且没有明显内收或大片覆盖时DONE，允许普通皱纹；没有可靠动作则ABSTAIN。DONE/ABSTAIN坐标、角度、运动类型和两个目的字段为null，is_smoothing=false。
常规抬升50mm，平移10–150mm；整理抬升10mm，平移10–30mm。已有Host负责物理检查。当前反馈未经独立验证，不能替代视觉判断。
只输出指定JSON。\n'''+io.GRASP_ANGLE_PROMPT+io.GRIPPER_BOUNDARY_PROMPT

def payload(pair,condition,feedback=None,oracle=None):
    from PIL import Image
    from pathlib import Path
    current=Path(pair['after'])/'rgb.png';body=io.payload(current,Path(pair['goal']),list(Image.open(current).size))
    body['instructions']=PROMPT;body['text']['format'].update(name='fast_feedback_action',schema=existing.SCHEMA)
    if condition=='B':
        p=pair['plan'];body['input'][0]['content'].extend([
            {'type':'input_text','text':f"第三张是上一步动作前图。实际执行状态={pair['execution']['status']}，Pa={p['grasp_px']}，Pb={p['target_px']}，抓前角度={p.get('grasp_angle_deg')}。未证实抓住布料。"},
            {'type':'input_image','image_url':io.image_data(Path(pair['before'])/'rgb.png'),'detail':'high'}])
    elif condition=='C':
        if not feedback:raise ValueError('C requires automatic feedback')
        body['input'][0]['content'].append({'type':'input_text','text':'上一动作CV反馈：'+feedback})
    elif condition=='D':
        if not oracle or oracle.get('confirmed_by') is None:raise ValueError('D requires independently confirmed labels')
        allowed={'grasp_status','area_change','motion_type','local_shape_change','visible_unfolding_progress'}
        facts={k:v for k,v in oracle['labels'].items() if k in allowed}
        if not facts or all(v=='UNKNOWN' for v in facts.values()):raise ValueError('no informative oracle outcomes')
        import json
        body['input'][0]['content'].append({'type':'input_text','text':'人工确认的上一动作结果（不是下一动作建议）：'+json.dumps({'Pa':pair['plan']['grasp_px'],'Pb':pair['plan']['target_px'],'facts':facts},ensure_ascii=False)})
    elif condition!='A':raise ValueError('unknown condition')
    return body
