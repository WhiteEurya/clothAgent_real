"""Per-turn visual grasp assessment for host retry control, not handoff memory."""
import json


def assessment_prompt():
    return (
        '\n最终输出格式覆盖纯代码要求：只输出 JSON，恰好包含 code 和 grasp_assessment，不加 Markdown。'
        'code 是完整动作代码字符串；grasp_assessment 恰好包含 result、cloth_motion、evidence。'
        'result 为 missed/held/uncertain，cloth_motion 为 moved/unchanged/uncertain，evidence 为具体图像依据字符串。'
        '这只是宿主重试控制需要的当轮评估，不是跨会话记忆。只评估上一轮已执行动作，不提前宣称本轮成功。'
        '比较抓点邻域、布边和预期落点；仅无展开进展不代表抓空；衣物移动、整体滑动或疑似中途脱落不写 missed。'
        '首次、IK修正或证据不足写 uncertain；只有图像明确支持没有抓起或搬动布料才写 missed 和 unchanged。'
        '宿主仅在 missed+unchanged 且可靠夹爪反馈支持空闭合时，本轮增加3 mm，上限30 mm。'
        '确认抓空时优先从新观测重新定位同一抓点附近，输出原始Z，考虑可能增加的3 mm，宿主重新验证实际轨迹。'
        'complete/blocked 也使用此 JSON 格式。'
    )


def parse_assessment_answer(answer):
    data=json.loads(answer)
    if not isinstance(data,dict) or set(data)!={'code','grasp_assessment'}:
        raise ValueError('answer must contain exactly code and grasp_assessment')
    code=data['code'];a=data['grasp_assessment']
    if not isinstance(code,str) or not code.strip():
        raise ValueError('code must be a nonempty string')
    if (not isinstance(a,dict) or set(a)!={'result','cloth_motion','evidence'}
            or a['result'] not in ('missed','held','uncertain')
            or a['cloth_motion'] not in ('moved','unchanged','uncertain')
            or not isinstance(a['evidence'],str) or not a['evidence'].strip()
            or len(a['evidence'])>2000):
        raise ValueError('invalid grasp assessment')
    return code,a
