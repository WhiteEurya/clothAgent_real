"""Model-authored session memory, persisted by the host without model file access."""
import json

FIELDS = ('current_state', 'previous_action_effect', 'uncertainties', 'lessons', 'next_step')


def memory_prompt(record, *, grasp_retry=False):
    assessment = (
        '\n展开抓取评估：memory 另外包含 grasp_assessment 对象，恰好包含 result、cloth_motion、evidence。'
        'result 为 missed/held/uncertain；cloth_motion 为 moved/unchanged/uncertain；evidence 是具体可见依据字符串。'
        '只评估上一轮已执行动作：比较前后图中的抓点邻域、布边及预期落点。'
        '仅无展开进展不代表抓空；衣物移动但未展开、整体滑动或疑似抓住后脱落均不能写 missed。'
        '只有图像明确支持没有抓起或搬动布料才写 missed 和 unchanged；首次、IK修正或证据不足写 uncertain。'
        '宿主仅在 missed+unchanged 且可靠夹爪反馈支持空闭合时，本轮额外下降增加3 mm，上限30 mm。'
        '明确抓空时优先重试当前图中同一抓点附近，重新读取坐标；不要直接复用旧坐标。'
        '仍输出原始观测Z；计划须考虑可能增加的3 mm，宿主会重新验证实际轨迹。'
        if grasp_retry else ''
    )
    return (
        '\n本次启用跨 iteration 记忆。宿主会保存你输出的 memory，并在下一轮原样提供。'
        '先根据当前图与上一轮动作前图评估上次动作效果，再更新记忆；'
        'EXECUTED 只代表指令执行，不证明抓布或复原成功。'
        '记住有效动作、失败尝试、连续无进展及待验证假设；多次无效时调整抓点或策略。'
        '旧坐标仅作历史记录，新的动作坐标必须从当前观测读取。'
        '本轮动作尚未执行，预期效果必须写成待验证，不能写成已成功。'
        'IK 修正轮没有新图且未执行，要保留这个事实，不虚构效果。'
        '\n此前记忆与宿主记录（首次为 null）：'+json.dumps(record, ensure_ascii=False)+
        '\n最终输出格式覆盖上文的纯代码输出要求：只输出一个 JSON 对象，不加 Markdown。'
        '对象恰好包含 code 和 memory。code 是包含完整 Python 动作代码的字符串；'
        'memory 是对象，包含以下五个字符串字段（启用展开抓取评估时还包含下述 grasp_assessment）：'
        'current_state、previous_action_effect、uncertainties、lessons、next_step。'
        'memory 合计不超过 12000 字符，压缩保留跨轮有用的信息。'
        'complete/blocked 也使用相同 JSON 格式，其 code 仍只能包含单独的终止指令。'
        '不要自行写文件，宿主负责将记忆落盘。\n'+assessment
    )


def parse_answer(answer):
    data = json.loads(answer)
    if not isinstance(data, dict) or set(data) != {'code', 'memory'}:
        raise ValueError('memory answer must contain exactly code and memory')
    code, memory = data['code'], data['memory']
    if not isinstance(code, str) or not code.strip():
        raise ValueError('memory answer code must be a nonempty string')
    if (not isinstance(memory, dict) or set(memory) not in (set(FIELDS), set(FIELDS) | {'grasp_assessment'})
            or any(not isinstance(memory[k], str) for k in FIELDS)
            or sum(len(memory[k]) for k in FIELDS) > 12000):
        raise ValueError('invalid iteration memory fields or length')
    if 'grasp_assessment' in memory:
        a = memory['grasp_assessment']
        if (not isinstance(a, dict) or set(a) != {'result', 'cloth_motion', 'evidence'}
                or a['result'] not in ('missed', 'held', 'uncertain')
                or a['cloth_motion'] not in ('moved', 'unchanged', 'uncertain')
                or not isinstance(a['evidence'], str) or not a['evidence'].strip()
                or len(a['evidence']) > 2000):
            raise ValueError('invalid grasp assessment')
    return code, memory
