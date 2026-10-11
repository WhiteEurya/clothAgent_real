"""Model-authored session memory, persisted by the host without model file access."""
import json

FIELDS = ('current_state', 'previous_action_effect', 'uncertainties', 'lessons', 'next_step')


def memory_prompt(record):
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
        'memory 是对象，恰好包含以下五个字符串字段：'
        'current_state、previous_action_effect、uncertainties、lessons、next_step。'
        'memory 合计不超过 12000 字符，压缩保留跨轮有用的信息。'
        'complete/blocked 也使用相同 JSON 格式，其 code 仍只能包含单独的终止指令。'
        '不要自行写文件，宿主负责将记忆落盘。\n'
    )


def parse_answer(answer):
    data = json.loads(answer)
    if not isinstance(data, dict) or set(data) != {'code', 'memory'}:
        raise ValueError('memory answer must contain exactly code and memory')
    code, memory = data['code'], data['memory']
    if not isinstance(code, str) or not code.strip():
        raise ValueError('memory answer code must be a nonempty string')
    if (not isinstance(memory, dict) or set(memory) != set(FIELDS)
            or any(not isinstance(v, str) for v in memory.values())
            or sum(map(len, memory.values())) > 12000):
        raise ValueError('invalid iteration memory fields or length')
    return code, memory
