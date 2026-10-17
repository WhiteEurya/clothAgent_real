"""Planner preferences and offline-only evidence; never an execution validator."""
from __future__ import annotations

import json
from pathlib import Path

POLICY = '''
动作目标是 folding execution robustness：可靠带动目标 cloth region，充分完成跨折线搬运，并容忍抓点和轨迹误差；不是让 planner 更保守或最小化位移。
允许正常多层 folding：不要求知道夹到具体哪一层；局部 overlap / layer ambiguity、相邻折层或看不见下层边缘，不能单独导致 NO_ROBUST_ACTION。只要 grasp 能合理带动目标 cloth region，即使可能同时夹住相邻折层，也允许提出动作；不要把“只夹单层”作为隐含条件。
Hard dispatch gate（仅以下拒绝条件）：① 无法确认抓点位于布料上；② 明显没有夹爪闭合空间；③ 候选点可能抓到与目标动作无关的结构；④ 当前 observation 中根本不存在可用于推进目标动作的 grasp handle；⑤ 已有 Host safety / workspace / IK / collision 检查结果不允许执行。
候选有上述问题时先考虑其他可用候选；没有可用动作时才返回 NO_ROBUST_ACTION，不强行生成 Pa/Pb。相邻折层不自动等于无关结构，应判断其能否一起带动目标区域。Host 检查仍由宿主执行，RGB 无法证明其通过不等于其拒绝，不得据此自行阻断或声称安全已验证。不新增 confidence 硬阈值、评分或解释字段。
现有 robustness 标签仅用于日志，不是执行条件；不要求静态 RGB 证明真实 safe_retreat，无证据时省略该标签，不据此拒绝动作。
在最终确定 Pa/Pb、yaw 和 motion 前，对每个候选内部检查（不输出长篇推理）：
1. Grasp：优先能带动整片目标区域的布边、突出 flap 或其他可用 handle；抓点稍入布边内，避免边界外、易滑脱或仅拉动局部的点。等效点优先视觉/几何证据清楚、闭合空间足够的位置，不因语义正确就随意选布片中部。局部重叠和多层抓取本身不是排除条件；关键是与目标区域的合理联系及带动能力。
2. Transport：布不是刚体，不以最短或最保守 Pa→Pb 位移为目标。袖子/衣身 inward fold 及 folding/tucking 优先 lift-and-transfer / arc-like 持布路径，避免低高度直拖；让整片目标区域明确跨过 fold line，不能只把抓点带到折线附近就结束。在 receiving region 和宿主约束内允许适度 overshoot，以抵消布料垂落及 Pa/Pb 小量变化，仍形成相近 terminal fold；不盲目拉远或追求最大位移。折叠关键段用已有 move_tcp waypoint 表达，不能用仅约束终点的 move_transfer 冒充受控弧线。
3. Placement：Pb 应位于有支撑的 receiving region 内部，与 fold line 和接收区边缘留有足够 margin，不能紧贴折线。规划持布跨线、转移及下降落放，使被搬运布片在 release 前已由接收区域支撑，避免仅抓点到位而布片形成 bridge / 悬空、释放后回弹。不把未知接触效果声称为已验证。落点、末端姿态和邻域应容纳开爪与宿主撤离的 swept area，避免把已放好的布推回或勾走；释放/撤离仍由宿主负责，不新增辅助动作或未提供的 release_direction。
动作选择严格按：① 能可靠带动目标 cloth region；② 能把目标区域充分跨过 fold line 并落入 receiving region；③ 对 Pa/Pb 和轨迹小误差不敏感；④ 前三项满足后，再尽量减少不必要扰动。不要为了减少扰动而牺牲充分跨线和稳定落放。所有动作仍须通过原有 Host 安全检查。
'''

CODE_PROMPT = POLICY + '''
仅在上述拒绝条件使当前无可用动作时，单独输出 motion.blocked(reason="NO_ROBUST_ACTION: 具体拒绝原因")，不附带 Pa/Pb 或移动指令；不要仅因层数不明或局部重叠而 blocked。
保持原 action schema；最终 Python 代码前仅加一行注释：
# ACTION_ROBUSTNESS {"tags":["clear_edge","sufficient_clearance","supported_target"],"confidence":0.0,"reason":"简短可见依据或缺失项"}
tags 只选当前观测支持的项（也可用 protruding_flap / reliable_geometry / low_sensitivity_transport），不得照抄示例或将推测写成事实。confidence 为 0..1 的执行稳健性估计，不是成功保证；无可靠 handle 时为低置信。complete 时 tags 为空。若外层要求 JSON，把这行注释放入原 code 字符串内，不增加顶层字段。标签仅供离线日志。
'''

GROUNDING_PROMPT = POLICY + '''
仅在已选 edge 和不可变模板允许的锚点/参数内应用以上优先级，不改 edge、primitive 数量或模板形状；不能改轨迹的模式不声称已优化轨迹。
仅在上述拒绝条件使当前无可用动作时：minimal 输出 null anchors、confidence=0；full 输出 NEED_MORE_INFORMATION、空 slots，missing_visual_parameters 简短写 NO_ROBUST_ACTION: 具体拒绝原因；不强行生成 Pa/Pb。多层抓取、局部重叠或无法判断具体层数本身不属于 missing_visual_parameters，也不构成视觉违规；未知物理效果只记风险。
额外只返回 robustness={"tags":[],"confidence":0.0,"reason":"简短可见依据或缺失项"}，沿用原有标签 schema，优先记录当前观测支持的 clear_edge / protruding_flap / sufficient_clearance / reliable_geometry / low_sensitivity_transport / supported_target / safe_retreat，不要求 low_layer_ambiguity 标签。数值 0..1，不作为拒绝阈值，不输出推理过程。该字段仅用于离线日志，非 Host 验证结果。
'''

TAGS = ('clear_edge', 'protruding_flap', 'sufficient_clearance', 'low_layer_ambiguity',
        'reliable_geometry', 'low_sensitivity_transport', 'supported_target', 'safe_retreat')
SCHEMA = dict(type='object', additionalProperties=False, required=['tags', 'confidence', 'reason'],
              properties=dict(tags=dict(type='array', items=dict(type='string', enum=list(TAGS)),
                                        maxItems=len(TAGS), uniqueItems=True),
                              confidence=dict(type='number', minimum=0, maximum=1),
                              reason=dict(type='string', maxLength=240)))
PREFIX = '# ACTION_ROBUSTNESS '


def save_evidence(folder, value):
    """Logging cannot grant/deny motion authorization or change existing validation."""
    import jsonschema
    report = dict(source='model_self_assessment', host_validated=False, status='MISSING')
    if value is not None:
        try:
            jsonschema.validate(value, SCHEMA)
            # JSONSchema's number range comparisons alone do not reject NaN.
            json.dumps(value, allow_nan=False)
            report.update(status='RECORDED', **value)
        except (ValueError, jsonschema.ValidationError):
            report.update(status='INVALID', reason='Malformed robustness metadata')
    path = Path(folder) / 'action_robustness.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


def strip_code_evidence(source, folder):
    """Remove diagnostic comments before motion/history can be exported to a Graph."""
    lines = source.splitlines(keepends=True)
    entries = [line.strip()[len(PREFIX):] for line in lines if line.strip().startswith(PREFIX)]
    value = None
    if entries:
        try:
            value = json.loads(entries[0]) if len(entries) == 1 else 'duplicate metadata'
        except ValueError:
            value = 'invalid JSON'
    save_evidence(folder, value)
    return ''.join(line for line in lines if not line.strip().startswith(PREFIX))


def strip_grounding_evidence(value, folder):
    clean = dict(value)
    save_evidence(folder, clean.pop('robustness', None))
    return clean
