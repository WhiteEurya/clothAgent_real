"""Accumulate 5 mm after confirmed empty closure; never infer it from execution status."""
import json
import math
from pathlib import Path


def retry_policy(folder, arm, outcome, applied_offset_mm, *, initial_offset_mm=5.):
    report=dict(assessment='unknown',applied_offset_mm=applied_offset_mm,initial_offset_mm=initial_offset_mm,
                next_offset_mm=applied_offset_mm,reason='no completed physical grasp')
    if outcome.get('status') not in ('EXECUTED','GRASP_EMPTY') or outcome.get('physical_execution') is not True:
        return report
    closes=[]
    for path in sorted(Path(folder).glob('gripper_*.json')):
        data=json.loads(path.read_text())
        if data.get('arm')==arm and data.get('action')=='close':closes.append((path,data))
    if len(closes)!=1:return report
    path,data=closes[0]
    report['evidence']=str(path.resolve())
    envelope=data.get('feedback',{});feedback=envelope.get('feedback',{});done=envelope.get('completion',{})
    if (envelope.get('command_result')!=0 or done.get('status')!='COMPLETED'
            or feedback.get('usable_for_completion') is not True
            or feedback.get('read_errors') or feedback.get('error_code')!=0):
        report['reason']='unreliable feedback';return report
    position=feedback.get('position_pulse');target=done.get('target_position_pulse');tol=done.get('position_tolerance_pulse')
    def number(value):
        return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    if not all(number(v) for v in (position,target,tol)) or tol<0:
        report['reason']='missing valid measured closure';return report
    report.update(position_pulse=position,target_position_pulse=target)
    if (feedback.get('state')=='stop' and feedback.get('mechanical_grasp_detected') is False
            and abs(position-target)<=tol):
        report.update(assessment='empty',next_offset_mm=applied_offset_mm+5.,
                      reason='fully closed without mechanical grasp')
    else:
        initial=done.get('before_command',{}).get('position_pulse')
        if (feedback.get('state')=='grasp' and feedback.get('mechanical_grasp_detected') is True
                and number(initial) and initial-position>tol and position-target>tol):
            report.update(assessment='contact_detected',next_offset_mm=applied_offset_mm,
                          reason='mechanical contact after closure; visual cloth confirmation still needed')
        else:report['reason']='ambiguous closure; keep offset unchanged'
    return report


def visual_retry_policy(folder, arm, outcome, applied_offset_mm, assessment, *, max_offset_mm=30.):
    """Require both visual miss and reliable empty closure; never interrupt a grasp."""
    mechanical = retry_policy(folder, arm, outcome, applied_offset_mm)
    report = dict(assessment='unconfirmed', applied_offset_mm=applied_offset_mm,
                  next_offset_mm=applied_offset_mm, limit_reached=False,
                  visual=assessment, mechanical=mechanical, max_offset_mm=max_offset_mm)
    if (isinstance(assessment, dict) and assessment.get('result') == 'missed'
            and assessment.get('cloth_motion') == 'unchanged'
            and isinstance(assessment.get('evidence'), str) and assessment['evidence'].strip()
            and mechanical['assessment'] == 'empty'):
        report.update(assessment='corroborated_miss',
                      next_offset_mm=min(applied_offset_mm + 3., max_offset_mm),
                      limit_reached=applied_offset_mm >= max_offset_mm)
    return report
