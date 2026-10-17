"""Concise user-facing termination summary from recorded facts only."""
import json
from pathlib import Path

REASONS={
 'COMPLETED':'重新观察确认到达已声明的终点。',
 'DRY_RUN_OK':'离线静态检查通过，未执行机器人。',
 'SOURCE_NODE_UNCHANGED':'动作后仍匹配到原节点，当前恢复策略已停止。',
 'EDGE_REPAIR_SELF_LOOP':'修复动作后仍匹配到原节点，本轮修复次数已用完。',
 'EDGE_INSTANTIATION_FAILURE':'所选动作的视觉参数或实例化条件未通过。',
 'SAFETY_REJECT':'规划或安全检查拒绝了本次动作，请查看具体错误。',
 'CONTROLLER_FAULT':'机械臂控制器报告故障或异常状态；已停止，不自动清故障或重试动作。',
 'ACTION_EXECUTION_FAILURE':'动作执行未正常完成，请查看控制器错误。',
 'APPLICABILITY_FRONTIER_FAILED':'适用性恢复中的临时规划或执行未完成。',
 'REQUIRES_FRESH_VALIDATION':'当前节点没有已经验证可 replay 的后续动作。',
 'NO_REPLAYABLE_EDGE':'当前没有可用的 replay 边。',
 'NO_REPLAYABLE_ACTION':'当前节点没有可用的 replay 动作。',
 'NO_MATCH':'当前观察未可靠匹配到已有节点。',
 'AMBIGUOUS':'当前节点判断存在歧义或置信度不足。',
 'UNEXPECTED_NODE':'动作后的匹配结果不是预计目标节点。',
 'SOURCE_REENTRY_LIMIT':'同一衣物状态的节点纠正反复冲突，已达到有界复核次数；未继续派发旧动作。',
 'STEP_LIMIT':'已达到本轮动作步数上限。',
 'FALLBACK_LIMIT':'已达到本轮临时规划次数上限。',
 'FALLBACK_FAILED':'临时规划或执行未完成，请查看具体错误。',
 'RECOVERY_UNRESOLVED':'恢复未能确认有效，或本轮恢复预算已用完；已停止，不连续恢复。',
 'INTERRUPTED':'运行被中断。',
 'ERROR':'运行发生异常。',
}


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def describe_stop(output,result):
    output=Path(output)
    episodes=sorted(output.glob('episode_[0-9]*'))
    if episodes and 'episodes' in result:
        output=episodes[-1];result=read(output/'cli_result.json') or read(output/'episode.json') or result
    status=result.get('status','UNKNOWN')
    steps=sorted(p for p in output.glob('step_[0-9]*') if p.is_dir())
    step=steps[-1] if steps else output
    selection=read(step/'host_selection.json') or read(step/'dispatch_intent.json')
    match=read(step/'node_match.json').get('result',{})
    retry=read(step/'grounding_retry_001/node_match.json').get('result',{})
    config=read(output/'runner_config.json')
    threshold=config.get('match_threshold')
    error=result.get('error') or result.get('execution',{}).get('error')
    reason=REASONS.get(status,'运行已结束，请查看状态码和详细记录。')
    if status=='AMBIGUOUS' and match.get('status')=='MATCH':
        if match.get('violated_conditions'):
            reason='模型返回 MATCH，但同时填写了违反条件，Host 未接受该匹配。'
            error='；'.join(match['violated_conditions'])
        elif match.get('unresolved_conditions'):
            reason='模型返回 MATCH，但仍有未解决条件，Host 未接受该匹配。'
            error='；'.join(match['unresolved_conditions'])
        elif threshold is not None and match.get('confidence',0)<threshold:
            reason=f"匹配置信度 {match.get('confidence')} 低于本轮阈值 {threshold}，Host 未接受该匹配。"
        elif not match.get('matched_evidence'):
            reason='模型返回 MATCH，但缺少匹配证据，Host 未接受该匹配。'
        elif set(match.get('candidate_nodes',[]))-{match.get('node_id')}:
            reason='模型返回 MATCH，但候选列表仍含其他节点，Host 未接受该匹配。'
    elif status=='AMBIGUOUS' and match.get('status')=='AMBIGUOUS':
        reason='图像对照仍存在多个可能状态，模型没有确认唯一节点。'
        if match.get('unresolved_conditions'):error='；'.join(match['unresolved_conditions'])
    if error=='grounding recapture no longer reliably matches selected edge source':
        reason='重拍后无法继续确认所选 edge 的起点；禁止沿用旧 edge，因而停止。'
    elif error=='joint image review could not resolve the source-node conflict':
        reason='已同时对照重拍前后图片和节点代表图，仍无法可靠确认当前节点。'
    elif error and 'minimal anchors unresolved/low confidence' in error:
        reason='抓点/落点仍未可靠确定，grounding 恢复未能完成。'
    elif error and 'applicability recovery budget exhausted' in error:
        reason='该 edge 本轮的适用性恢复次数已用完。'
    elif error and 'ModelInvocationError' in error:
        reason='视觉模型调用失败，未获得可用结果；这不等于图像匹配置信度低。'
    from .replay_failures import controller_fault_details
    fault=result.get('controller_fault') or controller_fault_details(error)
    if fault:
        status='CONTROLLER_FAULT'
        reason=f"{fault['arm']} 控制器 state={fault['controller_state']}、error={fault['error_code']}、warning={fault['warning_code']}；不是视觉 grounding 失败，不自动清故障或继续动作。"
    return dict(status=status,reason=reason,controller_fault=fault,node=selection.get('node_id',selection.get('source_node',match.get('node_id'))),
                edge=selection.get('edge_id',selection.get('selected_action')),
                recapture_match=retry or None,error=error,details=str(output.resolve()),
                match_result=match or None,match_threshold=threshold,
                cleanup_error=result.get('cleanup_error'),cleanup_errors=result.get('cleanup_errors',[]))


def print_stop_summary(output,result):
    summary=describe_stop(output,result)
    print('\n===== Replay 结束原因 =====',flush=True)
    print('状态：'+summary['status'])
    print('原因：'+summary['reason'])
    print('本步原节点：'+str(summary['node'] or '尚未确认')+'；所选 edge：'+str(summary['edge'] or '尚未选择'))
    match=summary['match_result']
    if match:
        print('本次匹配：'+str(match.get('node_id') or match.get('status'))
              +'；置信度：'+str(match.get('confidence'))
              +'；阈值：'+str(summary['match_threshold'] if summary['match_threshold'] is not None else '记录未提供'))
        if match.get('candidate_nodes'):print('匹配候选：'+', '.join(match['candidate_nodes']))
    retry=summary['recapture_match']
    if retry:
        print('重拍匹配：'+str(retry.get('node_id') or retry.get('status'))+'；置信度：'+str(retry.get('confidence')))
    if summary['error']:print('具体错误：'+str(summary['error']))
    if summary['cleanup_error']:print('退出清理错误：'+str(summary['cleanup_error']))
    for error in summary['cleanup_errors']:
        print('退出清理错误：'+error['resource']+'：'+error['error'])
    print('详细记录：'+summary['details'],flush=True)
    try:
        (Path(summary['details'])/'stop_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    except OSError:pass
    return summary
