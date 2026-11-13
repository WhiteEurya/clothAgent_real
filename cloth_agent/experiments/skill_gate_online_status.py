"""Read saved pilot evidence without camera, robot or model access."""
import json
from pathlib import Path


def read_current(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        # A running writer may be replacing its snapshot. Do not invent status.
        return {}


def resolve_batch(path, reports):
    if str(path) != 'latest':
        root = Path(path).resolve()
        if not (root/'batch_summary.json').is_file():
            raise ValueError('找不到批次状态文件：'+str(root))
        return root
    candidates = list(Path(reports).glob('skill_gate_online_*/batch_summary.json'))
    if not candidates:
        raise ValueError('尚无 Online Pilot 批次记录')
    return max(candidates, key=lambda p:p.stat().st_mtime_ns).parent


def describe_batch(root):
    root = Path(root)
    batch = read_current(root/'batch_summary.json')
    runs = []
    totals = {'SIMPLE':0, 'TANGENT':0}
    for item in batch.get('runs', []):
        folder = root/f"run_{item['run_id']:02d}"
        episode = read_current(folder/'episode_summary.json')
        initialization = episode.get('initialization') or read_current(folder/'initialization/result.json')
        steps = []
        for path in sorted(folder.glob('round_*/exploration.json')):
            record = read_current(path)
            if not record:
                continue
            execution = read_current(path.parent/'execution.json') or record.get('execution', {})
            rejection = read_current(path.parent/'translation_rejection.json')
            skill = record.get('forced_skill')
            if execution.get('status') == 'EXECUTED' and skill in totals:
                totals[skill] += 1
            steps.append(dict(iteration=int(path.parent.name[6:]), skill=skill,
                probability=record.get('selection_probability'),
                planning=record.get('planning_status'), requests=record.get('model_requests'),
                execution=execution.get('status','NOT_DISPATCHED'),
                reason=execution.get('error') or record.get('abstain_reason'),
                distance_mm=rejection.get('distance_mm'), minimum_mm=rejection.get('minimum_mm'),
                maximum_mm=rejection.get('maximum_mm'), evidence=str(path)))
        status = episode.get('status', item.get('status', 'UNKNOWN'))
        reason = episode.get('error') or initialization.get('error') or item.get('error')
        stage = '初始化' if status == 'RESET_FAILED' else '展开'
        if not steps and not initialization:
            stage = '启动准备/采集（等待阶段记录）'
        runs.append(dict(run_id=item['run_id'], status=status, stage=stage,
            reset_status=initialization.get('status','NOT_RECORDED'), reason=reason,
            steps=steps, evidence=str(folder/'episode_summary.json'),
            video_manifest=str(folder/'video_recording.json') if (folder/'video_recording.json').exists() else None))
    return dict(output=str(root), status=batch.get('status','UNKNOWN_OR_UPDATING'),
        run_limit=batch.get('run_limit',6), runs=runs, executed_skills=totals,
        reason=batch.get('reason') or batch.get('error'))


def print_report(report):
    print(f"\n实验状态：{report['status']}；已开始 {len(report['runs'])}/{report['run_limit']} 次", flush=True)
    print('目录：'+report['output'], flush=True)
    for run in report['runs']:
        print(f"Run {run['run_id']:02d}：{run['status']}；初始化={run['reset_status']}；阶段={run['stage']}")
        for step in run['steps']:
            print(f"  iter {step['iteration']:03d} {step['skill']}：规划={step['planning']}，执行={step['execution']}，请求={step['requests']}")
            if step['distance_mm'] is not None:
                print(f"    距离拒绝：{step['distance_mm']:.1f} mm；允许 {step['minimum_mm']}–{step['maximum_mm']} mm")
            elif step['reason']:
                print('    原因：'+str(step['reason']))
        if run['reason']:
            print('  结束原因：'+str(run['reason']))
    print('实际展开动作：'+', '.join(f'{k}={v}' for k,v in report['executed_skills'].items()))
    if report['reason']:
        print('批次停止原因：'+str(report['reason']))
    print('状态来自保存日志；RUNNING 不是进程存活或机器人运动的独立证明。', flush=True)
