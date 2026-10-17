#!/usr/bin/env python3
"""Export one frozen graph and its node images as a small offline demo."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloth_agent.experiments.graph_policy_replay import OnlineGrowth
from cloth_agent.experiments.replay_live_graph import PAGE, projection
from scripts.compare_graph_demo import read_snapshot, digest


def export_cases(graph, output):
    records, cases = [], []
    chosen = [graph['transitions'][0], next(t for t in graph['transitions'] if t['transition_type']=='NO_PROGRESS')]
    for label, transition in zip(('observed_progress', 'observed_self_loop'), chosen):
        before, after = [Path(transition[key]['path']).parent for key in ('before', 'after')]
        step = after.parent
        case = output/'cases'/label
        case.mkdir(parents=True)
        def keep(source, relative=None):
            source = source.resolve()
            row = dict(source=str(source.relative_to(ROOT)), sha256=digest(source), bytes=source.stat().st_size)
            if relative is not None:
                target = case/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                row['demo_file'] = str(target.relative_to(output))
            records.append(row)
        for phase, observation in (('before', before), ('after', after)):
            for name in ('rgb.png', 'observation.json', 'state.json'):
                keep(observation/name, phase+'/'+name)
            for name in ('world_xyz_mm.npy', 'uncorrected_world_xyz_mm.npy'):
                if (observation/name).exists():
                    keep(observation/name)  # Already in the curated evidence release.
        for name in ('execution.json', 'outcome_match.json'):
            keep(step/name, name)
        round_dir = step/'frontier/round_001'
        for name in ('motion.py', 'effective_motion.py', 'native_plan.json', 'dispatch_intent.json',
                     'outcome.json', 'execution_events.json', 'timing.json', 'planner/prompt.txt',
                     'planner/answer.txt', 'planner/result.json'):
            keep(round_dir/name, 'execution/'+name)
        for path in sorted(round_dir.glob('gripper_*.json')):
            keep(path, 'execution/'+path.name)
        (case/'transition.json').write_text(json.dumps(transition, ensure_ascii=False, indent=2)+'\n')
        execution = json.loads((step/'execution.json').read_text())
        cases.append(dict(name=label, transition_id=transition['transition_id'],
                          source_node=transition['source_node'], action=transition['selected_action'],
                          result_node=transition['result_node'], transition_type=transition['transition_type'],
                          execution_status=execution['status'], physical_execution=execution['physical_execution'],
                          completed_commands=execution['completed_commands']))
    (output/'actual_cases.json').write_text(json.dumps(dict(cases=cases, files=records), ensure_ascii=False, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True, help='frozen graph JSON, not pointer')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    graph = read_snapshot(args.snapshot)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.snapshot, args.output/'snapshot.json')
    shutil.copyfile(Path(str(args.snapshot)+'.sha256.json'), args.output/'snapshot.json.sha256.json')
    view = projection(graph)
    view.update(current=None, action=None, target=None, phase='STOPPED',
                new_nodes=[], events=[], recoveries=[], result='REFERENCE_SNAPSHOT')
    evidence = []
    (args.output/'images').mkdir()
    for node in view['nodes']:
        record = graph['nodes'][node['id']]['representative_observation']['rgb']
        source = Path(record['path'])
        if digest(source) != record['sha256']:
            raise ValueError(f'node image checksum mismatch: {source}')
        relative = 'images/'+node['id']+source.suffix
        shutil.copyfile(source, args.output/relative)
        node.pop('references', None)
        node['images'] = [dict(label=node['id']+' · 原始观测', src=relative)]
        evidence.append(dict(node=node['id'], file=relative, sha256=record['sha256']))
    runtime = object.__new__(OnlineGrowth)
    runtime.state = graph
    runtime.latest_path = args.output/'graph.json'
    runtime.render()  # Existing renderer; does not issue robot or model calls.
    export_cases(graph, args.output)
    expected = dict(source_snapshot=str(args.snapshot), source_sha256=digest(args.snapshot),
                    nodes=len(graph['nodes']), actions=len(graph['actions']),
                    transitions=len(graph['transitions']), metrics=runtime.metrics(), evidence=evidence,
                    result_scope='Observed experience graph; not a verified completed folding task')
    (args.output/'expected.json').write_text(json.dumps(expected, ensure_ascii=False, indent=2)+'\n')
    (args.output/'state.json').write_text(json.dumps(view, ensure_ascii=False, indent=2)+'\n')
    page = PAGE.replace('__SNAPSHOT__', json.dumps(view, ensure_ascii=False).replace('<', '\\u003c'))
    banner = '''<aside style="position:fixed;top:12px;left:12px;z-index:10;background:#152231ee;padding:12px 18px;border-radius:12px;font:14px sans-serif;color:#eef3fa">
    <b>Graph 复现基准 Demo</b><br>__COUNTS__<br>
    点击节点查看原始观测；滚轮缩放，双击背景复位。<br>
    <a style="color:#a9cbff" href="graph.svg">完整 State → Action → Outcome 图</a> ·
    <a style="color:#a9cbff" href="README.md">对照说明</a><br>
    <small>真实经验快照，包含失败和自环；不代表整件折叠已成功。</small></aside>'''
    banner = banner.replace('__COUNTS__', f"{len(graph['nodes'])} 个状态 · {len(graph['actions'])} 个动作 · {len(graph['transitions'])} 条已观测转移")
    page = page.replace('<body>', '<body>'+banner)
    # The sample is frozen: never poll a local live session over HTTP.
    page = page.replace('}poll();', '}/* frozen demo: polling disabled */')
    (args.output/'index.html').write_text(page)
    print(json.dumps(dict(output=str(args.output), nodes=expected['nodes'],
                          actions=expected['actions'], transitions=expected['transitions'])))


if __name__ == '__main__':
    main()
