"""Compare existing minimal one-shot with saved unfold requests, without hardware."""
import copy
import json
from pathlib import Path
import statistics
import sys
from datetime import datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
from cloth_agent.experiments.minimal_one_shot_6iter import PROMPT
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once


def main():
    source=Path(sys.argv[1]).resolve()
    out=Path('reports')/('oneshot_latency_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    cases=[]
    for index in range(3):
        path=source/f'round_{index:03d}/skill_selection/request_payload.json'
        current=json.loads(path.read_text(encoding='utf-8'))
        minimal=copy.deepcopy(current)
        minimal['instructions']=PROMPT
        schema=copy.deepcopy(io.ROTATION_SCHEMA)
        schema['properties']['grasp_angle_deg']={'type':'null'}
        minimal['text']['format'].update(name='unfold_single_step',schema=schema)
        order=['minimal','current'] if index%2==0 else ['current','minimal']
        for condition in order:
            cases.append((index,condition,minimal if condition=='minimal' else current))
    io.save(out/'protocol.json',dict(source=str(source),requests=6,model=io.MODEL,effort=io.EFFORT,
        baseline='cloth_agent/experiments/minimal_one_shot_6iter.py:PROMPT',
        same_input_images_and_user_content=True,sequential=True,robot_calls=0,
        scope='HTTP planning latency only; prompts and output schemas differ; no physical effectiveness comparison'))
    print(str(out),flush=True)
    headers=io.auth_headers();rows=[]
    for index,condition,body in cases:
        folder=out/f'round_{index:03d}_{condition}'
        try:
            plan,m=direct_once(folder,body,headers,300)
            usage=m.get('usage') or {}
            row=dict(round=index,condition=condition,status=m['status'],decision=plan['decision'],
                latency_s=m['wall_latency_s'],input_tokens=usage.get('input_tokens'),
                output_tokens=usage.get('output_tokens'),
                reasoning_tokens=usage.get('output_tokens_details',{}).get('reasoning_tokens'),
                cached_tokens=usage.get('input_tokens_details',{}).get('cached_tokens'))
        except Exception as exc:
            row=dict(round=index,condition=condition,status='ERROR',error=f'{type(exc).__name__}: {exc}')
        rows.append(row);io.save(out/'results.json',rows);print(json.dumps(row,ensure_ascii=False),flush=True)
    summary={}
    for condition in ['minimal','current']:
        values=[r['latency_s'] for r in rows if r['condition']==condition and 'latency_s' in r]
        summary[condition]=dict(n=len(values),mean_s=statistics.mean(values) if values else None,
            median_s=statistics.median(values) if values else None,values_s=values)
    io.save(out/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
