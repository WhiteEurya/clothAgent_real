"""Paired current-only versus previous-image replay using the current unfold prompt."""
import argparse
import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import statistics
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import replay_unfold_goal_offline as replay
from cloth_agent.experiments import unfold_topdown


class PayloadCaptured(Exception):
    pass


def capture_payload(root, source, folder, runtime):
    observation = replay.Observation.load(replay.read(source/'observation_used.json')['directory'])
    view = replay.DualPrimaryView(observation, source/'topdown', replay.sources(observation))
    collected = []

    def capture(directory, body, headers, timeout):
        collected.append(copy.deepcopy(body))
        raise PayloadCaptured()

    with patch.object(unfold_topdown, 'primary', return_value=view):
        try:
            replay.unfold.plan_round(root, folder, observation, {}, 1, capture, runtime=runtime)
        except PayloadCaptured:
            pass
    if len(collected) != 1:
        raise ValueError('expected exactly one prepared request')
    return collected[0]


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'goal').mkdir(exist_ok=True)
    replay.shutil.copyfile(args.goal, args.output/'goal/target.png')
    runtime = replay.unfold.configure_runtime(replay.PartitionedReplayRuntime(replay.unfold.DEFAULT_CONFIG, planning_only=True))
    jobs = []
    try:
        folders = sorted(path for path in args.source.glob('round_*') if (path/'plan.json').exists())
        for source in folders[1:]:
            index = int(source.name.split('_')[1])
            previous = args.source/f'round_{index-1:03d}'
            if replay.read(previous/'execution.json')['status'] != 'EXECUTED':
                raise ValueError('history must precede an actually executed action')
            if Path(replay.read(source/'observation_used.json')['directory']).resolve() != (previous/'after').resolve():
                raise ValueError('current observation does not follow the previous action')
            prepared = args.output/'prepared'/source.name
            prepared.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix='payload-', dir=prepared) as temporary:
                base = capture_payload(args.output, source, Path(temporary), runtime)
            history = copy.deepcopy(base)
            replay.unfold.add_previous_image(history, prepared, dict(
                planning_before_image=previous/'topdown/planning_rgb.png', source_round=previous.name))
            if replay.image_entries(history)[:2] != replay.image_entries(base):
                raise ValueError('history variant changed current/goal images')
            if len(replay.image_entries(base)) != 2 or len(replay.image_entries(history)) != 3:
                raise ValueError('unexpected image counts')
            if '不提供历史图' in history['instructions']:
                raise ValueError('history variant retains conflicting instruction')
            for forbidden in ('留余量', '合理余量', '不要一次拉到', '不一步拉到', '可分多轮展开'):
                if forbidden in base['instructions'] or forbidden in history['instructions']:
                    raise ValueError('old short-pull instruction remains: '+forbidden)
            for condition, body in (('current_only', base), ('with_previous', history)):
                replay.io.save(prepared/f'{condition}_request.json', body)
                (prepared/f'{condition}_prompt.txt').write_text(body['instructions'], encoding='utf-8')
            ordered = [('current_only', base), ('with_previous', history)]
            if index % 2 == 0:
                ordered.reverse()
            jobs.extend((source, condition, body) for condition, body in ordered)
    finally:
        runtime.close()
    replay.io.save(args.output/'protocol.json', dict(
        source=str(args.source.resolve()), goal=str(args.goal.resolve()), goal_sha256=replay.digest(args.goal),
        paired_states=len(jobs)//2, conditions=['current_only', 'with_previous'], image_counts=[2,3],
        history='only immediately previous actually executed round before-image; no previous action commands',
        excluded='round_000 has no previous unfolding action image',
        prompt='current production prompt after removal of short-pull guidance; both conditions share it',
        model=replay.io.MODEL, effort=replay.io.EFFORT, workers=2,
        order='alternate condition order per pair; two concurrent requests',
        repeats_per_condition=1, physical_execution=False, robot_calls=0, camera_calls=0,
        timing='HTTP model latency includes any selection/distance retries; offline total also includes geometry and preflight',
        limitations=['Saved states are not a new physical rollout', 'Single sample per condition; no statistical significance claim',
                     'Service variability and cache may affect latency; offline total includes IK compilation']))
    return jobs


def metrics(rows):
    model = [row['model_latency_s'] for row in rows]
    total = [row['offline_total_latency_s'] for row in rows]
    request_metrics = [item for row in rows for item in row['request_metrics']]
    distances = [row['geometry']['distance_mm'] for row in rows if 'geometry' in row]
    return dict(states=len(rows), model_mean_s=statistics.mean(model), model_median_s=statistics.median(model),
                offline_mean_s=statistics.mean(total), offline_median_s=statistics.median(total),
                requests=sum(row['model_requests'] for row in rows),
                decisions=dict(Counter(row.get('new_plan', {}).get('decision', 'ERROR') for row in rows)),
                skills=dict(Counter(row.get('new_selection', {}).get('skill') or 'none' for row in rows)),
                checks=dict(Counter(row['status'] for row in rows)),
                mean_distance_mm=statistics.mean(distances) if distances else None,
                input_tokens=sum(item.get('usage', {}).get('input_tokens', 0) for item in request_metrics),
                output_tokens=sum(item.get('usage', {}).get('output_tokens', 0) for item in request_metrics),
                reasoning_tokens=sum(item.get('usage', {}).get('output_tokens_details', {}).get('reasoning_tokens', 0) for item in request_metrics))


def report(args, rows):
    paired = []
    for name in sorted({row['source'] for row in rows}):
        pair = {row['condition']:row for row in rows if row['source']==name}
        if len(pair) != 2:
            continue
        baseline, history = pair['current_only'], pair['with_previous']
        old_plan, new_plan = baseline.get('new_plan', {}), history.get('new_plan', {})
        old_selection, new_selection = baseline.get('new_selection', {}), history.get('new_selection', {})
        valid = bool(old_plan and new_plan)
        both_action = old_plan.get('decision') == new_plan.get('decision') == 'ACTION'
        old_arm = baseline.get('geometry', {}).get('assignment', {}).get('active_arm')
        new_arm = history.get('geometry', {}).get('assignment', {}).get('active_arm')
        paired.append(dict(round=baseline['round'], source=name, valid=valid,
            decision_changed=old_plan.get('decision')!=new_plan.get('decision') if valid else None,
            skill_changed=old_selection.get('skill')!=new_selection.get('skill') if valid else None,
            points_changed=any(old_plan[key]!=new_plan[key] for key in ('grasp_px','target_px')) if both_action else None,
            arm_changed=old_arm!=new_arm if old_arm and new_arm else None,
            baseline_status=baseline['status'], history_status=history['status'],
            baseline_distance_mm=baseline.get('geometry', {}).get('distance_mm'),
            history_distance_mm=history.get('geometry', {}).get('distance_mm'),
            baseline_model_s=baseline['model_latency_s'], history_model_s=history['model_latency_s'],
            model_delta_s=history['model_latency_s']-baseline['model_latency_s'],
            baseline_evidence=old_selection.get('evidence'), history_evidence=new_selection.get('evidence')))
    summary = {condition: metrics([row for row in rows if row['condition']==condition])
               for condition in ('current_only', 'with_previous')}
    summary.update(pairs=len(paired), decision_changed=sum(row['decision_changed'] is True for row in paired),
                   skill_changed=sum(row['skill_changed'] is True for row in paired),
                   points_changed=sum(row['points_changed'] is True for row in paired),
                   arm_changed=sum(row['arm_changed'] is True for row in paired),
                   model_slower_pairs=sum(row['model_delta_s']>0 for row in paired), physical_execution=False)
    replay.io.save(args.output/'summary.json', summary)
    replay.io.save(args.output/'paired_results.json', paired)
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22)
    canvas = Image.new('RGB', (1680, len(paired)*410+60), '#edf2f6')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), '左：当前图 + 新目标图；中：增加上一轮动作前图。两组均使用修改后的距离提示，均未实机执行。', font=font, fill='#192c42')
    for index, pair in enumerate(paired):
        top = 60+410*index
        draw.text((15, top), f'第 {pair["round"]} 轮 / {pair["source"]}', font=font, fill='#192c42')
        for column, condition in enumerate(('current_only','with_previous')):
            image = args.output/condition/pair['source']/'new_action.png'
            if image.exists():
                picture = Image.open(image).convert('RGB').crop((165,160,430,385)).resize((424,360))
                canvas.paste(picture,(15+column*450,top+35))
        text = (f'模型耗时：{pair["baseline_model_s"]:.1f}s → {pair["history_model_s"]:.1f}s；'
                f'检查：{pair["baseline_status"]} → {pair["history_status"]}')
        cursor = replay.wrap(draw,text,(920,top+10),740,font)
        cursor = replay.wrap(draw,'无历史：'+str(pair['baseline_evidence']),(920,cursor+5),740,font)
        replay.wrap(draw,'有历史：'+str(pair['history_evidence']),(920,cursor+5),740,font)
        canvas.crop((0,top,1680,top+410)).save(args.output/f'{pair["source"]}_comparison.png')
    canvas.save(args.output/'comparison_overview.jpg', quality=93)
    chart = Image.new('RGB', (1500, 100+len(paired)*70), '#edf2f6')
    chart_draw = ImageDraw.Draw(chart)
    chart_draw.text((20, 12), '蓝色：当前图＋目标图；橙色：增加上一轮图像。每组每状态一次采样，未执行新动作。', font=font, fill='#192c42')
    for column, (key, label, unit) in enumerate((('model_s','模型耗时（包含重选请求）','s'),('distance_mm','候选抓放点实测水平距离','mm'))):
        left = column*750
        chart_draw.text((left+20, 50), label, font=font, fill='#192c42')
        maximum = max(pair[f'{prefix}_{key}'] or 0 for pair in paired for prefix in ('baseline','history')) or 1
        for index, pair in enumerate(paired):
            top = 95+index*70
            chart_draw.text((left+15,top+10), f'第{pair["round"]}轮', font=font, fill='#192c42')
            for offset, prefix, color in ((0,'baseline','#2776bf'),(30,'history','#db782a')):
                value = pair[f'{prefix}_{key}']
                width = (value or 0)/maximum*490
                chart_draw.rectangle((left+110,top+offset,left+110+width,top+offset+20),fill=color)
                text = f'{value:.1f} {unit}' if value is not None else '无有效动作'
                chart_draw.text((left+120+width,top+offset-5),text,font=font,fill='#192c42')
    chart.save(args.output/'latency_and_distance.png')
    print(replay.json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--goal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-model', action='store_true')
    args = parser.parse_args()
    jobs = prepare(args)
    print('VALIDATED '+str(len(jobs))+' requests over '+str(len(jobs)//2)+' paired states',flush=True)
    if not args.run_model:
        return
    for condition in ('current_only','with_previous'):
        (args.output/condition).mkdir(exist_ok=True)
    headers = replay.io.auth_headers()
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}
        for source, condition, body in jobs:
            saved = args.output/condition/source.name/'comparison.json'
            if saved.exists():
                row=replay.read(saved);row['condition']=condition;rows.append(row)
                continue
            future=pool.submit(replay.trial,source,args.output/condition,body,headers,900)
            futures[future]=condition
        for future in as_completed(futures):
            row=future.result();row['condition']=futures[future];rows.append(row)
            replay.io.save(args.output/'results.json',rows)
            print(replay.json.dumps({key:row.get(key) for key in ('round','condition','status','new_selection','model_latency_s','model_requests','error')},ensure_ascii=False),flush=True)
    rows.sort(key=lambda row:(row['round'],row['condition']))
    replay.io.save(args.output/'results.json',rows)
    report(args, rows)


if __name__ == '__main__':
    main()
