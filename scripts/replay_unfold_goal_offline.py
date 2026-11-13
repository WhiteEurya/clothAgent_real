"""Replay saved unfold requests with a replacement goal and no hardware access."""
import argparse
import ast
import base64
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloth_agent.dual_arm.observation import Observation
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once
from cloth_agent.experiments import partitioned_unfold as unfold
from cloth_agent.experiments import unfold_skill_planner as skills
from cloth_agent.experiments.unfold_dual_topdown import DualPrimaryView, sources
from cloth_agent.replay_dual_arm_runtime import PartitionedReplayRuntime
from cloth_agent.robot_collision import RobotCollision


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_entries(body):
    return [item for message in body['input'] for item in message['content']
            if item['type'] == 'input_image']


def validate_case(folder, goal):
    body = read(folder/'skill_selection/request_payload.json')
    entries = image_entries(body)
    if len(entries) != 2:
        raise ValueError(f'{folder}: expected exactly current and goal images')
    for entry, path in zip(entries, (folder/'topdown/planning_rgb.png', folder.parent/'goal/target.png')):
        if base64.b64decode(entry['image_url'].split(',', 1)[1]) != path.read_bytes():
            raise ValueError(f'{folder}: saved request does not match {path}')
    if body['model'] != io.MODEL or body['reasoning']['effort'] != io.EFFORT:
        raise ValueError('saved model/effort differs from current transport')
    replaced = copy.deepcopy(body)
    image_entries(replaced)[1]['image_url'] = io.image_data(goal)
    restored = copy.deepcopy(replaced)
    image_entries(restored)[1]['image_url'] = entries[1]['image_url']
    if restored != body:
        raise ValueError('replacement changed more than the goal image')
    observation = Observation.load(read(folder/'observation_used.json')['directory'])
    sources(observation)
    return replaced


def workspace_check(runtime, source, observation):
    config, program, assignment = runtime.compile(source, observation)
    actions = []
    for index, statement in enumerate(ast.parse(program).body[0].body, 1):
        call = statement.value
        arguments = {}
        if call.func.id == 'move':
            values = [ast.literal_eval(argument) for argument in call.args]
            arguments = dict(zip(('x', 'y', 'z', 'yaw'), values))
            try:
                config.validate_workspace_pose(*values[:3], relative_yaw_deg=values[3])
            except (ValueError, RuntimeError) as error:
                return dict(status='REJECTED', error=f'action {index}: {error}', assignment=assignment)
        actions.append(dict(name=call.func.id, args=arguments))
    checker = RobotCollision(runtime.config_path, observation.meta['joints_deg'], assignment['active_arm'], root=runtime.config.root)
    try:
        plans = checker.require_execution(config, actions)
    except (ValueError, RuntimeError) as error:
        return dict(status='REJECTED', error=f'{type(error).__name__}: {error}', assignment=assignment)
    return dict(status='PASSED', assignment=assignment, checked_actions=len(plans))


def trial(folder, output, body, headers, timeout):
    started = time.perf_counter()
    target = output/folder.name
    target.mkdir()
    shutil.copytree(folder/'topdown', target/'topdown', ignore=shutil.ignore_patterns('depth_resolution.json'))
    observation = Observation.load(read(folder/'observation_used.json')['directory'])
    view = DualPrimaryView(observation, target/'topdown', sources(observation))
    observation.planning_view = view
    runtime = unfold.configure_runtime(PartitionedReplayRuntime(unfold.DEFAULT_CONFIG, planning_only=True))
    manifest = read(folder/'planning_skills.json')
    template = copy.deepcopy(body)
    template['instructions'] = ''
    request_count = 0
    request_metrics = []

    def replay_direct(directory, generated, request_headers, request_timeout):
        nonlocal request_count
        suffix = '\n'+skills.ACTION_GUIDANCE+'\n'
        if not generated['instructions'].endswith(suffix):
            raise ValueError('unexpected planner prompt assembly')
        feedback = generated['instructions'][:-len(suffix)]
        actual = copy.deepcopy(body)
        actual['instructions'] += feedback
        if request_count == 0 and actual != body:
            raise ValueError('first request must differ only in goal image')
        request_count += 1
        (directory/'planner_prompt.txt').write_text(actual['instructions'], encoding='utf-8')
        io.save(directory/'actual_request_payload.json', actual)
        value, metrics = direct_once(directory, actual, request_headers, request_timeout)
        request_metrics.append(metrics)
        return value, metrics

    row = dict(round=int(folder.name.split('_')[1])+1, source=folder.name, robot_calls=0,
               old_selection=read(folder/'skill_selection/selection.json'),
               old_plan=read(folder/'plan.json'),
               old_execution_status=read(folder/'execution.json')['status'],
               physical_execution=False, ik_collision_check='NOT_EVALUATED')
    current = target
    try:
        for attempt in range(3):
            io.save(current/'planning_skills.json', manifest)
            plan, metrics = skills.run(output, current, view, template, headers, timeout,
                                      replay_direct, selection_prompt='', max_attempts=3)
            io.save(current/'plan.json', plan)
            selected = Path(read(current/'depth_reselection.json')['selected_attempt'])
            row.update(new_selection=read(selected/'skill_selection/selection.json'), new_plan=plan,
                       selected_attempt=str(selected), planning_directory=str(current), model_metrics=metrics)
            if plan['decision'] != 'ACTION':
                row['status'] = plan['decision']
                break
            try:
                source = unfold.prepare_action(runtime, observation, plan, current)
            except unfold.TranslationRejected as error:
                feedback = dict(error.feedback, rejected_plan=plan, rejection_reason='XY_TRANSLATION_OUT_OF_RANGE')
                io.save(current/'translation_rejection.json', feedback)
                if attempt == 2:
                    row.update(status='DISTANCE_REJECTED', error=str(error))
                    break
                current = target/f'replan_{attempt+1:02d}'
                current.mkdir()
                io.save(current/'rejection_feedback.json', feedback)
                continue
            row['geometry'] = read(current/'geometry.json')
            row['workspace_check'] = workspace_check(runtime, source, observation)
            row['ik_collision_check'] = 'OFFLINE_PREFLIGHT_FROM_SAVED_JOINTS'
            row['status'] = 'CANDIDATE' if row['workspace_check']['status'] == 'PASSED' else 'PREFLIGHT_REJECTED'
            break
        io.overlay(view.image, row['new_plan'], target/'new_action.png', 'NEW GOAL | OFFLINE | NOT EXECUTED')
    except Exception as error:
        row.update(status='ERROR', error=f'{type(error).__name__}: {error}')
    finally:
        runtime.close()
        for actual_path in target.rglob('actual_request_payload.json'):
            shutil.copyfile(actual_path, actual_path.with_name('request_payload.json'))
        row['model_requests'] = request_count
        row['request_metrics'] = request_metrics
        row['model_latency_s'] = sum(metrics['wall_latency_s'] for metrics in request_metrics)
        row['offline_total_latency_s'] = time.perf_counter()-started
        io.save(target/'comparison.json', row)
    return row


def wrap(draw, text, origin, width, font):
    left, top = origin
    line = ''
    for character in text:
        if draw.textlength(line+character, font=font) > width:
            draw.text((left, top), line, font=font, fill='#192c42')
            top += font.size+6
            line = ''
        line += character
    draw.text((left, top), line, font=font, fill='#192c42')
    return top+font.size+6


def visualize(source, output, rows):
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22)
    overview = Image.new('RGB', (1440, len(rows)*390+60), '#edf2f6')
    draw = ImageDraw.Draw(overview)
    draw.text((20, 15), '旧目标规划（左） vs 新目标规划（中） | 同一批历史图片，未执行新动作', font=font, fill='#192c42')
    for index, row in enumerate(rows):
        top = 60+index*390
        name = row['source']
        for column, path in enumerate((source/name/'proposed_action_overlay.png', output/name/'new_action.png')):
            if path.exists():
                picture = Image.open(path).convert('RGB').crop((165, 160, 430, 385)).resize((424, 360))
                overview.paste(picture, (column*450+10, top+25))
        old = row['old_selection']
        new = row.get('new_selection', {})
        draw.text((10, top), f'第 {row["round"]} 轮 / {name}', font=font, fill='#192c42')
        lines = [f'旧：{old["decision"]} / {old.get("skill")}',
                 f'新：{new.get("decision")} / {new.get("skill")}',
                 f'离线检查：{row["status"]}', new.get('evidence', row.get('error', ''))]
        cursor = top+20
        for line in lines:
            cursor = wrap(draw, line, (920, cursor), 500, font)+6
        overview.crop((0, top, 1440, top+390)).save(output/name/'comparison.png')
    overview.save(output/'comparison_overview.jpg', quality=93)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--goal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-model', action='store_true')
    parser.add_argument('--workers', type=int, default=2, choices=(1, 2))
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args()
    folders = sorted(path for path in args.source.glob('round_*') if (path/'plan.json').exists())
    if not folders:
        raise ValueError('no completed source planning rounds')
    cases = [(folder, validate_case(folder, args.goal)) for folder in folders]
    print(json.dumps(dict(status='VALIDATED', rounds=len(cases), goal_sha256=digest(args.goal), robot_calls=0)), flush=True)
    if not args.run_model:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'goal').mkdir()
    shutil.copyfile(args.goal, args.output/'goal/target.png')
    shutil.copyfile(args.source/'goal/target.png', args.output/'goal/old_target.png')
    io.save(args.output/'protocol.json', dict(source=str(args.source.resolve()), goal=str(args.goal.resolve()),
        goal_sha256=digest(args.goal), old_goal_sha256=digest(args.source/'goal/target.png'),
        first_request_change='goal image only; exact saved payload otherwise', rounds=len(cases),
        model=io.MODEL, effort=io.EFFORT, workers=args.workers, robot_calls=0, camera_calls=0,
        retries='existing candidate/depth and translation reselection, maximum 3 attempts each',
        checks='saved measured depth, geometry, translation, offline RobotCollision preflight from saved joints',
        limitations=['No physical execution or new trajectory rollout', 'No live controller state/reachability checks',
                     'One new run per saved state; model sampling variation is not isolated',
                     'Later inputs remain old trajectory states even if an earlier new plan differs or stops']))
    headers = io.auth_headers()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(trial, folder, args.output, body, headers, args.timeout) for folder, body in cases]
        for future in as_completed(jobs):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda item: item['round'])
            io.save(args.output/'results.json', rows)
            print(json.dumps({key: row.get(key) for key in ('round', 'status', 'new_selection', 'error', 'model_requests')}, ensure_ascii=False), flush=True)
    visualize(args.source, args.output, rows)
    print('COMPLETE '+str(args.output), flush=True)


if __name__ == '__main__':
    main()
