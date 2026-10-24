"""FastPlanner + shadow gate, with optional bounded pre-dispatch Codex recovery."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import secrets
import shutil
import signal
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from . import one_shot_goal_planner_probe_v1 as planner
from .trajectory_memory_full_episode_v1 import (
    EpisodeCollectionRuntime, prepare_action as prepare_collection_action, repeated_action, unchanged,
)
from cloth_agent.dual_arm.geometry import DualArmError, matrix_pose, pose_matrix
from cloth_agent.dual_arm.observation import DepthUnavailableError
from cloth_agent.gamepad_collect import TrajectoryRecorder, approach_program, tcp_command
from cloth_agent import random_fold_reset as reset
from cloth_agent.pre_action_gate import PreActionGate, GATE_PROMPT, GATE_SCHEMA
from cloth_agent.unfolding_planners import FastPlanner, ShadowPlannerRouter
from cloth_agent import planner_shadow_review
from cloth_agent import table_target_depth
from cloth_agent import codex_recovery
from cloth_agent import one_shot_views, one_shot_orientation

ROOT = Path('reports/one_shot_goal_full_episode_v1')
DEFAULT_GOAL = Path('reports/goal_only_full_episode_v1_restart02/goal/target.png')
save, read, sha = planner.save, planner.read, planner.sha
PROTECTED = ['cloth_agent/codex_native_runtime.py', 'cloth_agent/gamepad_collect_runtime.py',
             'cloth_agent/gamepad_collect.py', 'cloth_agent/random_fold_reset.py',
             'config/dual_arm.local.json', 'cloth_agent/experiments/one_shot_goal_planner_probe_v1.py',
             'cloth_agent/one_shot_views.py', 'cloth_agent/one_shot_orientation.py',
             'cloth_agent/grasp_angle.py', 'cloth_agent/dual_arm/observation.py',
             'cloth_agent/experiments/trajectory_memory_full_episode_v1.py']
INITIALIZATION_RELEASE_POLICY = 'HOME_Y_HOME_YAW_V1'


def prepare_action(runtime, before, plan, folder):
    return prepare_collection_action(runtime, before, plan, folder,
                                     target_sampler=table_target_depth.project_target,
                                     orientation_resolver=one_shot_orientation.resolve,
                                     check_grasp_height_spread=False,
                                     smoothing=bool(plan and plan.get("is_smoothing",False)))


def sample_initialization_release(seed):
    """Keep Home Y and orientation; retain existing X/height variation."""
    release = reset.sample_release(seed, 300., 50., 0., 40.)
    release['drop_offset_xy'][1] = 0.
    return release


def direct_once(folder, body, headers, timeout_s):
    """Exactly one HTTP dispatch. Persist attempt before dispatch; never retry."""
    folder.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(body, ensure_ascii=False).encode()
    with (folder/'attempt.json').open('x') as stream:
        json.dump(dict(started_at=planner.utc(), http_attempts=1, retries=0), stream)
    (folder/'request_payload.json').write_bytes(payload)
    result = dict(status='FAILED', http_attempts=1, retries=0, model_response_count=0, tool_calls=0)
    events = []
    started = time.perf_counter()
    try:
        with planner.requests.Session() as session:
            session.trust_env = False
            session.mount('https://', planner.requests.adapters.HTTPAdapter(max_retries=0))
            with session.post(planner.ENDPOINT, headers=headers, data=payload, stream=True,
                              timeout=(30, timeout_s), allow_redirects=False) as response:
                result['http_status'] = response.status_code
                if response.status_code != 200:
                    (folder/'raw_http_error.txt').write_text(response.text)
                    raise RuntimeError(f'HTTP {response.status_code}; no retry')
                with (folder/'raw_response.sse').open('wb') as raw:
                    for line in response.iter_lines():
                        raw.write(line+b'\n'); raw.flush()
                        if line.startswith(b'data: ') and line[6:] != b'[DONE]':
                            events.append(json.loads(line[6:]))
                            if events[-1].get('type') in ('response.completed', 'response.failed'):
                                break
        result['wall_latency_s'] = time.perf_counter()-started
        result['model_response_count'] = len({e['response']['id'] for e in events
                                              if e.get('type') == 'response.created'})
        response, text = planner.parse_events(events)
        save(folder/'raw_response.json', response)
        (folder/'raw_output.txt').write_text(text)
        if response.get('model') != planner.MODEL or response.get('reasoning', {}).get('effort') != planner.EFFORT:
            raise ValueError('returned model/effort differs from frozen protocol')
        result.update(returned_model=response['model'], reasoning=response['reasoning'], usage=response.get('usage'))
        value = json.loads(text)
        planner.jsonschema.validate(value, body['text']['format']['schema'])
        result['status'] = 'VALID_JSON'
        return value, result
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result.setdefault('wall_latency_s', time.perf_counter()-started)
        save(folder/'model_metrics.json', result)


def plan_round(root, folder, before, headers, timeout_s):
    started = time.perf_counter()
    with Image.open(before.image) as image:
        size = list(image.size)
    body = planner.payload(before.image, root/'goal/target.png', size,
                           right_view=folder/'right_view/rgb.png', allow_grasp_rotation=True)
    manifest = dict(model=planner.MODEL, effort=planner.EFFORT, input_image_count=3,
        current_rgb='before/rgb.png', current_sha256=sha(before.image), goal_rgb='goal/target.png',
        goal_sha256=sha(root/'goal/target.png'),
        right_view_rgb=str(folder/'right_view/rgb.png'), right_view_sha256=sha(folder/'right_view/rgb.png'),
        coordinate_frame='perception before/rgb.png', image_size_px=size,
        prompt_chars=len(body['instructions'])+len(body['input'][0]['content'][0]['text']),
        prompt_sha256=sha(root/'prompt.txt'), tools=[], tool_choice='none', retries=0,
        episode_history_supplied=False, trajectory_memory=False, depth_supplied_to_model=False,
        host_preparation_s=time.perf_counter()-started)
    save(folder/'request_manifest.json', manifest)
    plan, metrics = direct_once(folder, body, headers, timeout_s)
    # Preserve even a schema-valid but semantically invalid proposal; never repair it.
    save(folder/'plan.json', plan)
    planner.validate_plan(plan, size, require_grasp_rotation=True)
    planner.overlay(before.image, plan, folder/'proposed_action_overlay.png', folder.name)
    return plan, metrics


def press_dropped_cloth(runtime, folder, observation, headers, timeout_s, rpy, recorder, report):
    """One depth-grounded vertical touch to the configured hard-table Z, then lift."""
    support=runtime.config.raw.get('environment_confirmation',{}).get('support_layer',{})
    table=next((o for o in runtime.config.obstacles if o.get('name')=='table_z0_user_confirmed'),None)
    if (runtime.config.raw.get('frame_id')!='world_mm' or support.get('type')!='sponge'
            or table is None or float(table['max_mm'][2])!=0.0):
        raise DualArmError('post-drop press requires the configured sponge and hard-table world Z=0')
    observation.validate_for(runtime.config,live=True,check_age=False)
    size=list(Image.open(observation.image).size)
    body=planner.payload(observation.image,observation.image,size)
    body['instructions']=reset.PROMPT+'''\nChoose a visible central body-fabric patch for one vertical press after a drop.
Avoid any visible arm, gripper, table or narrow edge. Do not choose a hidden part.
Return only the region schema; no action or desired press force.'''
    body['input'][0]['content']=[dict(type='input_text',text=f'CURRENT RGB dimensions: {size} pixels.'),
        dict(type='input_image',image_url=planner.image_data(observation.image),detail='high')]
    body['text']['format'].update(name='post_drop_press_region',schema=reset.REGION)
    region,metrics=direct_once(folder/'press_region_request',body,headers,timeout_s)
    report.update(model_calls=2,press_region_request_s=metrics['wall_latency_s'],press_status='PLANNING')
    save(folder/'result.json',report)
    save(folder/'press_region.json',region)
    reset.validate_region(region,size)
    if runtime.cancel.is_set():raise KeyboardInterrupt('cancelled before post-drop press')
    pixel=region['center_px']
    surface=np.asarray(observation.sample(pixel,runtime.config.limits['max_surface_spread_mm']),dtype=float)
    report['press_center_px']=list(pixel)
    report['press_surface_world_mm']=surface.tolist()
    save(folder/'result.json',report)
    if surface.shape!=(3,) or not np.isfinite(surface).all() or not 5<=surface[2]<=35:
        raise DualArmError(f'post-drop press center has unsupported measured height above hard table: '
                           f'z={surface[2] if surface.shape==(3,) else None} mm, allowed 5..35 mm')
    rpy=np.asarray(rpy,dtype=float)
    above=np.r_[surface[:2],max(surface[2]+50,60),rpy]
    press=above.copy();press[2]=0.0
    arm=runtime.config.arms['left'];world_to_base=np.linalg.inv(arm.world_from_base)
    for pose in (above,press):arm.validate_point((world_to_base@np.r_[pose[:3],1])[:3])
    source=(approach_program(runtime,above)+'motion.gripper(arm="left", action="close")\n'
            +tcp_command(press)+tcp_command(above)+'motion.gripper(arm="left", action="open")\n')
    action=folder/'post_drop_press';action.mkdir()
    (action/'motion.py').write_text(source)
    save(folder/'press_plan.json',dict(pixel_xy=pixel,surface_world_xyz_mm=surface.tolist(),
        hard_table_world_z_mm=0.0,press_world_pose_mm_deg=press.tolist(),above_world_pose_mm_deg=above.tolist(),
        model_request_s=metrics['wall_latency_s'],source_observation_sha256=sha(observation.image),
        contact_policy='position to hard-table Z=0 over sponge; no force feedback; one press then lift'))
    plan_started=time.monotonic()
    initial={k:np.asarray(v['joints']) for k,v in runtime.coordinator.snapshots().items()}
    runtime.arrival({k:np.asarray(v) for k,v in observation.meta['joints_deg'].items()})
    # Entire route and return are planned before descending.
    operations=runtime.plan(source,initial,preserve_command_boundaries=True)
    report['press_planning_s']=time.monotonic()-plan_started
    if runtime.cancel.is_set():raise KeyboardInterrupt('cancelled before post-drop press execution')
    recorder.check();recorder.stage('post_drop_press')
    report.update(physical_actions=2,press_status='DISPATCHED')
    save(folder/'result.json',report)
    execution_started=time.monotonic()
    try:result=runtime._execute_operations(operations,action)
    finally:report['robot_execution_s']+=time.monotonic()-execution_started
    save(action/'result.json',result)
    if result['status']!='EXECUTED':raise DualArmError('post-drop press execution failed')
    return dict(status='EXECUTED',center_px=pixel,surface_z_mm=float(surface[2]),table_z_mm=0.0,
                action_folder='post_drop_press',source_image='after_drop/rgb.png')


def initialize(runtime, folder, headers, timeout_s, *, center_radius_mm=80., press_after_drop=False):
    """Existing reset geometry and sampling; direct region request instead of Agent."""
    folder.mkdir()
    started = time.monotonic()
    report = dict(status='PREPARING', physical_actions=0, robot_execution_s=0., model_calls=0,
                  seed=secrets.randbits(63), max_retries=0, acquisition='UNKNOWN_PENDING_HUMAN_REVIEW',
                  release_policy=INITIALIZATION_RELEASE_POLICY,
                  center_radius_mm=float(center_radius_mm),press_after_drop=bool(press_after_drop))
    recorder = None
    save(folder/'result.json', report)
    try:
        before = runtime.observe(folder/'before')
        before.validate_for(runtime.config, live=True, check_age=False)
        with Image.open(before.image) as image:
            size = list(image.size)
        body = planner.payload(before.image, before.image, size)
        body['instructions'] = reset.PROMPT+' Return only schema-valid JSON.'
        body['input'][0]['content'] = [dict(type='input_text', text=f'CURRENT RGB dimensions: {size} pixels.'),
            dict(type='input_image', image_url=planner.image_data(before.image), detail='high')]
        body['text']['format'].update(name='initialization_cloth_region', schema=reset.REGION)
        region, metrics = direct_once(folder/'region_request', body, headers, timeout_s)
        report['model_calls'] = 1
        report['planning_wall_s'] = metrics['wall_latency_s']
        save(folder/'region.json', region)
        reset.validate_region(region, size)
        seeds = np.random.SeedSequence(report['seed']).spawn(1)[0].generate_state(2, dtype=np.uint64)
        point = reset.sample_grasp(before, region, runtime.config.limits['max_surface_spread_mm'], center_radius_mm, int(seeds[0]))
        release = sample_initialization_release(int(seeds[1]))
        save(folder/'grasp.json', point)
        save(folder/'release.json', dict(seed=int(seeds[1]), policy=INITIALIZATION_RELEASE_POLICY, **release))
        reset.preview(before.image, region, point, folder/'grasp_preview.png')
        arm = runtime.config.arms['left']
        rpy = matrix_pose(arm.world_from_base @ pose_matrix(runtime.models['left'].forward(arm.home_joints)))[3:]
        source = reset.reset_program(runtime, point['world_xyz_mm'], rpy, grasp_offset=-15., **release)
        action_folder = folder/'lift_and_drop'; action_folder.mkdir()
        (action_folder/'motion.py').write_text(source)
        initial = {k: np.asarray(v['joints']) for k, v in runtime.coordinator.snapshots().items()}
        runtime.arrival({k: np.asarray(v) for k, v in before.meta['joints_deg'].items()})
        operations = runtime.plan(source, initial, preserve_command_boundaries=True)
        recorder = TrajectoryRecorder(runtime, folder)
        recorder.start(); recorder.stage('initialization_lift_and_drop'); recorder.check()
        runtime.arrival(initial)
        report.update(physical_actions=1, status='DISPATCHED')
        save(folder/'result.json', report)
        robot_start = time.monotonic()
        try:
            result = runtime._execute_operations(operations, action_folder)
            save(action_folder/'result.json', result)
            if result['status'] != 'EXECUTED':
                raise DualArmError('initialization execution failed')
            if runtime.cancel.wait(2.):
                raise KeyboardInterrupt('cancelled during initialization settling')
            recorder.check(); recorder.stage('return_perception')
            runtime.move_to_perception(folder/'return_perception')
            recorder.check()
        finally:
            report['robot_execution_s'] = time.monotonic()-robot_start
        if press_after_drop:
            recorder.check();recorder.stage('post_drop_press_observation')
            dropped=runtime.capture_current(folder/'after_drop')
            press_result=press_dropped_cloth(runtime,folder,dropped,headers,timeout_s,rpy,recorder,report)
            report.update(press=press_result,press_status='EXECUTED')
            save(folder/'result.json',report)
            recorder.check();recorder.stage('return_perception_after_press')
            return_started=time.monotonic()
            try:runtime.move_to_perception(folder/'return_perception_after_press')
            finally:report['robot_execution_s']+=time.monotonic()-return_started
            recorder.check()
        runtime.capture_current(folder/'after')
        report.update(status='EXECUTED', outcome='PENDING_HUMAN_REVIEW', comparison_model_calls=0)
    except BaseException as exc:
        report.update(status='STOPPED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if recorder:
            recorder.close()
        report['elapsed_s'] = time.monotonic()-started
        save(folder/'result.json', report)
    if recorder:
        recorder.check()
    return report


def wait_initialization_review(runtime, root, timeout_s=300.):
    """Keep arm stationary until the saved initialization images are reviewed."""
    folder = root/'initialization'
    request = dict(status='WAITING', before_sha256=sha(folder/'before/rgb.png'),
                   after_sha256=sha(folder/'after/rgb.png'), timeout_s=timeout_s)
    save(folder/'review_request.json', request)
    print('INITIALIZATION REVIEW WAITING; no unfolding model request or motion', flush=True)
    started = time.monotonic()
    path = folder/'review_decision.json'
    while not path.exists():
        if runtime.cancel.wait(.25):
            raise KeyboardInterrupt('cancelled during initialization review')
        if time.monotonic()-started >= timeout_s:
            raise DualArmError('initialization review timed out; no unfolding started')
    decision = read(path)
    if any(decision.get(k) != request[k] for k in ('before_sha256', 'after_sha256')):
        raise DualArmError('initialization review image hashes do not match')
    if decision.get('decision') != 'ACCEPT':
        raise DualArmError('initialization not accepted: '+str(decision.get('reason', '')))
    save(folder/'review_result.json', dict(**decision, wait_s=time.monotonic()-started))


def render(root, summary):
    rows, panels = [], []
    if (root/'initialization/before/rgb.png').exists():
        panel = Image.new('RGB', (1280, 285), 'white'); draw = ImageDraw.Draw(panel)
        for col, (name, path) in enumerate((('GOAL', root/'goal/target.png'),
                ('INITIALIZATION BEFORE', root/'initialization/before/rgb.png'),
                ('INITIALIZATION AFTER', root/'initialization/after/rgb.png'))):
            draw.text((col*425+8, 10), name, fill='black')
            if path.exists():
                with Image.open(path) as im:
                    im = im.convert('RGB'); im.thumbnail((420, 236)); panel.paste(im, (col*425, 40))
        panels.append(panel)
    for item in summary['rounds']:
        folder = root/item['round']
        plan = read(folder/'plan.json') if (folder/'plan.json').exists() else None
        t = item['timing']
        rows.append(dict(round=item['round'], decision=plan['decision'] if plan else 'NO_VALID_PLAN',
            execution=item['execution']['status'], planning_s=t['planning_s'], robot_execution_s=t['robot_execution_s'],
            planner=item.get('planner', 'FastPlanner'), codex_s=t.get('deliberative_planner_wall_s', 0.),
            view_observation_s=t.get('view_observation_s', 0.), view_return_s=t.get('view_return_s', 0.),
            planner_return_overlap_s=t.get('planner_return_overlap_s', 0.),
            model_wait_after_return_s=t.get('model_wait_after_return_s', 0.)))
        panel = Image.new('RGB', (1280, 285), 'white'); draw = ImageDraw.Draw(panel)
        views=[('GOAL', root/'goal/target.png'),(item['round']+' PERCEPTION', folder/'before/rgb.png')]
        if not (summary.get('human_preference_enabled') or summary.get('automatic_top5_enabled') or summary.get('visible_sleeve_hem_enabled')):
            views.append(('RIGHT VIEW',folder/'right_view/rgb.png'))
        views.append(('AFTER',folder/'after/rgb.png'))
        column_width=1280//len(views)
        for col, (name, path) in enumerate(views):
            draw.text((col*column_width+8, 10), name, fill='black')
            if path.exists():
                with Image.open(path) as im:
                    im = im.convert('RGB'); im.thumbnail((column_width-5, 236)); panel.paste(im, (col*column_width, 40))
        panels.append(panel)
    if rows:
        with (root/'latency.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if panels:
        sheet = Image.new('RGB', (1280, 285*len(panels)), 'white')
        for i, panel in enumerate(panels): sheet.paste(panel, (0, 285*i))
        sheet.save(root/'episode_contact_sheet.png')
    s = summary['planning_latency_s']
    lines = ['# one_shot_goal_full_episode_v1', '',
        f"终止：**{summary['termination_reason']}**；{summary['termination_detail']}", '',
        f"共 {summary['total_rounds']} rounds；展开 manipulation {summary['physical_actions']} 次（包含部分执行的 dispatch）；"
        f"初始化 {summary['initialization_physical_actions']} 次；安全拒绝 {summary['safety_rejections']} 次。", '',
        f"Planning latency：median {s['median']} s / mean {s['mean']} s / p95 {s['p95']} s。",
        f"展开总 planning {summary['total_planning_s']:.2f} s；展开 robot execution（含回观测位）"
        f"{summary['total_robot_execution_s']:.2f} s；整场 {summary['episode_wall_s']:.2f} s。", '',
        ('衣袖/下摆单动作模式：当前图+目标图+上一动作前图及真实动作；一次规划请求，不生成TopN、不评分、不使用学习参数；记录部位依据、局部效果及停止检查。'
         if summary.get('visible_sleeve_hem_enabled') else '自动候选模式：perception当前图+目标图，独立Ps/Pt评分；保存各自Top5，执行GPT明确选择的完整动作。无right view、alpha、Rθ、人工等待或学习；原有shadow gate及Host检查保留。'
         if summary.get('automatic_top5_enabled') else
         '人工候选模式每轮用perception当前图与目标图独立请求Ps与Pt；启用排名修正时按Ps+固定alpha*Pt+Rθ显示Top10，否则按Ps+alpha*Pt。人工选择后才检查并执行。评分期间机械臂保持观测位，人工等待单列。'
         'ACTION仍经过独立shadow gate和原有Host检查；无自动选择、重试或恢复替换。'
         if summary.get('human_preference_enabled') else
        'FastPlanner 每轮 perception RGB + right view RGB + 固定 goal RGB；优先可见边角，Pa/Pb 只用 perception 坐标。'
        '一次 direct gpt-6-astra/high 请求与慢速回程并行，回到 perception 核验后才使用结果；'
        'tools=[] / tool_choice=none / retries=0。ACTION 额外调用独立 shadow gate，仅记录标签。'
        'Fast 请求不做 repair 或第二次 grounding；memory、Graph、ANN 保持禁用。'
        f"本次 Codex 错误恢复启用：{summary.get('codex_recovery_enabled', False)}；"
        f"调用 {summary.get('codex_recovery_calls', 0)} 次，单列耗时 {summary.get('total_deliberative_planning_s', 0.):.2f}s。"
        '若启用，未执行动作的允许类错误可在新轮次交给 Codex，gate 仍无路由权。'), '',
        '初始化采用原 reset_program / sample_grasp / sample_release，单次直接请求定位可见衣身区域；'
        '不调用 Agent、不做模型 after 比较、不重复 reset。初始化耗时和调用单列在 summary。', '',
        'Host 沿用 CollectionRuntime 当前检查与 motion primitive。其原有 path validation 和 left 附加碰撞包络设置'
        '未启用，不将这些未启用检查报告为通过。Pa 保留实测深度检查；'
        'Pb 仅在缺深度且通过裸桌支撑面校验时使用 TABLE_PLANE_ESTIMATE，详见 target_depth.json。'
        '距离、workspace、IK 和执行检查继续使用最终 XYZ。', '',
        '**DONE 只是模型停止决定，不代表实验成功。人工 FOLD_READY 判定未填；请在 review.html 评审并导出标签。**', '',
        '|Round|Planner|Decision|Execution|Fast s|Codex s|Robot s|', '|---|---|---|---|---:|---:|---:|']
    lines += [f"|{r['round']}|{r['planner']}|{r['decision']}|{r['execution']}|{r['planning_s']:.2f}|{r['codex_s']:.2f}|{r['robot_execution_s']:.2f}|" for r in rows]
    if 'continuation' in summary:
        lines += ['', '## 用户授权续跑', '',
                  '本目录记录安全停止后的新段，未重新初始化；从当前衣物重新采集 RGB-D 并发起新一轮请求。'
                  '原拒绝动作没有直接重放。目标、prompt、每轮单请求限制和 Host 检查均保持不变。'
                  '原有已执行步骤仅用于 Host 重复动作保护，未提供给模型。',
                  f"累计 {summary['cumulative_rounds']} 轮、{summary['cumulative_physical_actions']} 次展开动作、"
                  f"{summary['cumulative_safety_rejections']} 次安全拒绝；"
                  f"累计 planning {summary['cumulative_planning_s']:.2f} 秒，"
                  f"robot execution {summary['cumulative_robot_execution_s']:.2f} 秒（不含初始化）。",
                  '[续跑来源与授权记录](continuation.json)']
    if summary['physical_actions'] == 0:
        lines += ['', '**本次没有执行展开动作，因此没有获得多步闭环展开能力的证据。**'
                  '初始化指令执行完成不证明抓住了衣服或产生了新的 crumpled state；'
                  '请结合初始化 before/after 和 execution_images 人工检查。']
    if s['n'] == 1:
        lines += ['', '本次只有 1 个规划样本，median / mean / p95 均为该单次耗时，不能用于判断稳定延迟。']
    if panels: lines += ['', '![全过程](episode_contact_sheet.png)']
    lines += ['', '[episode summary](episode_summary.json) · [人工评审](human_review.json)', '',
              '接口依据：[OpenAI Docs structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)。', '']
    (root/'report.md').write_text('\n'.join(lines))


def run(args):
    from cloth_agent import unfold_experiment_policy as experiment_policy
    no_improvement_streak = 0
    patience = getattr(args, 'no_improvement_patience', 0)
    max_actions = getattr(args, 'max_actions', 0)
    uncertain_counts = getattr(args, 'uncertain_is_no_improvement', False)
    if (patience or max_actions) and not getattr(args,'visible_sleeve_hem',False):
        raise ValueError('outcome-based stop policy requires contextual sleeve/hem mode')
    human_preference = getattr(args, 'human_preference', False)
    automatic_top5 = getattr(args, 'automatic_top5', False)
    stateful_context = getattr(args, 'stateful_context', False)
    restricted_mode = getattr(args, 'visible_sleeve_hem', False)
    if restricted_mode:
        if human_preference or automatic_top5 or getattr(args,'ranking_correction',False) or getattr(args,'codex_recovery',False) or getattr(args,'continue_from',None):
            raise ValueError('sleeve/hem experiment requires a new episode without candidate scoring, learning or recovery')
        from cloth_agent import restricted_context_one_shot as restricted
        stateful_context = True
    candidate_mode = human_preference or automatic_top5 or restricted_mode
    if stateful_context and not (automatic_top5 or restricted_mode):
        raise ValueError('previous-action context requires automatic Top5 mode')
    if automatic_top5 and (human_preference or getattr(args,'ranking_correction',False) or getattr(args,'codex_recovery',False)):
        raise ValueError('automatic Top5 excludes human preference, learned ranking and recovery')
    if automatic_top5:
        from cloth_agent import automatic_top5 as auto
    ranking_correction = getattr(args, 'ranking_correction', False)
    residual_state_path = None
    if ranking_correction and not human_preference:
        raise ValueError('ranking correction requires human preference mode')
    preference = None
    if candidate_mode:
        from cloth_agent import preference_one_shot as preference
    if human_preference:
        if getattr(args, 'codex_recovery', False) or getattr(args, 'continue_from', None):
            raise ValueError('human preference requires a new episode without automatic Codex recovery')
        alpha_state_path = getattr(args, 'alpha_state', preference.pref.DEFAULT_STATE)
        alpha_initial = preference.pref.load_state(alpha_state_path)
        if ranking_correction:
            from cloth_agent import ranking_residual, residual_features
            residual_state_path = getattr(args, 'residual_state', ranking_residual.DEFAULT_STATE)
            residual_initial = ranking_residual.load_state(residual_state_path, alpha_initial['alpha'])
    recovery_enabled = getattr(args, 'codex_recovery', False)
    recovery_limit = getattr(args, 'max_codex_replans', 4)
    incident_limit = getattr(args, 'max_replans_per_incident', 2)
    recovery_timeout = getattr(args, 'codex_timeout_s', 600.)
    source = getattr(args, 'continue_from', None)
    previous = None
    if source is not None:
        source = source.resolve()
        previous = read(source/'episode_summary.json')
        protocol = read(source/'protocol.json')
        if previous['experiment'] != ROOT.name or previous['termination_reason'] != 'SAFETY_STOP':
            raise ValueError('continuation requires this experiment stopped at SAFETY_STOP')
        expected_prompt = ((preference.AUTO_PS_CONTEXT_PROMPT+'\n\n'+auto.PROMPT+preference.PT_CONTEXT_PROMPT)
                           if stateful_context else (preference.AUTO_PS_PROMPT+'\n\n'+auto.PROMPT)
                           if automatic_top5 else planner.ROTATING_MULTIVIEW_PROMPT)
        if (protocol['prompt'] != expected_prompt or protocol['schema'] != planner.ROTATION_SCHEMA
                or protocol['model'] != planner.MODEL or protocol['effort'] != planner.EFFORT):
            raise ValueError('continuation must preserve the original planner')
        if bool(protocol.get('automatic_top5',{}).get('enabled')) != automatic_top5:
            raise ValueError('continuation must preserve automatic Top5 mode')
        if bool(protocol.get('stateful_context_enabled',False)) != stateful_context:
            raise ValueError('continuation must preserve previous-action context mode')
        prior_rounds = previous.get('cumulative_rounds', previous['total_rounds'])
        if args.max_rounds > 12-prior_rounds:
            raise ValueError('continuation exceeds remaining 12-round episode budget')
        args.goal_image = source/'goal/target.png'
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    (root/'goal').mkdir(); shutil.copyfile(args.goal_image, root/'goal/target.png')
    prompt_text = restricted.PROMPT if restricted_mode else ((preference.AUTO_PS_CONTEXT_PROMPT+'\n\n'+auto.PROMPT+preference.PT_CONTEXT_PROMPT)
                   if stateful_context else
                   ((preference.AUTO_PS_PROMPT if automatic_top5 else preference.PS_PROMPT)+'\n\n'+
                    (auto.PROMPT if automatic_top5 else preference.PT_PROMPT)
                    if candidate_mode else planner.ROTATING_MULTIVIEW_PROMPT))
    (root/'prompt.txt').write_text(prompt_text)
    if not candidate_mode:
        shutil.copyfile(one_shot_views.RIGHT_VIEW, root/'right_view_pose.json')
    (root/'gate_prompt.txt').write_text(GATE_PROMPT)
    protected = {p: sha(p) for p in PROTECTED}
    if candidate_mode:
        protected.update({p:sha(p) for p in ('cloth_agent/action_preference.py','cloth_agent/preference_one_shot.py')})
    if restricted_mode:
        protected.update({p:sha(p) for p in ('cloth_agent/restricted_context_one_shot.py','cloth_agent/geometry_report.py','cloth_agent/unfold_experiment_policy.py',
            'cloth_agent/one_shot_depth_recovery.py','cloth_agent/experiments/arm6_depth_recovery.py')})
    if automatic_top5:protected['cloth_agent/automatic_top5.py']=sha('cloth_agent/automatic_top5.py')
    if ranking_correction:
        protected.update({p:sha(p) for p in ('cloth_agent/ranking_residual.py','cloth_agent/residual_features.py',
            'cloth_agent/residual_review.html','cloth_agent/dino_correspondence.py')})
    target_depth_sources = {str(p): sha(p) for p in (
        Path(table_target_depth.__file__), table_target_depth.APPEARANCE_CONFIG,
        table_target_depth.INTRINSICS_FILE)}
    recovery_sources = {p: sha(p) for p in (
        'cloth_agent/codex_recovery.py', 'cloth_agent/unfolding_planners.py',
        'cloth_agent/experiments/trajectory_memory_full_episode_v1.py',
        'cloth_agent/experiments/goal_only_full_episode_v1.py',
        'cloth_agent/experiments/trajectory_memory_agent_probe.py')}
    frozen_paths=[root/'goal/target.png',root/'prompt.txt',root/'gate_prompt.txt']
    if not candidate_mode:frozen_paths.append(root/'right_view_pose.json')
    frozen = {str(p.relative_to(root)): sha(p) for p in frozen_paths}
    save(root/'protocol.json', dict(experiment=ROOT.name, model=planner.MODEL, effort=planner.EFFORT,
        schema=restricted.SCHEMA if restricted_mode else planner.ROTATION_SCHEMA, prompt=prompt_text, max_rounds=args.max_rounds,
        observation_views=['perception'] if candidate_mode else ['perception','right_view'],
        input_image_count=2 if candidate_mode else 3,
        stateful_context_enabled=stateful_context,
        stateful_context_policy=('from the second action: current RGB + goal RGB + previous pre-action RGB; '
                                 'previous executed Pa/Pb as text; assess local effect before choosing the next action'
                                 if stateful_context else None),
        target_depth_sampled=False,
        grasp_height_spread_check_enabled=False,
        missing_grasp_depth_recovery=dict(enabled=restricted_mode,offset_mm=20,max_views=2,
            fresh_rgb_replanning=True,additional_garment_iterations=False),
        gripper_orientation='image opening axis; adjust before grasp only; hold fixed after closing',
        view_joint_speed_cap_deg_s=one_shot_views.VIEW_SPEED_DEG_S,
        view_joint_accel_cap_deg_s2=one_shot_views.VIEW_ACCEL_DEG_S2,
        model_overlaps_view_return=not candidate_mode,
        global_gripper_boundary_prompt=planner.GRIPPER_BOUNDARY_PROMPT,
        tools=[], tool_choice='none', retries=0, model_calls_per_round=2 if (human_preference or automatic_top5) else 1,
        no_improvement_patience=patience,max_actions=max_actions,uncertain_is_no_improvement=uncertain_counts,
        visible_sleeve_hem_enabled=restricted_mode, topn_enabled=human_preference or automatic_top5,
        request_constraints_scope='FastPlanner only; gate and optional Codex recovery logged separately',
        episode_history=False, graph=False, memory=False, frozen=frozen, protected_sources=protected,
        initialization=('none; user-authorized continuation from current garment state' if previous else
                        'single existing random reset primitive; direct region request; no Agent/comparison/retry'),
        initialization_center_radius_mm=float(getattr(args,'initialize_center_radius_mm',80.)),
        post_drop_press_enabled=bool(getattr(args,'press_after_drop',False)),
        post_drop_press_policy=('fresh RGB-D after free drop; central visible cloth; hard-table world Z=0; lift and reobserve'
                                if getattr(args,'press_after_drop',False) else None),
        initialization_review_required=getattr(args, 'review_initialization', False),
        initialization_release_policy=INITIALIZATION_RELEASE_POLICY if not previous else None,
        planner_framework='FAST_WITH_HOST_RECOVERY_V1' if recovery_enabled else 'FAST_WITH_SHADOW_GATE_V1',
        codex_recovery=dict(enabled=recovery_enabled, max_calls_per_episode=recovery_limit,
            max_calls_per_incident=incident_limit, timeout_s=recovery_timeout,
            trigger='allowlisted pre-dispatch errors only', fresh_observation=True,
            retries_of_fast_request=0, hardware_fault_recovery=False,
            source_sha256=recovery_sources, agent_tool_loop=True),
        target_depth_policy=table_target_depth.PROJECT_POLICY,
        target_depth_sources=target_depth_sources,
        gate=dict(mode='SHADOW', model=planner.MODEL, effort=planner.EFFORT,
                  prompt=GATE_PROMPT, schema=GATE_SCHEMA, retries=0, action_only=True,
                  timeout_s=getattr(args, 'gate_timeout_s', 120.),
                  affects_execution=False, on_error='UNCERTAIN with ERROR status; no retry'),
        deliberative_enabled=recovery_enabled, memory_planner_enabled=False,
        human_review_required=False if restricted_mode else True))
    if human_preference:
        protocol=read(root/'protocol.json')
        protocol.update(planner_framework='PS_PT_HUMAN_PREFERENCE_V1',
            human_preference=dict(enabled=True,formula='P=Ps+alpha*Pt',
                alpha_state=str(Path(alpha_state_path).resolve()),initial_state=alpha_initial,
                point_prompt=preference.PS_PROMPT,point_schema=preference.PS_SCHEMA,
                trajectory_prompt=preference.PT_PROMPT,candidate_points=10,directions=['R','D','L','U'],
                length_px=120,display_count=10,automatic_choice=False,
                update_signal='human preference among shown actions; not execution outcome',
                host_scene_recheck=True,review_port=getattr(args,'preference_port',8767)))
        save(root/'protocol.json',protocol)
        if ranking_correction:
            protocol['planner_framework']='PS_PT_RESIDUAL_RANKING_V1'
            protocol['human_preference'].update(formula='S=Ps+fixed_alpha*Pt+R_theta(x)',
                alpha_fixed=True,residual_state=str(Path(residual_state_path).resolve()),
                residual_initial_revision=residual_initial['revision'],
                feature_version=residual_features.VERSION,learning=ranking_residual.LEARNING,
                update_signal='ordered human groups of displayed Top10; within-group ties ignored; execution choice independent',
                evaluation='all historical checkpoints on this state, measured before training this feedback')
            save(root/'protocol.json',protocol)
    if automatic_top5:
        protocol=read(root/'protocol.json')
        protocol.update(planner_framework='GPT_AUTOMATIC_TOP5_V1',human_review_required=False,
            automatic_top5=dict(enabled=True,alpha_used=False,residual_used=False,learning=False,
                previous_action_context=stateful_context,first_round_images=2,
                subsequent_round_images=3 if stateful_context else 2,
                stop_policy='DONE only when torso has no separate folded-over panel, both sleeves are mostly out, hem does not split into hanging lobes, and no major cloth panel remains hidden; goal-like long wrinkles allowed',
                done_allowed_after_point_scoring=True,done_allowed_after_trajectory_scoring=True,
                selection='GPT explicit selected_id, not argmax Pt or learned sum',
                point_prompt=preference.AUTO_PS_CONTEXT_PROMPT if stateful_context else preference.AUTO_PS_PROMPT,
                trajectory_prompt=auto.PROMPT+preference.PT_CONTEXT_PROMPT if stateful_context else auto.PROMPT,
                points=10,directions=['R','D','L','U'],length_px=120,
                saved_rankings=['Ps Top5','Pt Top5'],raw_request_audit_retained=True,host_scene_recheck=True))
        save(root/'protocol.json',protocol)
    source_dir = root/'source_snapshot'; source_dir.mkdir()
    for path in (__file__, planner.__file__, 'cloth_agent/experiments/trajectory_memory_full_episode_v1.py'):
        shutil.copyfile(path, source_dir/Path(path).name)
    for path in ('cloth_agent/pre_action_gate.py', 'cloth_agent/unfolding_planners.py',
                 'cloth_agent/planner_shadow_review.py', 'cloth_agent/table_target_depth.py',
                 'cloth_agent/codex_recovery.py', 'cloth_agent/one_shot_views.py',
                 'cloth_agent/one_shot_orientation.py', 'cloth_agent/grasp_angle.py'):
        shutil.copyfile(path, source_dir/Path(path).name)
    for path in (table_target_depth.APPEARANCE_CONFIG, table_target_depth.INTRINSICS_FILE):
        shutil.copyfile(path, source_dir/path.name)
    for path in recovery_sources:
        shutil.copyfile(path, source_dir/Path(path).name)
    if candidate_mode:
        for path in (preference.__file__,preference.pref.__file__):
            shutil.copyfile(path,source_dir/Path(path).name)
    if restricted_mode:
        shutil.copyfile(restricted.__file__,source_dir/Path(restricted.__file__).name)
        shutil.copyfile(restricted.geometry_report.__file__,source_dir/'geometry_report.py')
        shutil.copyfile(experiment_policy.__file__,source_dir/Path(experiment_policy.__file__).name)
        for path in ('cloth_agent/one_shot_depth_recovery.py','cloth_agent/experiments/arm6_depth_recovery.py'):
            shutil.copyfile(path,source_dir/Path(path).name)
    if automatic_top5:shutil.copyfile(auto.__file__,source_dir/Path(auto.__file__).name)
    if ranking_correction:
        for path in ('cloth_agent/ranking_residual.py','cloth_agent/residual_features.py',
                     'cloth_agent/residual_review.html','cloth_agent/dino_correspondence.py'):
            shutil.copyfile(path,source_dir/Path(path).name)
    summary = dict(experiment=ROOT.name, started_at=planner.utc(), termination_reason=None, termination_detail='',
        total_rounds=0, physical_actions=0, safety_rejections=0, initialization_physical_actions=0,
        total_planning_s=0., total_robot_execution_s=0., rounds=[], human_fold_ready=None,
        codex_recovery_enabled=recovery_enabled, codex_recovery_calls=0,
        total_deliberative_planning_s=0.)
    summary['human_preference_enabled']=human_preference
    summary['automatic_top5_enabled']=automatic_top5
    summary['stateful_context_enabled']=stateful_context
    summary['visible_sleeve_hem_enabled']=restricted_mode
    summary['ranking_correction_enabled']=ranking_correction
    if ranking_correction: ranking_residual.write_progress(residual_initial, root)
    save(root/'human_review.json', dict(choice=None, allowed_choices=['FOLD_READY', 'NOT_FOLD_READY', 'UNCERTAIN'],
        reviewer=None, notes='', acquisition_by_round={}))
    save(root/'episode_summary.json', summary)
    runtime = None; folder = None; history = []; started = time.monotonic()
    pending_recovery = None
    fast_planner = FastPlanner(plan_round)
    if previous:
        continuation = dict(source_episode=str(source), source_summary_sha256=sha(source/'episode_summary.json'),
            prior_total_rounds=prior_rounds,
            prior_physical_actions=previous.get('cumulative_physical_actions', previous['physical_actions']),
            prior_safety_rejections=previous.get('cumulative_safety_rejections', previous['safety_rejections']),
            prior_planning_s=previous.get('cumulative_planning_s', previous['total_planning_s']),
            prior_robot_execution_s=previous.get('cumulative_robot_execution_s', previous['total_robot_execution_s']),
            authorization='User requested continuation from the current garment state', reinitialization=False,
            model_history_supplied=False, prompt_changed=False, safety_thresholds_changed=False,
            source_hashes={})
        past_directories = {Path(p).parent for p in previous.get('continuation', {}).get('source_hashes', {})
                            if Path(p).name == 'plan.json'}
        past_directories.update(source/item['round'] for item in previous['rounds']
                                if item['execution']['status'] == 'EXECUTED')
        for past in sorted(past_directories):
            past_execution=read(past/'execution.json')
            if past_execution['status'] != 'EXECUTED':
                raise ValueError('Host history contains an unexecuted proposal')
            if not (past/'after/rgb.png').is_file():
                raise ValueError('executed prefix has no after observation')
            history.append(dict(plan=read(past/'plan.json'), before=past/'before/rgb.png',
                                after=past/'after/rgb.png',source_round=past.name,
                                completed_commands=past_execution.get('completed_commands'),
                                acquisition='UNKNOWN_PENDING_HUMAN_REVIEW'))
            for name in ('plan.json', 'before/rgb.png', 'after/rgb.png', 'execution.json'):
                continuation['source_hashes'][str(past/name)] = sha(past/name)
        save(root/'continuation.json', continuation)
        summary['continuation'] = continuation
    old_signal = signal.getsignal(signal.SIGTERM)
    def interrupted(*_): raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        headers = planner.auth_headers()  # Local read only; before connecting hardware.
        runtime = EpisodeCollectionRuntime(args.config)
        if not candidate_mode:
            one_shot_views.view_target(runtime, read(root/'right_view_pose.json'))
        if previous:
            print('CONTINUE from current state; no reset; fresh RGB-D; no model history', flush=True)
            startup = time.monotonic()
            runtime.observe(root/'startup_observation')
            summary['startup_observation_s'] = time.monotonic()-startup
        else:
            print('INITIALIZE one existing random pick-up/drop; direct region request; no retries', flush=True)
            init_options={}
            if getattr(args,'initialize_center_radius_mm',80.)!=80.:
                init_options['center_radius_mm']=args.initialize_center_radius_mm
            if getattr(args,'press_after_drop',False):init_options['press_after_drop']=True
            initialize(runtime, root/'initialization', headers, args.timeout_s,**init_options)
            if getattr(args, 'review_initialization', False):
                wait_initialization_review(runtime, root)
        for index in range(args.max_rounds):
            folder = root/f'round_{index:03d}'; folder.mkdir()
            timing = dict(planning_s=0., robot_execution_s=0., host_planning_s=0., observation_s=0.,
                          fast_planner_wall_s=0., gate_wall_s=0., host_validation_s=0., total_round_s=0.,
                          deliberative_planner_wall_s=0.)
            recovery_request, pending_recovery = pending_recovery, None
            plan = None
            item = dict(round=folder.name, execution=dict(status='NOT_EXECUTED', physical_execution=False), timing=timing)
            item['planner'] = 'CodexRecovery' if recovery_request else 'RestrictedContextOneShot' if restricted_mode else 'GPTAutomaticTop5' if automatic_top5 else 'HumanPreference' if human_preference else 'FastPlanner'
            summary['rounds'].append(item); summary['total_rounds'] += 1
            save(folder/'execution.json', item['execution']); save(root/'episode_summary.json', summary)
            router = ShadowPlannerRouter(folder, recovery_enabled=recovery_enabled)
            host_validation = dict(status='NOT_RUN', policy_failure_inferred=False)
            save(folder/'host_validation.json', host_validation)
            save(folder/'gate_status.json', dict(status='NOT_RUN', mode='SHADOW', affects_execution=False))
            began = time.monotonic(); phase = 'OBSERVATION'
            try:
                if recovery_request:
                    if not all(sha(p) == digest for p, digest in recovery_sources.items()):
                        raise DualArmError('frozen recovery source changed')
                    codex_recovery.check_stationary(runtime, recovery_request['expected_joints'],
                                                   folder/'recovery_preflight.json')
                capture = time.monotonic(); before = runtime.capture_current(folder/'before')
                timing['observation_s'] += time.monotonic()-capture
                if not all(sha(root/p) == value for p, value in frozen.items()):
                    raise DualArmError('frozen goal/prompt changed')
                if recovery_request:
                    one_shot_views.capture_and_return(runtime, root, folder, before, timing)
                    phase = 'DELIBERATIVE_MODEL'
                    summary['codex_recovery_calls'] += 1
                    save(root/'episode_summary.json', summary)
                    save(folder/'fast_status.json', dict(valid=False, model_called=False, reason='Codex recovery round'))
                    print(f"ROUND {index:03d} CODEX_RECOVERY {recovery_request['attempt']}/{incident_limit}", flush=True)
                    plan, metrics = codex_recovery.CodexRecoveryPlanner().plan(
                        root, folder, before, recovery_request, recovery_timeout, allow_grasp_rotation=True)
                    timing['deliberative_planner_wall_s'] = metrics['wall_latency_s']
                    codex_recovery.check_stationary(runtime, recovery_request['expected_joints'],
                                                   folder/'recovery_post_model_preflight.json')
                    router.accept_deliberative(plan)
                elif candidate_mode:
                    phase='CONTEXT_CHECK';print(f'ROUND {index:03d} '+('SLEEVE/HEM ONE_SHOT' if restricted_mode else 'Ps/Pt CANDIDATES'),flush=True)
                    score_options={'automatic':True} if automatic_top5 else {}
                    if stateful_context and history:
                        latest=history[-1]
                        same,gap=unchanged(latest['after'],before.image)
                        save(folder/'previous_context_continuity.json',dict(
                            passed=same,previous_after_sha256=sha(latest['after']),
                            current_sha256=sha(before.image),rgb_q95=gap,threshold=8,
                            source_round=latest['source_round']))
                        if not same:
                            raise DualArmError('garment changed since previous action outcome; previous-action context invalid')
                        score_options['previous_context']=dict(source_round=latest['source_round'],
                            before_image=latest['before'],after_image=latest['after'],plan=latest['plan'],
                            completed_commands=latest.get('completed_commands'))
                        item['previous_action_context_round']=latest['source_round']
                    phase='MODEL'
                    if restricted_mode:
                        plan,metrics=restricted.plan_round(root,folder,before,headers,args.timeout_s,direct_once,
                            previous_context=score_options.get('previous_context'))
                        timing['planning_s']=metrics['wall_latency_s']
                        timing['fast_planner_wall_s']=metrics['wall_latency_s']
                    else:
                        bundle,metrics=preference.score_round(
                            root,folder,before,headers,args.timeout_s,runtime.config,direct_once,**score_options)
                        timing['planning_s']=metrics['wall_latency_s']
                        timing['fast_planner_wall_s']=metrics['wall_latency_s']
                        # Score and review the current perception observation without a view tour.
                        if automatic_top5:
                            phase='AUTOMATIC_CHOICE'
                            plan=auto.choose(folder,before,bundle,runtime.cancel)
                        else:
                            phase='HUMAN_CHOICE'
                            plan=preference.choose_plan(root,folder,before,bundle,runtime,timing,alpha_state_path,
                                getattr(args,'preference_port',8767),getattr(args,'preference_timeout_s',1800.),
                                residual_state_path=residual_state_path)
                    for name in ('plan.json','fast_plan.json'):save(folder/name,plan)
                    planner.overlay(before.image,plan,folder/'proposed_action_overlay.png',folder.name+(' GPT CHOICE' if (automatic_top5 or restricted_mode) else ' HUMAN CHOICE'))
                    shutil.copyfile(folder/'proposed_action_overlay.png',folder/'fast_action_overlay.png')
                    save(folder/'fast_status.json',dict(valid=True,planner=item['planner'],model=planner.MODEL,
                        effort=planner.EFFORT,retries=0,automatic_slow_routing=False))
                    router.accept_fast(plan)
                    if automatic_top5:auto.render_round(folder)
                else:
                    phase = 'MODEL'; print(f'ROUND {index:03d} ONE_SHOT', flush=True)
                    plan, metrics = one_shot_views.plan_with_right_view(
                        runtime, root, folder, before,
                        lambda: fast_planner.plan(root, folder, before, headers, args.timeout_s), timing)
                    timing['planning_s'] = metrics['wall_latency_s']
                    timing['fast_planner_wall_s'] = metrics['wall_latency_s']
                    router.accept_fast(plan)
                print(f"ROUND {index:03d} {plan['decision']} {plan['grasp_px']} -> {plan['target_px']} "
                      f"grasp_angle_image_deg={plan.get('grasp_angle_deg')} (null/None = Home) is_smoothing={plan.get('is_smoothing',False)}", flush=True)
                experiment_stop = None
                if restricted_mode and (patience or max_actions):
                    if not restricted.REPORTS_PREVIOUS_EFFECT:patience=0
                    effect=read(folder/'decision_evidence.json').get('previous_effect')
                    verdict=effect['verdict'] if effect is not None else None
                    no_improvement_streak=experiment_policy.next_streak(no_improvement_streak,verdict,uncertain_counts)
                    experiment_stop=experiment_policy.stop_reason(plan['decision'],no_improvement_streak,
                        patience,len(history),max_actions)
                    evidence=dict(previous_effect_enabled=restricted.REPORTS_PREVIOUS_EFFECT,previous_action_round=history[-1]['source_round'] if history else None,
                        previous_effect=effect,consecutive_no_improvement=no_improvement_streak,
                        patience=patience,completed_actions=len(history),max_actions=max_actions,
                        uncertain_is_no_improvement=uncertain_counts,stop_reason=experiment_stop,
                        done_takes_precedence=True)
                    save(folder/'progress_stop.json',evidence)
                    summary['progress_stop']=evidence
                if experiment_stop:
                    summary.update(termination_reason=experiment_stop,
                        termination_detail=(f'连续{patience}个已完成动作无可见改善且未DONE' if experiment_stop=='NO_IMPROVEMENT'
                                            else '已评估最后一个动作效果，达到动作上限但未DONE'))
                    item['execution']['status']='NO_ACTION_'+experiment_stop
                    host_validation.update(status='NOT_APPLICABLE',reason=experiment_stop)
                    save(folder/'gate_status.json',dict(status='NOT_APPLICABLE',reason=experiment_stop,
                         mode='SHADOW',affects_execution=False,model_called=False))
                    runtime.capture_current(folder/'after');break
                if plan['decision'] != 'ACTION':
                    save(folder/'gate_result.json', dict(decision='UNCERTAIN'))
                    save(folder/'gate_status.json', dict(status='NOT_APPLICABLE', reason=item['planner']+' '+plan['decision'],
                         mode='SHADOW', affects_execution=False, model_called=False))
                    if (folder/'fast_action_overlay.png').exists():
                        shutil.copyfile(folder/'fast_action_overlay.png', folder/'gate_input_overlay.png')
                    host_validation.update(status='NOT_APPLICABLE', reason='No ACTION')
                    summary.update(termination_reason=plan['decision'], termination_detail=plan['short_reason'])
                    item['execution']['status'] = 'NO_ACTION_'+plan['decision']
                    runtime.capture_current(folder/'after'); break
                if not recovery_request:
                    phase = 'SHADOW_GATE'
                    print(f'ROUND {index:03d} SHADOW_GATE (no routing authority)', flush=True)
                    gate = PreActionGate(folder, headers, direct_once, getattr(args, 'gate_timeout_s', 120.))
                    try:
                        result = router.shadow(gate, before.image, root/'goal/target.png')
                    finally:
                        timing['gate_wall_s'] = gate.metrics.get('gate_wall_s', 0.)
                    print(f"ROUND {index:03d} GATE_{result['decision']} [{gate.metrics['status']}] SHADOW ONLY", flush=True)
                else:
                    save(folder/'gate_result.json', dict(decision='UNCERTAIN'))
                    save(folder/'gate_status.json', dict(status='NOT_APPLICABLE',
                        reason='Codex recovery is not a FastPlanner proposal', mode='SHADOW',
                        affects_execution=False, model_called=False))
                plan = router.execution_plan
                phase = 'HOST_CHECKS'
                check_started = time.monotonic()
                host_validation.update(status='CHECKING',
                    fast_plan_sha256=sha(folder/'fast_plan.json') if (folder/'fast_plan.json').exists() else None,
                    execution_planner=item['planner'], plan_sha256=sha(folder/'plan.json'))
                save(folder/'host_validation.json', host_validation)
                try:
                    if not all(sha(p) == digest for p, digest in target_depth_sources.items()):
                        raise DualArmError('frozen target-depth policy/calibration changed')
                    if recovery_request:
                        codex_recovery.reject_repeated(plan, recovery_request)
                    if repeated_action(plan, before.image, history):
                        raise DualArmError('unchanged observation with repeated same action; no redispatch')
                    execution_before=before
                    if candidate_mode:
                        execution_before=runtime.capture_current(folder/('pre_execution_observation' if (automatic_top5 or restricted_mode) else 'after_human_review'))
                        preference.verify_scene(before,execution_before,plan,folder,**({'automatic':True} if (automatic_top5 or restricted_mode) else {}))
                    missing_grasp_depth = None
                    if restricted_mode:
                        try:execution_before.sample(plan['grasp_px'],None)
                        except DepthUnavailableError as exc:missing_grasp_depth=exc
                    if missing_grasp_depth is not None:
                        from cloth_agent import one_shot_depth_recovery
                        before,execution_before,plan,prepared=one_shot_depth_recovery.recover(
                            runtime,root,folder,execution_before,plan,missing_grasp_depth,
                            headers,args.timeout_s,getattr(args,'gate_timeout_s',120.),direct_once,
                            prepare_action,score_options.get('previous_context'),timing,history=history)
                        operations,initial=prepared if prepared is not None else (None,None)
                        host_validation.update(plan_sha256=sha(folder/'plan.json'),
                            fast_plan_sha256=sha(folder/'fast_plan.json'),depth_view_recovered=True)
                    else:
                        operations, initial = prepare_action(runtime, execution_before, plan, folder)
                except DualArmError as exc:
                    # Retry only geometric target rejection, before any dispatch. One attempt.
                    if not restricted_mode or not str(exc).startswith(('table-target fallback rejected:','smoothing action rejected:')):
                        raise
                    archive=folder/'target_replan'/'original';archive.mkdir(parents=True)
                    for path in list(folder.iterdir()):
                        if path.name=='target_replan':continue
                        if path.is_dir():shutil.copytree(path,archive/path.name)
                        else:shutil.copyfile(path,archive/path.name)
                    feedback=dict(error=f'{type(exc).__name__}: {exc}',rejected_plan=plan,
                        target_validation=read(folder/'target_depth.json'),physical_execution=False,max_retries=1)
                    save(folder/'target_replan'/'feedback.json',feedback)
                    print(f'ROUND {index:03d} TARGET REJECT: fresh observation + one model replan',flush=True)
                    retry_started=time.monotonic()
                    before=runtime.capture_current(folder/'target_replan'/'observation')
                    shutil.copytree(before.directory,folder/'before',dirs_exist_ok=True)
                    prior_model_s=read(folder/'model_metrics.json')['wall_latency_s']
                    retry_promoted=False
                    try:
                        plan,metrics=restricted.replan_rejected(root,folder,before,headers,args.timeout_s,direct_once,
                            score_options.get('previous_context'),feedback)
                        retry_promoted=True
                        timing['target_replan_model_s']=metrics['wall_latency_s']
                    finally:
                        retry_metrics=folder/'target_replan'/'retry_01'/'model_metrics.json'
                        extra_model_s=prior_model_s if retry_promoted else (
                            read(retry_metrics).get('wall_latency_s',0.) if retry_metrics.exists() else 0.)
                        timing['depth_recovery_prior_planning_s']=timing.get('depth_recovery_prior_planning_s',0.)+extra_model_s
                        timing['target_replan_wall_s']=time.monotonic()-retry_started
                    if plan['decision']=='ACTION':
                        if repeated_action(plan,before.image,history):
                            raise DualArmError('unchanged observation with repeated same action; no redispatch')
                        if plan['grasp_px']==feedback['rejected_plan']['grasp_px'] and plan['target_px']==feedback['rejected_plan']['target_px']:
                            raise DualArmError('target replan repeated rejected coordinates; no redispatch')
                        execution_before=runtime.capture_current(folder/'target_replan'/'pre_execution_observation')
                        preference.verify_scene(before,execution_before,plan,folder,automatic=True)
                        operations,initial=prepare_action(runtime,execution_before,plan,folder)
                    else:
                        operations,initial=None,None
                    host_validation.update(plan_sha256=sha(folder/'plan.json'),
                        fast_plan_sha256=sha(folder/'fast_plan.json'),target_replanned=True)
                    save(folder/'gate_status.json',dict(status='NOT_APPLICABLE',mode='SHADOW',
                        reason='Target replan; original shadow evidence archived',affects_execution=False,model_called=False))
                    save(folder/'gate_result.json',dict(decision='UNCERTAIN'))
                finally:
                    timing['host_planning_s'] = (time.monotonic()-check_started
                        -timing.get('depth_recovery_wall_s',0.)+timing.get('depth_recovery_host_s',0.)
                        -timing.get('target_replan_model_s',0.))
                    timing['host_validation_s'] = timing['host_planning_s']
                if plan['decision']!='ACTION':
                    # A fresh side view can reveal DONE or cause ABSTAIN; never dispatch the old plan.
                    summary.update(termination_reason=plan['decision'],termination_detail=plan['short_reason'])
                    item['execution']['status']='NO_ACTION_'+plan['decision']
                    host_validation.update(status='NOT_APPLICABLE',reason='replanned '+plan['decision'])
                    view_return_started=time.monotonic()
                    runtime.move_to_perception(folder/'depth_recovery_return_perception')
                    timing['depth_recovery_return_s']=time.monotonic()-view_return_started
                    runtime.capture_current(folder/'after');break
                host_validation.update(status='PASS',
                    scope='Measured Pa; Pb ray/support-plane intersection without local depth; existing metric/IK/workspace/execution checks',
                    target_depth=read(folder/'target_depth.json') if (folder/'target_depth.json').exists() else None)
                save(folder/'host_validation.json', host_validation)
                phase = 'PRE_EXECUTION'
                recorder = TrajectoryRecorder(runtime, folder); robot_started = None
                try:
                    recorder.start(); recorder.stage('one_shot_single_manipulation'); recorder.check()
                    # Existing live arrival check is still performed after shadow latency.
                    phase = 'HOST_CHECKS'
                    check_started = time.monotonic()
                    try:
                        if recovery_request:
                            codex_recovery.check_stationary(runtime, recovery_request['expected_joints'],
                                                           folder/'recovery_pre_dispatch.json')
                        runtime.arrival(initial)
                    finally:
                        timing['host_validation_s'] += time.monotonic()-check_started
                        timing['host_planning_s'] = timing['host_validation_s']
                    phase = 'PRE_EXECUTION'
                    save(folder/'dispatch_intent.json', dict(one_manipulation=True, plan=plan, observation_sha256=sha(before.image),
                        host_observation_sha256=sha(execution_before.image)))
                    summary['physical_actions'] += 1
                    item['execution'] = dict(status='DISPATCHED', physical_execution=True)
                    save(folder/'execution.json', item['execution']); save(root/'episode_summary.json', summary)
                    phase = 'EXECUTION'; robot_started = time.monotonic()
                    execution = runtime._execute_operations(operations, folder)
                    item['execution'].update(execution)
                    save(folder/'execution.json', item['execution'])
                    if execution['status'] != 'EXECUTED': raise DualArmError('execution failed: '+str(execution))
                    recorder.check(); recorder.stage('return_perception')
                    runtime.open_at_home_on_return = True
                    try: runtime.move_to_perception(folder/'return_perception')
                    finally: runtime.open_at_home_on_return = False
                    recorder.check()
                finally:
                    if robot_started is not None: timing['robot_execution_s'] = time.monotonic()-robot_started
                    recorder.close()
                recorder.check(); phase = 'AFTER_OBSERVATION'
                capture = time.monotonic(); runtime.capture_current(folder/'after')
                timing['observation_s'] += time.monotonic()-capture
                item['execution']['acquisition'] = 'UNKNOWN_PENDING_HUMAN_REVIEW'
                history.append(dict(plan=plan,before=before.image,after=folder/'after/rgb.png',
                                    source_round=folder.name,completed_commands=execution.get('completed_commands'),
                                    acquisition='UNKNOWN_PENDING_HUMAN_REVIEW'))
                # History is used only by the repetition guard unless previous-action context is enabled.
                if index == args.max_rounds-1:
                    summary.update(termination_reason='ROUND_LIMIT', termination_detail='Configured maximum reached')
            except BaseException as exc:
                if isinstance(exc, one_shot_views.ViewError):
                    phase = 'VIEW_OBSERVATION_OR_RETURN'
                if phase == 'HOST_CHECKS':
                    host_validation.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'REJECT',
                                           error=f'{type(exc).__name__}: {exc}')
                    if not isinstance(exc, KeyboardInterrupt):
                        router.host_reject(host_validation['error'])
                if isinstance(exc, KeyboardInterrupt): status = 'USER_INTERRUPTED'
                elif phase in ('MODEL', 'DELIBERATIVE_MODEL'): status = 'MODEL_ERROR_NO_RETRY'
                elif not item['execution']['physical_execution']: status = 'SAFETY_REJECTED'
                else: status = 'EXECUTION_STOPPED'
                item['execution'].update(status=status, error=f'{type(exc).__name__}: {exc}', failure_phase=phase)
                if status == 'SAFETY_REJECTED': summary['safety_rejections'] += 1
                category = codex_recovery.recovery_category(exc, phase, item['execution']['physical_execution'])
                attempt = (recovery_request['attempt']+1) if recovery_request else 1
                eligible = bool(recovery_enabled and category)
                can_schedule = (eligible and attempt <= incident_limit
                    and summary['codex_recovery_calls'] < recovery_limit and index+1 < args.max_rounds)
                if recovery_enabled:
                    save(folder/'recovery_eligibility.json', dict(category=category, eligible=eligible,
                        scheduled=can_schedule, next_attempt=attempt, max_per_incident=incident_limit,
                        used_episode_calls=summary['codex_recovery_calls'], max_episode_calls=recovery_limit,
                        another_round_available=index+1 < args.max_rounds))
                if can_schedule:
                    failure = dict(round=folder.name, category=category, phase=phase,
                        error=f'{type(exc).__name__}: {exc}',
                        plan=plan if plan is not None else (read(folder/'plan.json') if (folder/'plan.json').exists() else None),
                        ik_failure=getattr(exc, 'ik_failure', None),
                        observation_sha256=sha(before.image), physical_execution=False)
                    failures = [*(recovery_request['failures'] if recovery_request else []), failure]
                    pending_recovery = dict(attempt=attempt, failures=failures,
                        expected_joints=before.meta['joints_deg'], future_observation_required=True,
                        original_action_executed=False, hardware_recovery_allowed=False)
                    save(folder/'recovery_handoff.json', pending_recovery)
                    save(folder/'after/unavailable.json', dict(reason='Rejected before dispatch; next round takes a fresh recovery observation'))
                    item['execution']['recovery_scheduled'] = True
                    print(f"ROUND {index:03d} rejected ({category}); fresh observation + Codex next round", flush=True)
                    continue
                raise
            finally:
                if (folder/'model_metrics.json').exists():
                    timing['planning_s'] = (read(folder/'model_metrics.json')['wall_latency_s']
                                            +timing.get('depth_recovery_prior_planning_s',0.))
                    timing['fast_planner_wall_s'] = timing['planning_s']
                    timing['geometry_report_s'] = read(folder/'model_metrics.json').get('geometry_wall_s',0.)
                    timing['structure_grounding_model_s'] = read(folder/'model_metrics.json').get('structure_model_wall_s',0.)
                    timing['structure_stage_s'] = read(folder/'model_metrics.json').get('structure_wall_s',0.)
                    timing['action_model_s'] = read(folder/'model_metrics.json').get('action_model_wall_s',timing['planning_s'])
                if (folder/'deliberative_metrics.json').exists():
                    timing['deliberative_planner_wall_s'] = read(folder/'deliberative_metrics.json')['wall_latency_s']
                timing['round_wall_s'] = time.monotonic()-began
                timing['total_round_s'] = timing['round_wall_s']
                summary['total_planning_s'] += timing['planning_s']
                summary['total_deliberative_planning_s'] += timing['deliberative_planner_wall_s']
                summary['total_robot_execution_s'] += timing['robot_execution_s']
                save(folder/'timing.json', timing); save(folder/'execution.json', item['execution'])
                host_validation['wall_s'] = timing['host_validation_s']
                save(folder/'host_validation.json', host_validation)
                save(root/'episode_summary.json', summary)
                if automatic_top5:auto.finalize_round(folder,item)
                if restricted_mode:restricted.render_round(folder)
    except BaseException as exc:
        summary.update(termination_reason='USER_STOP' if isinstance(exc, KeyboardInterrupt) else 'SAFETY_STOP',
                       termination_detail=f'{type(exc).__name__}: {exc}')
        (root/'exception.txt').write_text(traceback.format_exc())
        if runtime:
            runtime.cancel.set()
            try: summary['stop_results'] = runtime.stop()
            except Exception as stop_error: summary['stop_error'] = str(stop_error)
        if folder and not (folder/'after/rgb.png').exists():
            save(folder/'after/unavailable.json', dict(reason='Stopped; no extra movement or capture after rejection'))
    finally:
        signal.signal(signal.SIGTERM, old_signal)
        if runtime:
            try: runtime.close()
            except Exception as exc: summary['cleanup_error'] = str(exc)
        summary.update(finished_at=planner.utc(), episode_wall_s=time.monotonic()-started,
            total_view_observation_s=sum(r['timing'].get('view_observation_s', 0.) for r in summary['rounds']),
            total_view_return_s=sum(r['timing'].get('view_return_s', 0.) for r in summary['rounds']),
            total_planner_return_overlap_s=sum(r['timing'].get('planner_return_overlap_s', 0.) for r in summary['rounds']),
            planning_latency_s=planner.stats([r['timing']['planning_s'] for r in summary['rounds']
                                              if (root/r['round']/'attempt.json').exists()]),
            frozen_inputs_unchanged=all(sha(root/p) == value for p, value in frozen.items()),
            protected_sources_unchanged={p: sha(p) == value for p, value in protected.items()})
        if (root/'initialization/result.json').exists():
            init = read(root/'initialization/result.json'); summary['initialization'] = init
            summary['initialization_physical_actions'] = init['physical_actions']
        init_metrics = root/'initialization/region_request/model_metrics.json'
        summary['initialization_planning_s'] = ((read(init_metrics)['wall_latency_s'] if init_metrics.exists() else 0.)+
            (read(root/'initialization/press_region_request/model_metrics.json')['wall_latency_s']
             if (root/'initialization/press_region_request/model_metrics.json').exists() else 0.))
        summary['total_model_planning_s_including_initialization'] = (
            summary['total_planning_s']+summary['initialization_planning_s']+summary['total_deliberative_planning_s'])
        summary['total_robot_execution_s_including_initialization'] = summary['total_robot_execution_s']+summary.get('initialization', {}).get('robot_execution_s', 0.)
        if previous:
            summary.update(cumulative_rounds=prior_rounds+summary['total_rounds'],
                cumulative_physical_actions=continuation['prior_physical_actions']+summary['physical_actions'],
                cumulative_safety_rejections=continuation['prior_safety_rejections']+summary['safety_rejections'],
                cumulative_planning_s=continuation['prior_planning_s']+summary['total_planning_s'],
                cumulative_robot_execution_s=continuation['prior_robot_execution_s']+summary['total_robot_execution_s'],
                continuation_source_unchanged=sha(source/'episode_summary.json') == continuation['source_summary_sha256']
                    and all(sha(p) == value for p, value in continuation['source_hashes'].items()))
        planner_shadow_review.summarize(root, summary)
        if candidate_mode:
            summary['planner_layers']['proposal_source']='RestrictedContextOneShot' if restricted_mode else 'GPTAutomaticTop5' if automatic_top5 else 'HumanPreference'
            summary['planner_layers']['success_questions']['fast_20_25s']=dict(
                status='NOT_APPLICABLE',reason='single contextual action constrained to visible sleeve/hem' if restricted_mode else 'two-stage scoring with explicit GPT choice' if automatic_top5 else 'two-stage scoring plus mandatory human selection, not original FastPlanner')
        summary['total_gate_wall_s'] = summary['planner_layers']['total_gate_wall_s']
        summary['total_model_planning_s_including_gate_and_initialization'] = (
            summary['total_model_planning_s_including_initialization']+summary['total_gate_wall_s'])
        save(root/'episode_summary.json', summary)
        try:
            render(root, summary)
            planner_shadow_review.render_review(root, summary)
            if human_preference:
                preference.append_report(root,summary)
            if automatic_top5:auto.append_report(root,summary)
            else:
                planner_shadow_review.append_report(root, summary)
        except Exception as exc:
            summary['report_error'] = str(exc); save(root/'episode_summary.json', summary)
        print('EPISODE FINISHED '+str(summary['termination_reason'])+': '+summary['termination_detail'], flush=True)
    return 1 if summary['termination_reason'] == 'SAFETY_STOP' else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--goal-image', type=Path, default=DEFAULT_GOAL)
    parser.add_argument('--config', type=Path, default=Path('config/dual_arm.local.json'))
    parser.add_argument('--max-rounds', type=int, default=12)
    parser.add_argument('--timeout-s', type=float, default=600.)
    parser.add_argument('--gate-timeout-s', type=float, default=120.,
                        help='independent shadow gate request timeout; errors do not replace or block the fast action')
    parser.add_argument('--review-initialization', action='store_true',
                        help='wait stationary for image-bound initialization/review_decision.json')
    parser.add_argument('--initialize-center-radius-mm',type=float,default=80.,
                        help='maximum measured XY distance from the visible body-center sampling point (default 80 mm)')
    parser.add_argument('--press-after-drop',action='store_true',
                        help='after free drop, locate visible center, lower closed gripper to hard-table world Z=0 over sponge, lift and reobserve')
    parser.add_argument('--codex-recovery', action='store_true',
                        help='bounded Codex replanning on allowlisted pre-dispatch errors; never retries hardware faults')
    parser.add_argument('--max-codex-replans', type=int, default=4)
    parser.add_argument('--max-replans-per-incident', type=int, default=2)
    parser.add_argument('--codex-timeout-s', type=float, default=600.)
    parser.add_argument('--continue-from', type=Path,
                        help='explicitly authorized continuation; no reset; preserves goal/prompt and remaining budget')
    parser.add_argument('--human-preference',action='store_true',
                        help='score Ps and Pt separately; require human Top10 choice; learn alpha from preferences')
    parser.add_argument('--no-improvement-patience',type=int,default=0)
    parser.add_argument('--max-actions',type=int,default=0,help='requires an additional final observation round')
    parser.add_argument('--uncertain-is-no-improvement',action='store_true')
    parser.add_argument('--visible-sleeve-hem',action='store_true',help='single contextual action; only visible sleeve/hem grasps; no TopN')
    parser.add_argument('--automatic-top5',action='store_true',help='save Ps/Pt Top5; execute explicit GPT choice without learned preferences')
    parser.add_argument('--stateful-context',action='store_true',
                        help='automatic Top5: from round 2, show GPT the previous pre-action RGB and executed action; log its local-effect assessment')
    parser.add_argument('--alpha-state',type=Path,default=Path('data/experiments/one_shot_preference/alpha.json'))
    parser.add_argument('--ranking-correction', action='store_true',
                        help='freeze alpha; learn residual ranking from full human Top10 orders')
    parser.add_argument('--residual-state',type=Path,default=Path('data/experiments/one_shot_preference/residual.json'))
    parser.add_argument('--preference-port',type=int,default=8767)
    parser.add_argument('--preference-timeout-s',type=float,default=1800.)
    args = parser.parse_args()
    if not 1 <= args.max_rounds <= 50 or not 0 < args.timeout_s <= 1200 or not 0 < args.gate_timeout_s <= 1200:
        parser.error('invalid limits')
    if args.no_improvement_patience<0 or args.max_actions<0:
        parser.error('stop limits must be nonnegative')
    if args.max_actions and args.max_rounds<args.max_actions+1:
        parser.error('max-rounds must include final action assessment: at least max-actions+1')
    if not 0<args.initialize_center_radius_mm<=80:
        parser.error('initialization center radius must be in (0,80] mm')
    if not 1 <= args.max_codex_replans <= 8 or not 1 <= args.max_replans_per_incident <= 4 or not 0 < args.codex_timeout_s <= 1200:
        parser.error('invalid recovery limits')
    if not 0<=args.preference_port<=65535 or not 0<args.preference_timeout_s<=7200:
        parser.error('invalid human review limits')
    if args.human_preference and (args.codex_recovery or args.continue_from):
        parser.error('human preference requires a new episode without automatic recovery')
    if args.ranking_correction and not args.human_preference:
        parser.error('ranking correction requires --human-preference')
    if not args.goal_image.is_file() or args.output.exists(): parser.error('goal missing or output already exists')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
