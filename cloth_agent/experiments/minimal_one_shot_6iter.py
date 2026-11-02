"""Minimal-prompt, single-episode wrapper. Hardware only with --execute."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
import json
import re
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
MAX_ROUNDS = 12
PROMPT = """把当前T恤展开至目标状态。未完成时输出一个lift_pull_place动作：grasp_px是抓点Pa，target_px是放下点Pb；已展开则DONE；无法选择动作则ABSTAIN。
坐标为第一张图的整数像素，左上为原点，x向右、y向下。grasp_angle_deg填null，沿用Home夹爪方向。
DONE/ABSTAIN时坐标和motion_type均为null。short_reason只写一句。只输出指定JSON。"""


class DisabledShadowGate:
    """Disable only the nonbinding quality-model call; Host checks are untouched."""
    def __init__(self, folder, headers, request_once, timeout_s=120.):
        self.folder = Path(folder)
        self.metrics = dict(status='DISABLED', mode='DISABLED', affects_execution=False,
                            model_called=False, request_attempted=False, gate_wall_s=0.)

    def evaluate(self, current_rgb, goal_rgb, proposed_action):
        # The existing router requires an enum; this is explicitly NOT a model verdict.
        value = {'decision': 'UNCERTAIN'}
        self.metrics['reason'] = 'Minimal one-shot: no auxiliary quality-model request'
        for name, data in [('gate_status.json', self.metrics), ('gate_result.json', value)]:
            (self.folder/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
        return value


@contextmanager
def minimal_runner():
    """Process-local adapters only. No edits to shared prompts or robot modules."""
    from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode
    wire = episode.planner
    original = (wire.ROTATING_MULTIVIEW_PROMPT, wire.ROTATION_SCHEMA,
                episode.PreActionGate, episode.GATE_PROMPT, episode.save, episode.SINGLE_VIEW_PROMPT, episode.direct_once)
    schema = deepcopy(wire.ROTATION_SCHEMA)
    schema['properties']['grasp_angle_deg'] = {'type': 'null'}
    original_save = episode.save
    original_direct = episode.direct_once

    def audited_direct(folder, body, headers, timeout_s):
        if body['text']['format']['name'] == 'unfold_single_step':
            content = body['input'][0]['content']
            expected_text = body['input'][0]['content'][0]['text']
            feedback_path = Path(folder)/'retry_feedback.json'
            feedback = json.loads(feedback_path.read_text()) if feedback_path.exists() else None
            if feedback is not None:
                suffix = episode.retry_feedback_text(feedback)
                if not expected_text.endswith(suffix) or feedback.get('physical_execution') is not False:
                    raise ValueError('Retry feedback contract violated; no request dispatched')
                expected_text = expected_text[:-len(suffix)]
            if (body['instructions'] != PROMPT or len(body['input']) != 1
                    or len(content) != 3 or content[0]['type'] != 'input_text'
                    or [c['type'] for c in content[1:]] != ['input_image', 'input_image']
                    or re.fullmatch(r'当前 perception 图尺寸 \d+×\d+ 像素。第一张 current，第二张 goal。', expected_text) is None
                    or body['text']['format']['schema'] != schema
                    or body.get('tools') != [] or body.get('tool_choice') != 'none'
                    or 'previous_response_id' in body):
                raise ValueError('Minimal request contract violated; no request dispatched')
            original_save(Path(folder)/'request_content_audit.json', dict(
                prompt_exact_match=True, schema_exact_match=True, image_count=2,
                live_views=['perception'], static_goal=True, history=feedback is not None, skills=False,
                retry_feedback_supplied=feedback is not None,
                history_scope='last rejected proposal only' if feedback is not None else 'none',
                agent_cli=False, tools=[]))
        return original_direct(folder, body, headers, timeout_s)

    def save(path, value):
        path = Path(path)
        if path.name == 'protocol.json':
            value = dict(value)
            value.update(experiment='minimal_one_shot_6iter', prompt=PROMPT,
                         schema=schema, max_rounds=MAX_ROUNDS, model_calls_per_round=1,
                         single_view=True, observation_views=['perception'], input_image_count=2,
                         model_overlaps_view_return=False,
                         gripper_orientation='fixed Home orientation; grasp_angle_deg=null',
                         global_gripper_boundary_prompt='',
                         gate=dict(mode='DISABLED', model_called=False, affects_execution=False,
                                   reason='No auxiliary quality request; existing Host checks retained'),
                         minimal_one_shot=dict(single_episode=True, iteration_budget=MAX_ROUNDS,
                             max_actions=MAX_ROUNDS, additional_final_done_call=False,
                             automatic_restart=False, skill_injection=False,
                             history_policy='last rejected plan and reason on retry only',
                             codex_recovery=False, fixed_home_gripper_orientation=True,
                             same_prompt_retry=True, retry_consumes_iteration=True,
                             retry_plan_feedback=True,
                             reset_policy='Existing initialize() unchanged; --no-reset for manual state',
                             transport='Existing direct ChatGPT account Responses; no Agent CLI',
                             entrypoint_sha256=wire.sha(Path(__file__))))
            shutil.copyfile(__file__, path.parent/'minimal_entrypoint.py')
        original_save(path, value)

    wire.ROTATING_MULTIVIEW_PROMPT = PROMPT
    episode.SINGLE_VIEW_PROMPT = PROMPT
    episode.direct_once = audited_direct
    wire.ROTATION_SCHEMA = schema
    episode.PreActionGate = DisabledShadowGate
    episode.GATE_PROMPT = ''
    episode.save = save
    try:
        yield episode
    finally:
        (wire.ROTATING_MULTIVIEW_PROMPT, wire.ROTATION_SCHEMA,
         episode.PreActionGate, episode.GATE_PROMPT, episode.save, episode.SINGLE_VIEW_PROMPT, episode.direct_once) = original


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Run one real episode')
    mode.add_argument('--dry-run', action='store_true', help='Print contract only; no model or hardware')
    p.add_argument('--no-reset', action='store_true', help='Use your manually reset state as-is')
    p.add_argument('--output', type=Path)
    p.add_argument('--goal-image', type=Path,
                   default=REPO/'data/reference/deployment_example/goal.png')
    p.add_argument('--config', type=Path, default=REPO/'config/dual_arm.local.json')
    p.add_argument('--timeout-s', type=float, default=600.)
    p.add_argument('--initialize-seed', type=int)
    p.add_argument('--initialize-center-radius-mm', type=float, default=80.)
    p.add_argument('--press-after-drop', action='store_true', help='Existing optional reset press')
    return p


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if not args.execute and not args.dry_run:
        parser.print_help()
        return 0
    if not 0 < args.timeout_s <= 1200 or not 0 <= args.initialize_center_radius_mm <= 80:
        parser.error('invalid timeout or initialization radius')
    if args.no_reset and args.press_after_drop:
        parser.error('--press-after-drop requires automatic reset')
    if args.dry_run:
        print(json.dumps(dict(prompt=PROMPT, model='gpt-6-astra', effort='high',
            max_iterations=MAX_ROUNDS, episodes=1, planner_calls_per_iteration=1,
            reset='SKIP_MANUAL_STATE' if args.no_reset else 'EXISTING_NORMAL_RESET',
            initialization_calls='separate; existing region localization',
            images=['current perception RGB', 'static goal RGB'],
            live_camera_views=['perception'], side_view_motion=False,
            shadow_model_calls=0, codex_recovery=False, automatic_restart=False,
            same_prompt_retry=True, retry_consumes_iteration=True, retry_plan_feedback=True,
            hardware_connected=False), ensure_ascii=False, indent=2))
        return 0
    output = (args.output or REPO/'reports'/('minimal_one_shot_6iter_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if output.exists():
        parser.error('output already exists; choose a fresh directory')
    if not args.goal_image.is_file() or not args.config.is_file():
        parser.error('goal image or config missing')
    forwarded = ['minimal_one_shot_6iter', '--execute', '--single-view', '--same-prompt-retry',
                 '--retry-plan-feedback', '--max-rounds', str(MAX_ROUNDS),
                 '--output', str(output), '--goal-image', str(args.goal_image.resolve()),
                 '--config', str(args.config.resolve()), '--timeout-s', str(args.timeout_s),
                 '--initialize-center-radius-mm', str(args.initialize_center_radius_mm)]
    for flag in ['no_reset', 'press_after_drop']:
        if getattr(args, flag):
            forwarded.append('--'+flag.replace('_', '-'))
    if args.initialize_seed is not None:
        forwarded += ['--initialize-seed', str(args.initialize_seed)]
    print(f'ONE EPISODE; at most {MAX_ROUNDS} planner iterations; output: {output}', flush=True)
    previous_argv = sys.argv
    try:
        with minimal_runner() as episode:
            sys.argv = forwarded
            code = episode.main()
    finally:
        sys.argv = previous_argv
    summary_path = output/'episode_summary.json'
    if not summary_path.exists():
        print('STOPPED: no episode summary; no automatic restart.', flush=True)
        return code or 1
    summary = json.loads(summary_path.read_text())
    reason = summary.get('termination_reason')
    done = reason == 'DONE'
    note = dict(termination_reason=reason, iterations=summary.get('total_rounds', 0),
                iteration_limit=MAX_ROUNDS, model_done=done, physical_success_verified=False,
                manual_reset_required=not done, automatically_restarted=False)
    (output/'operator_notification.json').write_text(json.dumps(note, ensure_ascii=False, indent=2)+'\n')
    if done:
        print('DONE：本场已结束（模型判断，未经人工成功确认）。', flush=True)
        return code
    print('\a'+(f'{MAX_ROUNDS}次迭代已用完，仍未DONE。' if reason == 'ROUND_LIMIT' else f'提前停止：{reason}，尚未DONE。')+
          '\n已停止，不会自动reset或重启。请手动reset，再运行：\n'
          'bash scripts/run_minimal_one_shot_6iter.sh --execute --no-reset\n'
          f'记录：{output}', flush=True)
    return 2 if reason == 'ROUND_LIMIT' else (code or 1)


if __name__ == '__main__':
    raise SystemExit(main())
