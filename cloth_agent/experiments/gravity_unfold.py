"""Gravity-aware seed rollout. Import/help/baseline audit never connect a robot."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import signal

import numpy as np
from cloth_agent.codex_astra_pipeline import ROOT, run
from cloth_agent.codex_motion import parse_code

PRIMITIVES = ('LiftAndReveal', 'LiftDrapeRelease', 'PeelOverlap', 'ShortDragReveal')
OUTCOMES = ('STRUCTURE_REVEALED', 'PROGRESS', 'NO_PROGRESS', 'REGRESSION', 'UNKNOWN')
READY_KEYS = ('torso_open', 'most_landmarks_locatable', 'no_severe_self_occlusion', 'folding_graph_compatible')
POLICY = '''只做一条 gravity-aware unfolding seed rollout，不建图、不定义 node taxonomy、不训练。
目标 FOLD_READY：torso 基本展开，collar/sleeves/hem 中大部分可定位，无严重大面积自遮挡，能进入现有 folding graph；不追求 perfect flattening。
按顺序判断并给出物理证据：
1. 有可靠 exposed boundary/cuff/hem/flap/extremity 且可安全 lift -> LiftAndReveal：pick、vertical lift、悬垂1.5秒、现场拍照。它同时是 manipulation 和 diagnostic action。
2. 上一步 lift 的新图确认有意义 extremity 和清晰 hanging structure -> LiftDrapeRelease：保持 grasp，抬到安全高度，移至附近可见空闲区域上方，缓慢下降、release。
3. 有独立 overlap/flap edge -> PeelOverlap：小幅 lift，横向分离，下降、release。
4. 没有可靠 extremity 或无法安全 lift 且无独立 overlap -> ShortDragReveal：短距离低速拖动，路径总长<=50mm、高度变化<=15mm。
NO_PROGRESS 是当前 observation/decision state 的 failed evidence。下一步必须换 primitive/几何策略，禁止无理由重复桌面 drag；不自动更深抓或拖更远。
所有抓点从本轮 world_xyz_mm.npy 读取；米/弧度，right 单臂；使用已应用桌面深度修正的观测高度，GPT 和宿主均不额外降低抓取 Z（偏移为0mm），不要重复修正。
相机在 arm7 手腕：悬垂视角可能不可见，不能假设图像面积可直接比较；不充分写 UNKNOWN。
如果 holding=true，当前悬垂点云未应用桌面深度补丁，仅作结构诊断，禁止用于新的抓点或桌面高度。放置区域与桌面高度从 context.table_observation 的已修正点云读取，结合当前 RGB 确认可见空闲区域；若不能确认则 BLOCKED。只能给 LiftDrapeRelease 或 BLOCKED；若悬垂证据不足，BLOCKED，不能盲目放置或重新抓。
JSON 输出：primitive（四种之一或 FOLD_READY/BLOCKED），reason（具体依据），grasp（6元素 pose 或保持抓取时 null），targets（按顺序6元素pose数组），
evidence（exposed_extremity,safe_lift,hanging_structure,meaningful_extremity,overlap_edge 五个布尔值），
fold_ready（torso_open,most_landmarks_locatable,no_severe_self_occlusion,folding_graph_compatible 四个布尔值）。
LiftAndReveal 恰好一个 target，严格竖直向上、不旋转；LiftDrapeRelease 至少三个 target：lift、移到空闲区域上方、下降。
PeelOverlap 至少三个 target：小幅竖直 lift、横向分离、下降。ShortDragReveal 恰好一个 target。
FOLD_READY/BLOCKED 时 grasp=null、targets=[]。禁止输出动作代码、工具执行机器人或其他字段。
'''
EVALUATOR = '''比较附件1 before RGB、附件2 after RGB，独立评估刚才动作的真实效果。命令 EXECUTED 不等于进展。
JSON 输出 visible_area_change（含视角可比性和依据字符串）,new_landmarks（新暴露landmark/boundary字符串数组）,overlap_change（依据字符串）,
garment_axis_recoverability（依据字符串）,whole_garment_translation_only（布尔值）,
outcome（STRUCTURE_REVEALED/PROGRESS/NO_PROGRESS/REGRESSION/UNKNOWN），
fold_ready（torso_open,most_landmarks_locatable,no_severe_self_occlusion,folding_graph_compatible 四个布尔值），reason（具体视觉依据）。
新 landmark/boundary -> STRUCTURE_REVEALED；面积增加/overlap减少/结构更清楚 -> PROGRESS；仅整体平移/轻微wrinkle -> NO_PROGRESS；明显变糟 -> REGRESSION。
视角变化、遮挡或图像不足则 UNKNOWN，不把像素面积变化当成真实面积增加。悬挂不算桌上 fold-ready。
'''


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def ask(prompt, images, folder, model_call=run, *, session=None):
    folder = Path(folder)
    options = {'session_id': session.get('id')} if session is not None else {}
    call = model_call(prompt, output=folder, images=images, read_only=True, raw_prompt=True, **options)
    if call['status'] != 'AGENT_FINISHED':
        raise RuntimeError(f'model failed: {call["status"]}')
    if session is not None:
        returned = call.get('session_id')
        if not isinstance(returned, str) or not returned.strip() or (session.get('id') and session['id'] != returned):
            raise ValueError('missing or changed unfold session ID')
        session['id'] = returned
        save(folder.parent.parent / 'session.json', session)
    return json.loads((folder / 'answer.txt').read_text())


def ready(value):
    if not isinstance(value, dict) or set(value) != set(READY_KEYS) or any(type(v) is not bool for v in value.values()):
        raise ValueError('invalid fold_ready evidence')
    return all(value.values())


def evaluation(data):
    required = {'visible_area_change', 'new_landmarks', 'overlap_change', 'garment_axis_recoverability',
                'whole_garment_translation_only', 'outcome', 'fold_ready', 'reason'}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError('invalid progress fields')
    for key in ('visible_area_change', 'overlap_change', 'garment_axis_recoverability', 'reason'):
        value = data[key]
        # Preserve structured visual evidence without imposing prose subkeys.
        # Empty/numeric/boolean objects still cannot masquerade as evidence.
        if (isinstance(value, dict) and value
                and all(isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip()
                        for k, v in value.items())):
            data[key] = json.dumps(value, ensure_ascii=False)
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f'progress requires visual evidence: {key}')
    if (data['outcome'] not in OUTCOMES or type(data['whole_garment_translation_only']) is not bool
            or not isinstance(data['new_landmarks'], list)
            or any(not isinstance(x, str) or not x.strip() for x in data['new_landmarks'])):
        raise ValueError('invalid progress evaluation')
    ready(data['fold_ready'])
    if data['whole_garment_translation_only']:
        if data['new_landmarks']:
            raise ValueError('translation-only contradicts newly exposed structure')
        data['outcome'] = 'NO_PROGRESS'
    elif data['outcome'] not in ('UNKNOWN', 'REGRESSION') and data['new_landmarks']:
        data['outcome'] = 'STRUCTURE_REVEALED'
    elif data['outcome'] == 'STRUCTURE_REVEALED' and not data['new_landmarks']:
        raise ValueError('STRUCTURE_REVEALED requires new landmark/boundary')
    return data


def unknown(reason):
    return dict(visible_area_change='unknown', new_landmarks=[], overlap_change='unknown',
                garment_axis_recoverability='unknown', whole_garment_translation_only=False,
                outcome='UNKNOWN', fold_ready=dict.fromkeys(READY_KEYS, False), reason=reason)


def command(decision, *, holding=False, previous=None, held_pose=None):
    if set(decision) != {'primitive', 'reason', 'grasp', 'targets', 'evidence', 'fold_ready'}:
        raise ValueError('invalid decision fields')
    p = decision['primitive']; ev = decision['evidence']
    if (not isinstance(decision['reason'], str) or not decision['reason'].strip()
            or not isinstance(ev, dict) or set(ev) != {'exposed_extremity', 'safe_lift', 'hanging_structure', 'meaningful_extremity', 'overlap_edge'}
            or any(type(x) is not bool for x in ev.values())):
        raise ValueError('decision requires physical evidence')
    is_ready = ready(decision['fold_ready'])
    if p in ('FOLD_READY', 'BLOCKED'):
        if decision['grasp'] is not None or decision['targets'] != []:
            raise ValueError('terminal decision must not contain motion')
        if p == 'FOLD_READY' and (holding or not is_ready):
            raise ValueError('FOLD_READY requires all criteria and released cloth')
        return None
    if p not in PRIMITIVES:
        raise ValueError('unknown primitive')
    if previous and previous['progress_evaluation']['outcome'] == 'NO_PROGRESS' and previous['chosen_primitive'] == p:
        raise ValueError('NO_PROGRESS forbids immediate repetition of this action family')
    if holding:
        if p != 'LiftDrapeRelease' or decision['grasp'] is not None or not (ev['hanging_structure'] and ev['meaningful_extremity']):
            raise ValueError('held continuation requires observed hanging extremity')
        start = held_pose
    else:
        expected = ('LiftAndReveal' if ev['exposed_extremity'] and ev['safe_lift'] else
                    'PeelOverlap' if ev['overlap_edge'] else 'ShortDragReveal')
        # A failed lift is evidence to try another applicable strategy.
        failed = previous and previous['progress_evaluation']['outcome'] == 'NO_PROGRESS'
        if p != expected and not (failed and p == 'PeelOverlap' and ev['overlap_edge']):
            raise ValueError('action violates gravity hierarchy')
        start = decision['grasp']
    poses = np.asarray([start, *decision['targets']], dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 6 or len(poses) < 2 or not np.isfinite(poses).all():
        raise ValueError('invalid pose sequence')
    delta = np.diff(poses[:, :3], axis=0)
    if p == 'LiftAndReveal' and (len(delta) != 1 or delta[0, 2] <= 0 or
            not np.allclose(poses[0, [0, 1, 3, 4, 5]], poses[1, [0, 1, 3, 4, 5]], atol=1e-6)):
        raise ValueError('reveal must be one vertical lift without rotation')
    if p in ('PeelOverlap', 'LiftDrapeRelease'):
        if len(delta) < 3 or delta[0, 2] <= 0 or np.linalg.norm(delta[0, :2]) > .001 or delta[-1, 2] >= 0:
            raise ValueError('primitive requires lift, separation, descent')
        if p == 'PeelOverlap' and (delta[0, 2] > .06 or np.linalg.norm(delta[:, :2], axis=1).sum() > .15):
            raise ValueError('peel exceeds small-motion bounds')
        if p == 'LiftDrapeRelease' and np.linalg.norm(delta[:, :2], axis=1).sum() > .30:
            raise ValueError('drape must use nearby region')
    if p == 'ShortDragReveal' and (len(delta) != 1 or np.linalg.norm(delta, axis=1).sum() > .05 or abs(delta[0, 2]) > .015):
        raise ValueError('fallback drag exceeds limits')
    lines = [] if holding else [f'motion.grasp(right={poses[0].tolist()!r})']
    lines += [f'motion.move_tcp(right={pose.tolist()!r})' for pose in poses[1:]]
    if holding:
        lines += ['motion.gripper(arm="right", action="open")']
    source = '\n'.join(lines) + '\n'
    parse_code(source, allow_grasp=True)
    return source


def metrics(steps, fold_ready=False):
    physical = [s for s in steps if s['execution'].get('physical_execution') is True]
    n = len(physical)
    no = sum(s['progress_evaluation']['outcome'] == 'NO_PROGRESS' for s in physical)
    unknowns = sum(s['progress_evaluation']['outcome'] == 'UNKNOWN' for s in physical)
    return dict(physical_actions=n, no_progress_actions=no, no_progress_ratio=no / n if n else None,
                unknown_actions=unknowns,
                gravity_assisted_actions=sum(s['chosen_primitive'] in PRIMITIVES[:2] for s in physical),
                structure_revealed=sum(s['progress_evaluation']['outcome'] == 'STRUCTURE_REVEALED' for s in physical),
                actions_to_fold_ready=n if fold_ready else None)


def report(output, record):
    for step in record['steps']:
        step['outcome'] = step['progress_evaluation']['outcome']
    record['failed_evidence'] = [dict(decision_state=s['decision_state'], step=s['step'],
        action_family=s['chosen_primitive'], action_command=s['action_command'],
        evidence=s['progress_evaluation'], before_rgb=s['before_rgb'], after_rgb=s['after_rgb'])
        for s in record['steps'] if s['outcome'] == 'NO_PROGRESS']
    record['metrics'] = metrics(record['steps'], record['status'] == 'FOLD_READY')
    record['comparison'] = {'v0': record.get('baseline', {}).get('metrics'), 'v1': record['metrics']}
    save(output / 'unfolding_rollout_v1.json', record)
    lines = ['# Gravity-aware unfolding v1', '', f'Status: {record["status"]}',
             f'Baseline review: {record.get("baseline_review_status", "NOT_REVIEWED")}', '',
             'Counts are physical primitive attempts (including partial execution), not controller waypoints.',
             'UNKNOWN is not counted as success. v0 unknown metrics remain null.', '',
             '| Metric | Experience Graph v0 | v1 |', '|---|---:|---:|']
    for k, v in record['metrics'].items():
        base = (record['comparison']['v0'] or {}).get(k)
        lines.append(f'| {k} | {base} | {v} |')
    for s in record['steps']:
        lines += ['', f'## Step {s["step"]}: {s["chosen_primitive"]}',
                  f'Outcome: {s["progress_evaluation"]["outcome"]}',
                  f'Before: {s["before_rgb"]}', f'After: {s.get("after_rgb")}',
                  '```json', json.dumps(s, ensure_ascii=False, indent=2), '```']
    (output / 'unfolding_rollout_v1.md').write_text('\n'.join(lines) + '\n')


def rollout(runtime, output, baseline, *, max_actions=30, model_call=run):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    record = dict(version=1, status='ACTION_LIMIT', baseline=baseline, steps=[], failed_evidence=[])
    held_pose = None
    session = {}
    try:
        print('进入原 unfold 观测流程：开爪 → Home → perception → RGB-D。', flush=True)
        before = runtime.observe(output / 'initial')
        table_observation = str(before.directory)
        for index in range(1, max_actions + 1):
            folder = output / f'step_{index:03d}'; folder.mkdir()
            context = dict(observation=str(before.directory), table_observation=table_observation, holding=runtime.holding,
                           held_pose=held_pose, failed_evidence=record['failed_evidence'],
                           previous=record['steps'][-1] if record['steps'] else None)
            images = [before.image]
            if context['previous']:
                images.append(Path(context['previous']['before_rgb']))
            prompt = (POLICY + '\n按原 unfold 在同一个会话决定下一步，不另开动作后评估调用。'
                      '第一张是当前 RGB；若有第二张，是上一动作前 RGB。'
                      '最终 JSON 额外包含 previous_progress：首轮 null；其后填写上一动作效果，'
                      '字段遵循下方评估格式，但不要把评估作为顶层输出，顶层仍是下一步决策。\n'
                      + EVALUATOR.replace('附件1 before RGB、附件2 after RGB', '附件2 before RGB、附件1 after RGB')
                      + '\n上述评估对象只放在 previous_progress 内。顶层必须是 primitive/reason/grasp/targets/evidence/fold_ready/previous_progress。'
                      + '\n' + json.dumps(context, ensure_ascii=False))
            decision = ask(prompt, images, folder / 'planner', model_call, session=session)
            progress = decision.pop('previous_progress', None)
            if context['previous']:
                previous = context['previous']
                previous['progress_evaluation'] = evaluation(progress)
                save(output / f'step_{previous["step"]:03d}' / 'progress_evaluation.json', previous['progress_evaluation'])
                report(output, record)
            source = command(decision, holding=runtime.holding, previous=context['previous'], held_pose=held_pose)
            save(folder / 'decision.json', decision)
            if source is None:
                if decision['primitive'] == 'FOLD_READY':
                    record['status'] = 'FOLD_READY'; record['terminal_decision'] = decision
                    break
                # Match the original unfold loop: BLOCKED skips this round,
                # never declares task completion. A held grasp is observed in
                # place; no release or recovery motion is inferred from BLOCKED.
                record.setdefault('blocked_decisions', []).append(dict(round=index, decision=decision))
                print(f'Round {index}: BLOCKED，跳过动作并继续观测：{decision["reason"]}', flush=True)
                report(output, record)
                if index < max_actions:
                    before = runtime.observe(folder / 'blocked_observation')
                    if not runtime.holding:
                        table_observation = str(before.directory)
                    if context['previous']:
                        context['previous']['after_rgb'] = str(before.image.resolve())
                continue
            (folder / 'motion.py').write_text(source)
            step = dict(step=index, decision_state=str(before.directory.resolve()),
                        before_rgb=str(before.image.resolve()), after_rgb=None,
                        chosen_primitive=decision['primitive'], grasp=decision['grasp'], target=decision['targets'],
                        action_command=source, decision=decision,
                        execution={'status': 'PENDING', 'physical_execution': False},
                        progress_evaluation=unknown('not yet observed'))
            record['steps'].append(step); report(output, record)
            execution = runtime.execute_primitive(decision['primitive'], source, before, folder)
            step['execution'] = execution; save(folder / 'execution.json', execution)
            report(output, record)
            if execution['status'] != 'EXECUTED':
                # Try a stationary sensor capture only. Never open or move an
                # arm after a controller fault just to obtain an after image.
                if execution.get('physical_execution') is True and hasattr(runtime, 'observe_stationary'):
                    try:
                        after = runtime.observe_stationary(folder / 'after')
                        step['after_rgb'] = str(after.image.resolve())
                    except Exception as exc:
                        step['progress_evaluation'] = unknown(f'Fault after-observation/evaluation unavailable: {exc}')
                record['status'] = execution['status']; break
            held_pose = decision['targets'][-1] if runtime.holding else None
            after = runtime.observe(folder / 'after')
            step['after_rgb'] = str(after.image.resolve())
            report(output, record)
            before = after
            if not runtime.holding:
                table_observation = str(after.directory)
            report(output, record)
    except KeyboardInterrupt:
        runtime.stop(); record['status'] = 'INTERRUPTED'
    except Exception as exc:
        runtime.stop(); record.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
    finally:
        record['holding_at_stop'] = runtime.holding
        report(output, record)
        runtime.close()
    return record


def latest_completed(roots):
    candidates = []
    for root in roots:
        for p in Path(root).glob('**/*unfold*/result.json'):
            if json.loads(p.read_text()).get('status') in ('MODEL_REPORTED_COMPLETE', 'FOLD_READY'):
                candidates.append(p.parent)
    if not candidates:
        raise ValueError('No completed unfold found. Supply --baseline with an explicitly identified historical run; never infer completion from EXECUTED.')
    return max(candidates, key=lambda p: p.name)


def baseline_review_answer(prompt, images, folder, model_call=run):
    """Reuse completed matching offline requests; retain failed attempts intact."""
    folder = Path(folder)
    attempts = [folder, *sorted(folder.parent.glob(folder.name + '_retry_*'))]
    for attempt in attempts:
        try:
            status = json.loads((attempt / 'result.json').read_text())
            saved_prompt = (attempt / 'prompt.txt').read_text().strip()
            readonly_prefix = '本次仅做只读分析，不连接或操作机器人，不修改项目文件。\n'
            if (status.get('status') == 'AGENT_FINISHED'
                    and saved_prompt in (prompt.strip(), (readonly_prefix + prompt).strip())):
                value = json.loads((attempt / 'answer.txt').read_text())
                evaluation(value['progress_evaluation'])
                if (value.get('gravity_assisted') is not None
                        and type(value['gravity_assisted']) is not bool):
                    continue
                if not isinstance(value.get('gravity_evidence'), str) or not value['gravity_evidence'].strip():
                    continue
                print(f'Reusing baseline review: {attempt}', flush=True)
                return value
        except (OSError, ValueError, KeyError, TypeError):
            continue
    target = folder
    retry = 1
    while target.exists():
        target = folder.with_name(folder.name + f'_retry_{retry:03d}')
        retry += 1
    print(f'Reviewing baseline: {target}', flush=True)
    return ask(prompt, images, target, model_call)


def freeze_baseline(source, destination, *, evaluate=False, model_call=run, resume=False):
    """Freeze evidence copies in a new directory; never mutate historical data."""
    import shutil
    source = Path(source).resolve(); destination = Path(destination).resolve()
    if not source.is_dir() or source == destination or source in destination.parents:
        raise ValueError('baseline destination must be outside source')
    rounds = sorted(source.glob('round_*'))
    if not rounds:
        raise ValueError('baseline must be an existing unfold run with round_* records')
    destination.mkdir(parents=True, exist_ok=resume)
    files = []
    # Preserve all historical logs, graph files and observations; leave large
    # videos in their original location and hash them for provenance.
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        row = dict(source=str(path), relative=str(relative), sha256=h.hexdigest())
        if path.suffix.lower() not in ('.mp4', '.bag'):
            target = destination / 'evidence' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not resume or hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
                    raise ValueError(f'Existing review evidence differs: {target}')
            else:
                shutil.copyfile(path, target)
            row['snapshot'] = str(target)
        files.append(row)
    frozen = destination / 'evidence'
    steps = []
    for i, folder in enumerate(rounds):
        outcome_path = frozen / folder.name / 'outcome.json'
        if not outcome_path.is_file():
            continue
        outcome = json.loads(outcome_path.read_text())
        if outcome.get('physical_execution') is not True:
            continue
        before = frozen / folder.name / 'before/rgb.png'
        after = frozen / rounds[i + 1].name / 'before/rgb.png' if i + 1 < len(rounds) else None
        progress = unknown('Historical progress not assessed / after image unavailable')
        motion = frozen / folder.name / 'motion.py'
        code = motion.read_text() if motion.exists() else ''
        primitive = 'UNCLASSIFIED'
        if evaluate and before.is_file() and after and after.is_file():
            reviewed = baseline_review_answer(EVALUATOR + '\nHistorical baseline: wrap the above progress JSON in progress_evaluation. '
                           'Also return gravity_assisted (boolean or null) and gravity_evidence (string). '
                           'Use the exact command AND visual evidence to decide whether this was a gravity-assisted '
                           'lift/reveal or lift/drape; ordinary approach/retreat or tabletop drag do not count. '
                           'If uncertain return null. Do not infer task progress from commands. Command:\n' + code,
                           [before, after], destination / f'evaluate_{folder.name}', model_call)
            progress = evaluation(reviewed['progress_evaluation'])
            gravity = reviewed['gravity_assisted']
            if gravity is not None and type(gravity) is not bool:
                raise ValueError('invalid historical gravity annotation')
            if not isinstance(reviewed.get('gravity_evidence'), str) or not reviewed['gravity_evidence'].strip():
                raise ValueError('historical gravity annotation needs evidence')
            primitive = 'LiftAndReveal' if gravity is True else 'OTHER' if gravity is False else 'UNCLASSIFIED'
        steps.append(dict(step=folder.name, chosen_primitive=primitive, execution=outcome,
                          before_rgb=str(before), after_rgb=str(after) if after and after.exists() else None,
                          action_command=code, progress_evaluation=progress))
    m = metrics(steps)
    m['gravity_unclassified_actions'] = sum(s['chosen_primitive'] == 'UNCLASSIFIED' for s in steps)
    if m['gravity_unclassified_actions']:
        m['gravity_assisted_actions'] = None  # Never turn missing annotation into zero.
    if not evaluate or m['unknown_actions']:
        for key in ('no_progress_actions', 'no_progress_ratio', 'structure_revealed'):
            m[key] = None
    first_ready = next((i + 1 for i, s in enumerate(steps)
                        if ready(s['progress_evaluation']['fold_ready']) and s['progress_evaluation']['outcome'] != 'UNKNOWN'), None)
    m['actions_to_fold_ready'] = first_ready
    result_path = source / 'result.json'
    source_status = json.loads(result_path.read_text()).get('status', 'UNKNOWN') if result_path.is_file() else 'NO_FINAL_REPORT'
    record = dict(label='Experience Graph v0', source=str(source), files=files, source_status=source_status,
                  preservation='Existing experience only; no graph synthesis or node taxonomy.',
                  steps=steps, metrics=m, visual_evaluation_performed=evaluate)
    save(destination / 'experience_graph_v0.json', record)
    return record


def finish_baseline_review(record, output, baseline_path):
    """Offline comparison after robot resources are closed; never erase rollout."""
    output = Path(output)
    if record['status'] != 'FOLD_READY' or record.get('holding_at_stop', False):
        record['baseline_review_status'] = 'DEFERRED'
        print('展开尚未完成；保存当前轨迹，不启动 v0 总结对比。', flush=True)
        report(output, record)
        return
    if not record['baseline'].get('visual_evaluation_performed'):
        try:
            print('实机流程已结束并关闭连接；现在继续 v0 离线对比评估。', flush=True)
            review_output = output.with_name(output.name + '_v0_review')
            record['baseline'] = freeze_baseline(Path(baseline_path).parent / 'evidence',
                                                 review_output, evaluate=True, resume=True)
            record['baseline_review_status'] = 'COMPLETED'
        except (Exception, KeyboardInterrupt) as exc:
            record['baseline_review_status'] = 'PENDING'
            record['baseline_review_error'] = f'{type(exc).__name__}: {exc}'
            print('v0 对比评估未完成；新 rollout 已保留：' + record['baseline_review_error'], flush=True)
        report(output, record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze-v0', 'run'))
    parser.add_argument('--baseline', type=Path, help='freeze-v0: historical run; run: frozen experience_graph_v0.json')
    parser.add_argument('--output', type=Path, required=True, help='new directory, never overwritten')
    parser.add_argument('--evaluate', action='store_true', help='freeze-v0: offline visual review (model calls, no hardware)')
    parser.add_argument('--execute', action='store_true', help='explicitly enable physical experiment')
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument('--auto-execute', dest='auto_execute', action='store_true',
                           help='与原 unfold 一致：预览后自动执行（默认）')
    execution.add_argument('--manual-execute', dest='auto_execute', action='store_false',
                           help='显式要求等待 Viser 点击执行')
    parser.set_defaults(auto_execute=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/dual_arm.local.json')
    parser.add_argument('--viser-port', type=int, default=8088)
    parser.add_argument('--max-actions', type=int, default=30)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output exists; choose a new directory')
    if args.mode == 'freeze-v0':
        from cloth_agent.run_storage import storage_roots
        source = args.baseline or latest_completed(storage_roots(ROOT))
        freeze_baseline(source, args.output, evaluate=args.evaluate)
        return 0
    if not args.execute:
        parser.error('run requires --execute; no robot runtime was constructed')
    if args.max_actions < 1 or not args.baseline or not args.baseline.is_file():
        parser.error('run requires positive --max-actions and frozen --baseline JSON')
    baseline = json.loads(args.baseline.read_text())
    if baseline.get('label') != 'Experience Graph v0':
        parser.error('baseline must be frozen with freeze-v0')
    for row in baseline['files']:
        path = Path(row.get('snapshot', row['source']))
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
        if h.hexdigest() != row['sha256']:
            parser.error(f'baseline evidence changed: {path}')
    from .gravity_runtime import GravityRuntime
    from cloth_agent.trajectory_preview import TrajectoryPreview
    runtime = GravityRuntime(args.config)
    runtime.preview = TrajectoryPreview(runtime.config, port=args.viser_port, wait=not args.auto_execute)
    print('执行方式：与原 unfold 一致，自动执行' if args.auto_execute else '执行方式：等待 Viser 点击', flush=True)
    old = signal.getsignal(signal.SIGTERM)
    def terminate(*_):
        raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM, terminate)
    try:
        record = rollout(runtime, args.output, baseline, max_actions=args.max_actions)
        print(json.dumps(record['metrics'], ensure_ascii=False, indent=2))
    finally:
        runtime.close(); runtime.preview.close(); signal.signal(signal.SIGTERM, old)
    finish_baseline_review(record, args.output, args.baseline)
    return 0 if record['status'] == 'FOLD_READY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
