"""Online ranking correction with immutable pre-feedback evaluation and checkpoints.

R(x)=10*w.x; w starts at zero. Only past human orders train this model.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np

from . import action_preference as pref, residual_features

DEFAULT_STATE = Path('data/experiments/one_shot_preference/residual.json')
LEARNING = dict(score_scale=10., l2=.05, steps=40, replay_rounds=100)


def load_state(path, alpha):
    with pref.state_lock(path):
        if Path(path).exists():
            state = pref.io.read(path)
            if state['schema_version'] != 1 or state['learning'] != LEARNING or state['alpha'] != alpha:
                raise ValueError('residual protocol/fixed alpha mismatch; use a separate state file')
            if state['feature_version'] != residual_features.VERSION:
                raise ValueError('residual feature version mismatch')
            checkpoints = state['checkpoints']
            if len(checkpoints) != state['revision']+1 or [c['revision'] for c in checkpoints] != list(range(len(checkpoints))):
                raise ValueError('invalid residual checkpoint history')
            for c in checkpoints:
                w = np.asarray(c['weights'], float)
                if w.shape != (residual_features.DIM,) or not np.isfinite(w).all():
                    raise ValueError('invalid residual checkpoint')
            return state
        pref.ranked([], alpha)
        state = dict(schema_version=1, feature_version=residual_features.VERSION, identity=None,
            alpha=alpha, revision=0, learning=LEARNING, feedback={},
            checkpoints=[dict(revision=0, weights=[0.]*residual_features.DIM, trained_through=None)])
        pref.atomic_save(path, state)
        return state


def prepare(rows, state, features):
    if state['identity'] is not None and state['identity'] != features['identity']:
        raise ValueError('DINO feature identity changed')
    if features['identity']['version'] != residual_features.VERSION or features['identity']['dimension'] != residual_features.DIM:
        raise ValueError('invalid feature identity')
    base = pref.ranked(rows, state['alpha'])
    if set(features['features']) != {r['id'] for r in rows}:
        raise ValueError('features do not cover candidates exactly')
    vectors = np.asarray([features['features'][r['id']] for r in base])
    if vectors.shape != (len(base), residual_features.DIM) or not np.isfinite(vectors).all():
        raise ValueError('invalid action vectors')
    predictions = []
    for checkpoint in state['checkpoints']:
        scores = np.array([r['P'] for r in base])+LEARNING['score_scale']*(vectors @ np.array(checkpoint['weights']))
        scores = {r['id']: float(s) for r, s in zip(base, scores)}
        predictions.append(dict(revision=checkpoint['revision'], trained_through=checkpoint['trained_through'],
            scores=scores, order=sorted(scores, key=lambda k: (-scores[k], k))))
    current = predictions[-1]['scores']
    order = sorted([dict(r, base=r['P'], correction=current[r['id']]-r['P'], S=current[r['id']],
                        P=current[r['id']]) for r in base], key=lambda r: (-r['S'], r['id']))
    context = dict(mode='residual', revision=state['revision'], alpha_fixed=state['alpha'],
        feature_identity=features['identity'], features=features['features'],
        image_sha256=features['image_sha256'], checkpoints=predictions,
        previously_labeled_image=any(e['image_sha256']==features['image_sha256'] for e in state['feedback'].values()),
        previous_evaluation=(list(state['feedback'].values())[-1]['evaluation'] if state['feedback'] else None),
        evaluation_scope='only displayed Top10; not unshown actions or physical unfolding success',
        predicted_at=pref.io.utc())
    return order, context


def validate_order(order, ids):
    if not isinstance(order, list) or any(not isinstance(i, str) for i in order):
        raise ValueError('human_order must be a complete list of action IDs')
    if len(order) != len(ids) or set(order) != set(ids):
        raise ValueError('human_order must contain each displayed action exactly once')
    return order


def validate_groups(groups, ids):
    if not isinstance(groups, list) or not groups or any(not isinstance(g, list) or not g for g in groups):
        raise ValueError('human_groups must be nonempty ordered groups')
    validate_order([item for group in groups for item in group], ids)
    # Display order inside a group carries no preference, including in idempotence checks.
    return [sorted(group) for group in groups]


def feedback_groups(choice, ids):
    if 'human_groups' in choice:
        return validate_groups(choice['human_groups'], ids)
    # Old saved complete-order reviews remain readable as singleton groups.
    return [[i] for i in validate_order(choice.get('human_order'), ids)]


def metrics(order, human, groups=None):
    groups = [[i] for i in human] if groups is None else groups
    position = {k:i for i,k in enumerate(order)}
    pairs = [(a,b) for i,g in enumerate(groups) for later in groups[i+1:] for a in g for b in later]
    total = len(pairs)
    concordant = sum(position[a]<position[b] for a,b in pairs)
    accuracy = concordant/total if total else None
    all_pairs = len(human)*(len(human)-1)//2
    # System order is strict; human within-group ties are excluded from accuracy.
    tau_b = (2*concordant-total)/np.sqrt(total*all_pairs) if total else None
    return dict(pairwise_accuracy=accuracy, kendall_tau=float(tau_b) if tau_b is not None else None,
                kendall_variant='tau-b, human ties', top1_agreement=order[0] in groups[0],
                human_top1_rank=min(position[i]+1 for i in groups[0]),
                pair_count=total, ignored_tie_pairs=all_pairs-total, informative=total>0)


def evaluate(manifest, human, groups=None):
    context = manifest['ranking_context']
    ids = {r['id'] for r in manifest['displayed']}
    validate_order(human, ids)
    groups = [[i] for i in human] if groups is None else validate_groups(groups, ids)
    evaluations = []
    for checkpoint in context['checkpoints']:
        order = [i for i in checkpoint['order'] if i in ids]
        evaluations.append(dict(revision=checkpoint['revision'], trained_through=checkpoint['trained_through'],
            order=order, **metrics(order, human, groups)))
    current, base = evaluations[-1], evaluations[0]
    previous = evaluations[-2] if len(evaluations)>1 else None
    informative = current['pair_count']>0
    return dict(recorded_at=pref.io.utc(), before_training=True, human_order=human, human_groups=groups,
        system_order=[r['id'] for r in manifest['displayed']],
        baseline=base, current=current, previous=previous, checkpoints=evaluations,
        delta_vs_base=current['pairwise_accuracy']-base['pairwise_accuracy'] if informative else None,
        delta_vs_previous=None if previous is None or not informative else current['pairwise_accuracy']-previous['pairwise_accuracy'],
        eligible_new_image=not context['previously_labeled_image'],
        scope=context['evaluation_scope'],
        limitation='sequential on-policy states; exact repeated RGB excluded, not independent held-out trials')


def _pairs(events):
    vectors, offsets = [], []
    for e in events:
        rows = {r['id']:r for r in e['training_rows']}
        groups = e.get('human_groups', [[i] for i in e['human_order']])
        for i, group in enumerate(groups):
            for later in groups[i+1:]:
                for a in group:
                    for b in later:
                        vectors.append(np.array(rows[a]['x'])-np.array(rows[b]['x']))
                        offsets.append((rows[a]['base']-rows[b]['base'])/LEARNING['score_scale'])
    return np.array(vectors), np.array(offsets)


def train(weights, events):
    """Regularized pairwise logistic loss, replaying the last 100 reviewed states."""
    x, offsets = _pairs(events[-LEARNING['replay_rounds']:])
    if not len(offsets):
        return list(weights),dict(skipped=True, reason='no cross-group preference pairs',pair_count=0)
    w = np.array(weights, float)
    def objective(v):
        margins = offsets+x@v
        return float(np.logaddexp(0, -margins).mean()+LEARNING['l2']/2*(v@v))
    before = objective(w)
    # Conservative Lipschitz bound avoids unstable steps on small human datasets.
    rate = 1/(.25*np.square(x).sum(axis=1).mean()+LEARNING['l2'])
    for _ in range(LEARNING['steps']):
        margins = offsets+x@w
        probability = np.exp(-np.logaddexp(0, margins))
        gradient = -(x.T@probability)/len(x)+LEARNING['l2']*w
        new = w-rate*gradient
        if not np.isfinite(new).all() or objective(new)>objective(w)+1e-10:
            raise ValueError('residual optimization failed; state not updated')
        w = new
    return w.tolist(), dict(replay_loss_before=before, replay_loss_after=objective(w),
        replay_rounds=min(len(events), LEARNING['replay_rounds']), pair_count=len(x),
        meaning='training fit only, not evidence of future-state improvement')


def apply_feedback(path, manifest, choice, folder):
    """Write pre-update evidence first; commit weights+feedback atomically and once."""
    folder = Path(folder)
    context = manifest['ranking_context']
    ids = {r['id'] for r in manifest['displayed']}
    groups = feedback_groups(choice, ids)
    human = [i for group in groups for i in group]
    if choice['candidate_id'] not in ids:
        raise ValueError('chosen action not displayed')
    key = manifest['review_id']
    label = dict(human_order=human, chosen=choice['candidate_id'])
    if 'human_groups' in choice: label = dict(human_groups=groups, chosen=choice['candidate_id'])
    digest = hashlib.sha256(json.dumps(label, sort_keys=True).encode()).hexdigest()
    with pref.state_lock(path):
        state = pref.io.read(path)
        if key in state['feedback']:
            event = state['feedback'][key]
            if event['label_digest'] != digest:
                raise ValueError('review already has another ranking or choice')
            pref.atomic_save(folder/'pre_update_evaluation.json', event['evaluation'])
            pref.atomic_save(folder/'residual_feedback.json', event)
            return event
        if state['revision'] != context['revision'] or state['alpha'] != manifest['alpha']:
            raise ValueError('stale residual feedback rejected')
        if state['identity'] is not None and state['identity'] != context['feature_identity']:
            raise ValueError('feature identity changed since review')
        evaluation = evaluate(manifest, human, groups)
        # This write must succeed BEFORE optimization or persistent state mutation.
        pref.atomic_save(folder/'pre_update_evaluation.json', evaluation)
        event = dict(review_id=key, label_digest=digest, at=pref.io.utc(), chosen=choice['candidate_id'],
            human_order=human, human_groups=groups, image_sha256=context['image_sha256'], revision_before=state['revision'],
            revision_after=state['revision']+1, alpha=state['alpha'], evaluation=evaluation,
            training_rows=[dict(id=r['id'], base=r['base'], x=context['features'][r['id']])
                           for r in manifest['displayed']])
        started = time.monotonic()
        if evaluation['current']['pair_count']:
            weights, fit = train(state['checkpoints'][-1]['weights'], [*state['feedback'].values(), event])
        else:
            # All tied: no preference supervision, no replay-only update or L2 shrinkage.
            weights = state['checkpoints'][-1]['weights'][:]
            fit = dict(skipped=True, reason='all displayed actions tied; weights unchanged',pair_count=0)
        event.update(training_s=time.monotonic()-started, training_fit=fit)
        state.update(identity=context['feature_identity'], revision=state['revision']+1,
                     feedback_protocol='ordered_groups_v2')
        state['checkpoints'].append(dict(revision=state['revision'], weights=weights, trained_through=key))
        state['feedback'][key] = event
        pref.atomic_save(path, state)
        pref.atomic_save(folder/'residual_feedback.json', event)
    return event


def write_progress(state, folder):
    """Human-readable running evaluation on later states, never post-fit scores."""
    lines = ['# Rθ 后续状态评估', '', f"固定 α = {state['alpha']}；模型版本 {state['revision']}。",
        '每行均为接收本轮反馈后、训练本轮之前的评估。只比较展示Top10中的跨组偏好；组内并列不训练、不计高低。',
        '全部同组时不更新权重，排序一致率无定义，不计入改善汇总；执行选择仍可独立提交。',
        '0号模型就是 base。上一版指尚未学习最近一次反馈的模型；当前版包含此前所有反馈。', '',
        '|反馈序号|本轮训练前版本|新RGB|base一致率|上一版一致率|当前版一致率|对base变化|对上一版变化|人工第一在系统中的名次|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    eligible = []
    for i, event in enumerate(state['feedback'].values(), 1):
        e = event['evaluation']; previous = e['previous']
        fmt = lambda x: '—' if x is None else f'{x:.1%}'
        lines.append(f"|{i}|{e['current']['revision']}|{'是' if e['eligible_new_image'] else '重复，不计入'}|"
            f"{fmt(e['baseline']['pairwise_accuracy'])}|{fmt(previous['pairwise_accuracy'] if previous else None)}|"
            f"{fmt(e['current']['pairwise_accuracy'])}|{fmt(e['delta_vs_base'])}|{fmt(e['delta_vs_previous'])}|"
            f"{e['current']['human_top1_rank']}|")
        if e['eligible_new_image'] and e['current']['revision']>0 and e['current']['pairwise_accuracy'] is not None:
            eligible.append(e)
    lines += ['', f'包含已有学习结果的新图评估数：{len(eligible)}。']
    if eligible:
        lines += [f"当前模型相对base的平均成对一致率变化：{np.mean([e['delta_vs_base'] for e in eligible]):+.1%}。"]
    else:
        lines += ['尚无后续新状态证据，不能判断修正是否有用。']
    lines += ['完整历史模型在每张新图上的排名和指标保存在各轮 pre_update_evaluation.json 的 checkpoints 中。',
              '这是同一实际轨迹上对不同评分器的比较，不是不同执行策略的随机对照实验。']
    (Path(folder)/'learning_progress.md').write_text('\n'.join(lines)+'\n')


def render_review(folder, manifest):
    # Features/checkpoints stay server-side; the browser receives only display data.
    context = manifest['ranking_context']
    payload = {k:manifest[k] for k in ('review_id', 'token', 'simulated', 'alpha', 'displayed')}
    payload.update(revision=context['revision'], previous_evaluation=context.get('previous_evaluation'))
    template = Path(__file__).with_name('residual_review.html').read_text()
    template = template.replace('__PAYLOAD__', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c'))
    (Path(folder)/'index.html').write_text(template)
