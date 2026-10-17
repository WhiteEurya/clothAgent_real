"""Independent arm7 one-shot Seed -> Compile Graph -> Graph Replay experiment.

Importing this module does not import a robot runtime or connect to hardware.
seed/replay are physical operations; compile only invokes the multimodal model.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import signal
from pathlib import Path
import time

import jsonschema
from .discovery_data import canonical, digest, file_hash, read, save
from .discovery_goal import task_contract
from .graph_policy_schema import COMMON, PROMPTS, SCHEMAS
from .model_errors import ModelInvocationError
from .graph_evidence import builder_steps, compact_evidence, compiler_steps

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'graph-policy-v1'


def frozen(path, value):
    save(path, value, immutable=True)
    save(Path(str(path) + '.sha256.json'), {'sha256': file_hash(path)}, immutable=True)


def load_frozen(path):
    path = Path(path)
    if read(str(path) + '.sha256.json').get('sha256') != file_hash(path):
        raise ValueError(f'frozen artifact hash mismatch: {path}')
    return read(path)


def evidence_file(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': file_hash(path)} if path.is_file() else None


def checked_file(record):
    path = Path(record['path'])
    if file_hash(path) != record['sha256']:
        raise ValueError(f'evidence modified: {path}')
    return path


class Requests:
    """Fresh isolated multimodal calls with bounded repair, hashes and local caching."""
    def __init__(self, output, backend, *, no_cache=False, max_calls=64,
                 schemas=None, prompts=None, common=None, protocol=PROTOCOL):
        self.output = Path(output); self.backend = backend
        self.schemas = SCHEMAS if schemas is None else schemas
        self.prompts = PROMPTS if prompts is None else prompts
        self.common = COMMON if common is None else common
        self.protocol = protocol
        self.no_cache = no_cache; self.max_calls = max_calls
        self.model_calls = 0; self.cache_hits = 0; self.elapsed = 0.
        self.tool_calls = 0; self.visual_tool_calls = 0; self.completed_turns = 0
        self.tool_metrics_available = True

    def ask(self, kind, payload, images, validator=None):
        schema = self.schemas[kind]
        image_rows = [{'id': key, **evidence_file(path)} for key, path in images]
        prompt = self.common + '\n' + self.prompts[kind] + '\n' + canonical(dict(
            payload=payload, image_ids=[x['id'] for x in image_rows], schema=schema))
        key = digest(dict(protocol=self.protocol, model=self.backend.model, kind=kind,
                          prompt=prompt, images=[(x['id'], x['sha256']) for x in image_rows]))
        cache = self.output / 'cache' / f'{key}.json'
        def validate(value):
            jsonschema.validate(value, schema)
            if validator: validator(value)
        if cache.is_file() and not self.no_cache:
            value = read(cache); validate(value); self.cache_hits += 1
            self._log(dict(kind=kind, cache_key=key, cache_hit=True, model=self.backend.model,
                           images=image_rows, structured_response=value))
            return value
        for retry in range(2):
            if len(prompt) > 800_000:
                raise ValueError(f'Graph request text too large before model call: {len(prompt)} chars; limit 800000. No truncation or model call performed.')
            if self.model_calls >= self.max_calls:
                raise RuntimeError('graph experiment model-call budget exhausted')
            self.model_calls += 1
            row = dict(kind=kind, cache_key=key, cache_hit=False, model=self.backend.model,
                       images=image_rows, prompt_hash=digest(prompt), schema_hash=digest(schema),
                       retry_count=retry, prompt=prompt)
            started = time.monotonic()
            try:
                response = self.backend.invoke(prompt, [p for _, p in images], schema)
                self.tool_metrics_available &= 'tool_calls' in response
                self.tool_calls += response.get('tool_calls', 0)
                self.visual_tool_calls += response.get('visual_tool_calls', 0)
                self.completed_turns += response.get('completed_turns', 0)
                row.update(usage=response.get('usage'), raw_response=response['text'],
                           tool_calls=response.get('tool_calls'), visual_tool_calls=response.get('visual_tool_calls'),
                           completed_turns=response.get('completed_turns'))
                value = json.loads(response['text']); validate(value)
                row['structured_response'] = value
            except (ValueError, jsonschema.ValidationError) as exc:
                row['error'] = str(exc)
                if retry: raise
                prompt += '\nFormat/contract repair only. Return complete JSON. Error: ' + str(exc)
                continue
            except ModelInvocationError as exc:
                row.update(error=str(exc),model_failure=exc.details,model_retryable=exc.retryable)
                if retry or not exc.retryable:raise
                row['retry_reason']='MODEL_INVOCATION_ONLY_NO_PHYSICAL_DISPATCH'
                continue
            except BaseException as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'; raise
            finally:
                row['elapsed_s'] = time.monotonic() - started
                self.elapsed += row['elapsed_s']; self._log(row)
            save(cache, value)
            return value
        raise RuntimeError('unreachable')

    def _log(self, row):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / 'requests.jsonl').open('a') as f:
            f.write(canonical(row) + '\n')

    def metrics(self):
        return dict(model_calls=self.model_calls, cache_hits=self.cache_hits,
                    tool_calls=self.tool_calls if self.tool_metrics_available else None,
                    visual_tool_calls=self.visual_tool_calls if self.tool_metrics_available else None,
                    completed_model_turns=self.completed_turns, reasoning_time_s=self.elapsed,
                    metric_definition='Codex invocations and completed tool events; wall time includes model latency; attached RGBs are not tool calls; internal API calls unavailable')


def command(source):
    from cloth_agent.codex_motion import parse_code
    return {'source': source, 'operations': [dict(method=n, arguments=a)
            for n, a in parse_code(source, allow_complete=True, allow_grasp=True)],
            'units': 'world_m_rad'}


def collect_evidence(folder):
    """Native feedback is factual controller evidence, not cloth ground truth."""
    folder = Path(folder)
    files = sorted(set(folder.glob('gripper_*.json')) | set(folder.glob('*recording*.json')))
    files += [folder / n for n in ('native_plan.json', 'grasp_height_policy.json',
                                   'grasp_orientation_policy.json', 'planning_stats.json',
                                   'execution_events.json')]
    return {p.name: dict(artifact=evidence_file(p), content=read(p),
                         semantics='controller/host record; not proof of cloth acquisition')
            for p in files if p.is_file()}


def execute_recorded(runtime, source, before, folder):
    """Copy just this dispatch's controller events; do not change execution."""
    coordinator = getattr(runtime, 'coordinator', None)
    event_path = Path(coordinator.directory) / 'events.jsonl' if coordinator else None
    offset = event_path.stat().st_size if event_path and event_path.is_file() else 0
    try:
        return runtime.execute(source, before, folder)
    finally:
        rows = []
        if event_path and event_path.is_file():
            with event_path.open('rb') as f:
                f.seek(offset)
                for line in f:
                    try: rows.append(json.loads(line))
                    except ValueError: rows.append({'unavailable': 'incomplete controller event'})
        save(Path(folder) / 'execution_events.json', dict(events=rows,
            unavailable_reason=None if rows else 'no controller events available',
            semantics='controller commands/feedback; not cloth-state ground truth'))


def observation_geometry(observation):
    meta = getattr(observation, 'meta', {})
    return {k: meta[k] for k in ('frame_id', 'calibration_id', 'intrinsics',
        'world_from_camera_mm', 'camera', 'joints_deg', 'world_tilt_correction',
        'support_height_patch', 'corrected_world_from_camera_affine_mm') if k in meta}


def planner_metrics(seed):
    calls = tools = visual = 0; elapsed = 0.; turns = 0
    for folder in sorted(Path(seed).glob('round_*/planner')):
        calls += 1; result = read(folder / 'result.json')
        elapsed += result.get('duration_s', 0.); turns += result.get('completed_turns', 0)
        events = folder / 'events.jsonl'
        if not events.exists(): continue
        for line in events.read_text().splitlines():
            try: event = json.loads(line)
            except ValueError: continue
            item = event.get('item', {})
            if event.get('type') == 'item.completed' and item.get('type') in (
                    'command_execution', 'mcp_tool_call', 'web_search', 'image_view', 'tool_call'):
                tools += 1
                if item.get('type') == 'image_view' or 'view_image' in str(item.get('tool', '')):
                    visual += 1
    return dict(model_calls=calls, completed_model_turns=turns, tool_calls=tools,
                visual_tool_calls=visual, reasoning_time_s=elapsed,
                metric_definition='Codex process invocations and completed tool events; internal model requests unavailable')


class SeedRecorder:
    """Proxy the original loop, retaining observations while deferring its close."""
    def __init__(self, runtime):
        self.runtime = runtime; self.observations = []
    def __getattr__(self, name): return getattr(self.runtime, name)
    def observe(self, folder):
        obs = self.runtime.observe(folder); self.observations.append(obs); return obs
    def execute(self, source, before, folder):
        return execute_recorded(self.runtime, source, before, folder)
    def close(self): pass


def seed(output, runtime, *, subgoal='full_fold', max_rounds=None, model_call=None, timeout_s=900):
    from cloth_agent.codex_loop import loop, DEFAULT_REFERENCES
    from cloth_agent.codex_astra_pipeline import run
    if subgoal not in ('full_fold', 'right_sleeve'): raise ValueError('unsupported task scope')
    max_rounds = max_rounds if max_rounds is not None else (30 if subgoal == 'full_fold' else 8)
    if max_rounds < 1: raise ValueError('max_rounds must be positive')
    if subgoal == 'full_fold':
        references = DEFAULT_REFERENCES
        definition = dict(subgoal='full_fold', objective='Complete the entire arm7 folding demonstration in order',
            reference_stages=[role for _, role in references],
            side_convention='Mentally rotate collar above hem without mirroring; left/right are viewer sides.',
            completion_criterion='The entire garment must reach the final demonstration fold, including sleeves, '
                'both torso sides and the bottom hem. Finishing a sleeve or any intermediate stage is NOT task completion.',
            interpretation='These are demonstration goals, not predefined graph nodes. Infer nodes and edges only from actual seed evidence.')
        goal = dict(status='AVAILABLE', goal=definition, sha256=digest(definition),
                    sources=['cloth_agent.codex_loop.DEFAULT_REFERENCES'])
        task = ('只使用arm7按全部参考图顺序完成整件衣服的折叠：左袖、右袖、左侧衣身、右侧衣身、下摆。'
                '完成一个袖子或某个中间步骤不能输出complete；继续根据新观测处理后续步骤。'
                '只有整件衣服达到最后一张参考图的折叠终态，才输出motion.complete()。'
                '自行决定每个阶段需要多少轮动作，不读取已有graph或其他实验策略。\n' + canonical(definition))
    else:
        goal = task_contract('right_sleeve')
        if goal['status'] != 'AVAILABLE': raise ValueError('right_sleeve goal unavailable')
        references = [(ROOT / 'data/reference/fold_states/active/state_02_right_sleeve.png',
                       '当前 right_sleeve 子目标的参考形状')]
        task = ('本次唯一任务是 right_sleeve。仅完成右袖子目标，不继续折叠衣身或下摆。'
                '此任务中的叠好/complete 仅指下述右袖目标完成。不读取已有graph或其他实验策略。\n' + canonical(goal['goal']))
    for path, _ in references:
        if not Path(path).is_file(): raise ValueError(f'reference image missing: {path}')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    recorder = SeedRecorder(runtime)
    save(output / 'experiment.json', dict(protocol=PROTOCOL, subgoal=subgoal,
         task_contract=goal, max_rounds=max_rounds, runtime='Arm7Runtime',
         grasp_offset_mm=runtime.grasp_offset_mm, graph_supplied=False))
    result = None; after_error = None
    try:
        result = loop(task, recorder, output / 'seed', max_rounds=max_rounds,
                      model_call=model_call or run, timeout_s=timeout_s,
                      references=references)
        # Normal next-round capture supplies after. Only a bounded run ending in
        # an executed last action needs one extra terminal capture. Never move after a fault.
        rounds = sorted((output / 'seed').glob('round_*'))
        if rounds and result['status'] == 'ROUND_LIMIT' and read(rounds[-1] / 'outcome.json').get('status') == 'EXECUTED':
            try: recorder.observe(output / 'seed' / 'terminal_after')
            except Exception as exc: after_error = str(exc)
    finally:
        try: runtime.close()
        finally: export_seed(output, recorder.observations, result, after_error)
    return result


def export_seed(output, observations, result, after_error=None):
    output = Path(output); config = read(output / 'experiment.json')
    obs_by_dir = {str(o.directory.resolve()): o for o in observations}
    obs_ids = {p: f'o{i:03d}' for i, p in enumerate(obs_by_dir)}
    evidence = {}
    for directory, o in obs_by_dir.items():
        evidence[obs_ids[directory]] = dict(rgb=evidence_file(o.image),
            observation=evidence_file(o.directory / 'observation.json'),
            surface=evidence_file(o.directory / 'world_xyz_mm.npy'),
            geometry=observation_geometry(o))
    folders = sorted((output / 'seed').glob('round_*')); steps = []
    for index, folder in enumerate(folders):
        before = str((folder / 'before').resolve())
        if before not in obs_ids: continue
        after = None
        if index + 1 < len(folders): after = obs_ids.get(str((folders[index+1] / 'before').resolve()))
        if index == len(folders)-1: after = obs_ids.get(str((output / 'seed/terminal_after').resolve()))
        outcome = read(folder / 'outcome.json')
        if not outcome:
            events = read(folder / 'execution_events.json').get('events', [])
            dispatched = any(e.get('event') in ('astra_move_start', 'astra_gripper_start') for e in events)
            outcome = dict(status='UNAVAILABLE', physical_execution=dispatched,
                provenance='Missing outcome.json; dispatch flag recovered from execution_events.json',
                completion_unknown=True,
                completed_commands=sum(e.get('event') == 'astra_command_complete' for e in events))
        motion = folder / 'motion.py'; source = motion.read_text() if motion.is_file() else None
        effective = folder / 'effective_motion.py'
        if outcome['status'] in ('MODEL_REPORTED_COMPLETE', 'BLOCKED'):
            after = obs_ids[before]  # no manipulation performed
        steps.append(dict(step_id=f's{index:03d}', before_observation=obs_ids[before],
            goal=config['task_contract']['goal'], planner_output=source,
            commanded_action=command(effective.read_text() if effective.is_file() else source) if source else None,
            planner_artifact=evidence_file(folder / 'planner/answer.txt'),
            actual_execution=outcome, intermediate_evidence=collect_evidence(folder),
            after_observation=after, outcome={'controller': outcome,
                'physical_cloth_outcome': 'UNKNOWN; requires visual/evidence assessment',
                'after_unavailable_reason': None if after else (after_error or 'no safe later observation available')}))
    trajectory = dict(protocol=PROTOCOL, goal=config['task_contract']['goal'],
        grasp_offset_mm=config['grasp_offset_mm'], observations=evidence, steps=steps,
        result=result, metrics=planner_metrics(output / 'seed'))
    # Preserve controller events and recorded intermediate video as auditable artifacts.
    trajectory['controller_artifacts'] = [evidence_file(p) for p in (output / 'seed').rglob('*.jsonl')
                                        if 'planner' not in p.parts and 'planning_trace' not in p.name]
    trajectory['video_artifacts'] = [evidence_file(p) for p in (output / 'seed').rglob('*.mp4')]
    trajectory['goal_references'] = [dict(id=f'goal_reference_{i+1}', role=row['role'],
        artifact=evidence_file(row['path'])) for i,row in enumerate(read(output / 'seed/references.json') or [])]
    trajectory['coverage'] = dict(scope=config['subgoal'],
        status='SEED_PLANNER_REPORTED_COMPLETE' if result and result.get('status') == 'MODEL_REPORTED_COMPLETE' else 'PARTIAL_TRAJECTORY',
        physical_attempt_count=sum(s['actual_execution'].get('physical_execution') is True for s in steps),
        limitation='Planner completion is not independent verification; unobserved stages must not be invented.')
    trajectory['limitations'] = ['Video is recorded but not automatically interpreted by Graph Builder.',
                               'Controller completion does not establish cloth acquisition or task success.']
    frozen(output / 'seed_trajectory.json', trajectory)
    (output / 'seed_trajectory.md').write_text('# Seed trajectory\n\n'
        + '\n'.join(f"- {s['before_observation']} -- {s['step_id']} ({s['actual_execution']['status']}) --> {s['after_observation'] or 'UNAVAILABLE'}" for s in steps)
        + '\n\nController completion and planner completion claims are not physical cloth ground truth.\n')


def graph_validation(graph, trajectory, *, excluded_steps=()):
    jsonschema.validate(graph, SCHEMAS['compile'])
    nodes = {n['node_id']: n for n in graph['nodes']}
    if not nodes or len(nodes) != len(graph['nodes']): raise ValueError('duplicate/empty nodes')
    if len({e['edge_id'] for e in graph['edges']}) != len(graph['edges']): raise ValueError('duplicate edge IDs')
    steps = {s['step_id']: s for s in trajectory['steps'] if s['actual_execution'].get('physical_execution')}
    if set(excluded_steps) - set(steps): raise ValueError('unknown excluded step')
    steps = {k:v for k,v in steps.items() if k not in excluded_steps}
    seen = set(); obs = trajectory['observations']
    for n in nodes.values():
        if set(n['representative_observations']) - set(obs):
            raise ValueError('node must cite known seed observations')
        if not n['representative_observations']:
            incoming = [e for e in graph['edges'] if e['target_node'] == n['node_id']]
            outgoing = [e for e in graph['edges'] if e['source_node'] == n['node_id']]
            if (not incoming or outgoing or n['confidence'] != 0
                    or not n['observation_condition']['uncertainty_conditions']
                    or any(e['seed_step_id'] not in steps
                           or steps[e['seed_step_id']]['after_observation'] is not None for e in incoming)):
                raise ValueError('unobserved node must be an uncertain zero-confidence sink for missing after evidence')
        elif not n['observation_condition']['positive_evidence']:
            raise ValueError('observed node needs positive evidence')
    for e in graph['edges']:
        if e['source_node'] not in nodes or e['target_node'] not in nodes: raise ValueError('unknown edge node')
        s = steps.get(e['seed_step_id'])
        if not s or e['seed_step_id'] in seen: raise ValueError('edge must cite one unique physical seed attempt')
        seen.add(e['seed_step_id'])
        if e['seed_action'] != dict(exact_command=s['commanded_action'], source_observation=s['before_observation']):
            raise ValueError('seed exact_command/source mismatch')
        if s['before_observation'] not in nodes[e['source_node']]['representative_observations']:
            raise ValueError('source node must include actual before observation')
        if not s['after_observation'] and nodes[e['target_node']]['representative_observations']:
            raise ValueError('missing after evidence requires an unobserved target sink')
        if s['after_observation'] and s['after_observation'] not in nodes[e['target_node']]['representative_observations']:
            raise ValueError('target node must include actual after observation')
        allowed = set(obs) | set(steps)
        if not e['outcome_evidence'] or set(e['outcome_evidence']) - allowed: raise ValueError('unknown outcome evidence')
        execution_status = s['actual_execution']['status']
        failure = execution_status == 'EXECUTION_FAILED' or e['seed_outcome'] == 'FAILURE'
        if execution_status not in ('EXECUTED', 'EXECUTION_FAILED') and e['seed_outcome'] in ('SUCCESS','PROGRESS'):
            raise ValueError('unknown execution outcome cannot establish seed SUCCESS')
        if failure and (e['generalization_status'] != 'OBSERVED_FAILURE' or e['seed_outcome'] != 'FAILURE'):
            raise ValueError('failed seed edge must be OBSERVED_FAILURE')
        if not failure and e['generalization_status'] != 'ONE_SHOT_HYPOTHESIS': raise ValueError('invalid one-shot status')
    if seen != set(steps): raise ValueError('graph must cover every physical seed attempt')
    edges_by_step = {e['seed_step_id']: e for e in graph['edges']}
    sequence = list(steps.values())
    for previous, current in zip(sequence, sequence[1:]):
        if previous['after_observation'] and previous['after_observation'] == current['before_observation']:
            if edges_by_step[previous['step_id']]['target_node'] != edges_by_step[current['step_id']]['source_node']:
                raise ValueError('consecutive observed transitions must share a graph node')


def validate_replay_graph(graph, trajectory):
    review = graph.get('user_review')
    excluded = []
    if review:
        source = load_frozen(review['source_graph'])
        if file_hash(review['source_graph']) != review['source_sha256'] or source['seed_trajectory_hash'] != graph['seed_trajectory_hash']:
            raise ValueError('review source provenance mismatch')
        if review.get('authority') != 'USER' or not review.get('instruction'):
            raise ValueError('review requires explicit user instruction')
        excluded = review['excluded_seed_steps']
        previous = {e['seed_step_id'] for e in source['edges']}
        current = {e['seed_step_id'] for e in graph['edges']}
        if previous - current != set(excluded) or current - previous:
            raise ValueError('review exclusion coverage mismatch')
    elif graph.get('terminal_node_ids'):
        raise ValueError('explicit terminals require user review provenance')
    graph_validation({k:graph[k] for k in SCHEMAS['compile']['properties']},trajectory,excluded_steps=excluded)
    nodes = {n['node_id']:n for n in graph['nodes']}
    for terminal in graph.get('terminal_node_ids', []):
        if terminal not in nodes or not nodes[terminal]['representative_observations']:
            raise ValueError('terminal must have observed evidence')
        if any(e['source_node'] == terminal for e in graph['edges']):
            raise ValueError('terminal cannot have outgoing edges')


def goal_images(record):
    return [(ref['id'], checked_file(ref['artifact'])) for ref in record.get('goal_references', [])]


def compile_graph(run, requests):
    run = Path(run); trajectory = load_frozen(run / 'seed_trajectory.json')
    graph_path = run / 'graph_v1.json'
    if graph_path.exists():
        graph = load_frozen(graph_path)
        if graph['seed_trajectory_hash'] != digest(trajectory): raise ValueError('graph seed hash mismatch')
        from .graph_template_compiler import compile_policy
        compile_policy(graph_path, run / 'compiled_policy')
        return graph
    physical = [s for s in trajectory['steps'] if s['actual_execution'].get('physical_execution')]
    if not physical: raise ValueError('seed has no physical attempts to compile')
    images = [(key, checked_file(value['rgb'])) for key, value in trajectory['observations'].items()]
    images += goal_images(trajectory)
    steps = compiler_steps(trajectory['steps'])
    save(run / 'compile/input_projection.json', dict(version='visual-compiler-telemetry-v2',
        seed_trajectory_hash=digest(trajectory), step_ids=[s['step_id'] for s in steps],
        original_steps_chars=len(canonical(trajectory['steps'])), projected_steps_chars=len(canonical(steps)),
        all_commands_preserved=True, all_observations_preserved=True))
    graph = requests.ask('compile', dict(goal=trajectory['goal'], steps=steps,
        grasp_offset_mm=trajectory['grasp_offset_mm'], limitations=trajectory['limitations'],
        coverage=trajectory.get('coverage'), goal_reference_roles=trajectory.get('goal_references', []),
        observation_geometry={k: v.get('geometry', {}) for k,v in trajectory['observations'].items()}), images,
        validator=lambda x: graph_validation(x, trajectory))
    graph.update(protocol=PROTOCOL, goal=trajectory['goal'], seed_trajectory_hash=digest(trajectory),
                 seed_trajectory_path=str((run / 'seed_trajectory.json').resolve()),
                 grasp_offset_mm=trajectory['grasp_offset_mm'], compile_metrics=requests.metrics(),
                 goal_references=trajectory.get('goal_references', []), coverage=trajectory.get('coverage'))
    frozen(graph_path, graph)
    from .graph_template_compiler import compile_policy
    compile_policy(graph_path, run / 'compiled_policy')
    (run / 'graph_v1.md').write_text('# One-shot observation-action graph\n\n'
        'All reusable edges are hypotheses, not validated rules. Observed failures are excluded from replay.\n\n'
        + 'Coverage: ' + canonical(graph.get('coverage')) + '\n\n'
        + '```mermaid\nflowchart LR\n'
        + '\n'.join(f'    node{list(n["node_id"] for n in graph["nodes"]).index(e["source_node"])} -->|edge{i}| node{list(n["node_id"] for n in graph["nodes"]).index(e["target_node"])}' for i,e in enumerate(graph['edges']))
        + '\n```\n\n'
        + '\n'.join(f'- edge{i}: {e["edge_id"]} ({e["seed_step_id"]}), {e["source_node"]} → {e["target_node"]}' for i,e in enumerate(graph['edges'])) + '\n\n'
        + '\n\n'.join('## Node ' + n['node_id'] + '\n\n```json\n' + json.dumps(n, ensure_ascii=False, indent=2) + '\n```' for n in graph['nodes'])
        + '\n\n' + '\n\n'.join('## Edge ' + e['edge_id'] + '\n\n```json\n' + json.dumps(e, ensure_ascii=False, indent=2) + '\n```' for e in graph['edges']) + '\n')
    return graph


def reliable_match(value, graph, threshold):
    ids = {n['node_id'] for n in graph['nodes'] if n['representative_observations']}
    if set(value['candidate_nodes']) - ids: raise ValueError('unknown candidate node')
    if value['status'] == 'MATCH' and value['node_id'] not in ids: raise ValueError('unknown matched node')
    if value['status'] != 'MATCH' and value['node_id']: raise ValueError('non-match must not select node')
    return (value['status'] == 'MATCH' and value['confidence'] >= threshold
            and bool(value['matched_evidence']) and not value['violated_conditions']
            and not value['unresolved_conditions'] and set(value['candidate_nodes']) <= {value['node_id']})


def eligible_edges(graph, node_id):
    observed = {n['node_id'] for n in graph['nodes'] if n['representative_observations']}
    return [e for e in graph['edges'] if e['source_node'] == node_id
            and e['generalization_status'] == 'ONE_SHOT_HYPOTHESIS'
            and e['seed_outcome'] in {'SUCCESS', 'PROGRESS'}
            and e['source_node'] in observed and e['target_node'] in observed]


def replay_choices(graph, node_id, attempted_edges):
    edges = eligible_edges(graph, node_id)
    if graph.get('replay_policy', {}).get('edge_reuse') == 'ONCE_PER_RUN':
        edges = [e for e in edges if e['edge_id'] not in attempted_edges]
    return edges


def instantiate(value, edges, observation, *, max_spread_mm=20.):
    """No generated Python is accepted. Host emits the unchanged primitive skeleton."""
    import numpy as np
    edge = next((e for e in edges if e['edge_id'] == value['edge_id']), None)
    if not edge or value['action_family'] != edge['action_family']: raise ValueError('not an eligible existing edge/family')
    if value['status'] != 'READY' or value['unresolved_conditions']: raise ValueError('unresolved instantiation')
    ops = edge['seed_action']['exact_command']['operations']
    if len(ops) < 2 or ops[0]['method'] != 'grasp' or any(o['method'] not in ('move_tcp', 'move_transfer') for o in ops[1:]):
        raise ValueError('seed skeleton unsupported; no free-plan substitution')
    if any(set(o['arguments']) != {'right'} for o in ops): raise ValueError('arm7-only skeleton required')
    if [s['operation_index'] for s in value['slots']] != list(range(len(ops))): raise ValueError('primitive slots changed')
    measurements = []; lines = []
    for op, slot in zip(ops, value['slots']):
        if op['method'] == 'grasp' and slot['z_above_surface_mm'] != 0: raise ValueError('grasp offset must be applied only by host')
        xyz = np.asarray(observation.sample(slot['pixel_xy'], max_spread_mm), dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all(): raise ValueError('invalid measured surface')
        pose = [xyz[0]/1000, xyz[1]/1000, (xyz[2]+slot['z_above_surface_mm'])/1000, *slot['rpy_rad']]
        if not np.isfinite(pose).all(): raise ValueError('non-finite pose')
        pose = [float(v) for v in pose]
        lines.append(f'motion.{op["method"]}(right={pose!r})')
        measurements.append(dict(operation_index=slot['operation_index'], pixel_xy=slot['pixel_xy'],
                                 surface_world_mm=xyz.tolist(), commanded_world_m_rad=pose,
                                 evidence=slot['relation_evidence']))
    source = '\n'.join(lines) + '\n'
    command(source)  # same parser as normal arm7 pipeline
    return edge, source, measurements


def replay(graph_path, output, runtime, requests, *, max_steps=None, threshold=.8):
    graph_path = Path(graph_path); graph = load_frozen(graph_path)
    trajectory = load_frozen(graph['seed_trajectory_path'])
    validate_replay_graph(graph, trajectory)
    if digest(trajectory) != graph['seed_trajectory_hash']: raise ValueError('seed provenance changed')
    if runtime.grasp_offset_mm != graph['grasp_offset_mm']: raise ValueError('runtime grasp offset differs from seed')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    frozen(output / 'graph_used.json', graph)
    limit = max_steps if max_steps is not None else max(1, len(graph['edges']))
    if limit < 1: raise ValueError('max_steps must be positive')
    metrics = dict(node_match='NOT_ATTEMPTED', graph_reused=False, graph_only_success=False,
        reached_expected_target_node=False, fallback_used=False, failure_category='UNKNOWN',
        status='STARTED', action_families=[], attempted_edges=[], executed_edges=[], transitions=[])
    save(output / 'metrics.json', metrics)
    try:
        before = runtime.observe(output / 'initial_observation')
        for index in range(limit):
            folder = output / f'step_{index:03d}'; folder.mkdir()
            images = [('current', before.image), *goal_images(graph)]
            images += [(p['id'], checked_file(p['artifact'])) for p in graph.get('state_prototypes', [])]
            match = requests.ask('match', dict(goal=graph['goal'], graph=graph), images,
                validator=lambda v: reliable_match(v, graph, threshold))
            save(folder / 'node_match.json', match)
            metrics['node_match'] = match['status']
            if not reliable_match(match, graph, threshold):
                metrics.update(status='GRAPH_MISS' if match['status'] == 'NO_MATCH' else 'AMBIGUOUS',
                               failure_category='GRAPH_MISS' if match['status'] == 'NO_MATCH' else 'UNKNOWN')
                break
            if metrics['transitions'] and match['node_id'] != metrics['transitions'][-1]['expected_target_node']:
                metrics.update(status='INCONSISTENT_NODE_MATCH', failure_category='NODE_MISMATCH'); break
            if match['node_id'] in graph.get('terminal_node_ids', []):
                metrics.update(status='TERMINAL_NODE_NO_ACTION', failure_category='NONE',
                    reached_terminal_node=True, completion_basis='USER_DESIGNATED_TERMINAL_MATCH')
                break
            outgoing = [e for e in graph['edges'] if e['source_node'] == match['node_id']]
            edges = replay_choices(graph, match['node_id'], metrics['attempted_edges'])
            if not edges:
                metrics.update(status='NO_ELIGIBLE_EDGE', failure_category='GRAPH_MISS')
                if not outgoing:
                    assessment = requests.ask('assess', dict(goal=graph['goal'], graph=graph,
                        executed_edge=None, current_node=match['node_id']), images)
                    save(folder / 'assessment.json', assessment)
                    metrics.update(status='TERMINAL_NODE_NO_ACTION', final_outcome=assessment)
                    if (metrics['executed_edges'] and metrics['reached_expected_target_node']
                            and assessment['task_goal'] == 'COMPLETE' and assessment['confidence'] >= threshold
                            and assessment['evidence']):
                        metrics.update(graph_only_success=True, status='GRAPH_ONLY_SUCCESS', failure_category='NONE')
                break
            candidate = requests.ask('instantiate', dict(goal=graph['goal'], node_id=match['node_id'],
                eligible_edges=edges, attempted_edges=metrics['attempted_edges'],
                previous_transitions=metrics['transitions'], grasp_offset_mm=runtime.grasp_offset_mm,
                image_size=list(before.xyz.shape[1::-1]),
                current_geometry=observation_geometry(before)), images)
            save(folder / 'instantiation.json', candidate)
            if candidate['status'] == 'BLOCKED':
                metrics.update(status='INSTANTIATION_BLOCKED', failure_category='EDGE_GENERALIZATION_FAILURE'); break
            try: edge, source, measurements = instantiate(candidate, edges, before)
            except ValueError as exc:
                metrics.update(status='INSTANTIATION_REJECTED', failure_category='EDGE_GENERALIZATION_FAILURE', error=str(exc)); break
            save(folder / 'surface_measurements.json', measurements)
            audit = requests.ask('conformance', dict(goal=graph['goal'], existing_edge=edge,
                candidate=candidate, measurements=measurements, command=command(source),
                grasp_offset_mm=runtime.grasp_offset_mm,
                current_geometry=observation_geometry(before)), images)
            save(folder / 'conformance.json', audit)
            if not (audit['status'] == 'CONFORMS' and audit['confidence'] >= threshold
                    and audit['evidence'] and not audit['violated_constraints'] and not audit['unresolved_conditions']):
                metrics.update(status='TEMPLATE_NOT_CONFIRMED', failure_category='EDGE_GENERALIZATION_FAILURE'); break
            (folder / 'motion.py').write_text(source)
            # Durable intent precedes dispatch; this directory can never be reused.
            save(folder / 'dispatch_intent.json', dict(event='GRAPH_REUSE', graph_hash=digest(graph),
                 edge_id=edge['edge_id'], node_id=match['node_id'], command=command(source),
                 before=evidence_file(before.image), physical_execution_pending=True))
            metrics['graph_reused'] = True
            outcome = execute_recorded(runtime, source, before, folder)
            save(folder / 'outcome.json', outcome)
            metrics['action_families'].append(edge['action_family'])
            metrics['attempted_edges'].append(edge['edge_id'])
            if outcome.get('physical_execution') is True: metrics['executed_edges'].append(edge['edge_id'])
            if outcome.get('status') != 'EXECUTED':
                metrics.update(status='EXECUTION_FAILED', failure_category='ACTION_EXECUTION_FAILURE', final_outcome=outcome); break
            after = runtime.observe(folder / 'after')
            assessment = requests.ask('assess', dict(goal=graph['goal'], graph=graph,
                executed_edge=edge, original_match=match, command=command(source),
                actual_execution=outcome, intermediate_evidence=compact_evidence(collect_evidence(folder))),
                [('before', before.image), ('after', after.image), *goal_images(graph)])
            save(folder / 'assessment.json', assessment)
            reached = (assessment['expected_target'] == 'REACHED'
                and assessment['matched_node_id'] == edge['target_node']
                and assessment['confidence'] >= threshold and bool(assessment['evidence']))
            metrics['transitions'].append(dict(before=evidence_file(before.image), edge_id=edge['edge_id'],
                after=evidence_file(after.image), expected_target_node=edge['target_node'],
                assessment=assessment, reached_expected_target_node=reached))
            metrics.update(reached_expected_target_node=reached, final_outcome=assessment)
            terminal_ids = graph.get('terminal_node_ids', [])
            goal_reached = edge['target_node'] in terminal_ids if terminal_ids else assessment['task_goal'] == 'COMPLETE'
            if reached and goal_reached and outcome.get('physical_execution') is True:
                metrics.update(reached_terminal_node=bool(terminal_ids),
                    completion_basis='USER_DESIGNATED_TERMINAL_MATCH' if terminal_ids else 'MODEL_GOAL_ASSESSMENT')
                metrics.update(graph_only_success=True, status='GRAPH_ONLY_SUCCESS', failure_category='NONE'); break
            if not reached:
                category = assessment['failure_category']
                metrics.update(status='TARGET_NOT_CONFIRMED', failure_category=category if category != 'NONE' else 'UNKNOWN'); break
            before = after
            metrics.update(status='STEP_LIMIT', failure_category='UNKNOWN')
    except BaseException as exc:
        metrics.update(status='ERROR', error=f'{type(exc).__name__}: {exc}', failure_category='UNKNOWN')
        runtime.stop()
        raise
    finally:
        try: runtime.close()
        except Exception as exc: metrics['cleanup_error'] = f'{type(exc).__name__}: {exc}'
        metrics.update(requests.metrics())
        metrics['seed_comparison'] = dict(metrics=trajectory['metrics'], final_outcome=trajectory['result'],
            action_families=[e['action_family'] for e in graph['edges']],
            caveat='Seed completion is planner-reported; replay completion is a separate visual audit, not independent physical ground truth.')
        frozen(output / 'final_metrics.json', metrics); save(output / 'metrics.json', metrics)
        replay_report(output, metrics)
    return metrics


def replay_report(output, metrics):
    seed_metrics = metrics['seed_comparison']['metrics']
    lines = ['# Graph replay audit', '',
             f"Status: **{metrics['status']}**; failure category: **{metrics['failure_category']}**.", '',
             f"Graph-only success: {metrics['graph_only_success']}; fallback used: {metrics['fallback_used']}.", '',
             '| Metric | Seed free planner | Graph replay |', '|---|---:|---:|']
    for key in ('model_calls', 'visual_tool_calls', 'tool_calls', 'reasoning_time_s'):
        lines.append(f"| {key} | {seed_metrics.get(key, 'unavailable')} | {metrics.get(key, 'unavailable')} |")
    lines += ['', 'Calls count Codex invocations, including bounded format repair; internal API request count is unavailable.',
              'Reasoning time is observable model-call wall time, not private reasoning or robot motion time.',
              'Seed final outcome is planner-reported; replay uses a separate visual audit. This comparison is not a controlled speed benchmark.', '',
              '## Reused transitions', '']
    for i, transition in enumerate(metrics['transitions']):
        lines += [f"- Step {i}: existing **{transition['edge_id']}** -> **{transition['expected_target_node']}**; "
                  f"target confirmed: {transition['reached_expected_target_node']}. "
                  f"[Instantiation](step_{i:03d}/instantiation.json), [surface measurements](step_{i:03d}/surface_measurements.json), "
                  f"[template audit](step_{i:03d}/conformance.json), [outcome audit](step_{i:03d}/assessment.json)."]
    if not metrics['transitions']: lines += ['No completed, visually assessed transition. See step records and status.']
    lines += ['', '## Full metrics and evidence', '', '```json', json.dumps(metrics, ensure_ascii=False, indent=2), '```', '']
    (Path(output) / 'report.md').write_text('\n'.join(lines))


def make_runtime(config, port):
    # The same runtime, camera, calibration, offsets, IK and execution as arm7.
    from cloth_agent.codex_arm7_runtime import Arm7Runtime
    from cloth_agent.trajectory_preview import TrajectoryPreview
    runtime = Arm7Runtime(config)
    try: runtime.preview = TrajectoryPreview(runtime.config, port=port, wait=False)
    except BaseException:
        runtime.close(); raise
    return runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='phase', required=True)
    s = commands.add_parser('seed', help='physically execute the existing arm7 free planner')
    s.add_argument('--subgoal', choices=['full_fold', 'right_sleeve'], default='full_fold')
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--compile-after', action='store_true', help='compile graph and minimal Host templates after seed returns; no replay')
    s.add_argument('--max-rounds', type=int, help='default: full_fold 30, right_sleeve 8')
    c = commands.add_parser('compile', help='compile recorded seed using GPT; no robot')
    c.add_argument('--run', type=Path, required=True)
    r = commands.add_parser('replay', help='capture and physically replay after manually resetting cloth')
    r.add_argument('--graph', type=Path, required=True)
    r.add_argument('--output', type=Path)
    r.add_argument('--max-steps', type=int)
    r.add_argument('--match-threshold', type=float, default=.8)
    for p in (s, r):
        p.add_argument('--config', type=Path, default=ROOT / 'config/dual_arm.local.json')
        p.add_argument('--viser-port', type=int, default=8088)
    for p in (c, r):
        p.add_argument('--model', default='gpt-6-astra')
        p.add_argument('--no-cache', action='store_true')
        p.add_argument('--max-calls', type=int, default=256)
    for p in (s, c, r): p.add_argument('--timeout-s', type=float, default=900)
    args = parser.parse_args(argv)
    if args.timeout_s <= 0: parser.error('timeout must be positive')
    if args.phase == 'seed' and args.max_rounds is not None and args.max_rounds < 1: parser.error('max-rounds must be positive')
    if args.phase == 'replay' and (not 0 < args.match_threshold <= 1 or (args.max_steps is not None and args.max_steps < 1)):
        parser.error('invalid match-threshold/max-steps')
    runtime = None
    previous_signal = signal.getsignal(signal.SIGTERM)
    def terminate(*_): raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM, terminate)
    try:
        if args.phase == 'seed':
            if args.output.exists(): parser.error('seed output already exists; never repeat a physical run in place')
            runtime = make_runtime(args.config, args.viser_port)
            result = seed(args.output, runtime, subgoal=args.subgoal, max_rounds=args.max_rounds, timeout_s=args.timeout_s)
            if args.compile_after:
                runtime.close()
                if getattr(runtime, 'preview', None) is not None: runtime.preview.close()
                runtime = None
                from .discovery_model import CodexMultimodal
                from .graph_policy_schema import seed_compiler_schemas
                requests = Requests(args.output / 'compile', CodexMultimodal('gpt-6-astra', args.timeout_s),
                                    no_cache=True, max_calls=256, schemas=seed_compiler_schemas())
                compile_graph(args.output, requests)
                result = dict(seed_result=result,
                    compiled_policy_path=str((args.output/'compiled_policy/policy_graph.json').resolve()),
                    minimal_compilation=read(args.output/'compiled_policy/summary.json'))
        else:
            from .discovery_model import CodexMultimodal
            backend = CodexMultimodal(args.model, args.timeout_s)
            if args.phase == 'compile':
                from .graph_policy_schema import seed_compiler_schemas
                requests = Requests(args.run / 'compile', backend, no_cache=args.no_cache, max_calls=args.max_calls,
                                    schemas=seed_compiler_schemas())
                result = compile_graph(args.run, requests)
                result = {**result, 'compiled_policy_path': str((args.run/'compiled_policy/policy_graph.json').resolve()),
                          'minimal_compilation': read(args.run/'compiled_policy/summary.json')}
            else:
                output = args.output or args.graph.parent / ('replay_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
                if output.exists(): parser.error('replay output already exists; physical dispatch is never resumed')
                load_frozen(args.graph)  # reject bad artifact before constructing runtime
                runtime = make_runtime(args.config, args.viser_port)
                requests = Requests(output / 'model', backend, no_cache=args.no_cache, max_calls=args.max_calls)
                result = replay(args.graph, output, runtime, requests, max_steps=args.max_steps, threshold=args.match_threshold)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_signal)
        if runtime is not None:
            runtime.close()
            if getattr(runtime, 'preview', None) is not None: runtime.preview.close()


if __name__ == '__main__': raise SystemExit(main())
