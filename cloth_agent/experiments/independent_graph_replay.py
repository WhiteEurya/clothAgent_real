"""Read an archived experience graph as a fixed policy; never grow or resume it."""
import copy
import json
from pathlib import Path
import signal
from types import SimpleNamespace

from .discovery_data import file_hash, save
from .graph_policy import frozen, load_frozen, reliable_match
from .graph_policy_matching import topology_match
from .growth_edge_recovery import select_recorded_action
from .replay_evidence_paths import resolve_evidence_path
from .replay_failures import capture_cli_failure, close_cli_runtime, cli_failed, stop_after_failure


def load_snapshot(path):
    from .graph_policy_replay import OnlineGrowth, validate_growth_base, GROWTH_PROTOCOL
    # Reuse provenance/schema validation only; never call the growth constructor,
    # snapshot(), match(), transition(), or mutation methods.
    state = load_frozen(path)
    if state.get('protocol') != GROWTH_PROTOCOL:
        raise ValueError('expected an archived experience graph')
    reader = object.__new__(OnlineGrowth)
    reader.base_path = resolve_evidence_path(state['base_graph'])
    reader.base = load_frozen(reader.base_path)
    reader.trajectory = validate_growth_base(reader.base)
    reader.state = state
    reader.validate_state()
    graph = reader.view()
    graph['edges'] = [copy.deepcopy(a['edge']) for a in state['actions'].values() if a.get('edge')]
    trajectory = copy.deepcopy(reader.trajectory)
    trajectory['observations'].update(state['observations'])
    return state, graph, trajectory


def replay_snapshot(path, output, runtime, requests, *, max_steps=20, threshold=.8, live_graph_port=None):
    from .graph_policy_replay import growth_ground, execute_policy_edge
    from .replay_live_graph import create_live_graph
    if max_steps < 1 or not 0 < threshold <= 1:
        raise ValueError('invalid step/confidence limit')
    state, graph, trajectory = load_snapshot(path)
    original_hash = file_hash(path)
    if runtime.grasp_offset_mm != graph['grasp_offset_mm']:
        raise ValueError('runtime lowering differs from reference graph')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frozen(output / 'graph_used.json', state)
    result = dict(status='STARTED', mode='INDEPENDENT_FIXED_GRAPH', steps=[], physical_actions=0,
                  source_graph=str(path), source_sha256=original_hash, graph_modified=False)
    save(output / 'runner_config.json', dict(mode=result['mode'], max_steps=max_steps,
         match_threshold=threshold, online_growth=False, resume=False, recovery=False))
    live = create_live_graph(output, live_graph_port)
    pending = None
    try:
        before = runtime.observe(output / 'initial_observation')
        initial_image, previous_image = before.image, None
        for index in range(max_steps + 1):
            if file_hash(path) != original_hash:
                raise ValueError('reference graph changed during replay')
            folder = output / f'step_{index:04d}'
            folder.mkdir()
            live.update(state, phase='MATCHING', observations=trajectory['observations'])
            match = topology_match(graph, trajectory, before, requests, folder, result,
                                   pending=pending, threshold=threshold)
            save(folder / 'node_match.json', match)
            if not reliable_match(match, graph, threshold):
                result['status'] = 'NO_MATCH' if match['status'] == 'NO_MATCH' else 'AMBIGUOUS'
                break
            node = match['node_id']
            result['last_node'] = node
            live.update(state, current=node, phase='MATCHED', observation=str(before.image))
            if pending:
                result['steps'][-1]['actual_target_node'] = node
                if node == pending['source_node']:
                    result['status'] = 'SOURCE_NODE_UNCHANGED'
                    break
            if node in graph.get('terminal_node_ids', []):
                result['status'] = 'COMPLETED'
                break
            if index == max_steps:
                result['status'] = 'STEP_LIMIT'
                break
            action = select_recorded_action(SimpleNamespace(state=state), node)
            if action is None:
                result['status'] = 'NO_REPLAYABLE_ACTION'
                break
            row = dict(source_node=node, selected_action=action['action_id'],
                       expected_target_node=action['nominal_target'], execution={'status': 'NOT_DISPATCHED'})
            result['steps'].append(row)
            save(folder / 'selected_action.json', action)
            source = growth_ground(runtime, requests, graph, trajectory, action, before, folder,
                                   initial_image, previous_image)
            if file_hash(path) != original_hash:
                raise ValueError('reference graph changed before dispatch')
            save(output / 'final_metrics.json', result)
            live.update(state, current=node, action=action['action_id'], phase='EXECUTING')
            outcome = execute_policy_edge(runtime, source, before, folder, graph, action['edge'])
            row['execution'] = outcome
            save(folder / 'execution.json', outcome)
            if outcome.get('physical_execution'):
                result['physical_actions'] += 1
            if outcome.get('status') != 'EXECUTED' or not outcome.get('physical_execution'):
                result['status'] = 'ACTION_EXECUTION_FAILURE'
                result['error'] = outcome.get('error', outcome.get('status'))
                stop_after_failure(runtime, result)
                break
            pending = dict(source_node=node, target_node=action['nominal_target'])
            previous_image = before.image
            before = runtime.observe(folder / 'after_observation')
    except BaseException:
        save(output / 'final_metrics.json', result)
        raise
    finally:
        save(output / 'final_metrics.json', result)
        live.update(state, phase='STOPPED', result=result['status'])
        live.close()
    return result


def run_cli(args, parser):
    if (args.online_growth or args.growth_state or args.collect_only or args.recover_no_progress
            or args.episodes != 1 or args.max_fallbacks or args.repair_self_loop
            or args.retry_missed_grasp or args.debug_grasp_retry or args.max_recovery_attempts
            or args.grounding_mode != 'full' or args.matcher != 'current'
            or args.initial_node or args.grounding_templates or args.arm6_expanded_graph
            or args.arm6_route or args.arm6_review or args.arm6_live_start):
        parser.error('independent snapshot replay requires full grounding, no growth/recovery, and --max-recovery-attempts 0')
    if args.max_steps < 1 or args.max_calls < 1 or args.timeout_s <= 0 or not 0 < args.match_threshold <= 1:
        parser.error('invalid step/call/time/confidence limit')
    from datetime import datetime, timezone
    output = args.output or Path('reports') / ('independent_replay_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if output.exists():
        parser.error('output already exists; each run requires a new directory')
    state, graph, trajectory = load_snapshot(args.graph)
    if not args.execute:
        output.mkdir(parents=True)
        frozen(output / 'graph_used.json', state)
        result = dict(status='DRY_RUN_OK', mode='INDEPENDENT_FIXED_GRAPH', nodes=len(state['nodes']),
                      actions=len(state['actions']), robot_calls=0, model_calls=0,
                      physical_actions=0, graph_modified=False)
    else:
        from .discovery_model import CodexMultimodal
        from .graph_policy import make_runtime
        from .graph_policy_replay import policy_requests
        runtime = None
        result = {}
        old = signal.getsignal(signal.SIGTERM)
        def terminate(*_):
            raise KeyboardInterrupt('SIGTERM')
        signal.signal(signal.SIGTERM, terminate)
        try:
            requests = policy_requests(output / 'model', CodexMultimodal(args.model, args.timeout_s),
                                       max_calls=args.max_calls, no_cache=args.no_cache)
            runtime = make_runtime(args.config, args.viser_port)
            result = replay_snapshot(args.graph, output, runtime, requests, max_steps=args.max_steps,
                                     threshold=args.match_threshold,
                                     live_graph_port=None if args.no_live_graph else args.graph_port)
        except (Exception, KeyboardInterrupt) as exc:
            result = capture_cli_failure(output, exc)
            if runtime is not None:
                stop_after_failure(runtime, result)
        finally:
            signal.signal(signal.SIGTERM, old)
            close_cli_runtime(runtime, result)
    save(output / 'final_metrics.json', result)
    print(json.dumps(dict(output=str(output), **result), ensure_ascii=False, indent=2))
    return 1 if cli_failed(result) else 0
