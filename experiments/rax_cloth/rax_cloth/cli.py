from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time
from .mcp import Frontend
from .session import Session, write_json
from .upstream import make_backend, ROOT

# Explicit trusted diagnostic, not a strategy or agent-authored compiled policy.
FIXTURE = '''def main(ctx):
    observation = ctx.call("observe", {})
    return {"observation": observation, "task_success": None}
'''


def main(argv=None):
    p = argparse.ArgumentParser(description='RAX-Cloth offline setting validation (not physical exploration)')
    p.add_argument('--mode', choices=['mock', 'replay', 'sim', 'real_supervised'], default='mock')
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--replay-manifest', type=Path)
    p.add_argument('--max-hours', type=float, default=8)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--max-llm-calls', type=int, default=0)
    p.add_argument('--budget-usd', type=float, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--interval-seconds', type=float, default=1)
    p.add_argument('--driver', choices=['smoke', 'agent', 'mcp'], default='smoke')
    args = p.parse_args(argv)
    if args.mode not in {'mock', 'replay'}:
        p.error('sim unverified; real_supervised disabled in this offline setting')
    for name in ('max_hours', 'budget_usd', 'interval_seconds'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(f'{name} must be finite and nonnegative')
    if args.max_hours <= 0 or args.episodes < 1 or args.max_llm_calls < 0:
        p.error('positive duration/episodes and nonnegative call limit required')
    if args.mode == 'replay' and not args.replay_manifest:
        p.error('replay needs --replay-manifest')
    if args.driver == 'agent':
        from .sandbox import require_agent_ready
        require_agent_ready()
    config = dict(mode=args.mode, seed=args.seed, max_hours=args.max_hours,
                  episodes=args.episodes, max_llm_calls=args.max_llm_calls,
                  budget_usd=args.budget_usd, driver=args.driver,
                  interval_seconds=args.interval_seconds,
                  replay_manifest=str(args.replay_manifest.resolve()) if args.replay_manifest else None,
                  replay_manifest_sha256=hashlib.sha256(args.replay_manifest.read_bytes()).hexdigest() if args.replay_manifest else None,
                  model=None, harness='offline_contract_diagnostic',
                  upstream=json.loads((ROOT / 'upstream.json').read_text()))
    stop = threading.Event()
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, lambda *_: stop.set())
    session = Session(args.session, config, resume=args.resume)
    started = time.monotonic()
    status = 'complete'
    backend = None
    try:
        while len(session.state['episodes']) < args.episodes and not stop.is_set():
            if session.state['elapsed_seconds'] >= args.max_hours * 3600:
                status = 'time_budget_exhausted'
                break
            episode, row = session.episode()
            def audit(event, **data):
                session.audit(event, episode=row['id'], **data)
                with (episode / 'events.jsonl').open('a') as f:
                    f.write(json.dumps(dict(event=event, **data), allow_nan=False)+'\n')
            backend = make_backend(episode, dict(mode=args.mode, replay_manifest=config['replay_manifest']))
            frontend = Frontend(backend, audit)
            if args.driver == 'mcp':
                deadline = time.monotonic() + args.max_hours*3600-session.state['elapsed_seconds']
                frontend.serve(stop=stop, deadline=deadline)
                row['status'] = 'complete'
                if time.monotonic() >= deadline:
                    status = 'time_budget_exhausted'
                break
            request_id = 0
            def call(name, arguments):
                nonlocal request_id
                request_id += 1
                response = frontend.handle(dict(jsonrpc='2.0', id=request_id, method='tools/call',
                                                 params=dict(name=name, arguments=arguments)))
                if 'error' in response:
                    raise RuntimeError(response['error']['message'])
                return json.loads(response['result']['content'][0]['text'])
            audit('phase', phase='Explore', kind='interface_test_only', agent_reasoning=None)
            observed = call('observe', {})
            call('coordinates', {})
            call('robot_state', {})
            refused = call('move_tcp', {'arm': 'arm6', 'pose_mm_deg': [0, 0, 0, 0, 0, 0]})
            assert refused['status'] == 'NOT_EXECUTED' and refused['physical_outcome'] is None
            audit('phase', phase='Compile', kind='trusted_fixture_syntax_test')
            version = session.version(FIXTURE, row['id'], author='test_fixture')
            audit('phase', phase='Evaluate', kind='trusted_fixture_execution_only')
            # Fixed source compiled here is host-authored and never takes agent code.
            namespace = {}
            exec(compile(FIXTURE, '<trusted-interface-fixture>', 'exec'), namespace)
            class Context:
                def call(self, name, arguments):
                    return call(name, arguments)
            began = time.monotonic()
            evaluated = namespace['main'](Context())
            evaluation_s = time.monotonic()-began
            assert evaluated['task_success'] is None
            version.update(execution='TRUSTED_FIXTURE_PASSED', program_seconds=evaluation_s,
                           model_inference_seconds=None)
            write_json(session.root/'programs'/version['sha256']/f'test-{row["id"]:06d}.json', version)
            audit('phase', phase='Improve', kind='NO_AGENT_NO_STRATEGY_UPDATE', verified_skill=False)
            result = dict(mode=args.mode, interface_test='passed', task_success=None,
                          physical_actions=0, physical_execution_seconds=None,
                          observation_seconds=observed['observation_seconds'],
                          final_program_inference_seconds=None, fixture_execution_seconds=evaluation_s,
                          llm_calls=[], program_version=version, action_result=refused)
            write_json(episode / 'result.json', result)
            row['status'] = 'complete'
            backend.close()
            backend = None
            now = time.monotonic()
            session.state['elapsed_seconds'] += now-started
            started = now
            session.save()
            if len(session.state['episodes']) < args.episodes:
                remaining = args.max_hours*3600-session.state['elapsed_seconds']
                stop.wait(max(0, min(args.interval_seconds, remaining)))
        if stop.is_set():
            status = 'stopped'
    except BaseException as exc:
        status = 'failed'
        session.audit('failure', type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if backend is not None:
            backend.close()
        session.state['elapsed_seconds'] += time.monotonic()-started
        session.finish(status)
        session.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if args.driver != 'mcp':
        print(str(session.root / 'summary.md'))


if __name__ == '__main__':
    main()
