"""Budgeted offline batches of the existing official-workflow Codex launcher.

No simulation, robot authorization, policy, automatic error retry or synthetic
physical feedback. Each episode is an official session sharing its skill library.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import signal
import threading
from types import SimpleNamespace

from .session import write_json
from .upstream import ROOT

TERMINAL = {'agent_completed', 'session_time_budget_exhausted', 'model_call_budget_exhausted'}


def read(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def reconcile(root, row):
    private = root/row['directory']/'codex-adaptation'
    state = read(private/'agent-state.json', {})
    ledger = read(private/'model'/'ledger.json', {})
    # Absent durable evidence cannot refund an interrupted reservation.
    row['seconds'] = state.get('elapsed_seconds', row['seconds'])
    row['calls'] = ledger.get('calls_reserved', row['calls'])
    row['calls_measured'] = 'calls_reserved' in ledger
    if state.get('status') in TERMINAL:
        row['status'] = state['status']
    return state


def save(root, state):
    write_json(root/'batch.json', state)
    summary = dict(status=state['status'], mode=state['config']['mode'],
                   episodes=len(state['episodes']),
                   model_calls_reserved=(sum(r['calls'] for r in state['episodes'])
                       if all(r.get('calls_measured',False) for r in state['episodes']) else None),
                   model_call_budget_charged=sum(r['calls'] for r in state['episodes']),
                   charged_seconds=sum(r['seconds'] for r in state['episodes']),
                   physical_actions=0, task_success=None, cost_usd=None,
                   physical_feedback='unavailable', harness='official RAX workflow + Codex CLI adaptation')
    write_json(root/'results.json', summary)
    (root/'summary.md').write_text('# RAX-Codex offline batch\n\n'+json.dumps(summary, indent=2)+
        '\n\nMock/replay cannot establish cloth manipulation success. Model completion is not task success.\n'
        'Interrupted reservations without durable usage evidence remain charged conservatively.\n'
        'See each episode for raw observations, model requests/responses, MCP calls, code and memory.\n')
    return summary


def run(args, launcher=None):
    if args.mode not in ('mock', 'replay'):
        raise ValueError('overnight permits only mock/replay; no verified simulator or unattended real mode')
    if args.budget_usd is not None:
        raise ValueError('USD enforcement unavailable with this transport; use time/call limits')
    if (not math.isfinite(args.max_hours) or args.max_hours*3600 < 1
            or min(args.episodes, args.max_model_calls, args.episode_calls, args.episode_seconds) < 1
            or args.episode_seconds > 28800 or not 0 <= args.seed <= 4294967295):
        raise ValueError('invalid time/episode/call/seed budget')
    if launcher is None:
        from .codex_agent import launch
        launcher = launch
    if args.model is None:
        if args.profile=='chatgpt':
            from cloth_agent.codex_astra_pipeline import MODEL
            args.model=MODEL
        else:
            import os
            from .codex_profile import Profile
            args.model=Profile(Path(os.environ.get('CODEX_HOME',str(Path.home()/'.codex'))),args.profile).model
    root = args.session.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = args.replay_manifest.resolve()
    config = dict(mode=args.mode, model=args.model, profile=args.profile, seed=args.seed,
                  max_seconds=int(args.max_hours*3600), episodes=args.episodes,
                  max_model_calls=args.max_model_calls, episode_calls=args.episode_calls,
                  episode_seconds=args.episode_seconds,
                  manifest=str(manifest), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()
                  if args.mode=='replay' else None)
    with (root/'batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(root/'batch.json', None)
        if state is not None:
            if not args.resume or state['config'] != config:
                raise ValueError('existing batch requires --resume with identical configuration')
        else:
            if args.resume:raise ValueError('no batch to resume')
            state = {'config': config, 'episodes': [], 'status': 'running'}
        stopped = threading.Event()
        previous = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            while not stopped.is_set():
                rows = state['episodes']
                pending = rows[-1] if rows and rows[-1]['status'] not in TERMINAL else None
                child_state = reconcile(root, pending) if pending else {}
                if pending and pending['status'] in TERMINAL:
                    pending = None
                if pending is None:
                    remaining_calls = args.max_model_calls-sum(r['calls'] for r in rows)
                    remaining_seconds = config['max_seconds']-sum(r['seconds'] for r in rows)
                    if len(rows)>=args.episodes:
                        state['status']='episodes_completed';break
                    if remaining_calls<1 or remaining_seconds<1:
                        state['status']='batch_budget_exhausted';break
                    pending = dict(directory=f'episode_{len(rows):04d}', status='reserved',
                        call_cap=min(args.episode_calls,remaining_calls),
                        time_cap=min(args.episode_seconds,int(remaining_seconds)))
                    pending.update(calls=pending['call_cap'],seconds=pending['time_cap'])
                    rows.append(pending)
                    child_state={}
                # An ambiguous interrupted call reservation must never silently reset.
                if child_state.get('attempts') and not child_state.get('thread_id'):
                    state['status']='needs_episode_inspection';break
                pending['status']='running';state['status']='running';save(root,state)
                options=SimpleNamespace(session=root/pending['directory'],mode=args.mode,
                    workflow='upstream',robot_config=None,replay_manifest=manifest,model=args.model,
                    profile=args.profile,max_model_calls=pending['call_cap'],max_seconds=pending['time_cap'],
                    seed=args.seed,budget_usd=None,preflight_only=False,
                    resume=bool(child_state.get('thread_id')),progress_interval=args.progress_interval)
                try:
                    result=launcher(options)
                    pending['result']=result;pending['status']=result['status']
                except KeyboardInterrupt:
                    pending['status']='interrupted';stopped.set()
                except Exception as exc:
                    pending['status']='launch_failed';pending['error']=f'{type(exc).__name__}: {exc}'
                reconcile(root,pending)
                save(root,state)
                if pending['status'] not in TERMINAL:
                    state['status']='stopped_on_episode_error';break
            else:state['status']='interrupted'
        finally:
            for sig, handler in previous.items():signal.signal(sig,handler)
            summary=save(root,state)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--mode',choices=['mock','replay'],default='replay')
    parser.add_argument('--profile',default='chatgpt')
    parser.add_argument('--model',default=None)
    parser.add_argument('--replay-manifest',type=Path,default=ROOT.parent/'configs/replay.example.json')
    parser.add_argument('--max-hours',type=float,default=8)
    parser.add_argument('--episodes',type=int,default=24)
    parser.add_argument('--max-model-calls',type=int,default=256)
    parser.add_argument('--episode-calls',type=int,default=32)
    parser.add_argument('--episode-seconds',type=int,default=1200)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--budget-usd',type=float)
    parser.add_argument('--progress-interval',type=int,default=10)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not 1<=args.progress_interval<=300:parser.error('progress interval must be 1..300')
    result=run(args)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status'] in ('episodes_completed','batch_budget_exhausted') else 2)


if __name__=='__main__':main()
