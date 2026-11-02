"""Explicit single-action execution entry for an inspected contour-normal plan."""
import argparse
import json
from pathlib import Path

from cloth_agent.contour_normal_grasp import digest, execute_once, save
from cloth_agent.dual_arm.observation import Observation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observation', type=Path, required=True)
    p.add_argument('--proposal', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('config/dual_arm.local.json'))
    p.add_argument('--execute', action='store_true')
    p.add_argument('--skip-grasp-height-spread', action='store_true',
                   help='skip local height-spread rejection only; valid measured depth remains required')
    args = p.parse_args()
    if not args.execute:
        p.error('this entry dispatches one physical action; --execute is required')
    before = Observation.load(args.observation)
    proposal = json.loads(args.proposal.read_text())
    if proposal.get('observation_image_sha256') != digest(before.image):
        p.error('proposal and RGB-D observation do not match')
    # Fail before even constructing a runtime for a non-action or synthetic input.
    if before.meta.get('synthetic') is not False or proposal.get('status') != 'OFFLINE_CANDIDATE':
        p.error('requires a real RGB-D observation and a valid candidate')
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = None
    report = dict(status='PREPARING', action_attempts=0, automatic_retries=0)
    try:
        from cloth_agent.experiments.trajectory_memory_full_episode_v1 import EpisodeCollectionRuntime
        from cloth_agent.preference_one_shot import verify_scene
        runtime = EpisodeCollectionRuntime(args.config)
        current = runtime.observe(args.output/'current')
        verify_scene(before, current, proposal['plan'], args.output, automatic=True)
        report['action_attempts'] = 1
        result = execute_once(runtime, before, proposal, args.output/'action',
                              check_grasp_height_spread=not args.skip_grasp_height_spread)
        report.update(status=result['status'], execution=result)
    except BaseException as exc:
        report.update(status='STOPPED', error=f'{type(exc).__name__}: {exc}')
        if runtime is not None:
            runtime.cancel.set()
            runtime.stop()
        raise
    finally:
        save(args.output/'result.json', report)
        if runtime is not None:
            runtime.close()


if __name__ == '__main__':
    main()
