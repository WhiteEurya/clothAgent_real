import pytest

from cloth_agent.experiments import independent_graph_replay as replay
from cloth_agent.experiments.graph_policy_replay import OnlineGrowth, main
from tests.test_graph_policy_replay import policy, Runtime, Model, matched, match


def snapshot(tmp_path):
    base, _ = policy(tmp_path)
    archived = OnlineGrowth(base, tmp_path / 'archive')
    return archived.latest_path


def test_two_runs_are_independent_and_never_write_growth(tmp_path):
    path = snapshot(tmp_path)
    original = path.read_bytes()
    results = []
    for name in ('first', 'second'):
        runtime = Runtime()
        folder = tmp_path / name
        result = replay.replay_snapshot(path, folder, runtime, Model(), max_steps=3)
        results.append(result)
        assert len(runtime.executions) == 1
        assert result['physical_actions'] == 1
        assert result['steps'][0]['source_node'] == 'N0'
        assert result['steps'][0]['actual_target_node'] == 'N1'
        assert not list(folder.glob('growth*'))
        assert not (folder / 'latest.json').exists()
        assert not (folder / 'episodes.json').exists()
    assert results[0]['status'] == results[1]['status'] == 'NO_REPLAYABLE_ACTION'
    assert path.read_bytes() == original


@pytest.mark.parametrize('response,status', [(match('NO_MATCH'), 'NO_MATCH'), (matched('N0'), 'SOURCE_NODE_UNCHANGED')])
def test_no_match_or_no_progress_stops_without_growth_or_repair(tmp_path, response, status):
    path = snapshot(tmp_path)
    runtime = Runtime()
    result = replay.replay_snapshot(path, tmp_path / 'run', runtime, Model([matched('N0'), response]))
    assert result['status'] == status
    assert len(runtime.executions) == 1
    assert not list((tmp_path / 'run').rglob('growth_*.json'))


def test_last_action_is_observed_and_matched_even_at_budget(tmp_path):
    path = snapshot(tmp_path)
    result = replay.replay_snapshot(path, tmp_path / 'run', Runtime(), Model(), max_steps=1)
    assert result['status'] == 'STEP_LIMIT'
    assert result['steps'][0]['actual_target_node'] == 'N1'


def test_cli_dry_run_and_rejects_resume(tmp_path):
    path = snapshot(tmp_path)
    args = ['--graph', str(path), '--max-recovery-attempts', '0', '--output', str(tmp_path / 'run')]
    assert main(args) == 0
    with pytest.raises(SystemExit):
        main(args + ['--growth-state', str(path)])
    with pytest.raises(SystemExit):
        main(args)
