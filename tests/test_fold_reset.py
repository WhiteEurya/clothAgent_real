import json
from types import SimpleNamespace

import pytest

from cloth_agent.fold_reset import FoldReset, confirm
from cloth_agent.fold_exploration_pipeline import (
    FOLD_STEP_IDS, FoldExplorationPipeline, FoldSupervisor,
    _merge_supervisor_completion_ledger, validate_supervisor_payload,
)
from cloth_agent.fold_exploration_viser import _reset_markdown
from scripts.watch_fold_exploration import classify_child_exit


def decision():
    return dict(status='BLOCKED', current_step='BLOCKED', completed_steps=[],
                garment_visibility='PARTIAL', trajectory_decision='REQUEST_RESET', confidence=0.9,
                evidence=['Repeated attempts left the shirt tangled beyond reach.'],
                reason='请将衣物重新展开，放回相机可见的桌面工作区。')


def test_reset_takes_precedence_over_old_completion_ledger():
    payload = decision()
    payload.update(status='COMPLETE', current_step='COMPLETE', completed_steps=list(FOLD_STEP_IDS))
    result = validate_supervisor_payload(payload)
    result = _merge_supervisor_completion_ledger(result, [])
    assert result['status'] == result['current_step'] == 'BLOCKED'
    assert result['trajectory_decision'] == 'REQUEST_RESET'


def test_confirmation_is_required_and_survives_restart(tmp_path, monkeypatch):
    reset = FoldReset(tmp_path / 'workspace')
    request = reset.request(tmp_path / 'iteration_001', decision(), stage='before')
    reset = FoldReset(tmp_path / 'workspace')
    assert reset.pending() == request
    confirmation = tmp_path / 'iteration_001/reset_confirmation.json'
    confirmation.write_text(json.dumps({'request_id': 'old', 'confirmed': True}))
    slept = []

    def human_confirmation(delay):
        slept.append(delay)
        confirm(request['request_path'], request['request_id'])

    monkeypatch.setattr('cloth_agent.fold_reset.time.sleep', human_confirmation)
    completed = reset.wait(request)
    assert len(slept) == 1  # A stale receipt cannot release the wait.
    assert completed['status'] == 'CONFIRMED'
    assert reset.pending() is None
    with pytest.raises(ValueError, match='stale'):
        confirm(request['request_path'], request['request_id'])


def test_confirmed_reset_excludes_old_state_but_keeps_new_experience(tmp_path):
    reset = FoldReset(tmp_path / 'workspace')
    request = reset.request(tmp_path / 'iteration_001', decision(), stage='after')
    with pytest.raises(ValueError, match='stale'):
        confirm(request['request_path'], 'wrong-id')
    confirm(request['request_path'], request['request_id'])
    reset.wait(request)
    old = {'created_at': '2000-01-01T00:00:00+00:00', 'completed_steps': ['left_sleeve']}
    new = {'created_at': '2999-01-01T00:00:00+00:00'}
    assert reset.current_history([old, {}, new]) == [new]
    second = reset.request(tmp_path / 'iteration_002', decision(), stage='before')
    with pytest.raises(ValueError, match='stale'):
        confirm(request['request_path'], request['request_id'])
    assert reset.pending()['request_id'] == second['request_id']


def test_fallback_cannot_request_reset(tmp_path):
    reset = FoldReset(tmp_path)
    with pytest.raises(ValueError, match='explicit Claude'):
        reset.request(tmp_path / 'iteration_001', {**decision(), 'fallback': True}, stage='before')


def test_pipeline_publishes_wait_and_refreshes_perception_after_confirmation(tmp_path, monkeypatch):
    pipeline = FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    pipeline.session = SimpleNamespace(workspace=tmp_path / 'workspace')
    pipeline._debug = lambda *args, **kwargs: None
    pipeline.reuse_latest_perception = True
    summary = {'status': 'RUNNING'}
    output = tmp_path / 'output'
    output.mkdir()

    def human_confirmation(_):
        pending_summary = json.loads((output / 'summary.json').read_text())
        assert pending_summary['status'] == 'WAITING_FOR_RESET'
        assert pending_summary['restart_safe'] is False
        markdown = _reset_markdown(pending_summary)
        assert '等待人工 RESET' in markdown and decision()['reason'] in markdown
        request = pending_summary['reset_request']
        assert request['confirmation_command'] in markdown
        confirm(request['request_path'], request['request_id'])

    monkeypatch.setattr('cloth_agent.fold_reset.time.sleep', human_confirmation)
    pipeline._wait_for_manual_reset(output, summary, iteration_dir=output / 'iteration_001',
                                   decision=decision(), stage='before')
    assert summary['status'] == 'RUNNING'
    assert pipeline.reuse_latest_perception is False
    assert summary['reset_request']['status'] == 'CONFIRMED'


def test_watchdog_never_restarts_pending_reset():
    for returncode in (0, 1, 130):
        assert classify_child_exit(returncode, {'status': 'WAITING_FOR_RESET'}, None,
                                   restart_on_max_iterations=True) == 'STOP'


def test_supervisor_prompt_allows_human_reset(tmp_path):
    bundle = FoldSupervisor._write_context_bundle(tmp_path, images=[], video_evidence=[], history=[], screen={})
    from pathlib import Path
    instructions = (Path(bundle['directory']) / '01_instructions.md').read_text()
    assert 'trajectory_decision=REQUEST_RESET' in instructions
    assert 'explicit human confirmation' in instructions
