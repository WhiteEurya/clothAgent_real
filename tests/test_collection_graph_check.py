import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from cloth_agent.collection_graph_check import GraphCheck, result_title


def test_launch_is_read_only_uses_snapshot_and_cancel_unwinds(tmp_path, monkeypatch):
    image = tmp_path/'rgb.png'; image.touch()
    graph = tmp_path/'graph.json'; graph.touch()
    memory = tmp_path/'memory'; memory.mkdir(); (memory/'memory.json').touch()
    calls = []
    signals = []
    process = SimpleNamespace(poll=lambda: None, send_signal=signals.append,
                              wait=lambda timeout=None: 0)
    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return process
    monkeypatch.setattr('cloth_agent.collection_graph_check.subprocess.Popen', launch)
    check = GraphCheck(image, tmp_path/'graph_checks', graph, memory)
    check.start()
    command, kwargs = calls[0]
    assert command[command.index('-m')+1] == 'cloth_agent.experiments.retrieval_matcher'
    assert '--execute' not in command and '--real' not in command
    assert command[command.index('--image')+1] == str(image)
    assert command[command.index('--graph')+1] == str(graph)
    assert '--gpt' in command
    assert not check.directory.exists()  # CLI owns creation; no exist_ok conflict
    assert kwargs['start_new_session'] is True
    check.close()
    assert signals == [signal.SIGINT]
    assert check.log.closed


@pytest.mark.parametrize('status', ['MATCH', 'NO_MATCH', 'AMBIGUOUS'])
def test_results_keep_original_ann_ranks_and_only_promote_confirmed_match(tmp_path, status):
    trajectory = tmp_path/'trajectory.json'
    trajectory.write_text(json.dumps({'observations': {
        'o1': {'rgb': {'path': '/first.png'}}, 'o2': {'rgb': {'path': '/second.png'}}}}))
    graph = tmp_path/'graph.json'
    graph.write_text(json.dumps({'source_trajectory': {'path': str(trajectory)}}))
    check = GraphCheck(tmp_path/'query.png', tmp_path/'checks', graph)
    (check.directory/'matching').mkdir(parents=True)
    match = dict(status=status, node_id='n2' if status == 'MATCH' else '', confidence=.9)
    (check.directory/'result.json').write_text(json.dumps({'match': match}))
    (check.directory/'matching/retrieval.json').write_text(json.dumps({'ranking': [
        dict(image_id='o1', node_id='n1', similarity=.95),
        dict(image_id='o2', node_id='n2', similarity=.9)]}))
    result, cards = check.results()
    assert cards[0]['ann_rank'] == (2 if status == 'MATCH' else 1)
    assert cards[0]['path'] == ('/second.png' if status == 'MATCH' else '/first.png')
    assert len(cards) == 2
    if status == 'NO_MATCH':
        assert '不代表全图一定没有' in result_title(result['match'])


def test_cancel_reaps_real_subprocess_with_finally(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time
    image = tmp_path/'rgb.png'; image.touch()
    graph = tmp_path/'graph.json'; graph.touch()
    memory = tmp_path/'memory'; memory.mkdir(); (memory/'memory.json').touch()
    ready = tmp_path/'ready'; cleaned = tmp_path/'cleaned'
    script = tmp_path/'fake_matcher.py'
    script.write_text('from pathlib import Path\nimport time\n'
                      f'Path({str(ready)!r}).touch()\n'
                      'try:\n    time.sleep(60)\n'
                      f'finally:\n    Path({str(cleaned)!r}).touch()\n')
    real_popen = subprocess.Popen
    monkeypatch.setattr('cloth_agent.collection_graph_check.subprocess.Popen',
                        lambda cmd, **kwargs: real_popen([sys.executable, str(script)], **kwargs))
    check = GraphCheck(image, tmp_path/'checks', graph, memory)
    try:
        check.start()
        deadline = time.monotonic()+5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
    finally:
        check.close()
    assert cleaned.exists()
    assert check.process.poll() is not None
