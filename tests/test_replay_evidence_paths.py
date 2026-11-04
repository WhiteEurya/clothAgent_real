import hashlib
import json

import pytest

from cloth_agent.experiments.graph_policy import checked_file, frozen, load_frozen
from cloth_agent.experiments.replay_evidence_paths import resolve_evidence_path


def test_restored_evidence_keeps_original_hashes(tmp_path, monkeypatch):
    monkeypatch.setenv('CLOTH_REPLAY_EVIDENCE_ROOT', str(tmp_path))
    target = tmp_path / 'workspace/runs/example.json'
    frozen(target, {'original': '/home/sja/clothAgent_real/unchanged'})
    original = '/home/sja/clothAgent_real/runs/example.json'
    record = {'path': original, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    assert checked_file(record) == target
    assert load_frozen(original)['original'] == '/home/sja/clothAgent_real/unchanged'
    target.write_text('{}')
    with pytest.raises(ValueError, match='modified'):
        checked_file(record)
    with pytest.raises(ValueError, match='hash mismatch'):
        load_frozen(original)


def test_only_known_roots_map_and_no_escape(tmp_path, monkeypatch):
    monkeypatch.setenv('CLOTH_REPLAY_EVIDENCE_ROOT', str(tmp_path))
    assert str(resolve_evidence_path('/mnt/newssd/sja/a')) == str(tmp_path / 'ssd/sja/a')
    assert str(resolve_evidence_path('/home/sja/clothAgent_real_other/a')) == '/home/sja/clothAgent_real_other/a'
    with pytest.raises(ValueError, match='escapes'):
        resolve_evidence_path('/home/sja/clothAgent_real/../../escape')


def test_svg_without_graphviz(tmp_path, monkeypatch):
    from tests.test_graph_policy_replay import policy
    from cloth_agent.experiments.graph_policy_replay import OnlineGrowth
    monkeypatch.setattr('shutil.which', lambda _: None)
    path, _ = policy(tmp_path)
    growth = OnlineGrowth(path, tmp_path / 'output')
    growth.render()
    import xml.etree.ElementTree as ET
    root = ET.parse(growth.latest_path.with_suffix('.svg')).getroot()
    text = ''.join(root.itertext())
    assert 'N0' in text and 'N1' in text and 'E0' in text
