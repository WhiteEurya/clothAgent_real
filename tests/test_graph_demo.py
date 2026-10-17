import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.compare_graph_demo import canonical, compare, read_snapshot


DEMO = Path(__file__).resolve().parents[1]/'examples/graph_demo'


def test_real_reference_and_all_exported_evidence_hashes():
    snapshot = read_snapshot(DEMO/'snapshot.json')
    assert (len(snapshot['nodes']), len(snapshot['actions']), len(snapshot['transitions'])) == (7, 14, 11)
    for row in json.loads((DEMO/'actual_cases.json').read_text())['files']:
        if 'demo_file' in row:
            assert hashlib.sha256((DEMO/row['demo_file']).read_bytes()).hexdigest() == row['sha256']


def test_compare_tolerates_only_snapshot_counter_and_workspace_prefix():
    snapshot = read_snapshot(DEMO/'snapshot.json')
    changed = json.loads(json.dumps(snapshot).replace('/home/sja/clothAgent_real/runs/', '/different/root/runs/'))
    changed['version'] += 2
    assert compare(changed, snapshot)['status'] == 'PASS'
    changed['actions']['F0000000']['preferred'] = False
    result = compare(changed, snapshot)
    assert result['status'] == 'FAIL' and 'actions' in result['differing_sections']
    changed = copy.deepcopy(snapshot)
    changed['transitions'][0]['result_node'] = 'P000006'
    assert compare(changed, snapshot)['status'] == 'FAIL'


def test_frozen_reference_tamper_is_rejected(tmp_path):
    source = DEMO/'snapshot.json'
    target = tmp_path/'graph.json'
    target.write_bytes(source.read_bytes()+b' ')
    Path(str(target)+'.sha256.json').write_bytes((DEMO/'snapshot.json.sha256.json').read_bytes())
    with pytest.raises(ValueError, match='checksum'):
        read_snapshot(target)
