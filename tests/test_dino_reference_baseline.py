import json

import numpy as np
from PIL import Image
import pytest

from cloth_agent.experiments import dino_reference_baseline as baseline
from cloth_agent.dino_correspondence import DIM, GRID, image_metadata, sha


def test_max_per_node_keeps_matching_edge_and_stable_ties():
    refs = [dict(node_id=n, edge_id=e) for n, e in
            [('n1', 'e1'), ('n1', 'e2'), ('n3', 'e3'), ('n2', 'e4')]]
    features = np.zeros((4, DIM), np.float32)
    features[:, :2] = [[0, 1], [1, 0], [.8, .6], [.8, .6]]
    query = np.zeros(DIM, np.float32); query[0] = 1
    ranked = baseline.rank_nodes(refs, features, query)
    assert [(r['node_id'], r['edge_id']) for r in ranked] == [('n1','e2'), ('n2','e4'), ('n3','e3')]
    with pytest.raises(ValueError, match='normalized'):
        baseline.rank_nodes(refs, features*2, query)


def test_cache_query_encodes_only_current_and_preserves_binding(tmp_path, monkeypatch):
    class Encoder:
        images = []
        def __init__(self, *args):
            self.contract = {'test': True}; self.calls = 0
        def encode_joint(self, image):
            self.calls += 1; self.images.append(image.getpixel((0,0)))
            cls = np.zeros(DIM, np.float32); cls[0 if image.getpixel((0,0))[0] else 1] = 1
            return cls, np.tile(cls, (GRID*GRID, 1)), image_metadata(image.size), 0.
    monkeypatch.setattr(baseline, 'JointEncoder', Encoder)
    red, blue = tmp_path/'red.png', tmp_path/'blue.png'
    Image.new('RGB',(32,24),'red').save(red); Image.new('RGB',(32,24),'blue').save(blue)
    refs = [dict(node_id=n, edge_id=e, image=dict(path=str(p), sha256=sha(p)), edge=dict(edge_id=e))
            for n,e,p in [('n1','e1',red),('n1','e2',blue),('n2','e3',red)]]
    monkeypatch.setattr(baseline, 'references', lambda _: (refs, {}))
    cache, output = tmp_path/'cache', tmp_path/'query'
    manifest = baseline.prepare('unused', cache)
    assert manifest['offline_forward_calls'] == 2
    assert manifest['references'][0]['source_image']['path'] == str(red)
    result = baseline.query(cache, blue, output)
    assert len(Encoder.images) == 3
    assert result['current_forward_calls'] == 1
    assert result['online_reference_forward_calls'] == 0
    assert result['selected']['edge_id'] == 'e2'
    assert json.loads((output/'selected_action_edge.json').read_text())['edge_id'] == 'e2'
    assert result['selected_action_executed'] is False
    (cache/'cls.npy').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        baseline.query(cache, blue, tmp_path/'bad')
