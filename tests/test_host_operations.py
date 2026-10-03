import pytest
from cloth_agent.harness.host_operations import compute, execute, validate_bindings
from cloth_agent.harness.policy import PolicyError


def test_geometry_and_registry_ranking():
    assert compute('reflect_point', {'point': [2, 1], 'line_start': [0, 0], 'line_end': [1, 1]}, {}) == [1, 2]
    assert compute('affine_point', {'point': [2, 3], 'matrix': [0, -1, 10, 1, 0, 20]}, {}) == [7, 22]
    registry = {'candidates': [{'candidate_id': 'R001', 'pixel_xy': [2, 1]},
                               {'candidate_id': 'R002', 'pixel_xy': [5, 1]}]}
    args = {'candidate_ids': ['R001', 'R002'], 'origin': [0, 0], 'axis': [2, 0]}
    ranked = compute('rank_candidates', args, registry)
    assert [r['candidate_id'] for r in ranked] == ['R002', 'R001']
    assert ranked[0]['projection'] == 5
    args['candidate_ids'] = ['R999']
    with pytest.raises(ValueError, match='Unknown'):
        compute('rank_candidates', args, registry)


@pytest.mark.parametrize('measurements', [{}, {'p': None},
    {'p': [1, 2], 'a': [0, 0], 'b': [0, 0]},
    {'p': [float('nan'), 2], 'a': [0, 0], 'b': [1, 1]}])
def test_missing_or_invalid_measurements_are_unknown(measurements):
    stage = {'context': ['measure'], 'host_operations': [{'id': 'mirror', 'op': 'reflect_point',
        'source_stage': 'measure', 'bindings': {'point': 'p', 'line_start': 'a', 'line_end': 'b'}}]}
    record = execute(stage, {'measure': {'measurements': measurements}}, {})[0]
    assert record['status'] == 'UNKNOWN' and record['output'] is None
    assert 'missing_information' in record
    stage['host_operations'][0]['source_stage'] = 'future'
    with pytest.raises(PolicyError):
        validate_bindings(stage)
