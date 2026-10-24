import copy
import pytest
from scripts.extend_arm6_archive_graph import extend_evidence, next_node_number


def observation(path):
    return {key: {'path': path + key, 'sha256': 'abc'} for key in ('rgb', 'observation', 'surface')}


def step(sid, manifest, before, after):
    return dict(step_id=sid, commanded_action={'manifest': {'path': manifest, 'sha256': 'abc'}},
                before_observation=before, after_observation=after)


def test_incremental_ids_survive_reenumeration_and_resume():
    old = dict(steps=[step('s010', 'old', 'o020', None)], observations={'o020': observation('old')})
    original = copy.deepcopy(old)
    imported = dict(steps=[step('s000', 'old', 'o000', None), step('s001', 'new', 'o000', 'o001')],
                    observations={'o000': observation('old'), 'o001': observation('new')},
                    sessions=[], excluded_iterations=[])
    combined, steps, images = extend_evidence(old, imported)
    assert old == original
    assert combined['steps'][0] == old['steps'][0]
    assert steps[0]['step_id'] == 's011'
    assert steps[0]['before_observation'] == 'o020'
    assert steps[0]['after_observation'] == 'o021'
    assert list(images) == ['o021']
    resumed, more, refs = extend_evidence(combined, imported)
    assert resumed == combined and not more and not refs
    imported['steps'][0]['commanded_action']['manifest']['sha256'] = 'changed'
    with pytest.raises(ValueError, match='demonstration changed'):
        extend_evidence(old, imported)


def test_changed_shared_observation_is_rejected():
    old = dict(steps=[step('s000', 'old', 'o000', None)], observations={'o000': observation('shared')})
    imported = dict(steps=[step('s000', 'new', 'o000', None)], observations={'o000': observation('shared')},
                    sessions=[], excluded_iterations=[])
    imported['observations']['o000']['surface']['sha256'] = 'changed'
    with pytest.raises(ValueError, match='observation evidence changed'):
        extend_evidence(old, imported)


def test_node_ids_do_not_reuse_merged_away_ids():
    graph = dict(nodes=[{'node_id': 'n077'}, {'node_id': 'unknown_s007'}],
                 revision_history=[{'merges': [{'source_nodes': ['n077', 'n078']}]}])
    assert next_node_number(graph) == 79
