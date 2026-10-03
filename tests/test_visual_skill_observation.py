import json
from pathlib import Path

from PIL import Image
import pytest

from cloth_agent.auto_exploration import VISUAL_PLAN_JSON_SCHEMA
from cloth_agent.planner_backend import BackendResult
from cloth_agent.remote_fold import RemoteFoldClient
from cloth_agent.visual_skill_observation import load_observation_skill, prepare_skill_observations


SKILL = Path(__file__).resolve().parents[1]/'data/skills/experimental/visual_information.json'


@pytest.fixture
def images(tmp_path):
    paths = []
    for name in ['camera_A_rgb_upright.png', 'camera_A_rxxx_overlay_upright.png', 'reference.png']:
        path = tmp_path/name
        Image.new('RGB', (100, 80), 'purple').save(path)
        paths.append(path)
    return paths


def request(source='image_0'):
    return {'gap_id': 'edge', 'skill_id': 'overlay_occlusion', 'source_image_id': source,
            'roi': [.1, .2, .6, .7], 'degrees_clockwise': 0, 'enlarge': False,
            'expected_information_gain': 'Determine whether overlay hides the boundary'}


def observer(monkeypatch, requests):
    from cloth_agent.harness import model
    class Stub:
        def __init__(self, **kwargs): self.calls = []
        def invoke(self, **kwargs):
            self.calls.append({'stage': kwargs['stage']})
            assert kwargs['stage'] == 'skill_observation'
            return {'observation_requests': requests}
    monkeypatch.setattr(model, 'RuntimeClaude', Stub)


def test_host_views_have_mapping_and_preserve_original_roots(tmp_path, images, monkeypatch):
    observer(monkeypatch, [request()])
    paths, bundle, edits = prepare_skill_observations(images=images, context={'objective': 'fold'},
        instructions='Select a marker', artifact=load_observation_skill(SKILL), root=tmp_path,
        ssh_host='not-contacted', timeout_s=20)
    assert paths[:3] == images
    assert len(paths) == 5 and edits == 2 and bundle['remaining_visual_edit_budget'] == 4
    assert bundle['images'][3]['to_original'] == [1, 0, 10, 0, 1, 16]
    assert bundle['images'][3]['original_image_id'] == 'image_0'
    assert bundle['images'][4]['original_image_id'] == 'image_1'
    assert all(p.exists() for p in paths)
    reports = list((tmp_path/'observation_skill').glob('*/report.json'))
    assert json.loads(reports[0].read_text())['status'] == 'PREPARED'
    assert 'expected_information_gain' not in json.dumps(bundle)


def test_invalid_root_fails_without_silent_fallback(tmp_path, images, monkeypatch):
    observer(monkeypatch, [request('image_2')])
    with pytest.raises(ValueError, match='clean root'):
        prepare_skill_observations(images=images, context={'objective': 'fold'}, instructions='select',
            artifact=load_observation_skill(SKILL), root=tmp_path, ssh_host='not-contacted', timeout_s=20)
    report = json.loads(next((tmp_path/'observation_skill').glob('*/report.json')).read_text())
    assert report['status'] == 'FAILED'


@pytest.mark.parametrize('with_skill', [False, True])
def test_original_selection_contract_and_grounding_roots_unchanged(tmp_path, images, monkeypatch, with_skill):
    from cloth_agent import visual_skill_observation
    def forbidden_prepare(**kwargs):
        pytest.fail('Inline mode must not start a separate observer or preparation stage')
    monkeypatch.setattr(visual_skill_observation, 'prepare_skill_observations', forbidden_prepare)
    class Backend:
        ssh_host = 'not-contacted'
        def __init__(self): self.calls = []
        def invoke(self, **kwargs):
            self.calls.append(kwargs)
            plan = {'garment_observation': 'cloth', 'opening_strategy': 'fold', 'confidence': .7,
                    'selected_reference': {'camera': 'A', 'reference_id': 'R001', 'reason': 'visible'},
                    'motion_intent': 'fold', 'expected_observation': 'folded', 'safety_notes': ['validate']}
            return BackendResult(json.dumps(plan), '', 0, ())
    b = Backend()
    client = RemoteFoldClient(backend=b, observation_skill_path=SKILL if with_skill else None)
    client._remote_context = {'objective': 'fold'}
    client._remote_images = images.copy()
    result = client._visual_plan(images, 'original task', tmp_path)
    assert len(b.calls) == 1
    call = b.calls[0]
    assert call['usage_stage'] == 'visual_planning'
    assert result.decision.selected_reference['reference_id'] == 'R001'
    assert call['schema'] == VISUAL_PLAN_JSON_SCHEMA
    assert call['image_paths'] == images
    assert call['image_edit_limit'] == 6
    assert call['information_tools'] is with_skill
    assert client._remote_images == images
    assert 'prepared_observation_views' not in client._remote_context
    assert 'prepared_observation_views' not in ''.join(call['context_files'].values())
    assert ('OBSERVATION POLICY EXPERIMENT' in call['prompt']) == with_skill
    if with_skill:
        assert 'success_check' in call['prompt'] and 'on_insufficient' in call['prompt']
        assert 'returns the actual image in this conversation' in call['prompt']
    assert not (tmp_path/'observation_skill').exists()


def test_empty_observations_are_valid_and_opt_in_defaults_off(tmp_path, images, monkeypatch):
    observer(monkeypatch, [])
    paths, _, edits = prepare_skill_observations(images=images, context={'objective': 'fold'},
        instructions='select', artifact=load_observation_skill(SKILL), root=tmp_path,
        ssh_host='not-contacted', timeout_s=20)
    assert paths == images and edits == 0
    from cloth_agent.fold_exploration_pipeline import build_parser
    assert build_parser().parse_args([]).observation_skill is None
