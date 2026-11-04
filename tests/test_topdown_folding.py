"""Geometry, independent model branches, abstention and no robot dispatch."""
import copy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cloth_agent.dual_arm.observation import save_observation, sha
from cloth_agent.site_observation import load_site
from cloth_agent.tabletop_view import reconstruct, TabletopView
from cloth_agent.experiments import topdown_folding as folding
from cloth_agent.experiments.partitioned_folding import load_policy


@pytest.fixture
def scene(tmp_path):
    profile = load_site()
    y, x = np.indices((180, 200))
    # Camera looks down: X right, Y image-down opposite base Y, Z down.
    K = [[200., 0., 100.], [0., 200., 90.], [0., 0., 1.]]
    T = [[1., 0., 0., 500.], [0., -1., 0., -400.], [0., 0., -1., 1000.], [0., 0., 0., 1.]]
    xyz = np.stack([500 + (x-100)*5., -400 - (y-90)*5., np.zeros_like(x)], axis=-1)
    rgb = np.full((180, 200, 3), 200, np.uint8)
    rgb[30:150, 30:170] = [200, 15, 15]
    # Raised cloth surface: projectively consistent depth, not flat-plane Z.
    garment = rgb[..., 0] == 200
    garment &= rgb[..., 1] == 15
    xyz[garment, 0] = 500 + (x[garment]-100)*4.9
    xyz[garment, 1] = -400 - (y[garment]-90)*4.9
    xyz[garment, 2] = 20.
    obs = save_observation(tmp_path/'obs', rgb, xyz, NS(raw={'calibration_id': profile['calibration_id']}), {},
        synthetic=True, extra=dict(camera_serial=profile['camera']['serial'], common_frame='right_arm_base',
            intrinsics=K, world_from_camera_mm=T, execution_allowed=False))
    config = dict(version=1, camera_serial=profile['camera']['serial'], image_size_wh=[200, 180],
        table_polygon_px=[[1, 1], [198, 1], [198, 178], [1, 178]], resolution_mm=2.5, garment_mask='red_hsv_largest_component')
    view = reconstruct(obs, tmp_path/'view', config)
    node = lambda key: dict(node_id=key, task_context='test', observation_condition=dict(
        positive_evidence=['visible'], negative_evidence=[], uncertainty_conditions=[]), representative_observations=['o'])
    graph = dict(nodes=[node('n0'), node('n1')], terminal_node_ids=['n1'], edges=[dict(
        edge_id='e', source_node='n0', target_node='n1', action_family='fold', action_template=dict(
            grasp_relation='side', target_relation='center', grasp_z_rule='surface', yaw_rule='site', other_constraints=[]))])
    trajectory = dict(observations={'o': dict(rgb=dict(path=str(obs.image), sha256=sha(obs.image)))})
    return NS(profile=profile, obs=obs, config=config, view=view, graph=graph, trajectory=trajectory,
        templates={'e': dict(arm='left', axis=6)}, policy=load_policy(), tmp=tmp_path)


def match(status='MATCH', node='n0'):
    return dict(status=status, node_id=node if status=='MATCH' else '', candidate_nodes=[node],
        confidence=.95, matched_evidence=['visible'], violated_conditions=[], unresolved_conditions=[])


def decision():
    return dict(status='READY', edge_id='e', confidence=.95, centerline_px=[[100, 50], [100, 130]],
        grasp_px=[135, 80], target_px=[85, 90], evidence=['visible'], unresolved_conditions=[])


class Backend:
    model = 'fake-test-model'
    def __init__(self, answers):
        self.answers = copy.deepcopy(answers)
        self.inputs = []
    def invoke(self, prompt, images, schema):
        self.inputs.append((prompt, list(images)))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return dict(text=json.dumps(answer), tool_calls=0, visual_tool_calls=0, completed_turns=1)


def view_decision(scene):
    result = decision()
    def locate(pixel):
        distances = np.linalg.norm(scene.view.source_xy - pixel, axis=-1)
        distances[~scene.view.mask] = np.inf
        y, x = np.unravel_index(np.argmin(distances), distances.shape)
        return [int(x), int(y)]
    result['centerline_px'] = [locate(p) for p in result['centerline_px']]
    for key in ('grasp_px', 'target_px'):
        result[key] = locate(result[key])
    return result


def branch(scene, backend, *, view=None, name='branch'):
    return folding.propose_branch(scene.obs, scene.graph, scene.trajectory, scene.templates,
        scene.profile, scene.policy, scene.view.meta['table_normal_common'],
        view.image if view else scene.obs.image, scene.tmp/name, backend, view=view)


def test_inverse_mapping_and_real_height(scene):
    d = view_decision(scene)
    request = folding.mapped_request(scene.obs, d, 'n0', scene.view)
    assert request['grasp_px'] == decision()['grasp_px']
    backend = Backend([match(), d])
    result = branch(scene, backend, view=scene.view)
    assert result['status'] == 'PREVIEW_READY'
    assert result['plan']['grasp_common_mm'][2] == pytest.approx(20.)
    assert result['plan']['selected_arm'] == 'left'
    assert result['plan']['execution_allowed'] is False
    assert {o['arm'] for o in result['plan']['operations']} == {'left'}


def test_mapping_hashes_and_observation_binding(scene):
    other = copy.deepcopy(scene.obs)
    other.meta['observation_id'] = 'different'
    with pytest.raises(ValueError, match='different observation'):
        TabletopView.load(scene.view.directory, other)
    np.save(scene.view.directory/'source_xy.npy', np.zeros_like(scene.view.source_xy))
    with pytest.raises(ValueError, match='artifact changed'):
        TabletopView.load(scene.view.directory, scene.obs)


@pytest.mark.parametrize('pixel', [[0, 0], [-1, 5], [9999, 5], [1.2, 4]])
def test_no_background_or_out_of_range_mapping(scene, pixel):
    with pytest.raises(ValueError):
        scene.view.source_pixel(pixel)


def test_missing_real_depth_never_uses_plane(scene):
    d = view_decision(scene)
    x, y = scene.view.source_pixel(d['grasp_px'])
    scene.obs.xyz[y-2:y+3, x-2:x+3] = np.nan
    result = branch(scene, Backend([match(), d]), view=scene.view)
    assert result['status'] == 'BLOCKED'
    assert 'depth' in result['error']
    assert result['plan'] is None


def test_ambiguous_state_and_goal_do_not_generate_actions(scene):
    for name, answer, expected in [('ambiguous', match('AMBIGUOUS'), 'STATE_UNCERTAIN'),
                                   ('goal', match(node='n1'), 'GOAL_OBSERVED')]:
        backend = Backend([answer])
        result = branch(scene, backend, name=name)
        assert result['status'] == expected and result['plan'] is None
        assert len(backend.inputs) == 1


def test_wrong_edge_preserves_failure(scene):
    d = decision(); d['edge_id'] = 'invented'
    result = branch(scene, Backend([match(), d, d]))
    assert result['status'] == 'BLOCKED'
    assert result['metrics']['model_calls'] == 3
    assert result['plan'] is None


def test_no_action_when_competing_candidate_remains(scene):
    m = match(); m['candidate_nodes'] = ['n0', 'n1']
    result = branch(scene, Backend([m]))
    assert result['status'] == 'STATE_UNCERTAIN'
    assert result['plan'] is None


@pytest.mark.parametrize('primary_ready', [True, False])
def test_paired_branch_isolation_no_raw_fallback(scene, monkeypatch, primary_ready):
    monkeypatch.setattr(folding, 'compile_templates', lambda _: (scene.graph, scene.trajectory, scene.templates))
    primary_answers = [match(), view_decision(scene)] if primary_ready else [match('AMBIGUOUS')]
    backends = [Backend(primary_answers), Backend([match(), decision()])]
    index = iter(backends)
    config_path = scene.tmp/'config.json'; config_path.write_text(json.dumps(scene.config))
    output = scene.tmp/'experiment'
    comparison = folding.plan_observation(scene.obs, output, view_config_path=config_path, backend_factory=lambda: next(index))
    primary = json.loads((output/'primary_plan.json').read_text())
    assert (output/'comparison.png').is_file()
    assert comparison['shadow_status'] == 'PREVIEW_READY'
    assert primary['status'] == ('PREVIEW_READY' if primary_ready else 'STATE_UNCERTAIN')
    assert primary['execution_allowed'] is False
    if not primary_ready:
        assert primary['operations'] == []
    assert backends[0].inputs[0][1][0] != backends[1].inputs[0][1][0]
    assert backends[0].inputs[0][1][1:] == backends[1].inputs[0][1][1:]
    raw_prompt = backends[1].inputs[0][0]
    assert 'selected_arm' not in raw_prompt and 'topdown-paired' not in raw_prompt
    assert comparison['robot_calls'] == 0
    if primary_ready:
        assert comparison['grasp_common_mm']['distance_mm'] == pytest.approx(0.)


def test_bad_camera_prevents_all_model_calls(scene, monkeypatch):
    monkeypatch.setattr(folding, 'compile_templates', lambda _: (scene.graph, scene.trajectory, scene.templates))
    scene.config['camera_serial'] = 'wrong'
    p = scene.tmp/'bad.json'; p.write_text(json.dumps(scene.config))
    def forbidden():
        pytest.fail('Model must not be constructed on invalid geometry')
    with pytest.raises(ValueError, match='mismatch'):
        folding.plan_observation(scene.obs, scene.tmp/'experiment', view_config_path=p, backend_factory=forbidden)
    result = json.loads((scene.tmp/'experiment/primary_plan.json').read_text())
    assert result['model_calls'] == 0 and result['operations'] == []


def test_no_execute_option():
    with pytest.raises(SystemExit) as error:
        folding.main(['--execute'])
    assert error.value.code == 2


def test_explicit_table_patches_are_separate_from_garment_roi(scene):
    config = dict(scene.config, plane_fit_polygons_px=[[[2, 2], [197, 2], [197, 25], [2, 25]]])
    view = reconstruct(scene.obs, scene.tmp/'patch_view', config)
    assert view.meta['selected_table_candidates'] < view.meta['neutral_table_candidates']
    assert view.meta['table_inlier_fraction'] > .99
    assert view.mask.any()
    assert (view.directory/'table_fit_candidates.png').is_file()


def test_invalid_table_patches_rejected(scene):
    config = dict(scene.config, plane_fit_polygons_px=[[[2, 2], [999, 2], [10, 25]]])
    with pytest.raises(ValueError, match='plane fit polygon'):
        reconstruct(scene.obs, scene.tmp/'bad_patch', config)
